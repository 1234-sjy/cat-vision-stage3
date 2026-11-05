import importlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image


class Delivery:
    def __init__(self):
        self.messages = []
        self.fail = False

    def publish(self, job_id):
        if self.fail:
            raise ConnectionError('broker unavailable')
        self.messages.append(job_id)


class Stage3ServiceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('catvision.stage3_api'), '第三阶段持久化服务尚未实现')
        api = importlib.import_module('catvision.stage3_api')
        settings = importlib.import_module('catvision.stage3_settings')
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = settings.Settings(database_url='sqlite:///' + (self.root/'database.db').as_posix(),
                                          data_dir=self.root/'jobs', jwt_secret='test-secret-'*6)
        self.delivery = Delivery()
        self.app = api.create_app(self.settings, publisher=self.delivery)
        self.client = self.enterContext(TestClient(self.app))
        self.a = self.account('first_user')
        self.b = self.account('second_user')

    def account(self, username):
        response = self.client.post('/api/auth/register', json={'username':username,'password':'test-password-123'})
        self.assertEqual(response.status_code, 201, response.text)
        return {'Authorization':'Bearer ' + response.json()['access_token']}

    def photo(self):
        output = io.BytesIO()
        Image.new('RGB', (24, 24), 'orange').save(output, 'PNG')
        return output.getvalue()

    def submit(self):
        response = self.client.post('/api/recognize', headers=self.a,
                                    files={'file':('cat.png', self.photo(), 'image/png')})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()['id']

    def test_upload_requires_login(self):
        response = self.client.post('/api/recognize', files={'file':('cat.png', self.photo(), 'image/png')})
        self.assertEqual(response.status_code, 401)

    def test_password_is_hashed_and_login_checks_password(self):
        response = self.client.post('/api/auth/login', json={'username':'first_user','password':'incorrect-password'})
        self.assertEqual(response.status_code, 401)
        good = self.client.post('/api/auth/login', json={'username':'first_user','password':'test-password-123'})
        self.assertEqual(good.status_code, 200)
        from sqlalchemy import text
        with self.app.state.store.engine.connect() as connection:
            value = connection.execute(text('select password_hash from users where username=:name'), {'name':'first_user'}).scalar_one()
        self.assertNotIn('test-password-123', value)
        self.assertTrue(value.startswith('scrypt$'))

    def test_expired_token_is_rejected(self):
        import jwt
        token = jwt.encode({'sub':'unused','exp':1}, self.settings.jwt_secret, algorithm='HS256')
        self.assertEqual(self.client.get('/api/auth/me', headers={'Authorization':'Bearer '+token}).status_code, 401)

    def test_another_user_cannot_read_job_or_original_photo(self):
        identity = self.submit()
        self.assertEqual(self.client.get('/api/jobs/'+identity, headers=self.b).status_code, 404)
        self.assertEqual(self.client.get('/api/files/'+identity+'/input.png', headers=self.b).status_code, 404)
        self.assertEqual(self.client.get('/api/files/'+identity+'/input.png', headers=self.a).status_code, 200)

    def test_history_survives_new_application_instance(self):
        identity = self.submit()
        api = importlib.import_module('catvision.stage3_api')
        other = self.enterContext(TestClient(api.create_app(self.settings, publisher=self.delivery)))
        job = other.get('/api/jobs/'+identity, headers=self.a)
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()['state'], 'queued')
        self.assertEqual(other.get('/api/jobs', headers=self.a).json()['items'][0]['id'], identity)

    def test_invalid_file_does_not_create_or_publish_a_task(self):
        response = self.client.post('/api/recognize', headers=self.a, files={'file':('cat.png',b'not an image','image/png')})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.delivery.messages, [])

    def test_broker_failure_is_saved_in_history(self):
        self.delivery.fail = True
        response = self.client.post('/api/recognize', headers=self.a, files={'file':('cat.png',self.photo(),'image/png')})
        self.assertEqual(response.status_code, 503)
        jobs = self.client.get('/api/jobs', headers=self.a).json()['items']
        self.assertEqual(jobs[0]['state'], 'error')
        self.assertIn('投递', jobs[0]['message'])

    def test_active_task_cannot_be_hidden_and_completed_label_can_be_updated(self):
        identity = self.submit()
        self.assertEqual(self.client.delete('/api/jobs/'+identity, headers=self.a).status_code, 409)
        self.app.state.store.finish(identity, {'traditional':{'breed':'Bengal'},'vlm':{'breed':'Bengal'}}, '识别完成')
        renamed = self.client.patch('/api/jobs/'+identity, headers=self.a, json={'label':'窗边的猫'})
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()['label'], '窗边的猫')
        self.assertEqual(self.client.delete('/api/jobs/'+identity, headers=self.a).status_code, 204)
        self.assertEqual(self.client.get('/api/jobs/'+identity, headers=self.a).status_code, 404)
        self.assertTrue((self.settings.data_dir/identity/'input.png').is_file())

    def test_question_is_owned_and_queued_after_recognition(self):
        identity = self.submit()
        self.app.state.store.finish(identity, {}, '识别完成')
        response = self.client.post('/api/jobs/'+identity+'/questions', headers=self.a, json={'question':'耳朵是什么形状？'})
        self.assertEqual(response.status_code, 202)
        question = self.client.get('/api/jobs/'+response.json()['id'], headers=self.a).json()
        self.assertEqual(question['parent_id'], identity)
        self.assertEqual(question['kind'], 'image_question')
        self.assertEqual(self.client.post('/api/jobs/'+identity+'/questions', headers=self.b, json={'question':'耳朵？'}).status_code, 404)

    def test_concurrent_admission_respects_queue_capacity(self):
        from catvision.stage3_store import Store
        limited = Store(replace(self.settings,max_pending=2))
        self.addCleanup(limited.engine.dispose)
        user = limited.user(username='first_user')
        def submit(_):
            try:
                limited.add_job(user.id)
                return True
            except OverflowError:
                return False
        with ThreadPoolExecutor(max_workers=6) as pool:
            admitted = list(pool.map(submit,range(6)))
        self.assertEqual(sum(admitted),2)

    def test_capacity_rejected_uploads_leave_no_untracked_photos(self):
        api = importlib.import_module('catvision.stage3_api')
        other = self.enterContext(TestClient(api.create_app(replace(self.settings,max_pending=0),publisher=self.delivery)))
        for _ in range(3):
            response = other.post('/api/recognize',headers=self.a,files={'file':('cat.png',self.photo(),'image/png')})
            self.assertEqual(response.status_code,429)
        self.assertEqual(list(self.settings.data_dir.glob('*/input.png')),[])

    def test_capacity_rejected_retry_does_not_copy_original(self):
        identity = self.submit()
        self.app.state.store.change(identity,state='error')
        api = importlib.import_module('catvision.stage3_api')
        other = self.enterContext(TestClient(api.create_app(replace(self.settings,max_pending=0),publisher=self.delivery)))
        response = other.post('/api/jobs/'+identity+'/retry',headers=self.a)
        self.assertEqual(response.status_code,429)
        self.assertEqual(len(list(self.settings.data_dir.glob('*/input.png'))),1)


if __name__ == '__main__':
    unittest.main()
