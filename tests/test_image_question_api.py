import io
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from PIL import Image
from catvision.server import create_app


class QuestionRunner:
    def __init__(self):
        self.received = []

    def __call__(self, image, directory, progress):
        return {'traditional':{'status':'ok'},'vlm':{'status':'ok'}}

    def answer_question(self, image, question):
        self.received.append((image.size,image.getpixel((0,0)),question))
        return {'answer':'耳朵直立。','inference_seconds':0.1}


def wait_job(client, identity):
    for _ in range(100):
        job = client.get('/api/jobs/'+identity).json()
        if job['state'] in {'completed','error'}:
            return job
        time.sleep(0.01)
    raise AssertionError('Job did not finish')


class ImageQuestionApiTests(unittest.TestCase):
    def test_server_binds_results_to_task_and_saved_image_not_runner_metadata(self):
        class ForeignMetadata(QuestionRunner):
            def __call__(self, image, directory, progress):
                return {**super().__call__(image, directory, progress),
                    'task_id': 'b'*32, 'input_sha256': 'wrong-image'}

            def answer_question(self, image, question):
                return {**super().answer_question(image, question),
                    'task_id': 'b'*32, 'parent_id': 'b'*32, 'input_sha256': 'wrong-image'}

        buffer=io.BytesIO();Image.new('RGB',(5,3),(14,36,71)).save(buffer,'PNG')
        expected=hashlib.sha256(b'(5, 3)'+bytes([14,36,71])*15).hexdigest()
        with tempfile.TemporaryDirectory() as folder, TestClient(create_app(Path(folder),ForeignMetadata())) as client:
            accepted=client.post('/api/recognize',files={'file':('A.png',buffer.getvalue(),'image/png')}).json()
            self.assertEqual(accepted.get('input_sha256'),expected)
            self.assertEqual(accepted.get('filename'),'A.png')
            image=wait_job(client,accepted['id'])
            self.assertEqual(image['result']['task_id'],accepted['id'])
            self.assertEqual(image['result']['input_sha256'],expected)
            queued=client.post('/api/jobs/'+image['id']+'/questions',json={'question':'耳朵是什么样？'}).json()
            self.assertEqual(queued.get('parent_id'),image['id'])
            answer=wait_job(client,queued['id'])
            self.assertEqual(answer['result']['task_id'],queued['id'])
            self.assertEqual(answer['result']['parent_id'],image['id'])
            self.assertEqual(answer['result']['input_sha256'],expected)

    def test_question_reads_the_same_saved_pixels_and_is_persisted_separately(self):
        buffer=io.BytesIO();Image.new('RGB',(5,3),(14,36,71)).save(buffer,'PNG')
        runner=QuestionRunner()
        with tempfile.TemporaryDirectory() as folder, TestClient(create_app(Path(folder),runner)) as client:
            photo=client.post('/api/recognize',files={'file':('cat.png',buffer.getvalue(),'image/png')}).json()['id']
            self.assertEqual(wait_job(client,photo)['state'],'completed')
            response=client.post('/api/jobs/'+photo+'/questions',json={'question':'  耳朵是什么样的？  '})
            self.assertEqual(response.status_code,202)
            answer=wait_job(client,response.json()['id'])
            self.assertEqual(answer['result']['answer'],'耳朵直立。')
            self.assertEqual(answer['parent_id'],photo)
            self.assertEqual(runner.received,[((5,3),(14,36,71),'耳朵是什么样的？')])
            self.assertTrue((Path(folder)/'runtime/results'/answer['id']/'result.json').is_file())
            self.assertIn('traditional',client.get('/api/jobs/'+photo).json()['result'])

    def test_absent_parent_and_empty_or_oversized_question_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, TestClient(create_app(Path(folder),QuestionRunner())) as client:
            path='/api/jobs/'+'a'*32+'/questions'
            self.assertEqual(client.post(path,json={'question':'图里有什么？'}).status_code,404)
            self.assertEqual(client.post(path,json={'question':'   '}).status_code,422)
            self.assertEqual(client.post(path,json={'question':'字'*501}).status_code,422)


if __name__ == '__main__':
    unittest.main()
