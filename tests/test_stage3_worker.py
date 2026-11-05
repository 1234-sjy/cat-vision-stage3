import importlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import threading
import time
from types import SimpleNamespace

from PIL import Image
from catvision.stage3_settings import Settings
from catvision.stage3_store import Store


class Runner:
    def __init__(self):
        self.calls = 0
        self.questions = []
        self.fail = False

    def __call__(self,image,directory,progress):
        self.calls += 1
        if self.fail:
            raise RuntimeError('device unavailable')
        progress('两条路线正在同时识别')
        image.save(directory/'traditional.png')
        return {'traditional':{'breed':'Bengal'},'vlm':{'breed':'Bengal'},'fresh_run':True,'total_seconds':0.1}

    def answer_question(self,image,question):
        self.questions.append((image.size,question))
        return {'answer':'能看到竖起的耳朵','inference_seconds':0.1}


class Stage3WorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('catvision.stage3_worker'),'队列工作端尚未实现')
        self.worker = importlib.import_module('catvision.stage3_worker')
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(database_url='sqlite:///'+(self.root/'database.db').as_posix(),
                                 data_dir=self.root/'jobs',jwt_secret='testing-secret-'*5)
        self.store = Store(self.settings)
        self.addCleanup(self.store.engine.dispose)
        self.user = self.store.add_user('worker_test','test hash')
        self.runner = Runner()

    def submit(self):
        job = self.store.add_job(self.user.id)
        self.store.change(job['id'],dispatched=True)
        folder = self.settings.data_dir/job['id']
        folder.mkdir()
        Image.new('RGB',(37,29),'orange').save(folder/'input.png')
        return job['id']

    def test_duplicate_delivery_does_not_run_model_again(self):
        identity = self.submit()
        self.worker.process_job(self.store,identity,self.runner)
        self.worker.process_job(self.store,identity,self.runner)
        self.assertEqual(self.runner.calls,1)
        self.assertEqual(self.store.get(identity)['state'],'completed')
        saved = json.loads((self.settings.data_dir/identity/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['state'],'completed')
        self.assertEqual(saved['result'].get('task_id'),identity)
        self.assertEqual(len(saved['result'].get('input_sha256','')),64)

    def test_question_reads_parent_original(self):
        parent = self.submit()
        self.worker.process_job(self.store,parent,self.runner)
        child = self.store.add_job(self.user.id,kind='image_question',parent_id=parent,question='耳朵？')
        self.worker.process_job(self.store,child['id'],self.runner)
        self.assertEqual(self.runner.questions,[((37,29),'耳朵？')])
        self.assertEqual(self.store.get(child['id'])['result']['answer'],'能看到竖起的耳朵')
        self.assertEqual(self.store.get(child['id'])['result'].get('task_id'),child['id'])
        self.assertEqual(self.store.get(child['id'])['result'].get('parent_id'),parent)

    def test_failure_keeps_traceback_and_stops_inference(self):
        identity = self.submit()
        self.runner.fail = True
        with self.assertRaisesRegex(RuntimeError,'device unavailable'):
            self.worker.process_job(self.store,identity,self.runner)
        self.assertEqual(self.store.get(identity)['state'],'error')
        error = (self.settings.data_dir/identity/'error.log').read_text(encoding='utf-8')
        self.assertIn('Traceback',error)
        self.assertIn('device unavailable',error)
        self.worker.process_job(self.store,identity,self.runner)
        self.assertEqual(self.runner.calls,1)

    def test_worker_restart_marks_running_interrupted_without_retry(self):
        identity = self.submit()
        self.store.claim(identity)
        self.store.interrupt_running()
        self.worker.process_job(self.store,identity,self.runner)
        self.assertEqual(self.runner.calls,0)
        self.assertEqual(self.store.get(identity)['state'],'interrupted')

    def test_explicit_recovery_interrupts_task_without_rerunning_model(self):
        self.assertTrue(hasattr(self.worker,'recover_interrupted'),'需要显式恢复操作')
        identity = self.submit()
        self.store.claim(identity)
        self.worker.recover_interrupted(self.store,self.root)
        self.assertEqual(self.store.get(identity)['state'],'interrupted')
        self.assertEqual(self.runner.calls,0)

    def test_consumer_returns_to_io_loop_while_inference_is_pending(self):
        self.assertTrue(hasattr(self.worker,'QueueConsumer'),'GPU工作应从AMQP回调移到独立线程')
        identity = self.submit()
        release = threading.Event()
        scheduled = []
        acknowledged = []
        class SlowRunner(Runner):
            def __call__(self,image,directory,progress):
                release.wait(1)
                return super().__call__(image,directory,progress)
        connection = SimpleNamespace(add_callback_threadsafe=scheduled.append)
        channel = SimpleNamespace(basic_ack=lambda tag:acknowledged.append(tag),basic_nack=lambda *a,**k:None)
        consumer = self.worker.QueueConsumer(self.store,SlowRunner(),connection)
        try:
            started = time.perf_counter()
            consumer.receive(channel,SimpleNamespace(delivery_tag=7),None,json.dumps({'job_id':identity}).encode())
            self.assertLess(time.perf_counter()-started,0.15)
            self.assertEqual(acknowledged,[])
            release.set()
        finally:
            release.set()
            consumer.close()
        self.assertEqual(self.store.get(identity)['state'],'completed')
        self.assertEqual(len(scheduled),1)
        scheduled[0]()
        self.assertEqual(acknowledged,[7])


if __name__=='__main__':
    unittest.main()
