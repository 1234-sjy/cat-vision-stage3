import io
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
