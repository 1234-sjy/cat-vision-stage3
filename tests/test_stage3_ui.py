import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class Stage3UITests(unittest.TestCase):
    def authenticated_page(self,job,history=None):
        def response(method,url,**kwargs):
            result=requests.Response();result.status_code=200
            if '/api/files/' in url:
                content=io.BytesIO();Image.new('RGB',(8,6),'orange').save(content,'PNG')
                result._content=content.getvalue()
                return result
            if '/api/info' in url:
                payload={'worker':{'ready':True,'message':'ready'},'breeds':['布偶猫']}
            elif '/api/jobs?' in url:
                payload={'items':history or [],'total':len(history or [])}
            else:
                payload=job
            result._content=json.dumps(payload).encode()
            return result
        app=AppTest.from_file(str(ROOT/'catvision/stage3_ui.py'))
        app.session_state['token']='test-session'
        app.session_state['username']='tester'
        app.session_state['active_job']='a'*32
        with patch('requests.request',side_effect=response):
            app.run(timeout=15)
        self.assertEqual(len(app.exception),0)
        return app

    def test_running_job_disables_upload_and_history_switching(self):
        job={'id':'a'*32,'kind':'comparison','state':'running','message':'正在识别A图',
             'label':'A','questions':[],'worker':{'ready':True},'result':None}
        app=self.authenticated_page(job,[job])
        self.assertTrue(app.get('file_uploader')[0].proto.disabled)
        self.assertTrue(next(button for button in app.button if button.key=='open-'+'a'*32).disabled)

    def test_foreign_task_response_never_renders_its_breed(self):
        job={'id':'b'*32,'kind':'comparison','state':'completed','message':'识别完成','label':'B','questions':[],
             'result':{'task_id':'b'*32,'input_sha256':'1'*64,
                       'traditional':{'status':'ok','breed_zh':'不属于A的猫'},'vlm':{}}}
        app=self.authenticated_page(job)
        self.assertTrue(any('任务' in error.value and '不匹配' in error.value for error in app.error))
        self.assertFalse(any(header.value=='不属于A的猫' for header in app.subheader))

    def test_login_and_register_are_available_without_models(self):
        app = AppTest.from_file(str(ROOT/'catvision/stage3_ui.py')).run(timeout=15)
        self.assertEqual(len(app.exception),0)
        self.assertEqual([tab.label for tab in app.tabs],['登录','注册'])

    def test_signed_in_page_renders_upload_with_installed_streamlit(self):
        def response(method,url,**kwargs):
            result = requests.Response()
            result.status_code=200
            if '/api/info' in url:
                payload = {'worker':{'ready':True,'message':'ready'},'breeds':['布偶猫']}
            else:
                payload = {'items':[],'total':0}
            result._content=json.dumps(payload).encode()
            return result
        app = AppTest.from_file(str(ROOT/'catvision/stage3_ui.py'))
        app.session_state['token']='test-session'
        app.session_state['username']='tester'
        with patch('requests.request',side_effect=response):
            app.run(timeout=15)
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any(title.value=='一张猫图，两个答案。' for title in app.title))


if __name__=='__main__':
    unittest.main()
