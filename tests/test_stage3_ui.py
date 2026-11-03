import json
from pathlib import Path
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class Stage3UITests(unittest.TestCase):
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
