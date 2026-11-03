import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from catvision.server import create_app


class AdaptedReportTests(unittest.TestCase):
    def test_report_follows_selected_adapter_and_retains_test_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'configs').mkdir()
            (root/'configs/active-classifier.json').write_text(json.dumps({'classes': ['A', 'B']}))
            active = root/'configs/active-vlm.json'
            active.write_text(json.dumps({'adapter': 'weights/cat', 'adapter_sha256': 'chosen'}))
            summary = root/'artifacts/optimization-v4/comparison-summary.json'
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'adapter_sha256': 'chosen',
                'fresh_matched': {'traditional': {'all': {'count': 20, 'accuracy': .8}},
                                 'vlm': {'all': {'count': 20, 'accuracy': .75}}},
                'traditional_full_test': {'all': {'count': 30, 'accuracy': .85}},
                'regression_120': {'previously_evaluated_images': True}}))
            with TestClient(create_app(root, lambda *args: {})) as client:
                report = client.get('/api/report').json()
                self.assertEqual(report['experiment'], 'adapted-v4')
                self.assertEqual(report['vlm']['accuracy'], .75)
                self.assertEqual(report['vlm']['count'], 20)
                active.write_text(json.dumps({'adapter': 'weights/cat', 'adapter_sha256': 'different'}))
                pending = client.get('/api/report').json()
                self.assertIsNone(pending['vlm'])
                self.assertEqual(pending['assessment'], 'pending')


if __name__ == '__main__':
    unittest.main()
