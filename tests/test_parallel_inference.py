from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from catvision.server import LocalComparison
from catvision.run_control import inference_stream


class ParallelInferenceTests(unittest.TestCase):
    def test_resident_engine_reuses_stream_to_bound_allocator_caches(self):
        class Engine: pass
        class Stream:
            def wait_stream(self,other): pass
            def synchronize(self): pass
        a,b=Engine(),Engine()
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.Stream',side_effect=[Stream(),Stream()]) as factory,patch('torch.cuda.default_stream',return_value=None),patch('torch.cuda.stream',side_effect=lambda stream:nullcontext()):
            for _ in range(5):
                with inference_stream(owner=a): pass
                with inference_stream(owner=b): pass
            self.assertEqual(factory.call_count,2)

    def test_both_routes_run_together_with_independent_original_images(self):
        barrier = threading.Barrier(2, timeout=2)
        seen = []
        class Engine:
            load_seconds = 0
            def predict(self, image):
                seen.append(image.getpixel((0, 0)))
                image.putpixel((0, 0), (0, 0, 0))
                barrier.wait()
                return {'status': 'no_cat', 'has_cat': False, 'breed': None}
            def close(self): pass
        pipeline = LocalComparison(keep_loaded=True)
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_make_engine', side_effect=lambda route: Engine()), patch('catvision.server.inference_stream', return_value=nullcontext(), create=True):
            result = pipeline._run(Image.new('RGB', (8, 8), (21, 31, 41)), Path(folder), lambda text: None)
        self.assertEqual(seen, [(21, 31, 41)] * 2)
        self.assertEqual(result['execution_mode'], 'parallel_routes')
        self.assertTrue(result['fresh_run'])

    def test_failure_closes_models_only_after_other_worker_finishes(self):
        sibling_done = threading.Event()
        barrier = threading.Barrier(2, timeout=2)
        closed = []
        class Engine:
            load_seconds = 0
            def __init__(self, route): self.route = route
            def predict(self, image):
                barrier.wait()
                if self.route == 'traditional':
                    raise RuntimeError('diagnostic failure')
                sibling_done.set()
                return {'status': 'no_cat', 'breed': None}
            def close(self): closed.append(sibling_done.is_set())
        pipeline = LocalComparison(keep_loaded=True)
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_make_engine', side_effect=Engine), patch('catvision.server.inference_stream', return_value=nullcontext(), create=True), patch('catvision.server.ModelTask', return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
                pipeline(Image.new('RGB', (8, 8)), Path(folder), lambda text: None)
            self.assertTrue((Path(folder) / 'partial-result.json').exists())
        self.assertEqual(closed, [True, True])


if __name__ == '__main__': unittest.main()
