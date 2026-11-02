from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from catvision.run_control import ModelTask, gpu_checkpoint


class RunControlTests(unittest.TestCase):
    def test_fast_inference_keeps_gpu_sync_and_removes_only_inference_pause(self):
        events=[]
        class FakeCuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def synchronize(): events.append('sync')
        with patch.dict('os.environ',{'CATVISION_FAST_INFERENCE':'1'}):
            actual=gpu_checkpoint(cuda=FakeCuda,sleep=lambda seconds:events.append(seconds))
            self.assertEqual(actual,0)
            self.assertEqual(events,['sync'])
            actual=gpu_checkpoint(cuda=FakeCuda,sleep=lambda seconds:events.append(seconds),pause_seconds=0.05)
            self.assertEqual(actual,0.05)
            self.assertEqual(events[-2:],['sync',0.05])

    def test_second_task_is_rejected_while_first_is_active(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with ModelTask("download", root):
                with self.assertRaisesRegex(RuntimeError, "已有模型任务"):
                    with ModelTask("validate", root):
                        self.fail("Two model tasks entered together")

    def test_failure_is_logged_and_next_run_requires_manual_handling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "original diagnostic"):
                with ModelTask("validation", root):
                    print("before failure")
                    raise ValueError("original diagnostic")
            logs = list((root / "artifacts/run-logs").glob("*/stderr.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("original diagnostic", logs[0].read_text(encoding="utf-8"))
            self.assertIn("Traceback", logs[0].read_text(encoding="utf-8"))
            with self.assertRaisesRegex(RuntimeError, "人工处理"):
                with ModelTask("download", root):
                    self.fail("Failed task was restarted")

    def test_success_allows_next_sequential_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with ModelTask("download", root):
                print("download finished")
            with ModelTask("validate", root):
                print("next task")

    def test_checkpoint_synchronizes_before_short_wait(self):
        events = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def synchronize():
                events.append("synchronize")

        gpu_checkpoint(cuda=FakeCuda(), sleep=lambda seconds: events.append(seconds))
        self.assertEqual(events, ["synchronize", 0.2])


if __name__ == "__main__":
    unittest.main()
