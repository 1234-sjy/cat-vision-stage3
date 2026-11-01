"""One model task at a time, full logs, and no automatic recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from .common import ROOT, save_json


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.logfile.write(text)
        self.logfile.flush()
        return self.console.write(text)

    def flush(self):
        self.logfile.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class ModelTask:
    """An OS lock prevents overlap; a failed/interrupted run blocks later tasks.

    The state file is intentionally not cleared automatically after a failure.
    Human-directed recovery must inspect the logs before changing its state.
    """
    def __init__(self, name: str, root: Path = ROOT):
        self.name, self.root = name, root
        self.lock = None

    def _release_lock(self):
        if self.lock is not None:
            self.lock.close()  # OS releases the byte/flock lock, including on process death.
            self.lock = None

    def __enter__(self):
        runtime = self.root / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        self.state_path = runtime / "model-task-state.json"
        self.lock = (runtime / "model-task.lock").open("a+b")
        if self.lock.tell() == 0:
            self.lock.write(b"\0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._release_lock()
            raise RuntimeError("已有模型任务运行，下载与GPU任务不得同时执行") from exc
        try:
            if self.state_path.exists():
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
                if previous.get("state") in {"running", "failed", "interrupted"}:
                    raise RuntimeError("上次模型任务失败或中断，须先人工处理日志；禁止自动重试或重启")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.log_dir = self.root / "artifacts/run-logs" / f"{stamp}-{self.name}-{uuid4().hex[:8]}"
            self.log_dir.mkdir(parents=True)
            self.stdout_log = (self.log_dir / "stdout.log").open("w", encoding="utf-8", buffering=1)
            self.stderr_log = (self.log_dir / "stderr.log").open("w", encoding="utf-8", buffering=1)
            self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(sys.stdout, self.stdout_log)
            sys.stderr = _Tee(sys.stderr, self.stderr_log)
            self.state = {"task": self.name, "state": "running", "pid": os.getpid(),
                          "created_utc": datetime.now(timezone.utc).isoformat(),
                          "logs": self.log_dir.relative_to(self.root).as_posix(),
                          "automatic_retry": False, "automatic_restart": False,
                          "argv": sys.argv}
            save_json(self.state_path, self.state)
            save_json(self.log_dir / "run.json", self.state)
            print(f"TASK START {self.name}; logs={self.state['logs']}", flush=True)
            return self
        except BaseException:
            self._release_lock()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            self.state.update(state="failed" if exc is not None else "completed",
                              finished_utc=datetime.now(timezone.utc).isoformat())
            if exc is not None:
                self.state["error"] = f"{exc_type.__name__}: {exc}"
            save_json(self.state_path, self.state)
            save_json(self.log_dir / "run.json", self.state)
            print(f"TASK END {self.name}; state={self.state['state']}", flush=True)
        finally:
            sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
            self.stdout_log.close()
            self.stderr_log.close()
            self._release_lock()
        return False


def gpu_checkpoint(*, cuda=None, sleep=time.sleep, pause_seconds=0.2):
    """Synchronize a finished GPU batch, then briefly yield the machine."""
    if cuda is None:
        import torch
        cuda = torch.cuda
    if cuda.is_available():
        cuda.synchronize()
    sleep(pause_seconds)


def set_low_load_threads():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(2)


def require_training_enabled():
    policy = json.loads((ROOT / "configs/runtime-policy.json").read_text(encoding="utf-8"))
    if not policy.get("allow_formal_training", False):
        raise RuntimeError("按用户要求暂停正式训练；当前只允许单任务下载和低负载验证")
