"""Terminal progress emitted by the parent process during long solve stages."""

from datetime import datetime
import sys
from threading import Event, Lock, Thread
from time import monotonic


def _duration(seconds):
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SolveProgress:
    def __init__(self, case_id, heartbeat_seconds=30):
        self.case_id = case_id
        self.heartbeat_seconds = heartbeat_seconds
        self.stream = sys.stdout
        self.started = monotonic()
        self.message = "starting"
        self.sample_started = None
        self.stopped = Event()
        self.lock = Lock()
        self.thread = Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stopped.set()
        self.thread.join()
        if error_type is not None:
            self.log(f"aborted: {error_type.__name__}: {error}")

    def _write(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = _duration(monotonic() - self.started)
        print(
            f"[{timestamp}] [sensitivity_analysis][{self.case_id}] "
            f"{message}; elapsed={elapsed}",
            file=self.stream,
            flush=True,
        )

    def log(self, message):
        with self.lock:
            self.message = message
            self._write(message)

    def samples(self, completed, total, detail=""):
        if self.sample_started is None:
            self.sample_started = monotonic()
        elapsed = monotonic() - self.sample_started
        remaining = (
            _duration(elapsed / completed * (total - completed))
            if completed
            else "pending"
        )
        percentage = 100 * completed / total if total else 100.0
        self.log(
            f"samples: completed={completed}/{total} ({percentage:.1f}%); "
            f"sample_elapsed={_duration(elapsed)}; ETA={remaining}"
            + (f"; {detail}" if detail else "")
        )

    def _heartbeat(self):
        while not self.stopped.wait(self.heartbeat_seconds):
            with self.lock:
                self._write(f"still running; last update: {self.message}")