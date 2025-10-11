# src/tsforecast/utils/progress.py
from __future__ import annotations
import time
from typing import Optional, Dict

def _fmt_secs(s: float) -> str:
    s = int(max(0, s))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

class ProgressTracker:
    """
    Minimal, notebook-freundliches Progress-Logging mit ETA.
    """
    def __init__(self, name: str, total_units: Optional[int], print_every: int = 1):
        self.name = name
        self.total = total_units
        self.print_every = max(1, int(print_every))
        self.start = time.perf_counter()
        self.last = self.start
        self.done = 0

    def update(self, n: int = 1, extra: Optional[Dict] = None):
        self.done += n
        now = time.perf_counter()
        elapsed = now - self.start
        if (self.done % self.print_every) == 0 or self.done == self.total:
            rate = self.done / elapsed if elapsed > 0 else 0.0
            if self.total:
                rem = (self.total - self.done) / rate if rate > 0 else 0.0
                pct = 100.0 * self.done / self.total
                msg = f"[{self.name}] {self.done}/{self.total} ({pct:4.1f}%)  elapsed={_fmt_secs(elapsed)}  eta={_fmt_secs(rem)}  rate={rate:5.2f}/s"
            else:
                msg = f"[{self.name}] {self.done}  elapsed={_fmt_secs(elapsed)}  rate={rate:5.2f}/s"
            if extra:
                tail = "  " + " ".join(f"{k}={v}" for k, v in extra.items())
            else:
                tail = ""
            print(msg + tail, flush=True)
        self.last = now

    def finish(self, note: str = ""):
        elapsed = time.perf_counter() - self.start
        if self.total:
            print(f"[{self.name}] done in {_fmt_secs(elapsed)} (total={self.total}) {note}", flush=True)
        else:
            print(f"[{self.name}] done in {_fmt_secs(elapsed)} {note}", flush=True)
