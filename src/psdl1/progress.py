"""진행 상황 JSON 줄. import_png와 같은 형식: {"event": "progress", "phase": ..., ...}를 stderr로 즉시 낸다."""
from __future__ import annotations

import json
import sys
import time


def stderr_sink(record):
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


class Progress:
    """sink가 없으면 아무것도 하지 않는다. due()는 interval초마다 한 번만 True라 배치 진행 줄이 넘치지 않는다."""

    def __init__(self, sink=None, interval=10.0):
        self.sink, self.interval, self._last = sink, interval, time.monotonic()

    def emit(self, phase, **record):
        if self.sink is not None:
            self.sink({"event": "progress", "phase": phase, **record})

    def due(self):
        now = time.monotonic()
        if self.sink is None or now - self._last < self.interval:
            return False
        self._last = now
        return True
