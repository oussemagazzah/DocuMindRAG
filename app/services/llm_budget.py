"""Per-process daily cap on LLM calls, to protect a free-tier API quota in the public demo."""

import threading
from datetime import datetime, timezone


class DailyBudget:
    def __init__(self, limit: int):
        self.limit = limit  # 0 = unlimited
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll(self):
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def try_consume(self) -> bool:
        if self.limit <= 0:
            return True
        with self._lock:
            self._roll()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int | None:
        if self.limit <= 0:
            return None
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)
