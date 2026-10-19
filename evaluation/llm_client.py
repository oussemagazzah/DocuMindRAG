"""Throttled, retrying, cached Gemini calls for the evaluation.

The free tier allows only a few requests per minute, so calls are spaced by
`min_interval_s` and retried on 429/5xx. Successful responses are cached on
disk (keyed by model, generation settings and prompt), so an interrupted run
can resume without spending quota twice; pass use_cache=False to force fresh calls.
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from google.api_core import exceptions as gexc

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("eval.llm")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".eval_cache"
RETRYABLE = (gexc.ResourceExhausted, gexc.ServiceUnavailable, gexc.InternalServerError,
             gexc.DeadlineExceeded, gexc.TooManyRequests)


class QuotaExhausted(RuntimeError):
    """The API asked us to wait longer than the call budget (e.g. the free tier's daily limit)."""


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    latency_ms: float | None  # None when served from cache
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached: bool = False


class GeminiClient:
    def __init__(self, min_interval_s: float = 6.5, max_attempts: int = 4, use_cache: bool = True,
                 attempt_timeout_s: float = 90, call_budget_s: float = 360):
        genai.configure(api_key=get_settings().gemini_api_key)
        self.min_interval_s = min_interval_s
        self.max_attempts = max_attempts
        # The SDK's own timeout is not always honoured, so each attempt also runs
        # under a hard wall-clock limit, and a whole call has a total budget.
        self.attempt_timeout_s = attempt_timeout_s
        self.call_budget_s = call_budget_s
        self._pool = ThreadPoolExecutor(max_workers=4)
        self.use_cache = use_cache
        self._last_call = 0.0
        CACHE_DIR.mkdir(exist_ok=True)

    def _wait_turn(self):
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def generate(self, model: str, system: str, prompt: str, temperature: float, max_output_tokens: int,
                 json_output: bool = False) -> LLMResult:
        key = hashlib.sha256(json.dumps([model, system, prompt, temperature, max_output_tokens, json_output]).encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return LLMResult(**(data | {"cached": True}))

        config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens,
                                        response_mime_type="application/json" if json_output else None)
        gm = genai.GenerativeModel(model_name=model, system_instruction=system)
        deadline = time.monotonic() + self.call_budget_s
        for attempt in range(1, self.max_attempts + 1):
            self._wait_turn()
            start = time.perf_counter()
            future = self._pool.submit(gm.generate_content, prompt, generation_config=config,
                                       request_options={"timeout": self.attempt_timeout_s})
            try:
                response = future.result(timeout=self.attempt_timeout_s + 5)
            except (*RETRYABLE, FutureTimeout) as e:
                delay = _retry_delay(e, attempt)
                if isinstance(e, gexc.ResourceExhausted) and delay > self.call_budget_s:
                    raise QuotaExhausted(f"{model}: quota exhausted, API says retry in {delay / 3600:.1f} h") from e
                if time.monotonic() + delay > deadline:
                    break
                print(f"    {model}: {type(e).__name__} on attempt {attempt}, retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            candidate = response.candidates[0] if response.candidates else None
            try:
                text = response.text
            except ValueError:  # no text parts (blocked, or all tokens spent on thinking)
                text = ""
            usage = response.usage_metadata
            result = LLMResult(
                text=text,
                finish_reason=candidate.finish_reason.name if candidate else "NO_CANDIDATE",
                latency_ms=round(latency_ms, 1),
                prompt_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count,
            )
            cache_file.write_text(json.dumps(result.__dict__ | {"cached": False}), encoding="utf-8")
            return result
        raise RuntimeError(f"{model}: gave up (attempts or {self.call_budget_s:.0f}s budget exhausted)")


def _retry_delay(error: Exception, attempt: int) -> float:
    match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(error))
    if match:
        return float(match.group(1)) + 2
    return min(10 * 2 ** (attempt - 1), 120)
