
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from toffee.utils import compute_cost

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0


_CACHE_MIN_CHARS = 4000


def _should_cache(model: str, content: str) -> bool:
    if not content or len(content) < _CACHE_MIN_CHARS:
        return False
    lower = model.lower()
    return "claude" in lower or "anthropic" in lower


def _apply_cache_marker(messages: List[Dict], model: str) -> List[Dict]:
    out = []
    for msg in messages:
        content = msg.get("content", "")

        if not isinstance(content, str):
            out.append(msg)
            continue
        if msg.get("role") == "system" and _should_cache(model, content):
            out.append({
                **msg,
                "content": [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }],
            })
        else:
            out.append(msg)
    return out


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    finish_reason: str = ""


class OpenRouterClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        import httpx
        self._client = OpenAI(
            base_url=base_url,
            api_key=self._api_key,
            http_client=httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)),
        )
        self._stream = False
        self._anthropic = None

    def call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, Usage]:
        cached_messages = _apply_cache_marker(messages, model)


        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                create_kwargs = dict(
                    model=model,
                    messages=cached_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/toffee-synthesis",
                        "X-Title": "TOFFEE Synthesis Pipeline",
                    },
                    timeout=300,
                )
                if stop:
                    create_kwargs["stop"] = stop
                if self._stream:
                    create_kwargs["stream"] = True
                    create_kwargs["stream_options"] = {"include_usage": True}

                response = self._client.chat.completions.create(**create_kwargs)

                if self._stream:
                    parts = []
                    finish_reason = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    for chunk in response:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                parts.append(delta.content)
                            fr = getattr(chunk.choices[0], "finish_reason", None)
                            if fr:
                                finish_reason = fr
                        if chunk.usage:
                            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                            completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                    content = "".join(parts)
                else:


                    choices = getattr(response, "choices", None)
                    if not choices:
                        err = getattr(response, "error", None)
                        raise RuntimeError(
                            f"OpenRouter returned no choices for {model}: "
                            f"{err or repr(response)[:300]}"
                        )
                    choice = choices[0]
                    msg = getattr(choice, "message", None)
                    content = (getattr(msg, "content", None) or "") if msg else ""
                    finish_reason = getattr(choice, "finish_reason", "") or ""
                    raw_usage = getattr(response, "usage", None)
                    prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0

                cost = compute_cost(prompt_tokens, completion_tokens, model)
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    model=model,
                    finish_reason=finish_reason,
                )
                return content, usage

            except Exception as exc:
                last_exc = exc

                is_timeout = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                if is_timeout:
                    log.warning("OpenRouter call timed out (attempt %d/%d): %s — not retrying",
                                attempt, _MAX_RETRIES, exc)
                    raise
                if attempt < _MAX_RETRIES:
                    wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                    log.warning(
                        "OpenRouter call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, _MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error("OpenRouter call failed after %d attempts: %s", _MAX_RETRIES, exc)

        raise RuntimeError(f"OpenRouter call failed after {_MAX_RETRIES} retries") from last_exc
