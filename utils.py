
from __future__ import annotations

import hashlib
import json
import os
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from toffee.config import DATA_SOURCE_CATALOG, MODEL_COSTS_PER_M

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FMT))
        root.addHandler(ch)

    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve()) for h in root.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FMT))
        root.addHandler(fh)


def sha256_short(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def compute_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    pin, pout = MODEL_COSTS_PER_M.get(model, (0.0, 0.0))
    return prompt_tokens * pin / 1_000_000 + completion_tokens * pout / 1_000_000


def load_catalog(path: Optional[str] = None) -> List[Dict[str, Any]]:
    from toffee.config import PACKAGE_ROOT
    p = path or DATA_SOURCE_CATALOG
    with open(p) as f:
        data = json.load(f)
    sources = data.get("sources", data) if isinstance(data, dict) else data
    for src in sources:
        df = src.get("data_file", "")
        if df and not os.path.isabs(df) and not os.path.exists(df):
            src["data_file"] = os.path.join(PACKAGE_ROOT, df)
        src["extra_sources"] = [
            e if os.path.isabs(e) or os.path.exists(e) else os.path.join(PACKAGE_ROOT, e)
            for e in (src.get("extra_sources") or [])
        ]
    return sources


def extract_json_string_field(blob: str, field: str) -> str:
    if not blob:
        return ""

    needle = f'"{field}"'
    pos = 0
    while True:
        idx = blob.find(needle, pos)
        if idx < 0:
            return ""

        after = blob[idx + len(needle):].lstrip()
        if not after.startswith(":"):
            pos = idx + 1
            continue
        after = after[1:].lstrip()
        if not after.startswith('"'):
            pos = idx + 1
            continue

        start = blob.index('"', blob.index(":", idx + len(needle)) + 1)
        i = start + 1
        while i < len(blob):
            ch = blob[i]
            if ch == '\\':
                i += 2
            elif ch == '"':
                raw = blob[start:i + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw[1:-1]
            else:
                i += 1
        return ""


def is_metadata_sql_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = (
        "pragma ",
        "select sql from sqlite_master",
        "select name, sql from sqlite_master",
        "from sqlite_master",
        "pragma_table_info(",
        ".tables",
        ".schema",
    )
    return any(pattern in q for pattern in patterns)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class BudgetTracker:

    def __init__(self, max_dollars: float = 8.00, max_tokens: int = 500_000):
        self.max_dollars = max_dollars
        self.max_tokens = max_tokens
        self.spent_dollars = 0.0
        self.spent_tokens = 0
        self.spent_exec_dollars = 0.0
        self.exec_seconds = 0.0
        self.exec_count = 0
        self._lock = threading.Lock()

    def record(self, cost: float, tokens: int) -> None:
        with self._lock:
            self.spent_dollars += cost
            self.spent_tokens += tokens

    def record_execution(self, seconds: float, vcpus: float = 1.0) -> float:
        """Metered tool execution, priced at EXEC_VCPU_RATE_PER_HOUR and
        charged against the same per-task budget as model calls."""
        from toffee.config import EXEC_VCPU_RATE_PER_HOUR
        cost = max(0.0, float(seconds)) / 3600.0 * vcpus * EXEC_VCPU_RATE_PER_HOUR
        with self._lock:
            self.spent_dollars += cost
            self.spent_exec_dollars += cost
            self.exec_seconds += max(0.0, float(seconds))
            self.exec_count += 1
        return cost

    def exhausted(self) -> bool:
        with self._lock:
            return self.spent_dollars >= self.max_dollars or self.spent_tokens >= self.max_tokens

    def remaining_dollar_ratio(self) -> float:
        with self._lock:
            return max(0.0, 1.0 - self.spent_dollars / self.max_dollars)

    def remaining_token_ratio(self) -> float:
        with self._lock:
            return max(0.0, 1.0 - self.spent_tokens / self.max_tokens)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def extract_answer_text(content: str) -> str:
    text = content or ""
    end = text.find("</think>")
    if end >= 0:
        text = text[end + len("</think>"):]
    text = text.strip()
    try:
        parsed = json.loads(strip_code_fences(text))
        if isinstance(parsed, dict):
            for key in ("answer", "visible_text"):
                v = str(parsed.get(key) or "").strip()
                if v:
                    return v
    except (json.JSONDecodeError, TypeError):
        pass
    marker = text.find('{"tool_name"')
    if marker > 0:
        text = text[:marker].strip()
    return text
