
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from toffee.config import TOOL_TIMEOUT_S

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    elapsed_s: float = 0.0

    @property
    def output(self) -> str:
        return self.stdout if self.success else f"ERROR: {self.stderr}"

    @property
    def nonempty(self) -> bool:
        return bool(self.stdout.strip())


def _run(cmd: list[str], timeout: int = TOOL_TIMEOUT_S, **kw) -> ToolResult:
    tool_name = kw.pop("tool_name", cmd[0])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            start_new_session=True, **kw,
        )
        stdout = proc.stdout
        truncated = False
        if len(stdout) > 50_000:
            stdout = stdout[:50_000] + "\n... [truncated]"
            truncated = True
        return ToolResult(
            tool_name=tool_name, success=proc.returncode == 0,
            stdout=stdout, stderr=proc.stderr, truncated=truncated,
        )
    except subprocess.TimeoutExpired as te:

        try:
            os.killpg(os.getpgid(te.process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError, AttributeError):
            pass
        return ToolResult(tool_name=tool_name, success=False, stderr=f"Timeout after {timeout}s")
    except Exception as exc:
        return ToolResult(tool_name=tool_name, success=False, stderr=str(exc))


def execute_sql(db_path: str, query: str) -> ToolResult:
    if not Path(db_path).is_file():
        return ToolResult(tool_name="execute_sql", success=False, stderr=f"DB not found: {db_path}")
    return _run(["sqlite3", "-readonly", db_path, "-header", "-column", query], tool_name="execute_sql")


def execute_python(code: str) -> ToolResult:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="toffee_") as tmpdir:
        return _run([sys.executable, "-c", code], tool_name="execute_python", cwd=tmpdir)


def list_tables(db_path: str) -> ToolResult:
    res = execute_sql(db_path, ".tables")
    res.tool_name = "list_tables"
    return res


def get_table_schema(db_path: str, table: str) -> ToolResult:
    res = execute_sql(db_path, f"PRAGMA table_info('{table}');")
    res.tool_name = "get_table_schema"
    return res


def _resolve_safe_path(working_dir: Path, rel_path: str) -> Optional[Path]:
    try:
        target = Path(rel_path)
        if target.is_absolute():
            resolved = target.resolve()
        else:
            resolved = (working_dir / rel_path).resolve()
    except Exception:
        return None
    try:
        resolved.relative_to(working_dir)
    except ValueError:
        return None
    return resolved


def read_file(path: str, max_bytes: int = 100_000, working_dir: Optional[str] = None) -> ToolResult:
    try:
        if working_dir:
            safe = _resolve_safe_path(Path(working_dir).resolve(), path)
            if safe is None:
                return ToolResult(tool_name="read_file", success=False,
                                  stderr="Access denied: path outside working directory")
            p = safe
        else:
            p = Path(path)
        if not p.is_file():
            return ToolResult(tool_name="read_file", success=False, stderr=f"Not found: {path}")
        data = p.read_text(errors="replace")
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes] + "\n... [truncated]"
        return ToolResult(tool_name="read_file", success=True, stdout=data, truncated=truncated)
    except Exception as exc:
        return ToolResult(tool_name="read_file", success=False, stderr=str(exc))


def list_directory(path: str = ".", working_dir: Optional[str] = None) -> ToolResult:
    try:
        if working_dir:
            safe = _resolve_safe_path(Path(working_dir).resolve(), path)
            if safe is None:
                return ToolResult(tool_name="list_directory", success=False,
                                  stderr="Access denied: path outside working directory")
            p = safe
        else:
            p = Path(path)
        if not p.is_dir():
            return ToolResult(tool_name="list_directory", success=False, stderr=f"Not a directory: {path}")
        entries = sorted(e.name for e in p.iterdir())
        return ToolResult(tool_name="list_directory", success=True, stdout="\n".join(entries))
    except Exception as exc:
        return ToolResult(tool_name="list_directory", success=False, stderr=str(exc))


def write_file(
    path: str,
    content: str,
    mode: str = "w",
    working_dir: Optional[str] = None,
) -> ToolResult:
    if mode not in ("w", "a"):
        return ToolResult(tool_name="write_file", success=False, stderr=f"Bad mode: {mode}")
    try:
        if working_dir:
            safe = _resolve_safe_path(Path(working_dir).resolve(), path)
            if safe is None:
                return ToolResult(tool_name="write_file", success=False,
                                  stderr="Access denied: path outside working directory")
            p = safe
        else:
            p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode, encoding="utf-8") as f:
            f.write(content or "")
        action = "overwrite" if mode == "w" else "append"
        return ToolResult(
            tool_name="write_file", success=True,
            stdout=f"{action}: wrote {len((content or '').encode('utf-8'))} bytes to {p.name}",
        )
    except Exception as exc:
        return ToolResult(tool_name="write_file", success=False, stderr=str(exc))


_BASH_FORBIDDEN_PATTERNS = (
    "rm -rf /", "rm -rf ~", "rm -rf *", "mkfs", "dd if=", ":(){", "fork bomb",
    "shutdown", "reboot", "poweroff", "halt", "init 0", "init 6",
    "chmod -r 777 /", "chmod -r 000 /", "curl|sh", "wget|sh", "curl|bash", "wget|bash",
    "> /dev/sd", "mv /", "sudo ", "/etc/passwd", "/etc/shadow",
)


_BASH_MAX_TIMEOUT = 120


def run_bash(
    command: str,
    timeout: int = TOOL_TIMEOUT_S,
    cwd: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> ToolResult:
    if not command or not command.strip():
        return ToolResult(tool_name="run_bash", success=False, stderr="command is required")


    cmd_norm = command.lower().replace(" ", "").replace("\t", "")
    for pat in _BASH_FORBIDDEN_PATTERNS:
        if pat.lower().replace(" ", "") in cmd_norm:
            return ToolResult(tool_name="run_bash", success=False,
                              stderr="Command blocked for safety")

    timeout = max(1, min(int(timeout or TOOL_TIMEOUT_S), _BASH_MAX_TIMEOUT))

    if working_dir:
        base = Path(working_dir).resolve()
        if cwd:
            safe_cwd = _resolve_safe_path(base, cwd)
            run_cwd = safe_cwd if (safe_cwd and safe_cwd.is_dir()) else base
        else:
            run_cwd = base
    else:
        run_cwd = Path(cwd).resolve() if cwd else None

    return _run(["bash", "-c", command], timeout=timeout, tool_name="run_bash",
                cwd=str(run_cwd) if run_cwd else None)


TOOL_DISPATCH = {
    "execute_sql": execute_sql,
    "execute_python": execute_python,
    "list_tables": list_tables,
    "get_table_schema": get_table_schema,
    "read_file": read_file,
    "list_directory": list_directory,
    "write_file": write_file,
    "run_bash": run_bash,
}


_ARG_ALIASES = {
    "execute_sql": {"sql": "query", "sql_query": "query"},
    "get_table_schema": {"table_name": "table", "name": "table"},
    "execute_python": {"script": "code", "python_code": "code", "source": "code"},
    "read_file": {"file_path": "path", "filename": "path", "file": "path"},
    "list_directory": {"dir_path": "path", "directory": "path", "dir": "path"},
    "write_file": {"filename": "path", "file_path": "path", "file": "path", "data": "content", "text": "content"},
    "run_bash": {"cmd": "command", "shell": "command", "bash": "command"},
}


_ALLOWED_ARGS = {
    "execute_sql": {"db_path", "query"},
    "execute_python": {"code"},
    "list_tables": {"db_path"},
    "get_table_schema": {"db_path", "table"},
    "read_file": {"path", "max_bytes", "working_dir"},
    "list_directory": {"path", "working_dir"},
    "write_file": {"path", "content", "mode", "working_dir"},
    "run_bash": {"command", "timeout", "cwd", "working_dir"},
}


_SANDBOXED_TOOLS = {"read_file", "list_directory", "write_file", "run_bash"}


def _normalise_args(tool_name: str, arguments: dict) -> dict:
    aliases = _ARG_ALIASES.get(tool_name, {})
    out = {}
    for k, v in arguments.items():
        out[aliases.get(k, k)] = v
    allowed = _ALLOWED_ARGS.get(tool_name)
    if allowed is not None:
        out = {k: v for k, v in out.items() if k in allowed}
    return out


def execute_tool(tool_name: str, arguments: dict, env_context: Optional[dict] = None) -> ToolResult:
    import time as _time
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return ToolResult(tool_name=tool_name, success=False, stderr=f"Unknown tool: {tool_name}")

    arguments = _normalise_args(tool_name, arguments)

    if env_context:

        if "db_path" in env_context:
            if tool_name in ("execute_sql", "list_tables", "get_table_schema") and "db_path" not in arguments:
                arguments = {**arguments, "db_path": env_context["db_path"]}
            if tool_name in ("list_directory", "read_file") and "path" not in arguments:
                default_path = (
                    env_context.get("working_dir")
                    or os.path.dirname(env_context["db_path"])
                )
                arguments = {**arguments, "path": default_path}


        if tool_name in _SANDBOXED_TOOLS and "working_dir" not in arguments:
            wd = env_context.get("working_dir")
            if wd is None and env_context.get("db_path"):
                wd = os.path.dirname(env_context["db_path"])
            if wd:
                arguments = {**arguments, "working_dir": wd}

    t0 = _time.perf_counter()
    try:
        result = fn(**arguments)
    except TypeError as exc:
        return ToolResult(tool_name=tool_name, success=False, stderr=f"Bad arguments: {exc}",
                          elapsed_s=_time.perf_counter() - t0)
    result.elapsed_s = _time.perf_counter() - t0
    return result
