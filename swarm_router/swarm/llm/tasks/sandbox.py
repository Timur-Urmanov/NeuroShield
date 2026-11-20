"""Run untrusted candidate code in a separate `python -I` process.

Isolation: the candidate file is written to a fresh temp dir which is also cwd/HOME/TMPDIR, minimal
env, isolated interpreter (-I: no user site, no PYTHONPATH, script dir not trusted), wall-clock
timeout (whole process group killed, bounded wait for the pipes afterwards), and POSIX rlimits where
available (CPU seconds, address space, file size, open files, number of processes, no core dumps).
This is NOT a security boundary: there is no filesystem isolation (the process can read/write
anything the user can), no network or syscall filtering, and RLIMIT_NPROC is per-user and ignored
for root. It is meant for model-written solutions to small tasks. Run real experiments in a container.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass

try:
    import resource
except ImportError:          # non-POSIX
    resource = None


@dataclass
class RunResult:
    passed: bool
    returncode: int
    timed_out: bool
    stdout: str
    stderr: str


NPROC = 64            # per-user process cap inside the child (blocks fork bombs for non-root users)


def _limits(cpu_s: int, mem_mb: int):
    def f():
        if resource is None:
            return
        for lim, val in ((resource.RLIMIT_CPU, cpu_s), (resource.RLIMIT_AS, mem_mb * 2**20),
                         (resource.RLIMIT_FSIZE, 10 * 2**20), (resource.RLIMIT_NOFILE, 64),
                         (resource.RLIMIT_CORE, 0), (getattr(resource, "RLIMIT_NPROC", None), NPROC)):
            if lim is None:
                continue
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    return f


def run_tests(code: str, tests: list, setup: str = "", timeout: float = 5.0, mem_mb: int = 512) -> RunResult:
    """Candidate passes iff the process exits 0 and prints a per-run random marker after all asserts."""
    marker = "__PASSED_" + secrets.token_hex(8)
    prog = "\n".join([code, "", setup or "", *tests, f"print({marker!r})", ""])
    with tempfile.TemporaryDirectory(prefix="sbx_") as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prog)
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "TMPDIR": d, "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        kw = dict(cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                  text=True)
        if os.name == "posix":
            kw.update(start_new_session=True, preexec_fn=_limits(int(timeout) + 1, mem_mb))
        p = subprocess.Popen([sys.executable, "-I", path], **kw)
        try:
            out, err = p.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL) if os.name == "posix" else p.kill()
            except ProcessLookupError:
                pass
            try:
                out, err = p.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:      # a grandchild escaped the group and holds the pipes
                for f in (p.stdout, p.stderr):
                    try:
                        f.close()
                    except Exception:
                        pass
                out, err = "", "sandbox: pipes not closed after kill"
            timed_out = True
    passed = (not timed_out) and p.returncode == 0 and marker in (out or "")
    return RunResult(passed, p.returncode, timed_out, (out or "")[-2000:], (err or "")[-2000:])
