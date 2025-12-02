"""Run opt-in local Ray monitoring tests with a strict 30-second process limit."""

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from geotrace.logging_config import configure_logging


def main() -> int:
    """Run opt-in Ray checks with a deadline and preserve their complete log."""
    configure_logging()
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, GEOTRACE_TEST_RAY="1")
    if sys.platform == "win32":
        # Detached Windows children can have invalid native stdout handles.
        environment["RAY_LOG_TO_STDERR"] = "1"
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_monitoring.py",
        "-v",
    ]
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = process.communicate(timeout=30)
        assert process.returncode is not None
        code = process.returncode
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False
            )
        else:
            kill_group = os.killpg
            kill_group(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=5)
        output += "\nRay integration exceeded the 30-second limit.\n"
        code = 124
    target = root / "outputs" / "ray-monitoring-smoke.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    logging.getLogger(__name__).info(
        "ray_check_finished", extra={"returncode": code, "log_file": str(target)}
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
