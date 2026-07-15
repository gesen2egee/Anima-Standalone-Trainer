"""Start the Node UI and give it a Python parent process to monitor."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    ui_dir = Path(__file__).resolve().parent
    node = shutil.which("node")
    if not node:
        print("[ERROR] 找不到 Node.js。", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["ANIMA_PYTHON_PID"] = str(os.getpid())
    child = subprocess.Popen(
        [node, str(ui_dir / "server.js"), *sys.argv[1:]],
        cwd=ui_dir,
        env=env,
    )
    try:
        return child.wait()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
