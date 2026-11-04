"""Fail if any git-tracked (or, outside git, any non-ignored-looking) file exceeds --max-mb.

Run before pushing:  python scripts/check_large_files.py --max-mb 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SKIP = {".git", ".venv", "data", "runs", "__pycache__", "data_synthetic", "runs_smoke"}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
        files = [Path(p) for p in out.stdout.decode().split("\0") if p]
        if files:
            return files
    except Exception:
        pass
    return [p for p in Path(".").rglob("*") if p.is_file() and not (set(p.parts) & SKIP)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=50)
    a = ap.parse_args()
    big = [(p, p.stat().st_size / 2**20) for p in tracked_files() if p.exists()]
    big = [(p, s) for p, s in big if s > a.max_mb]
    for p, s in big:
        print(f"TOO LARGE: {p} ({s:.1f} MB > {a.max_mb} MB)")
    if not big:
        print(f"OK: no file over {a.max_mb} MB")
    return 1 if big else 0


if __name__ == "__main__":
    sys.exit(main())
