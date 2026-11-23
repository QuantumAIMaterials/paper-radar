#!/usr/bin/env python3
"""Compatibility entry for the shared Mac publisher; never deploys a service."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'radar'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise SystemExit("Commit tracked changes before publishing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        return subprocess.call(["lab-release", "check" if args.check else "sync", PROJECT,
                                "--expect-sha", commit])
    except FileNotFoundError:
        raise SystemExit("Install lab-release-tools first; see docs/release-workflow.md") from None


if __name__ == "__main__":
    raise SystemExit(main())
