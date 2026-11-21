#!/usr/bin/env python3
"""Publish this commit to the organization only after personal-repository CI."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = "paper-radar"
WORKFLOW = "ci.yml"
ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without pushing")
    args = parser.parse_args()
    if run("git", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Commit tracked changes before publishing.")
    branch = run("git", "symbolic-ref", "--short", "HEAD")
    sha = run("git", "rev-parse", "HEAD")
    endpoint = (f"repos/feiniao0071/{REPOSITORY}/actions/workflows/{WORKFLOW}/runs"
                f"?head_sha={sha}&event=push&per_page=100")
    data = json.loads(run("gh", "api", endpoint))
    runs = [r for r in data.get("workflow_runs", [])
            if r.get("head_sha") == sha and r.get("event") == "push"
            and r.get("path") == f".github/workflows/{WORKFLOW}"]
    if not runs:
        raise ValueError("No personal-repository CI run for this commit. Push there first.")
    latest = max(runs, key=lambda r: (r["id"], r.get("run_attempt", 1)))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError(f"CI is {latest.get('status')}/{latest.get('conclusion')}: "
                         + latest["html_url"])
    print(f"Verified {sha}: {latest['html_url']}", flush=True)
    if args.check:
        return
    if run("git", "rev-parse", "HEAD") != sha or run(
            "git", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Source changed during verification; retry from the new commit.")
    target = f"https://github.com/QuantumAIMaterials/{REPOSITORY}.git"
    subprocess.run(["git", "push", target, f"{sha}:refs/heads/{branch}"],
                   cwd=ROOT, check=True)
    actual = run("git", "ls-remote", target, f"refs/heads/{branch}").split()[0]
    if actual != sha:
        raise ValueError("Organization branch changed; inspect before retrying.")
    print(f"Published and verified {REPOSITORY}:{branch} at {sha}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, KeyError, IndexError) as error:
        print(f"Publish stopped: {error}", file=sys.stderr)
        sys.exit(1)
