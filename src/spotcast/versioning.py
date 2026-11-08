"""Which code produced a result: the git commit, and whether there were uncommitted changes."""

import subprocess

from spotcast.config import PROJECT_ROOT


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def code_version() -> dict[str, str]:
    """'dirty' means there were uncommitted changes, so the commit alone can't reproduce it."""
    commit = _git("rev-parse", "--short", "HEAD")
    changes = _git("status", "--porcelain")
    return {
        "git_commit": commit or "unknown",
        "git_dirty": "unknown" if changes is None else str(bool(changes)).lower(),
    }
