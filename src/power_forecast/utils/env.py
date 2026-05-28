from __future__ import annotations

import os


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "For local runs, add it to .env. For GitHub Actions, add it as a repository secret."
        )
    return value
