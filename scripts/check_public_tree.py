"""Reject tracked files that do not belong in the public repository."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

_GENERATED_DIRECTORIES = {
    ".e2b",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "vendor",
}

_LOCAL_ENVIRONMENT_DIRECTORIES = {".venv", "ENV", "env", "venv"}
_FORBIDDEN_FILENAMES = {
    "credentials.json",
    "credentials.toml",
    "credentials.yaml",
    "credentials.yml",
    "e2b.toml",
}
_PRIVATE_KEY_SUFFIXES = {".key", ".p12", ".pfx"}
_GENERATED_FILE_PATTERNS = (
    re.compile(r"^template-manifest(?:\..+)?\.json$"),
    re.compile(r"^build-ledger(?:\..+)?\.json$"),
)


def _is_forbidden(path: str) -> bool:
    normalized = PurePosixPath(path).as_posix()
    parts = PurePosixPath(normalized).parts
    filename = parts[-1] if parts else ""

    if filename in _FORBIDDEN_FILENAMES or PurePosixPath(filename).suffix in _PRIVATE_KEY_SUFFIXES:
        return True
    if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example"):
        return True
    if any(pattern.fullmatch(filename) for pattern in _GENERATED_FILE_PATTERNS):
        return True
    return any(
        part in _GENERATED_DIRECTORIES
        or part in _LOCAL_ENVIRONMENT_DIRECTORIES
        or part.endswith(".egg-info")
        for part in parts[:-1]
    )


def find_forbidden_paths(paths: Iterable[str]) -> list[str]:
    """Return sorted tracked paths that violate the publication policy."""
    return sorted(path for path in paths if _is_forbidden(path))


def _publication_paths(root: Path) -> list[str]:
    """List Git-tracked paths or, for an archive, every extracted filesystem entry."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        top = None
    if top is not None and top.returncode == 0 and Path(top.stdout.strip()).resolve() == root:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return [path for path in completed.stdout.split("\0") if path]
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def main(root: Path | None = None) -> int:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    forbidden = find_forbidden_paths(_publication_paths(root))
    if not forbidden:
        print("Public-tree policy passed.")
        return 0

    print("Public-tree policy rejected these tracked paths:")
    for path in forbidden:
        print(f"- {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
