from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from cybergym_e2b.config import (
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    is_digest_locked_image,
    require_digest_locked_image,
)
from cybergym_e2b.inventory import mutable_project_images, resolve_task

_MANIFEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_RATE_LIMIT_MARKERS = ("429", "Too Many Requests", "TOOMANYREQUESTS")
_RATE_LIMIT_BACKOFF_SECONDS = (15.0, 45.0, 90.0)


def _repository(image: str) -> str:
    """Strip a tag from an OCI reference without confusing a registry port for a tag."""
    without_digest = image.split("@", 1)[0]
    final_slash = without_digest.rfind("/")
    final_colon = without_digest.rfind(":")
    return without_digest[:final_colon] if final_colon > final_slash else without_digest


class DockerBuildxResolver:
    """Resolve registry manifests without downloading image layers."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._sleep = sleep
        version = runner(
            ["docker", "buildx", "version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not version:
            raise ValueError("docker buildx did not report a resolver version")
        self.provenance = {
            "tool": "docker buildx imagetools inspect",
            "version": version,
            "method": "manifest-descriptor-digest",
        }

    def __call__(self, image: str) -> str:
        command = [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image,
            "--format",
            "{{json .Manifest}}",
        ]
        for attempt, backoff in enumerate((*_RATE_LIMIT_BACKOFF_SECONDS, None)):
            try:
                completed = self._runner(command, check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or "").strip()
                rate_limited = any(marker in detail for marker in _RATE_LIMIT_MARKERS)
                if rate_limited and backoff is not None:
                    self._sleep(backoff)
                    continue
                raise RuntimeError(
                    f"registry inspection failed for {image!r}"
                    f"{' after ' + str(attempt + 1) + ' attempts' if rate_limited else ''}: "
                    f"{detail or 'no error output'}"
                ) from exc
        try:
            descriptor = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"registry inspection did not return one unambiguous manifest for {image!r}"
            ) from exc
        digest = descriptor.get("digest") if isinstance(descriptor, dict) else None
        if not isinstance(digest, str) or not _MANIFEST_DIGEST.fullmatch(digest):
            raise ValueError(
                f"registry inspection did not return one unambiguous SHA-256 manifest digest "
                f"for {image!r}"
            )
        return require_digest_locked_image(
            f"{_repository(image)}@{digest}", label=f"resolved image for {image!r}"
        )


def _validate_provenance(raw: Any) -> dict[str, str]:
    required = {"tool", "version", "method"}
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or not all(isinstance(raw[key], str) and raw[key] for key in required)
    ):
        raise ValueError("image resolver provenance must identify a tool, version, and method")
    return {key: raw[key] for key in ("tool", "version", "method")}


def _required_images(upstream: Path, tasks: Iterable[str] | None) -> tuple[str, ...]:
    if not tasks:
        return mutable_project_images(upstream)
    images = {resolve_task(upstream, task).build_image for task in tasks}
    return tuple(sorted(image for image in images if not is_digest_locked_image(image)))


def create_image_lock(
    upstream: Path,
    *,
    resolver: Callable[[str], str] | None = None,
    tasks: Iterable[str] | None = None,
    existing: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve mutable project images into a deterministic lock payload.

    Without ``tasks`` the whole inventory is resolved. With ``tasks`` only the images those
    tasks need are resolved and merged over ``existing`` entries, so a rate-limited registry
    can be walked in slices.
    """
    resolver = resolver or DockerBuildxResolver()
    provenance = _validate_provenance(getattr(resolver, "provenance", None))
    required = _required_images(upstream, tasks)
    images: dict[str, str] = dict(existing or {}) if tasks else {}
    for source in required:
        if source in images:
            continue
        target = resolver(source)
        require_digest_locked_image(target, label=f"resolved image for {source!r}")
        if (
            source == FFMPEG_IMAGE
            and target.split("@", 1)[1] != FFMPEG_IMAGE_DIGEST.split("@", 1)[1]
        ):
            raise ValueError("resolved image for FFmpeg does not preserve the known FFmpeg digest")
        images[source] = target
    if not set(required) <= set(images):
        raise ValueError("image resolution was incomplete")
    return {
        "schema_version": 1,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "resolver": provenance,
        "images": dict(sorted(images.items())),
    }


def write_image_lock(
    upstream: Path,
    output: Path,
    *,
    resolver: Callable[[str], str] | None = None,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Resolve the requested images, then atomically publish the lock.

    A task-scoped run merges into an existing lock for the same upstream inputs.
    """
    existing: dict[str, str] | None = None
    if tasks and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(previous, dict) and previous.get("upstream") == {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        }:
            existing = previous.get("images") or {}
    payload = create_image_lock(upstream, resolver=resolver, tasks=tasks, existing=existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload
