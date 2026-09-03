from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cybergym_e2b.config import (
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    require_digest_locked_image,
)
from cybergym_e2b.inventory import mutable_project_images

_MANIFEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


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
    ) -> None:
        self._runner = runner
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
        completed = self._runner(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                image,
                "--format",
                "{{json .Manifest}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
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


def create_image_lock(
    upstream: Path,
    *,
    resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Resolve the complete mutable image inventory into a deterministic lock payload."""
    resolver = resolver or DockerBuildxResolver()
    provenance = _validate_provenance(getattr(resolver, "provenance", None))
    required = mutable_project_images(upstream)
    images: dict[str, str] = {}
    for source in required:
        target = resolver(source)
        require_digest_locked_image(target, label=f"resolved image for {source!r}")
        if (
            source == FFMPEG_IMAGE
            and target.split("@", 1)[1] != FFMPEG_IMAGE_DIGEST.split("@", 1)[1]
        ):
            raise ValueError("resolved image for FFmpeg does not preserve the known FFmpeg digest")
        images[source] = target
    if set(images) != set(required):
        raise ValueError("image resolution was incomplete")
    return {
        "schema_version": 1,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "resolver": provenance,
        "images": images,
    }


def write_image_lock(
    upstream: Path,
    output: Path,
    *,
    resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Resolve every image before atomically publishing a complete lock."""
    payload = create_image_lock(upstream, resolver=resolver)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload
