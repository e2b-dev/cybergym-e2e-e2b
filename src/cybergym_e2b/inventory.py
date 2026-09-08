from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from cybergym_e2b.config import (
    DEFAULT_BUILD_IMAGE,
    DEFAULT_PATCH_FILE,
    DEFAULT_REMOTE_APT_RETRY,
    DEFAULT_REMOTE_INSTALL_CODEX,
    DEFAULT_REMOTE_SMOKE,
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    is_digest_locked_image,
    normalize_task,
    require_digest_locked_image,
)


@dataclass(frozen=True)
class ResolvedTask:
    task: str
    project: str
    task_id: str
    build_image: str
    runtime_image: str
    repo_to_patch: str


def resolve_task(
    upstream: Path,
    task: str,
    *,
    image_map: dict[str, str] | None = None,
) -> ResolvedTask:
    task = normalize_task(task)
    project, task_id = task.split("/", 1)
    project_dir = upstream / "projects" / project
    task_dir = project_dir / task_id
    project_config = project_dir / "project.toml"
    task_config = task_dir / "config.toml"
    if not project_config.is_file() or not task_config.is_file():
        raise FileNotFoundError(f"task is absent from pinned CyberGym-E2E checkout: {task}")
    merged = tomllib.loads(project_config.read_text(encoding="utf-8"))
    merged.update(tomllib.loads(task_config.read_text(encoding="utf-8")))
    build_image = merged.get("build_image", DEFAULT_BUILD_IMAGE)
    runtime_image = (image_map or {}).get(
        build_image,
        FFMPEG_IMAGE_DIGEST if build_image == FFMPEG_IMAGE else build_image,
    )
    return ResolvedTask(
        task=task,
        project=project,
        task_id=task_id,
        build_image=build_image,
        runtime_image=runtime_image,
        repo_to_patch=merged["repo_to_patch"],
    )


def project_images(upstream: Path) -> tuple[str, ...]:
    """Return every unique effective build image in deterministic order."""
    return tuple(sorted({row["build_image"] for row in inventory(upstream)["images"]}))


def mutable_project_images(upstream: Path) -> tuple[str, ...]:
    """Return the unique upstream project images that still require resolution."""
    return tuple(image for image in project_images(upstream) if not is_digest_locked_image(image))


def load_image_map(path: Path, *, upstream: Path) -> dict[str, str]:
    """Load a lock written by `images lock` and check it against the pinned inventory.

    The lock may cover a subset of the inventory; each task is re-checked at resolution.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"image lock is missing: {path}; run `cybergym-e2b images lock --task <project>/<task>`"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("runtime image lock must use schema version 1")
    if raw.get("upstream") != {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT}:
        raise ValueError("image lock is for different CyberGym-E2E source inputs")
    resolver = raw.get("resolver")
    if (
        not isinstance(resolver, dict)
        or set(resolver) != {"tool", "version", "method"}
        or not all(isinstance(value, str) and value for value in resolver.values())
    ):
        raise ValueError("image lock is missing resolver provenance")
    images = raw.get("images")
    if not isinstance(images, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in images.items()
    ):
        raise ValueError("image lock must contain a string-to-string 'images' object")
    for source, target in images.items():
        require_digest_locked_image(target, label=f"image lock value for {source!r}")
        if (
            source == FFMPEG_IMAGE
            and target.rsplit(":", 1)[1] != FFMPEG_IMAGE_DIGEST.rsplit(":", 1)[1]
        ):
            raise ValueError("image lock value for FFmpeg must preserve the known FFmpeg digest")
    unexpected = sorted(set(images) - set(mutable_project_images(upstream)))
    if unexpected:
        raise ValueError(
            f"image lock has {len(unexpected)} image(s) absent from the pinned inventory; "
            f"first unexpected image: {unexpected[0]}"
        )
    return images


def require_immutable_runtime_image(resolved: ResolvedTask) -> None:
    """Fail before artifact reuse or sandbox creation on a mutable project image."""
    if not is_digest_locked_image(resolved.runtime_image):
        raise ValueError(
            f"runtime image for CyberGym-E2E task {resolved.task} is not digest-locked "
            f"(absent from the image lock): {resolved.runtime_image!r}; "
            f"run `cybergym-e2b images lock --task {resolved.task}`"
        )


def inventory(upstream: Path) -> dict:
    tasks: list[ResolvedTask] = []
    for task_config in sorted((upstream / "projects").glob("*/*/config.toml")):
        task = f"{task_config.parent.parent.name}/{task_config.parent.name}"
        tasks.append(resolve_task(upstream, task))
    image_counts = Counter(item.build_image for item in tasks)
    project_counts = Counter(item.project for item in tasks)
    return {
        "tasks": len(tasks),
        "projects": len(project_counts),
        "unique_build_images": len(image_counts),
        "single_use_build_images": sum(count == 1 for count in image_counts.values()),
        "images": [
            {"build_image": image, "task_count": count}
            for image, count in image_counts.most_common()
        ],
        "project_task_counts": dict(sorted(project_counts.items())),
    }


def build_code_bundle(
    upstream: Path,
    resolved: ResolvedTask,
    *,
    patch_file: Path = DEFAULT_PATCH_FILE,
    remote_smoke: Path = DEFAULT_REMOTE_SMOKE,
    remote_install_codex: Path = DEFAULT_REMOTE_INSTALL_CODEX,
    remote_apt_retry: Path = DEFAULT_REMOTE_APT_RETRY,
) -> bytes:
    """Pack upstream scripts plus one project/task; HF source blobs stay out of this bundle."""
    with tempfile.TemporaryDirectory(prefix="cybergym-e2b-bundle-") as temp_name:
        temp = Path(temp_name)
        shutil.copytree(upstream / "scripts", temp / "scripts")
        project_target = temp / "projects" / resolved.project
        project_target.mkdir(parents=True)
        shutil.copy2(upstream / "projects" / resolved.project / "project.toml", project_target)
        shutil.copytree(
            upstream / "projects" / resolved.project / resolved.task_id,
            project_target / resolved.task_id,
        )
        shutil.copy2(remote_smoke, temp / "scripts" / "e2b_smoke.py")
        shutil.copy2(remote_install_codex, temp / "scripts" / "install_codex.sh")
        shutil.copy2(remote_apt_retry, temp / "scripts" / "apt_retry.sh")
        subprocess.run(
            [
                "git",
                "apply",
                "--whitespace=nowarn",
                str(patch_file.resolve()),
            ],
            cwd=temp,
            check=True,
            capture_output=True,
            text=True,
        )
        if resolved.runtime_image != resolved.build_image:
            config_path = project_target / resolved.task_id / "config.toml"
            task_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            task_config["build_image"] = resolved.runtime_image
            config_path.write_text(tomli_w.dumps(task_config), encoding="utf-8")
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            archive.add(temp / "scripts", arcname="scripts")
            archive.add(temp / "projects", arcname="projects")
        return output.getvalue()
