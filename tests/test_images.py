from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cybergym_e2b.config import (
    FFMPEG_IMAGE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from cybergym_e2b.images import DockerBuildxResolver, create_image_lock, write_image_lock
from cybergym_e2b.inventory import load_image_map, require_immutable_runtime_image, resolve_task


def _write_project(
    upstream: Path,
    project: str,
    task: str,
    image: str,
    *,
    task_image: str | None = None,
) -> None:
    project_dir = upstream / "projects" / project
    task_dir = project_dir / task
    task_dir.mkdir(parents=True)
    (project_dir / "project.toml").write_text(
        f'build_image = "{image}"\nrepo_to_patch = "https://example.invalid/{project}"\n',
        encoding="utf-8",
    )
    task_config = f'build_image = "{task_image}"\n' if task_image else ""
    (task_dir / "config.toml").write_text(task_config, encoding="utf-8")


class _Resolver:
    provenance = {
        "tool": "test-registry-inspector",
        "version": "1.2.3",
        "method": "manifest-descriptor-digest",
    }

    def __init__(self, resolved: dict[str, str]) -> None:
        self.resolved = resolved
        self.calls: list[str] = []

    def __call__(self, image: str) -> str:
        self.calls.append(image)
        return self.resolved[image]


def test_image_lock_is_deterministic_complete_and_records_provenance(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    _write_project(upstream, "alpha", "task-a", "registry.example/alpha:v1")
    _write_project(
        upstream,
        "beta",
        "task-b",
        "registry.example/beta:v2",
        task_image="registry.example/override:v3",
    )
    _write_project(upstream, "beta", "task-c", "registry.example/beta:v2")
    _write_project(
        upstream,
        "locked",
        "task-d",
        "registry.example/locked@sha256:" + "d" * 64,
    )
    resolver = _Resolver(
        {
            "registry.example/alpha:v1": "registry.example/alpha@sha256:" + "a" * 64,
            "registry.example/beta:v2": "registry.example/beta@sha256:" + "b" * 64,
            "registry.example/override:v3": "registry.example/override@sha256:" + "c" * 64,
        }
    )

    first = create_image_lock(upstream, resolver=resolver)
    second = create_image_lock(upstream, resolver=_Resolver(resolver.resolved))

    assert first == second
    assert first == {
        "schema_version": 1,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "resolver": _Resolver.provenance,
        "images": {
            "registry.example/alpha:v1": "registry.example/alpha@sha256:" + "a" * 64,
            "registry.example/beta:v2": "registry.example/beta@sha256:" + "b" * 64,
            "registry.example/override:v3": "registry.example/override@sha256:" + "c" * 64,
        },
    }
    assert resolver.calls == [
        "registry.example/alpha:v1",
        "registry.example/beta:v2",
        "registry.example/override:v3",
    ]

    lock_path = tmp_path / "images.lock.json"
    lock_path.write_text(json.dumps(first), encoding="utf-8")
    assert load_image_map(lock_path, upstream=upstream) == first["images"]

    # A partial lock is usable for the tasks it covers and refused for the rest.
    del first["images"]["registry.example/beta:v2"]
    lock_path.write_text(json.dumps(first), encoding="utf-8")
    partial = load_image_map(lock_path, upstream=upstream)
    covered = resolve_task(upstream, "alpha/task-a", image_map=partial)
    require_immutable_runtime_image(covered)
    assert covered.runtime_image == "registry.example/alpha@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="images lock --task beta/task-c"):
        require_immutable_runtime_image(resolve_task(upstream, "beta/task-c", image_map=partial))

    first["images"]["registry.example/stranger:v9"] = "registry.example/stranger@sha256:" + "e" * 64
    lock_path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the pinned inventory"):
        load_image_map(lock_path, upstream=upstream)


def test_image_lock_can_be_scoped_to_tasks_and_merged(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    _write_project(upstream, "alpha", "task-a", "registry.example/alpha:v1")
    _write_project(upstream, "beta", "task-c", "registry.example/beta:v2")
    resolved = {
        "registry.example/alpha:v1": "registry.example/alpha@sha256:" + "a" * 64,
        "registry.example/beta:v2": "registry.example/beta@sha256:" + "b" * 64,
    }

    resolver = _Resolver(resolved)
    scoped = create_image_lock(upstream, resolver=resolver, tasks=["alpha/task-a"])
    assert set(scoped["images"]) == {"registry.example/alpha:v1"}
    assert resolver.calls == ["registry.example/alpha:v1"]

    # Repeated scoped runs accumulate into one lock, so a rate-limited registry can
    # be walked in slices without ever re-resolving what is already pinned.
    output = tmp_path / "images.lock.json"
    write_image_lock(upstream, output, resolver=_Resolver(resolved), tasks=["alpha/task-a"])
    second = _Resolver(resolved)
    merged = write_image_lock(upstream, output, resolver=second, tasks=["beta/task-c"])
    assert second.calls == ["registry.example/beta:v2"]
    assert merged["images"] == resolved
    assert json.loads(output.read_text())["images"] == resolved


def test_docker_buildx_resolver_retries_rate_limited_inspections() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        if command[-1] == "version":
            return subprocess.CompletedProcess(command, 0, "buildx 1.0\n", "")
        attempts.append(command[4])
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="ERROR: unexpected status from HEAD request: 429 Too Many Requests",
            )
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"digest": "sha256:" + "a" * 64}), ""
        )

    resolver = DockerBuildxResolver(runner=runner, sleep=sleeps.append)
    assert resolver("registry.example/app:v1") == "registry.example/app@sha256:" + "a" * 64
    assert len(attempts) == 3
    assert len(sleeps) == 2


def test_docker_buildx_resolver_surfaces_registry_error_text() -> None:
    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        if command[-1] == "version":
            return subprocess.CompletedProcess(command, 0, "buildx 1.0\n", "")
        raise subprocess.CalledProcessError(1, command, stderr="ERROR: manifest unknown")

    resolver = DockerBuildxResolver(runner=runner, sleep=lambda _seconds: None)
    with pytest.raises(RuntimeError, match="manifest unknown"):
        resolver("registry.example/app:v1")


def test_docker_buildx_resolver_uses_manifest_descriptor_without_pulling_layers() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["docker", "buildx", "version"]:
            return subprocess.CompletedProcess(
                command, 0, "github.com/docker/buildx v0.32.1 abc\n", ""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "digest": "sha256:" + "a" * 64,
                }
            ),
            "",
        )

    resolver = DockerBuildxResolver(runner=runner)

    assert resolver("registry.example/team/app:v1") == (
        "registry.example/team/app@sha256:" + "a" * 64
    )
    assert resolver.provenance == {
        "tool": "docker buildx imagetools inspect",
        "version": "github.com/docker/buildx v0.32.1 abc",
        "method": "manifest-descriptor-digest",
    }
    assert calls[-1] == [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        "registry.example/team/app:v1",
        "--format",
        "{{json .Manifest}}",
    ]


@pytest.mark.parametrize(
    "descriptor",
    [
        {},
        {"digest": "sha256:" + "a" * 64 + "\nsha256:" + "b" * 64},
        {"digest": "sha512:" + "a" * 64},
    ],
)
def test_docker_buildx_resolver_rejects_incomplete_or_ambiguous_results(
    descriptor: dict[str, str],
) -> None:
    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        stdout = "buildx 1.0\n" if command[-1] == "version" else json.dumps(descriptor)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    resolver = DockerBuildxResolver(runner=runner)

    with pytest.raises(ValueError, match="unambiguous SHA-256 manifest digest"):
        resolver("registry.example/team/app:v1")


def test_image_lock_rejects_a_changed_known_ffmpeg_tag(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    _write_project(upstream, "ffmpeg", "task", FFMPEG_IMAGE)

    with pytest.raises(ValueError, match="known FFmpeg digest"):
        create_image_lock(
            upstream,
            resolver=_Resolver({FFMPEG_IMAGE: "cybergym/e2e@sha256:" + "f" * 64}),
        )
