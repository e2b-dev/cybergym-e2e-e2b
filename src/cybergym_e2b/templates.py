from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

from e2b import Template, default_build_logger

from cybergym_e2b.config import (
    BASE_BUILDER_IMAGES,
    BASE_TEMPLATE_IMAGE,
    DATASET_REVISION,
    DEFAULT_BUILD_LEDGER,
    DEFAULT_FFMPEG_TEMPLATE_NAME,
    DEFAULT_MANIFEST,
    DEFAULT_TEMPLATE_NAME,
    DEFAULT_TEMPLATE_REQUIREMENTS,
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    OPUS_MODEL_CACHE,
    OPUS_MODEL_FILENAME,
    OPUS_MODEL_SHA256,
    UPSTREAM_COMMIT,
    TemplateManifest,
    TemplateRef,
    require_digest_locked_image,
)

TEMPLATE_RECIPE_SCHEMA = "cybergym-e2e-template-v1"
UBUNTU_SNAPSHOT = "https://snapshot.ubuntu.com/ubuntu/20260831T000000Z"
DOCKER_SIGNING_MATERIAL_SHA256 = "1500c1f56fa9e26b9b8f42452a553675796ade0807cdce11975eb98170b3a570"
DOCKER_PACKAGES = (
    "docker-ce=5:27.5.1-1~ubuntu.22.04~jammy",
    "docker-ce-cli=5:27.5.1-1~ubuntu.22.04~jammy",
    "containerd.io=1.7.25-1",
    "docker-buildx-plugin=0.20.0-1~ubuntu.22.04~jammy",
    "docker-compose-plugin=2.32.4-1~ubuntu.22.04~jammy",
)
SYSTEM_PACKAGES = (
    "ca-certificates",
    "curl",
    "file",
    "git",
    "gnupg",
    "jq",
    "lsb-release",
    "patch",
    "python3-venv",
    "ripgrep",
    "rsync",
    "tar",
    "xz-utils",
)
_LOCKED_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s\\]+")
_TEMPLATE_PREFIX = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_BASE_FINALIZE_COMMAND = "docker system df && rm -rf /root/.cache /tmp/* && sync"
_READY_COMMAND = "docker info >/dev/null"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_requirements_lock(path: Path) -> bytes:
    """Require every requirement group to use an exact version and at least one hash."""
    payload = path.read_bytes()
    groups: list[list[str]] = []
    current: list[str] = []
    for raw_line in payload.decode("utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line[:1].isspace():
            if current:
                groups.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)
        else:
            raise ValueError(f"requirements lock begins with an option: {path}")
    if current:
        groups.append(current)
    if not groups:
        raise ValueError(f"requirements lock is empty: {path}")
    for group in groups:
        statement = " ".join(group).replace("\\", " ")
        if not _LOCKED_REQUIREMENT.match(group[0]) or "--hash=sha256:" not in statement:
            raise ValueError(
                "template requirements must be fully locked with == versions and SHA-256 hashes"
            )
    return payload


def _recipe(kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    normalized_inputs = json.loads(json.dumps(inputs, sort_keys=True))
    payload = {
        "identity_schema": TEMPLATE_RECIPE_SCHEMA,
        "kind": kind,
        "inputs": normalized_inputs,
    }
    return {**payload, "sha256": _canonical_sha256(payload)}


def _tagged_name(prefix: str, recipe_sha256: str) -> str:
    if not _TEMPLATE_PREFIX.fullmatch(prefix):
        raise ValueError("template name prefix must contain lowercase letters, digits, and hyphens")
    return f"{prefix}:recipe-{recipe_sha256[:16]}"


@cache
def verify_template_ref(ref: TemplateRef) -> dict[str, str]:
    """Verify that a reusable recipe tag still resolves to its recorded build."""
    tags = Template.get_tags(ref.template_id)
    for assigned in tags:
        if assigned.tag != ref.tag:
            continue
        if assigned.build_id != ref.build_id:
            raise RuntimeError(
                f"template tag {ref.reference} no longer points to recorded build "
                f"{ref.build_id}; found {assigned.build_id}"
            )
        return {
            "reference": ref.reference,
            "template_id": ref.template_id,
            "build_id": ref.build_id,
        }
    raise RuntimeError(f"template tag {ref.reference} is not assigned to {ref.template_id}")


def _base_commands(requirements_payload: bytes) -> tuple[str, str, str]:
    encoded_requirements = base64.b64encode(requirements_payload).decode("ascii")
    docker_packages = " ".join(shlex.quote(package) for package in DOCKER_PACKAGES)
    system_packages = " ".join(shlex.quote(package) for package in SYSTEM_PACKAGES)
    install = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
rm -f /etc/apt/sources.list.d/ubuntu.sources
printf '%s\n' \
  'deb [check-valid-until=no] {UBUNTU_SNAPSHOT} jammy main restricted universe multiverse' \
  'deb [check-valid-until=no] {UBUNTU_SNAPSHOT} jammy-updates main restricted universe multiverse' \
  'deb [check-valid-until=no] {UBUNTU_SNAPSHOT} jammy-security main restricted universe multiverse' \
  > /etc/apt/sources.list
apt-get -o Acquire::Check-Valid-Until=false update
apt-get install -y --no-install-recommends {system_packages}
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker.asc
echo '{DOCKER_SIGNING_MATERIAL_SHA256}  /tmp/docker.asc' | sha256sum -c -
gpg --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker.asc
rm /tmp/docker.asc
chmod a+r /etc/apt/keyrings/docker.gpg
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable' > /etc/apt/sources.list.d/docker.list
apt-get -o Acquire::Check-Valid-Until=false update
apt-get install -y --no-install-recommends {docker_packages}
systemctl enable docker.service containerd.service
apt-get clean
rm -rf /var/lib/apt/lists/*
""".strip()
    daemon = f"""
set -euo pipefail
install -d -m 0755 /etc/docker /opt/cybergym-e2e /work/cybergym-e2e
printf '%s\n' '{{"storage-driver":"overlay2","log-driver":"local","log-opts":{{"max-size":"20m","max-file":"3"}},"max-concurrent-downloads":8}}' > /etc/docker/daemon.json
printf '%s\n' 'vm.mmap_rnd_bits=28' > /etc/sysctl.d/99-cybergym-sanitizers.conf
python3 -m venv /opt/cybergym-e2e-venv
printf '%s' {shlex.quote(encoded_requirements)} | base64 -d > /tmp/template-requirements.lock
/opt/cybergym-e2e-venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/template-requirements.lock
rm /tmp/template-requirements.lock
""".strip()
    ensure_docker = """
set -euo pipefail
if ! docker info >/dev/null 2>&1; then
  nohup dockerd >/var/log/cybergym-dockerd-build.log 2>&1 &
  for _ in $(seq 1 60); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
fi
docker info --format '{{.Driver}}' | grep -Fx overlay2
""".strip()
    return install, daemon, ensure_docker


def _pull_command(image: str) -> str:
    quoted = shlex.quote(image)
    return f"docker pull {quoted} && docker image inspect {quoted} >/dev/null"


def _base_recipe(
    requirements: Path,
    *,
    name: str,
    cpu_count: int,
    memory_mb: int,
    disk_limit_gb: int,
) -> tuple[dict[str, Any], bytes]:
    if min(cpu_count, memory_mb, disk_limit_gb) <= 0:
        raise ValueError("template resource values must be positive")
    requirements_payload = _validate_requirements_lock(requirements)
    if not _TEMPLATE_PREFIX.fullmatch(name):
        raise ValueError("template name prefix must contain lowercase letters, digits, and hyphens")
    build_commands = _base_commands(requirements_payload)
    require_digest_locked_image(BASE_TEMPLATE_IMAGE, label="base template image")
    for image in BASE_BUILDER_IMAGES:
        require_digest_locked_image(image, label="preloaded builder image")
    recipe = _recipe(
        "base",
        {
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "name_prefix": name,
            "base_image": BASE_TEMPLATE_IMAGE,
            "ubuntu_snapshot": UBUNTU_SNAPSHOT,
            "system_packages": SYSTEM_PACKAGES,
            "docker_signing_material_sha256": DOCKER_SIGNING_MATERIAL_SHA256,
            "docker_packages": DOCKER_PACKAGES,
            "python_requirements_sha256": _sha256_bytes(requirements_payload),
            "template_steps": {
                "user": "root",
                "workdir": "/root",
                "environment": {"DEBIAN_FRONTEND": "noninteractive"},
                "command_sha256": [
                    *[_sha256_bytes(value.encode()) for value in build_commands],
                    *[
                        _sha256_bytes(_pull_command(image).encode())
                        for image in BASE_BUILDER_IMAGES
                    ],
                    _sha256_bytes(_BASE_FINALIZE_COMMAND.encode()),
                ],
                "ready_command_sha256": _sha256_bytes(_READY_COMMAND.encode()),
            },
            "preloaded_images": BASE_BUILDER_IMAGES,
            "resources": {
                "cpu_count": cpu_count,
                "memory_mb": memory_mb,
                "disk_limit_gb": disk_limit_gb,
            },
        },
    )
    return recipe, requirements_payload


def _ffmpeg_commands() -> tuple[str, str, str]:
    model_source = f"rsync://media.xiph.org/media/opus/models/{OPUS_MODEL_FILENAME}"
    return (
        f"docker pull {shlex.quote(FFMPEG_IMAGE_DIGEST)}",
        (
            f"install -d -m 0755 {shlex.quote(str(Path(OPUS_MODEL_CACHE).parent))} && "
            f"rsync --timeout=120 --contimeout=30 {shlex.quote(model_source)} "
            f"{shlex.quote(OPUS_MODEL_CACHE)} && "
            f"echo {shlex.quote(f'{OPUS_MODEL_SHA256}  {OPUS_MODEL_CACHE}')} | sha256sum -c -"
        ),
        f"docker image inspect {shlex.quote(FFMPEG_IMAGE_DIGEST)} >/dev/null && sync",
    )


def _ffmpeg_recipe(manifest: TemplateManifest, *, name: str) -> dict[str, Any]:
    require_digest_locked_image(FFMPEG_IMAGE_DIGEST, label="FFmpeg image")
    if not _TEMPLATE_PREFIX.fullmatch(name):
        raise ValueError("template name prefix must contain lowercase letters, digits, and hyphens")
    build_commands = _ffmpeg_commands()
    return _recipe(
        "ffmpeg",
        {
            "base_template": manifest.base.reference,
            "base_recipe_sha256": manifest.base.recipe_sha256,
            "name_prefix": name,
            "ffmpeg_image": FFMPEG_IMAGE_DIGEST,
            "opus_model_filename": OPUS_MODEL_FILENAME,
            "opus_model_sha256": OPUS_MODEL_SHA256,
            "template_steps": {
                "user": "root",
                "workdir": "/root",
                "command_sha256": [_sha256_bytes(value.encode()) for value in build_commands],
                "ready_command_sha256": _sha256_bytes(_READY_COMMAND.encode()),
            },
            "resources": {
                "cpu_count": manifest.cpu_count,
                "memory_mb": manifest.memory_mb,
                "disk_limit_gb": manifest.disk_limit_gb,
            },
        },
    )


class _BuildLedger:
    """Local identity gate for reusing immutable E2B build references."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            self.payload: dict[str, Any] = {"schema_version": 2, "records": {}}
            return
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        if self.payload.get("schema_version") != 2 or not isinstance(
            self.payload.get("records"), dict
        ):
            raise ValueError(f"unsupported template build ledger: {path}")

    def get(self, recipe: dict[str, Any]) -> TemplateRef | None:
        record = self.payload["records"].get(recipe["sha256"])
        if not isinstance(record, dict) or record.get("recipe") != recipe:
            return None
        raw = record.get("template")
        if not isinstance(raw, dict):
            return None
        ref = TemplateRef(
            name=raw["name"],
            tag=raw["tag"],
            template_id=raw["template_id"],
            build_id=raw["build_id"],
            recipe_sha256=raw["recipe_sha256"],
            images=tuple(raw.get("images", ())),
        )
        if ref.recipe_sha256 != recipe["sha256"]:
            return None
        _ = ref.reference
        return ref

    def put(self, recipe: dict[str, Any], ref: TemplateRef) -> None:
        if ref.recipe_sha256 != recipe["sha256"]:
            raise ValueError("template reference does not match build recipe")
        self.payload["records"][recipe["sha256"]] = {
            "recipe": recipe,
            "template": asdict(ref),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def _docker_builder(requirements: Path):
    requirements_payload = _validate_requirements_lock(requirements)
    install, daemon, ensure_docker = _base_commands(requirements_payload)
    return (
        Template()
        .from_image(BASE_TEMPLATE_IMAGE)
        .set_user("root")
        .set_workdir("/root")
        .set_envs({"DEBIAN_FRONTEND": "noninteractive"})
        .run_cmd(install)
        .run_cmd(daemon)
        .run_cmd(ensure_docker)
    )


def _pull(builder, images: tuple[str, ...]):
    for image in images:
        require_digest_locked_image(image, label="preloaded image")
        builder = builder.run_cmd(_pull_command(image))
    return builder


def build_base_template(
    *,
    requirements: Path = DEFAULT_TEMPLATE_REQUIREMENTS,
    name: str = DEFAULT_TEMPLATE_NAME,
    manifest_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_BUILD_LEDGER,
    cpu_count: int = 8,
    memory_mb: int = 8192,
    disk_limit_gb: int = 120,
) -> TemplateManifest:
    recipe, _ = _base_recipe(
        requirements,
        name=name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        disk_limit_gb=disk_limit_gb,
    )
    ledger = _BuildLedger(ledger_path)
    reused = ledger.get(recipe)
    if reused is not None:
        manifest = TemplateManifest(
            base=reused,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            disk_limit_gb=disk_limit_gb,
        )
        manifest.write(manifest_path)
        return manifest

    template_name = _tagged_name(name, recipe["sha256"])
    builder = _pull(_docker_builder(requirements), BASE_BUILDER_IMAGES)
    builder = builder.run_cmd(_BASE_FINALIZE_COMMAND).set_ready_cmd(_READY_COMMAND)
    build = Template.build(
        builder,
        template_name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        on_build_logs=default_build_logger(),
        request_timeout=900,
    )
    ref = TemplateRef(
        name=name,
        tag=f"recipe-{recipe['sha256'][:16]}",
        template_id=build.template_id,
        build_id=build.build_id,
        recipe_sha256=recipe["sha256"],
        images=(BASE_TEMPLATE_IMAGE, *BASE_BUILDER_IMAGES),
    )
    ledger.put(recipe, ref)
    manifest = TemplateManifest(
        base=ref,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        disk_limit_gb=disk_limit_gb,
    )
    manifest.write(manifest_path)
    return manifest


def build_ffmpeg_template(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_BUILD_LEDGER,
    name: str = DEFAULT_FFMPEG_TEMPLATE_NAME,
) -> TemplateManifest:
    manifest = TemplateManifest.load(manifest_path)
    recipe = _ffmpeg_recipe(manifest, name=name)
    ledger = _BuildLedger(ledger_path)
    reused = ledger.get(recipe)
    if reused is not None:
        manifest.hot[FFMPEG_IMAGE] = reused
        manifest.write(manifest_path)
        return manifest

    template_name = _tagged_name(name, recipe["sha256"])
    pull_ffmpeg, install_model, inspect_ffmpeg = _ffmpeg_commands()
    builder = (
        Template()
        .from_template(manifest.base.reference)
        .set_user("root")
        .set_workdir("/root")
        .run_cmd(pull_ffmpeg)
        .run_cmd(install_model)
        .run_cmd(inspect_ffmpeg)
        .set_ready_cmd(_READY_COMMAND)
    )
    build = Template.build(
        builder,
        template_name,
        cpu_count=manifest.cpu_count,
        memory_mb=manifest.memory_mb,
        on_build_logs=default_build_logger(),
        request_timeout=900,
    )
    ref = TemplateRef(
        name=name,
        tag=f"recipe-{recipe['sha256'][:16]}",
        template_id=build.template_id,
        build_id=build.build_id,
        recipe_sha256=recipe["sha256"],
        images=(*manifest.base.images, FFMPEG_IMAGE_DIGEST),
    )
    ledger.put(recipe, ref)
    manifest.hot[FFMPEG_IMAGE] = ref
    manifest.write(manifest_path)
    return manifest
