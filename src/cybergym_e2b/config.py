from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any


def asset_path(relative: str) -> Path:
    """Return an installed package asset as a concrete filesystem path."""
    return Path(str(files("cybergym_e2b.assets").joinpath(relative)))


UPSTREAM_LOCK = asset_path("upstream.lock.json")


@dataclass(frozen=True)
class UpstreamInputs:
    code_repository: str
    code_commit: str
    dataset_repository: str
    dataset_revision: str


def load_upstream_lock(path: Path = UPSTREAM_LOCK) -> UpstreamInputs:
    """Load and strictly validate the sole executable source-input contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "code", "dataset"}:
        raise ValueError("upstream lock must contain schema_version, code, and dataset")
    if raw["schema_version"] != 1:
        raise ValueError("unsupported upstream lock schema")
    code = raw["code"]
    dataset = raw["dataset"]
    if not isinstance(code, dict) or set(code) != {"repository", "commit"}:
        raise ValueError("upstream lock code entry is malformed")
    if not isinstance(dataset, dict) or set(dataset) != {"repository", "revision"}:
        raise ValueError("upstream lock dataset entry is malformed")
    if not isinstance(code["repository"], str) or not code["repository"]:
        raise ValueError("upstream lock code repository is malformed")
    if not isinstance(code["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", code["commit"]):
        raise ValueError("upstream lock code commit must be a 40-character lowercase Git SHA")
    if not isinstance(dataset["repository"], str) or not dataset["repository"]:
        raise ValueError("upstream lock dataset repository is malformed")
    if not isinstance(dataset["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", dataset["revision"]
    ):
        raise ValueError("upstream lock dataset revision must be a 40-character lowercase SHA")
    return UpstreamInputs(
        code_repository=code["repository"],
        code_commit=code["commit"],
        dataset_repository=dataset["repository"],
        dataset_revision=dataset["revision"],
    )


_UPSTREAM_INPUTS = load_upstream_lock()
UPSTREAM_REPOSITORY = _UPSTREAM_INPUTS.code_repository
UPSTREAM_COMMIT = _UPSTREAM_INPUTS.code_commit
DATASET_REPOSITORY = _UPSTREAM_INPUTS.dataset_repository
DATASET_REVISION = _UPSTREAM_INPUTS.dataset_revision

DEFAULT_BUILD_IMAGE = (
    "gcr.io/oss-fuzz-base/base-builder"
    "@sha256:8eda74a11e800aead5a041ee479a65b33dab3150d6e89e5694e2b6eb27be98fc"
)
BASE_TEMPLATE_IMAGE = (
    "ubuntu@sha256:79676deb51ebb02885b0b9d33788e78a37cf1045ad79d1bb04c6a222c3556b3d"
)
BASE_BUILDER_IMAGES = (
    "gcr.io/oss-fuzz-base/base-builder"
    "@sha256:fba1033c6a64433642ab97b6ea987ddaa9938e06596c6cace1c786130fc1461b",
    DEFAULT_BUILD_IMAGE,
)
FFMPEG_IMAGE = "cybergym/e2e:ffmpeg"
FFMPEG_IMAGE_DIGEST = (
    "cybergym/e2e@sha256:1f29cf62de96e8ff6411d3f903372c0dd5dc93e93ab166f9863483eaa43ffd25"
)
OPUS_MODEL_SHA256 = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e"
OPUS_MODEL_FILENAME = f"opus_data-{OPUS_MODEL_SHA256}.tar.gz"
OPUS_MODEL_CACHE = f"/opt/cybergym-e2e-cache/{OPUS_MODEL_FILENAME}"


DEFAULT_UPSTREAM = Path("vendor/cybergym-e2e")
DEFAULT_MANIFEST = Path("artifacts/templates/manifest.json")
DEFAULT_BUILD_LEDGER = Path("artifacts/templates/build-ledger.json")
DEFAULT_IMAGE_LOCK = Path("artifacts/images.lock.json")
DEFAULT_NETWORK_POLICY = asset_path("policies/network.json")
DEFAULT_PATCH_FILE = asset_path("patches/openai-compatible.patch")
DEFAULT_REMOTE_SMOKE = asset_path("remote/e2b_smoke.py")
DEFAULT_REMOTE_INSTALL_CODEX = asset_path("remote/install_codex.sh")
DEFAULT_REMOTE_APT_RETRY = asset_path("remote/apt_retry.sh")
DEFAULT_TEMPLATE_REQUIREMENTS = asset_path("template-requirements.lock")
# Project-local .env by default; override with CYBERGYM_KEYS_FILE for other setups.
DEFAULT_KEYS_FILE = Path(os.environ.get("CYBERGYM_KEYS_FILE", ".env"))
DEFAULT_TEMPLATE_NAME = "cybergym-e2e-dind"
DEFAULT_FFMPEG_TEMPLATE_NAME = "cybergym-e2e-ffmpeg"
DEFAULT_MODEL = "openai.gpt-5.4"
DEFAULT_MODEL_PROVIDER = "bedrock"
_TASK_PATH = re.compile(r"^[A-Za-z0-9_.+-]+/[A-Za-z0-9_.+-]+$")
TEMPLATE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_E2B_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,}$")
_RECIPE_TAG = re.compile(r"^recipe-([0-9a-f]{16})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def is_digest_locked_image(image: str) -> bool:
    """Return whether an OCI image reference carries an exact SHA-256 digest."""
    return bool(_DIGEST_IMAGE.fullmatch(image))


def require_digest_locked_image(image: str, *, label: str = "runtime image") -> str:
    if not is_digest_locked_image(image):
        raise ValueError(
            f"{label} must be digest-locked as repository@sha256:<64 hex chars>: {image!r}"
        )
    return image


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    """Load a simple dotenv file without ever printing its values."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def normalize_task(task: str) -> str:
    task = task.strip()
    if not _TASK_PATH.fullmatch(task) or ".." in task:
        raise ValueError(f"invalid CyberGym-E2E task path: {task!r}")
    return task


@dataclass(frozen=True)
class TemplateRef:
    name: str
    tag: str
    template_id: str
    build_id: str
    recipe_sha256: str
    images: tuple[str, ...] = ()

    @property
    def reference(self) -> str:
        if (
            not TEMPLATE_NAME.fullmatch(self.name)
            or not _E2B_ID.fullmatch(self.template_id)
            or not _E2B_ID.fullmatch(self.build_id)
            or not _SHA256.fullmatch(self.recipe_sha256)
        ):
            raise ValueError("template does not have a valid tagged build receipt")
        tag_match = _RECIPE_TAG.fullmatch(self.tag)
        if not tag_match or tag_match.group(1) != self.recipe_sha256[:16]:
            raise ValueError("template tag does not match its build recipe")
        for image in self.images:
            require_digest_locked_image(image, label="recorded template image")
        return f"{self.name}:{self.tag}"


@dataclass
class TemplateManifest:
    base: TemplateRef
    hot: dict[str, TemplateRef] = field(default_factory=dict)
    cpu_count: int = 8
    memory_mb: int = 8192
    # Expected account/tier root-disk allocation, not a per-template guarantee: the E2B
    # build API (TemplateBuildRequestV3) only accepts cpu_count and memory_mb, so disk is
    # controlled account-side. 120 GB is the current requested-tier working max; the real
    # safety net is the runtime free-disk preflight, not this recorded value.
    disk_limit_gb: int = 120

    def route(self, build_image: str) -> TemplateRef:
        return self.hot.get(build_image, self.base)

    def write(self, path: Path = DEFAULT_MANIFEST) -> None:
        payload = {
            "schema_version": 4,
            "resources": {
                "cpu_count": self.cpu_count,
                "memory_mb": self.memory_mb,
                "disk_limit_gb": self.disk_limit_gb,
            },
            "upstream": {
                "code_commit": UPSTREAM_COMMIT,
                "dataset_revision": DATASET_REVISION,
            },
            "base": asdict(self.base),
            "hot": {image: asdict(ref) for image, ref in sorted(self.hot.items())},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST) -> TemplateManifest:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 4:
            raise ValueError("unsupported template manifest schema")
        if raw.get("upstream") != {
            "code_commit": UPSTREAM_COMMIT,
            "dataset_revision": DATASET_REVISION,
        }:
            raise ValueError("template manifest is for different CyberGym-E2E inputs")
        resources = raw["resources"]
        base_raw = raw["base"]
        base = TemplateRef(
            name=base_raw["name"],
            tag=base_raw["tag"],
            template_id=base_raw["template_id"],
            build_id=base_raw["build_id"],
            recipe_sha256=base_raw["recipe_sha256"],
            images=tuple(base_raw.get("images", ())),
        )
        hot = {
            image: TemplateRef(
                name=value["name"],
                tag=value["tag"],
                template_id=value["template_id"],
                build_id=value["build_id"],
                recipe_sha256=value["recipe_sha256"],
                images=tuple(value.get("images", ())),
            )
            for image, value in raw.get("hot", {}).items()
        }
        manifest = cls(
            base=base,
            hot=hot,
            cpu_count=resources["cpu_count"],
            memory_mb=resources["memory_mb"],
            disk_limit_gb=resources["disk_limit_gb"],
        )
        _ = manifest.base.reference
        for ref in manifest.hot.values():
            _ = ref.reference
        return manifest
