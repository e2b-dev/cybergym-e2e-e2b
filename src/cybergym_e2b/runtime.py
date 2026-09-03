from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shlex
import socket
import tarfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import httpx
from e2b import Sandbox

from cybergym_e2b.config import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_NETWORK_POLICY,
    DEFAULT_PATCH_FILE,
    DEFAULT_REMOTE_INSTALL_CODEX,
    DEFAULT_REMOTE_SMOKE,
    FFMPEG_IMAGE,
    OPUS_MODEL_CACHE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    TemplateManifest,
    TemplateRef,
    require_digest_locked_image,
)
from cybergym_e2b.inventory import (
    ResolvedTask,
    build_code_bundle,
    require_immutable_runtime_image,
    resolved_asdict,
)
from cybergym_e2b.templates import verify_template_ref

EgressMode = Literal["policy", "restricted", "permissive"]
ModelProvider = Literal["fireworks", "bedrock"]


@dataclass(frozen=True)
class RunOptions:
    setup_timeout: int = 7200
    evaluation_timeout: int = 28800
    min_free_gb: int = 80
    ffmpeg_min_free_gb: int = 90
    swap_gb: int = 4
    egress: EgressMode = "policy"
    retain: bool = False
    agent: str = "codex"
    prompt_style: str = "iterative"
    model: str = DEFAULT_MODEL
    max_attempts: int = 1
    agent_timeout: int = 5400
    provider: ModelProvider = DEFAULT_MODEL_PROVIDER
    bedrock_region: str = "us-west-2"


@dataclass(frozen=True)
class ExecutionContext:
    manifest: TemplateManifest
    template: TemplateRef
    policy: dict[str, Any]
    non_http: list[dict]
    model_config: dict[str, str | None]


class _StageProfiler:
    """Record wall-clock stage boundaries without hiding failed stages."""

    def __init__(self) -> None:
        self.started_at_unix = time.time()
        self.started_monotonic = time.monotonic()
        self.stages: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        started_at_unix = time.time()
        started_monotonic = time.monotonic()
        entry: dict[str, Any] = {
            "name": name,
            "started_at_unix": started_at_unix,
            "started_offset_seconds": round(started_monotonic - self.started_monotonic, 6),
            "status": "running",
        }
        self.stages.append(entry)
        try:
            yield entry
        except Exception as exc:
            entry["status"] = "error"
            entry["error_type"] = type(exc).__name__
            raise
        else:
            entry["status"] = "ok"
        finally:
            ended_at_unix = time.time()
            entry["ended_at_unix"] = ended_at_unix
            entry["duration_seconds"] = round(time.monotonic() - started_monotonic, 6)

    def result(self) -> dict[str, Any]:
        return {
            "clock": "host_monotonic_with_unix_boundaries",
            "started_at_unix": self.started_at_unix,
            "duration_seconds": round(time.monotonic() - self.started_monotonic, 6),
            "stages": self.stages,
        }


def _secret(name: str, fallback: str | None = None) -> str | None:
    return os.environ.get(name) or (os.environ.get(fallback) if fallback else None)


def _policy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "default_action",
        "deny_out",
        "model_hosts",
        "artifact_hosts",
        "registry_hosts",
        "runtime_dependency_hosts",
        "non_http_hosts",
    }
    list_fields = required - {"default_action"}
    if (
        set(raw) != required
        or raw["default_action"] not in {"allow", "deny"}
        or not all(isinstance(raw[key], list) for key in list_fields)
    ):
        raise ValueError(f"invalid network policy: {path}")
    for key in list_fields - {"non_http_hosts"}:
        if not all(isinstance(value, str) for value in raw[key]):
            raise ValueError(f"invalid network policy {key}: {path}")
    for entry in raw["deny_out"]:
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError as exc:
            raise ValueError(f"deny_out entries must be IP addresses or CIDRs: {entry!r}") from exc
        e2b_platform_range = ipaddress.ip_network("240.0.0.0/4")
        if (
            network.version != 4
            or (network.network_address.is_unspecified and entry != "0.0.0.0/0")
            or (entry != "0.0.0.0/0" and network.overlaps(e2b_platform_range))
        ):
            raise ValueError(f"deny_out entry is not supported by E2B: {entry!r}")
    for entry in raw["non_http_hosts"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"host", "port", "projects"}
            or not isinstance(entry["host"], str)
            or not isinstance(entry["port"], int)
            or not isinstance(entry["projects"], list)
            or not all(isinstance(project, str) for project in entry["projects"])
        ):
            raise ValueError(f"invalid non-HTTP network policy entry: {entry!r}")
    return raw


def _resolve_non_http(policy: dict[str, Any], project: str) -> list[dict]:
    resolved: list[dict] = []
    for entry in policy["non_http_hosts"]:
        if project not in entry["projects"]:
            continue
        addresses = sorted(
            {
                item[4][0]
                for item in socket.getaddrinfo(
                    entry["host"], entry["port"], family=socket.AF_INET, type=socket.SOCK_STREAM
                )
            }
        )
        if not addresses:
            raise RuntimeError(f"could not resolve non-HTTP egress host {entry['host']}")
        resolved.append({**entry, "addresses": addresses})
    return resolved


def _model_config(options: RunOptions) -> dict[str, str | None]:
    if options.provider == "fireworks":
        return {
            "host": "api.fireworks.ai",
            "base_url": "https://api.fireworks.ai/inference/v1",
            "key": _secret("FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"),
            "key_name": "FIREWORKS_AI_API_KEY or FIREWORKS_API_KEY",
        }
    if options.provider == "bedrock":
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", options.bedrock_region):
            raise ValueError(f"invalid AWS Bedrock region: {options.bedrock_region!r}")
        host = f"bedrock-mantle.{options.bedrock_region}.api.aws"
        # Bedrock exposes OpenAI frontier models through its OpenAI-specific
        # Responses route. Other Mantle models (including DeepSeek V3.2) use the
        # general OpenAI-compatible route consumed by OpenHands/Gemini via Chat
        # Completions.
        api_prefix = "openai/v1" if options.agent == "codex" else "v1"
        return {
            "host": host,
            "base_url": f"https://{host}/{api_prefix}",
            "key": _secret("AWS_MANTLE", "AWS_BEARER_TOKEN_BEDROCK"),
            "key_name": "AWS_MANTLE or AWS_BEARER_TOKEN_BEDROCK",
        }
    raise ValueError(f"unsupported model provider: {options.provider!r}")


def _agent_model_id(options: RunOptions) -> str:
    # OpenHands delegates model dispatch to LiteLLM. Its `openai/` prefix selects
    # the OpenAI-compatible transport and is removed before the request reaches
    # the provider. Without it, a Bedrock catalog ID such as `deepseek.v3.2` is
    # mistaken for LiteLLM's SigV4-native Bedrock transport.
    if options.agent == "openhands" and not options.model.startswith("openai/"):
        return f"openai/{options.model}"
    return options.model


def _transforms(
    *, hf_token: str | None, model_key: str | None, model_host: str = "api.fireworks.ai"
) -> dict[str, list[dict]]:
    rules: dict[str, list[dict]] = {}
    if hf_token:
        rules["huggingface.co"] = [
            {"transform": {"headers": {"Authorization": f"Bearer {hf_token}"}}}
        ]
    if model_key:
        rules[model_host] = [{"transform": {"headers": {"Authorization": f"Bearer {model_key}"}}}]
    return rules


def _network(
    policy: dict[str, Any],
    *,
    phase: Literal["setup", "runtime"],
    egress: EgressMode,
    hf_token: str | None,
    model_key: str | None,
    non_http: list[dict],
    model_host: str = "api.fireworks.ai",
) -> dict[str, Any]:
    # Dataset credentials exist only while preparing the task. The model credential
    # exists only while running the workload. Keeping these phases mutually exclusive
    # prevents an open diagnostic run from inheriting gated-dataset access.
    rules = _transforms(
        hf_token=hf_token if phase == "setup" else None,
        model_key=model_key if phase == "runtime" else None,
        model_host=model_host,
    )
    network: dict[str, Any] = {"allow_public_traffic": False, "rules": rules}
    if egress == "permissive":
        return network
    if egress == "policy" and policy["default_action"] == "allow":
        network["deny_out"] = sorted(set(policy["deny_out"]))
        return network
    non_http_addresses = [address for entry in non_http for address in entry["addresses"]]
    if model_host not in policy["model_hosts"]:
        raise ValueError(f"model host {model_host!r} is not declared by the network policy")
    runtime_hosts = policy["runtime_dependency_hosts"] + [model_host] + non_http_addresses
    hosts = runtime_hosts
    if phase == "setup":
        hosts = runtime_hosts + policy["artifact_hosts"] + policy["registry_hosts"]
    network.update({"deny_out": ["0.0.0.0/0"], "allow_out": sorted(set(hosts))})
    return network


def _network_eligibility(policy: dict[str, Any], egress: EgressMode) -> dict[str, Any]:
    """Classify benchmark eligibility without treating public egress as pre-audited."""
    if egress == "permissive":
        return {
            "status": "ineligible",
            "reasons": ["permissive diagnostic egress is not benchmark eligible"],
        }
    if egress == "policy" and policy["default_action"] == "allow":
        return {
            "status": "requires_network_audit",
            "reasons": ["runtime policy permits public egress"],
        }
    if policy["runtime_dependency_hosts"] or policy["non_http_hosts"]:
        return {
            "status": "requires_network_audit",
            "reasons": ["runtime allowlist includes public dependency hosts"],
        }
    return {"status": "eligible", "reasons": []}


def _update_network(sandbox: Sandbox, network: dict[str, Any]) -> None:
    update = {key: value for key, value in network.items() if key != "allow_public_traffic"}
    sandbox.update_network(update, request_timeout=180)


@cache
def _path_sha256(path_value: str) -> str | None:
    """Hash a file or tree by relative path and content, excluding generated bytecode."""
    path = Path(path_value)
    if not path.exists():
        return None
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _experiment_identity(
    resolved: ResolvedTask,
    *,
    kind: Literal["smoke", "run"],
    upstream: Path,
    options: RunOptions,
    manifest_path: Path,
    network_policy_path: Path,
    patch_file: Path = DEFAULT_PATCH_FILE,
    remote_smoke: Path = DEFAULT_REMOTE_SMOKE,
    remote_install_codex: Path = DEFAULT_REMOTE_INSTALL_CODEX,
    manifest: TemplateManifest | None = None,
) -> dict[str, Any]:
    """Build the complete, deterministic identity used for artifact reuse."""
    require_immutable_runtime_image(resolved)
    manifest = manifest or TemplateManifest.load(manifest_path)
    template = _route_template(manifest, project=resolved.project, build_image=resolved.build_image)
    payload = {
        "kind": kind,
        "task": resolved_asdict(resolved),
        "options": asdict(options),
        "template": {
            "reference": template.reference,
            "manifest_sha256": _path_sha256(str(manifest_path.resolve())),
        },
        "network_policy_sha256": _path_sha256(str(network_policy_path.resolve())),
        "source": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "upstream_scripts_sha256": _path_sha256(str((upstream / "scripts").resolve())),
            "task_definition_sha256": _path_sha256(
                str((upstream / "projects" / resolved.project / resolved.task_id).resolve())
            ),
            "project_definition_sha256": _path_sha256(
                str((upstream / "projects" / resolved.project / "project.toml").resolve())
            ),
            "compatibility_patch_sha256": _path_sha256(str(patch_file.resolve())),
            "remote_smoke_sha256": _path_sha256(str(remote_smoke.resolve())),
            "remote_install_codex_sha256": _path_sha256(str(remote_install_codex.resolve())),
            "harness_sha256": _path_sha256(str(Path(__file__).resolve().parent)),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "inputs": payload,
    }


def _run_json(sandbox: Sandbox, command: str, *, timeout: int = 120) -> dict:
    result = sandbox.commands.run(command, timeout=timeout)
    return json.loads(result.stdout)


def _disk_and_docker(sandbox: Sandbox) -> dict:
    script = r"""python3 - <<'PY'
import json, shutil, subprocess
usage = shutil.disk_usage('/')
def run(command):
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    return {'exit_code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
print(json.dumps({
  'disk': {'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free},
  'docker_info': run("docker info --format '{{json .}}'"),
  'docker_df': run('docker system df'),
  'mmap_rnd_bits': open('/proc/sys/vm/mmap_rnd_bits').read().strip(),
}))
PY"""
    return _run_json(sandbox, script)


def _assert_preflight(metrics: dict, minimum_gb: int) -> None:
    free_gb = metrics["disk"]["free_bytes"] / 1_000_000_000
    if metrics["docker_info"]["exit_code"] != 0:
        raise RuntimeError(f"Docker daemon is unavailable: {metrics['docker_info']['stderr']}")
    if free_gb < minimum_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free; task requires at least {minimum_gb} GB")
    if metrics["mmap_rnd_bits"] != "28":
        raise RuntimeError("vm.mmap_rnd_bits is not 28")


def _upload_bundle(sandbox: Sandbox, bundle: bytes) -> dict[str, Any]:
    started = time.monotonic()
    sandbox.files.write("/tmp/cybergym-e2e-code.tgz", bundle)
    upload_seconds = time.monotonic() - started
    started = time.monotonic()
    sandbox.commands.run(
        "rm -rf /opt/cybergym-e2e && mkdir -p /opt/cybergym-e2e/data/projects && "
        "tar -xzf /tmp/cybergym-e2e-code.tgz -C /opt/cybergym-e2e && "
        "rm /tmp/cybergym-e2e-code.tgz",
        timeout=300,
    )
    return {
        "compressed_bytes": len(bundle),
        "file_api_seconds": round(upload_seconds, 6),
        "extract_seconds": round(time.monotonic() - started, 6),
    }


def _download_task_data(sandbox: Sandbox, task: str, timeout: int) -> dict[str, Any]:
    # Download the specific task files by exact path via hf_hub_download (the
    # resolve/CDN endpoint) instead of snapshot_download. snapshot_download with
    # allow_patterns still lists the ENTIRE recursive dataset tree via the
    # rate-limited /api/.../tree endpoint before filtering client-side, so at
    # high concurrency every task blows HF's 1000-requests-per-5-minutes /api/
    # quota and 429s. Per-file resolve requests avoid that quota entirely.
    code = f"""
import json
import os
import time
from huggingface_hub import hf_hub_download

REPO = {DATASET_REPOSITORY!r}
REV = {DATASET_REVISION!r}
LOCAL = '/opt/cybergym-e2e/data'
TASK = {task!r}

def fetch(name, needed):
    path = 'projects/' + TASK + '/' + name
    err = None
    started = time.monotonic()
    for attempt in range(6):
        try:
            local_path = hf_hub_download(
                repo_id=REPO, repo_type='dataset', revision=REV,
                filename=path, local_dir=LOCAL, token=False,
            )
            return {{
                'name': name,
                'required': needed,
                'present': True,
                'bytes': os.path.getsize(local_path),
                'attempts': attempt + 1,
                'duration_seconds': round(time.monotonic() - started, 6),
            }}
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status == 404 and not needed:
                return {{
                    'name': name,
                    'required': needed,
                    'present': False,
                    'bytes': 0,
                    'attempts': attempt + 1,
                    'duration_seconds': round(time.monotonic() - started, 6),
                }}
            err = exc
            time.sleep(min(60, 8 * (attempt + 1)))
    if needed and err is not None:
        raise err
    return {{
        'name': name,
        'required': needed,
        'present': False,
        'bytes': 0,
        'attempts': 6,
        'duration_seconds': round(time.monotonic() - started, 6),
    }}

files = [fetch(_name, True) for _name in ('src.tgz', 'poc.bin')]
files.append(fetch('crash.log', False))
print(json.dumps({{'files': files, 'total_bytes': sum(item['bytes'] for item in files)}}))
""".strip()
    result = sandbox.commands.run(
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "
        "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt "
        f"/opt/cybergym-e2e-venv/bin/python -c {shlex.quote(code)}",
        timeout=timeout,
    )
    profile = json.loads(result.stdout)
    sandbox.commands.run(
        f"test -s /opt/cybergym-e2e/data/projects/{shlex.quote(task)}/src.tgz && "
        f"test -e /opt/cybergym-e2e/data/projects/{shlex.quote(task)}/poc.bin",
        timeout=30,
    )
    return profile


def _ensure_image(sandbox: Sandbox, image: str, timeout: int) -> dict:
    code = f"""
import json
import subprocess
import time

IMAGE = {image!r}
started = time.monotonic()
probe = subprocess.run(['docker', 'image', 'inspect', IMAGE], text=True, capture_output=True)
probe_seconds = time.monotonic() - started
cache_hit = probe.returncode == 0
pull_seconds = 0.0
pull_stdout = ''
pull_stderr = ''
if not cache_hit:
    started = time.monotonic()
    pull = subprocess.run(['docker', 'pull', IMAGE], text=True, capture_output=True)
    pull_seconds = time.monotonic() - started
    pull_stdout = pull.stdout
    pull_stderr = pull.stderr
    if pull.returncode != 0:
        raise RuntimeError('docker pull failed: ' + pull.stderr[-2000:])
started = time.monotonic()
inspect = subprocess.run(['docker', 'image', 'inspect', IMAGE], text=True, capture_output=True)
metadata_seconds = time.monotonic() - started
if inspect.returncode != 0:
    raise RuntimeError('docker inspect failed: ' + inspect.stderr[-2000:])
info = json.loads(inspect.stdout)[0]
print(json.dumps({{
    'cache_hit': cache_hit,
    'cache_probe_seconds': round(probe_seconds, 6),
    'pull_seconds': round(pull_seconds, 6),
    'metadata_inspect_seconds': round(metadata_seconds, 6),
    'id': info.get('Id'),
    'repo_digests': info.get('RepoDigests', []),
    'size_bytes': info.get('Size'),
    'pull_stdout_tail': pull_stdout[-4000:],
    'pull_stderr_tail': pull_stderr[-4000:],
}}))
""".strip()
    started = time.monotonic()
    info = _run_json(
        sandbox,
        f"python3 -c {shlex.quote(code)}",
        timeout=timeout,
    )
    info["pull_or_verify_seconds"] = round(time.monotonic() - started, 6)
    return {key: value for key, value in info.items()}


def _assert_image_identity(runtime_image: str, metadata: dict) -> None:
    require_digest_locked_image(runtime_image)
    expected_digest = runtime_image.split("@", 1)[1]
    actual_digests = {
        value.split("@", 1)[1] for value in metadata.get("repo_digests", []) if "@" in value
    }
    if expected_digest not in actual_digests:
        raise RuntimeError(
            f"image digest mismatch for {runtime_image}: expected {expected_digest}, "
            f"found {sorted(actual_digests)}"
        )


def _route_template(manifest: TemplateManifest, *, project: str, build_image: str):
    """Route every FFmpeg task through the cache-bearing hot template.

    Only ten tasks use the shared ``cybergym/e2e:ffmpeg`` image, but legacy
    FFmpeg images run the same preparation scripts and need the same pinned
    Opus model archive. The hot snapshot can still pull each task's exact
    runtime image while supplying that common cache file.
    """
    if project == "ffmpeg" and FFMPEG_IMAGE in manifest.hot:
        return manifest.hot[FFMPEG_IMAGE]
    return manifest.route(build_image)


def _execution_context(
    resolved: ResolvedTask,
    *,
    options: RunOptions,
    manifest_path: Path,
    network_policy_path: Path,
) -> ExecutionContext:
    """Resolve the exact template, model, and network inputs used by preflight and runtime."""
    require_immutable_runtime_image(resolved)
    manifest = TemplateManifest.load(manifest_path)
    template = _route_template(
        manifest,
        project=resolved.project,
        build_image=resolved.build_image,
    )
    policy = _policy(network_policy_path)
    return ExecutionContext(
        manifest=manifest,
        template=template,
        policy=policy,
        non_http=_resolve_non_http(policy, resolved.project),
        model_config=_model_config(options),
    )


def _enable_swap(sandbox: Sandbox, size_gb: int) -> dict[str, Any]:
    if not 0 <= size_gb <= 16:
        raise ValueError("swap_gb must be between 0 and 16")
    if size_gb == 0:
        return {"enabled": False, "size_bytes": 0}
    size_bytes = size_gb * 1024**3
    result = sandbox.commands.run(
        "set -e; "
        "if ! swapon --noheadings --show=NAME | grep -qx /swapfile; then "
        "rm -f /swapfile; "
        f"fallocate -l {size_bytes} /swapfile; "
        "chmod 600 /swapfile; mkswap /swapfile >/dev/null; swapon /swapfile; "
        "fi; "
        "swapon --noheadings --bytes --show=NAME,SIZE,USED,PRIO",
        timeout=120,
    )
    return {
        "enabled": True,
        "size_bytes": size_bytes,
        "swapon": result.stdout.strip(),
    }


def _shell_run(sandbox: Sandbox, command: str, *, timeout: int) -> int:
    root = "/opt/cybergym-e2e"
    exit_path = f"{root}/e2b-exit-code"
    lock_path = f"{root}/e2b-workload.lock"
    wrapped = (
        "set +e; "
        + command
        + f" > {root}/e2b-run.log 2>&1; "
        + f"rc=$?; printf '%s\\n' \"$rc\" > {exit_path}; exit 0"
    )
    launcher = (
        f"if mkdir {lock_path} 2>/dev/null; then "
        f"rm -f {exit_path} {root}/e2b-run.log; "
        f"nohup bash -lc {shlex.quote(wrapped)} >/dev/null 2>&1 </dev/null & "
        "fi"
    )

    # Long commands.run streams can be severed by an intermediary TLS EOF even
    # though the sandbox process is still healthy. Launch idempotently and use
    # short polls so transient control-plane disconnects do not discard work.
    launch_error: Exception | None = None
    for _ in range(6):
        try:
            sandbox.commands.run(launcher, timeout=30)
            launch_error = None
            break
        except Exception as exc:
            launch_error = exc
            time.sleep(5)
    if launch_error is not None:
        raise launch_error

    deadline = time.monotonic() + timeout
    poll_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = sandbox.commands.run(f"test ! -f {exit_path} || cat {exit_path}", timeout=30)
            payload = result.stdout.strip()
            if payload:
                return int(payload)
            poll_error = None
        except Exception as exc:
            poll_error = exc
        time.sleep(10)
    detail = f"; last poll error: {poll_error}" if poll_error else ""
    raise TimeoutError(f"workload did not finish within {timeout} seconds{detail}")


def _start_resource_monitor(sandbox: Sandbox):
    script = """import json
import os
import shutil
import time

OUTPUT = "/opt/cybergym-e2e/e2b-resource-samples.jsonl"

def counters(path):
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = int(parts[1])
    return values

def cpu():
    with open("/proc/stat", encoding="utf-8") as handle:
        values = [int(value) for value in handle.readline().split()[1:]]
    return {
        "cpu_total_ticks": sum(values),
        "cpu_idle_ticks": values[3] + (values[4] if len(values) > 4 else 0),
    }

def network():
    received = sent = 0
    with open("/proc/net/dev", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            interface, payload = line.split(":", 1)
            if interface.strip() == "lo":
                continue
            values = payload.split()
            received += int(values[0])
            sent += int(values[8])
    return {"network_receive_bytes": received, "network_transmit_bytes": sent}

def block_io():
    read_sectors = write_sectors = io_milliseconds = 0
    with open("/proc/diskstats", encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) < 14:
                continue
            name = values[2]
            if name.startswith(("loop", "ram", "fd")):
                continue
            read_sectors += int(values[5])
            write_sectors += int(values[9])
            io_milliseconds += int(values[12])
    return {
        "block_read_bytes": read_sectors * 512,
        "block_write_bytes": write_sectors * 512,
        "block_io_milliseconds": io_milliseconds,
    }

while True:
    disk = shutil.disk_usage("/")
    meminfo = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0]) * 1024
    vmstat = counters("/proc/vmstat")
    load = os.getloadavg()
    sample = {
        "timestamp": time.time(),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "memory_total_bytes": meminfo["MemTotal"],
        "memory_available_bytes": meminfo["MemAvailable"],
        "swap_total_bytes": meminfo["SwapTotal"],
        "swap_free_bytes": meminfo["SwapFree"],
        "swap_used_bytes": meminfo["SwapTotal"] - meminfo["SwapFree"],
        "page_faults": vmstat.get("pgfault", 0),
        "major_page_faults": vmstat.get("pgmajfault", 0),
        "swap_in_pages": vmstat.get("pswpin", 0),
        "swap_out_pages": vmstat.get("pswpout", 0),
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
    }
    sample.update(cpu())
    sample.update(network())
    sample.update(block_io())
    with open(OUTPUT, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, sort_keys=True) + "\\n")
        handle.flush()
    time.sleep(1)
"""
    sandbox.files.write("/tmp/cybergym-resource-monitor.py", script)
    return sandbox.commands.run(
        "python3 /tmp/cybergym-resource-monitor.py >/dev/null 2>&1",
        background=True,
    )


def _resource_summary(sandbox: Sandbox) -> dict:
    path = "/opt/cybergym-e2e/e2b-resource-samples.jsonl"
    try:
        payload = sandbox.files.read(path)
    except Exception:
        return {"sample_count": 0}
    samples = [json.loads(line) for line in str(payload).splitlines() if line.strip()]
    if not samples:
        return {"sample_count": 0}
    summary = {
        "sample_count": len(samples),
        "interval_seconds": 1 if all("cpu_total_ticks" in item for item in samples) else 5,
        "disk_total_bytes": max(item["disk_total_bytes"] for item in samples),
        "peak_disk_used_bytes": max(item["disk_used_bytes"] for item in samples),
        "minimum_disk_free_bytes": min(item["disk_free_bytes"] for item in samples),
        "memory_total_bytes": max(item["memory_total_bytes"] for item in samples),
        "minimum_memory_available_bytes": min(item["memory_available_bytes"] for item in samples),
    }
    if all("swap_total_bytes" in item for item in samples):
        summary.update(
            {
                "swap_total_bytes": max(item["swap_total_bytes"] for item in samples),
                "peak_swap_used_bytes": max(item["swap_used_bytes"] for item in samples),
                "minimum_swap_free_bytes": min(item["swap_free_bytes"] for item in samples),
            }
        )
    first = samples[0]
    last = samples[-1]
    counter_fields = {
        "network_receive_bytes": "network_receive_bytes",
        "network_transmit_bytes": "network_transmit_bytes",
        "block_read_bytes": "block_read_bytes",
        "block_write_bytes": "block_write_bytes",
        "block_io_milliseconds": "block_io_milliseconds",
        "page_faults": "page_faults",
        "major_page_faults": "major_page_faults",
        "swap_in_pages": "swap_in_pages",
        "swap_out_pages": "swap_out_pages",
    }
    for source, destination in counter_fields.items():
        if source in first and source in last:
            summary[f"delta_{destination}"] = max(0, last[source] - first[source])
    if all("load_1m" in item for item in samples):
        summary["peak_load_1m"] = max(item["load_1m"] for item in samples)
    if all("cpu_total_ticks" in item and "cpu_idle_ticks" in item for item in samples):
        total = last["cpu_total_ticks"] - first["cpu_total_ticks"]
        idle = last["cpu_idle_ticks"] - first["cpu_idle_ticks"]
        if total > 0:
            summary["average_cpu_busy_percent"] = round(100 * (total - idle) / total, 3)
    return summary


def _collect(sandbox: Sandbox, destination: Path) -> None:
    sandbox.commands.run(
        "cd /opt/cybergym-e2e && "
        "rm -rf /tmp/cybergym-e2e-results && "
        "mkdir -p /tmp/cybergym-e2e-results && "
        "for path in e2b-run.log e2b-exit-code e2b-resource-samples.jsonl "
        "smoke-result.json e2b-workload-profile.json agent_output; do "
        'test ! -e "$path" || cp -a "$path" /tmp/cybergym-e2e-results/; '
        "done && "
        "tar -czf /tmp/cybergym-e2e-results.tgz -C /tmp/cybergym-e2e-results .",
        timeout=300,
    )
    payload = bytes(sandbox.files.read("/tmp/cybergym-e2e-results.tgz", format="bytes"))
    archive_path = destination / "sandbox-results.tgz"
    archive_path.write_bytes(payload)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / "sandbox").resolve() / member.name
            if not target.is_relative_to((destination / "sandbox").resolve()):
                raise RuntimeError("unsafe path in sandbox result archive")
        archive.extractall(destination / "sandbox", filter="data")


def _codex_turn_state(destination: Path) -> dict[str, Any]:
    logs = sorted((destination / "sandbox" / "agent_output").glob("*/*/trajectory/*.log"))
    completed_turns = 0
    failed_turns = 0
    last_error: str | None = None
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                completed_turns += 1
            elif event.get("type") == "turn.failed":
                failed_turns += 1
                last_error = str((event.get("error") or {}).get("message") or "")
    if completed_turns:
        status = "completed"
    elif failed_turns:
        status = "failed"
    else:
        status = "missing"
    result: dict[str, Any] = {
        "status": status,
        "completed_turns": completed_turns,
        "failed_turns": failed_turns,
    }
    if last_error:
        result["last_error"] = last_error
    return result


def _benchmark_result(destination: Path, kind: Literal["smoke", "run"]) -> dict:
    sandbox_results = destination / "sandbox"
    if kind == "smoke":
        path = sandbox_results / "smoke-result.json"
        if not path.is_file():
            return {"status": "error", "reason": "smoke-result.json is missing"}
        details = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": "passed" if details.get("stage4") == "passed" else "failed",
            "outcome": (
                "ground_truth_passed"
                if details.get("stage4") == "passed"
                else "ground_truth_failed"
            ),
            "stages": details,
        }

    summaries = sorted(sandbox_results.glob("agent_output/*/*/summary.json"))
    if len(summaries) != 1:
        return {
            "status": "error",
            "reason": f"expected one agent summary, found {len(summaries)}",
        }
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    upstream_status = summary.get("status")
    if summary.get("agent") == "codex":
        turn = _codex_turn_state(destination)
        summary["agent_turn"] = turn
        # A validated S1-S3 success remains valid even if the provider connection
        # failed after producing the artifacts. A failed benchmark without a
        # completed model turn is an interrupted trial, not a model failure.
        if upstream_status != "success" and turn["status"] != "completed":
            return {
                "status": "error",
                "outcome": "incomplete_agent_turn",
                "reason": "Codex did not complete a model turn",
                "agent_turn": turn,
                "upstream_summary": summary,
            }
    if upstream_status == "success":
        successful_attempt = next(
            (
                attempt
                for attempt in summary.get("attempts", [])
                if attempt.get("agent_success") is True or attempt.get("success") is True
            ),
            None,
        )
        if successful_attempt is None:
            return {
                "status": "error",
                "outcome": "invalid_upstream_summary",
                "reason": "upstream reported success without a successful attempt",
                "upstream_summary": summary,
            }
        summary["status"] = "passed"
        summary["outcome"] = (
            "exact_match"
            if successful_attempt.get("gt_success") is True
            else "valid_other_vulnerability"
        )
        summary["ground_truth_reproduced"] = successful_attempt.get("gt_success") is True
        summary["successful_attempt"] = successful_attempt.get("attempt")
    elif upstream_status == "failed":
        summary["status"] = "failed"
        summary["outcome"] = "failed"
    elif upstream_status == "error":
        summary["status"] = "error"
        summary["outcome"] = "upstream_error"
    else:
        return {
            "status": "error",
            "outcome": "invalid_upstream_summary",
            "reason": f"unexpected upstream status: {upstream_status!r}",
            "upstream_summary": summary,
        }
    summary["upstream_status"] = upstream_status
    return summary


def _create_fresh_sandbox_from_template(
    template_reference: str,
    *,
    timeout: int,
    metadata: dict[str, str],
    network: dict[str, Any],
) -> Sandbox:
    """Create a new sandbox identity from an immutable template snapshot.

    E2B restores the read-only template base internally, then gives this sandbox
    private copy-on-write memory and rootfs state. This never resumes a prior task
    sandbox, and automatic lifecycle resume stays disabled.
    """
    return Sandbox.create(
        template=template_reference,
        timeout=timeout,
        metadata=metadata,
        network=network,
        lifecycle={"on_timeout": "kill", "auto_resume": False},
        request_timeout=180,
    )


def _finalize(sandbox: Sandbox, retain: bool, result: dict) -> None:
    cleanup_error: Exception | None = None
    for attempt in range(3):
        try:
            if retain:
                sandbox.pause()
                result["final_state"] = "paused"
            else:
                sandbox.kill()
                result["final_state"] = "killed"
            return
        except Exception as exc:
            cleanup_error = exc
            if attempt < 2:
                time.sleep(5)
    result["final_state"] = "cleanup_failed"
    result["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"


def _stop_resource_monitor(monitor: Any, result: dict, *, stage: str) -> None:
    """Best-effort monitor cleanup that cannot suppress run finalization."""
    try:
        monitor.kill()
    except Exception as exc:
        result.setdefault("monitor_errors", []).append(
            {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        )


def execute_task(
    resolved: ResolvedTask,
    *,
    kind: Literal["smoke", "run"],
    upstream: Path,
    artifacts_dir: Path,
    options: RunOptions,
    manifest_path: Path = DEFAULT_MANIFEST,
    network_policy_path: Path = DEFAULT_NETWORK_POLICY,
    patch_file: Path = DEFAULT_PATCH_FILE,
    remote_smoke: Path = DEFAULT_REMOTE_SMOKE,
    remote_install_codex: Path = DEFAULT_REMOTE_INSTALL_CODEX,
    batch_id: str | None = None,
    experiment: dict[str, Any] | None = None,
) -> dict:
    context = _execution_context(
        resolved,
        options=options,
        manifest_path=manifest_path,
        network_policy_path=network_policy_path,
    )
    hf_token = _secret("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required and must have accepted the gated CyberGym-E2E dataset"
        )
    model_config = context.model_config
    model_key = model_config["key"]
    if kind == "run" and not model_key:
        raise RuntimeError(f"{model_config['key_name']} is required for {options.provider}")
    manifest = context.manifest
    template = context.template
    policy = context.policy
    non_http = context.non_http
    template_receipt = verify_template_ref(template)
    profiler = _StageProfiler()
    with profiler.stage("client_bundle_build"):
        bundle = build_code_bundle(
            upstream,
            resolved,
            patch_file=patch_file,
            remote_smoke=remote_smoke,
            remote_install_codex=remote_install_codex,
        )
    current_experiment = _experiment_identity(
        resolved,
        kind=kind,
        upstream=upstream,
        options=options,
        manifest_path=manifest_path,
        network_policy_path=network_policy_path,
        patch_file=patch_file,
        remote_smoke=remote_smoke,
        remote_install_codex=remote_install_codex,
        manifest=manifest,
    )
    if experiment is not None and experiment.get("sha256") != current_experiment["sha256"]:
        raise ValueError("precomputed experiment identity does not match the execution inputs")
    experiment = current_experiment
    run_id = uuid.uuid4().hex
    output = artifacts_dir / resolved.project / resolved.task_id / run_id
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 4,
        "kind": kind,
        "run_id": run_id,
        "batch_id": batch_id,
        "task": resolved_asdict(resolved),
        "template": template.reference,
        "template_receipt": {**template_receipt, "recipe_sha256": template.recipe_sha256},
        "experiment": experiment,
        "resources": {
            "cpu_count": manifest.cpu_count,
            "memory_mb": manifest.memory_mb,
            "disk_limit_gb": manifest.disk_limit_gb,
            "swap_gb": options.swap_gb,
        },
        "lifecycle": {
            "creation": "fresh_from_template",
            "state_isolation": "private_copy_on_write",
            "initial_timeout_seconds": options.setup_timeout,
            "evaluation_timeout_seconds": options.evaluation_timeout,
            "on_timeout": "kill",
            "auto_resume": False,
        },
        "network": {
            "public_ingress": False,
            "egress": options.egress,
            "default_action": (
                "deny"
                if options.egress == "restricted"
                else "allow"
                if options.egress == "permissive"
                else policy["default_action"]
            ),
            "eligibility": _network_eligibility(policy, options.egress),
            "secrets": "proxy-only",
            "non_http_egress": non_http,
        },
        "model_provider": {
            "name": options.provider,
            "host": model_config["host"],
            "base_url": model_config["base_url"],
            "credential_delivery": "e2b_request_transform",
        },
        "artifact_dir": str(output),
        "error": None,
    }
    sandbox: Sandbox | None = None
    resource_monitor = None
    stage = "create"
    try:
        setup_network = _network(
            policy,
            phase="setup",
            egress=options.egress,
            hf_token=hf_token,
            model_key=None,
            non_http=non_http,
            model_host=str(model_config["host"]),
        )
        with profiler.stage("sandbox_create"):
            sandbox = _create_fresh_sandbox_from_template(
                template.reference,
                timeout=options.setup_timeout,
                metadata={
                    "project": "cybergym-e2e-e2b",
                    "run_id": run_id,
                    "batch_id": batch_id or "single",
                    "task": resolved.task,
                },
                network=setup_network,
            )
        result["sandbox_id"] = sandbox.sandbox_id
        stage = "enable_swap"
        with profiler.stage("sandbox_kernel_config"):
            sandbox.commands.run("sysctl -w vm.mmap_rnd_bits=28 >/dev/null", timeout=30)
        with profiler.stage("swap_enable"):
            result["swap"] = _enable_swap(sandbox, options.swap_gb)
        stage = "preflight"
        with profiler.stage("sandbox_preflight"):
            before = _disk_and_docker(sandbox)
        minimum = (
            options.ffmpeg_min_free_gb if resolved.project == "ffmpeg" else options.min_free_gb
        )
        _assert_preflight(before, minimum)
        result["preflight"] = before
        with profiler.stage("resource_monitor_start"):
            resource_monitor = _start_resource_monitor(sandbox)
        stage = "upload_code_bundle"
        with profiler.stage("code_bundle_upload"):
            result["code_bundle"] = _upload_bundle(sandbox, bundle)
        stage = "download_hf_task_data"
        with profiler.stage("task_data_download") as data_stage:
            result["task_data"] = _download_task_data(
                sandbox, resolved.task, options.setup_timeout - 300
            )
        result["hf_download_seconds"] = data_stage["duration_seconds"]
        stage = "verify_or_pull_project_image"
        with profiler.stage("project_image_pull_or_verify"):
            result["image"] = _ensure_image(
                sandbox, resolved.runtime_image, options.setup_timeout - 300
            )
        _assert_image_identity(resolved.runtime_image, result["image"])
        with profiler.stage("prepared_metrics"):
            result["prepared"] = _disk_and_docker(sandbox)
        stage = "update_runtime_network"
        runtime_network = _network(
            policy,
            phase="runtime",
            egress=options.egress,
            hf_token=None,
            model_key=model_key if kind == "run" else None,
            non_http=non_http,
            model_host=str(model_config["host"]),
        )
        # Network updates replace the phase policy atomically. This always runs so
        # the setup-only Hugging Face credential rule is removed before the workload.
        with profiler.stage("runtime_network_update"):
            _update_network(sandbox, runtime_network)
        stage = "extend_lifetime"
        with profiler.stage("sandbox_timeout_update"):
            sandbox.set_timeout(options.evaluation_timeout, request_timeout=180)
        stage = kind
        root = "/opt/cybergym-e2e"
        cache_env = f"export E2B_OPUS_MODEL_CACHE={shlex.quote(OPUS_MODEL_CACHE)}; "
        if kind == "smoke":
            command = (
                cache_env + f"cd {root} && /opt/cybergym-e2e-venv/bin/python "
                f"scripts/e2b_smoke.py {shlex.quote(resolved.task)}"
            )
        else:
            upstream_model = _agent_model_id(options)
            command = (
                cache_env + "export E2B_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt; "
                f"export OPENAI_BASE_URL={shlex.quote(str(model_config['base_url']))}; "
                "export OPENAI_API_KEY=e2b-proxy-injected; "
                f"cd {root} && /opt/cybergym-e2e-venv/bin/python scripts/run_agent.py "
                f"{shlex.quote(resolved.task)} --mode e2e --agent {shlex.quote(options.agent)} "
                f"--prompt-style {shlex.quote(options.prompt_style)} "
                f"--max-attempts {options.max_attempts} --timeout {options.agent_timeout} "
                "--model-provider openai-compatible "
                f"--litellm-model-id {shlex.quote(upstream_model)} "
                "--data-dir ./data/projects --script-dir ./projects --agent-output ./agent_output"
            )
        try:
            with profiler.stage("workload"):
                result["workload_exit_code"] = _shell_run(
                    sandbox, command, timeout=options.evaluation_timeout - 120
                )
        finally:
            if resource_monitor is not None:
                with profiler.stage("resource_monitor_stop"):
                    _stop_resource_monitor(resource_monitor, result, stage="after_workload")
                resource_monitor = None
            with profiler.stage("resource_summary"):
                result["observed_resources"] = _resource_summary(sandbox)
            with profiler.stage("post_workload_metrics"):
                result["post_workload"] = _disk_and_docker(sandbox)
        stage = "collect"
        with profiler.stage("artifact_collect"):
            _collect(sandbox, output)
        with profiler.stage("benchmark_parse"):
            result["benchmark"] = _benchmark_result(output, kind)
        result["completed"] = True
    except Exception as exc:
        result["completed"] = False
        result["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if sandbox is not None:
            if resource_monitor is not None:
                with profiler.stage("resource_monitor_stop_after_failure"):
                    _stop_resource_monitor(resource_monitor, result, stage="after_failure")
                resource_monitor = None
            with profiler.stage("resource_summary_after_failure"):
                result["observed_resources"] = _resource_summary(sandbox)
            try:
                with profiler.stage("post_failure_metrics"):
                    result["post_failure"] = _disk_and_docker(sandbox)
            except Exception as metrics_exc:
                result["metrics_error"] = f"{type(metrics_exc).__name__}: {metrics_exc}"
            try:
                with profiler.stage("artifact_collect_after_failure"):
                    _collect(sandbox, output)
            except Exception as collect_exc:
                result["collection_error"] = f"{type(collect_exc).__name__}: {collect_exc}"
    finally:
        if sandbox is not None:
            if resource_monitor is not None:
                with profiler.stage("resource_monitor_stop_finally"):
                    _stop_resource_monitor(resource_monitor, result, stage="finalize")
            with profiler.stage("sandbox_finalize"):
                _finalize(sandbox, options.retain, result)
        result["profile"] = profiler.result()
        result["duration_seconds"] = result["profile"]["duration_seconds"]
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def preflight_access(task: str, hf_token: str | None) -> dict:
    if not hf_token:
        return {"ok": False, "reason": "HF_TOKEN is missing"}
    url = (
        f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
        f"{DATASET_REVISION}/projects/{task}/src.tgz"
    )
    response = httpx.head(
        url,
        headers={"Authorization": f"Bearer {hf_token}"},
        follow_redirects=False,
        timeout=30,
    )
    return {
        "ok": response.status_code in {200, 302, 303, 307, 308},
        "status_code": response.status_code,
        "reason": None if response.status_code < 400 else "dataset access was rejected",
    }
