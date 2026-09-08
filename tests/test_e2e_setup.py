from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from cybergym_e2b.cli import _already_completed, _batch, _options, _parser, main
from cybergym_e2b.config import (
    BASE_BUILDER_IMAGES,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_NETWORK_POLICY,
    DEFAULT_PATCH_FILE,
    DEFAULT_REMOTE_APT_RETRY,
    DEFAULT_REMOTE_INSTALL_CODEX,
    DEFAULT_REMOTE_SMOKE,
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    TemplateManifest,
    TemplateRef,
    normalize_task,
)
from cybergym_e2b.inventory import build_code_bundle, inventory, load_image_map, resolve_task
from cybergym_e2b.runtime import (
    RunOptions,
    _agent_model_id,
    _assert_image_identity,
    _benchmark_result,
    _create_fresh_sandbox_from_template,
    _experiment_identity,
    _model_config,
    _network,
    _network_eligibility,
    _policy,
    _resource_summary,
    _route_template,
    _shell_run,
    _stop_resource_monitor,
    _summary_wire_api,
)

UPSTREAM = Path("vendor/cybergym-e2e")


def test_validated_8c8g_configuration_is_the_default() -> None:
    assert DEFAULT_MANIFEST.parts[-3:] == ("artifacts", "templates", "manifest.json")
    assert DEFAULT_MODEL == "openai.gpt-5.4"
    assert DEFAULT_MODEL_PROVIDER == "bedrock"
    assert RunOptions().reasoning_effort == "high"


def test_reasoning_effort_is_explicit_and_configurable() -> None:
    default_args = _parser().parse_args(["run", "curl/arvo_66012"])
    assert _options(default_args).reasoning_effort == "high"

    args = _parser().parse_args(["run", "curl/arvo_66012", "--reasoning-effort", "xhigh"])
    assert _options(args).reasoning_effort == "xhigh"


def test_summary_wire_api_matches_agent_transport() -> None:
    assert _summary_wire_api("codex") == "responses"
    assert _summary_wire_api("openhands") == "chat-completions"


def test_apt_retry_wrapper_covers_apt_and_apt_get(tmp_path: Path) -> None:
    fake_real_dir = tmp_path / "real"
    wrapper_dir = tmp_path / "wrapper"
    fake_real_dir.mkdir()
    wrapper_dir.mkdir()
    fake_command = """#!/bin/sh
count=$(cat "$E2B_APT_COUNTER" 2>/dev/null || printf 0)
count=$((count + 1))
printf '%s' "$count" > "$E2B_APT_COUNTER"
printf '%s\\n' "$*" >> "$E2B_APT_ARGUMENTS"
if [ "$1" = "-o" ] && [ "$count" -lt 3 ]; then exit 42; fi
exit 0
"""

    for command_name in ("apt", "apt-get"):
        real_command = fake_real_dir / command_name
        real_command.write_text(fake_command, encoding="utf-8")
        real_command.chmod(0o755)
        wrapper = wrapper_dir / command_name
        shutil.copy2(DEFAULT_REMOTE_APT_RETRY, wrapper)
        wrapper.chmod(0o755)
        counter = tmp_path / f"{command_name}.counter"
        arguments = tmp_path / f"{command_name}.arguments"
        env = {
            **os.environ,
            "E2B_APT_REAL_DIR": str(fake_real_dir),
            "E2B_APT_RETRY_SLEEP": "0",
            "E2B_APT_COUNTER": str(counter),
            "E2B_APT_ARGUMENTS": str(arguments),
        }

        subprocess.run([str(wrapper), "update", "-qq"], env=env, check=True)
        assert counter.read_text(encoding="utf-8") == "3"
        calls = arguments.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 3
        assert all("Acquire::Retries=3" in call for call in calls)
        assert all("Acquire::http::No-Cache=true" in call for call in calls)

        subprocess.run([str(wrapper), "install", "-y", "git"], env=env, check=True)
        assert counter.read_text(encoding="utf-8") == "4"
        assert arguments.read_text(encoding="utf-8").splitlines()[-1] == "install -y git"


def test_default_egress_enforces_the_default_allow_policy() -> None:
    args = _parser().parse_args(["run", "curl/arvo_66012"])
    assert args.egress == "policy"

    policy = _policy(DEFAULT_NETWORK_POLICY)
    assert policy["default_action"] == "allow"
    assert "169.254.0.0/16" in policy["deny_out"]


def test_default_public_egress_requires_a_network_audit_for_eligibility() -> None:
    assert _network_eligibility(_policy(DEFAULT_NETWORK_POLICY), "policy") == {
        "status": "requires_network_audit",
        "reasons": ["runtime policy permits public egress"],
    }
    assert _network_eligibility(
        _policy(DEFAULT_NETWORK_POLICY.with_name("network-locked.json")), "policy"
    ) == {"status": "eligible", "reasons": []}
    assert _network_eligibility(_policy(DEFAULT_NETWORK_POLICY), "permissive") == {
        "status": "ineligible",
        "reasons": ["permissive diagnostic egress is not benchmark eligible"],
    }
    assert _network_eligibility(_policy(DEFAULT_NETWORK_POLICY), "restricted") == {
        "status": "requires_network_audit",
        "reasons": ["runtime allowlist includes public dependency hosts"],
    }


def test_network_credentials_are_scoped_to_their_phase() -> None:
    policy = _policy(DEFAULT_NETWORK_POLICY)
    setup = _network(
        policy,
        phase="setup",
        egress="policy",
        hf_token="hf-secret",
        model_key="model-secret",
        non_http=[],
    )
    assert set(setup["rules"]) == {"huggingface.co"}
    assert set(setup["deny_out"]) == set(policy["deny_out"])
    assert "allow_out" not in setup
    runtime = _network(
        policy,
        phase="runtime",
        egress="policy",
        hf_token="hf-secret",
        model_key="model-secret",
        non_http=[],
    )
    assert set(runtime["rules"]) == {"api.fireworks.ai"}
    assert set(runtime["deny_out"]) == set(policy["deny_out"])
    assert "allow_out" not in runtime

    locked_policy = _policy(DEFAULT_NETWORK_POLICY.with_name("network-locked.json"))
    locked_runtime = _network(
        locked_policy,
        phase="runtime",
        egress="policy",
        hf_token="hf-secret",
        model_key="model-secret",
        non_http=[],
    )
    assert locked_runtime["deny_out"] == ["0.0.0.0/0"]
    assert locked_runtime["allow_out"] == ["api.fireworks.ai"]
    assert set(locked_runtime["rules"]) == {"api.fireworks.ai"}

    legacy_restricted = _network(
        policy,
        phase="runtime",
        egress="restricted",
        hf_token="hf-secret",
        model_key="model-secret",
        non_http=[],
    )
    assert legacy_restricted["deny_out"] == ["0.0.0.0/0"]
    assert "api.fireworks.ai" in legacy_restricted["allow_out"]

    bedrock = _network(
        policy,
        phase="runtime",
        egress="policy",
        hf_token=None,
        model_key="bedrock-secret",
        non_http=[],
        model_host="bedrock-mantle.us-west-2.api.aws",
    )
    assert set(bedrock["rules"]) == {"bedrock-mantle.us-west-2.api.aws"}


def test_bedrock_provider_uses_mantle_responses_endpoint() -> None:
    args = _parser().parse_args(
        [
            "run",
            "curl/arvo_66012",
            "--provider",
            "bedrock",
            "--model",
            "openai.gpt-5.4",
        ]
    )
    options = RunOptions(provider=args.provider, bedrock_region=args.bedrock_region)
    config = _model_config(options)
    assert config["host"] == "bedrock-mantle.us-west-2.api.aws"
    assert config["base_url"] == ("https://bedrock-mantle.us-west-2.api.aws/openai/v1")


def test_bedrock_openhands_uses_mantle_chat_completions_endpoint() -> None:
    options = RunOptions(
        provider="bedrock",
        agent="openhands",
        model="deepseek.v3.2",
        bedrock_region="us-west-2",
    )
    config = _model_config(options)
    assert config["host"] == "bedrock-mantle.us-west-2.api.aws"
    assert config["base_url"] == "https://bedrock-mantle.us-west-2.api.aws/v1"
    assert _agent_model_id(options) == "openai/deepseek.v3.2"
    assert _agent_model_id(replace(options, agent="codex")) == "deepseek.v3.2"


def test_network_policy_rejects_unsupported_deny_cidrs(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_NETWORK_POLICY.read_text(encoding="utf-8"))
    raw["deny_out"] = ["0.0.0.0/8"]
    path = tmp_path / "network.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        _policy(path)
    except ValueError as exc:
        assert "not supported by E2B" in str(exc)
    else:
        raise AssertionError("an E2B-invalid deny CIDR was accepted")

    raw["deny_out"] = ["240.0.0.0/4"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        _policy(path)
    except ValueError as exc:
        assert "not supported by E2B" in str(exc)
    else:
        raise AssertionError("E2B's platform-address range was accepted in deny_out")


def test_pinned_inventory_shape() -> None:
    result = inventory(UPSTREAM)
    assert result["tasks"] == 920
    assert result["projects"] == 139
    assert result["unique_build_images"] == 509
    assert result["single_use_build_images"] == 487
    counts = {row["build_image"]: row["task_count"] for row in result["images"]}
    assert sum(counts[image] for image in BASE_BUILDER_IMAGES) == 344
    assert counts[FFMPEG_IMAGE] == 10


def test_task_resolution_uses_known_ffmpeg_digest_pin() -> None:
    original = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305")
    assert original.build_image == FFMPEG_IMAGE
    assert original.runtime_image == FFMPEG_IMAGE_DIGEST


def test_image_map_accepts_only_digest_locked_values(tmp_path: Path) -> None:
    digest = FFMPEG_IMAGE_DIGEST.rsplit(":", 1)[1]
    path = tmp_path / "images.json"
    path.write_text(
        json.dumps({"images": {FFMPEG_IMAGE: f"mirror.invalid/cybergym/e2e@sha256:{digest}"}}),
        encoding="utf-8",
    )
    image_map = load_image_map(path)
    mapped = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305", image_map=image_map)
    assert mapped.build_image == FFMPEG_IMAGE
    assert mapped.runtime_image == f"mirror.invalid/cybergym/e2e@sha256:{digest}"

    path.write_text(
        json.dumps({"images": {FFMPEG_IMAGE: "mirror.invalid/cybergym/e2e:ffmpeg"}}),
        encoding="utf-8",
    )
    try:
        load_image_map(path)
    except ValueError as exc:
        assert "digest-locked" in str(exc)
    else:
        raise AssertionError("a mutable image-map value was accepted")

    path.write_text(
        json.dumps({"images": {FFMPEG_IMAGE: "mirror.invalid/cybergym/e2e@sha256:" + "b" * 64}}),
        encoding="utf-8",
    )
    try:
        load_image_map(path)
    except ValueError as exc:
        assert "known FFmpeg digest" in str(exc)
    else:
        raise AssertionError("an image map changed the pinned FFmpeg image content")


def test_bundle_is_task_scoped_and_applies_provider_patch() -> None:
    assert "if ! command -v curl" in DEFAULT_REMOTE_INSTALL_CODEX.read_text()
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    payload = build_code_bundle(
        UPSTREAM,
        resolved,
        patch_file=DEFAULT_PATCH_FILE,
        remote_smoke=DEFAULT_REMOTE_SMOKE,
    )
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
        run_agent = archive.extractfile("scripts/run_agent.py")
        assert run_agent is not None
        source = run_agent.read().decode()
        utils = archive.extractfile("scripts/utils.py")
        assert utils is not None
        utils_source = utils.read().decode()
        validator_installer = archive.extractfile("scripts/install_validate_deps.sh")
        assert validator_installer is not None
        validator_installer_source = validator_installer.read().decode()
    assert "projects/curl/arvo_66012/config.toml" in names
    assert "scripts/apt_retry.sh" in names
    assert not any("projects/ffmpeg/" in name for name in names)
    assert '"openai-compatible"' in source
    assert 'wire_api = "responses"' in source
    assert 'model_reasoning_effort = "{args.reasoning_effort}"' in source
    assert source.index("model_reasoning_effort") < source.index("[model_providers.openai_http]")
    assert 'parser.add_argument("--reasoning-effort"' in source
    assert "if attempt < args.max_attempts:" in source
    assert 'model_provider == "openai-compatible"' in utils_source
    assert "client.responses.create(" in utils_source
    assert 'httpx.Client(verify=os.environ.get("E2B_CA_BUNDLE", True))' in utils_source
    assert "command -v sudo" in utils_source
    assert 'exec \\"$@\\"' in utils_source
    assert "command -v git" in utils_source
    assert 'for apt_command in ("apt", "apt-get")' in utils_source
    assert 'scripts_dir / "apt_retry.sh"' in utils_source
    assert "poc_file is None or not poc_file.exists()" in source
    assert "patch_file is None or not patch_file.exists()" in source
    assert "if ! command -v curl" in validator_installer_source
    compile(source, "scripts/run_agent.py", "exec")
    compile(utils_source, "scripts/utils.py", "exec")
    assert 'os.environ.get("E2B_CA_BUNDLE")' in source
    assert 'os.getenv("E2B_OPUS_MODEL_CACHE")' in utils_source
    assert '"LLM_MAX_INPUT_TOKENS": "131072"' in utils_source
    assert '"LLM_MAX_OUTPUT_TOKENS": "8192"' in utils_source
    assert '"LLM_NATIVE_TOOL_CALLING": "true"' in utils_source


def test_openai_compatible_summary_uses_agent_wire_api(monkeypatch) -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    payload = build_code_bundle(UPSTREAM, resolved)
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        utils = archive.extractfile("scripts/utils.py")
        assert utils is not None
        module = ast.parse(utils.read().decode())
    call_llm_node = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "call_llm"
    )

    calls: list[tuple[str, dict]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.responses = SimpleNamespace(
                create=lambda **call: (
                    calls.append(("responses", call))
                    or SimpleNamespace(output_text="responses summary")
                )
            )
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **call: (
                        calls.append(("chat-completions", call))
                        or SimpleNamespace(
                            choices=[
                                SimpleNamespace(message=SimpleNamespace(content="chat summary"))
                            ]
                        )
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    namespace = {
        "boto3": SimpleNamespace(),
        "httpx": SimpleNamespace(Client=lambda **kwargs: ("http-client", kwargs)),
        "os": os,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[call_llm_node], type_ignores=[])),
            "utils.py",
            "exec",
        ),
        namespace,
    )
    call_llm = namespace["call_llm"]

    monkeypatch.setenv("E2B_OPENAI_WIRE_API", "responses")
    assert (
        call_llm(
            "summarize",
            model_provider="openai-compatible",
            litellm_model_id="openai/openai.gpt-5.4",
        )
        == "responses summary"
    )
    assert calls[-1] == (
        "responses",
        {
            "model": "openai.gpt-5.4",
            "input": "summarize",
            "max_output_tokens": 2000,
        },
    )

    monkeypatch.setenv("E2B_OPENAI_WIRE_API", "chat-completions")
    assert (
        call_llm(
            "summarize",
            model_provider="openai-compatible",
            litellm_model_id="openai/openai.gpt-5.4",
        )
        == "chat summary"
    )
    assert calls[-1] == (
        "chat-completions",
        {
            "model": "openai.gpt-5.4",
            "messages": [{"role": "user", "content": "summarize"}],
            "max_tokens": 2000,
            "temperature": 0.0,
        },
    )


def test_manifest_round_trip_and_routing(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    base = TemplateRef(
        name="base",
        tag="recipe-aaaaaaaaaaaaaaaa",
        template_id="template-base",
        build_id="build-base-1234",
        recipe_sha256="a" * 64,
        images=BASE_BUILDER_IMAGES,
    )
    hot = TemplateRef(
        name="ffmpeg",
        tag="recipe-bbbbbbbbbbbbbbbb",
        template_id="template-ffmpeg",
        build_id="build-ffmpeg-1234",
        recipe_sha256="b" * 64,
        images=(*BASE_BUILDER_IMAGES, FFMPEG_IMAGE_DIGEST),
    )
    manifest = TemplateManifest(base=base, hot={FFMPEG_IMAGE: hot})
    manifest.write(path)
    loaded = TemplateManifest.load(path)
    assert loaded.route(FFMPEG_IMAGE).reference == hot.reference
    assert loaded.route("n132/arvo:1-fix").reference == base.reference
    raw = json.loads(path.read_text())
    assert raw["resources"]["disk_limit_gb"] == 120


def test_all_ffmpeg_tasks_use_cache_bearing_hot_template() -> None:
    base = TemplateRef(
        name="base",
        tag="recipe-aaaaaaaaaaaaaaaa",
        template_id="template-base",
        build_id="build-base-1234",
        recipe_sha256="a" * 64,
        images=BASE_BUILDER_IMAGES,
    )
    hot = TemplateRef(
        name="ffmpeg",
        tag="recipe-bbbbbbbbbbbbbbbb",
        template_id="template-ffmpeg",
        build_id="build-ffmpeg-1234",
        recipe_sha256="b" * 64,
        images=(*BASE_BUILDER_IMAGES, FFMPEG_IMAGE_DIGEST),
    )
    manifest = TemplateManifest(base=base, hot={FFMPEG_IMAGE: hot})

    assert (
        _route_template(
            manifest,
            project="ffmpeg",
            build_image="cybergym/e2e:ffmpeg-legacy-task-image",
        ).reference
        == hot.reference
    )
    assert (
        _route_template(
            manifest,
            project="curl",
            build_image="gcr.io/oss-fuzz-base/base-builder",
        ).reference
        == base.reference
    )


def test_task_path_rejects_traversal() -> None:
    try:
        normalize_task("curl/../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal was accepted")


def test_ffmpeg_digest_is_enforced_even_through_a_mirror() -> None:
    digest = FFMPEG_IMAGE_DIGEST.split("@", 1)[1]
    _assert_image_identity(
        FFMPEG_IMAGE_DIGEST,
        {"repo_digests": [f"mirror.invalid/cybergym/e2e@{digest}"]},
    )
    try:
        _assert_image_identity(
            FFMPEG_IMAGE_DIGEST,
            {"repo_digests": ["mirror.invalid/cybergym/e2e@sha256:wrong"]},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("an unexpected FFmpeg digest was accepted")


def test_benchmark_result_separates_benchmark_status_from_infrastructure(tmp_path: Path) -> None:
    summary = tmp_path / "sandbox/agent_output/curl_task/run/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status":"failed","attempts":[]}\n', encoding="utf-8")
    assert _benchmark_result(tmp_path, "run") == {
        "status": "failed",
        "outcome": "failed",
        "upstream_status": "failed",
        "attempts": [],
    }

    smoke = tmp_path / "sandbox/smoke-result.json"
    smoke.write_text('{"stage4":"passed","raw_exit_code":1}\n', encoding="utf-8")
    assert _benchmark_result(tmp_path, "smoke")["status"] == "passed"


def test_benchmark_result_rejects_interrupted_codex_failure_but_keeps_validated_success(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sandbox/agent_output/curl_task/run"
    summary = run_dir / "summary.json"
    trajectory = run_dir / "trajectory/attempt_1.log"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps({"type": "turn.failed", "error": {"message": "provider unavailable"}}) + "\n",
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps({"status": "failed", "agent": "codex", "attempts": []}) + "\n",
        encoding="utf-8",
    )
    interrupted = _benchmark_result(tmp_path, "run")
    assert interrupted["status"] == "error"
    assert interrupted["agent_turn"]["status"] == "failed"

    summary.write_text(
        json.dumps(
            {
                "status": "success",
                "agent": "codex",
                "attempts": [
                    {
                        "attempt": 1,
                        "agent_success": True,
                        "gt_success": True,
                        "success": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    exact = _benchmark_result(tmp_path, "run")
    assert exact["status"] == "passed"
    assert exact["outcome"] == "exact_match"
    assert exact["upstream_status"] == "success"

    summary.write_text(
        json.dumps(
            {
                "status": "success",
                "agent": "codex",
                "attempts": [
                    {
                        "attempt": 1,
                        "agent_success": True,
                        "gt_success": False,
                        "success": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    other = _benchmark_result(tmp_path, "run")
    assert other["status"] == "passed"
    assert other["outcome"] == "valid_other_vulnerability"


class _Files:
    @staticmethod
    def read(_path: str) -> str:
        return "\n".join(
            [
                '{"disk_total_bytes":1000,"disk_used_bytes":400,"disk_free_bytes":600,'
                '"memory_total_bytes":800,"memory_available_bytes":500}',
                '{"disk_total_bytes":1000,"disk_used_bytes":700,"disk_free_bytes":300,'
                '"memory_total_bytes":800,"memory_available_bytes":200}',
            ]
        )


class _Sandbox:
    files = _Files()


def test_resource_summary_reports_worst_observation() -> None:
    assert _resource_summary(_Sandbox()) == {
        "sample_count": 2,
        "interval_seconds": 5,
        "disk_total_bytes": 1000,
        "peak_disk_used_bytes": 700,
        "minimum_disk_free_bytes": 300,
        "memory_total_bytes": 800,
        "minimum_memory_available_bytes": 200,
    }


class _SwapFiles:
    @staticmethod
    def read(_path: str) -> str:
        return "\n".join(
            [
                '{"disk_total_bytes":1000,"disk_used_bytes":400,"disk_free_bytes":600,'
                '"memory_total_bytes":800,"memory_available_bytes":500,'
                '"swap_total_bytes":400,"swap_free_bytes":350,"swap_used_bytes":50}',
                '{"disk_total_bytes":1000,"disk_used_bytes":700,"disk_free_bytes":300,'
                '"memory_total_bytes":800,"memory_available_bytes":200,'
                '"swap_total_bytes":400,"swap_free_bytes":100,"swap_used_bytes":300}',
            ]
        )


class _SwapSandbox:
    files = _SwapFiles()


def test_resource_summary_reports_swap_pressure() -> None:
    result = _resource_summary(_SwapSandbox())
    assert result["swap_total_bytes"] == 400
    assert result["peak_swap_used_bytes"] == 300
    assert result["minimum_swap_free_bytes"] == 100


def test_resource_monitor_stop_failure_is_recorded_without_raising() -> None:
    class Monitor:
        def kill(self) -> None:
            raise ConnectionError("monitor channel closed")

    result: dict = {}
    _stop_resource_monitor(Monitor(), result, stage="after_workload")

    assert result["monitor_errors"] == [
        {
            "stage": "after_workload",
            "type": "ConnectionError",
            "message": "monitor channel closed",
        }
    ]


def test_batch_accounts_for_every_submitted_future_after_fail_fast(
    monkeypatch, tmp_path: Path
) -> None:
    tasks = ["curl/one", "curl/two", "curl/three"]
    digest = "example.invalid/project@sha256:" + "a" * 64
    resolved = {
        task: type(
            "Resolved",
            (),
            {
                "task": task,
                "project": "curl",
                "task_id": task.rsplit("/", 1)[1],
                "build_image": digest,
                "runtime_image": digest,
                "repo_to_patch": "https://example.invalid/repo",
            },
        )()
        for task in tasks
    }
    manifest = object()
    monkeypatch.setattr("cybergym_e2b.cli._tasks", lambda _args: tasks)
    monkeypatch.setattr("cybergym_e2b.cli.load_image_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "cybergym_e2b.cli.resolve_task", lambda _upstream, task, **_kwargs: resolved[task]
    )
    monkeypatch.setattr("cybergym_e2b.cli.require_immutable_runtime_image", lambda _task: None)
    monkeypatch.setattr("cybergym_e2b.cli.TemplateManifest.load", lambda _path: manifest)
    monkeypatch.setattr(
        "cybergym_e2b.cli._experiment_identity", lambda *_args, **_kwargs: {"sha256": "x"}
    )

    def execute(_args, task: str, **_kwargs):
        if task == tasks[0]:
            raise RuntimeError("sandbox create failed")
        return {"task": task, "completed": True, "benchmark": {"status": "passed"}}

    monkeypatch.setattr("cybergym_e2b.cli._execute", execute)
    args = Namespace(
        concurrency=1,
        continue_on_error=False,
        reuse_completed=False,
        artifacts_dir=tmp_path,
        upstream=tmp_path,
        image_map=tmp_path / "images.json",
        manifest=tmp_path / "manifest.json",
        kind="run",
        network_policy=DEFAULT_NETWORK_POLICY,
        patch_file=DEFAULT_PATCH_FILE,
        remote_smoke=DEFAULT_REMOTE_SMOKE,
        remote_install_codex=DEFAULT_REMOTE_SMOKE,
        setup_timeout=1,
        evaluation_timeout=1,
        min_free_gb=1,
        ffmpeg_min_free_gb=1,
        swap_gb=0,
        egress="policy",
        retain=False,
        agent="codex",
        prompt_style="iterative",
        model="openai.gpt-5.4",
        max_attempts=1,
        agent_timeout=1,
        provider="bedrock",
        bedrock_region="us-west-2",
    )

    summary = _batch(args)

    assert len(summary["results"]) == summary["submitted"] == 3
    assert summary["infrastructure_completed"] + summary["infrastructure_failures"] == 3
    assert summary["infrastructure_failures"] >= 1


def test_batch_infrastructure_failure_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("E2B_API_KEY", "present")
    monkeypatch.setattr("cybergym_e2b.cli._verify_upstream", lambda _path: "commit")
    monkeypatch.setattr(
        "cybergym_e2b.cli._batch",
        lambda _args: {"infrastructure_failures": 1, "results": []},
    )

    assert main(["batch", "--upstream", str(tmp_path), "--artifacts-dir", str(tmp_path)]) == 1


def test_preflight_accepts_the_same_execution_options_as_run() -> None:
    preflight = _parser().parse_args(
        [
            "preflight",
            "--task",
            "curl/arvo_66012",
            "--provider",
            "fireworks",
            "--model",
            "accounts/fireworks/models/test",
            "--agent",
            "openhands",
            "--egress",
            "restricted",
        ]
    )

    options = _options(preflight)
    assert options.provider == "fireworks"
    assert options.model == "accounts/fireworks/models/test"
    assert options.agent == "openhands"
    assert options.egress == "restricted"


def test_resume_skips_graded_model_errors_but_retries_smoke_errors(tmp_path: Path) -> None:
    task_dir = tmp_path / "curl" / "arvo_66012" / "run"
    task_dir.mkdir(parents=True)
    result_path = task_dir / "result.json"

    result_path.write_text(
        json.dumps(
            {
                "completed": True,
                "experiment": {"sha256": "expected"},
                "benchmark": {
                    "status": "failed",
                    "attempts": [
                        {
                            "stage1": "error",
                            "stage2": "skipped",
                            "stage3": "skipped",
                            "stage4": "skipped",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    assert _already_completed(tmp_path, "curl/arvo_66012", "expected")
    assert not _already_completed(tmp_path, "curl/arvo_66012", "different")
    result_path.write_text(
        json.dumps(
            {
                "completed": True,
                "experiment": {"sha256": "expected"},
                "benchmark": {
                    "status": "failed",
                    "stages": {"stage4": "error"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert not _already_completed(tmp_path, "curl/arvo_66012", "expected")


def test_resume_retries_interrupted_codex_turn(tmp_path: Path) -> None:
    task_dir = tmp_path / "curl" / "arvo_66012" / "run"
    trajectory = task_dir / "sandbox/agent_output/task/run/trajectory/attempt_1.log"
    trajectory.parent.mkdir(parents=True)
    result_path = task_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "completed": True,
                "experiment": {"sha256": "expected"},
                "benchmark": {"status": "failed", "agent": "codex", "attempts": []},
            }
        ),
        encoding="utf-8",
    )
    trajectory.write_text(
        json.dumps(
            {
                "type": "turn.failed",
                "error": {"message": "rate limit exceeded"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not _already_completed(tmp_path, "curl/arvo_66012", "expected")

    trajectory.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _already_completed(tmp_path, "curl/arvo_66012", "expected")

    result_path.write_text(
        json.dumps(
            {
                "completed": True,
                "experiment": {"sha256": "expected"},
                "benchmark": {
                    "status": "passed",
                    "outcome": "exact_match",
                    "attempts": [
                        {
                            "stage1": "passed",
                            "stage2": "passed",
                            "stage3": "passed",
                            "stage4": "passed",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    assert _already_completed(tmp_path, "curl/arvo_66012", "expected")


def test_experiment_identity_changes_with_kind_and_model(tmp_path: Path) -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    manifest_path = tmp_path / "manifest.json"
    TemplateManifest(
        base=TemplateRef(
            name="base",
            tag="recipe-aaaaaaaaaaaaaaaa",
            template_id="template-base",
            build_id="build-base-1234",
            recipe_sha256="a" * 64,
            images=BASE_BUILDER_IMAGES,
        )
    ).write(manifest_path)
    common = {
        "upstream": UPSTREAM,
        "manifest_path": manifest_path,
        "network_policy_path": DEFAULT_NETWORK_POLICY,
    }
    run = _experiment_identity(resolved, kind="run", options=RunOptions(), **common)
    smoke = _experiment_identity(resolved, kind="smoke", options=RunOptions(), **common)
    other_model = _experiment_identity(
        resolved,
        kind="run",
        options=replace(RunOptions(), model="accounts/fireworks/models/another-model"),
        **common,
    )
    other_reasoning_effort = _experiment_identity(
        resolved,
        kind="run",
        options=replace(RunOptions(), reasoning_effort="xhigh"),
        **common,
    )
    assert len(run["sha256"]) == 64
    assert run["sha256"] != smoke["sha256"]
    assert run["sha256"] != other_model["sha256"]
    assert run["sha256"] != other_reasoning_effort["sha256"]


def test_fresh_sandbox_creation_uses_template_and_disables_auto_resume(monkeypatch) -> None:
    captured = {}
    expected = object()

    def create(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("cybergym_e2b.runtime.Sandbox.create", create)
    actual = _create_fresh_sandbox_from_template(
        "template-name:build-id-1234",
        timeout=60,
        metadata={"run_id": "run"},
        network={"allow_public_traffic": False},
    )
    assert actual is expected
    assert captured["template"] == "template-name:build-id-1234"
    assert captured["lifecycle"] == {"on_timeout": "kill", "auto_resume": False}


def test_shell_run_survives_transient_poll_disconnect(monkeypatch) -> None:
    class Commands:
        def __init__(self) -> None:
            self.polls = 0

        def run(self, command: str, *, timeout: int):
            assert timeout == 30
            if "nohup bash" in command:
                return type("Result", (), {"stdout": ""})()
            self.polls += 1
            if self.polls == 1:
                raise ConnectionError("transient TLS EOF")
            return type("Result", (), {"stdout": "7\n"})()

    sandbox = type("Sandbox", (), {"commands": Commands()})()
    monkeypatch.setattr("cybergym_e2b.runtime.time.sleep", lambda _seconds: None)
    assert _shell_run(sandbox, "do-work", timeout=60) == 7
    assert sandbox.commands.polls == 2
