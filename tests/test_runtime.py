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

import pytest

from cybergym_e2b.cli import _already_completed, _batch, _verify_upstream, main
from cybergym_e2b.config import (
    BASE_BUILDER_IMAGES,
    DEFAULT_NETWORK_POLICY,
    DEFAULT_PATCH_FILE,
    DEFAULT_REMOTE_APT_RETRY,
    DEFAULT_REMOTE_INSTALL_CODEX,
    DEFAULT_REMOTE_SMOKE,
    FFMPEG_IMAGE,
    FFMPEG_IMAGE_DIGEST,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    TemplateManifest,
    TemplateRef,
    asset_path,
    normalize_task,
)
from cybergym_e2b.inventory import build_code_bundle, inventory, load_image_map, resolve_task
from cybergym_e2b.runtime import (
    COLLECTION_RESERVE_SECONDS,
    RunOptions,
    _agent_model_id,
    _assert_image_identity,
    _benchmark_result,
    _execution_context,
    _experiment_identity,
    _extract_results,
    _model_config,
    _network,
    _network_eligibility,
    _observe_after_workload,
    _policy,
    _require_runnable_policy,
    _resource_summary,
    _route_template,
    _shell_run,
    _StageProfiler,
)

UPSTREAM = Path("vendor/cybergym-e2e")


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
        model_host="api.fireworks.ai",
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
        model_host="api.fireworks.ai",
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
        model_host="api.fireworks.ai",
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
        model_host="api.fireworks.ai",
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


def test_bedrock_routes_codex_to_responses_and_openhands_to_chat_completions() -> None:
    codex = RunOptions(provider="bedrock", model="openai.gpt-5.4")
    assert _model_config(codex)["base_url"] == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
    assert _agent_model_id(codex) == "openai.gpt-5.4"

    openhands = replace(codex, agent="openhands", model="deepseek.v3.2")
    assert _model_config(openhands)["base_url"] == "https://bedrock-mantle.us-west-2.api.aws/v1"
    assert _agent_model_id(openhands) == "openai/deepseek.v3.2"


def test_network_policy_rejects_unsupported_deny_cidrs(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_NETWORK_POLICY.read_text(encoding="utf-8"))
    raw["deny_out"] = ["0.0.0.0/8"]
    path = tmp_path / "network.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="not supported by E2B"):
        _policy(path)

    raw["deny_out"] = ["240.0.0.0/4"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not supported by E2B"):
        _policy(path)


def test_pinned_inventory_shape() -> None:
    result = inventory(UPSTREAM)
    assert result["tasks"] == 920
    assert result["projects"] == 139
    assert result["unique_build_images"] == 509
    assert result["single_use_build_images"] == 487
    counts = {row["build_image"]: row["task_count"] for row in result["images"]}
    assert sum(counts[image] for image in BASE_BUILDER_IMAGES) == 344
    assert counts[FFMPEG_IMAGE] == 10


def _lock(images: dict[str, str]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
            "resolver": {"tool": "test", "version": "1", "method": "manifest-descriptor-digest"},
            "images": images,
        }
    )


def test_image_lock_accepts_only_digest_locked_values(tmp_path: Path) -> None:
    unmapped = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305")
    assert (unmapped.build_image, unmapped.runtime_image) == (FFMPEG_IMAGE, FFMPEG_IMAGE_DIGEST)

    digest = FFMPEG_IMAGE_DIGEST.rsplit(":", 1)[1]
    path = tmp_path / "images.lock.json"
    path.write_text(_lock({FFMPEG_IMAGE: f"mirror.invalid/cybergym/e2e@sha256:{digest}"}))
    image_map = load_image_map(path, upstream=UPSTREAM)
    mapped = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305", image_map=image_map)
    assert mapped.build_image == FFMPEG_IMAGE
    assert mapped.runtime_image == f"mirror.invalid/cybergym/e2e@sha256:{digest}"

    path.write_text(_lock({FFMPEG_IMAGE: "mirror.invalid/cybergym/e2e:ffmpeg"}))
    with pytest.raises(ValueError, match="digest-locked"):
        load_image_map(path, upstream=UPSTREAM)

    path.write_text(_lock({FFMPEG_IMAGE: "mirror.invalid/cybergym/e2e@sha256:" + "b" * 64}))
    with pytest.raises(ValueError, match="known FFmpeg digest"):
        load_image_map(path, upstream=UPSTREAM)

    path.write_text(json.dumps({FFMPEG_IMAGE: f"mirror.invalid/cybergym/e2e@sha256:{digest}"}))
    with pytest.raises(ValueError, match="schema version 1"):
        load_image_map(path, upstream=UPSTREAM)

    with pytest.raises(FileNotFoundError, match="images lock --task"):
        load_image_map(tmp_path / "absent.json", upstream=UPSTREAM)


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


def test_manifest_round_trip_and_ffmpeg_routing(tmp_path: Path) -> None:
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
    path = tmp_path / "manifest.json"
    TemplateManifest(base=base, hot={FFMPEG_IMAGE: hot}).write(path)
    manifest = TemplateManifest.load(path)

    assert manifest.route(FFMPEG_IMAGE).reference == hot.reference
    assert manifest.route("n132/arvo:1-fix").reference == base.reference
    # Every ffmpeg task needs the cached Opus archive, even with a task-specific legacy image.
    legacy = "cybergym/e2e:ffmpeg-legacy-task-image"
    assert _route_template(manifest, project="ffmpeg", build_image=legacy) == hot
    assert _route_template(manifest, project="curl", build_image=legacy) == base


def test_task_path_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        normalize_task("curl/../secret")


def test_ffmpeg_digest_is_enforced_even_through_a_mirror() -> None:
    digest = FFMPEG_IMAGE_DIGEST.split("@", 1)[1]
    _assert_image_identity(
        FFMPEG_IMAGE_DIGEST,
        {"repo_digests": [f"mirror.invalid/cybergym/e2e@{digest}"]},
    )
    with pytest.raises(RuntimeError):
        _assert_image_identity(
            FFMPEG_IMAGE_DIGEST, {"repo_digests": ["mirror.invalid/cybergym/e2e@sha256:wrong"]}
        )


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


def _sample(**overrides) -> dict:
    base = {
        "timestamp": 0,
        "disk_total_bytes": 1000,
        "disk_used_bytes": 400,
        "disk_free_bytes": 600,
        "memory_total_bytes": 800,
        "memory_available_bytes": 500,
        "swap_total_bytes": 400,
        "swap_free_bytes": 350,
        "swap_used_bytes": 50,
        "page_faults": 10,
        "major_page_faults": 1,
        "swap_in_pages": 0,
        "swap_out_pages": 0,
        "load_1m": 0.5,
        "load_5m": 0.4,
        "load_15m": 0.3,
        "cpu_total_ticks": 1000,
        "cpu_idle_ticks": 900,
        "network_receive_bytes": 100,
        "network_transmit_bytes": 50,
        "block_read_bytes": 0,
        "block_write_bytes": 0,
        "block_io_milliseconds": 0,
    }
    return {**base, **overrides}


class _SampleSandbox:
    def __init__(self, lines: list[str]) -> None:
        self.files = SimpleNamespace(read=lambda _path: "\n".join(lines))


def test_resource_summary_reports_worst_observation_and_skips_truncated_lines() -> None:
    first = _sample()
    last = _sample(
        disk_used_bytes=700,
        disk_free_bytes=300,
        memory_available_bytes=200,
        swap_free_bytes=100,
        swap_used_bytes=300,
        load_1m=2.5,
        cpu_total_ticks=1400,
        cpu_idle_ticks=1000,
        network_receive_bytes=600,
    )
    sandbox = _SampleSandbox([json.dumps(first), json.dumps(last), '{"disk_total_bytes": 10'])

    summary = _resource_summary(sandbox)

    assert summary["sample_count"] == 2
    assert summary["peak_disk_used_bytes"] == 700
    assert summary["minimum_disk_free_bytes"] == 300
    assert summary["minimum_memory_available_bytes"] == 200
    assert summary["peak_swap_used_bytes"] == 300
    assert summary["minimum_swap_free_bytes"] == 100
    assert summary["peak_load_1m"] == 2.5
    assert summary["delta_network_receive_bytes"] == 500
    assert summary["average_cpu_busy_percent"] == 75.0


def test_observe_after_workload_records_errors_without_raising() -> None:
    class Monitor:
        def kill(self) -> None:
            raise ConnectionError("monitor channel closed")

    class Commands:
        def run(self, _command: str, *, timeout: int):
            raise TimeoutError("metrics command hung")

    sandbox = SimpleNamespace(
        files=SimpleNamespace(read=lambda _path: json.dumps(_sample())),
        commands=Commands(),
    )
    result: dict = {}

    _observe_after_workload(sandbox, result, _StageProfiler(), Monitor())

    assert result["observed_resources"]["sample_count"] == 1
    assert "post_workload" not in result
    assert [(item["step"], item["type"]) for item in result["observation_errors"]] == [
        ("monitor_stop", "ConnectionError"),
        ("post_workload_metrics", "TimeoutError"),
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
        image_lock=tmp_path / "images.json",
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


def test_shell_run_fails_fast_when_sandbox_stops_answering(monkeypatch) -> None:
    class Commands:
        def __init__(self) -> None:
            self.polls = 0

        def run(self, command: str, *, timeout: int):
            if "nohup bash" in command:
                return SimpleNamespace(stdout="")
            self.polls += 1
            raise ConnectionError("sandbox is gone")

    sandbox = SimpleNamespace(commands=Commands())
    monkeypatch.setattr("cybergym_e2b.runtime.time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="stopped answering"):
        _shell_run(sandbox, "do-work", timeout=10_000)
    assert sandbox.commands.polls == 12


def _write_manifest(path: Path) -> None:
    TemplateManifest(
        base=TemplateRef(
            name="base",
            tag="recipe-aaaaaaaaaaaaaaaa",
            template_id="template-base",
            build_id="build-base-1234",
            recipe_sha256="a" * 64,
            images=BASE_BUILDER_IMAGES,
        )
    ).write(path)


def test_execution_context_reserves_time_for_collection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    resolved = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305")
    with pytest.raises(ValueError, match="evaluation_timeout"):
        _execution_context(
            resolved,
            kind="smoke",
            options=RunOptions(evaluation_timeout=COLLECTION_RESERVE_SECONDS),
            manifest_path=manifest,
            network_policy_path=DEFAULT_NETWORK_POLICY,
        )


def test_execution_context_resolves_non_http_hosts_only_for_allowlists(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    resolved = resolve_task(UPSTREAM, "ffmpeg/oss-fuzz_431665305")
    lookups: list[str] = []

    def getaddrinfo(host, port, **_kwargs):
        lookups.append(host)
        return [(None, None, None, None, ("203.0.113.7", port))]

    monkeypatch.setattr("cybergym_e2b.runtime.socket.getaddrinfo", getaddrinfo)

    public = _execution_context(
        resolved,
        kind="smoke",
        options=RunOptions(),
        manifest_path=manifest,
        network_policy_path=DEFAULT_NETWORK_POLICY,
    )
    assert public.non_http == []
    assert lookups == []

    restricted = _execution_context(
        resolved,
        kind="smoke",
        options=RunOptions(egress="restricted"),
        manifest_path=manifest,
        network_policy_path=DEFAULT_NETWORK_POLICY,
    )
    assert sorted(lookups) == ["fate-suite.ffmpeg.org", "samples.ffmpeg.org"]
    assert all(entry["addresses"] == ["203.0.113.7"] for entry in restricted.non_http)

    def failing(host, port, **_kwargs):
        raise OSError("name resolution failed")

    monkeypatch.setattr("cybergym_e2b.runtime.socket.getaddrinfo", failing)
    with pytest.raises(RuntimeError, match="fate-suite.ffmpeg.org"):
        _execution_context(
            resolved,
            kind="smoke",
            options=RunOptions(egress="restricted"),
            manifest_path=manifest,
            network_policy_path=DEFAULT_NETWORK_POLICY,
        )


LOCKED_POLICY = asset_path("policies/network-locked.json")


def test_patch_keeps_anthropic_return_ahead_of_openai_compatible_branch() -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    payload = build_code_bundle(UPSTREAM, resolved)
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        utils = archive.extractfile("scripts/utils.py")
        assert utils is not None
        source = utils.read().decode()
    anthropic_return = source.index("return response.content[0].text")
    new_branch = source.index('model_provider == "openai-compatible"')
    assert anthropic_return < new_branch


def test_patch_carries_context_and_applies_without_unidiff_zero() -> None:
    check = subprocess.run(
        ["git", "apply", "--check", str(DEFAULT_PATCH_FILE.resolve())],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def _git_repo_with_commit(root: Path) -> str:
    (root / "scripts").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "scripts" / "validate.py").write_text("pinned\n")
    subprocess.run(["git", "add", "scripts/validate.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "pin"],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_verify_upstream_rejects_dirty_checkout(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "upstream"
    head = _git_repo_with_commit(repo)
    monkeypatch.setattr("cybergym_e2b.cli.UPSTREAM_COMMIT", head)

    assert _verify_upstream(repo) == head

    # IDE metadata or notes outside the shipped trees must not block every command.
    (repo / "notes.md").write_text("scratch\n")
    assert _verify_upstream(repo) == head

    (repo / "scripts" / "validate.py").write_text("edited validator\n")
    with pytest.raises(RuntimeError, match="dirty") as excinfo:
        _verify_upstream(repo)
    # sync-upstream refuses dirty trees too, so the message must name a remedy that works,
    # and every git step in it must target the vendored checkout, not the caller's repo.
    assert f"git -C {repo} checkout -- ." in str(excinfo.value)
    assert f"git -C {repo} clean -fd" in str(excinfo.value)

    (repo / "scripts" / "validate.py").write_text("pinned\n")
    (repo / "scripts" / "extra.py").write_text("untracked but shipped\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _verify_upstream(repo)


def _tarball(members: dict[str, bytes | Path | tuple[str, str]]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            if isinstance(data, Path):
                info.type = tarfile.SYMTYPE
                info.linkname = str(data)
                archive.addfile(info)
            elif isinstance(data, tuple):
                info.type = tarfile.LNKTYPE
                info.linkname = data[1]
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def test_extract_results_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"../escape.txt": b"x"}))
    with pytest.raises(RuntimeError, match="unsafe path"):
        _extract_results(archive_path, tmp_path / "sandbox")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_results_extracts_nested_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"./agent_output/run/log.txt": b"ok"}))
    _extract_results(archive_path, tmp_path / "sandbox")
    assert (tmp_path / "sandbox" / "agent_output" / "run" / "log.txt").read_bytes() == b"ok"


def test_extract_results_drops_unsafe_links_but_keeps_in_tree_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(
        _tarball(
            {
                "./agent_output/fix.patch": Path("/etc/passwd"),
                "./agent_output/summary.json": b"{}",
                # GNU cp -a followed by tar emits the second hardlink name as LNKTYPE.
                "./agent_output/summary-copy.json": ("hardlink", "./agent_output/summary.json"),
            }
        )
    )
    _extract_results(archive_path, tmp_path / "sandbox")
    out = tmp_path / "sandbox" / "agent_output"
    assert (out / "summary.json").read_bytes() == b"{}"
    assert (out / "summary-copy.json").read_bytes() == b"{}"
    assert not (out / "fix.patch").is_symlink()
    assert not (out / "fix.patch").exists()


@pytest.mark.parametrize("policy_path", [DEFAULT_NETWORK_POLICY, LOCKED_POLICY])
def test_packaged_policies_accept_any_bedrock_region(policy_path: Path) -> None:
    policy = _policy(policy_path)
    host = "bedrock-mantle.eu-west-1.api.aws"
    network = _network(
        policy,
        phase="runtime",
        egress="restricted",
        hf_token=None,
        model_key="secret",
        non_http=[],
        model_host=host,
    )
    assert host in network["allow_out"]
    assert set(network["rules"]) == {host}


def test_packaged_policies_share_host_lists() -> None:
    default = _policy(DEFAULT_NETWORK_POLICY)
    locked = _policy(LOCKED_POLICY)
    for key in ("model_hosts", "artifact_hosts", "registry_hosts"):
        assert default[key] == locked[key], key


def test_locked_policy_refuses_agent_runs_but_allows_smoke() -> None:
    locked = _policy(LOCKED_POLICY)
    with pytest.raises(ValueError, match="agent tooling"):
        _require_runnable_policy(locked, kind="run", egress="policy")
    _require_runnable_policy(locked, kind="smoke", egress="policy")
    default = _policy(DEFAULT_NETWORK_POLICY)
    _require_runnable_policy(default, kind="run", egress="policy")
    _require_runnable_policy(default, kind="run", egress="restricted")


def test_cli_validates_asset_overrides_before_doing_work(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "instal_codex.sh"
    code = main(["preflight", "--remote-install-codex", str(missing)])
    assert code == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["type"] == "FileNotFoundError"
    assert "instal_codex.sh" in error["message"]


def test_batch_refuses_unrunnable_policy_before_submitting_work(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    # main() requires the E2B key before dispatching to batch; keep the test hermetic.
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("curl/arvo_66012\n")
    code = main(
        [
            "batch",
            "--kind",
            "run",
            "--tasks-file",
            str(tasks),
            "--network-policy",
            str(LOCKED_POLICY),
        ]
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["type"] == "ValueError"
    assert "agent tooling" in error["message"]
