from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from cybergym_e2b.agents import AGENTS, agent, codex_turn_state
from cybergym_e2b.cli import _parser
from cybergym_e2b.config import (
    BASE_BUILDER_IMAGES,
    DEFAULT_NETWORK_POLICY,
    TemplateManifest,
    TemplateRef,
)
from cybergym_e2b.inventory import build_code_bundle, resolve_task
from cybergym_e2b.providers import PROVIDERS, EndpointRequest, provider
from cybergym_e2b.runtime import RunOptions, _experiment_identity, _model_config

UPSTREAM = Path("vendor/cybergym-e2e")


def _request(**overrides) -> EndpointRequest:
    base = {"wire_api": "responses", "region": "us-west-2", "base_url": None, "key_env": None}
    return EndpointRequest(**{**base, **overrides})


def test_cli_choices_come_from_the_registries() -> None:
    parser = _parser()
    commands = next(a for a in parser._subparsers._actions if a.dest == "command")
    run_actions = commands.choices["run"]._actions
    assert list(next(a for a in run_actions if a.dest == "agent").choices) == sorted(AGENTS)
    assert list(next(a for a in run_actions if a.dest == "provider").choices) == sorted(PROVIDERS)


def test_unknown_agent_or_provider_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported agent 'claude-code'"):
        agent("claude-code")
    with pytest.raises(ValueError, match="unsupported provider 'anthropic'"):
        provider("anthropic")


def test_openai_compatible_provider_takes_any_https_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("MY_VLLM_KEY", "secret")
    options = RunOptions(
        provider="openai-compatible",
        model_base_url="https://llm.example.internal:8443/v1/",
        model_key_env="MY_VLLM_KEY",
    )
    assert _model_config(options) == {
        "host": "llm.example.internal",
        "base_url": "https://llm.example.internal:8443/v1",
        "key": "secret",
        "key_name": "MY_VLLM_KEY",
    }

    with pytest.raises(ValueError, match="--model-base-url"):
        _model_config(RunOptions(provider="openai-compatible", model_key_env="MY_VLLM_KEY"))
    with pytest.raises(ValueError, match="https URL"):
        _model_config(
            RunOptions(
                provider="openai-compatible",
                model_base_url="http://llm.example.internal/v1",
                model_key_env="MY_VLLM_KEY",
            )
        )
    with pytest.raises(ValueError, match="--model-key-env"):
        _model_config(
            RunOptions(provider="openai-compatible", model_base_url="https://x.example/v1")
        )


def test_fixed_providers_resolve_credentials_from_named_env(monkeypatch) -> None:
    monkeypatch.delenv("FIREWORKS_AI_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fallback")
    assert PROVIDERS["fireworks"].credential(_request()) == (
        "fallback",
        "FIREWORKS_AI_API_KEY or FIREWORKS_API_KEY",
    )
    monkeypatch.delenv("FIREWORKS_API_KEY")
    assert PROVIDERS["fireworks"].credential(_request())[0] is None
    with pytest.raises(ValueError, match="invalid AWS region"):
        PROVIDERS["bedrock"].endpoint(_request(region="mars-1"))


def _installer(bundle: bytes) -> str:
    with tarfile.open(fileobj=BytesIO(bundle), mode="r:gz") as tar:
        member = tar.extractfile("scripts/install_codex.sh")
        assert member is not None
        return member.read().decode()


def test_agent_bundle_scripts_are_shipped_only_for_that_agent() -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    # Without an agent's scripts the bundle carries upstream's installer untouched.
    assert "command -v curl" not in _installer(build_code_bundle(UPSTREAM, resolved))
    hardened = build_code_bundle(UPSTREAM, resolved, scripts=AGENTS["codex"].bundle_scripts)
    assert "command -v curl" in _installer(hardened)
    with pytest.raises(ValueError, match="bare filename"):
        build_code_bundle(UPSTREAM, resolved, scripts={"../escape.sh": Path(__file__)})


def test_experiment_identity_covers_agent_scripts_and_overrides(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    TemplateManifest(
        base=TemplateRef(
            name="base",
            tag="recipe-aaaaaaaaaaaaaaaa",
            template_id="template-base",
            build_id="build-base-1234",
            recipe_sha256="a" * 64,
            images=BASE_BUILDER_IMAGES,
        )
    ).write(manifest)
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    common = {
        "upstream": UPSTREAM,
        "manifest_path": manifest,
        "network_policy_path": DEFAULT_NETWORK_POLICY,
    }
    codex = _experiment_identity(resolved, kind="run", options=RunOptions(), **common)
    assert list(codex["inputs"]["source"]["bundle_scripts_sha256"]) == ["install_codex.sh"]

    smoke = _experiment_identity(resolved, kind="smoke", options=RunOptions(), **common)
    assert smoke["inputs"]["source"]["bundle_scripts_sha256"] == {}

    override = tmp_path / "install_codex.sh"
    override.write_text("#!/bin/sh\n")
    overridden = _experiment_identity(
        resolved,
        kind="run",
        options=RunOptions(),
        bundle_script_overrides={"install_codex.sh": override},
        **common,
    )
    assert overridden["sha256"] != codex["sha256"]


def test_codex_turn_state_reads_trajectory_events(tmp_path: Path) -> None:
    log = tmp_path / "sandbox/agent_output/task/run/trajectory/attempt_1.log"
    log.parent.mkdir(parents=True)
    log.write_text('{"type":"turn.failed","error":{"message":"429"}}\nnot json\n')
    assert codex_turn_state(tmp_path) == {
        "status": "failed",
        "completed_turns": 0,
        "failed_turns": 1,
        "last_error": "429",
    }
    assert AGENTS["openhands"].turn_state is None
