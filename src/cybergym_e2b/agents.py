"""Agent harness registry.

An :class:`AgentHarness` describes everything the adapter needs to know about one upstream
``--agent`` value: how it speaks to an OpenAI-compatible endpoint, whether it installs tooling
inside the task container at runtime, which extra scripts it needs in the code bundle, and how to
tell an interrupted model turn from a genuine failure. Everything else about running the agent is
upstream's business (``scripts/run_agent.py``).

To add an agent that upstream already supports: add an entry to :data:`AGENTS`. To add one it
does not, teach upstream first, then register it here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cybergym_e2b.config import DEFAULT_REMOTE_INSTALL_CODEX

WireApi = Literal["responses", "chat-completions"]


def _identity(model: str) -> str:
    return model


def _litellm_openai_model(model: str) -> str:
    # OpenHands dispatches through LiteLLM. The ``openai/`` prefix selects the OpenAI-compatible
    # transport and is stripped before the request leaves; without it a catalog ID such as
    # ``deepseek.v3.2`` is mistaken for LiteLLM's SigV4-native Bedrock transport.
    return model if model.startswith("openai/") else f"openai/{model}"


def codex_turn_state(destination: Path) -> dict[str, Any]:
    """Classify a Codex run from its JSONL trajectory: completed, failed, or missing turn."""
    logs = sorted((destination / "sandbox" / "agent_output").glob("*/*/trajectory/*.log"))
    completed_turns = failed_turns = 0
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
    status = "completed" if completed_turns else "failed" if failed_turns else "missing"
    result: dict[str, Any] = {
        "status": status,
        "completed_turns": completed_turns,
        "failed_turns": failed_turns,
    }
    if last_error:
        result["last_error"] = last_error
    return result


@dataclass(frozen=True)
class AgentHarness:
    """Adapter-side description of one upstream agent."""

    name: str
    wire_api: WireApi
    # True when upstream installs the agent CLI inside the task container during the run, which
    # an allowlist policy must permit via ``runtime_dependency_hosts``.
    installs_tooling_at_runtime: bool
    # Files copied into the bundle's ``scripts/`` directory, overriding upstream's copies.
    bundle_scripts: Mapping[str, Path] = field(default_factory=dict)
    # Maps the operator's model ID to the value passed as upstream ``--litellm-model-id``.
    model_id: Callable[[str], str] = _identity
    # Distinguishes an interrupted model turn from a graded failure; None means trust upstream.
    turn_state: Callable[[Path], dict[str, Any]] | None = None


AGENTS: dict[str, AgentHarness] = {
    harness.name: harness
    for harness in (
        AgentHarness(
            name="codex",
            wire_api="responses",
            installs_tooling_at_runtime=True,
            bundle_scripts={"install_codex.sh": DEFAULT_REMOTE_INSTALL_CODEX},
            turn_state=codex_turn_state,
        ),
        AgentHarness(
            name="openhands",
            wire_api="chat-completions",
            installs_tooling_at_runtime=True,
            model_id=_litellm_openai_model,
        ),
    )
}


def agent(name: str) -> AgentHarness:
    try:
        return AGENTS[name]
    except KeyError:
        raise ValueError(f"unsupported agent {name!r}; known: {sorted(AGENTS)}") from None
