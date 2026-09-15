"""Model provider registry.

A :class:`ModelProvider` turns operator options into an OpenAI-compatible endpoint (host plus
base URL) and names the environment variables that may hold its credential. The credential
itself never enters the sandbox: E2B's egress proxy injects it on requests to ``host`` only.

Fixed providers (Fireworks, Bedrock Mantle) are registered in :data:`PROVIDERS`. Any other
OpenAI-compatible endpoint works through the ``openai-compatible`` provider with
``--model-base-url`` and ``--model-key-env``, no code change required. To add a fixed provider,
add an entry to :data:`PROVIDERS` and its host to both packaged network policies.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from cybergym_e2b.agents import WireApi

_AWS_REGION = re.compile(r"[a-z]{2}(?:-gov)?-[a-z]+-\d")


@dataclass(frozen=True)
class EndpointRequest:
    """What a provider may use to build its endpoint."""

    wire_api: WireApi
    region: str
    base_url: str | None
    key_env: str | None


@dataclass(frozen=True)
class Endpoint:
    host: str
    base_url: str


@dataclass(frozen=True)
class ModelProvider:
    name: str
    endpoint: Callable[[EndpointRequest], Endpoint]
    # Environment variables that may hold the credential; the first one set wins. An empty tuple
    # means the operator names it with --model-key-env.
    key_env: tuple[str, ...]

    def credential(self, request: EndpointRequest) -> tuple[str | None, str]:
        """Return (secret or None, human-readable name of where it was looked for)."""
        names = self.key_env or ((request.key_env,) if request.key_env else ())
        if not names:
            raise ValueError(f"provider {self.name!r} needs --model-key-env")
        for name in names:
            value = os.environ.get(name)
            if value:
                return value, " or ".join(names)
        return None, " or ".join(names)


def _fireworks(_request: EndpointRequest) -> Endpoint:
    return Endpoint(host="api.fireworks.ai", base_url="https://api.fireworks.ai/inference/v1")


def _bedrock_mantle(request: EndpointRequest) -> Endpoint:
    if not _AWS_REGION.fullmatch(request.region):
        raise ValueError(f"invalid AWS region: {request.region!r}")
    host = f"bedrock-mantle.{request.region}.api.aws"
    # Bedrock serves OpenAI frontier models on an OpenAI-specific Responses route; other Mantle
    # models use the general OpenAI-compatible Chat Completions route.
    path = "openai/v1" if request.wire_api == "responses" else "v1"
    return Endpoint(host=host, base_url=f"https://{host}/{path}")


def _openai_compatible(request: EndpointRequest) -> Endpoint:
    if not request.base_url:
        raise ValueError("provider 'openai-compatible' needs --model-base-url")
    parts = urlsplit(request.base_url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"--model-base-url must be an https URL with a host: {request.base_url!r}")
    return Endpoint(host=parts.hostname, base_url=request.base_url.rstrip("/"))


PROVIDERS: dict[str, ModelProvider] = {
    provider.name: provider
    for provider in (
        ModelProvider(
            name="fireworks",
            endpoint=_fireworks,
            key_env=("FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"),
        ),
        ModelProvider(
            name="bedrock",
            endpoint=_bedrock_mantle,
            key_env=("AWS_MANTLE", "AWS_BEARER_TOKEN_BEDROCK"),
        ),
        ModelProvider(name="openai-compatible", endpoint=_openai_compatible, key_env=()),
    )
}


def provider(name: str) -> ModelProvider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unsupported provider {name!r}; known: {sorted(PROVIDERS)}") from None
