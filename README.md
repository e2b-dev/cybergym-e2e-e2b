# CyberGym-E2E on E2B

`cybergym-e2b` runs [CyberGym-E2E](https://github.com/sunblaze-ucb/cybergym-e2e) tasks on
[E2B](https://e2b.dev) sandboxes: one fresh Docker-in-Docker sandbox per task, running the
upstream harness (`run_agent.py`, the S1–S4 validators) unchanged apart from a small
compatibility patch. It covers the source-only end-to-end mode. The classic binary-only CyberGym
benchmark is a different project.

Supported agents are `codex` and `openhands`, speaking to an OpenAI-compatible endpoint on
Fireworks or Amazon Bedrock (Mantle). See [Deviations from upstream](#deviations-from-upstream)
for what this adapter does not reproduce.

## Prerequisites

- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker with the Buildx plugin (used only to resolve image tags to digests; nothing is pulled
  locally)
- An E2B account with template builds and enough sandbox concurrency for your batch size
- Accepted access to the gated
  [sunblaze-ucb/cybergym-e2e](https://huggingface.co/datasets/sunblaze-ucb/cybergym-e2e) dataset
- Credentials in a local `.env` (copy `.env.example`): `E2B_API_KEY`, `HF_TOKEN`, and either
  `AWS_MANTLE` (Bedrock) or `FIREWORKS_AI_API_KEY`

## Quickstart

```bash
uv sync --locked
uv run cybergym-e2b sync-upstream                       # pinned upstream code under vendor/
uv run cybergym-e2b images lock --task curl/arvo_66012  # pin the task's image to a digest

uv run cybergym-e2b templates build-base                # Docker-in-Docker base template
uv run cybergym-e2b templates build-ffmpeg              # FFmpeg image + Opus model cache

uv run cybergym-e2b preflight --task curl/arvo_66012 --provider bedrock --model openai.gpt-5.4
uv run cybergym-e2b smoke curl/arvo_66012               # infra check: ground-truth S4 only
uv run cybergym-e2b run curl/arvo_66012 --agent codex --provider bedrock --model openai.gpt-5.4
```

Every command prints one JSON document. Results land under
`artifacts/e2e/<project>/<task>/<run-id>/`: `result.json` (orchestration, timings, resource
samples, benchmark verdict) and `sandbox/` (the upstream `agent_output` tree, run log, exit code).

Run many tasks with bounded concurrency. The task file is one `project/task` per line; the
upstream `scripts/tasks.txt` lists all 920:

```bash
uv run cybergym-e2b batch --kind run --tasks-file ./my-tasks.txt --concurrency 4 \
  --agent codex --provider bedrock --model openai.gpt-5.4
```

`batch` skips tasks that already have a completed result for the exact same experiment
fingerprint (task, agent, model, template build, image digest, upstream revisions, patch, policy,
options). A completed run whose agent finished a turn and failed the benchmark counts as done.
Infrastructure errors and interrupted agent turns are retried. Pass `--no-reuse-completed` to
rerun everything.

### Image locks and Docker Hub

Upstream references project images by mutable tag. The adapter refuses to run a task until its
image is pinned to a digest in `artifacts/images.lock.json`. `images lock --task` (repeatable)
resolves only what you need and merges into the existing lock. `images lock` with no task
resolves all 506 tags, which exceeds Docker Hub's anonymous manifest budget (about 100 requests
per hour per address); use it only with an authenticated `docker login`. The lock records the
resolver tool and version, and the runner verifies the pulled image's digest inside the sandbox.

## How a run works

1. Resolve the task's project and task config from the pinned upstream checkout; select the base
   template, or the FFmpeg template for any `ffmpeg/*` task.
2. Verify the template tag still points at the build recorded in `artifacts/templates/manifest.json`.
3. Create a fresh sandbox (8 vCPU, 8 GB RAM by default, kill on timeout, no auto-resume) with the
   setup-phase network policy. `HF_TOKEN` is injected by E2B's egress proxy on requests to
   `huggingface.co` only; the sandbox never holds it.
4. Set `vm.mmap_rnd_bits=28`, enable 4 GB of swap, check free disk and Docker health.
5. Upload the upstream scripts (patched), the one project/task directory, and the adapter's
   helper scripts. Download only that task's data files from Hugging Face. Pull and digest-verify
   the project image.
6. Switch to the runtime network policy: the Hugging Face rule is removed and the model provider
   credential is injected on the provider host only. Extend the sandbox timeout.
7. Run upstream `run_agent.py` (or the S4 smoke) in the background and poll for its exit code.
8. Collect `agent_output`, logs, and resource samples; normalize the verdict; kill the sandbox
   (`--retain` pauses it instead for debugging).

`result.json` separates `completed` (did the adapter finish and collect artifacts) from
`benchmark.status` (`passed`, `failed`, or `error`) and `benchmark.outcome`
(`exact_match`, `valid_other_vulnerability`, `failed`, `incomplete_agent_turn`, ...).

## Network policy and result eligibility

The packaged default policy (`--network-policy`, `assets/policies/network.json`) mirrors upstream:
the agent container has public internet, minus private, loopback, link-local, carrier-grade NAT,
and multicast IPv4 ranges. CyberGym's prompt tells agents that network use invalidates the
result, so every default-policy result is marked `requires_network_audit`; inspect the trajectory
before publishing it.

- `--egress restricted` replaces public egress with an allowlist of the policy's model, dependency,
  and registry hosts. Results still require an audit.
- `--egress permissive` removes the adapter's filters entirely, for diagnostics; never eligible.
- `assets/policies/network-locked.json` allows only the model host at runtime and is eligible
  without an audit. Because this release installs Node and the agent CLI at runtime, `run` and
  `batch --kind run` refuse it; `smoke` works under it.

`preflight` reports the eligibility classification for the chosen policy and egress mode.

## Deviations from upstream

The adapter aims to change nothing that affects grading. These are the intentional differences;
each is either recorded in the experiment fingerprint or documented here.

| Area | Upstream | This adapter |
|---|---|---|
| Agents | `claude-code` (default), `codex`, `openhands`, `gemini-cli` | `codex`, `openhands` only |
| Model access | Anthropic API, Bedrock (SigV4), or a LiteLLM proxy | Direct OpenAI-compatible endpoint (Fireworks, Bedrock Mantle); no LiteLLM |
| Codex reasoning | Codex default | Explicit `--reasoning-effort high` (configurable), `wire_api = "responses"` |
| Machine shape | Whatever host runs `run_agent.py` | 8 vCPU, 8 GB RAM, 4 GB swap, shared disk tier; `--cpu-count`/`--memory-mb` at template build |
| Project images | Mutable tags | Digest-pinned via the image lock |
| Dataset | Full `hf download` | Per-task `src.tgz`, `poc.bin`, `crash.log` |
| Agent container setup | `apt-get update && apt-get install sudo git` | `apt`/`apt-get` retry wrapper, a `sudo` shim if absent, dead third-party apt sources disabled before `apt-get update`, hardened Codex installer |
| Trajectory summary | LLM summary after every failed attempt | Skipped on the final attempt, so that attempt's `feedback_attempt_N.txt` lacks the summary; it only feeds the next attempt |
| FFmpeg tasks | Fetch the Opus model archive from `media.xiph.org` at prepare time | Checksum-pinned archive baked into the FFmpeg template |
| Validation dependencies | `apt-get install curl` unconditionally | Skipped when `curl` is already present |

The `apt`/`sudo`/apt-source changes exist because many project images pin apt repositories that
no longer publish a Release file, which aborts upstream's setup before the agent starts. They
alter what an agent can install inside its container (a removed Kitware or LLVM source is gone
for the agent too). Treat results on tasks where the agent depended on such a source with care.

## Pinned inputs

`src/cybergym_e2b/assets/upstream.lock.json` pins the upstream code commit and dataset revision;
all runtime constants derive from it. Template builds pin the Ubuntu base image and OSS-Fuzz
builder images by digest, the Ubuntu package archive at a dated snapshot, exact Docker package
versions and signing-key checksum, and a hash-locked Python environment
(`assets/template-requirements.lock`). Template tags are content-addressed
(`cybergym-e2e-dind:recipe-<16 hex>`), and the runner refuses a tag whose E2B build ID no longer
matches the manifest. Losing `artifacts/templates/build-ledger.json` causes a rebuild under the
same tag, never reuse of an unverified one.

Runtime assets can be overridden with `--patch-file`, `--remote-smoke`, `--remote-install-codex`,
and `--network-policy`; overrides change the experiment fingerprint.

## Known limitations

- No Claude Code or Anthropic API path, so upstream's default configuration cannot be reproduced.
- No plain OpenAI provider; only Fireworks and Bedrock Mantle hosts are declared in the policies.
- The full 506-image lock needs an authenticated Docker Hub session.
- 8 GB RAM plus 4 GB swap has been validated on a sample of tasks, not the whole inventory.
  Watch `observed_resources.minimum_memory_available_bytes` and swap usage in `result.json`.
- E2B's template build API does not take a disk size; the recorded `--disk-limit-gb` is the
  expected account tier, and the runtime free-space check is what actually protects a run.
- Remaining engineering follow-ups are tracked in [FOLLOW-UPS.md](FOLLOW-UPS.md).

## Development

```bash
uv sync --locked
uv run cybergym-e2b sync-upstream     # several tests read the vendored checkout
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/check_public_tree.py
uv build --no-build-isolation --clear && uv run twine check dist/*
```

CI runs the suite on Python 3.12 and 3.13, installs the wheel outside the checkout, and reruns
the tests from the extracted source distribution. Security reports follow
[SECURITY.md](SECURITY.md). Licensed under Apache-2.0.
