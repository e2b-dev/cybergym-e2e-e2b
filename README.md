# CyberGym-E2E on E2B

`cybergym-e2b` runs [CyberGym-E2E](https://github.com/sunblaze-ucb/cybergym-e2e) on
[E2B](https://e2b.dev): one fresh Docker-in-Docker sandbox per task, running the upstream harness
and S1–S4 validators with a small compatibility patch. Source-only end-to-end mode only; the
classic binary-only CyberGym is a different project.

Agents: `codex`, `openhands`. Providers: Fireworks or Amazon Bedrock (Mantle), via their
OpenAI-compatible endpoints. See [Deviations from upstream](#deviations-from-upstream).

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/); Docker with Buildx (only to resolve image
  tags to digests)
- An E2B account with template builds and enough sandbox concurrency for your batch size
- Access to the gated [dataset](https://huggingface.co/datasets/sunblaze-ucb/cybergym-e2e)
- A `.env` (copy `.env.example`) with `E2B_API_KEY`, `HF_TOKEN`, and `AWS_MANTLE` or
  `FIREWORKS_AI_API_KEY`

## Quickstart

```bash
uv sync --locked
uv run cybergym-e2b sync-upstream                       # pinned upstream code -> vendor/
uv run cybergym-e2b images lock --task curl/arvo_66012  # pin the task's image to a digest
uv run cybergym-e2b templates build-base                # Docker-in-Docker base template
uv run cybergym-e2b templates build-ffmpeg              # FFmpeg image + Opus model cache
uv run cybergym-e2b preflight --task curl/arvo_66012 --provider bedrock --model openai.gpt-5.4
uv run cybergym-e2b smoke curl/arvo_66012               # infra only: ground-truth S4
uv run cybergym-e2b run curl/arvo_66012 --agent codex --provider bedrock --model openai.gpt-5.4
```

Batch (task file: one `project/task` per line; upstream `scripts/tasks.txt` lists all 920):

```bash
uv run cybergym-e2b batch --kind run --tasks-file ./my-tasks.txt --concurrency 4 \
  --agent codex --provider bedrock --model openai.gpt-5.4
```

`batch` skips tasks that already have a completed result for the same experiment fingerprint
(task, agent, model, template build, image digest, upstream revisions, patch, policy, options).
Infrastructure errors and interrupted agent turns are retried; `--no-reuse-completed` reruns all.

Results: `artifacts/e2e/<project>/<task>/<run-id>/result.json` plus `sandbox/` with upstream's
`agent_output`, run log, and exit code. `completed` says whether the adapter finished and collected
artifacts; `benchmark.status` is `passed`, `failed`, or `error`, and `benchmark.outcome` names the
verdict (`exact_match`, `valid_other_vulnerability`, `incomplete_agent_turn`, ...).

**Image locks.** Upstream references project images by mutable tag; a task runs only once its
image is digest-pinned in `artifacts/images.lock.json`. `images lock --task` (repeatable) resolves
what you need and merges. `images lock` alone resolves all 506 tags, which exceeds Docker Hub's
anonymous limit (about 100 manifest requests per hour); use it only after `docker login`.

## How a run works

1. Resolve the task from the pinned checkout; route `ffmpeg/*` tasks to the FFmpeg template.
2. Verify the template tag still points at the build recorded in `artifacts/templates/manifest.json`.
3. Create a fresh sandbox (8 vCPU, 8 GB RAM, 4 GB swap by default; killed on timeout). Secrets
   are injected by E2B's egress proxy per host: `HF_TOKEN` during setup, the model key during the
   run. The sandbox never holds either.
4. Upload patched upstream scripts and the one task; download only that task's data; pull and
   digest-verify the project image; set `vm.mmap_rnd_bits=28`.
5. Run upstream `run_agent.py` (or the S4 smoke), collect artifacts, kill the sandbox (`--retain`
   pauses it instead).

## Network policy and eligibility

The default policy (`assets/policies/network.json`) mirrors upstream: public internet, minus
private, loopback, link-local, carrier-grade NAT, and multicast IPv4 ranges. CyberGym tells agents
that network use invalidates the result, so default-policy results are marked
`requires_network_audit`; inspect the trajectory before publishing.

- `--egress restricted`: allowlist of the policy's model, dependency, and registry hosts. Still
  requires an audit.
- `--egress permissive`: no adapter filter, for diagnostics. Never eligible.
- `assets/policies/network-locked.json`: model host only at runtime, eligible without audit.
  Agent runs refuse it because Node and the agent CLI are installed at runtime; `smoke` works.

## Deviations from upstream

Nothing here changes grading. Each item is recorded in the experiment fingerprint or listed below.

| Area | Upstream | This adapter |
|---|---|---|
| Agents | `claude-code` (default), `codex`, `openhands`, `gemini-cli` | `codex`, `openhands` |
| Model access | Anthropic API, Bedrock SigV4, or LiteLLM proxy | Direct OpenAI-compatible endpoint; no LiteLLM |
| Codex | Codex defaults | `--reasoning-effort high` (configurable), `wire_api = "responses"` |
| Machine | Whatever host runs `run_agent.py` | 8 vCPU / 8 GB / 4 GB swap (`--cpu-count`, `--memory-mb` at build) |
| Images | Mutable tags | Digest-pinned via the image lock |
| Dataset | Full download | Per-task files only |
| Container setup | `apt-get install sudo git` | `apt` retry wrapper, `sudo` shim, dead third-party apt sources disabled, hardened Codex installer |
| Trajectory summary | After every failed attempt | Skipped on the final attempt (it only feeds the next one) |
| FFmpeg | Opus archive fetched at prepare time | Checksum-pinned archive baked into the template |

The container-setup changes exist because many project images pin apt repositories that no longer
publish a Release file, which aborts upstream's setup before the agent starts. They also remove
those sources for the agent; treat tasks where the agent needed one with care.

## Pinned inputs

`assets/upstream.lock.json` pins the upstream commit and dataset revision. Templates pin the base
and OSS-Fuzz builder images by digest, the Ubuntu archive at a dated snapshot, Docker package
versions, and a hash-locked Python environment. Template tags are content-addressed
(`cybergym-e2e-dind:recipe-<16 hex>`) and refused if their E2B build ID no longer matches the
manifest. `--patch-file`, `--remote-smoke`, `--remote-install-codex`, and `--network-policy`
override runtime assets and change the fingerprint.

## Known limitations

- No Claude Code / Anthropic path, so upstream's default configuration is not reproducible here.
- No plain OpenAI provider.
- The full image lock needs an authenticated Docker Hub session.
- The 8 GB shape is validated on a sample of tasks, not all 920; check
  `observed_resources` in `result.json` for memory and swap pressure.
- E2B's build API takes no disk size; `--disk-limit-gb` is recorded only, and the runtime
  free-space check is what protects a run.
- Open engineering items: [FOLLOW-UPS.md](FOLLOW-UPS.md).

## Development

```bash
uv sync --locked && uv run cybergym-e2b sync-upstream   # tests read the vendored checkout
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
uv run python scripts/check_public_tree.py
```

CI runs the suite on Python 3.12 and 3.13, installs the wheel outside the checkout, and reruns the
tests from the extracted sdist. Security reports: [SECURITY.md](SECURITY.md). Apache-2.0.
