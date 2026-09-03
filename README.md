# CyberGym-E2E on E2B

This repository is the standalone E2B adapter for
[CyberGym-E2E](https://github.com/sunblaze-ucb/cybergym-e2e). It runs one source-only
CyberGym-E2E task in one private E2B sandbox and delegates agent execution and the S1–S4
validators to the pinned upstream harness. It does not implement the classic binary-only
CyberGym benchmark.

The integration pins:

- CyberGym-E2E code commit `b861317f11641b14ab6ba08b5179d0b044601057`;
- dataset revision `a65d1d273eb7ee5db7525418120fc2434b887203`;
- the E2B template base image and preloaded OSS-Fuzz images by SHA-256 digest;
- the Ubuntu package archive at a dated snapshot;
- every Docker package version and the Docker repository signing-key checksum;
- every Python template dependency and transitive dependency by version and distribution hash;
- the FFmpeg project image and Opus model archive by SHA-256 digest.

The pinned upstream inventory contains 920 tasks across 139 projects. The gated dataset is never
baked into a template. Each sandbox downloads only the data files for its requested task.

## Repository and artifact boundary

This public repository contains the adapter, compatibility patch, runtime helpers, network
policies, construction locks, and tests. Those assets ship inside the Python wheel, so an installed
`cybergym-e2b` command does not depend on the current working directory or a source checkout.

The adapter was extracted from E2B's benchmark-conversion work and is maintained here as the
artifact source of truth. The upstream lock in
`src/cybergym_e2b/assets/upstream.lock.json` is the executable source of truth for the code and
dataset inputs used by the extraction. Runtime constants are loaded from that packaged lock rather
than maintained as duplicate pins.
The repository intentionally excludes gated dataset contents, API keys, experiment campaigns,
customer or operator reports, model trajectories, run results, generated template manifests, and
template build ledgers. Those files are local operational artifacts and are ignored by Git.

## Security and identity contracts

Every runtime project image must use the form `repository@sha256:<64 lowercase hex characters>`.
Mutable tags are rejected before artifact lookup or sandbox creation. Generate the complete local
lock for the pinned inventory before running any task:

```bash
uv run cybergym-e2b images lock
```

The command inspects registry manifest descriptors with Docker Buildx; it does not download image
layers. It resolves all 506 unique mutable project-image tags, records the pinned upstream commit
and exact resolver tool/version/method, and atomically writes `artifacts/images.lock.json` only
after every result is an unambiguous SHA-256 digest. It also refuses a moved FFmpeg tag that no
longer matches the independently pinned FFmpeg digest. Run, smoke, batch, and preflight consume
this lock by default and reject missing, partial, stale, extra, mutable, or provenance-free maps.
Use `--image-lock PATH` to select another generated lock; `--image-map` remains a compatibility
alias. After Docker pulls or finds an image in the sandbox, the runner verifies that the observed
repository digests contain the locked digest.

Templates use normal E2B aliases with content-addressed tags such as
`cybergym-e2e-dind:recipe-0123456789abcdef`. The recipe digest covers the immutable construction
inputs, upstream revisions, dependency locks, preloaded images, and resource request. A local atomic
build ledger at `artifacts/templates/build-ledger.json` reuses a tagged template only when the
complete recipe matches. The generated manifest records the stable tag, immutable E2B template ID,
immutable build ID, and full recipe digest. Preflight and runtime query E2B and reject the manifest
if its tag no longer points to the recorded build. Losing the ledger causes a rebuild attempt under
the same recipe tag; it never reuses an unverified moving tag.

Secrets belong in a local `.env` or another file selected with `CYBERGYM_KEYS_FILE` or
`--keys-file`. `HF_TOKEN` and the model-provider credential are delivered with E2B request
transforms: the sandbox receives placeholders while the egress proxy substitutes authorization on
approved hosts. Public ingress is disabled. The Hugging Face rule is removed before model runtime.

## Prerequisites

- Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker with the Buildx plugin, plus registry credentials for any non-public project images
- an E2B account with enough template and sandbox capacity
- accepted access to the gated
  [CyberGym-E2E dataset](https://huggingface.co/datasets/sunblaze-ucb/cybergym-e2e)
- `E2B_API_KEY`, `HF_TOKEN`, and a credential for the selected model provider

Install dependencies and fetch the exact upstream source:

```bash
uv sync --locked
uv run cybergym-e2b sync-upstream
uv run cybergym-e2b inventory
uv run cybergym-e2b images lock
```

`sync-upstream` refuses to replace a non-Git path or modify a dirty managed checkout. It fetches
and detaches at the pinned commit under `vendor/cybergym-e2e` by default.

## Build the E2B templates

Build the base template and then the FFmpeg cache-bearing derivative:

```bash
uv run cybergym-e2b templates build-base \
  --cpu-count 8 \
  --memory-mb 8192 \
  --disk-limit-gb 120
uv run cybergym-e2b templates build-ffmpeg
```

Both commands write `artifacts/templates/manifest.json`. The disk value records the expected E2B
account-tier root-disk allocation; E2B's template build API accepts CPU and memory but does not
accept a disk-size field. Runtime free-space preflight is authoritative.

The base template contains Docker-in-Docker, the hash-locked Python environment, the sanitizer
ASLR setting, and both pinned OSS-Fuzz base-builder images. The FFmpeg template adds the pinned
FFmpeg image and the checksum-verified Opus archive needed during upstream preparation.

## Validate and run

Preflight resolves the same model, network policy, project-aware template route, and immutable task
image as runtime. It verifies the upstream checkout, selected provider credential, E2B tag-to-build
receipt, and gated dataset access:

```bash
uv run cybergym-e2b preflight \
  --task curl/arvo_66012 \
  --provider bedrock \
  --model openai.gpt-5.4
```

An infrastructure smoke compiles a fresh nested project and runs the upstream ground-truth S4
validator:

```bash
uv run cybergym-e2b smoke curl/arvo_66012
```

Run one source-only agent task:

```bash
uv run cybergym-e2b run curl/arvo_66012 \
  --agent codex \
  --provider bedrock \
  --model openai.gpt-5.4
```

Run a bounded batch with an operator-owned task file:

```bash
uv run cybergym-e2b batch \
  --kind run \
  --tasks-file ./local-tasks.txt \
  --concurrency 4 \
  --agent codex \
  --provider bedrock \
  --model openai.gpt-5.4
```

The upstream Docker runner does not disable networking, so the comparison-compatible default keeps
public egress while denying private, loopback, link-local, carrier-grade NAT, and multicast IPv4
ranges. CyberGym instructs agents that network use invalidates the result. The adapter therefore
marks default-policy results `requires_network_audit`; inspect trajectories and network evidence
before including them in a published comparison. `--egress restricted` uses a public dependency
allowlist and also requires an audit. `--egress permissive` is diagnostic and is always ineligible.
The packaged `network-locked.json` policy permits only the selected model endpoint during runtime
and is eligible without a public-egress audit after all task dependencies have been preloaded.

Runtime assets can be overridden with `--patch-file`, `--remote-smoke`,
`--remote-install-codex`, and `--network-policy`. Overrides are included in the experiment
fingerprint.

## Execution and result semantics

For each task, the runner:

1. merges the upstream project and task configuration and resolves an immutable runtime image;
2. selects the base or FFmpeg recipe tag and verifies it still resolves to the manifest's immutable
   build receipt;
3. creates a fresh sandbox with kill-on-timeout and automatic resume disabled;
4. configures bounded swap, Docker, ASLR, and free-space preflight;
5. uploads only the upstream scripts, compatibility assets, project configuration, and requested
   task directory;
6. downloads only the requested gated data and verifies the project-image digest;
7. replaces setup-only dataset authorization with the runtime model authorization;
8. runs the upstream agent or smoke workload and collects its outputs; and
9. kills the sandbox unless `--retain` requests a paused debugging sandbox.

Results are written below `artifacts/e2e/<project>/<task>/<run-id>/result.json`. `completed`
describes orchestration and artifact collection. `benchmark.status` is separately normalized to
`passed`, `failed`, or `error`; `benchmark.outcome` distinguishes an exact ground-truth match from
another valid vulnerability.

Artifact reuse requires an exact experiment fingerprint covering task kind, agent and model
configuration, verified template build receipt, immutable runtime image, source and dataset revisions,
compatibility assets, network policy, and runtime options. A model failure with a completed agent
turn is reusable evidence. Infrastructure errors, interrupted turns, mutable identities, and legacy
results without the current fingerprint are retried or rejected.

## Development and release checks

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_public_tree.py
actionlint .github/workflows/ci.yml
uv build --no-build-isolation --clear
uv run twine check dist/*
```

CI runs tests on Python 3.12 and 3.13, checks the public tree, verifies the installed wheel outside
the checkout, and reruns the suite from the extracted source distribution.

Security reports follow [SECURITY.md](SECURITY.md). The project is licensed under Apache-2.0.
