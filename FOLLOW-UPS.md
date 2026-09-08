# Follow-ups

Confirmed findings from the independent release review of PR #1 that PR #2 deliberately left
alone. None affects the correctness of a single run: with PR #2 applied, a live `run` and a live
`smoke` of `curl/arvo_66012` completed cleanly on E2B. They matter for long batches and for
reliably preserving trajectories. Issues are disabled on this repository, so this file is the
tracker; delete entries as they land.

## Runtime reliability (`src/cybergym_e2b/runtime.py`)

1. **Unguarded post-workload collection path.** The `finally` after the workload and the failure
   handler call `_resource_summary` and `_disk_and_docker` without try/except. One truncated line
   in the monitor's JSONL turns a completed run into a `JSONDecodeError`, and on the failure path
   the exception escapes before `_collect` runs, so a multi-hour trajectory is never downloaded.
   Wrap those calls and always attempt `_collect`.
2. **No liveness check while polling the workload.** `_shell_run` swallows every poll exception,
   including a killed sandbox, and sleeps until the deadline. A dead sandbox or an OOM-killed
   workload that never writes the exit file pins a batch worker for the full evaluation timeout.
   Check that the sandbox and process are alive and fail fast.
3. **No time left for collection after a workload timeout.** The poll deadline is
   `evaluation_timeout - 120`, but the post-timeout path runs disk checks (timeout 120) and a
   tarball download (timeout 300) before `on_timeout=kill` fires. Reserve budget for collection
   and validate `evaluation_timeout` so a small value cannot produce a negative poll window.
4. **Resource monitor output deleted by bundle upload.** The monitor starts just before
   `_upload_bundle` runs `rm -rf /opt/cybergym-e2e && mkdir -p`, discarding early samples and
   racing an unguarded `open()` that can kill the monitor silently. Start the monitor after the
   upload, or write samples elsewhere, and record monitor health in the summary.
5. **Host-side DNS for FFmpeg hosts can fail the whole project.** `_resolve_non_http` calls
   `socket.getaddrinfo` with no error handling for every FFmpeg task, even under the default
   policy where the addresses are never used. Skip it unless the policy enforces an allowlist and
   turn a lookup failure into a clear per-task error.

## Inherited from the compatibility patch (`assets/patches/openai-compatible.patch`)

6. The `poc_file is None or ...` and `patch_file is None or ...` guards turn upstream
   `run_agent()`'s exception path into a graded `no_poc` / `no_patch` failure instead of upstream's
   retryable `error` status. Codex is rescued by `_codex_turn_state`; OpenHands is not.
7. `LLM_MAX_INPUT_TOKENS=131072` equals DeepSeek V3.2's full context window, leaving no headroom
   for the 8192-token completion on that model via OpenHands.

## Also noted

8. `_network_eligibility` returns `eligible` for a model-only policy that
   `_require_runnable_policy` refuses for agent runs. Consistent today, because no agent run can
   carry that status, but `_network`, `_network_eligibility`, and `_require_runnable_policy`
   duplicate the same egress predicate and should share one.
9. Per-process caching of the template tag-to-build check is intentional (no rebuild per run).
   Listed so it is not re-reported.
