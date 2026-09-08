# Follow-ups

Issues are disabled on this repository, so this file is the tracker. Delete entries as they land.
None of these affects the correctness of a completed run.

## Compatibility patch (`assets/patches/openai-compatible.patch`)

1. The `poc_file is None or ...` and `patch_file is None or ...` guards turn upstream
   `run_agent()`'s exception path into a graded `no_poc` / `no_patch` failure instead of
   upstream's retryable `error` status. Codex results are rescued by the trajectory check in
   `_benchmark_result`; OpenHands results are not.
2. `LLM_MAX_INPUT_TOKENS=131072` equals DeepSeek V3.2's full context window, leaving no headroom
   for the 8192-token completion on that model via OpenHands.

## Coverage

3. Add a plain `openai` provider (host `api.openai.com`) so customers without Fireworks or Bedrock
   can run Codex. The runtime already speaks the OpenAI Responses API; this needs a
   `--provider` choice, a host entry in both packaged policies, and a key name.
4. Upstream's default agent is `claude-code` with the Anthropic API. Supporting it needs
   `ANTHROPIC_API_KEY` delivery through the egress proxy (upstream passes it as a container
   environment variable) and `api.anthropic.com` in the packaged policies; upstream installs the
   CLI with `npm install -g @anthropic-ai/claude-code` inside the agent container. Without it the
   paper's reference configuration cannot be reproduced on E2B.

## Notes

5. The template tag-to-build check is cached per process on purpose; rebuilding or re-verifying
   per run is not wanted.
