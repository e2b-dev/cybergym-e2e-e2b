# Follow-ups

Issues are disabled on this repository; this file is the tracker. None affects a completed run.

1. **Patch:** the `poc_file is None or ...` / `patch_file is None or ...` guards turn upstream's
   exception path into a graded `no_poc` / `no_patch` failure instead of a retryable `error`.
   Codex results are rescued by the trajectory check; OpenHands results are not.
2. **Patch:** `LLM_MAX_INPUT_TOKENS=131072` equals DeepSeek V3.2's full context, leaving no room
   for the 8192-token completion via OpenHands.
3. **Provider:** add plain `openai` (`api.openai.com`): a `--provider` choice, a host in both
   packaged policies, a key name.
4. **Agent:** `claude-code` with the Anthropic API is upstream's default. Needs
   `ANTHROPIC_API_KEY` via the egress proxy, `api.anthropic.com` in the policies, and the
   upstream `npm install -g @anthropic-ai/claude-code` step.

Intentional, do not re-report: the template tag-to-build check is cached per process.
