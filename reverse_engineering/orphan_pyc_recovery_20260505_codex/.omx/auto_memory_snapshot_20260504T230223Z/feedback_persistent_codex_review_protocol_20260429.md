---
name: PROTOCOL — persistent recursive codex CLI xhigh adversarial review
description: 2026-04-29 PM. User mandate. As work returns, launch background recursive adversarial extreme-rigor senior-engineer reviews via codex CLI xhigh. Each review covers a different perspective from previous rounds. Output to /tmp/codex_reviews/<labeled>.log.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule**: every meaningful landing (subagent commit, Modal lane completion, new module, new lane script, codex council session output) triggers a recursive adversarial review via raw codex CLI gpt-5.5 xhigh. Run as background process. Different perspectives each round.

**Invocation pattern**:
```bash
codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    "<rigorous review prompt with file paths + previous-round perspectives + new perspectives>" \
    > /tmp/codex_reviews/<labeled>.log 2>&1 &
```

**Why**: independent extreme-rigor review catches what subagent-author + my-self-review miss. Codex with xhigh reasoning produces senior-engineer-level critique. Asynchronous so doesn't block forward progress.

**Round-tracker** (perspectives rotate):
- **Round 1** (Selfcomp port): clamp range bug, non-existent kwarg, dead noise_std (3 CRITICAL caught)
- **Round 2** (Selfcomp polish): bilinear-vs-bicubic at inflate, yuv420p vs gray pix_fmt, NaN/Inf passthrough, LUT divergence, format-compat (1 CRITICAL + 4 Medium)
- **Round 3** (Subagent H hardening + Lane MM v2 falsification): perspectives A-H = numerical-edge / codec-roundtrip / format-compat / batch-chunking-correctness / EMA+gradclip / frame-indices / pair-weights / STRICT-bypasses
- **Round 4+** (per-landing): different perspectives each round

**Persistent automation** (when applicable):
- Wrap codex invocation in a small helper script: `tools/codex_review.sh <label> <prompt-file>` that handles BG launch + log labeling + status check.
- Keep `/tmp/codex_reviews/` directory clean; log labels include date+timestamp.
- After codex returns, classify findings as CRITICAL/Medium/Low and act:
  - CRITICAL: stop, fix, re-test before promoting
  - Medium: queue as task, fix during cooldown windows
  - Low: queue for batch fix later

**Failure modes to avoid**:
- Bash 144 SIGURG kill on long sessions: use `Agent` tool wrapper if codex prompt requires >3min.
- Codex auth expiry: requires `codex login` interactive (human action).
- ChatGPT-account routing rejects o3 model fallback.
- 0-output codex sessions: log file has 0 lines. Re-run via Agent wrapper.

**3-clean-pass gate** (CLAUDE.md non-negotiable for major training-code changes):
- 3 consecutive rounds with zero CRITICAL findings before code is cleared for promotion.
- Counter resets on any CRITICAL.

**Coverage targets**:
- All landed Subagent commits
- All Modal lane completions (rc=0 score interpretation + rc!=0 root-cause)
- All new src/tac/ modules
- All new lane scripts
- All preflight check additions
