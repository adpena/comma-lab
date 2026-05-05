---
name: No Hacks TODOs or Workarounds Ever
description: Non-negotiable standard — every implementation must be complete, correct, tested end-to-end, and production-hardened. No partial work.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Rule

NEVER leave TODOs, workarounds, hacks, partial implementations, or monkey-patches in the codebase. Every change must be:

1. **Fully implemented** — no stubs, no placeholders, no "fix later"
2. **Completely correct** — no "good enough", no approximations unless mathematically justified
3. **Tested end-to-end** — run the actual pipeline, verify output, try to break it
4. **Production-hardened** — hyperscale grade, NVIDIA/OpenAI/Cloudflare standard
5. **Extremely optimized** — for performance, efficiency, memory, compute

**Why:** The monkey-patch of AllNorm.forward was a hack that masked the real issue (non-contiguous tensors from channels_last layout). Hacks accumulate and eventually cause cascading failures that are impossible to debug. The proper fix is always to address the root cause.

**How to apply:** Before committing any code change:
- Run the full training pipeline end-to-end
- Verify output is numerically correct
- Try adversarial inputs (empty tensors, wrong shapes, edge cases)
- Check for non-contiguous tensors, memory leaks, resource leaks
- Never use `# TODO`, `# HACK`, `# FIXME`, `# WORKAROUND` — fix it now or don't commit
