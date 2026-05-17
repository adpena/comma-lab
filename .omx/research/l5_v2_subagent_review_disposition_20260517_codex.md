# L5 v2 subagent review disposition - 2026-05-17

Scope: preserve and disposition the `Euler the 3rd` L5-v2 readiness review
returned on 2026-05-17 before pushing `main`, so the review signal is not lost
in chat.

Evidence axis: planning/custody hardening only. This memo is not a score claim,
not promotion evidence, and not exact-eval dispatch authorization.

## Findings disposition

1. Stale Modal provider blocker bypassed into executable next action.
   Disposition: fixed in current `main` before this memo. The architecture
   packet now suppresses Modal execute while the active provider blocker is
   present, and tests cover stale/invalid blocker artifacts.

2. Stale Lightning source custody still reported dry-run readiness.
   Disposition: fixed in current `main` before this memo. The paired-axis plan
   status now separates structural dry-run validity from execution-current
   source custody.

3. Per-axis TT5L evidence rows preserved stale paired-axis state.
   Disposition: fixed in current `main` before this memo. The diagnostic anchor
   consumes the pair-level machine artifact and keeps the measured CPU/CUDA pair
   non-promotional instead of relying on stale per-axis row flags.

4. Materialized work-unit runtime content equality not enforced at that layer.
   Disposition: fixed in current `main` before this memo. The materialized work
   unit status now rejects CPU/CUDA runtime-content SHA mismatch.

5. Follow-on signal-preservation gap found while reviewing the subagent output:
   TT5L side-info effect-curve status surfaced only blockers, hiding the partial
   negative evidence that the lone observed CUDA/trained cell had all-zero
   side-info liveness and that nine paired cells were missing. This memo's
   companion patch promotes that evidence into the machine status and rendered
   architecture packet.

## Current TT5L side-info effect-curve status

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `artifact_valid=false`
- observed cells: one `contest_cuda/trained` cell
- observed cell score: `3.9007398365396795`
- observed side-info liveness: `0 / 27000` nonzero values
- missing cells include all required CPU variants and CUDA controls

Next unblocked action remains to resolve the Modal provider blocker or make the
Lightning alternate provider executable-current before running the paired
side-info effect curve.
