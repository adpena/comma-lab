# Track A apogee_int6 Lightning T4 dispatch — outcome

- **Date**: 2026-05-07T10:16Z
- **Status**: REFUSED at submit step (Lightning API)
- **Predispatch sanity**: override accepted (exit 65) — logged to `.omx/state/predispatch_overrides.log`
- **Operator override reason** (≥40 chars): "2026-05-07 operator pivot: continue with all (parallel A+B+C+D); apogee_int6 dispatch produces real [contest-CUDA] anchor at 1.55% rel_err (third lossy anchor calibrating predictor across [0.24%,7.09%] gap). Race-mode rigor inversion non-negotiable applies post-leader-shift (top 3: 0.193/0.195/0.195 published)."
- **Lane claim**: `.omx/state/active_lane_dispatch_claims.md` — terminal status `refused_dispatch_lightning_aws_t4_capacity_at_2026_05_07T10_16Z`
- **Staging**: COMPLETE — 1653 files / 200MB uploaded; manifest_sha256=`dea0c7a6c60feb3685aadaa8b509e50280a2f820be6fe4a9107f6d967612f46d`
- **Submit error**: Lightning SDK ApiException(400) — `"accelerator T4 not found for this AWS cluster"`
- **Predicted band (still pending)**: [0.190, 0.204] [predicted-band, NOT contest-CUDA]
- **Forensic warning carried over**: `forensic_byte_only_invalidated_by_int4_exact_negative` (per repack_metadata)

## Recovery options

1. Retry Lightning at next loop tick (~25 min) — capacity may return.
2. Pivot to Vast.ai 4090 — `scripts/launch_lane_on_vastai.py full --disk 60 --min-cuda-vers 12.4 ...`. Cost ~$0.25/hr, ~30-45 min wall-clock.
3. Wait for codex partner queue (per `feedback_codex_partner_coordination_state_20260501T1310Z.md`).

## Cross-references

- Override log: `.omx/state/predispatch_overrides.log`
- Lane claims: `.omx/state/active_lane_dispatch_claims.md`
- Source manifest: `source_manifest.json` (this dir, gitignored due to size)
- Invocation env: `_dispatch_invocation.env` (this dir)
- Canonical Lightning recipe: `~/.claude/projects/.../memory/reference_lightning_studio_canonical_dispatch_recipe_20260505.md`
