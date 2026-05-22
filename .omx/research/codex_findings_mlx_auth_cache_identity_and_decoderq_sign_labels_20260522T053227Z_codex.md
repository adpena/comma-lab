# Codex Findings: MLX Auth-Cache Identity And Decoder-Q Sign Labels

utc: 2026-05-22T05:32:27Z
lane: lane_codex_mlx_auth_hardening_swarm_20260522
status: LANDED
evidence_grade: macOS-MLX-research-signal
score_claim: false
score_claim_valid: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false

## Findings

The existing Modal/Linux CPU scorer-input hash packet is internally strict:

`experiments/results/mlx_fec6_auth_cache_identity_20260522/cache_vs_modal_cpu_hash_identity.json`

Result:

- `passed=true`
- `verdict=PASS_CACHE_AUTH_EVAL_IDENTITY`

This proves the auth-side hash packet is self-consistent. It does not yet make
the current local macOS MLX cache production-eligible, because the local cache
was built from a different decoded raw surface. The remaining materialization
step is still to recover/export Modal/Linux raw bytes or Modal/Linux scorer
input tensors, then rebuild the local cache against that surface.

The decoder-q response-surface advisory gate now preserves failed
surface-guided candidates as signed calibration labels:

- schema: `decoder_q_surface_sign_calibration_labels.v1`
- allowed use: local decoder-q sign calibration only
- forbidden use: score claim, rank/kill, promotion, or submission
- regressing labels recommend suppressing same-sign atoms and trying inverse
  signs in the next waterbucket pass

`tools/plan_decoder_q_signed_waterbucket.py` can consume those labels through
`--decoder-q-signed-calibration` and prioritize inverse-of-regressing atoms
without turning advisory data into score authority.

## Subagent Audit Harvest

- Grayscale LUT A100 timeout was not OOM. `best.pt` is recoverable into GLV1
  bytes, but the timed-out run never exported `0.bin` or `archive.zip`.
  Next patch: add export-only-from-checkpoint and soft-deadline early stop.
- NSCS06 v8 rc=1 advanced past the old mode-hardcode guard and failed because
  full mode still reached CPU. Next patch: add recipe/protocol pre-dispatch
  full-mode device guards before Modal spend.
- HFV2 sidecar dirty WIP is coherent but unrelated to MLX; it remains
  intentionally unstaged and untouched.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_input_cache.py ...`
  emitted `PASS_CACHE_AUTH_EVAL_IDENTITY` for the Modal CPU hash packet.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_mlx_auth_cache_materialization.py ...`
  emitted `READY_USE_EXISTING_AUTH_TRANSFER_CACHE` for the auth-side hash
  packet and `AUTH_CACHE_MATERIALIZATION_REQUIRED` for the current local-vs-
  Modal mismatch packet.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check ...` passed for
  touched MLX local-acceleration, decoder-q, and test files.
- Focused MLX/decoder-q pytest passed: `203 passed in 16.17s`.
- After `main` advanced to `5b5a4fb74`, post-advance focused pytest passed:
  decoder-q/scorer-response `62 passed in 1.01s`; MLX auth/calibration/contract
  `55 passed in 0.74s`.
- `git diff --check` passed.
- Generated experiment outputs remain ignored by `.gitignore:61`
  (`experiments/results/*`).

## Next Actions

1. Export grayscale LUT `best.pt` into a byte-closed archive locally, then run
   smoke inflate before any exact eval.
2. Add soft train deadline and early-stop export for grayscale LUT so the next
   paid run cannot time out before archive export.
3. Add NSCS06 full-mode device guards at the operator-authorize/preflight layer.
4. Add the Modal/Linux tensor-cache export hook required to turn the strict
   auth-side hash packet into a local MLX production cache.
