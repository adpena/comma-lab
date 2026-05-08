# Cross-Paradigm ADMM x Op1 macOS Advisory Negative - 2026-05-08

Scope: classify the byte-closed `153,513 B` cross-paradigm ADMM continuous-K
plus Op1 finalizer archive on the fast local CPU advisory axis. This is not a
contest score and cannot promote, rank, or kill.

## Candidate custody

- Lane family: `cross_paradigm_admm_continuous_k_plus_op1_finalizer`
- Archive path:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/archive.zip`
- Archive bytes: `153513`
- Archive SHA-256:
  `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`
- Build manifest:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/build_manifest.json`
- Build manifest declared `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`, and `cuda_eval_worth_testing=true`
  with blockers `cpu_build_rel_err_proxy_not_score_evidence`,
  `exact_cuda_auth_eval_not_yet_harvested`, and
  `requires_contest_auth_eval_json_before_score_promotion_rank_or_kill`.

## Local runtime patch

The generated cross-paradigm `inflate.sh` defaults to the contest-facing
`uv run --no-project --with ...` path. For local advisory eval on macOS, the
builder now emits an explicit `PYTHON` override path:

```bash
if [ -n "${PYTHON:-}" ]; then
  "$PYTHON" "$HERE/inflate.py" "$SRC" "$DST"
else
  "$UV_BIN" run --no-project "${UV_WITH_INFLATE_DEPS[@]}" python "$HERE/inflate.py" "$SRC" "$DST"
fi
```

The harvested runtime was patched the same way for this advisory eval. The
charged archive bytes and archive SHA did not change.

## Local macOS CPU advisory eval

Claim:

- `cross_paradigm_admm_x_op1_macos_cpu_advisory_20260508T234734Z`
- terminal status: `completed_macos_cpu_advisory_score_0p328444`

Command:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/archive.zip \
  --inflate-sh experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/macos_cpu_advisory_work \
  --json-out experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Result:

- Evidence grade: `[macOS-CPU advisory]`
- Evidence semantics: `non_contest_cpu_auth_eval_advisory`
- Canonical score: `0.32844434076752543`
- PoseNet distortion: `0.00014180`
- SegNet distortion: `0.00188570`
- Rate contribution: `0.10221800`
- Archive bytes: `153513`
- Runtime tree SHA-256:
  `4a3fdcb6fbe8aed4263b283da89a96ec6f0dff8dba1efdcd3811fda5228ecdea`
- Durable JSON:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/contest_auth_eval.macos_cpu_advisory.json`

Interpretation:

- The rate term improved to `0.102218`, but SegNet dominates the failure:
  `100 * seg = 0.18857`, already near the whole public-medal score before
  pose and rate are added.
- Pose is not the primary failure mode on this advisory axis.
- The `rel_err≈4%` CPU-build proxy is confirmed insufficient as a promotion
  or dispatch criterion for this config.

Disposition:

- Retire this measured archive/config for score work unless a formal exact
  negative is explicitly desired.
- Do not kill the cross-paradigm family. Reactivation requires
  scorer-aware/seg-boundary-aware per-tensor allocation, a lower-distortion
  trust region, or a reconstruction-preserving retrain before exact CUDA or
  contest-CPU spend.
