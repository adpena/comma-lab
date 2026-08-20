# ddm_fb1 FA1 Soft Velocity Blend Backtest Receipt

## Backtest Table

| Corpus | State custody found | Gradient custody found | Backtest status | Measured/audited signal |
|---|---:|---:|---|---|
| J3 finalsmoke 366 opening window | no previous mapped boundary moment | 1/4 reconstructable vectors | `PARTIAL_STEP1_ONLY_REFUSED` | step1 telemetry grad norm `1.3735015609101762`; reconstructed first-step gradient RMS `0.026064301628862335`; uncorrected Adam eta(1) `3.1622776601683773`; step4 n600 `d_seg=0.027603208753797744`, `d_pose=163.0613308426994` |
| J4 warm-start reform 366 opening window | diagnosis says warm-start optimizer state loadable `false` | 0/4 vectors | `REFUSED_MISSING_REPLAY_CUSTODY` | step1 grad norm `1.372286340806738`; LR rewarmup factor `0.1`; realized boundary crossed `false` |
| JD-line TR1 continuation windows | sample checkpoint has 20 `opt::*.m` moment keys | 0/4 vectors | `REFUSED_SCALAR_TELEMETRY_ONLY` | 6 windows found; sample `tr1_jd4_cont_ep1766_q3on`; telemetry first row has no gradient vector; reset arm `B` |

Machine receipt: `.omx/research/ddm_fb1_20260807/backtest_receipt.json`.

## Verdict Row

`LESSON-ONLY-confirmed`.

The FA1 soft-blend treatment is designed and represented by real arithmetic:
`m_new=(1-alpha(t))*m_mapped+alpha(t)*m_fresh`, with alpha over a beta2-derived
optimizer-step window `ceil(2.0/(1-0.999)) = 2000` steps. The banked corpus does
not support an honest comparison against discrete COLD reset, because no row has
both previous boundary first-moment state and the required recorded gradient
sequence. No ADOPT-build-ready verdict is claimed.

Named consumer trainer for a future hook:
`experiments/train_levelset_witness_realized_through_R_mlx.py`.

Fire condition: capture a real stage-boundary checkpoint with previous mapped
optimizer state, deterministic post-boundary gradient vectors for the beta2
window or a pre-registered shorter opening slice, and matched cold/reset
controls. Then rerun the scorer-free arithmetic backtest and only enable the
trainer consumer if the soft blend wins both controls at matched update RMS.

## Built Artifacts

- `src/tac/optimization/stage_transition_soft_velocity_blend.py`: pure numpy
  optimizer-state arithmetic, fail-closed on missing/invalid arrays.
- `src/tac/witness_dsl/curriculum_dsl.py`: `StageTransitionSoftVelocityBlend`
  DSL factory, default OFF, emits no trainer flags, refuses `enabled=True` until
  a real consumer exists.
- `tools/backtest_fa1_stage_transition_soft_velocity_blend.py`: scorer-free
  banked-corpus audit tool that writes the JSON receipt.
- Tests:
  `src/tac/optimization/tests/test_stage_transition_soft_velocity_blend.py` and
  added DSL inert/refusal coverage in `src/tac/tests/test_witness_curriculum_dsl.py`.

## Recall Evidence

Searched and read beyond the charter seeds:

- FA1/AH1 receipts: `.omx/research/ddm_fa1_20260807/RECEIPT.md`,
  `.omx/research/ddm_fa1_20260807/CROSSWALK.md`,
  `.omx/research/ddm_ah1_20260807/FA1_STAGE_TRANSITION_SOFT_VELOCITY_BLEND.md`.
- Warm-start receipts: `.omx/research/codex_findings_ddm_j3_366_fullrun_mode_ticket_reseal_20260723_codex.md`,
  `.omx/research/codex_findings_ddm_j4_366_warm_start_reform_20260723_codex.md`,
  `.omx/research/ddm_gc15_fresh_vs_warm_20260731.md`,
  `.omx/research/p0_resume_warmup_geometry_build_20260717.md`.
- Existing code surfaces: `src/tac/optimization/reset_operator.py`,
  `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`,
  `experiments/train_levelset_witness_realized_through_R_mlx.py`.
- Banked corpus:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j3_366_fullrun_finalsmoke_64c421698c_20260723T030700Z`,
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j4_366_warmstart_smoke_9c3575aa_20260723T042700Z`,
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1766_q3on`.

Key plan change from recall: the existing GC15 reset operator and TR1 reset DSL
arm are not the FA1 levelset soft-blend consumer. Therefore the only honest FB1
implementation here is default-off arithmetic plus a refusing backtest receipt,
not a trainer launch or an enabled flag.

## Boundaries

No training launched. No scorer launched. No archive built. No exact-eval row.
No `/tmp` evidence path. This is `[optimizer-state arithmetic / scorer-free audit]`.

Own-vehicle live frontier remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
`0.19108 [contest-CPU]` remains borrowed and unmoved.
