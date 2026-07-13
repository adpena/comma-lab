# REPLACE round 3 — attack the frozen-SegNet costate fidelity wall

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round3_fidelity_wall_20260713`  
**Authority:** local macOS CPU fp32 training-gradient research evidence only  
**Status:** `NO_GO_REGISTERED_ROUND3_RUNGS`; `research_only=true`; `score_claim=false`; `promotion_eligible=false`  
**Pointer delta:** `NONE`

## Executive verdict

**Winning measured rung: fixed RFF, but the registered rung is a scoped NO-GO.** Its aggregate heldout input-costate cosine is `0.0016791964165317613`, below the preregistered `0.07078966932743762` bar by `0.06911047291090586`. It is only `1.1860462356199448x` round-2 noise, not `50x`; relative L2 is `1.0000003871015077`. Linear is nearly identical at cosine `0.0016650255538056325`.

Neither target reformulation passes. At realized `4.7017415%` area, source-margin risk retains `16.3468%` of exact input-costate L2-square mass (`3.4767x` uniform; conditional exact cosine `0.40431`); the learned RFF mass head retains only `2.44265%` (`0.5195x` uniform; cosine `0.15629`). The same-area oracle retains `52.7815%`, so the support-localization family remains open even though both registered localizers fail.

The prefix costs a DERIVED `0.571411805%` of full exact-teacher forward-plus-input-backward convolution FLOPs, or a `175.005x` conv-only ideal ratio. This is not a wall-speed claim: host contention contaminated exact/surrogate timing intervals.

Clean-run teacher economics are `15x` label-only and `12x` inclusive. Conservatively charging every harness retry and one interrupted start gives the campaign-honest values: `626` training starts and `746` total starts, hence `11.501597444089457x` label-only and `9.651474530831099x` inclusive. FORE remains uncomposed for the current instance.

## What was held settled

Round 2 proved the fixed-replay custody, deterministic convex-head machinery, full-batch ridge contraction, and cached-label economics. It also measured the fidelity wall: heldout input-costate cosine `0.0014157933865487525` and relative L2 `1.0000018705777456` for the 31-feature hand chart. This round does not re-derive those results and does not edit the committed round-2 module, policy, or memo.

The round-2 negative remains `FORMULATION x INSTANCE`. It does not cover frozen-stem features, fixed RFF lifts, target localization, transition-complete FORE, on-policy replay, other charts, or other seeds.

## Preregistered rule, before measurement

- Population: 600 unique real V9 states; checkpoint epochs 150/251/275; deterministic seed455; 480 train and 120 heldout.
- Rung 1: convex ridge on a local pre-SE frozen SegNet prefix chart.
- Rung 2: one fixed seed455 16-frequency RFF lift; no width/seed sweep.
- Rung 3: source-margin risk then learned RFF costate-mass localization at 4.7% area.
- Direction admission: aggregate heldout input-costate cosine `>=0.07078966932743762` and positive-dot state fraction `>=0.60`.
- Localizer admission: retained exact input-costate L2-square fraction `>=0.47`; conditional masked-exact cosine is `sqrt(retained fraction)`.
- Stop: first passing rung in EV order wins; later rungs are not measured.

The cosine bar is `50x` the measured round-2 noise, corresponding to at least `0.501118%` normalized projected directional energy. It replaces the dead negative round-1 bar.

## Formulations and math

The local chart is `bias + 32 frozen prefix activations + 5 source-class one-hot + tanh(source margin) + 3 stage one-hot`, sampled at stride 4 on the prefix lattice during fit. The trainable object remains a ridge head. It predicts the exact adjoint at the frozen prefix; the predicted RGB costate is obtained through the exact local-prefix VJP:

`lambda_hat_x = J_phi(x)^T (X_phi W)`.

The RFF rung appends one deterministic fixed sinusoidal lift. Because the nonlinear map is fixed and only `W` is fitted, the registered optimization remains convex and inherits the same spectral-scale contraction proof.

For target localization, a binary support projector `M` gives the exact identity

`cos(lambda, M lambda) = ||M lambda||_2 / ||lambda||_2 = sqrt(rho)`.

This reports where exact costate mass lies. It does not claim a learned direction outside `M`, and it does not reduce a dense exact SegNet VJP by itself.

## Tileability and routed sensitivity waterfill

The prefix cuts before the first squeeze-excite/global pooling operation, so the surrogate feature extractor is local and tileable. This composes with the routed three-tier margin × class-pair sensitivity waterfill as a future formulation. It does not contradict the settled exact-teacher wall: the exact SegNet has a 685-pixel dependency halo and 23 global squeeze-excite reductions.

The inherited `~4.7% area / ~97% d_seg` statement belongs to a specific annulus/island accounting. A separate committed bulk-boundary artifact reports `4.7365977%` area but `26.8038%` of flips. Neither is silently equated to this round's independently measured input-costate L2-square mass.

## FORE composition

FORE status is `NO_GO_CURRENT_INSTANCE__CONDITIONAL_FORMULATION_OPEN`. Correct weights would be `occupancy_current / occupancy_replay`, but the current 600 isolated states are not transition-complete and have no current-policy support receipt. No weighting is applied and zero calls are attributed to FORE. A stage-frozen, transition-complete successor remains queued.

## Economics law

The honest decomposition is

`C_teacher = A_label + V_validation + c_label D`.

Here `A_label=480`, `V_validation=120`, `D=7200`, and `c_label=0` under same-state cached reuse. Label-only amortization is `15x`; inclusive amortization is `12x`. Validation calls are charged but do not amortize training labels. Invalid pre-fit attempts are preserved and excluded from the admitted receipt; they remain visible as engineering custody, not formulation evidence.

## Measurement and custody

- Immutable receipt: `experiments/results/replace_round3_fidelity_wall_20260713/measurement_receipt.json`, `172738` bytes, SHA-256 `83704e64d1e5a70c00cf96c19330ff8453459e1024f957bceb48f99972157d75`.
- Completion: `2026-07-13T18:27:28.772018Z`; measurement-start git HEAD `31bb1e324fe7b4a649442b98c1f0ce4da06c8827`; deterministic CPU Torch/NumPy, one intra/inter-op thread, MPS absent.
- Exact inputs: GT cache SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`; SegNet SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`; checkpoint SHAs `2599ad8b...`, `c59cdec6...`, `1676e4d4...` in the receipt.
- Receiver parity: `MEASURED PASS`, zero differing elements at the representative state for all three checkpoints.
- Final admitted teacher ledger: `1800` rows = `600` starts + `600` completions + `600` unit batches; SHA-256 `676d25577e0cb70c55e9e0e5b4f3d1fdc35c61b3316a0ddcde3c2b9c0bb75c36`.
- Campaign accounting receipt: `campaign_teacher_call_accounting.json`, `2719` bytes, SHA-256 `51fcd984bfd93662e74d64e3ad577ed0e302097ed1b55b2c92075dbaead0b664`.
- Cleanup: 120 certified exact heldout caches, `283307760` bytes, deleted only after all reductions sealed; manifest SHA-256 `4116054ba3ca9ee12ea7931d6b585f61e31ff0d6a5b394b1b32b7fb9c2e0ad83`; blockers `[]`.

### Measured decision table

| Rung | Heldout metric | Preregistered gate | Result |
|---|---:|---:|---|
| local frozen-prefix linear | costate cosine `0.0016650255538056325`; rel-L2 `1.0000004372846978`; positive-dot `0.916667` | cosine `>=0.07078966932743762` and positive-dot `>=0.60` | **FAIL** cosine |
| fixed RFF prefix lift | costate cosine `0.0016791964165317613`; rel-L2 `1.0000003871015077`; positive-dot `0.916667` | same | **FAIL** cosine; best direction rung |
| source-margin support | mass `0.1634677541848741`; conditional exact cosine `0.40431145690528497` | mass `>=0.47` | **FAIL** |
| RFF log-mass ridge | mass `0.024426459564827255`; conditional exact cosine `0.1562896655727027` | mass `>=0.47` | **FAIL** |
| exact same-area oracle | mass `0.5278150212253758`; conditional exact cosine `0.72650878950318` | diagnostic, not deployable | family-open witness |

The RFF direction lift changes cosine by only `1.4170862726128799e-05` absolute (`0.8511%` relative) versus linear. Renderer-gradient cosine is larger (`0.0856221` linear, `0.0857091` RFF) but remains diagnostic: the primary exact input-costate gate fails by roughly `42.16x`.

### Convex-fit caveat

The realized linear/RFF Hessian operators have derived spectral contraction `gamma=0.3333333515` and `0.3333333858`, and every objective-gap ratio and residual bound passes. Do not overstate this as an every-iterate fp32 parameter contraction: maximum observed parameter ratios are `0.3336913636` and `0.3358835234`, exceeding their gammas by `0.0003580121` and `0.0025501377`. The fixed-feature heads remain convex; the fp32 trace needs that explicit rounding/noise caveat.

### Invalid-attempt custody

Three pre-verdict attempts are preserved and never admitted as formulation evidence: two one-call canary failures and a 143-completion/one-interrupted partial cache stopped before fit when cross-rung CPU grad-mode drift was found. Across them are `146` conservative training starts (`145` completed). They are why the campaign-honest economics are lower than the clean receipt's economics; no retry is hidden.

## Verdict scope and reformulation queue

**Verdict scope: `FORMULATION x INSTANCE`.** The negative covers this first-block local pre-SE prefix-adjoint target, its 42-column linear chart, its one seed455 16-frequency RFF lift, its log-mass ridge, fixed V9 n600 replay/split, and macOS CPU axis. It does not cover trainable nonlinear heads, direct top-k support classification, class-pair heads, globally conditioned deeper scorer features, transition-complete FORE, on-policy replay, other charts/seeds, or contest axes.

EV-ordered reformulation queue:

1. **Direct support-ranking target, not log mass regression.** The same-area oracle passes (`52.78% >47%`), proving useful support exists. Fit a weighted top-k/quantile or pairwise ranking target with explicit calibration; the present RFF log-mass head is worse than uniform.
2. **Margin × class-pair localization.** Source margin is informative (`3.48x` uniform) but captures only `30.97%` of oracle mass. Add the committed ~120x class-pair sensitivity spread instead of asking a class-agnostic scalar margin to carry it.
3. **Class-pair block heads on scorer features.** Preserve convex custody while preventing cancellation across Lane/Road/Undrivable regimes.
4. **Richer or deeper frozen scorer features with a new cost/tileability receipt.** The current cut is already immediately before the first SE. Any genuinely deeper scorer prefix inherits global SE state and must not reuse the local/tileable claim.
5. **Transition-complete stage-frozen FORE successor.** Collect explicit `(s,a,s')`, current occupancy, replay occupancy, and support-overlap custody before applying `occupancy_current/occupancy_replay`.
6. **On-policy nonlinear learner under a matched exact controller.** This is a new family/instance and owes its own stability and teacher-economics gates.

## Triality and system wire-in

- DSL: `tac.witness_dsl.replace_round3_fidelity_wall_policy`, typed default-off and `live_trainer_argv=[]`.
- Equation: `tac.canonical_equations.replace_round3_fidelity_wall_20260713`.
- DAG: `.omx/research/replace_round3_fidelity_wall_DAG_FEED_20260713.md`; shared hot DAG deferred to main review.
- Sensitivity: exact/predicted heldout costate and renderer-gradient reductions plus target-mass localization.
- Pareto: fidelity × exact-call count × prefix compute; no bytes or evaluator score measured.
- Bit allocator: non-binding; future compute waterfill reuses the canonical margin × class-pair sensitivity field.
- Cathedral/autopilot: no dispatch hook activated; research-only local probe.
- Continual learning: advisory scoped-KILL row `replace_round3_fidelity_wall_v9_n600_seed455_20260713` is registered; same-area oracle keeps the localization family open.
- Probe disambiguator: ordered linear/RFF/target modes in `tools/probe_replace_round3_fidelity_wall.py`.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`.
- v7.5 and v8 canonical specifications.
- `reports/latest.md`; lane registry; subagent progress; master gradient anchors; Modal call ledger; cost-band and continual-learning posteriors; probe outcomes.
- latest Codex findings/session summary, council T3, V9 design, and last-24-hour directives.
- committed round-2 memo/module/policy/receipt and settled #462 economics law.
- `.omx/research/fore_occupancy_ratio_dig_20260713.md`.
- `.omx/research/replace_round2_directive_tileable_architecture_20260713.md` and `.omx/research/cheapen_real95_directive_stratified_sensitivity_20260713.md`.

Lane maturity is L1 (`impl_complete=true`, `research_only=true`); real-archive, contest-axis, review-seal, memory, and deploy gates remain deliberately unclaimed.
