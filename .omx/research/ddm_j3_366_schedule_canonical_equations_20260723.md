---
equation_id: ddm_j3_366_full_run_schedule_v1
utc: 2026-07-23T03:25:00Z
status: research_only
score_claim: false
---

# DDM J3 #366 full-run schedule law

Let `N=600` pairs and measured training batch `B=4`. One complete exposure is

`P = ceil(N/B) = ceil(600/4) = 150 steps`.

The three inherited energy-force admissions each receive one complete exposure, so

`K_stage=P=150`, and `K_total=3P=450`.

The periodic crash-loss bound is one quarter-exposure:

`C = floor(P/4) = 37 steps`,

with every stage boundary separately and immutably checkpointed. Live parameters, EMA shadow, Adam first/second moments, exact run cursor, verdict history, and realized archive identity are all saved atomically.

The v16 receiver measurement invalidates first-order extrapolation beyond one uint8 realization quantum. Therefore

`q_lr = 1/4 quantum`, `q_line_search in {1/4, 1/8, 1/16}`,

and the inherited pair-447 warm-up duration is

`K_warm = ceil(1/q_lr) = 4 steps`.

No linearized plateau can transition a stage. Stage decisions consume only exact candidate archive parse-back through paint, uint8, `R`, and frozen scorers. A stage may advance on a realized target, a realized two-verdict no-descent plateau, or one complete exposure. Any component regression checkpoints and blocks the campaign before stage advance.

With measured `t_step in [100.87168033304624, 104.09510249993764] s`, chunked n600 verdict `t_V in [248.62050645798445, 276.30817591701634] s`, and measured startup `t_0=51.49326074984856 s`, the preregistered ten-verdict campaign band (one baseline plus three third-exposure/stage-exit decisions per stage) is

`T = K_total*t_step + 10*t_V + t_0`,

which yields `T in [13.31387624311125, 13.79371420691443] hours`.

The memory projection law is

`M_projected = max(1.2*M_measured + 1 GiB, M_measured + 2 GiB)`.

For `M_measured=10.47418212890625 GiB`, `M_projected=13.5690185546875 GiB`, safely below the operator ceiling of `116 GiB`. This admission does not override the separate exact-efficacy blocker.

Value provenance: `N`, targets, and force ordering are inherited preregistration; timings and peak RSS are MEASURED actual full-run values; step counts, checkpoint interval, quantum bounds, wall-clock band, and memory projection are DERIVED. No constant is guessed.
