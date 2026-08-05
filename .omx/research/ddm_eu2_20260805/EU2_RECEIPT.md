# EU2 Receipt - Eureka Candidate #2

Answer: KEEP the 10K-parameter video-invariant comma10k/openpilot micro-student as a queued $0 successor experiment, but only for address ordering and token/context entropy reduction. Do not route it as a renderer replacement and do not route any scorer-slot work from EU2.

The 50K candidate is CONDITIONAL: it can only continue if the 10K cell shows a real scaling slope on cached ordering/context bytes. The 250K candidate is DEAD as the first cell for this line: its int8 packet costs 250,000 B, 0.166464738281 S, and is larger than the entire 90,000-155,000 B legal task-description corridor before any video-specific residuals.

This receipt is design/literature/arithmetic only. I ran no training, no scorer forwards, no launch, no archive exact eval, and no `upstream/evaluate.py`.

## Verdict

| candidate | counted-weight assumption | raw bytes | rate S | fraction of GP1 perfect-ordering ceiling | share of TR1 rehearsal tokens | share of current token stream | EU2 verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 10K int8 | 1 byte/param, framing omitted | 10,000 | 0.006658589531 | 9.35% | 2.00% | 2.89% | KEEP for first experiment |
| 10K fp16 | 2 bytes/param, framing omitted | 20,000 | 0.013317179062 | 18.70% | 4.00% | 5.77% | KEEP only if int8 not faithful |
| 50K int8 | 1 byte/param, framing omitted | 50,000 | 0.033292947656 | 46.75% | 10.01% | 14.43% | CONDITIONAL |
| 50K fp16 | 2 bytes/param, framing omitted | 100,000 | 0.066585895312 | 93.50% | 20.02% | 28.86% | DO NOT start here |
| 250K int8 | 1 byte/param, framing omitted | 250,000 | 0.166464738281 | 233.75% | 50.04% | 72.15% | DEAD for first cell |
| 250K fp16 | 2 bytes/param, framing omitted | 500,000 | 0.332929476561 | 467.49% | 100.08% | 144.31% | DEAD |

Rate arithmetic is DERIVED from `25 * bytes / 37,545,489`, so each counted byte costs `6.658589531e-7 S`. The GP1 perfect-ordering ceiling is MEASURED/DERIVED as 106,954 B = 0.07122 S. The TR1 rehearsal token stream was MEASURED as 499,587 B. The current hot-state token stream is MEASURED as 346,478 B.

## Why This Is Alive

The useful hypothesis is not "ship a tiny replacement SegNet." That is too strong and not supported by our priors. The useful hypothesis is narrower:

1. A public-data, video-invariant micro-student may learn the road/lane/object semantic prior that lets the receiver rank contested sites or choose token contexts better than local hand features.
2. The counted weight packet is cheap at 10K int8: it only has to save more than 10,000 token/context bytes, or recover more than 9.35% of the GP1 free-to-oracle address gap, to break even on byte economics.
3. PA2 already showed a no-new-scorer shared-context win of 3,975 B / 5.6405% on the persisted per-pair context stream. Projected onto the TR1 rehearsal token stream, the same fraction is 28,179 B; projected onto the current token stream, it is 19,543 B. A 10K int8 student can pay if it captures a modest increment over that context floor. A 50K student cannot pay on PA2-scale savings alone.

The best first use is therefore a context/orderer cell, not a pixel renderer and not a learned scorer surrogate.

## Why This Is Not Enough By Itself

The measured student prior is warning-heavy. Task #74's small learned student reached a best 8-pair 40KB advisory row at 46,248 B, `d_seg = 0.003436406434047967`, `d_pose = 0.00242719395646418`, `S = 0.5302298310635942`. The 60KB row was worse, at 71,278 B, `d_seg = 0.008823394833598286`, `d_pose = 1.4408006370067596`, `S = 4.725588555311784`. That is not a direct comma10k/openpilot test, but it kills any assumption that "small student" implies monotone fidelity or score movement.

Quantizr shows that an 88K-ish learned component can be packed into this contest's byte regime: its archive was 299,970 B, with a ~64KB `model.pt.br` from FP4 88K parameters. But Quantizr was a contest-clip trained joint frame generator with mask/pose sections. It is an existence proof for compact counted neural payloads, not evidence that a video-invariant comma10k student can replace the current trained renderer or tokens.

TR1/TB1 shows the renderer is not the current byte wall. In the EG1 rehearsal packet, archive bytes were 504,736 B and section bytes were:

| section | bytes | share |
|---|---:|---:|
| tokens | 499,587 | 99.0% |
| LOTTO renderer | 3,341 | 0.7% |
| selector | 535 | 0.1% |
| pose stub | 83 | 0.0% |

Current hot state is the same shape: 346,478 B tokens plus renderer, selector, and pose. A video-invariant student must attack token choice, context, or address ordering. Replacing the 3KB renderer cannot move the score enough.

## Legality And Rule-118 Boundary

| object | counted/free status | EU2 legality view |
|---|---|---|
| Generic student architecture and inference code | free in `inflate.py` | Legal generic algorithm, subject to runtime budget. |
| Weights trained only on public comma10k/openpilot-domain data | counted in `archive.zip` if shipped | Plausibly legal counted public-data prior. Must still survive contest-compliance review before shipping. |
| Weights distilled from frozen contest scorer outputs | counted in `archive.zip` if shipped | Governance-unclear because of the no-scorer-weights/no-scored-model-smuggling boundary. Requires operator ruling before any shipping path. |
| Weights trained on this contest clip or its scorer targets | counted in `archive.zip` if shipped | Legal only if fully counted and not hidden in code; not the EU2 "video-invariant" claim. |
| Per-frame outputs, lookup tables, masks, or token decisions derived from this video | counted in `archive.zip` | Not free code. Hiding them in `inflate.py` would be a fake implementation. |

The cleanest EU2 variant is trained on public comma10k/openpilot-domain data and then used as a counted semantic prior at decode time. A scorer-distilled variant is not a first move; it needs an explicit compliance ruling.

External source check, used only for data-source viability: `YassineYousfi/comma10k-baseline` is a public PyTorch/Lightning baseline for comma10k, and `commaai/comma10k` is the public comma10k semantic driving dataset. This does not create scorer authority; it only confirms that a public-domain training source exists.

## Economics Table

| measured prior | value | implication for EU2 |
|---|---:|---|
| GP1 free-to-oracle ordering gap | 106,954 B = 0.07122 S | 10K int8 needs 9.35% of this gap; 50K int8 needs 46.75%; 250K int8 cannot pay as an orderer. |
| PA2 Brotli shared-context win | 3,975 B / 5.6405% on persisted stream | Projects to 28,179 B on EG1 tokens and 19,543 B on current tokens; enough to justify 10K, not enough for 50K. |
| PA2 LZMA raw shared-context win | 1,067 B / 1.5461% | Projects to 7,724 B on EG1 tokens and 5,357 B on current tokens; below 10K unless the student adds information. |
| EG1 rehearsal token section | 499,587 B | The student should target token/context bytes, not the 3.3KB renderer. |
| Current hot-state token stream | 346,478 B | 10K int8 costs 2.89% of this stream; 50K costs 14.43%. |
| GC18 legal task-description corridor | 90,000-155,000 B | 10K is 6.5-11.1% of the corridor; 50K is 32.3-55.6%; 250K exceeds the corridor alone. |
| #74 best small-student row | 46,248 B, S 0.530229831064 advisory | Shows compact student training can work partially but remains far from a score row. |
| Quantizr model packet | ~64KB for 88K FP4 params | Shows compact counted learned payloads are plausible, but not that video-invariant weights solve this vehicle. |

## RECALL EVIDENCE

Queries and sources consulted:

| recall path | search/query | what it changed |
|---|---|---|
| Memory registry | `rg -n "eu2|codex_runs|common_contract|frontier|charter" /Users/adpena/.codex/memories/MEMORY.md` | Enforced scoped absence language and kept advisory/frontier separation explicit. |
| Governing docs | `PROGRAM.md`, `AGENTS.md`, `CLAUDE.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Confirmed no-fake, rule-118, scorer-slot, serializer, advisory-axis, and current pointer constraints. |
| Distillation prior | `.omx/research/distillation_smaller_student_20260610T191237Z.md`, `.omx/research/task74_distill_candidate_rows_20260610.json` | Converted "student may help" into a warning: small-student fidelity is unstable and non-monotone. |
| GP1/GC16 student pricing | `.omx/research/ddm_gp1_selective_gt_student_pricing_20260803.md`, `.omx/research/ddm_gc16_from_here_20260803.md` | Reframed the candidate as address/context ordering, not scorer replacement. |
| TR1/TB1 bytes | `.omx/research/ddm_tb1_renderer_build_20260728.md`, `.omx/research/ddm_eg1_tr1_rehearsal_20260728.json`, `.omx/research/ddm_sub1_canonical_submission_chain_20260804.md`, hot state | Located the byte wall in tokens, not renderer weights. |
| PA2/PK1/GC18 current side evidence | `.omx/research/ddm_pa2_20260805/PA2_RECEIPT.md`, `.omx/research/ddm_pk1_20260805/PK1_RECEIPT.md`, `.omx/research/ddm_gc18_20260805/GC18_CONVOCATION_RECEIPT.md` | Set the context-savings floor, legal corridor, and warning that byte-positive grammars can be scorer-negative. |
| Quantizr | `docs/quantizr_archive_layout_confirmation_20260504.md`, Quantizr section in `AGENTS.md` | Supplied a compact-neural-payload precedent while blocking transfer of clip-trained renderer evidence to a video-invariant student. |
| Canonical equations | `tools/list_canonical_equations.py --json` filtered for `distill`, `student`, `surrogate`, `hinton`, `token`, `context`, `rule118`, `openpilot` | Found adjacent laws for distilled surrogates and token/context waterfill, but no direct measured comma10k micro-student law. |
| External public-data check | web search for `YassineYousfi/comma10k-baseline` and `commaai/comma10k` | Confirmed public-data source plausibility only; no contest authority. |

I did not find, in the searched scopes above, a measured byte-closed row for a comma10k/openpilot-trained video-invariant micro-student on the current TR1/qo1 vehicle. This receipt is therefore a pricing verdict and experiment design, not an adoption verdict.

## Decisive First Experiment - Do Not Run In EU2

Name: `EU2-X1-10K-context-orderer`.

Purpose: price whether a 10K int8 public-data semantic prior can buy more token/context/address savings than it costs, without consuming the scorer slot.

Data:

- Training data: public comma10k/openpilot-domain images and masks only. No contest clip frames, no contest scorer outputs for training.
- Evaluation data: existing cached n600 receiver frames and cached GP1/order-margin surfaces only. No new scorer forwards.
- Controls: a 1KB named-feature logistic/Rudin-style head and the PA2 shared-context baseline.

Candidate:

- Primary: <=10K parameters, int8 export first, fp16 only if int8 fails parse-back fidelity.
- Inputs: receiver-available RGB/YUV, row/col, gradients, decoded class field, boundary distance, and token/context state. No hidden per-frame table in code.
- Output: order/context features for contested sites or entropy contexts. Not pixels, not final masks, not a score.

Artifacts required:

- Real packed `archive.zip` section for the weight payload with bytes and sha256.
- Rank-agreement JSON against cached oracle ordering/margin fields.
- Token/context byte delta under the same coder used for the current token stream.
- Parse-back proof that the receiver consumes the packet exactly once.

Wall-clock bound:

- Cached n600 inference/coder pass: cap at 30 minutes. This is a DERIVED cap from the heavier TB1 T2 n600 40-epoch TR1 renderer run, which was 1,477 seconds for the LOTTO cell; the micro-student pass should be cheaper, but the cap is the pre-registered stop.
- Public-data training smoke: cap at 30 minutes for the first <=2,000-image split. If the 10K model cannot finish a bounded smoke under that cap, record NO-GO for this immediate route and do not escalate to 50K.

Pre-registered GO/NO-GO:

| result | condition |
|---|---|
| GO 10K | Actual packet <=15,000 B and either recovers >=30% of GP1's 106,954 B ordering gap, or saves >=30,000 token/context bytes against EG1/current-equivalent coding, with deterministic parse-back. |
| WEAK GO | Packet <=15,000 B and recovers 15-30% of GP1 gap, or saves 15,000-30,000 bytes. Queue one width/feature ablation only; no scorer slot. |
| NO-GO | Packet >15,000 B, or recovers <15% of GP1 gap, or saves <15,000 bytes, or requires contest-clip/scorer labels to work. Fold EU2 candidate #2 as an orderer. |
| ESCALATE 50K | Only after GO 10K plus a measured slope showing extra capacity buys at least 1.2 saved bytes per added counted byte. |
| KILL 250K | Remains killed unless a later measured route shows wholesale token replacement exceeding 250,000 B saved before any scorer claim. |

No scorer-slot escalation is authorized by EU2. If a later owner wants scorer authority, the queued artifact must first pass the no-scorer cached economics bar above, then request a lane under the scorer-slot rule.

## Attack On This Conclusion

The KEEP verdict could still be wrong in three ways:

1. Public comma10k masks may not align with frozen SegNet's private boundary/margin behavior, so zero-shot ordering on this video may be weak.
2. The student may save entropy bytes while making the wrong errors at high-score pixels; PK1 already showed byte-positive representation can be scorer-negative.
3. The compliance line around scorer-distilled students remains unresolved. A public-mask prior is cleaner than a frozen-scorer surrogate, but no shipping path should treat this as settled law.

The receipt therefore keeps the arm only as a cheap cached-ordering test. It does not promote a score, a route, or a launch.

## NEXT_IF_RESUMED

1. Build `EU2-X1-10K-context-orderer` only if rz1 has no claim on the same cached/coder surface and only as a no-scorer, no-launch experiment.
2. Start with the 1KB named-feature control and 10K int8 student. Do not start with 50K.
3. Measure actual packed bytes and cached ordering/context savings before any training expansion.
4. If GO 10K passes, queue an operator compliance question for public-data weights vs scorer-distilled weights before any ship path.

```json
{
  "receipt": "ddm_eu2_20260805",
  "candidate": "eureka_candidate_2_video_invariant_comma10k_micro_student",
  "axis": "pricing_and_experiment_design_only",
  "ran_training": false,
  "ran_scorer_forward": false,
  "ran_evaluate_py": false,
  "primary_verdict": "KEEP_10K_CONTEXT_ORDERER_ONLY",
  "candidate_verdicts": {
    "10k_int8": "KEEP_FOR_QUEUED_NO_SCORER_EXPERIMENT",
    "10k_fp16": "KEEP_ONLY_IF_INT8_FAILS_FIDELITY",
    "50k_int8": "CONDITIONAL_ON_10K_SCALING_SLOPE",
    "50k_fp16": "DO_NOT_START_HERE",
    "250k": "KILL_FOR_FIRST_CELL"
  },
  "rate_per_byte_s": 0.0000006658589531,
  "gp1_perfect_ordering_gap_bytes": 106954,
  "gp1_perfect_ordering_gap_s": 0.07122,
  "tr1_rehearsal_token_bytes": 499587,
  "current_hot_state_token_bytes": 346478,
  "decisive_experiment": "EU2-X1-10K-context-orderer",
  "go_bar": {
    "max_packet_bytes": 15000,
    "ordering_gap_recovery_fraction": 0.30,
    "token_context_bytes_saved": 30000
  },
  "requires_operator_compliance_ruling_before_shipping": true,
  "follow_on": "QUEUED-WITH-FIRE-ORDER"
}
```

Follow-on status: QUEUED-WITH-FIRE-ORDER. Fire order is cached economics first, then compliance ruling, then scorer-lane request only if both pass.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
