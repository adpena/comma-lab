# ddm_m1r3 - science review of sealed M1 n120 receiver ticket

Tags: [no-triality] [p0-ledger-ok]
Axis: [macOS-CPU advisory / macOS-MLX research-signal / apparatus-source review]
Scope: read-only review, no Metal, no launch, no scorer, no ticket mutation.
Recommendation: CLEAN_PASS_3_OF_3

## Answer First

M1R3 is CLEAN on the science lens. I found no finding that should reset the
sealed-ticket counter from this pass.

Concurrency caveat: M1R3 and M1R2 review the same unchanged sealed state. If
either pass reports a finding, both passes are void and the 3-clean-pass counter
resets to 0 because any cure changes the reviewed artifact.

This pass measured no new score and moved no pointer. It reviewed the sealed
M1 ticket, receipts, and current recall surfaces only.

## Provenance Pins

| artifact | sha256 |
|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `dafb9e7792564de2ad3b68875792c5b4e75eb91b72360e99db78d6bed441fd47` |
| `.omx/research/ddm_m1_20260808/sigma/sigma_harvest_receipt.json` | `bfdd921982eef458b90c53567bb60abffacff68a558f7a31cee838629663fe6d` |
| `.omx/research/ddm_m1_20260808/run/n120_metal/mem_probe/mem_probe_receipt.json` | `91ad0bee7e16827205b5baff82de9087b261aec74df49f01f7e377cb59709ef9` |
| `.omx/research/ddm_gc21_20260808/GC21_CONVOCATION.md` | `15f6d2febc23e7eb779ebaa93d902d7470aec612a9a4c6bba54cd9f6de1d06ee` |
| `.omx/research/ddm_ng1_20260808/NG1_CROSSWALK.md` | `26f76a4496ad55a2a15af69889f0a60d28a724c372c4f94ea59375b96ce28845` |
| `.omx/research/ddm_wc3_20260808/WC3_FINDINGS.md` | `89810e8e9d27d46b8b1e99f18bbc78a784231ac601efaf0b964018d3e5dd207b` |
| `.omx/state/main_hot_state.md` | `ba1236bc7c8231de562e8db37e6468bd19307570ca7d807f17f03c9cb546fa0c` |

## RECALL EVIDENCE

| scope | query / artifact | found beyond charter seeds | plan impact |
|---|---|---|---|
| Governing files | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/tmp/codex_runs/_common_contract.md`, `.omx/state/main_hot_state.md` | The live board pins own-vehicle frontier `S=0.7534578126155775 @ 357,837 B`, PR130 gap dominated by seg, and M1 v5 as `COMPOSED_UNSEALED` with remaining review/guard/fire sequence. | Kept this as a read-only review, no scorer, no launch, own-vehicle frontier line at end. |
| Charter-subject receipts | Opened ticket, sigma receipt, mem-probe receipt, GC21, NG1, WC3. | The ticket now contains the mem-probe and sigma fields that hot state listed as remaining before those receipts landed. | Review uses current ticket receipts, not stale hot-state gate order. |
| Corpus map | `rg` for `#984`, `PR130`, `n120 receiver`, `HPAC`, then opened `ddm_map1_competition_design_space_20260808.md`, `ddm_plan15_20260808.md`, `ddm_lx1_20260808/*`, `ddm_hb2_20260808/HB2_FINDINGS.md`. | HB2 added a fresh measured HPAC exact-label component: tq1c labels model+tokens `112,044 B`, exact decode, score_claim=false. LX1 ranks receiver discriminator first and HB2 label bytes second. | Strengthened the composed vehicle case while preserving the rule that semantic bytes cannot become score before receiver consumption. |
| Task/ledger surfaces | `rg` over `.omx/state/canonical_task_status.jsonl`, operator ledgers, map1, and LX1 for `#984/#982/#978`, `HPAC`, `token family`, `receiver discriminator`. | Did not find a better live Metal-window consumer than the n120 receiver discriminator in searched task/status scopes. HPAC and coder alternatives are byte-only or already running independently. | Counterfactual compares M1 mainly against the renderer-backward kernel and HPAC/rate path, not against a hidden active scorer row. |
| Canonical equations | `tools/list_canonical_equations.py --json` filtered for `trajectory_derived_stopping_law_v1`, `ddm_rr9_mem_probe_fire_protocol_v1`, `ddm_rr8_stage_rc_success_contract_v1`, `score_marginal_lagrange_multipliers_v1`, `witness_fp_reorder_transform_bit_identity_wall_v1`, `ddm_hb1_semantic_label_incumbent_transfer_v1`. | The relevant laws are safety-cap-not-convergence, required Metal mem-probe receipt, stage-rc honesty, score marginal arithmetic, fp-order caution, and HPAC transfer requiring target-payload training plus exact decode equality. | M1 is admissible as a guarded trajectory burn, not an exact-score claim or HPAC shortcut. |
| Source checks | `experiments/ddm_mx1_pr130_semantic_renderer.py`, `tools/mx1_fire_guard.py`, `src/tac/pr130_lift/mlx_semantic_renderer.py`, `src/tac/optimization/trajectory_stopping.py`. | Source matches the ticket: GPU MLX train refuses without in-process guard; resume checks pair IDs; checkpoints are atomic NPZ writes; trajectory stopping reports safety bounds separately from convergence. | No waste-mode finding from source review. |
| Dirty worktree | `git status --porcelain=v1`, `git diff --cached --name-only`. | Many unrelated dirty/untracked files exist; no staged index entries were present before my edit. | I touched only `.omx/research/ddm_m1r3_20260808/M1R3_REVIEW.md`. |

## Per-Question Verdict Table

| question | verdict | deciding evidence | what would have falsified it |
|---|---|---|---|
| 1. Right burn | CLEAN | M1 attacks the seg axis, the dominant live gap: hot state gives PR130 gap about `0.5813`, with seg about `0.4010` and still majority (`.omx/state/main_hot_state.md:19-20`). Map1 places PR130 in the semantic-carriage cluster and says the #984 vehicle is semantic carriage plus trained receiver plus pose (`ddm_map1...md:35-37`, `58-62`). VEH/CAP shows gt-to-gt capacity at `d_seg=0.0010862350` while decode-side VEH is flat, so the next object is receiver capacity on good semantic tokens (`VEH_CAP_N32_VERDICT.md:20-50`). | A better live Metal-window object with measured higher expected score value, or evidence that M1 targets rate/pose while seg remains binding. |
| 2. Optimal form | CLEAN | The ticket uses PR130 w96 init, GT labels, n120 stratified sample, saturated throughput flags, dense eval, checkpoint cadence, CPU-torch authority, and event stop. LR `2e-7` is source-verified to PR130 stage-08 but explicitly caveated for batch geometry; the ticket records `(accumulated-batch, 2e-7)` as n32 measured-descending and names lr as an EXTEND lever if trajectory crawls (`launch_ticket...json:668-678`). | Treating `2e-7` as globally optimal, omitting the batch-geometry caveat, using prefix/n8 evidence, or removing the EXTEND/lr fallback. |
| 3. Interpretability | CLEAN | The ticket labels axis as MLX train telemetry with CPU-torch d_seg authority and no contest claim (`launch_ticket...json:508`). It records stratified n120, not prefix (`launch_ticket...json:666`), sigma/fp16 falsifiers (`sigma_harvest_receipt.json:11-32`), CPU-torch facets on checkpoints, and per-eval fields (`launch_ticket...json:794-798`). | A plateau being written as family death, a descent being written as n600/exact score, or gate rows reading a different objective basis than the stop predicate. |
| 4. Counterfactual | CLEAN | Highest alternative Metal-window use found is renderer-backward kernel work from WC3. It targets a real bottleneck: renderer backward is 57% of fp32 graph time (`WC3_FINDINGS.md:22-31`). But GC21 and Plan15 both route it after-first-burn or mid-burn because it is a throughput lever, not the first receiver trajectory (`GC21_CONVOCATION.md:186-208`, `ddm_plan15_20260808.md:67-75`). HPAC/rate rows are byte-only or already independent: HB2 gives `112,044 B` exact tq1c labels but no scorer and unchanged frontier (`HB2_FINDINGS.md:5-10`, `100-101`); map1 says gt-HPAC is live separately (`ddm_map1...md:101-109`). | A parity-clean renderer-backward kernel already ready before seal with same-object whole-step speedup, or a Metal-dependent rate/pose action that produces a conditioned receiver base sooner than M1. |
| 5. Waste modes | CLEAN | Unrecoverable crash: checkpoint every 250 and latest checkpoint are in argv and source (`launch_ticket...json:101-104`, `678`; `experiments/...py:3139-3220`; `mlx_semantic_renderer.py:358-400`). Silent freeze/inert lever: eval every 50, event predicate, sigma envelope, and CPU-torch facet checks are recorded (`launch_ticket...json:769-793`, `711-766`). Guard refusal: mem-probe receipt status passed and receipt carries saturated config (`mem_probe_receipt.json:1-24`, `1847-1850`); GPU train re-runs the guard in-process and refuses mismatch (`experiments/...py:3822-3897`). Byte-close limitation: ticket does not claim byte-closed archive and labels `score_claim=false` (`launch_ticket...json:703-704`). | Missing `--resume-from` consumption, stale old-schema mem-probe accepted, fixed 3250 stop written as convergence, or absent CPU-torch checkpoint facets. |
| 6. Other science lens | CLEAN | The strongest new recall item, HB2, makes semantic carriage more plausible but also sharpens the dependency: bytes need receiver consumption. LX1's top queue ranks the receiver discriminator first, HB2 bytes second (`LX1_CROSSWALK.md:201-217`). | Promoting HB2 or #984 component arithmetic as a score before receiver/pose/archive closure. |

## Arithmetic And Objective Fit

The burn attacks the right score derivative. At the current own-vehicle line,
`d_seg=0.004305420`, while PR130's bar uses `d_seg=0.000296600`. The remaining
seg swing to PR130 is:

```text
100 * (0.004305420 - 0.000296600) = 0.400882 S
```

That is consistent with the live board's `seg 0.4010` gap statement
(`.omx/state/main_hot_state.md:19-20`) and LX1's component skeleton
(`LX1_CROSSWALK.md:67-71`). M1 does not itself move composed `S`, because it
does not build the HPAC stream, pose leg, archive, or exact eval. Its expected
endpoint case is therefore DERIVED, not measured:

- If the n120 receiver approaches the n32 CAP shape (`d_seg=0.0010862350`),
  the receiver seg term is about `0.1086235`, which would remove about
  `0.3219185 S` from the current own-vehicle seg term before rate/pose/archive
  accounting.
- If it reaches the PR130 renderer class floor (`d_seg=0.000296600`), the
  whole `0.400882 S` seg gap closes.
- If it plateaus, the conclusion is INSTANCE/CONFIG: this exact n120
  saturated, PR130-init, accumulated-batch, lr-2e-7 trajectory plateaued. It
  does not kill the receiver family or semantic-carriage vehicle.

That is the right burn because seg is the binding axis and M1 buys the first
interpretable receiver trajectory on that axis.

## Optimal-Form Notes

The only science risk that needed adversarial treatment is the `2e-7` learning
rate under changed batch geometry. I do not make it a finding because the ticket
does not hide the transfer: it records the PR130 source provenance, batch-size-2
caveat, n32 measured-descending support, and an EXTEND lever if the trajectory
crawls (`launch_ticket...json:668-678`). This is acceptable for a first
event-driven burn. It would become a finding if the final interpretation says
"receiver family plateau" rather than "this lr/batch/schedule instance
plateaued."

The saturated throughput config is optimal-form for this vehicle at the current
evidence level: WC3 measured saturated at `7.659 s/step` versus `10.421`
baseline on n32, with d_seg sanity inside the observed schedule spread
(`WC3_FINDINGS.md:62-88`). The mem-probe remeasured the n120 saturated memory
surface at the real config and passed (`mem_probe_receipt.json:1-34`,
`1847-1850`). The sigma receipt showed bit-identical fp16 repeat checkpoints and
zero d_seg delta between fp16 and fp32 checkpoint verdicts at the calibration
horizon (`sigma_harvest_receipt.json:3-9`, `44-79`). These are sufficient for
admission, not a proof of final-horizon fp16 equivalence; the ticket preserves
that boundary.

## Outcome Interpretability

When M1 ends we can conclude:

- CLEAN descent: this n120 stratified receiver trajectory improved the
  scorer-facing d_seg objective under the ticketed saturated config, with
  checkpoint/facet receipts deciding whether to extend or harvest.
- CLEAN plateau: this exact instance did not buy enough d_seg at the measured
  lr/batch/schedule and should route to the pre-declared EXTEND/lr/config
  branch, not family death.
- CLEAN guard refusal: the ticket did its job and the fire should stop until
  the receipt/config mismatch is cured.

We cannot conclude:

- no n600 result,
- no contest-CPU/CUDA score,
- no byte-closed archive,
- no pose/rate composition,
- no public leaderboard claim,
- no family-level kill from a single n120 trajectory.

## Counterfactual Pricing

Concrete alternative: spend the same approximately 12h Metal window on the
renderer-backward custom kernel line routed by WC3/wc2.

M1 side:

- Cost: approximately 12h buys about `12*3600/29.3218 = 1473` n120 saturated
  training steps at the mem-probe measured `29.3218 s/step`
  (`mem_probe_receipt.json:1934`), plus eval/checkpoint receipts.
- Value: directly observes the first stratified n120 trajectory on the
  `0.400882 S` seg gap.
- Decision value: feeds extend/stop/retreat and conditions pose/rate follow-ons.

Kernel side:

- Cost: unknown implementation plus parity plus whole-step bench plus review.
- Best measured target: renderer backward is `0.442 s` of `0.775 s` fp32 chunk
  graph time, 57% of graph time, while about 40% of whole-step time was outside
  the loss graph (`WC3_FINDINGS.md:19-31`).
- Value: throughput only. It does not create the first receiver trajectory, and
  GC21 says it wins before-burn only if already parity-clean and bench-clean
  before seal with at least `1.25x` additional end-to-end n120 speedup
  (`GC21_CONVOCATION.md:197-208`).

Verdict: M1 wins the window. The kernel remains a good mid-burn/checkpoint
adoption candidate after parity and bench proof. HPAC/rate alternatives do not
consume this Metal window and are already routed separately.

## Waste-Mode Audit

| waste mode | protection observed | verdict |
|---|---|---|
| Unrecoverable crash | Distinct `mlx_stage_step%06d.npz` plus `mlx.latest.npz`; source writes atomic NPZ tmp then replace; resume load checked at end (`experiments/...py:3139-3220`, `mlx_semantic_renderer.py:358-400`). | CLEAN |
| Silent freeze / inert lever | Dense eval every 50 and event stop predicate; stop law uses `continue_projected`, `marginal_below_bar`, `converged_projected`, `safety_bound_REPORTED` (`launch_ticket...json:769-793`; `trajectory_stopping.py:22-27`). | CLEAN |
| ep_loss=0 / bad telemetry class | Ticket records per-eval JSONL fields: step, wall, loss, lr, d_seg, best d_seg, checkpoint path and sha (`launch_ticket...json:794-798`). This review did not launch, so no live log checked. | CLEAN with launch-time monitor owed |
| Guard refuses mid-run | GPU entrypoint re-evaluates guard and refuses malformed/mismatched verdicts (`experiments/...py:3822-3897`); guard returns refused rc if status is not passed (`tools/mx1_fire_guard.py:512-527`). | CLEAN |
| Checkpoint cannot be byte-closed | Stage checkpoints are loadable MLX checkpoints and CPU-torch verdict inputs, not archive closure. Ticket makes no byte-closed archive claim (`launch_ticket...json:703-704`). | CLEAN boundary |

## Findings

None.

## Recommendation

CLEAN_PASS_3_OF_3

If M1R2 also returns clean on the unchanged sealed state, MAIN may count this
as the science pass for the 3-clean-pass seal and proceed to the final guard
gate. If M1R2 finds anything, this pass is void with it.

## Frontier Line

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
Contest pointer remains borrowed/unmoved at `0.19108`.
