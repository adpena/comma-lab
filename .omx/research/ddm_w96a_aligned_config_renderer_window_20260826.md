---
arm: ddm_w96a_aligned_config_renderer_window
utc: 2026-08-26
status: BLOCKED_QUEUED_WITH_FIRE_ORDER
axis: "[macOS-CPU receipt replay + storage preflight; no aligned training, scorer, Metal, n600, Modal, or contest evaluation]"
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: "NO CONFIGURATION VERDICT; the aligned W96 instance remains unmeasured"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_w96a — aligned W96 window is still unmeasured; the canonical SSD cannot retain the two-seed experiment

## Answer first

**Disposition: QUEUED-WITH-A-FIRE-ORDER.** I did not launch an aligned W96 window. The required
two-seed, 65-epoch, keep-every-payload experiment projects to **36,931,633,152 B** from the completed
same-form 65-epoch seed. The mandatory APDataStore tier had **22,319,071,232 B free** at preflight.
Even with no safety reserve it is short by **14,612,561,920 B**; with the 8 GiB reserve used here it
is short by **23,202,496,512 B**. Vertigo had only **8,980,893,696 B free**. The common contract says
to stop when the SSD preflight fails, and forbids a fallback symlink. No local-disk fallback, payload
discard, cleanup, scorer call, Metal call, n600 leg, or training launch occurred.

This is **not** a negative on the configuration hypothesis. There are zero aligned-config W96 seed
rows, so neither the `>=5x` LIVE branch nor the `<2x at >=2 seeds` closure branch fired. The exact
contest pointer remains **S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]**, archive SHA
`ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.

## What was measured

I re-ran the chartered S1E instrument against the already-retained OFF-config W96 payloads. It
reproduced **35 fixed evenly-strided n60 checkpoint rows** across both seeds, all
`[Darwin-mps frozen-scorer advisory]`, and again found zero corner crossings. This replay did not run
a scorer; it verified and recomposed the retained scorer and receiver receipts.

| configuration | seed | retained rows | named row | composed delta vs renderer break-even | hard d_seg | d_pose | selected renderer packet |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | 20260815 | 14 | best/ep65 | +0.1554134085557307 | 0.0008149888599291444 | 0.0009353572968393564 | 38,847 B |
| OFF | 20260816 | 21 | ep65 | +0.19887521337211275 | 0.0007870144327171147 | 0.0020441312808543444 | 38,847 B |
| OFF | 20260816 | 21 | best/ep75 | +0.14815737243836 | 0.0008060031686909497 | 0.000816427986137569 | 38,847 B |
| **ALIGNED** | 20260815 | **0** | not launched | **UNMEASURED** | — | — | — |
| **ALIGNED** | 20260816 | **0** | not launched | **UNMEASURED** | — | — | — |

The ep75 row is reported only as the best observed OFF row with its 21-evaluation selection count;
the chartered equal-window comparison remains ep65. No ratio is computed against an absent aligned
row. The re-run receipt is 296,698 B, SHA
`3037d264f097cd1b239cd96fc2302f5d812e0f3384eea7813fdc5cb074b60b18`.

## Aligned configuration derivation

This is the sealed intended configuration. It is a transfer prescription, not a claim that the
current WD3 code already implements it.

| constant / invariant | sealed value | source receipt and status |
|---|---|---|
| Renderer form | `film_amortized_flat_w96`, four full pointwise blocks, Film W96 | RJ1 archive 179,290 B, SHA `34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21`; `SOLE_ADMISSIBLE_RJ1_W96_FORM_FOR_S1_BUILD` in the S1 receipt. SVD-r32 remains forbidden. |
| Strict initializer | 253,955 B, SHA `e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434` | `ddm_s1a_stage_a_adapter.py` and the retained RJ1 initializer; both seed births must strict-load this same state while preserving distinct RNG states. |
| Seg objective allocation | 100% expected-flip margin; no CE and no softplus phase | CE1: `ce_fraction=0.0`, `softplus_fraction=0.0`; the source selector therefore enters `expected_flip` for the complete window. This is the measured best-aligned one of the three CE1 objectives, not a proven optimum. |
| Seg law | `100 * mean(sigmoid(-(z_target - max(z_other))/tau))` on the WD3 selected real scorer cells | The lifted oracle defines the exact margin and expected-flip law; coefficient 100 is the contest d_seg coefficient. Current WD3 instead uses calibrated `1-p_target`, so a reviewed implementation is required before a launch can truthfully be called aligned. |
| Tau schedule | linearly 0.15 -> 0.05 over normalized full-window progress | Exact lifted-oracle source law, SHA `ffdf098801863ff8bffe8bd818ce101928dd75b4937cbbffb2e225bddbc12f4b`. CW1 established that this fraction-of-window schedule was inherited and unswept; it is a reference constant, not optimal-form evidence. |
| Peak LR | `2e-5` reference rung | CE1 EF3000/EF6000 measured descent at this value. CW1 later measured `6e-5` weaker and `1e-5` null, so UP is dead and `2e-5` is the reference plateau value. It is still borrowed across architectures. |
| LR schedule | cosine over the complete window, `eta_min = 0.01 * lr` | Exact CE1 trainer source. Do not use WD3's current `0.02 * lr` while calling the transfer exact. |
| Pose gate | `sqrt(10 * MSE(Pose6_student, Pose6_original))`, coefficient 1, active from step 0 | Current WD3 real-scorer loss and JF2 addendum. There is no delayed pose phase; pose supplied at least 93% of every prior diagonal death. |
| Preservation constraints | retain current WD3 teacher-margin, teacher-KL, decode-MSE, and teacher-pose adaptive constraints | Reference-form transfer; removing them would be a mechanism reduction and would not test the chartered family. |
| Packet path | real packet quantizer in-loop; `cheap_to_shrink=off` for the first aligned comparison | Same-form single-treatment comparison. A shrink treatment would confound objective allocation with representation. |
| Seeds | 20260815 and 20260816, separately reported | The S1A apparatus already cured #1251: identical strict-loaded birth weights, distinct retained RNG states. This contradicts the charter's stale “seed has never varied” premise for the OFF campaign, but aligned training still has zero seeds. |
| Window | 65 epochs maximum, batch 1, sequential, watched | Charter cap and S1A reference window. Cheap checkpoints first; no early result becomes a family verdict. |
| Checkpoints / screens | atomic full-state checkpoint and retained n60 screen at epoch 1, every 5 epochs, and stage end | Existing WD3/S1A P0 path; checkpoint carries live weights, EMA, optimizer, scheduler, generator, controller, allocation, selection binding, config, and history. |
| Screen population | IDs `0,10,...,590`, fixed evenly-strided n60 | S1E comparability only. Advisory screen, never a population negative or score. |
| Full sampled leg | all n600, chunks <=120, only after a seed checkpoint moves the screen at least 5x | Charter gate plus common single-flight contract. This arm does not own the n600 scorer slot, so the leg is queued for MAIN rather than fired. |

## Apparatus readiness

- **Crash resume: FOLDED.** Commits `a5fd9ace0b` and `bfa780756e` mask only `resume_from` and the
  self-referential builder hash in the resume identity while retaining source pinning. The focused
  regression suite passed **7/7** in this run.
- **Two actual seeds: FOLDED.** The retained S1A births already prove identical initialized weights
  and distinct generator states for 20260815 and 20260816.
- **Aligned objective in WD3: QUEUED-WITH-A-FIRE-ORDER.** Source SHA
  `662acc06adf110840f077c65526e9696f375b35f72e305da59d1d06cec35b758` has only calibrated
  soft disagreement, not the CE1 target-margin expected-flip law. No config-only edit can make the
  current trainer execute that law.
- **Storage: QUEUED-WITH-A-FIRE-ORDER.** The retained blocker receipt is 1,870 B, SHA
  `4e62de066d3304e970a20002aa08400789015f9d26d69f143d82f4227f3e1056`.
- **Lanes: not claimed.** The coordination ledger reported zero active claims at the 24-hour TTL.
  Claiming lanes before storage and implementation were green would create phantom authority.

## Fire order

1. **QUEUED-WITH-A-FIRE-ORDER — storage unblock.** Owner: MAIN/operator. Consumer store:
   `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`. Fire trigger: APDataStore has at least
   **45,521,567,744 free bytes** after a separately certified cleanup or capacity expansion. No
   retained byte may be deleted or moved without its own certify-or-block record.
2. **QUEUED-WITH-A-FIRE-ORDER — implement and review the exact transferred objective.** Owner:
   `ddm_w96a` implementation successor. Consumer store: the reviewed WD3/S1A source plus the same AP
   root for generated configs. Fire trigger: storage is green; expected-flip margin, tau schedule,
   `eta_min=0.01*lr`, step-0 pose, config identity, and resume determinism have tests; every edited
   Python function has two review-tracker passes.
3. **QUEUED-WITH-A-FIRE-ORDER — two sequential aligned windows.** Owner: MAIN. Consumer store:
   `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`. Fire trigger: step 2 is green, memory
   preflight passes, and fresh distinct scorer and Metal claims are present. Run seed 20260815 then
   20260816, retaining every checkpoint, receiver/scorer field, candidate archive, repeat, config,
   and SHA receipt.
4. **QUEUED-WITH-A-FIRE-ORDER — n60 adjudication.** Owner: `ddm_w96a`. Consumer store: same AP
   root plus this memo. Fire trigger: each checkpoint lands. Re-run the exact S1E instrument; report
   each seed separately and compare ep65 against OFF ep65.
5. **QUEUED-WITH-A-FIRE-ORDER — n600 sampled leg.** Owner: MAIN, which owns the global scorer
   slot. Consumer store: same AP root. Fire trigger: an aligned seed checkpoint reaches
   `composed_delta <= OFF/5`, the full n600 slot is idle and freshly claimed, and the selected
   checkpoint/config/payload SHAs are sealed. If both seed-65 screens remain worse than OFF/2, do not
   fire n600; close only that two-seed aligned instance.

## Verdict

**NO CONFIGURATION VERDICT.** The prior-law `0.155 -> <=0.031` prediction remains live but untested.
Storage, followed by the absent exact objective path, blocked the first aligned checkpoint. The OFF
family's 35 retained screen rows remain advisory negatives at the OFF formulation; they do not close
the aligned formulation. The exact pointer is unchanged.

## GESTALT-DELTA

Before this unit, the working story was “one OFF seed refused at +0.155, seed variation and crash
resume are outstanding, so run the aligned config.” Source recall changes that picture in three ways:

1. The OFF evidence now contains two genuinely varied seeds and 35 retained rows; #1251 is already
   cured for the apparatus, although it remains uncured for aligned training.
2. EF3000, EF6000, and CW1 show that the ancestor expected-flip regime descends, but also that the
   13.6x factor is architecture-specific, the LR optimum was not established, higher LR weakened the
   result, lower LR was null, and the tau schedule is inherited. The factor is a falsifiable prior,
   not a multiplier that may be applied to W96 data.
3. The current W96 trainer is not a flag-compatible copy of CE1: it uses calibrated target
   probability, not the target-margin expected-flip law. The first honest aligned measurement needs
   a real objective implementation, not a JSON rename.

The scientific question is therefore still valuable and still open. The immediate blocker is mundane
but binding: retained reference-form evidence does not fit on the canonical SSDs right now.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content for `W96`, `film_amortized_flat_w96`,
`expected_flip`, `ce_fraction`, `softplus_fraction`, `aligned objective`, `s1e`, `pose gate`, `#1251`,
`#1273`, and `renderer corner`; searched `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks,
the canonical equations JSON, design/SPEC surfaces, source, retained AP/Vertigo receipts, and canonical
task history.

Beyond the charter seeds, this found:

- `ddm_ef3000_first_descent_verdict_20260817.md` and
  `ddm_ef6000_double_window_verdict_20260817.md`: the aligned ancestor crossed its own initialization,
  but only on its own advisory instrument and architecture.
- `ddm_cw1_corrected_window_20260817.md`: the 13.6069x allocation result re-derived; `lr=2e-5` is a
  borrowed constant; tau is a fraction-of-window confound; CW1-LR6E5 weakened, seed-2 widened the
  band, and LR1E5D was null. This removed any license to tune LR upward in W96.
- The S1A #1273 cure receipts and current source: resume and seed variation are already repaired.
- The live S1E retention tree: the charter's 14-row snapshot has grown to 35 rows across both OFF
  seeds. This changed the baseline report but not the need for aligned seeds.
- Current WD3 source: expected-flip is absent, so the aligned window cannot be created by editing only
  a compiled config. This added an implementation gate before launch.
- The canonical equations registry contained no compiled equation that transfers CE1's exact law into
  the film-W96 WD3 objective; that scoped absence prevents pretending the bridge is already governed.

## Retained receipts

| path | bytes | SHA-256 | disposition |
|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/off_baseline_s1e_rerun.json` | 296,698 | `3037d264f097cd1b239cd96fc2302f5d812e0f3384eea7813fdc5cb074b60b18` | KEEP; replay of retained OFF rows |
| `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/STORAGE_PREFLIGHT_BLOCKER.json` | 1,870 | `4e62de066d3304e970a20002aa08400789015f9d26d69f143d82f4227f3e1056` | KEEP; launch blocker and fire trigger |

No aligned checkpoint, aligned screen receipt, trained packet, archive, receiver field, scorer field,
or n600 payload exists. Nothing was discarded, moved, or deleted.

**Own-vehicle frontier: S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600], GB1 groupbin8 archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` — UNMOVED by ddm_w96a.**
