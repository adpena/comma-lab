# ddm_dj1 — dual-lineage carrier adjudication on rc2

`date_utc: 2026-08-21` · `arm: ddm_dj1_dual_lineage_carrier`

`status: CLOSED(measured-arithmetic)` · `score_claim: false` · `promotion_eligible: false`
· no Modal · no scorer forward · no archive build · no frozen-packet or `upstream/` mutation

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`,
archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` —
UNMOVED by ddm_dj1.** This unit produced no candidate and no exact row.

## ANSWER FIRST

**Close the joint DALI ∥ PyAV carrier family on rc2 before build. No formulation can make a
CPU-declared packet beat the CUDA packet we already ship, and the live rc2 CPU decoder cannot
finish inside the contest wall.** These are independent failures.

1. **Score:** hold rc2's rate fixed and give a carrier-only CPU formulation the impossible gifts
   of zero added bytes and `d_pose_cpu = 0`. SegNet still reads frame 1, which the frame-0 carrier
   does not alter, so the PyAV-lineage seg term remains. The absolute optimistic floor is
   **S = 0.154898267387**, already **+0.006619796167** worse than rc2's CUDA score. A perfect
   corrector that removed *both* lineage penalties at zero bytes could only **tie** rc2 at
   0.148278471220; it cannot beat it without becoming a different, general score lever.
2. **Wall:** the native corrector port that the charter treated as a gate has already landed and
   been measured on the exact rc2 archive. `[contest-CPU wall measurement; no score]` inflation
   still took **2,850.781 s**; **token decode alone took 2,427.166 s**, which is **627.166 s over
   the entire 1,800 s job wall before render or evaluation begins**. `upstream/evaluate.py` never
   ran. A carrier head or post-render analytic corrector cannot cure that token-stage wall.

The public 0.162 bar is therefore not the strategic admission rule. Some impossible pose-zero
bounds can beat 0.162, but **none beats 0.148278**, which the charter explicitly requires before a
CPU declaration becomes interesting. Building or firing would spend against a dominated packet.

## RECALL EVIDENCE

Searched the full required surfaces before adjudicating:

- `.omx/research/` by content, not only filenames. Queries included `DALI`, `PyAV`,
  `GT lineage`, `carrier re-solve`, `0.0991`, `99.725`, `21.80`, `dual-head`,
  `analytic corrector`, `0.20513189`, `831.5`, `4369.6`, `#1054`, and `#1142`.
- `.venv/bin/python tools/list_canonical_equations.py --json`; consumed
  `cw1_gt_lineage_additive_pose_offset_v1` and its cpu1 anchor.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, the GT-lineage FEED blocks in
  `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, current design/SPEC files, and
  `.omx/state/{canonical_task_status.jsonl,operator_p0_ledger.jsonl,active_lane_dispatch_claims.md}`.
- The actual `upstream/evaluate.py` device fork and score formula, plus `upstream/README.md`'s
  1,800 s / 4-CPU rule. `upstream/` was read only.
- Live retained receipts under `/Volumes/APDataStore/pact/ddm_{cpu1,fs3,rc2}/`.

What was found beyond the charter seeds, and what it changed:

1. **The cd1 gate is stale.** rc2 already uses `NativeFreeCorrector`. Its exact CPU attempt
   completed the child inflate in 2,850.781 s after the 1,800 s harness timeout. This replaces
   the jg5 4,369.600 s wall for current-body routing. The port helped, but did not open CPU.
2. **Task #1142's consumer premise was already repaired.** `ddm_dg1` found that commit
   `809199d24f` had repointed `qs1.GT_POSE`, both `mt1` defaults, and the nine `qs1` importers to
   DALI. Its 11 remaining findings were declaration defects, not eight consumers optimizing the
   wrong objective. No rewiring belongs in this arm.
3. **`0.0991 B/pair` is not a dual-head price.** `ddm_fs3` measured a same-slot coefficient
   replacement: 454 pairs / 5,119 changed coefficients changed an already-existing carrier
   archive by +45 B. Its density ladder is non-monotone. A second head adds another coefficient
   object and must be counted separately; transferring 0.0991 B/pair to it would be fake pricing.
4. **The cheapest analytic-corrector family already has a negative.** `ddm_up1`'s realized global
   photometric sweep was minimized at offset 0; every integer offset hurt pose. A more complex
   generic map remains unmeasured, but the perfect-map strategic bound below makes it unnecessary.

The canonical research index did not expose a later direct `dj1` or dual-lineage build receipt in
the searched scope; the current DAG FEED, live receipts, and task ledger supplied the later state.

## PRIMARY ARITHMETIC

All score arithmetic uses the exact evaluator formula
`100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`. The evaluator's printed two-decimal score is not
used.

### rc2 and the projected PyAV row

| quantity | value | authority |
|---|---:|---|
| rc2 rate term | `0.120158243244615` | DERIVED from 180,456 exact bytes |
| rc2 `d_seg` | `0.00020139` | MEASURED `[contest-CUDA T4, n600]` |
| rc2 `d_pose` | `0.00000637` | MEASURED `[contest-CUDA T4, n600]`, report resolution |
| rc2 S | **`0.148278471220309`** | recomputed from components |
| cpu1 PyAV `d_seg` | `0.000347400241428` | MEASURED `[macOS-CPU advisory, n600]` on jg5 raw |
| cpu1 PyAV `d_pose` | `0.000147010909813` | MEASURED `[macOS-CPU advisory, n600]` on jg5 raw |
| GT pose-table separation `C` | `0.000140615093997` | MEASURED n600, same clip |
| projected rc2 PyAV S | **`0.193240269136597`** | DERIVED: cpu1 components with rc2's −169 B rate rider |

The projected rc2 PyAV number is **not** a `[contest-CPU]` score. It transfers only the lossless
169-byte rider into cpu1's same-raw lineage decomposition. The exact rc2 CPU raw was not scored.

### The geometry the joint formulation must obey

Let `A` and `B` be the PyAV and DALI GT PoseNet tables and `P` the candidate pose outputs. With
population MSE `C = mean||A-B||^2`, triangle inequality in root-MSE units requires

`sqrt(d_pose(P,A)) + sqrt(d_pose(P,B)) >= sqrt(C)`.

The pw2 falsifier requested `d_pose_pyav <= 6e-5` while holding
`d_pose_dali <= 1.3e-5`. But

- `sqrt(6e-5) + sqrt(1.3e-5) = 0.011351517968`,
- `sqrt(C) = 0.011858123544`.

The requested balls do not intersect. Equivalently, at `d_pose_dali = 1.3e-5`, the best possible
PyAV value is **`6.810494906e-5`**, already above the falsifier. This is an optimistic
unconstrained-output proof; the 12-dimensional quantized carrier can only do worse.

## FORMULATION ADJUDICATION

Every score below is a DERIVED optimistic bound over MEASURED inputs, not a score row. `delta_B = 0`
means a deliberately favorable lower bound unless explicitly stated; it is not a byte price.

| formulation | best honest arithmetic on rc2 | CPU-axis action | CUDA-axis action | counted bytes | decode-wall delta | verdict |
|---|---|---:|---:|---:|---:|---|
| **A. One equal-MSE joint head** | The unconstrained minimax point is the GT midpoint: `d_pose=C/4=3.51537735e-5` on each lineage. CPU S **0.173647606976**; CUDA S **0.159046582833**. | **−0.019592662161** vs projected PyAV rc2 | **+0.010768111612** vs rc2 CUDA | exact archive delta **UNMEASURED**; 0 B assumed for the bound | no token-stage cure | **CLOSED, FORMULATION.** Its CPU row misses public 0.162, while its CUDA row is worse than rc2; pw2's joint falsifier is analytically impossible. |
| **A2. One CPU-endpoint head** | Give CPU `d_pose=0` and keep its PyAV seg term: CPU S **0.154898267387**. The same pose output pays at least `C` on DALI, giving CUDA S **0.177795922421**. | **−0.038342001749** | **+0.029517451201** | exact archive delta unmeasured; 0 B assumed | no token-stage cure | **CLOSED, FORMULATION.** It beats public only under perfect pose, but is still +0.006619796167 worse than what we ship and destroys CUDA. |
| **B. Dual head, shared basis + per-lineage coefficients** | At the measured same-basis DALI floor `6.365873831e-6`, a zero-extra-byte PyAV head projects CPU S **0.162876910023**: +0.000876910023 over public and +0.014598438803 over rc2. Even the impossible `d_pose_cpu=0`, `delta_B=0` floor is A2's **0.154898267387**, still dominated by rc2. | up to −0.030363359113 at the measured floor; −0.038342001749 only at impossible zero pose | 0 if the DALI head remains byte-identical | exact delta **UNMEASURED**. The in-scope append-only design must add a nonempty counted stream unless the CPU head is deterministically derived, in which case it is formulation C. Existing real coder emits a 9,759 B Rice payload for one 600x12 head; fs3's +45 B replacement is not a second-head price. | select-one-head parsing cannot remove 2,427.166 s token decode | **CLOSED, STRATEGIC.** The favorable 0-byte bound already fails our ship bar, so counting a real appended head cannot reverse the decision. |
| **C. CPU-side analytic corrector** | Pose-only perfection has A2's 0.154898 floor. Perfect zero-byte correction of both measured lineage penalties returns exactly rc2's components and **ties 0.148278471220**; it does not beat them. | at most removes the CPU lineage gap; any common-axis gain is a different lever | unchanged if correctly axis-gated | 0 B only for genuinely generic code; every fitted/video-derived value is counted | adds work after a token stage already 627.166 s beyond the whole wall | **CLOSED, STRATEGIC.** The perfect in-scope bound is a tie and the simple global form is empirically dead. |

### Why no real-coder dual-head price was run

The charter correctly requires both heads to be counted. No current-body PyAV coefficient head
exists, so a real price would first require the full solve the arithmetic is deciding whether to
build. The proper admission test is therefore the favorable `delta_B=0` lower bound. It already
fails the required `S < 0.14827847122030852` condition by at least 0.006619796167 for a
carrier-only formulation. The in-scope dual-head design preserves the DALI stream byte-identically
and appends the CPU coefficients, so running a coder after that would measure a non-deciding added
term. A joint representation redesign that also shrinks the original stream is a separate rate
lever, not this dual-head formulation. The 9,759 B figure above is the existing coder's raw
fixed-shape output, not an archive-delta claim.

## DECLARED-RUNTIME STRATEGIC TABLE

| row | S | vs public 0.162 | vs shipped CUDA 0.148278471220 | runtime status |
|---|---:|---:|---:|---|
| rc2 shipped | **0.148278471220** `[contest-CUDA T4, n600]` | −0.013721528780 | 0 | **PASS**, requested runner `linux-nvidia-t4` |
| unchanged rc2 on PyAV objective | **0.193240269137** DERIVED, not a CPU score | +0.031240269137 | +0.044961797916 | CPU inflate REFUSED |
| one-head midpoint, ideal 0 B | **0.173647606976** DERIVED | +0.011647606976 | +0.025369135756 | CPU inflate REFUSED |
| dual-head at current measured carrier floor, ideal 0 B | **0.162876910023** DERIVED | +0.000876910023 | +0.014598438803 | CPU inflate REFUSED |
| carrier-only absolute floor (`d_pose=0`, ideal 0 B) | **0.154898267387** DERIVED | −0.007101732613 | +0.006619796167 | CPU inflate REFUSED |
| perfect full-lineage generic corrector, ideal 0 B/0 s | **0.148278471220** DERIVED | −0.013721528780 | **tie, not beat** | actual CPU token stage still REFUSED |

**Strategic verdict:** keep `linux-nvidia-t4`. There is no CPU-declared packet in this formulation
space that is better than the packet already shipped. Public 0.162 is a weaker bar and cannot
override that comparison.

## CURRENT-BODY DECODE RECEIPT

The exact rc2 CPU attempt is `[contest-CPU wall measurement; no score]`, Linux x86_64, four torch
threads, archive `df7fd266…` @ 180,456 B, `NativeFreeCorrector`:

| stage | seconds |
|---|---:|
| token decode / checkpoint load | **2,427.166** |
| neural render and resize | 410.183 |
| frame-0 selector and I/O | 2.825 |
| archive setup | 0.533 |
| child inflate total | **2,850.781** |
| contest wall | **1,800.000** |

The harness killed its bash parent at 1,800 s; the decoder child completed and emitted the full
stage report. `upstream/evaluate.py` never ran, so this is wall authority only. It supersedes
jg5's 4,369.600 s number for current-body routing, while confirming the same conclusion.

Custody caveat: this prior run's result says `inflated_outputs_volume_manifest was not produced`;
the local APDataStore harvest contains the 39,829 B result JSON, not the 3.66 GB raw. That violates
the desired payload-custody shape, so the raw hash is not treated as a retained scoring payload.
The wall report remains self-contained for stage timing. ddm_dj1 launched nothing and materialized
no payload.

## RETENTION RECEIPTS CONSUMED

| artifact | bytes | sha256 | use |
|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_rc2/cpu_row_r1/MODAL_REMOTE_RESULT.json` | 39,829 | `ec9dab16706bcd28c30118b2d465e7717cc5bb4a093d16e33b2ca92b57850145` | current-body CPU wall |
| `/Volumes/APDataStore/pact/ddm_rc2/t4_row_r2/MODAL_REMOTE_RESULT.json` | 129,146 | `38e195853a645dc2fbe433cb0b9c09a38f022ccbc1b4a5b6948bc6740ce0bd46` | rc2 CUDA components / wall |
| `/Volumes/APDataStore/pact/ddm_cpu1/retained/cpu1_attribution_n600.json` | 3,237 | `940cbafa82c929a801c0a129a257fa9ed6d999822172079bab645668759eada3` | same-raw two-lineage components |
| `/Volumes/APDataStore/pact/ddm_cpu1/retained/cpu1_per_pair_n600.npz` | 20,234 | `f00afa49ff05a27e0e4141ea87f3a1fe4ff2087c0e9697c026a37af5fc895bbf` | retained per-pair lineage decomposition |
| `/Volumes/APDataStore/pact/ddm_fs3/carrier_ladder/FS3_CARRIER_PRICE_LADDER.json` | 6,044 | `2388d1110b3e0b109d4167e6a114f62cd98c612c5fc18f415220cc97cefabff3` | same-slot real-coder price and its non-monotonicity |

No new bulky artifact was created. No Vertigo write occurred.

## PRE-REGISTERED FALSIFIERS AND DISPOSITIONS

| formulation | falsifier stated before any new measurement | result | disposition |
|---|---|---|---|
| one joint head | close if `d_pose_pyav <= 6e-5` and `d_pose_dali <= 1.3e-5` cannot coexist under measured `C` | **FIRED analytically:** minimum PyAV value at the DALI cap is 6.8105e-5 | **FOLDED / CLOSED FORMULATION** |
| dual head | close for CPU declaration if even `delta_B=0`, `d_pose_cpu=0` fails to beat rc2 CUDA | **FIRED:** 0.154898267387 > 0.148278471220 | **FOLDED / CLOSED STRATEGIC** |
| analytic corrector | close if perfect removal of only the lineage penalty does not strictly beat rc2, or if token decode alone exceeds 1,800 s | **FIRED twice:** perfect correction ties; token stage is 2,427.166 s | **FOLDED / CLOSED STRATEGIC** |

No scorer measurement was run after these falsifiers. No follow-on is named: the closure is strict
against the charter's own CPU-declaration criterion, so there is no fire order for this arm.

## BUILD STATUS

**CLOSED(measured-arithmetic), no build.** No rc2 body was mutated; no archive, runtime branch,
coefficient head, seal, or launch manifest was created. This is the mandate's allowed refusal path:
all three formulations fail before build under optimistic zero-byte / perfect-correction bounds.

## LIVE-HYPOTHESES

- **None in scope.** A more complex generic DALI→PyAV transform might remove more of the lineage
  gap than the dead global-offset form, but even perfect removal only ties the CUDA packet and the
  CPU token stage still misses the wall. It is therefore not worth pursuing as a CPU-declaration
  lever on rc2.

## DEAD-ENDS

- **Single shared carrier head:** closed at FORMULATION scope. The two requested lineage balls do
  not intersect, and the equal-MSE midpoint loses on both strategic bars.
- **CPU-specific second head:** closed for CPU-declaration strategy. Even perfect pose at zero
  counted bytes remains 0.00662 S worse than rc2; a real second head only adds bytes.
- **Global analytic photometric corrector:** closed at FORMULATION scope by ddm_up1; offset 0 was
  the measured optimum and integer offsets hurt.
- **Any carrier/corrector build before a token-decoder cure:** closed on current-body wall
  measurement. Token decode alone is 627.2 s over the whole contest budget.
- **Using 0.0991 B/pair as a dual-head price:** closed as a category error. It prices replacement
  values inside one existing stream, not a second counted stream.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED.**
