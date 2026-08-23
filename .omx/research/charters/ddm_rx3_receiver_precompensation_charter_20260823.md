# ddm_rx3_receiver_precompensation — R and uint8 REPAIR 11,685 broken positions for free; PR98's zero-byte decode-side transform is a proven family on the ancestor; nobody has mined THIS body's measured repair structure for one

## MANDATE

Routed finding, three memos, no operator verbatim. **This arm's output must cost ~ZERO counted
bytes or it is not this arm.**

**(1) The realization path already repairs, and we measured how much.**
`ddm_mst1_manufactured_stage_split_20260822.md` (`1c33f278920b91bf922e9620deb9ce20615135e8`)
measured the stage ladder: the native render manufactures **+22,321 net** seg errors, then
**R repairs −6,980** and **uint8 repairs −771**. `ddm_wj1_cost_error_position_join_20260823.md`
(`72975fcaa1`) resolved the membership: of **28,602 gross native breaks**, **11,685 positions are
LATER REPAIRED** by the downstream path, carrying **2,252.103297 B** of modeled cost —
**5.313820% of the 42,382 B demand** — and **10,491 of those 11,685 (2,249.700879 B, 99.89% of the
repaired mass) sit in the top 1% cost set.** The realization path is not only a destroyer; on
11,685 positions it is a free corrector we are paying top-1% rates to fight.

**(2) The exploit family is PROVEN — on an ancestor.** Canonical leaderboard lesson **L28**
(PR98/PR101 decode-side channel postprocess): subtract 1.0 from frame_0 RED, frame_0 BLUE, and
frame_1 GREEN — **3 lines in inflate.py, 0 archive bytes, measured −0.0001 to −0.0005 S.** That is
a zero-byte deterministic transform applied at decode time, chosen to work WITH the frozen
scorers rather than against them. Per L18 the NUMBER does not transfer to our vehicle; the
MECHANISM CLASS does, and **it has never been mined against dx2's own measured break/repair
structure.**

**(3) Why it is worth an arm now, and why it is not the paint arm.** The distortion side is worth
**42,235 B** of rate budget (zero-distortion ceiling 180,218.3 B vs 180,368 shipped; seg alone
30,248 B), and it currently has NO live owner: the paint/partition-repaint route is measured DEAD
on this body (`ddm_fp1` receiver floor d_seg **0.008305** = 41× dx2's live 0.00020139;
`ddm_qa92` measured even perfect GT paint as joint net-POSITIVE +0.30 S via SegNet's 85px ERF
collateral), and the joint-descent line terminated by self-refusal (`EXACT_DELTA_NONNEGATIVE`).
**This arm does not repaint anything.** It leaves the render, the tokens, the model and the
archive byte-identical, and asks whether a GENERIC deterministic function inserted between render
and R buys distortion for free. That is a different actuator in a different place.

**(4) THE HARD LINE, and it is this charter's central discipline.** Rule-118: a GENERIC ALGORITHM
in `inflate.py` is FREE; VIDEO-DERIVED content is COUNTED. A transform whose PARAMETERS were fit
to this video is video-derived — **its parameters must be stored in `archive.zip` and priced at
6.658590e-07 S/B, no exceptions.** PR98's three offsets were video-derived too; they were simply
tiny. **Every rung must report BOTH: the parameter byte cost (counted, real, in the archive) and
the realized distortion delta.** A rung claiming "zero bytes" while carrying fitted constants in
the receiver source is the hide-data-in-code fake (NO-FAKE #6/#7) and must not be reported.

## SCOPE

1. **Verify pins; REUSE wj1's target list and mst1's stage split read-only; refuse on drift.** DX2
   archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B ·
   RC64 token stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @
   113,777 B. WJ1's position list sha **`bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d`**
   — verify it independently before use. Reproduce the 28,602 gross / 11,685 repaired split before
   proposing anything; a disagreement IS the finding. Do NOT re-derive the exchange rate —
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **CHARACTERIZE the repair operator before proposing a transform.** On the 11,685 repaired
   positions vs the 16,917 terminal-persistent ones, measure what DISTINGUISHES them: pre-R
   residual magnitude · sign · distance to the class boundary · local margin · neighbourhood
   composition · class pair. **A transform proposed without a measured discriminator is a guess.**
   Report the discriminator's separation strength with its denominator. If repaired and persistent
   positions are indistinguishable in every measured coordinate, say so — that closes the exploit
   on measured evidence and is a complete result.
3. **Build ≥3 zero-or-near-zero-byte candidate transforms and measure each through the REAL path.**
   Each is a deterministic function applied to the rendered RGB BEFORE R (camera resolution), then
   R → uint8 → frozen SegNet/PoseNet, n600. Candidate classes (propose better if the SCOPE-2
   discriminator suggests them): global per-channel offset (the direct L28 analogue) · a
   parameterized local operator keyed on the measured discriminator · a boundary-band-scoped
   variant. Per candidate report: **parameter COUNT and its real archive byte cost · realized
   Δd_seg per class with Lane on its own row · realized Δd_pose · ΔS_distortion · ΔS_rate ·
   net ΔS.** Do NOT interpolate distortion between parameter settings — `ddm_ri1`/`ddm_ni1`
   measured amplification exponent **16.69**; measure each.
4. **Report the POSE axis on every rung, never as an afterthought.** Any pre-R transform changes
   the frames PoseNet reads. `#1127`'s SD1M ladder died on exactly this: a change that looked
   locally cheap carried d_pose 0.286 uncompensated. **A rung reporting only d_seg has not been
   measured.** If a rung is seg-positive and pose-negative, report the joint ΔS and say which
   dominates.
5. **Adjudicate honestly, including the empty outcome.** If every candidate is net-positive, say so
   plainly: the zero-byte receiver-transform family is closed on this body, the L28 mechanism does
   not transfer to the dx2 render, and the distortion axis has no free move at this actuator. That
   is a complete result and it narrows the campaign. **Build NO shipping candidate here**; the
   per-candidate table is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- **DO NOT set `F26_TOKEN_DECODER=native-hpac`.** This generation refuses it
  (`f26_inflate.py:437`, the ddm_rr2 corrector is python-only). It also means **prefix advisories
  are IMPOSSIBLE on this lineage** — the prefix branch REQUIRES native-hpac and line 437 refuses
  it, so the two guards are jointly unsatisfiable. **Every scorer row runs at FULL n600; a strided
  pilot is not available to you and must not be attempted.** Order candidates by a scorer-free
  proxy instead. (MAIN-owned defect, filed; not yours to cure.)
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every transformed frame set, every re-encoded archive, every per-class
  argmax field persists with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation/` —
  BOTH SSD TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister
  arms at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN
  destination per the disk rule while the tiers are full.** Do NOT write to `/Volumes/*`. Say which
  tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place; build your transform as a NEW receiver
  variant. The jo1 r9 run dir is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every d_seg/d_pose states its GT lineage (DALI-GT where the tool family expects it; a
  PyAV-lineage GT on the pose axis is a measured wrong-objective defect).
- File ownership: WJ1 owns the join + target list · MST1 the stage split · BL1 the cost field ·
  AR1B the residue census · **JF1/MP3/AP1 are running concurrently on the RATE axis — this arm is
  DISTORTION-side and must not duplicate or touch their trees.** CITE them.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_fp1` (receiver floor d_seg **0.008305**, FORMULATION-scoped) + `ddm_qa92` (perfect GT paint
  joint **+0.30 S** via 85px ERF collateral) — **the paint/repaint family is DEAD, and deader on
  dx2 than on the ancestor it was killed on (0.008305 is 41× dx2's live 0.00020139).** Do NOT
  propose painting, compositing, or class-field substitution. This arm perturbs the rendered RGB
  by a generic function; it does not author content.
- SD1M mass ladder (`ddm_mz2_frozen_section_representation_attack_20260815.md` lineage, memo §5, commit `c30f92fbc9`) — the amplification lesson: render amplification
  ~**38,700×**, damage ∝ mse^~0.4 falls SLOWER than credit at every depth, and pose went 0.286
  uncompensated. **A small-looking perturbation is not a small distortion; measure, never model.**
- `ddm_ri1` + `ddm_ni1` — whole-body lossy re-representations DEAD (43.66× and 247.71× over
  ceiling), amplification exponent **16.69**. Seg responds violently and non-linearly.
- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` (`5e8d6011ba`) · `ddm_to2` · `ddm_ef1` ·
  `ddm_oe1` · `ddm_ad2` — the RATE axis on this body is at a sharp joint local optimum in every
  tested direction. **Do NOT propose a coder, ordering, estimator, model-member, or field change;
  those axes are owned and closed. Your actuator is the receiver's pre-R function.**
- `ddm_wj1_cost_error_position_join_20260823.md` (`72975fcaa1`) — its own honest limit: the
  **6,846.84 B** gross-manufactured mass is **incumbent MODELED cost, not realized savings**, and
  it is only **16.16% of the demand**. Enrichment vs independence is enormous (90.96× count,
  257.48× bit) but the absolute ceiling is modest. **Do not inflate the enrichment into a byte
  claim.** Also: **69.39% of gross-manufactured modeled mass lies OUTSIDE Lane** — this is not a
  Lane story; do not scope the transform to Lane by assumption.

## OPTIMAL FORM

- **REFERENCE FORM: the shipped dx2 receiver with ONE additional deterministic function between
  render and R, everything else byte-identical.** The render, tokens, semantic tensors, HPAC model
  and coder are FIXED. Family exemplar: **L28** (PR98/PR101 decode-side channel postprocess, 0
  archive bytes, measured −0.0001..−0.0005 S on its own vehicle) — that is the FORM to match, at
  its own optimum for THIS body, not its constants (L18: ancestor = lessons, never numbers).
- Family exemplar for CONDUCT: `ddm_wj1_cost_error_position_join_20260823.md` — it confirmed its
  prediction at 45× the bar AND corrected its own charter's arithmetic in the same memo (the
  22,321 headline is a NET STAGE DELTA, not a membership set; the valid mask has 28,602 gross
  transitions), and it labelled its 6,846.84 B "incumbent modeled mass, not realized savings"
  rather than letting a big number stand unqualified. **Match that: correct me where I am wrong,
  and qualify every mass you report.**
- SCOPE reductions declared per row. **The prefix reduction is UNAVAILABLE (see HARD CONSTRAINTS) —
  every scorer row is n600.** MECHANISM reductions FORBIDDEN: a modelled distortion estimate, a
  weight-space proxy, or a transform not carried through the real render→R→uint8→frozen-scorer
  path is the exact SD1M defect. **Measure through the real path or do not report the row.**
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B · HPAC model
  13,515 B · residue 66,591 B. DX2 S 0.14821987563243377 · rate 25·180368/37545489 = 0.1200996 ·
  seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B
  → shed **42,382 B**; **6.658590e-07 S/B**; **1.2731082153 B/flip**. Distortion is worth **42,235 B**
  of rate budget; **0.001 S of distortion = 1,502 B**. WJ1: 28,602 gross breaks / 6,846.84 B;
  11,685 repaired / 2,252.103297 B; 16,917 terminal-persistent.
- **PRIOR-LAW PREDICTION (falsifiable):** the frozen scorers have a systematic decode-side bias the
  render does not model — that is why L28's three constants paid on the ancestor and why R repairs
  6,980 errors here for free. **At least one candidate achieves Δd_seg ≤ −5.0e-6 (≥590 eliminated
  flips, ≥751 B-equivalent of rate budget at 1.2731082153 B/flip) at a parameter cost under 100 B,
  with d_pose not worse than +2.0e-7 S.**
  **FALSIFIER:** every candidate is net-positive in joint ΔS, or every seg-positive candidate is
  pose-negative by more than it gains. Then the zero-byte receiver-transform family is closed on
  this body, L28's mechanism does not transfer to the dx2 render, and — composed with the paint
  family's death and the joint line's self-refusal — **the seg-distortion axis has no free actuator
  left at the receiver**, which routes the remaining distortion budget entirely to a body change
  (`ddm_nr1_taskcell_quotient_prebuild_20260822.md`). **Count it plainly if it lands; both outcomes
  route the campaign.**

## DELIVERABLE

`.omx/research/ddm_rx3_receiver_precompensation_20260823.md` — WJ1's 28,602/11,685 split reproduced
+ the SCOPE-2 repaired-vs-persistent discriminator with its separation strength and denominator +
the ≥3-candidate table with, per candidate: **parameter count and REAL archive byte cost · realized
Δd_seg per class (Lane on its own row) · realized Δd_pose · ΔS_distortion · ΔS_rate · net ΔS** +
the best candidate's coordinates or the honest all-positive verdict + verdict_scope at the NARROWEST
level the evidence supports. Every figure carries its denominator and its GT lineage. No shipping
candidate. Commit via the serializer. End with the own-vehicle frontier line.
