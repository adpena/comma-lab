---
schema: ddm_v4d_composed_candidate.v1
date_utc: 2026-07-31
arm: ddm_v4d (adaptive + hybrid composed candidate — v4c's successor; the ph3 rung stack)
lane_id: "lane_ddm_v4d_adaptive_hybrid_20260731"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver + frozen PoseNet; composed byte-close DONE, n600 evaluate gate STAGED — MAIN fires]"
operator_binding: "ddm_v4d — compose the ph3 rung stack (QA65 dim0 offset re-solve + QA66 per-pair shear + QA69 refinement waterfill + QA70 min-entropy probe + QA72a stage attribution) under the §1 realization doctrine; STAGE the gate (MAIN fires)"
tools:
  - "experiments/ddm_v4d_resolve.py (modes qa66/refine/qa70/qa72a/resummarize)"
  - "experiments/inflate_runner_v4d.py (v4d receiver PFS1WPD1: static two-plane + photo (a,b) + per-pair beta + optional dim0 offset; rule-118 free)"
  - "experiments/ddm_v4d_build_composed_archive.py (byte-close build, kl1 quad + beta section + DEFLATE container)"
  - "experiments/ddm_v4d_verify_decode.py (parse-back bijection #417 + decode identity + dim0-offset round-trip + beta path + rebuild)"
  - "experiments/stage_v4d_realized_gate.sh (STAGED gate — MAIN fires)"
  - "experiments/launch_v4d_detached.sh (ppid-1 detach + tac-hijack guard)"
data: "SSD ddm_v4d_20260731/{final_qa66.jsonl(600), refine.partial.jsonl(600), final_refine.jsonl(600), refine_receipt.json, qa72a receipts, qa70 receipts, v4d archives+build+verify receipts, logs}"
tokens: "[no-triality] [p0-ledger-ok] [magnitude-ok]"
---

# ddm_v4d — adaptive+hybrid composed candidate: QA65 dim0-offset re-solve × QA66 per-pair shear × QA69 refinement waterfill (+QA72a attribution, +QA70 gauge probe)

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`, per-pair realized through the real
receiver + frozen PoseNet, priced through the real byte-closed member. This arm lowers the pfs1
warp-base ADVISORY vehicle (v4c gate MEASURED S 0.992972), which is FAR from the pointer — it does
NOT move the pointer and does NOT achieve the GOAL (sub-0.15). **The n600 `evaluate.py` gate (MAIN
fires, ONE-n600 rule) is the authority; my composed S is a PREDICTION the gate verifies** (fidelity
law: v4b 3e-6, v4c 1.38e-4 — two anchors; the standing re-anchor duty applies, residual >1e-3
falsifies the fidelity leg).

## §1 Headline (advisory)

v4d composes the ph3 rung stack on the MEASURED v4c gate (0.992972 = seg 0.431179 + pose 0.322248
+ rate 0.239543; 359,750 B) under the §1 realization doctrine (proposals cheap; the uint8/coder/
render path is the only acceptor; realized ΔS≤0 acceptance; every member priced byte-closed).

**v4d-full (PRIMARY candidate): byte-closed 360,238 B, sha f1f3288062468e97…, verify
all_checks_ok, deterministic rebuild → same sha, gate STAGED. Predicted composed S = 0.963986
(−0.028986 vs v4c 0.992972).** First predicted sub-0.97 own-vehicle row.

| axis | v4d-full | how obtained |
|---|---:|---|
| 100·d_seg | **0.431179** | v4c gate MEASURED (cell_drop50 tokens BYTE-IDENTICAL — sha-checked; evaluate band ±2.8e-5) |
| pose √(10·d̄) | **0.292939** | advisory, per-pair realized (d̄ 0.008581 over 600; integrated dim0-offset re-solve + (a,b) re-fit + per-pair beta, monotone-safe best-of vs the QA66 floor) |
| rate 25·B/37.5M | **0.239868** | MEASURED (360,238 B byte-closed; +488 B vs v4c) |
| **predicted composed S** | **0.963986** | seg + pose + rate |

**QA66-only (FLOOR candidate, zero new evals): 359,890 B, sha d5149d811b4a…, verify
all_checks_ok, rebuild-stable. Predicted S = 0.979578 (−0.013394 vs v4c).** Its pose d̄ 0.009533 is
EXACT (realized in the v4c photo pass; poses/ab bit-identical to v4c) — the lowest-risk row.

## §2 THE RUNG STACK — per-rung realized deltas + admission verdicts

Stage decomposition of the v4d-full pose win (full-600, realized, monotone-safe composition;
refine-win 545/600, QA66-kept 55):

| rung | actuator | realized d̄ step | ΔS (pose) | Δbytes | admission |
|---|---|---|---:|---:|---|
| QA66 | per-pair beta {0,0.5,1.0}, sign free from yaw | 0.010384→0.009533 | **−0.013485** | +140 (beta 111 B + framing) | **ADMITTED** (win ≫ bytes at water) |
| QA65 | dim0 offset-coded lattice (19.3× finer), re-solved | (isolated slice) 0.010384→0.009800 | **−0.009196** | +319 (tp member 6,059→6,378) | **ADMITTED** (92% pairs improved) |
| (a,b) re-fit + integration | 2-param GN at new dim0 + joint beta re-select | remainder to 0.008581 | **−0.006628** | ~+18 | ADMITTED (rides the same pass) |
| **total v4d-full** | | **0.010384→0.008581** | **−0.029309** | **+488 (+0.000325 S rate)** | net **−0.028986 S** |

In v4d-full the shipped beta counts shift to 459/65/76 (from QA66-only 415/84/101) — after dim0+ab
refinement, fewer pairs need the shear (the rungs COMPETE for the same error mass: non-additive
pools law, measured here at the pair level).

## §3 QA65 dim0 offset-coding — the pi2 f16-marginal law CONFIRMED (2nd anchor)

MEASURED over the v4c 600 poses: dim0 mean +31.520, std 2.324; f16 quantum at magnitude **0.0201**
vs at the mean-subtracted residual **0.00104** → **19.3× finer**; dims 1-5 ratio ~1.0× (offset
coding is dim0-ONLY, exactly pi2's prediction). The acceptance lattice IS the storage lattice
(dim0 = offset + f16(residual) round-trips bit-exact; verify B_pose_reconstruct_exact). Realized
full-600: dim0-slice −0.009196 S at +319 B coding cost (offset residuals brotli slightly worse
than raw magnitudes — a real, honest coding tax the win absorbs 30×over).

## §4 QA69 refinement curve + the MacKay pose-floor reading

The integrated refine IS the coarse→fine race (Contrarian brake honored: raced per stream, not
adopted as law). Full-600 curve: `frac_pairs_dim0_improved = 0.92`, dim0 precision gain
−0.009196 S, but the per-pair gain is HEAVILY tail-concentrated (hardest-164 slice: dim0 31%,
photometric 26%, beta 42% of the stage gain; easy-400: near-zero every stage). **Pose-floor
reading: the residual pose contribution (0.2929) is CONTENT/TAIL-limited, not storage-limited —**
after 19.3×-finer dim0 + per-pair photometric + per-pair shear, the mean is dominated by the same
hard tail (MEASURED on the ship field: top-17 pairs = 74.3% of total d̄ mass — Tao's QA48 17-pair
hard core re-surfaced intact through every rung; top-50 = 85.9%, top-100 = 91.1%; median d 0.00088
vs mean 0.00858). This is ja1's #4 ranking (QA65
DERIVED-SMALL) refined: QA65 pays real S (−0.0092) but the remaining axis is the QA68
content-tail, not more precision. QA72a sharpens this attribution below.

## §5 QA72a stage attribution + QA70 min-entropy probe

- **QA72a (ja1 #2, the $0 decider) — FIRED, verdict CONTENT/RE-SOLVE-LIMITED.** Hardest-80
  continuous-dim0 solve (golden-section fp32 through the uint8 render) → re-quantize plain-f16 vs
  offset-f16. MEASURED: recoverable (v4c→continuous) d 0.004927 (S slice 0.0301); f16-storage
  penalty d 0.000518 = **10.5% of recoverable**; offset-f16 penalty d 0.0000823 (the offset
  lattice eliminates ~84% of the storage tax; S_plain→offset 0.00246). **Verdict: pose loss is
  CONTENT/RE-SOLVE-limited, not storage-limited — QA65-as-future-direction DEMOTED (ja1 #4
  CONFIRMED by measurement); the banked v4d-full win was mostly re-solve content; the next pose
  spend belongs to the QA68 tail, storable at plain f16 (offset stays: its −0.0025 S-class tax
  refund is already shipped at +319 B).** Receipt `qa72a_receipt.json`.
- **QA70 (ph3 §1 Kolmogorov) — FIRED, verdict INTRINSIC WHITENESS at this chart.** Hardest-120:
  re-pick dim0 among realized-no-worse optima (accept only d≤plain; mean d actually dipped
  0.04541→0.04477) toward the temporal-neighbor prediction. MEASURED: prediction-residual kl1
  member bytes 260→260 (**entropy reduction = 0**); mean |dim0−neighbor| 1.812→1.797 (**MAD
  reduction 0.77%**). The tie-break cannot pull dim0 toward neighbors even at no distortion cost.
  **Verdict scope (honest bound): LOCAL (±0.05-window, dim0-only) member selection is refuted; a
  non-local/JOINT gauge traverse (dim0×pitch×dz aliasing, sc1 law) remains unprobed. At this
  chart the whiteness is intrinsic — 4th white solver-field receipt; kl1 family verdict extends.**
  Receipt `qa70_receipt.json`.

## §6 BYTE-CLOSE + VERIFY + STAGED GATE

- **v4d grammar PFS1WPD1** (rule-118 FREE): pose_warp.stp = tp_member(kl1 f16 6×n; dim0 column =
  offset residual when manifest `pose_dim0_offset` present) + st_coded(r7) + sel_coded(brotli
  packbits) + ab_member(kl1 f16 2×n) + **beta_coded(brotli n uint8 indices)** [NEW section 5].
  Manifest: `rs_beta_mags=[0,0.5,1.0]`, `pose_dim0_offset=31.515625` (v4d-full). Members
  tokens/renderer/selector/pose_stub = cell_drop50 verbatim (sha-verified identical to v4c ⇒ d_seg
  unchanged); DEFLATE on manifest+selector.sec.
- **VERIFY all_checks_ok (both candidates):** (A) #417 parse-back bijection (off==len; 600×6 pose
  + 600×2 ab + 600 beta + selector); (B) fields BIT-EXACT vs final JSONL incl the dim0-offset
  round-trip; (C) independent recompute byte-exact on sampled pairs INCLUDING a beta≠0 pair (shear
  path exercised) + two-plane genuinely differs; (D) deterministic rebuild → same sha (both).
- **EXACT FIRE COMMANDS (MAIN; fire ONE — v4d-full is the primary):**
```
bash experiments/stage_v4d_realized_gate.sh cpu refine_celldrop50    # PRIMARY  S~0.9640
bash experiments/stage_v4d_realized_gate.sh cpu qa66_celldrop50      # FLOOR    S~0.9796
```
  Verify at the gate (primary): realized d_seg ≈ 0.004312 (±2.8e-5 band) · realized d_pose ≈
  0.008581 (fidelity band ~1e-3 duty) · realized S ≈ 0.9640 < v4c 0.992972.

## §7 HONEST-DEFER (defer-at-source; admission-rule + rate-EV reasons)

- **QA68 +Movable-depth expert (rung-C):** measured MIXED (10/17 improve, 6 degrade) — needs a
  selector; admission marginal by the Hotz rule. v4d's realized menu is already {single,two-plane}
  × {beta 0/0.5/1.0} = a 6-expert per-pair mixture. The +Movable expert + receiver-PREDICTED
  selector are the named next races; the §4 tail-concentration says this content-tail direction
  (not more precision) is where the remaining pose mass lives. DEFERRED with reason, not churned
  into the validated build.
- **QA47 Jacobian-basis rate lever:** pose member ~6.4 KB of 360 KB; a 1-2 coeff/pair basis saves
  ~0.001 S class at a grammar change. Rate-EV small; DEFERRED to a rate slot.
- **QA67 unsolved-350 polish: RETIRED-SUBSUMED** — the v4d refine re-solved ALL 600 pairs
  (dim0+ab+beta); the residual "350 polish" question is now the full 6-DOF re-solve of the easy
  pairs, measured near-zero marginal in this pass.

## §8 Confounds + honesty rails

- **tac HIJACK guarded** (launcher asserts main src/tac). **Ops incident (recorded):** the first
  full-refine launch was killed externally at 570/600 (no traceback; SIGURG-orphan class suspect);
  the resumable JSONL made the loss zero — relaunch etiquette briefly created a DUPLICATE resolve
  (pgrep matched a zsh snapshot, not the python), killed within seconds; 1 benign duplicate row
  (last-wins load). Lesson: liveness checks must match the PYTHON pid, not `pgrep -f` string.
- **d_seg UNCHANGED by construction:** members sha-identical to v4c (checked all four).
- **QA66-only pose d̄ is EXACT** (realized in the v4c photo pass; no new evals).
- **v4d-full is monotone-safe:** per-pair best-of(refine, QA66-floor); no pair ships worse than
  its floor.
- **Advisory everywhere:** frozen-PoseNet, macOS-CPU, non-promotable, `score_claim=false`.
- **Verdict scope:** QA65/QA66 wins INSTANCE (this vehicle/base/scorers); pi2 f16-marginal law now
  FORMULATION-level (pi2 predicted → v4d measured 19.3× + realized −0.0092 S); the
  rungs-compete-for-error-mass observation (beta counts 415→459 b0) is another instance of the
  standing non-additive-pools law.

## §9 Ledger routing (defer-at-source; patched same commit)

- QA65 FIRED (dim0 offset lattice shipped + re-solved; −0.009196 S isolated slice, +319 B).
- QA66 FIRED (per-pair beta member shipped; −0.013485 S realized at +140 B; in v4d-full the joint
  re-select yields 459/65/76).
- QA69 FIRED-as-instrument (the refinement race ran; verdict: pose is content/tail-limited past
  the offset lattice — fixed-finer-quantum beyond dim0-offset does NOT clear water; per-stream
  falsifier honored).
- QA72a FIRED (hardest-80 stage attribution: storage fraction 10.5%; verdict
  CONTENT/RE-SOLVE-LIMITED; QA65-as-future demoted, ja1 #4 confirmed).
- QA70 FIRED (hardest-120 local gauge probe: entropy reduction 0 B, MAD −0.77% at no-worse d;
  intrinsic whiteness at this chart; non-local/joint traverse remains the only open gauge door).
- QA68 (+Movable expert / predicted selector) DUE — named the top pose direction by §4. QA47
  rate lever DEFERRED. QA67 RETIRED-SUBSUMED.
- NEW: **QA78 v4d gate FIRE (MAIN, DUE)** — primary `refine_celldrop50` sha f1f32880…, floor
  `qa66_celldrop50` sha d5149d81… (QA71-77 already minted by ja1/parallel arms; collision checked).
