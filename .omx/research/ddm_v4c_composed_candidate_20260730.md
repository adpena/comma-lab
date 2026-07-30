---
schema: ddm_v4c_composed_candidate.v1
date_utc: 2026-07-30
arm: ddm_v4c (composed candidate v4c — the critical-path successor to the MEASURED v4b gate)
lane_id: "lane_ddm_v4c_composed_candidate_20260730"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver + frozen PoseNet; composed byte-close DONE, n600 evaluate gate STAGED — MAIN fires]"
operator_binding: "MAIN v4c dispatch — base adjudication x pose re-solve x photometric rungs x lossless trio; STAGE the gate (do not fire)"
tools:
  - "experiments/ddm_v4c_resolve.py (transfer/solve/photo, generic-base oracle)"
  - "experiments/launch_v4c_detached.sh (ppid-1 detach + tac-hijack guard)"
  - "experiments/inflate_runner_v4c.py (v4c receiver: static two-plane + photometric, rule-118 free)"
  - "experiments/ddm_v4c_build_composed_archive.py (byte-close build, kl1 codec + DEFLATE container)"
  - "experiments/ddm_v4c_verify_decode.py (parse-back bijection + decode identity + shear path)"
  - "experiments/ddm_v4c_qa58_exposure_race.py (QA58 $0 coding race)"
  - "experiments/stage_v4c_realized_gate.sh (STAGED gate — MAIN fires)"
data: "SSD ddm_v4c_20260730/{transfer_celldrop50.partial.jsonl(600), solve_celldrop50.partial.jsonl(250), photo_celldrop50_resolve.partial.jsonl(600), receipts, v4c archive+build+verify receipts, qa50_rider + qa58 race}"
tokens: "[no-triality] [p0-ledger-ok] [magnitude-ok]"
---

# ddm_v4c — base-adjudicated composed candidate: cell_drop50 x static re-solve x photometric x lossless trio

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number
below is `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`, per-pair
realized through the real receiver + frozen PoseNet. This arm lowers the pfs1
warp-base ADVISORY vehicle (v4b measured 1.534258), which is FAR from the
pointer — it does NOT move the pointer. **The n600 `evaluate.py` gate (MAIN
fires, ONE-n600 rule) is the authority; my composed S is a PREDICTION the gate
verifies** (v4b precedent: prediction agreement 3e-6).

## §1 Headline (advisory)

The v4c composed archive is **byte-closed (359,750 B, sha b6365270…), verified
(all_checks_ok), deterministic-rebuild-proven (same sha), and the n600 gate is
STAGED**. Predicted composed **S = 0.99283** — **−0.5414 vs the MEASURED v4b
gate 1.534258 (−35%)** · −1.2638 vs ref pfs1 D1 2.256641 (−56%). First sub-1.0
composed operating point on this vehicle line.

| axis | value | how obtained |
|---|---:|---|
| 100·d_seg | **0.43104** | gr1 n600 MEASURED on cell_drop50 (0.004310; evaluate band ±2.8e-5 → ±0.0028 S) |
| pose contribution √(10·d̄) | **0.322248** | advisory, per-pair realized (d̄ 0.0103844 over 600; re-solve + rung-B; gate band ~1e-6 per ck1 control + v4b 3e-6 precedent) |
| rate 25·B/37,545,489 | **0.239543** | MEASURED (359,750 B byte-closed) |
| **predicted composed S** | **0.992834** | seg + pose + rate |

Per-stage delta decomposition vs v4b (measured 1.534258 → predicted 0.992834):

| stage | Δ axis | ΔS |
|---|---|---:|
| base swap Knee-A→cell_drop50 (gr1) | seg −0.12264 · rate +0.05643 | **−0.0662** |
| pose transfer damage (base swap) | pose 0.79781→0.85920 | +0.0614 |
| pose re-solve through static mask (QA59+QA60) | pose 0.85920→0.41141 | **−0.4478** |
| photometric rung-B (a,b) (QA62) | pose 0.41141→0.32225 | **−0.0892** |
| (a,b)+selector members + DEFLATE container | rate +529 B net | +0.0004 |
| **net** | | **−0.5414** |

## §2 STAGE 1 — BASE ADJUDICATION (MEASURED; cell_drop50 WINS)

Candidates: Knee-A (v4b shipped; pose SOLVED) vs gr1 cell_drop50 (359,221 B @
n600 d_seg 0.004310; seg+rate 0.6702 vs Knee-A 0.7365 = −0.066, but |g|-sum
ordering is seg-only/pose-BLIND → pose damage unknown).

**The cheap measurement:** transfer the v4b shipped pose field (600×6 f16 +
selector) onto the cell_drop50 base through the REALIZABLE static two-plane
compose (`--mode transfer`), full 600:

- **Positive control (Knee-A):** reproduces the v4b ship table `d` exactly
  (pair-level bit-agreement; substrate identity with the gate instrument).
- **cell_drop50 pre-resolve: mean d_pose 0.07382** (contribution 0.8592) vs
  Knee-A 0.06365 (0.7978) — damage only **+0.010 d_pose / +0.061 S**, the
  RECOVERABLE class (ck1 precedent: stale-params damage, not capability loss;
  the gr1 drop was seg-|g| ordered yet the pose cost is tiny because the ship
  poses were solved through a two-plane compose that is largely
  frozen-far-field-invariant by construction).
- **Arithmetic at decision time:** even PRE-resolve, cell_drop50 composed
  (0.4310 + 0.8592 + 0.2392 = 1.5294) ≈ v4b (1.5343); with re-solve recovery
  → ~1.468. **cell_drop50 wins the base; Knee-A retires as the v4b row.**

## §3 STAGE 2 — POSE RE-SOLVE THROUGH THE STATIC MASK (QA59+QA60, MEASURED)

Re-solve per pair THROUGH the realizable static compose (far = rows<437 → H∞ /
ground → full H; horizon DERIVED at decode; NOT the GT mask the v4b poses were
solved on), best-of {single-plane 6-DOF GN, two-plane static GN (starts: ship
pose + single-static solution)}, f16-monotone acceptance, resumable JSONL,
detached ppid-1. Solved the 250 hardest by transfer damage; two-plane enabled
for all 250 (QA60: two-plane extends beyond the v4b tail-112).

- **Solved-250: mean d_pose 0.13730 (transferred ship) → 0.02732 re-solved
  (−0.110); 188/250 select two-plane** (vs v4b's 95/600 — QA60 confirmed: the
  two-plane pays far beyond the old tail).
- **HONEST composed pre-photo (solved-250 + transfer-350): mean d_pose
  0.016926 → contribution 0.41141.** Pre-photo composed arithmetic:
  0.43104 seg + 0.41141 pose + 0.23919 rate = **1.0816** (−0.4526 vs v4b).
- Mechanism (same as ck1/qa45): the v4b ship poses were GT-mask-solved and
  Knee-A-base-solved — doubly stale for this receiver; re-solving through the
  ACTUAL static compose on the ACTUAL base recovers nearly all of it. The
  static mask is again not a cost: it is the exact physics the solve exploits.

## §4 STAGE 3 — PHOTOMETRIC RUNGS (QA62, MEASURED) — TO FILL

Per pair at the final pose: rung-B auto-exposure (a,b) 2-param GN (f16-quantized
re-score, honest), then rung-A rolling-shutter row-shear restricted to the
SHIPPABLE candidates only (beta = g·sign(yaw), g ∈ {0.5, 1.0}; sign derived
free at decode from pose dim-5; g is ONE global manifest constant — the
per-pair-best beta of pm1's probe is NOT shippable at 0 bytes and is not used).
Per-g d recorded per pair → the build picks the global g* by exact arithmetic.

**MEASURED (600 pairs, receipt `photo_celldrop50_resolve_receipt.json`):**
- **rung-B: mean d_pose 0.016926 → 0.010384, contribution 0.41141 → 0.32225
  (−0.0892 S)** at 1,820 counted (a,b)-member bytes (+0.0012 S rate) — the pm1
  family verdict CONFIRMED on the re-solved knee-successor base, now over the
  full 600 (pm1's −0.1039 was full-base tail-only; same magnitude class).
- **rung-A global: g\* = 0.0 — HONEST NEGATIVE.** 600-means: g=0 → 0.010384,
  g=0.5 → 0.011843, g=1.0 → 0.016223. The shippable one-global-constant
  rolling-shutter shear HURTS on net at the re-solved poses (pm1's 96/112
  improvement was per-pair-best beta, which needs per-pair sign+magnitude
  bits). Per-pair-best over all 600 = 0.009533 (0.30876, a further −0.0135 S)
  at ~2 bits/pair ≈ 150 B (+0.0001 S) — a POSITIVE trade, but a grammar change
  past receiver freeze → routed to v4d (ledger row), not churned into v4c.
- a ≈ 1 for 99.7% of pairs (median 1.0, 85 exact-1); b is the live axis
  (median +0.02, range dominated by hard pairs) — kl1's preliminary law holds.

## §5 STAGE 4 — LOSSLESS TRIO (kl1) + container

- Pose field → kl1 byte-plane codec member (KL1PWF01, bit-exact, rule-118-free
  receiver decode).
- Selector → brotli packbits (75 B raw; colex parked — see §8 honesty).
- (a,b) exposure field → kl1 byte-plane member.
- Container: DEFLATE on manifest.json + selector.sec (receiver reads the
  UNZIPPED dir; ZIP method transparent; rate = archive.zip size).
- NOT shipped (measured-worse per kl1): spline/AR/rank-1 pose codes; any QA61
  dz-carrier (§7: the standardized SVD kills it on this field too).

## §6 BYTE-CLOSE + VERIFY + STAGED GATE (all PASS)

- **Build** (`v4c_composed_static_photo_celldrop50_build_receipt.json`):
  **359,750 B, sha b6365270ddc55fde…**, +529 B vs the cell_drop50 base.
  pose_warp.stp 8,175 B = tp kl1-member 6,059 + st 189 (verbatim) + selector
  79 + (a,b) kl1-member 1,820; manifest+selector.sec DEFLATE'd (container win
  offsets most of the (a,b) cost). rs_beta_global=0.0. Members
  tokens/renderer/selector/pose_stub = cell_drop50 bytes verbatim.
- **Verify** (`v4c_verify_receipt.json`, **all_checks_ok=true**): (A) #417
  parse-back bijection (600×6 poses + 600×2 ab + 224 two-select, off==len);
  (B) decoded fields BIT-EXACT vs the photo JSONL (poses/ab/selector); (C)
  independent fresh-code recompute byte-exact on 6 sampled pairs + two-plane
  genuinely differs from single; (D) deterministic rebuild → **same sha**
  (re-ran the build end-to-end; b6365270… reproduced).
- **Stage smoke (exact gate layout):** staged RUN_SUB replica (template deps +
  v4c receiver as `inflate_runner.py` + unzipped archive) decodes pairs
  0/44/300 through the real entry path — `STAGE_SMOKE_OK`.
- **EXACT FIRE COMMAND (MAIN):**
```
bash experiments/stage_v4c_realized_gate.sh cpu static_photo_celldrop50
```
  Verify at the gate: realized d_seg ≈ 0.004310 (gr1 band ±2.8e-5) · realized
  pose contribution ≈ 0.3222 (band ~1e-6) · realized S ≈ 0.9928 ≪ v4b 1.5343.

## §7 QA50 rider — the re-solve correction field has NO low-rank law (standardized)

SVD of (p_best_static − p_ship) over the solved 250: raw energy [0.921, 0.068,
0.011, ~0] LOOKS rank-1 dim0 — but **standardized energy = [0.221, 0.180,
0.171, 0.158, 0.139, 0.130] = FLAT/isotropic**: the exact kl1-B3 scale
artifact, re-measured on the NEW field. Per-pair GN corrections are white
across dims once scale-normalized. **QA61's rank-1 dz carrier premise does NOT
transfer to the v4c correction field** — closes QA61 for this vehicle
(INSTANCE; consistent with kl1's FAMILY-level falsification for solver-output
fields).

## §8 Confounds + honesty rails

- **tac HIJACK guarded:** every launch through `launch_v4c_detached.sh` which
  asserts `tac.__file__` resolves to main `src/tac` (bare venv still points at
  the eg1 worktree — hygiene row QD15 stands).
- **d_seg on cell_drop50 is NOT re-measured here** (bounded-scorer etiquette;
  gr1's n600 evaluate-validated render band ±2.8e-5 is the citation); the gate
  measures it exactly.
- **Selector coding:** shipped as brotli-packbits (74-80 B), not colex (~47 B)
  — the 188/250+ two-selector set is denser than v4b's 95; colex advantage
  shrinks and the packbits path is the decode-proven one. Parked (≈30 B).
- **Unsolved-350 pose:** carries the TRANSFER-measured d on this base (not the
  Knee-A d) — the honest per-pair realizable value; further single-static
  re-solve of the 350 is a residual lever (parked, diminishing: their sum is
  3.32 of 10.16 total pre-photo).
- **Advisory everywhere:** frozen-PoseNet, macOS-CPU, non-promotable,
  `score_claim=false`. The gate is the authority; the pointer is UNMOVED.
- **Verdict scope:** base-adjudication + re-solve wins are INSTANCE (this
  vehicle, these bases, frozen scorers); the static-resolve-recovers law is
  now 2-for-2 (ck1 Knee-A, v4c cell_drop50) — FORMULATION-level for this
  receiver family.

## §9 QA58 exposure race ($0, FIRED) + owed items stated plainly

**QA58 verdict: NO-LAW.** The full-600 (a,b) stream is temporally WHITE
(std(diff)/std = 1.42 for a, 1.37 for b); raw byte-plane 1,804 B beats
lossless delta-u16 1,932 and AR(1)-residual 1,935. The camerad-AE
smooth-control-loop hypothesis is falsified at the stream level — third white
solver-output field (pose, tail-correction, now exposure); kl1's family
verdict extends. Receipt `qa58_exposure_race.json`.

**Owed / not done this round (stated plainly, defer-at-source):**
- **dim0 offset-coded lattice (kl1 §9 / pi2, ~0.03 S potential): NOT
  implemented.** The charter asked the re-solve to ride the 16×-finer dim0
  quantum; my solve accepted on the plain f16 lattice (the ck1-proven
  instrument). Folding it now = pose-member grammar change past receiver
  freeze + a ~1.5 h re-solve → the exact validated-build churn the v4b §9
  one-rule forbids. Routed to v4d (ledger row), with the honest note that the
  (a,b) member already absorbs part of the output-side dim0 error.
- **Per-pair rung-A beta (−0.0135 S at ~150 B): positive trade, grammar
  change → v4d** (§4).
- **Unsolved-350 single-static polish** (their pre-photo sum 3.32/10.16;
  photo already ate part) — diminishing, parked.

## §10 Ledger routing (defer-at-source; patched in the same commit batch)

- QA59 FIRED (re-solve through the static mask on the ADJUDICATED base).
- QA60 FIRED (two-plane beyond the tail: 188/250 solve-two, 224 shipped).
- QA62 FIRED (photometric fold: rung-B shipped; rung-A global g*=0 negative).
- QA58 FIRED (NO-LAW; raw byte-plane).
- QA61 CLOSED-NEGATIVE for the v4c field (standardized-SVD isotropic, §7).
- NEW: QA64 v4c gate FIRE (MAIN, DUE) · QA65 dim0 offset lattice (v4d) ·
  QA66 per-pair rung-A beta member (v4d).
