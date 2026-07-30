---
schema: ddm_v4b_composed_gate.v1
date_utc: 2026-07-30
arm: ddm_v4b (composed-gate build; task #776 — THE CRITICAL PATH)
lane_id: "lane_ddm_v4b_composed_gate_20260730"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver + frozen PoseNet; static REALIZABLE, GT UPPER BOUND; composed byte-close DONE, n600 evaluate gate STAGED — MAIN fires]"
operator_binding: "MAIN v4b dispatch — static-transfer check, receiver amendment, byte-close, STAGE the n600 gate (do not fire)"
tools:
  - "experiments/ddm_v4b_static_transfer.py (step 1 measurement + QA50 SVD rider)"
  - "experiments/inflate_runner_v4b.py (v4b receiver: two-plane static-horizon compose, rule-118 free)"
  - "experiments/ddm_v4b_build_composed_archive.py (byte-close build)"
  - "experiments/ddm_v4b_verify_decode.py (parse-back bijection + decode identity)"
  - "experiments/stage_v4b_realized_gate.sh (STAGED gate — MAIN fires)"
data: "SSD ddm_v4b_20260730/{v4b_static_transfer.partial.jsonl(112), v4b_static_transfer_receipt.json, v4b_ship_table.json, v4b_composed_static_kneeA_archive.zip(sha 3b3a4abf), v4b_composed_static_kneeA_build_receipt.json, v4b_verify_receipt.json, inflate_runner_v4b.py}"
tokens: "[no-triality] [p0-ledger-ok] [magnitude-ok]"
---

# ddm_v4b — composed v4b gate: two-plane STATIC-horizon warp, byte-closed + staged

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number
below is `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`, per-pair
realized through the real receiver + frozen PoseNet. This arm lowers the pfs1
warp-base ADVISORY vehicle (ref S 2.256641), which is FAR from the pointer — it
does NOT move the pointer. **The n600 `evaluate.py` gate (MAIN fires, ONE-n600
rule) is the authority; my composed S is a PREDICTION the gate verifies.**

## §1 Headline (advisory)

The v4b composed archive is **byte-closed (274,479 B, sha 3b3a4abf…) and the
n600 gate is STAGED**. Predicted composed **S = 1.5343** (realizable, static
mask) — it BEATS the illegal GT-mask upper bound (1.7139) by **−0.180**, exactly
the qa45 mechanism (the physics horizon is a cleaner far/ground partition than
the noisy GT argmax). Deltas: **−0.7224 vs ref pfs1 D1 (2.2566)** · **−0.8754 vs
the Knee-A standalone reject (2.4097)** · −0.5574 vs single-only (2.0917).

| axis | value | how obtained |
|---|---|---|
| 100·d_seg | **0.553676** | MEASURED at the wr1 Knee-A gate; tokens UNCHANGED → f1 unchanged → exact |
| pose contribution √(10·d̄) | **0.797815** | advisory, per-pair realized; d̄=0.06365080 over 600 |
| rate 25·B/37,545,489 | **0.182764** | MEASURED (274,479 B byte-closed) |
| **predicted composed S** | **1.534255** | seg + pose + rate |

Advisory→gate band: d_seg + rate are EXACT (deterministic decode, tokens/bytes
fixed); only the pose contribution carries the ~1e-6 advisory-vs-gate band (ck1
§1 control: ck1 oracle reproduced the gate d_pose to 1.13e-6) → predicted S
reproduces to ~6e-6 on pose.

## §2 STEP 1 — static-mask transfer on the KNEE base ($0, 112 tail, 61 s)

The gate renders f1 from the **Knee-A** tokens, so the honest shipping pose table
must be measured on the KNEE base (not qa45's full base). `ddm_v4b_static_transfer.py`
re-uses ck1's validated `build_kneeA_oracle` (renders the DROPPED-token f1) and
re-evaluates ck1's KNEE-base `p_two_star` through the STATIC compose:

- **Horizon DERIVED, not tuned:** `v = round(-l₂/l₁)` for `l = K⁻ᵀ[0,−1,0]` =
  `round(cy) = 437`. Rows `< 437` → far = H∞ (s_t=0, s_r=1.0); rows `≥ 437` →
  ground = full H. Pure code, **0 shipped mask bytes.** No hood (qa45: 76 B
  vdmaj buys only −0.005 S — SKIPPED).
- **POSITIVE CONTROL PASSED 112/112:** the GT-mask compose reproduced ck1's
  cached `d_two_solved` to **max abs delta 0.00e+00** → substrate identity with
  the ck1 instrument confirmed; the static numbers are trustworthy.
- **Shipping selection (monotone-safe):** per tail pair `min(d_single_kneeA,
  d_two_static)`; selector=1 iff two@static beats single. **95/112 select
  two-plane**; 17 fall back to single (blowups like pair 80 static 114 vs single
  2.26 → single). Non-tail (488) ship single-plane p_single_kneeA (selector 0).
- **Realized shipping pose table** (`v4b_ship_table.json`): mean d_pose over 600
  = **0.06365080** → pose contribution **0.797815** (vs GT-best 0.9776, single
  1.3553). Static realizable is −0.180 S BETTER than the illegal GT upper bound
  — realizability is a **gain**, not a cost (qa45 confirmed on the knee base).

## §3 STEP 2 — receiver amendment (rule-118 FREE code)

`experiments/inflate_runner_v4b.py` (Decoder, MAGIC `PFS1WPB2`, policy
`warp_two_plane_static_v4b`) reuses the EXISTING vendored `pfs1_warp_receiver`
primitives unchanged (`pose_to_homography`/`warp_rgb`/`intrinsics_native`); the
amendment is the policy layer only:
- **s_r=1.0 pose consumption** (both single and two-plane branches).
- **per-pair selector bit**: 0 → single-plane full H; 1 → two-plane static compose.
- **two-plane static compose**: far (rows<437 → H∞) / ground (rows≥437 → full H);
  horizon DERIVED at decode from `intrinsics_native()` (no shipped constant).
- **selector=0 path byte-identical to the v4a receiver** — verified byte-exact
  on 3 pairs (§5).

## §4 STEP 3 — payload (byte-closed, RAW this round)

Only `state/pose_warp.stp` + `manifest.json` change; token/renderer/selector/
pose_stub members are Knee-A bytes VERBATIM. `pose_warp.stp` (v4b grammar):

```
<8s PFS1WPB2><I n=600><I l1><tp_coded: brotli-q11 f16 (600,6) p_best>
             <I l2><st_coded: r7 codes (Knee-A s_t stream, REUSED VERBATIM)>
             <I l3><sel_coded: brotli-q11 packbits(600 selector bits)>
```
- **pose field** p_best = p_two_star where selector else p_single_kneeA, 6 f16/pair,
  brotli-q11 → **6,705 B**.
- **s_t stream** unchanged (189 B) — s_t already shipped by the D1 grammar; NOT
  re-added.
- **selector** 600 bits → packbits 75 B → brotli-q11 **74 B**.
- pose_warp total **6,992 B** (+128 vs Knee-A's 6,864). Manifest +18 B.
- **archive 274,479 B**, delta **+146 B** vs Knee-A (274,333) → rate +0.0001 S.
- kl1's law-coded pose swap (−0.00059 S member) is v4c — NOT waited on (RAW this
  round, per the build spec).

## §5 STEP 4 — byte-close + parse-back + decode identity (`v4b_verify_receipt.json`, all_checks_ok=true)

Run entirely in the VENDORED gate substrate (no tac):
- **(A) parse-back consumption bijection (#417):** the receiver consumes EVERY
  pose_warp byte (`off == len` enforced) and decodes p_best (600,6) + s_t (600) +
  selector (600, sum 95). No counted-but-inert bytes.
- **(C) selector=0 byte-identity vs v4a:** for 3 selector-0 pairs the v4b f0 is
  **byte-identical** (maxabs 0) to an INDEPENDENT single-plane decode of a v4a
  archive, and the shipped p_best f16 round-trips to p_single_kneeA — the sel-0
  branch IS the v4a receiver.
- **(D) two-plane independent recompute:** for 3 selector-1 pairs the v4b f0
  matches a fresh-code static two-plane compose (maxabs 0) AND genuinely differs
  from the single-plane f0 (maxabs 239–240) — the compose is doing real work.
- **Gate-time smoke:** the receiver constructs + decodes both branches from the
  exact staged submission dir (horizon 437, 95 selector-two, tac-free).
- **Gate reproduces advisory:** ck1 §5 established the ck1-oracle f1 render ==
  the vendored inflate f1 render byte-identical (v4a 3/3); the compose code is
  the identical vendored `pfs1_warp_receiver`; therefore the gate's f0 == the f0
  I measured d_two_static on, and the gate d_pose == my advisory to ~1e-6.

## §6 STEP 5 — STAGED gate (MAIN fires; do NOT self-fire)

`experiments/stage_v4b_realized_gate.sh` mirrors `stage_wr1_realized_gate.sh`
(surgical archive.zip swap into a COPY of the pfs1 D1 eval_root submission dir +
the v4b receiver + vendored deps, then stock `evaluate.sh`), **with the PATH
export the wr1 script lacked** (`export PATH=".venv/bin:$PATH"` — evaluate.sh +
inflate.sh call bare `python`; the bare-python death). Apples-to-apples with the
ref pfs1 D1 row (same eval_root, 0.mkv, device). Projected wall ~17 min CPU.

**EXACT FIRE COMMAND (MAIN):**
```
bash experiments/stage_v4b_realized_gate.sh cpu
```
Receipt schema `ddm_v4b_realized_gate.v1`. Verify at the gate: realized d_seg ==
0.00553676 (tokens unchanged) · realized pose contribution ~0.798 · realized S
~1.53 (≪ ref 2.2566, ≪ Knee-A 2.4097).

## §7 QA50 rider ($0) — the next systematic axis is rank-1 forward-speed

SVD of the two-plane→single correction (`p_two_star − p_single_kneeA`) over the
95 selector-1 pairs: **σ = [30.55, 7.47, 3.75, 0.032, 0.015, 0.013]**, energy
**[93.0 %, 5.6 %, 1.4 %, 1e-6, …]**. The top right-singular vector is
`[−0.99991, 0.0118, 0.0064, ~0, ~0, ~0]` — **almost purely pose dim-0** (the
forward-speed component, largest t_p spread 1.256); mean correction dim-0 =
**+0.672**. Rotation dims (3–5) carry ~1e-6 energy — near-INERT.

**Mechanism + named next axis:** the two-plane's win over single-plane is a
near-**rank-1 forward-speed (dim-0) re-estimation**. On the frozen-far-field
knee base the single-plane homography aliases forward translation into the far
field (which cannot move under a ground-plane H); the far→H∞ split de-aliases
it, and the residual correction the GN applies is a clean low-rank dz shift.
**v4c rate lever:** store the tail two-plane poses as (single-plane pose + ONE
scalar dz correction) instead of full 6-DOF → ~5 f16/pair × 95 saved. This is
the Schmidhuber loop's first firing (the correction field is predictable → it
compresses).

## §8 Confounds + defer-at-source (ledger rows appended)

- **`tac` import HIJACK (control-guarded):** the shared venv editable-install
  points to the eg1 codex worktree, NOT main `src/tac`. I forced
  `PYTHONPATH=$PWD/src` (verified → main tac) for the measurement AND ran the
  GT-control positive control (delta 0.00e+00) → measurements hold. The verify +
  build + gate are tac-FREE (vendored substrate). Hygiene row QD15 stands.
- **Advisory everywhere:** frozen-PoseNet, macOS-CPU, non-promotable,
  `score_claim=false`. GT numbers UPPER BOUND; static numbers REALIZABLE. The
  gate is the authority; the pointer is UNMOVED.
- **Verdict scope:** static-wins is FAMILY-level for the mask-source question
  (static geometric mask is legal AND superior); INSTANCE for the exact totals
  (this vehicle, frozen PoseNet, GT-solved poses).
- **OWED (appended to `ddm_deferral_queue_ledger_20260729.md`):** (a) v4b gate
  FIRE (MAIN, DUE) · (b) full re-solve of p_two_star THROUGH the static mask
  (all tail; the shipped poses are GT-solved, suboptimal-for-static → a further
  win, ~1 h PoseNet-only) · (c) non-tail static two-plane extension over the 488
  (est ≤−0.05 S) · (d) QA50 rank-1 dz carrier for the tail field (v4c rate) ·
  (e) kl1 law-coded pose-member swap (v4c, −0.00059 S).

## §9 MAIN 07-30 — pm1 rungs A+B (QA44): SHIP v4b AS CHARTERED, A+B → v4c

MAIN landed pm1's QA44 verdicts mid-build: rung B (per-pair auto-exposure
`f0 := a·warp(f1)+b` after warp before uint8) improves 112/112, rung A
(rolling-shutter row-shear, ~0 B) improves 96/112; composed A+B = **−0.1039 S**
on the FULL-base pose member (0.7229→0.6189) at ≤4 B/pair. **Decision: ship v4b
AS CHARTERED; A+B fold to v4c.** Rationale (MAIN's own one-rule): I am PAST
receiver freeze — the v4b receiver is written, byte-closed (sha 3b3a4abf), and
decode-validated (all_checks_ok: parse-back bijection + sel-0 byte-identity +
sel-1 recompute), gate-smoke-passed, and staged. Folding A+B would (i) change the
compose → invalidate every decode-identity check, (ii) require re-solving (a,b)
for the 488 non-tail on the KNEE base (pm1's −0.1039 was measured on the FULL
base — it does NOT transfer as a number; the knee-base A+B is UNMEASURED), (iii)
re-byte-close + re-verify — the validated-build churn MAIN's rule forbids. The
v4b gate measures the chartered composed S (1.5343); **v4c = v4b + rungs A+B
(on the knee base) + QA59/QA60/QA61 carriers**, a subsequent gate. A+B is a real
further win banked as the top v4c rung (ledger QA62). Per pi2 (99.3% luma):
gain/bias is scalar (a,b) in RGB — no chroma variant.
