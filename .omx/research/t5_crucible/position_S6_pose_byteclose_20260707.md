# T5 CRUCIBLE — SEAT S6 POSITION — POSE + BYTE-CLOSE + EXACT-PATH (apparatus charter)

**Seat:** S6 · **Date:** 2026-07-07 · **Anti-anchoring honored:** no other position_S*.md read.
**Axis discipline:** every number below is tagged. All byte-close rows run TODAY are
`[macOS-CPU advisory] NON-PROMOTABLE` for score semantics, but the **archive bytes are exact**
(a zip stat is host-independent). Pointer **0.19110 UNMOVED** — everything here is MEANS.

**NEW MEASURED ROWS (this seat, today, inline <10 min each):** the FIRST byte-close of the
council-designed clean baseline (mod32cap ep650 BEST), in three compositions, all with the
2-pair bit-exact decode gate PASSING:

| composition | archive.zip | rate_term | marginal | bit-exact |
|---|---:|---:|---:|---|
| weights-only (111,095 params, int8+brotli) | **83,430 B** | **0.05555** | — | PASS |
| + lane render band (LBND2 RD, u_mask) | 125,267 B | 0.08340 | **+0.02767** | PASS |
| + store-nothing pose (#257 derive-H, ξ delta_ar q4096) | 90,393 B | 0.06018 | **+0.00464** | PASS |

Reports: `reports/t5_s6_byteclose_mod32cap_ep650_{weightsonly,band,posesn}_20260707.json`.
Full Arm-A stack projection (weights + band + pose): ≈ 131.9 KB → **rate ≈ 0.0879** (LBND2)
or ≈ 0.0808 (LBND4, decode build owed — see G2).

---

## Position

### P1 — §5B: POSE **ON** for the derived-optimal run (two-track), and pose is the SECOND WALL, not a line-item

**Decision: (c)-as-two-track = the dossier's recommendation, accepted with sharpened thresholds.**
Diagnostic A/B arms stay seg-only (`w_pose=0`, attribution purity). The derived-optimal
pointer-moving run trains **pose ON: `w_pose > 0` + FiLM-on-ξ + store-nothing ξ carrier
(#257 derive-H) + ξ-consistent null-texture (the L3 mechanism)**, staged-on at the tau-stage
boundary with a warmup ramp, #227 (seg⊥pose decoupling) engaged.

**The score-law arithmetic that forces this (DERIVED from `S = 100·d_seg + √(10·d_pose) + rate`):**

- **Pose OFF is not a submittable row.** mod32cap's measured pose-blind operating point is
  d_pose 125.833 (`costate_shadow.jsonl`, dossier §5B) → pose term √(10·125.833) ≈ **35.5**.
  A stored sidecar does NOT fix it — the scorer runs PoseNet on the FRAMES (byte-close tool's
  own pose_note, VERIFIED-VIA-SOURCE below). Pose must be TRAINED-IN.
- **The best MEASURED store-nothing floor is itself insufficient.** R1 (#245) measured the
  trained store-nothing pose descending through-R: d_pose 62.44 → **0.0011 plateau**
  (ep1074/1093), d_seg HELD ~0.0046 (DAG FEED rows at sub015 DAG:7496–7527). Term =
  √(10·0.0011) = **0.105**. At the T_3 working point (d_seg 0.00092 → 0.092; rate 0.0602
  measured today) that gives S ≈ 0.092 + 0.0602 + 0.105 = **0.257 — above even T_1.**
- **Feasibility thresholds (this seat's derivation, from today's measured rate):**
  - T_1-crossing (S < 0.19110) at d_seg 0.00092 / rate 0.0602 → pose term ≤ 0.0389 →
    **d_pose ≤ 1.51e-4** (R1's floor misses this by ~7×).
  - T_3 (S < 0.15) per dossier arithmetic (d_seg 0.00077 → 0.077; rate ~0.055) → pose term
    ≤ ~0.018 → **d_pose ≤ 3.2e-5**.
  - **Headline: pose is the SECOND wall.** Even a perfect d_seg run cannot cross the pointer
    unless the pose mechanism beats R1's measured floor by ~7×. §5B is not a side-question.
- **Why beating R1 by 7× is credible (not vibes):** R1 trained with **w_pose=0** — "the null
  was never used for pose" (pose-carrier optimal-form symposium §8,
  `council_pose_carrier_optimal_form_symposium_20260703.md`). The L3 mechanism (w_pose>0 +
  FiLM consuming ξ + ξ-consistent texture painted into the 99.95% seg⊥pose null, #206) is the
  designed, unfired lever. Its achievable floor is UNMEASURED — that is exactly what the
  bounded pose-ON smoke measures (R-3/R-5 below). If it floors above 1.5e-4, the symposium's
  L1 fallback (Jacobian-coefficient inverse-solve, ~0.010–0.020 rate, $0 gate defined in
  symposium §6) activates.

**Costs of pose ON (all measured, no vibes):**
- **Bytes:** +6,929 B section → **+0.00464 rate** (MEASURED today; coded ξ 6,367 B delta_ar
  @ q_levels 4096, H_bytes = 0 — derive-H live). q_levels is an unswept rate knob (R-4).
- **Training time: NEGATIVE.** The pose stack is a measured **speed-SAVER**: removing it made
  the step +0.71 s/step SLOWER (P1 ablation, `per_lever_compute_audit_20260705.md` §P1 —
  the carrier warp replaces a second full INR render). "Never drop pose to save time."
- **Risk (the pose-collapse history, correctly attributed):** the d_pose 2.67–12.66 collapse
  was the **amortized-luma-carrier** composition (CLAUDE.md §Pose-is-SOLVED note; memory L68);
  the catastrophic 3.7–10.3 START d_pose was **warp-real-luma** (L68/L69 — EXCLUDED, wrong
  mover, byte-close docstring: adds counted bytes without lowering realized d_pose). Neither
  is the store-nothing mechanism. Store-nothing's measured interference risk is LOW: R1 held
  d_seg ~0.0046 while d_pose fell 62.44→0.0011; seg⊥pose is 99.95% null (#206). The remaining
  confound is the FiLM machinery itself → handled by the seg-only twin arm (two-track).

**Staging (Q12/Q13):** w_pose staged-on at the tau-stage boundary (not ep0): the CE stage is
where the partition births and clean d_seg attribution against the seg-only twin matters most
early; seg⊥pose nullity makes late engagement cheap. Warmup ramp over the stage-transition
re-warmup window (the existing `--stage-transition-rewarmup-*` machinery, no new flags).
#227 gates the run (engage the decoupling port); #248 P-B FiLM read-back fires as the bounded
probe BEFORE the full run (R-5). Fixed-linear w_pose near d_pose→0 (#302 row 24, the
5/√(10·d_pose) divergence): defer a derived w_pose schedule to the costate DECIDE layer
(interface to S3) rather than hand-tuning — the λ-marginal signal is exactly this derivative.

### P2 — Byte-close readiness audit: the tool byte-closes the Arm-A stack TODAY, with 5 named gaps

**What works today (verified by execution, not by reading):**
- **Weights + codes:** int8+brotli 0.bin, accounting matches canonical `quantize_levelset_blob`
  (no WARN emitted). The mod32cap BEST npz carries full `__cfg_*`/`__bank_*` provenance
  (hosc β 2.9489 mid-anneal persisted, chroma, palette, render_hw) — decode reproduces the
  trained forward, **BIT-EXACT vs the numpy-fp32 oracle** (2-pair gate, all 3 compositions).
- **Band section (folded arm B):** `--lane-render-band` fits n600 from the GT cache
  (checkpoint-independent by construction), LBND2 RD codec default: **41,562 B counted →
  +0.02767 rate** (u_mask on). **CORRECTION to the convening record §3:** the band is NOT
  "near-zero byte" — it is a 41.5 KB counted payload. The compendium's "+0.0206" is VERIFIED
  as the **LBND4** rate term (30,892 B varint, −25.6% vs LBND2, decode-reencode bit-identical:
  `experiments/results/lane_band_res_coder_20260707/lane_band_res_coder_n600_measured.json`).
- **Pose section:** `--pose-carrier --pose-carrier-mode store_nothing` ships the #257 v2
  layout: **H_bytes = 0** (derived FREE at decode via exp_se3 + plane homography, rule-118),
  kf_of_pair junk dropped, ξ delta-AR coded. FINDING-1's 52,135 B → 6,929 B (7.5×).
- **Islands (SeedIslandBirth / AmplifyIsland / LADDER):** training-time levers — they live in
  the WEIGHTS; no new archive section or decode change needed. No byte-close gap.
- **#239 and #257 both verified FIXED by today's runs:** #239 (byte-close could not reproduce
  the pose decode — PCAR1 block landed `18927a1ae`, memo
  `n205_pose_aware_byte_close_confirmed_20260702T221839Z.md`) — today's bit-exact gate passed
  WITH the carrier composed. #257 (derive-H + ξ coder) — today's report shows `H_bytes: 0`,
  `xi_coder: delta_ar`, decode derives H.

**The 5 gaps (each with disposition):**
- **G1 — self-orient freq override hazard (CONFOUND-CLASS, silent).** `detect_self_orient`
  (tool L306) sources freq_across/freq_along/tau/iters **from CLI defaults only**
  (`--so-freq-along` default **4**), ignoring the checkpoint's persisted `__cfg_freq_along =
  8.0` (mod32cap trained at 8). A default-flag byte-close of mod32cap decodes with the WRONG
  directional features — and the bit-exact gate does NOT catch it (inflate and oracle share
  the same wrong manifest); only realized-parity vs the training-side d_seg catches it. Today's
  runs passed `--so-freq-along 8` explicitly. **BUILD (small): make `_load_levelset_ckpt`
  consume `__cfg_freq_*` with CLI as override-only + a mismatch refusal** (an L2 confound gate
  per the Confound self-protection non-negotiable). Until then: every mod32cap-family
  byte-close MUST pass `--so-freq-along 8`.
- **G2 — LBND4 decode not inlined.** The shipped `_INFLATE_PY` lacks the LBND4 decode half; a
  shipped LBND4 packet fails CLOSED at parity (tool's own help text — NO-FAKE correct).
  Shippable band today = LBND2 at +0.02767. **BUILD (small): inline the LBND4 decode** →
  band drops to +0.02057 (−0.0071 S, free).
- **G3 — render_aa unsupported at decode.** Zero `render_aa` handling in the byte-close tool;
  `__cfg_render_aa` is persisted but unconsumed. A checkpoint trained with AA supersample
  would byte-close with the WRONG (non-AA) forward. **AACoverageRender is a shipping blocker
  until the AA decode lands** (consistent with the known compose-after-downsample gap, Q3).
- **G4 — the `--run-exact-eval` leg has NEVER fired end-to-end.** No
  `reports/levelset_byte_close_*.json` exists with `exact_eval.ran = true`. The
  `run_upstream_evaluate` code path (subprocess evaluate.py, regex parse, recomputed-S
  cross-check, 18,000 s timeout) is written but unexecuted. **MEASURE: one advisory dry-run
  (R-6)** before the real row depends on it.
- **G5 — no n600 realized-parity row on the clean baseline.** The training-side 0.0033662 is
  NOT the byte-closed number; d_seg through the real decode is UNMEASURED on mod32cap.
  **MEASURE: R-1 (the first RECESS item).**

### P3 — Exact-eval chain status: decode budget PROVEN with ~2× headroom; two gates remain

- **30-min inflate budget: CLOSED in principle (#214, FEED-05z, real-weights).** Measured on a
  91,943-param self-orient hosc ckpt: numpy-fp64 4-worker **13.9 min bit-exact + 2-run
  deterministic**; torch-fp32 CPU **6.59 min score-preserving** (argmax 99.9995%, d_pose
  Δ3.2e-10); T4 <0.5 min PROJECTED (staged, unrun). `inflate.sh` clean-env rc=0, no scorer
  weights. fp32 REJECTED as bit-exact authority (±12 LSB).
- **New-stack headroom (INFERRED, to be measured in R-1):** mod32cap is 111,095 params
  (+21%); the band raster and the store-nothing warp are vectorized numpy adds (the warp
  REPLACES an INR render at train time; at decode it adds one bilinear warp/pair). Projected
  bit-exact wall ≈ 15–18 min < 30 with ~1.7× headroom. R-1 measures it for free.
- **Gates remaining before a real row can land:** (i) R-1 n600 parity row (G5); (ii) the
  exact-eval dry-run (G4); (iii) the **contest-CPU authority host** — macOS is
  `[macOS-CPU advisory]` forever; the pointer row runs `upstream/evaluate.py --device cpu` on
  Linux x86_64 (Modal CPU ~$0.06/hr per CLAUDE.md, 600-pair 1–2 h) on the EXACT archive bytes;
  (iv) operator GO for the converged-stack byte-close + eval (heavy).
- **A useful apparatus fact for the crucible:** the byte-close tool auto-drains the activation
  ledger (`record_measured_for_run`) and records costates into the cross-run posterior ONLY
  when parity actually runs (`byte_close_verdict_landed`). Measurement plans should prefer
  parity-ON byte-closes so fired levers get marked measured (closes the ledger-not-wired gap
  the grounding packet flags).

### P4 — Measurement-plan skeleton (deliverable-6 input; POWERPLAY cheapest-decisive first)

| # | probe | cost | predicted band (grounding) | kill/proceed |
|---|---|---|---|---|
| M0 | bytes+rate rows, 3 compositions (DONE today) | $0, 3×~2 min | rate 0.0556/0.0834/0.0602 | — (landed) |
| M1 = R-1 | n600 inflate+parity, ep650 BEST, band/pose OFF | ~30–60 min | realized d_seg = 0.0034 ± 3e-4 (int8+R delta prior: 100·Δ=−8.5e-5 at n24, tool header L945); inflate ≤ 20 min | KILL if Δd_seg > +5e-4 (decode/quant defect — fix before ANY run) or inflate > 25 min |
| M2 = R-2 | M1 + `--lane-render-band --lane-band-umask` | ~30–60 min | band removes most lane mass (lane = 19.1% of flips ≈ 6.4e-4 of d_seg at baseline → up to −0.064 seg term) vs +0.02767 rate | PROCEED iff net ΔS < 0; else band DEFERS pending LBND4 (G2) + trained-with-band arm |
| M3 = R-3 | byte-close+parity the **20260703T120444Z** ep300 store-nothing pose-TRAINED ckpt (`--pose-carrier-mode store_nothing`) | ~30–60 min | FIRST realized-d_pose-through-real-decode row; band: within ~2× of its training-side d_pose (chain-validation, not a floor claim) + CONFIRM triad bit-exact | KILL if `frame0_decode_bit_exact` false (#239-class regression) |
| M4 = R-4 | ξ q_levels sweep at byte-close (4096/1024/256) on M3's ckpt | $0, minutes | coded ξ 6.4 KB → ~2–4 KB at q1024 with d_pose penalty < 1e-5 (fp16≈q4096 is over-precise vs the 1.51e-4 budget) | keep the largest q with Δd_pose-term < 0.002 |
| M5 = R-5 | **the §5B decisive probe**: bounded pose-ON n600 smoke (w_pose>0 + FiLM + store-nothing + null-texture; #248 P-B read-back on its ckpt) — training launch → **operator GO** | bounded run | d_pose < 0.01 by mid-run AND d_seg within +5% of the seg-only twin at matched epoch | KILL pose-ON-as-designed if converged d_pose > **1.5e-4** (T_1 infeasible) → activate symposium L1 Jacobian-coefficient $0 gate |
| M6 = R-6 | advisory end-to-end `--run-exact-eval` dry-run on ep650 (fires the never-fired leg) | ~2–3 h | evaluate.py parses; recomputed-S delta < 1e-5 (tool cross-check); wall < timeout | KILL any pointer-row plan until this leg has fired once |
| M7 | THE ROW: converged pose-ON stack → byte-close (G1 fix in, LBND4 if built) → Linux x86_64 contest-CPU `upstream/evaluate.py` | operator GO | S < 0.19110 (aim 0.15) | the only success definition |

Ordering rationale: M1 de-risks every later row (decode integrity + wall-clock); M2 prices the
band with a real numerator; M3+M4 close the pose APPARATUS chain at $0 before any training
spend; M5 is the one bounded-training decision point; M6 is a pure never-fired-leg burn-down;
M7 is the END.

---

## Derivations + assumption tags (#363)

- **Byte-close rows (3 compositions), bit-exact PASS, section bytes/rates** —
  VERIFIED-VIA-ANCHOR(`reports/t5_s6_byteclose_mod32cap_ep650_{weightsonly,band,posesn}_20260707.json`,
  run inline today). Archive bytes exact; score semantics advisory.
- **Scorer reads frames / sidecar does not lower realized d_pose** —
  VERIFIED-VIA-SOURCE(`tools/levelset_byte_close_and_eval.py:2301-2309` pose_note + :2410-2417 POSE-BLIND print).
- **store_nothing derive-H + ξ coder live (#257)** — VERIFIED-VIA-ANCHOR(today's posesn report:
  `H_bytes: 0`, `xi_coder: delta_ar`, byte_optimal_note cites FINDING-1) +
  VERIFIED-VIA-SOURCE(tool :2606-2618 flags; `serialize_pose_carrier_store_nothing` :549).
- **#239 fixed** — VERIFIED-VIA-ANCHOR(`n205_pose_aware_byte_close_confirmed_20260702T221839Z.md`,
  commit 18927a1ae) + re-verified today (bit-exact with carrier composed).
- **R1 (#245) d_pose 62.44→0.0011 plateau, d_seg held ~0.0046** —
  VERIFIED-VIA-ANCHOR(DAG `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md:7496,7506`).
  Its "w_pose=0, null never used for pose" — VERIFIED-VIA-ANCHOR(pose symposium §8).
- **Pose-blind operating point 125.833 → term ≈35.5** — VERIFIED-VIA-ANCHOR(dossier §5B citing
  `costate_shadow.jsonl`); the dossier's "≈31" used O(100); both catastrophic.
- **Pose stack = speed-saver (−0.71 s/step when removed)** —
  VERIFIED-VIA-ANCHOR(`per_lever_compute_audit_20260705.md` §P1, n24 ratios; n600 absolutes are the wall-clock authority).
- **Feasibility thresholds (1.51e-4 / 3.2e-5)** — DERIVED(score law + today's measured rates +
  dossier's d_seg working points; arithmetic shown inline).
- **G1 freq hazard** — VERIFIED-VIA-SOURCE(tool :306-332 `detect_self_orient` CLI-only) +
  VERIFIED-VIA-ANCHOR(BEST npz `__cfg_freq_along = 8.0` printed today; CLI default 4 at :2648).
- **LBND4 −25.6% / rate 0.02057 / decode-not-inlined** —
  VERIFIED-VIA-ANCHOR(`lane_band_res_coder_n600_measured.json`) + VERIFIED-VIA-SOURCE(tool :2637-2645 help).
- **render_aa unsupported at decode** — VERIFIED-VIA-SOURCE(grep: zero matches in the tool;
  `__cfg_render_aa` persisted in npz, unconsumed by `_load_levelset_ckpt` :221-290).
- **Exact-eval leg never fired** — VERIFIED-VIA-ANCHOR(no `exact_eval.ran=true` in any
  `reports/levelset_byte_close_*.json`; only `first_row` (6-pair, parity-era) + `inflate_legal_*` exist).
- **#214 inflate budget numbers (13.9 / 6.59 min)** — VERIFIED-VIA-ANCHOR(DAG FEED-05z :8268;
  `reports/inflate_legal_byteclose_20260705.json`). New-stack ≈15–18 min — INFERRED(+21% params,
  vectorized adds; measured in M1).
- **20260703T120444Z = store-nothing pose-trained ckpt with surviving EMA npz** —
  VERIFIED-VIA-ANCHOR(its `launch.sh`: `--w-pose 1.0 --pose-carrier-source generated`;
  `levelset_witness_ema_BEST.npz` present, ep300 d_seg 0.004752).
- **L3 mechanism beats R1's floor** — ASSUMED-with-mechanism(unavoidable: the null-texture
  lever is designed-but-unfired; this is exactly what M5 measures; verdict PROVISIONAL until then).
- **"0.05499 current rate" (FEED-07a) vs today's 0.05555** — both real; different
  checkpoints/serializations. INFERRED(provenance divergence); today's ep650-BEST row is the
  clean-baseline authority going forward — do not mix the two in budgets.

## PR95 cargo-cult audit (my face)

| element | verdict | basis |
|---|---|---|
| stored-target pose sidecar (Quantizr/PR95 `poses.pt` pattern) | **DROP/REPLACED** | scorer-reads-frames (source-verified) makes a stored sidecar dead bytes; replacement = store-nothing ξ TRAINED-IN via FiLM — derived from the witness's own screw/quotient math (H = f(ξ; K, pitch, d) exact, FINDING-1 §3; group-theory Q1 orbit-coding) |
| monolithic 0.bin + int8+brotli weight blob (PR95 L20/L32 lineage) | JUSTIFIED-KEPT | re-derived through our own blob; accounting matches canonical `quantize_levelset_blob` (verified today, no WARN); measured 83,430 B beats PR95's 178,417 B at ~47% |
| "inflate ≤100 LOC" (PR95 L4) | **DROP** | replaced by the rule-118 free-generator paradigm — the generator IS the decoder; the binding constraint is the MEASURED 30-min budget (#214: 13.9 min), not a LOC proxy |
| temporal-delta + entropy coding of per-pair payloads (PR95 L25 echo) | JUSTIFIED-KEPT | re-derived from the smooth 6-dof ego trajectory (Shannon argument, FINDING-1 §7); MEASURED: ξ delta_ar 7,232→6,367 B; LBND4 −25.6% |
| ξ q_levels 4096 default ("fp16 parity" reflex) | SUSPECT — unswept | fp16-precision inheritance, not derived from the 1.51e-4 d_pose budget; M4 sweeps it |
| pose-term weight as fixed linear w_pose (PR95-style static loss weight) | SUSPECT — defer to costate | the score's local pose gradient 5/√(10·d_pose) diverges near 0 (#302 row 24); a derived schedule belongs to the costate DECIDE layer, not a hand constant |

## RECESS measurement proposals

(Full table in P4; exact command sketches here. All foreground, chunked, governor-respecting;
each writes a durable report under `reports/` and, where parity runs, auto-drains the
activation ledger.)

**R-1 (M1) — clean-baseline realized-parity row (~30–60 min, <8 GiB):**
```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z \
  --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --so-freq-along 8 --verify-bit-exact \
  --out reports/t5_s6_parity_mod32cap_ep650_n600.json
```
Band: realized d_seg 0.0034 ± 3e-4; inflate ≤ 20 min. Kill: Δ > +5e-4 vs 0.0033662.

**R-2 (M2) — band ROI numerator:** R-1 + `--lane-render-band --lane-band-umask`.
Kill: net ΔS ≥ 0 → band DEFER-at-LBND2 (re-price after G2 LBND4 decode build).

**R-3 (M3) — pose-chain validation on the surviving store-nothing-trained ckpt:**
```bash
.venv/bin/python tools/levelset_byte_close_and_eval.py \
  --ckpt-dir experiments/results/levelset_n600_witness_20260703T120444Z \
  --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pose-carrier --pose-carrier-mode store_nothing --verify-bit-exact \
  --out reports/t5_s6_parity_storenothing_ep300_n600.json
```
First realized-d_pose row through the REAL decode. Kill: CONFIRM triad not bit-exact.

**R-4 (M4) — ξ q_levels rate sweep:** R-3 with `--pc-xi-qlevels {1024,256}`; minutes each.

**R-5 (M5) — the §5B decisive bounded pose-ON smoke — OPERATOR GO (training launch, outside
my envelope):** seg-only twin + pose-ON arm (w_pose staged-on at tau boundary, FiLM, #227,
null-texture), byte-closed per-stage. Proceed/kill per P4 (1.5e-4 converged threshold).

**R-6 (M6) — exact-eval leg dry-run (advisory):** R-1 command + `--run-exact-eval` (macOS
advisory; or Modal Linux CPU for the contest-CPU axis). ~2–3 h; fires the never-fired leg.

**Builds (small, named):** B-1 = G1 fix (consume `__cfg_freq_*`, mismatch refusal — confound
gate); B-2 = G2 LBND4 inflate decode (−0.0071 S free on the band); B-3 = G3 AA decode
(only if S1 adopts AACoverageRender in the shippable stack).

## Interfaces

- **From S1 (basis):** any Arm-A freq rebalance (freq-along/across change) MUST be threaded to
  the byte-close `--so-*` flags until B-1 lands — the decode silently uses CLI defaults (G1).
  AA render (AACoverageRender) is a SHIPPING blocker until B-3 (G3) — decide AA's stack
  membership with that cost priced in.
- **From S2 (schedule/curriculum):** the w_pose stage-on point + warmup shape (I recommend the
  tau boundary + existing rewarmup machinery); per-stage EMA checkpoints give me a byte-close
  row per stage boundary (early exact-relevant rows from ONE run).
- **From S3 (costate):** a derived w_pose schedule from the λ-marginal (the 5/√(10·d_pose)
  divergence is a costate-native signal); note the byte-close auto-records costates + drains
  the activation ledger only on parity-ON runs — plan verdicts through parity.
- **To S4 (rate):** measured section economics: weights 0.0556 @ 111 K params; band +0.02767
  (LBND2) / +0.02057 (LBND4, build owed); pose +0.00464 (q4096, sweepable to ~half). λ_bytes
  6.659e-7 S/byte prices every section.
- **To S5 (lever ledger):** MEASURED correction — `AnalyticLaneRenderBand` is NOT
  "near-zero-byte" (convening record §3): 41,562 B / +0.02767 today, 30,892 B / +0.02057 after
  B-2. Its d_seg payment must clear that bar (M2). Store-nothing pose lever: fired-and-measured
  on the rate axis today; d_pose axis owed via R-3/R-5.
- **To synthesis:** the two-wall picture — d_seg (3.7×) AND pose (7× vs R1's floor) — both must
  fall in the SAME run; the measurement plan above sequences the apparatus so the one heavy run
  is de-risked end-to-end (decode integrity → band ROI → pose chain → exact-eval leg → GO).
