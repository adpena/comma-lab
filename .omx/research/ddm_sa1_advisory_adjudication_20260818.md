# ddm_sa1 advisory adjudication — same-instrument base leg + rank-1 verdict (2026-08-18)

Axis: **[macOS-CPU advisory n600, env-mismatch grade]** — score_claim=false, promotable=false.
Deltas below are SAME-INSTRUMENT (identical harness, mirror, device, shim'd PATH);
absolutes are never compared to T4 rows. Admit rule and bar from the sealed sa1 §7
fire-order; bar re-derived at fire time from the live pointer (sz1 composed,
S 0.15771357797660338, canonical band ±3.5e-6).

## The instrument (the base leg — durable, reusable for every sa1 candidate)

rr4 BASE archive (`35ac2b9beb7e6fa8…`, 181,161 B) through the identical local
advisory harness (attempt_0002, rc=0, 1,196 s, payload kept):

| row | S | d_seg | d_pose | bytes |
|---|---|---|---|---|
| BASE rr4 (CPU advisory) | 0.20174349667996583 | 0.00042714 | 0.00014747 | 181,161 |
| BASE rr4 (T4 exact, sealed) | 0.15853325 | 0.00029611 | 6.88e-6 | 181,161 |

CPU-vs-T4 drift on the SAME bytes: pose 21.4× worse on CPU (consistent with the
#1054 frontier-bytes measurement), seg 1.44× worse. This is why cross-instrument
adjudication is forbidden (#1034 lesson) — the base leg prices the drift out.

Receipt: `/Volumes/APDataStore/pact/ddm_sa1/advisory_n600_cpu/rr4_base/attempt_0002/contest_auth_eval.json`
(upstream snapshot `fa7c4bf51d47a614…`, matches the sm3r leg's recorded snapshot exactly).

## Rank 1 — sm3r_keep01: REFUSED, net ΔS = +0.15698

Candidate row (attempt_0006, rc=0, payload kept): S 0.35871954844376963,
d_seg 0.00043191, d_pose 0.00387399, 178,272 B (archive `67422cf0d49b7af0…`).

Same-instrument decomposition vs the base leg:

| term | ΔS |
|---|---|
| rate (−2,889 B) | −1.9237e-3 |
| seg (Δd_seg +4.77e-6) | +4.77e-4 |
| pose (√(10·d_pose): 0.038402 → 0.196825) | **+0.158423** |
| **net** | **+0.156976** (bar: < −3.5e-6) |

**Mechanism (measured, not drift):** pruning 99% of the semantic FiLM rows leaves
seg almost invariant (+4.8e-6 — the carrier-side seg proofs held) but collapses
pose **26×** on the same instrument. The deleted rows carry pose-relevant render
signal. The pose damage is 82× the rate credit. This prices the distortion the
sa1 memo flagged unpriced and confirms §9.3 with a sharp asymmetry: the semantic
tensor is nearly seg-inert and strongly pose-load-bearing.

The fire-order's own pre-registered ceiling (d_pose ≤ 1.043e-5 if seg held) was
exceeded 371×. verdict_scope: INSTANCE (keep_percent=1 on this base). DERIVED
family read (not measured): the SM3R row-drop family looks dead — even 1/100 of
this pose damage at keep87's −130 B credit is ~80× over bar — but rank 3
(sm3r_keep87, the slope anchor) fires only if rank 2's mechanism read makes the
slope worth buying.

## Rank 2 — S2_film23_q2_top3_q3: REFUSED, net ΔS = +0.06021

Different mechanism (precision reduction on film23 rows, not deletion; SD1M
family), 179,828 B (−1,333 B, credit −8.876e-4 S), archive `a36890b6541cf259…`.
Row (attempt_0001, rc=0, 1,186 s, payload kept): S 0.26195030040592626,
d_seg 0.00042886, d_pose 0.00098653.

| term | ΔS |
|---|---|
| rate (−1,333 B) | −8.876e-4 |
| seg (Δd_seg +1.72e-6) | +1.72e-4 |
| pose (d_pose 1.47e-4 → 9.87e-4, 6.7×) | **+0.060922** |
| **net** | **+0.060207** (bar: < −3.5e-6) |

Quantization is ~2.6× gentler than deletion on pose (6.7× degradation vs 26×)
but needed ≥180× — refused by ~68× over its credit. **Two mechanistically
distinct edits now concord: lossy perturbation of the semantic FiLM tensor pays
~70–80× its rate credit in pose, while seg stays nearly invariant both times.**

## Rank 3 — sm3r_keep87: REFUSED, net ΔS = +0.04427 (the slope anchor)

181,031 B (−130 B, credit −8.656e-5 S), archive `a16a58e55f63b141…`, deletes
only the 13% least-important rows (rgb_rms-ranked). Row (attempt_0001, rc=0,
1,186 s, payload kept): S 0.24601735879880116, d_seg 0.00042778,
d_pose 0.0006839.

| term | ΔS |
|---|---|
| rate (−130 B) | −8.656e-5 |
| seg (Δd_seg +6.4e-7) | +6.40e-5 |
| pose (d_pose 1.47e-4 → 6.84e-4, 4.64×) | **+0.044296** |
| **net** | **+0.044274** (bar: < −3.5e-6) |

Damage is **512×** the rate credit — the worst ratio of the three, because the
credit is tiny while the per-row pose load is not. The admit path required the
bottom-13% rows to be pose-inert (~250× sub-linear damage); measured damage is
**LINEAR** in deleted mass: keep01/keep87 deleted-mass ratio 7.62, absolute
d_pose-increase ratio 6.95 → per-row damage 0.91× of exactly linear. Every
semantic FiLM row carries ~equal pose load. rgb_rms is now proven pose-blind
twice (keep01 and keep87 anchor both ends of its own ranking).

## FAMILY VERDICT — lossy uncompensated semantic-FiLM edits: CLOSED, REFUSED 3/3

verdict_scope: **FAMILY** — lossy UNCOMPENSATED perturbation of the semantic
FiLM tensor on the rr4/cp135 base (deletion-heavy · quantization ·
deletion-light; three mechanistically distinct edits, one instrument).

The measured law: **the semantic FiLM tensor is nearly seg-INERT
(Δd_seg ≤ +4.8e-6 all three rows) and uniformly pose-LOAD-BEARING
(damage 68–512× the rate credit, linear in perturbed mass).** It is a pose
carrier wearing a semantic name — consistent with eu4's route map (pose routes
through the joint representation) and with why PR135 ships it whole.

Three-point table (all same-instrument vs the bought base leg):

| rank | mechanism | Δbytes | rate ΔS | pose ΔS | net ΔS | damage/credit |
|---|---|---|---|---|---|---|
| 1 keep01 | delete 99% of rows | −2,889 | −1.924e-3 | +0.158423 | +0.156976 | 82× |
| 2 S2 quant | q2/q3 precision cut | −1,333 | −8.876e-4 | +0.060922 | +0.060207 | 68× |
| 3 keep87 | delete bottom-13% (rgb_rms) | −130 | −8.656e-5 | +0.044296 | +0.044274 | 512× |

Reactivation criteria (either reopens the family, neither is more pruning):
1. **Pose-COMPENSATED semantic edit** — the qs5-proven in-compile Schur
   compensation solved against the EDITED tensor (asserted in code, never
   carried across regimes per the qs4 lesson). qs5 measured frame-1 seg edits
   at ~zero pose tax once compensation is in-compile; the same machinery has
   never been pointed at a semantic-tensor edit.
2. **The js1/js8 joint-representation line** — a jointly-learned semantic+pose
   representation replaces the tensor instead of editing it (the named major
   pose route; every sa1 refusal this chain reinforces it).

Cost of the whole family measurement: 4 advisory runs ≈ 80 min Metal-free CPU,
$0 marginal (the base leg amortizes), all payloads retained under
`/Volumes/APDataStore/pact/ddm_sa1/`.

## Apparatus lessons banked this chain

1. **Mirror self-pollution class**: the evaluate step writes `__pycache__` into
   the canonical mirror; the NEXT run's provenance hash refuses executable
   bytecode. Cure: mirror restored (hash re-verified `fa7c4bf5…` = the sealed
   value) + `PYTHONDONTWRITEBYTECODE=1` now in the advisory harness env
   (numerically inert). Memory: sister of the `._*` AppleDouble class.
2. The base leg is a bought instrument — every subsequent sa1 candidate
   adjudicates against it at $0 marginal instrument cost.
