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

## Rank 2 — S2_film23_q2_top3_q3: LIVE

Different mechanism (precision reduction on film23 rows, not deletion; SD1M
family), 179,828 B (−1,333 B, credit −8.874e-4 S), archive `a36890b6541cf259…`.
Staged at `generations/S2_film23_q2_top3_q3` (inflate.py re-pinned per the
staging lesson), fired attempt_0001 on the same instrument. Admit needs
pose+seg damage < ~8.84e-4 S — i.e., quantization must be ≥180× gentler than
deletion on the pose axis.

## Apparatus lessons banked this chain

1. **Mirror self-pollution class**: the evaluate step writes `__pycache__` into
   the canonical mirror; the NEXT run's provenance hash refuses executable
   bytecode. Cure: mirror restored (hash re-verified `fa7c4bf5…` = the sealed
   value) + `PYTHONDONTWRITEBYTECODE=1` now in the advisory harness env
   (numerically inert). Memory: sister of the `._*` AppleDouble class.
2. The base leg is a bought instrument — every subsequent sa1 candidate
   adjudicates against it at $0 marginal instrument cost.
