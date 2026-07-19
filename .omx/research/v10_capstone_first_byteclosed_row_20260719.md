# v10 CAPSTONE — first byte-closed S row through the OFFICIAL evaluator (2026-07-19)

**Axis: [macOS-CPU advisory] NON-PROMOTABLE · score_claim=false · pointer 0.1910828242 [contest-CPU] UNMOVED.**
Operator goal 2026-07-19: "Our goal today is to reach successful byte closed v ten capstone." — REACHED
in the loop-closure sense: the FULL official chain executed end-to-end on counted bytes.

## The row (MEASURED, unmodified upstream/evaluate.py, 600 samples, --device cpu, seed 1234)

| term | value | S-contribution |
|---|---|---|
| SegNet distortion | 0.00015196 | 0.015196 |
| PoseNet distortion | 0.00010184 | 0.031912 |
| archive bytes | 409,526,925 | rate 10.90748678 → 272.687 |
| **S** | | **272.73** |

Report: `<SSD>/evidence/c1_two_plane_receiver_20260719/capstone_eval/report.txt` (launch manifest sibling).

## The chain (every joint on the official contract)

1. Payload → `archive.zip` 409,526,925 B, sha256 e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42,
   canonical-ZIP-identical; packet 0.bin sha aa1dbb5e2efff28cd0d31f5ee2a4b0575a248a27a431151bfcae64eb320d385b;
   codec predictor-residual-u8.v1 (spatial-smooth-121 predictor; EXACT planes; Y0/Y1 shas match the C1
   frozen custody 5e86e419… / 6a731946…).
2. `inflate.sh <archive_dir> <output_dir> <video_names_file>` (literal contest signature) → scorer-free
   receiver → EXACT factor-2 integer solve → 874×1164 frames (out_serial/0.raw, TensorVideoDataset ABI).
   All 707,788,800 numerators verified; both planes exact.
3. Official evaluate.py scored the decoded output against upstream/videos/0.mkv (37,545,489 B — the
   exact score-law normalizer).

## What the row establishes (each MEASURED)

- **Distortion is frontier-class at the exact-plane endpoint:** 0.047 total < the ~0.073 distortion
  budget implied by the 0.19108 frontier. The vehicle's realization + receiver lose almost nothing.
- **d_seg 1.52e-4 = the preimage-fp32 noise class** predicted by
  f32_receiver_arithmetic_exactness_admissibility_v1 (~1.2e-4 measured class): same rational plane,
  different uint8 camera preimage → fp32 resize noise only. First confirmation of that law THROUGH the
  OFFICIAL evaluator at n600 (anchor appended to the law this landing).
- **The open axis is rate alone (99.98% of S).** Consistent with the 07-19 state: exact-residual family
  rate-dead (~336KB/pair floor); descent = cheaper DESCRIPTIONS (banded/secant → counted C2 generator),
  each now a drop-in payload swap into this officially-scored spine.
- **C1 timing (local, 4-worker):** full decode+verify ~302 s; the integer solve itself ~3.5 s/600 pairs;
  ~236 s is raw-frame IO. Contest-budget verdict remains honestly
  BLOCKED_CANONICAL_FULL_EVALUATE_RECEIPT_VALIDATOR_OWED — the Modal contest-CPU run (#381 ≤$20,
  operator-GO) is BOTH the timing authority and the exact-axis receipt for this same archive.

## Verdict scope

Loop-closure + realization/receiver fidelity + rate-axis attribution ONLY. NOT a frontier candidate,
NOT a promotion, NOT a pointer claim (S=272.73 ≫ 0.19108, rate-dominated by construction — the payload
is the known rate-dead exact-plane endpoint, chosen to prove the spine, not to descend).

## Consumers

- SPEC_v10 §8: C1 receiver/ABI leg CLOSED locally (Modal receipt owed); the S-composition harness for
  C6/C9 now exists end-to-end.
- f32_receiver_arithmetic law: official-evaluator n600 anchor appended (this landing).
- #571: the counted-base C2 byte-close drops into `capstone_submission/` unchanged.
- Operator GO decision: Modal contest-CPU run of THIS archive = C1 CLOSE + first [contest-CPU] v10 row.
