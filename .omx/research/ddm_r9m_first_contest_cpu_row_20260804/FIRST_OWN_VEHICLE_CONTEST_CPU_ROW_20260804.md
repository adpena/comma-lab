# THE FIRST OWN-VEHICLE CONTEST-CPU EXACT ROW — 2026-08-04

## The row (MEASURED, full custody)

**S = 0.7541524541914318 [contest-CPU]** on fz4 `sub_final` — the first exact contest-axis row ever
measured on a vehicle that is OURS end-to-end (no borrowed lineage anywhere in the archive).

| field | value |
|---|---|
| axis | contest-CPU (Modal Linux x86_64, device=cpu, CUDA_VISIBLE_DEVICES empty) |
| n_samples | 600 (full population) |
| avg SegNet distortion | 0.00431185 |
| avg PoseNet distortion | 0.00071460 |
| rate (unscaled) | 0.009537337494791985 (358,084 / 37,545,489) |
| S recomputed from components | **0.7541524541914318** (evaluate.py print "0.75" — 2-decimal, ignored per m82/#877) |
| archive sha256 | `ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66` (358,084 B — byte-identical to the advisory row's archive) |
| transport zip sha256 | `3f192ebda174aa83499f3c1e23c584612abb3be6f45c09da03fa391f97de8155` (validated remotely BEFORE extraction) |
| runtime files digest | `6cf1e8178fa3075d4156d9b13d6cfe0bcf68207d903d5ba1880959ff745da955` (the 71a790be2e environment-free custody object — validated remotely on the EXTRACTED tree) |
| upstream evaluate.py sha256 | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` (6,005 B) |
| upstream snapshot sha256 | `92f2b06a2a0940868aa715d8d365d05f0fc9ae362110c8f455e05baf93a88e54` |
| inflate elapsed | 631.56 s · evaluate elapsed 174.27 s · total remote 821.0 s |
| call_id | fc-01KZ74QCPKHDH48NTR0R9RPDJ3 (attempt 7; job ddm_r9m_sub_final_contest_cpu_20260804c) |
| dispatch claim | lane_ddm_r9m_first_own_vehicle_contest_cpu_20260804 — terminal `completed_contest_cpu_modal_auth_eval` 19:56:43Z |
| artifacts | `.omx/research/ddm_r9m_first_contest_cpu_row_20260804/modal_exact_eval_results/contest_cpu_b/` |
| spend | ~821 s Modal CPU (+ ~42 s across 6 failed attempts, all fail-closed pre-eval) — well inside the r9m ≤$2 sub-envelope of #381 |

Authority per CLAUDE.md: `evidence_grade=contest-CPU`, authoritative for the public-leaderboard CPU
axis. NOT a CUDA row; NOT a submission claim (shipping requires BOTH axes); NOT a pointer move
(borrowed 0.19108 [contest-CPU custody] unmoved — this row is 0.754 on our own vehicle).

## THE FINDING — the advisory axis is calibrated to contest-CPU at ~1e-5

| axis | d_seg | d_pose | rate | S |
|---|---|---|---|---|
| macOS-CPU advisory (fz4 receipt) | 0.00431179 | 0.00071459 | 0.00953734 | 0.7541459 |
| **contest-CPU (this row)** | 0.00431185 | 0.00071460 | 0.00953734 | **0.7541525** |
| delta | +6e-8 | +1e-8 | 0 | **+6.55e-6** |

MEASURED: the entire advisory ladder this campaign has been climbing (v4d → pw1 → ms8 → dc1 → pj2 →
cx1 → pu2 → fz4, seven pointer moves) transfers to the contest-CPU axis within **6.6e-6 S** — the
same magnitude as the historical PR107 macOS↔GHA agreement (6e-6). Consequences:
1. Advisory ΔS comparisons on this vehicle/receiver are trustworthy at the 1e-5 scale on the CPU
   axis; the ordering of our banked candidates is contest-CPU-real, not advisory-only hope.
2. Every future advisory row inherits a measured transfer prior of +0.0000066 ± (one-point estimate;
   scope: INSTANCE — this archive/receiver family; widen with the next candidate's paired row).
3. The CUDA axis remains UNMEASURED for the own vehicle (separate evidence space per CLAUDE.md;
   never inferred from CPU).

## What made it land (the apparatus story)

Seven attempts; the first six were the five recurring failure classes (memory
`modal-dispatch-five-failure-classes-permanent-fixes`). The permanent fixes that carried attempt 7:
- **Files-digest custody** (71a790be2e): environment-free `runtime_files_sha256` computed identically
  by both validators — the deadlock (local projection 3ea13f96 vs remote tree 9982203b) dissolved,
  and BOTH legacy tree hashes now appear in provenance as observational fields only.
- **Exact-basename allowlist** (6e127af6ac): `ddm_r7_token_coder.py` (the receiver's token coder)
  passes the secrets marker without weakening the "token" scan for everything else.
- Detached fire (nohup+disown) + pre-claim `require_active` + detached blocking harvest with
  terminal claim closure — the class-1/3 operational laws executed as designed.

## NEXT (owed consumers)
- The paired contest-CUDA row for any submission-grade claim (queued behind budget/priority — not
  needed while S is 4.4× above the 0.172 bar).
- se1/sb1/ed1 candidates: each future n600 advisory row can now cite the measured CPU-axis transfer
  prior; the NEXT Modal row should be spent on a candidate that BEATS 0.754, not on re-calibration.
- Costate/organ: this receipt is the calibration anchor joining the advisory ladder to the contest
  axis (evidence join for ev1-lineage stores).
