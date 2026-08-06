---
schema: ddm_et4_next_if_resumed.v1
date_utc: 2026-08-06
arm: ddm_et4
score_claim: false
promotion_eligible: false
tokens: [no-triality, p0-ledger-ok]
---

# DDM ET4 Next If Resumed

## Fire Order

1. Continue the n600 rows from the SSD checkpoint, with chunk size `<=120`.
2. Do not bank or adjudicate the first-8 prefix. The common contract says n=8 banks nothing.
3. Once `600/600` rows and patch records exist, build the counted ET4 overlay archive through
   `tac.submission_chain`.
4. Inflate and evaluate the byte-closed archive, then report realized `d_seg`, `d_pose`, bytes, `S`, and
   `dS` versus `S=0.7534578126155775 @ 357,837 B`.
5. If `dS < 0`, queue MAIN adjudication for bank/pointer handling. If `dS >= 0`, write the honest
   formulation-scoped negative and keep the family intact.

## Resume Commands

Current completed rows: `8/600`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 120 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 120 --pair-stop 240 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 240 --pair-stop 360 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 360 --pair-stop 480 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 480 --pair-stop 600 --resume
```

Then:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 600 --resume --build-archive --run-inflate --run-evaluate
```

## Expected Timing

Measured first-8 mean: `71.04250931739807 s/pair`. Serial remaining estimate for 592 pairs:
`42057.16551589966 s` (`11.68 h`). This is a scheduling estimate only.

## Guardrails

- Keep the parent archive fixed at
  `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`.
- The runner refuses a parent sha different from
  `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.
- Do not substitute `tq1c_base`.
- Do not run more than one full-n600 scorer job at a time.
- Do not claim a score until the byte-closed archive is evaluated.
- Keep bulk artifacts under `/Volumes/VertigoDataTier/pact/ddm_et4_20260806`.
