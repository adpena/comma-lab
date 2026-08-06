---
schema: ddm_et4_checkpoints.v1
date_utc: 2026-08-06
arm: ddm_et4
score_claim: false
promotion_eligible: false
tokens: [no-triality, p0-ledger-ok]
---

# DDM ET4 Checkpoints

## Current State

Completed rows: `8/600`, pairs `0..7`.

Durable SSD state:

| artifact | path |
|---|---|
| rows JSONL | `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl` |
| summary JSON | `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_summary.json` |
| patch records | `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/patch_records/` |
| receipt copy of summary | `.omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json` |

Hashes:

| artifact | sha256 |
|---|---|
| rows JSONL | `2d4f9d1be067c7b682aaf360c2ed9651cb907cfffb3cf2d9d93c91864a05e8d2` |
| summary JSON | `ab6df3e5a443279f3a898b919a9598d002ea99c422001ae869496bfa2b0c5a60` |

Patch-record hashes:

| pair | sha256 |
|---:|---|
| 0 | `aeb256d696eed676bc09c24f0b02687fea00a5cbb7a388c4a6145a8c64694d35` |
| 1 | `7beb75cd071f9cb1a3f1914e3e2761c108703a5563cdc9a8dcf7c5c5022eaa14` |
| 2 | `dcb28f088a37dc634e4e3dea31a1854316a952c782102708b10d7acc7fd9d198` |
| 3 | `3b87a07c59d8b67838dcd16236ae2209e98269ad97c4d3ed239d3d798462372f` |
| 4 | `f56bb3ceab4e6b78d099e9958324104a08b7deeabf28f1b00465c3be41636a36` |
| 5 | `416d0b8f01c65cb7a000e75f761c796d3bacc062b1a01cc41dc047c8cb28ac74` |
| 6 | `bcb8d4f6ff2429660a6ae6290543b16875bcee01df6a94d250e5eb5ab19d84e9` |
| 7 | `781337d76352ec3e7b43c4163290bcab95e7096df1cf8ebdfe17e0affbcc3d69` |

## Storage Preflight

SSD bulk dir: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806`.

Measured free bytes at first-8 receipt time: `97701175296`; minimum required by runner:
`21474836480`. Preflight passed. No persisted evidence path is under `/tmp`.

## Resume Semantics

The runner writes one JSONL row and one `.npz` patch record per pair. With `--resume`, an existing row is
accepted only if its patch record exists. A missing patch for an existing row is fail-closed.

Chunk guard: requested scoring chunks must be `<=120` pairs unless `--build-archive` is used after all
600 rows exist.

## Next Chunk Commands

Run at most one full-n600 scorer job at a time.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 120 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 120 --pair-stop 240 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 240 --pair-stop 360 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 360 --pair-stop 480 --resume
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 480 --pair-stop 600 --resume
```

After 600 rows and patch records exist:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 600 --resume --build-archive --run-inflate --run-evaluate
```

## No-Bank Boundary

This checkpoint is not a score, not a candidate archive, and not a pointer movement. It is a resumable
timing and first-8 prefix measurement only.
