# ddm_ps2 — RECEIPT

UTC 2026-08-18T01:06:40Z · git `26e7a2eb075248dd6e4aff716ea5b6e66764e203` · host Primary

Axis `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false` · pointer UNMOVED.
No scorer was run. No Modal. No Metal. $0. All figures are arithmetic over fo2h retained rows.
Determinism: both JSONs reproduce byte-identically across a repeat run modulo the `utc` stamp.

## artifacts

| artifact | bytes | sha256 |
|---|---:|---|
| `PS2_F2_JOINT_ADJUDICATION.json` | 9659 | `b68393ce88de3da938c5ba3c72b113c56ee9e84e773a660692f68a2985f17804` |
| `PS2_JOINT_GATE_SURVIVOR.json` | 39284 | `ce52df8a02909ff7ae3be492528a587d52f52a2ef86bf1b5a8bc652a32a19070` |
| `PROGRESS.jsonl` | 3840 | `002964804dcfd8fe267c7ec78d59c16fd815e97a20432f40733d0b5dd4d25457` |

## upstream inputs (fo2h, read-only, unmodified)

| artifact | sha256 |
|---|---|
| `ddm_fo2h_eta_hardening/null_shardA/ETA_GATE_ROWS.jsonl` | `b1485992ddcfff4a79a0f8422ef978e27a0db6eec00e4840b6c74c246bb34367` |
| `ddm_fo2h_eta_hardening/null_shardB/ETA_GATE_ROWS.jsonl` | `1c0fc5ed2bf47697c80ea26d95481708b5bcb74823a17cebd7d49d3692745ef1` |
| `ddm_fo2h_eta_hardening/free_matched16/ETA_GATE_ROWS.jsonl` | `c5a96514e60a7ee7789eba3a43c15af3cd646f7414f3ef66346094d0bfb639f2` |
| `ddm_fo2h_eta_hardening/FO2H_WATERFILL_MEASURED.json` | `b9dd1a1adcac9275e643c0bed625afe3814769c51a99cf34564924f3b783b1ff` |
| `ddm_fo2h_eta_hardening/FO2H_ETA_ADJUDICATION.json` | `7e154b2c7a4cda3c2d1e472954bfc4ff0597a86f8025585c47bad1be3e2da5cc` |

## producing code (committed)

- `experiments/ddm_ps2_f2_joint_adjudicate.py`
- `experiments/ddm_ps2_joint_gate_survivor.py`
- `src/tac/tests/test_ddm_ps2_joint_pose_arithmetic.py` (9 tests, all pass)
- memo `.omx/research/ddm_ps2_pose_projection_nscaling_20260818.md`
