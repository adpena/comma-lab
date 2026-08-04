# ddm_rt1 -- RATE-axis charter receipt

**Date:** 2026-08-04  
**Charter:** `.omx/tmp/codex_runs/rt1_prompt.md` + `.omx/tmp/codex_runs/_common_contract.md`  
**Axis:** byte-only / scorer-free rate apparatus.  
**Score claim:** false. **Promotion eligible:** false. **Rank/kill eligible:** false.

## Outcome

The rt1 charter is closed as a byte-only rate receipt. No scorer ran here, no archive was promoted, and
the own-vehicle frontier did not move in this arm.

The underlying implementation and primary memo were already landed as `ddm_tz1` in commit
`eb3d47ce90` (`ddm_tz1: archive token-sweep RATE attack harness + byte-only leg landed`). This rt1
receipt records the charter-specific execution surface, today's current-gap recomputation, and the
scorer fire-order so the follow-ons are not orphaned.

## Evidence Consulted

Primary committed artifacts:
- `.omx/research/ddm_tz1_token_sweep_rate_attack_20260804.md`
- `experiments/ddm_tz1_token_sweep_rate_attack.py`
- commit `eb3d47ce90`

SSD receipts:
- `/Volumes/VertigoDataTier/pact/ddm_rt1_20260804/rt1_byte_only_reverify.json`
  - bytes: 14303
  - sha256: `713d491185ada2865af005e717e0a8dd9f62bf635f485d2559a675e30a7978d7`
- `/Volumes/VertigoDataTier/pact/ddm_rt1_20260804/rt1_lean_byte_receipt.json`
  - bytes: 4446
  - sha256: `ed4664daeef739d2fabd0507ba10e62a0f2f9e9f4b776beb3472555ff5e8ca23`
- `/Volumes/VertigoDataTier/pact/ddm_tz1_20260804/tz1_byte_only_receipt.json`
  - bytes: 6695
  - sha256: `3ce4586c28941f6aa4de54ccdbda08e8c91646b469dddda136a55f6938a98685`

Source inputs from `rt1_lean_byte_receipt.json`:
- tokens: `/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy`
  - shape: `[600, 24, 32, 4]`
  - dtype: `uint8`
  - sha256: `d4eacbf619d09aeda1c15a5015b0cd45ab2d3de33d349c881b7e0f59dc803a56`
- field: `/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy`
  - shape: `[24, 32]`
  - dtype: `int64`
  - sha256: `31586a87e2f82662602fda554f5a6bf286dceeea5ec9b4a062929f67f58d184d`

Calibration used for this memo:
- current own-vehicle frontier: `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`
- PR130 bar: `0.172141`
- current gap: `0.5820049`
- rate coefficient: `25 / 37_545_489 = 6.658589531221714e-7 S/B`

The older tz1 receipt's percent-of-gap column used the pre-fz2 gap `0.6189279`. Raw bytes and `dS_rate`
are unchanged; the percent-of-gap values below are recomputed against today's common-contract gap
`0.5820049`.

## Measured Byte Legs

| Leg | Measured bytes | Rate-only dS | Current gap pct | Break-even scorer price |
|---|---:|---:|---:|---:|
| Live ix2 global `L=14` | `-24,605 B` | `-0.01638346` | `2.82%` | pays iff `delta d_seg < 1.638e-4` |
| Live ix2 global `L=8` | `-106,099 B` | `-0.07064697` | `12.14%` | pays iff `delta d_seg < 7.065e-4` |
| Adaptive margin-coupled `[16,12,8,4]` | `-113,555 B net` | `-0.07561161` | `12.99%` | pays iff `delta d_seg < 7.561e-4` |
| Adaptive derived-activity `[16,12,8,4]` | `-62,502 B net` | `-0.04161752` | `7.15%` | pays iff `delta d_seg < 4.162e-4` |
| Adaptive margin-coupled `[16,8]` | `-64,231 B net` | `-0.04276879` | `7.35%` | pays iff `delta d_seg < 4.277e-4` |
| Adaptive derived-activity `[16,8]` | `-35,840 B net` | `-0.02386438` | `4.10%` | pays iff `delta d_seg < 2.386e-4` |

`L=14` reproduces the stale smevr/tr1 provenance number exactly (`-23,655 B`) but corrects the live ix2
vehicle to `-24,605 B`. The live receiver is ix2, so the `-24,605 B` number is the one to use.

## Task #933

The shipped token range clamp mass is measured:

| Quantity | Value |
|---|---:|
| `lvl0` mass | `30.1651%` |
| `lvl15` mass | `3.1309%` |
| total endpoint mass | `33.2960%` |
| ch0 endpoint mass | `34.4301%` |
| ch1 endpoint mass | `43.6133%` |
| ch2 endpoint mass | `38.0888%` |
| ch3 endpoint mass | `17.0516%` |

The range-refit half is QUEUED, not claimed: the continuous pre-clamp tokens are absent from the shipped
artifact, so widening/narrowing the `+/-1.0` literal cannot be honestly reconstructed from `cx1_tokens.npy`.
It needs a retrain or an unclamped-token artifact before a scorer run.

## Task #869

The 768-cell adaptive waterfill byte surface is measured at real coder/receiver round-trip. The strongest
byte-only row is margin-coupled `[16,12,8,4]`:

- member bytes: `227,647`
- gross saved bytes: `113,648`
- stored map cost: `93 B`
- net saved bytes: `113,555 B`
- level histogram: `552 x L4`, `87 x L8`, `86 x L12`, `43 x L16`
- round-trip: true

The 0-byte fallback row is derived-activity `[16,12,8,4]`:

- member bytes: `278,793`
- gross/net saved bytes: `62,502 B`
- map cost: `0 B` because the activity map is decoder-derivable from tokens
- level histogram: `432 x L4`, `110 x L8`, `148 x L12`, `78 x L16`
- round-trip: true

Byte-only domination is measured: the margin-coupled adaptive map saves more bytes than global `L=8`
while retaining 43 cells at full `L16` and 86 at `L12`. The joint rate x d_seg claim is scorer-gated and
not made here.

## Coder And Depth Boundaries

The token-bulk coder axis stays shut for this arm. On the live residual payload:

- Brotli q11: `339,970 B`
- best LZMA1 variant (`lc=2, lp=0, pb=0`): `347,609 B`
- shipped LZMA1 (`lc=3, lp=0, pb=0`): `348,438 B`

The base micro-block has a tiny tuned-LZMA edge (`1,248 B` vs Brotli `1,297 B`), but the token bulk is
the residual stream and the charter-moving coder result is negative. This does not reopen task #918/#940.

Sub-nibble token bit-packing is not adopted. At charter-relevant `L=8`, tight 3-bit packing worsens coded
bytes:

- fixed nibble coded bytes: `234,156 B`
- tight 3-bit coded bytes: `274,295 B`
- net: `-40,139 B`

The full receipt also shows a small positive tight-pack row at `L=4` (`+5,760 B`), but that row is coupled
to an aggressive lossy coarsening and a receiver format change; it is not a scorer-free adoption.

## Optional #921 Siblings

Bounded source lookup found the optional siblings already on the guarded-constant path, so rt1 does not
duplicate them:

- `thr_wall = 2.5e-4` has an explicit force-ledger row
  `thr_wall.pose_gate_3275x`; it is correct arithmetic but stale-target / owed-adjudication against the
  PR130 pose bar.
- `total_archive_ceiling_bytes: Literal[200000]` is flagged unladdered and queued as ca1 O2 for whoever
  re-fires the composer. The protected compose file is not touched by this arm.

## Fire Order

1. QUEUED after the current scorer owner clears: adaptive margin-coupled `[16,12,8,4]` joint scorer run,
   using the same byte-close row; accept only if `delta d_seg < 7.561e-4`.
2. QUEUED fallback: adaptive derived-activity `[16,12,8,4]`; accept only if `delta d_seg < 4.162e-4`.
3. QUEUED attribution sweep if adaptive fails or needs a clean baseline: global `L in {15,14,12,10,8}`;
   `L=14` pays iff `delta d_seg < 1.638e-4`.
4. QUEUED-BLOCKED: `+/-1.0` range refit, blocked on continuous pre-clamp tokens or a retrain artifact.
5. FOLDED: optional `thr_wall` / `total_archive_ceiling_bytes` siblings to the existing fl2/gk1 guarded
   constant ledger and composer re-fire queue.

## Boundaries

- I ran no scorer and did not consume the n600 scorer slot.
- I did not edit `upstream/`, `tac.submission_chain`, the receiver, or
  `src/tac/optimization/direct_description_carrier_compose.py`.
- I attempted a fresh full harness replay to
  `/Volumes/VertigoDataTier/pact/ddm_rt1_20260804/rt1_rederive_receipt.json`; it was interrupted after
  more than five minutes in `F_depth_x_coder` Brotli compression, emitted no target receipt, and ran no
  scorer. The existing full rt1 receipt listed above is the byte evidence for this memo.
- All byte numbers are real-coder, round-trip-backed measurements on the cited token artifact. No
  d_seg/d_pose result is measured here.

Own-vehicle frontier: `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600` -- UNMOVED by rt1.
