# Task #576 — counted PDW2 coefficient spatial receiver

## Outcome

**MEASURED [macOS-CPU advisory]: the sealed 138-byte PDW2 packet is genuinely consumed and
deterministically maps an explicit real rank-4 quotient field to spatial argmax labels, but it is
not a self-contained spatial generator and cannot be through-R equivalent.** The exact blocker is
`PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY`. `d_seg` and `d_pose` remain unmeasured. No hard
CPU Torch oracle was run because there are no decoded RGB bytes to score. The preserved pointer is
`0.1910828242 [contest-CPU Linux x86_64]`, **UNMOVED**.

This is the optimal-form residual requested by the gate, not a family kill. It closes only the
map `packet bytes -> arbitrary video spatial partition`. The broader
`packet + counted spatial generator -> quotient field -> scorer-free RGB pullback -> R` family
remains open.

## What now exists

- `pdw2_spatial_receiver.py` strictly parses and canonical-reencodes PDW2/PDP2, accepts only
  `float32 [N,384,512,4]`, and streams native-fp32 first-max labels, counts, and hashes per pair.
- `probe_pdw2_spatial_receiver.py` reads the quotient `.npy` as a read-only memmap, runs n24/n600,
  records RSS, proves coefficient consumption by mutation, and emits no large label artifact.
- The integer-plane byte-close receipt now emits the exact blocker ID while retaining
  `pdw2_spatial_receiver_consumed=false`.
- `pdw2_coefficient_only_spatial_nonidentifiability_v1` is registered with the measured receipt as
  an empirical anchor.

## Measurements

| Quantity | n24 | n600 | Authority |
|---|---:|---:|---|
| spatial cells | 4,718,592 | 117,964,800 | MEASURED [macOS-CPU advisory] |
| quotient content SHA-256 | `240d09bb...cd62` | `cb6c28b4...6317` | MEASURED |
| partition-label SHA-256 | `a49962d8...bae2` | `7b4558bc...d870` | MEASURED |
| label counts, classes 0..4 | `[1077102,29883,2330251,72381,1208975]` | `[8980186,224892,98568848,429289,9761585]` | MEASURED |
| wall time | 0.92 s | 12.32 s | MEASURED local |
| `ru_maxrss` | 164,495,360 B | 1,976,549,376 B | MEASURED |
| macOS peak physical footprint | 71,582,296 B | 72,566,608 B | MEASURED |

The n600 replay independently reproduced the packet hash, quotient-content hash, partition hash,
all five counts, and the mutation canary bit-identically. macOS counts the file-backed mapping in
`ru_maxrss`; the separately reported physical-footprint counter stayed 72.6 MB. Both values are
preserved rather than conflated.

The packet is 138 raw bytes, SHA-256
`93c0d3320e6673aed1975426a6c8c1bbc41475f295ea62b357ad7a6bf9427568`. Brotli-q11 is 133 bytes.
Their isolated rate terms are respectively `9.188853553085965e-05` and
`8.855924076524879e-05`; neither is an archive saving because the necessary spatial generator and
RGB pullback are absent.

## Why the blocker is exact

The packet declares a global affine head. For a quotient feature vector `z(x)`, it computes

`P(x) = argmax_c l_c(z(x))`.

It does not contain `z(x)`. Under the same sealed bytes, the executable witness maps constant
`z_a=(-4,0,0,0)` to class 1 and `z_b=(0,0,0,0)` to class 2. Therefore no function of the packet
bytes alone can identify both spatial partitions. On the real n24 quotient field, moving the first
relative coefficient by float32 `+1.0` changes 108 labels, proving the implementation consumes the
coefficients rather than copying cached labels.

The real quotient file itself is 1,887,436,928 bytes and is not counted receiver state. Even if a
compact generator produced it, the current code still owes an offline-compiled, scorer-free
RGB/camera pullback whose decoded uint8 bytes survive the actual resize/frozen-trunk cells. A target
label tensor is not such a pullback.

## Adversarial review

Round 1 found and fixed six bug classes: an always-failing packet validator, a non-distinct witness,
whole-memmap reductions, uint8 label-hash truncation, missing probe/integration, and false contest-axis
provenance. The durable review is
`.omx/research/pdw2_spatial_receiver_576_round1_review_20260719.json`. The focused suite is 57/57
green; Ruff and `py_compile` are clean.

## Reformulation queue

1. Compile a counted temporal/spatial generator for the rank-4 field; the 1.887 GB memmap is evidence,
   not legal payload.
2. Compile an offline RGB/preimage receiver that does not ship or invoke SegNet/PoseNet and prove its
   exact uint8/resize parse-back.
3. Only then run byte-close n600 Seg/Pose and exact contest-CPU/CUDA replays.
4. If a residual/carrier becomes admissible, obey the later operator directives: rank in the
   Fisher/margin metric, use the corrected inner-Jacobian secant/QP law, sparsify with
   curvelet/shearlet rather than Fourier, factor pose through one SE(3) xi, and stop at the KKT rate
   waterline. Per-flip sparse patching remains out of scope and dominated.

## Triality and custody

- **DSL:** N/A with rationale. This unit adds no trainer lever or config; inventing a flag for a
  target-only receiver would create an argv marker without runtime authority.
- **DAG:** `FEED-PDW2-SPATIAL-RECEIVER-576` records the measured stop and reactivation edges.
- **Equations:** `pdw2_coefficient_only_spatial_nonidentifiability_v1` carries the n600 anchor.
- **Artifact custody:** packet, n24/n600 probes, replay, receipt, and review are under
  `.omx/research/pdw2_spatial_receiver_576_measurement/` and the adjacent blocker receipt.

## STORES CONSULTED

Delegated Task #576 authority; `CLAUDE.md`; `AGENTS.md`; operating manual; PROGRAM; v7.5/v8/v10
specs; vehicle OS; #553 packet memo/receipt; M1 #575 findings/session summary; canonical task/lane,
equation, subagent, and inbox state; FEED-STEP2-CONVERGENCE; frozen frame-195 diagnostic; real n600
quotient memmap; receiver/emitter/power-diagram code and tests. Inbox directives were consumed
through `2026-07-19T19:48:01Z`. No paid provider, scorer runtime, or unrelated vehicle family was
consulted.

## MAIN landing requirement

This branch is recovery-written and not main authority. MAIN must review the complete base-to-head
diff, rerun the focused tests and n24 probe, verify the receipt/packet hashes, and only then merge.
