# Codex Findings: DDM G4 n600 spatial stationarity

Date: 2026-07-22
Verdict: `MEASURED_ADVISORY_SPATIAL_STATIONARITY_COMPLETE_XI_PROXY_SCOPED`

## Outcome

The spatial-stationarity premise is strongly confirmed in image coordinates:
`3,963,354 / 4,011,236 = 98.806303%` of exact v12 flip events repeat the same
predicted-to-target class transition at the same 512x384 pixel at least twice.
Only 47,882 transition loci are singletons.  A target-cache metric-Pose G1 xi
proxy links just 2 singleton events; this is not an independent physical-BEV
measurement and its metric-Pose transport side information is not free.

The compact receipt SHA-256 is
`bea555b95aeaa11f4209df5333010c41c5495dd789def2a4f7a2a91973f3408c`.
Its 17 typed rows and 21 receipt/output/source custody rows revalidate.  The
analysis reuses settled G3/V12 frozen-scorer cells; it runs no scorer forward
and creates no archive candidate.

## Spatial concentration

| Pixel budget | Pixels | Flip mass | Fraction of 4,011,236 |
|---|---:|---:|---:|
| Top 1% | 1,967 | 850,575 | 21.204811% |
| Top 5% | 9,831 | 2,756,634 | 68.722808% |
| Top 10% | 19,661 | 3,605,023 | 89.873121% |

The map shows stable horizon/road-edge bands, the two Lane-corridor edges,
the broad Movable/Undrivable mid-band, and the hood rim.  PNGs and resumable
arrays live under
`/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_20260722T212138Z/`.
The receipt carries each path, byte count, and SHA-256.

## Stationarity partition

The partition is hierarchical and disjoint: same-pixel exact-transition
recurrence first, then length-at-least-two xi-proxy tracks among singleton
events, then transient.

| Stratum | Flip mass | Static image | Xi proxy | Transient |
|---|---:|---:|---:|---:|
| All | 4,011,236 | 3,963,354 (98.806303%) | 2 (0.000050%) | 47,880 (1.193647%) |
| Boundaries | 1,139,611 | 1,124,391 (98.664457%) | 2 (0.000175%) | 15,218 (1.335368%) |
| Hood rim | 15,646 | 15,400 (98.427713%) | 0 | 246 (1.572287%) |
| Lane corridor | 1,002,333 | 980,077 (97.779580%) | 0 | 22,256 (2.220420%) |
| Movable band | 1,083,972 | 1,074,751 (99.149332%) | 0 | 9,221 (0.850668%) |

The exact recurrence bands are:

| Exact-transition k | Loci | Flip-event mass |
|---|---:|---:|
| 1 | 47,882 | 47,882 |
| 2-4 | 24,994 | 64,230 |
| 5-9 | 12,304 | 82,779 |
| 10-29 | 18,835 | 338,583 |
| 30-59 | 12,107 | 515,201 |
| 60-119 | 11,199 | 943,243 |
| 120-600 | 9,558 | 2,019,318 |

The typed ledger retains every exact nonzero `k`, not only these bands.

## Amortization ranking

Bytes are actual selected lengths from raw/zlib9/Brotli-11/raw-LZMA plus a
one-byte codec tag.  “Reach” is collateral-aware cell events fixed across the
600 settled pairs; `Delta d_seg` is a cell-space opportunity, not a receiver
claim.

| Rank | One-time field | Bytes | Coder | Reach | Cell-space Delta d_seg |
|---:|---|---:|---|---:|---:|
| 1 | Movable rows 174-215, Lane→Road | 12 | raw | 159,604 | 0.001352980 |
| 2 | Horizon row 212 ±4, Undrivable→Road | 12 | raw | 119,546 | 0.001013404 |
| 3 | Two-line Lane wedge ±4, Lane→Road | 27 | raw | 10,145 | 0.000086000 |
| 4 | Movable-band sparse static field, 7,793 rules | 1,533 | Brotli-11 | 437,531 | 0.003708996 |
| 5 | Full sparse static field, 19,661 rules | 4,107 | raw-LZMA | 920,921 | 0.007806744 |
| 6 | Boundary sparse static field | 4,111 | raw-LZMA | 918,017 | 0.007782127 |
| 7 | Lane-corridor sparse static field | 4,030 | raw-LZMA | 818,885 | 0.006941774 |
| 8 | Hood sparse static field | 34 | zlib9 | 4,015 | 0.000034036 |
| 9 | One xi-proxy seed/lifetime | 19 | raw | 2 | 0.000000017 |
| 10 | Quadratic hood curve | 21 | raw | 0 | 0 |

The 12-byte bands are the first successor smoke controls.  They do not
authorize an RGB receiver: class-cell overrides must still be realized through
the actual renderer/resize/uint8 chain, and Pose collateral must be measured.
The xi row prices only the real track record; total xi bytes remain unresolved
because metric-Pose transport side information is not decoder-free.

## Free context

- Context-free Jeffreys-KT: `25,254,954.1711` bits.
- Per-pixel causal KT at zero context bytes: `12,343,747.2289` bits, a
  `51.123462%` ideal reduction.
- Real generic aggregate-pixel traversal: `490,794 -> 401,633` selected bytes,
  saving `89,161` bytes (`18.166685%`).
- Predictor argmax boundary-distance proxy: `41.913343%` ideal KT reduction,
  but the measured real stream is `683,211` bytes, `192,417` bytes worse than
  context-free.  It is rejected as the real coder despite its entropy proxy.

The old #141 producer is real frozen-SegNet autograd, but its stored
measurement is vehicle-relative to bc20 and the standing recalibration audit
explicitly forbids reuse on a new vehicle.  V12/G3 retained argmax cells and
target-margin bands, not the current predictor's full margin tensor.  G4
therefore does not falsely call the boundary-distance proxy a margin map.

## Round-1 adversarial review and blocker delta versus #603

Round 1 found three implementation defects before sealing: ordinal receiver
Pose6 codes were initially misused as metric coordinates; small Accelerate
`matmul` surfaced invalid floating-point flags; and a stale final checkpoint
could self-enter a rerun receipt and break resume.  The final pass uses
SHA-bound metric Pose6 with explicit non-free scope, deterministic scalar
homography products, and excludes the final marker from receipt inputs.  Eight
focused tests pass; hash-validated resume completes in 0.36 seconds.

Resolved versus #603: exact n600 concentration, recurrence `k`, per-stratum
stationarity, real-coded parametric/sparse field prices, and zero-payload
context gains.  Still owed: current predictor-margin custody, independently
observed cross-pair homography/liveCalibration, total xi side-information
bytes, RGB receiver realization, frozen Seg/Pose delta, and contest CPU/CUDA
authority.

Bounded re-derivation:

```sh
/Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_ddm_g4_spatial_stationarity.py \
  --config .omx/research/configs/ddm_g4_spatial_stationarity_n600_20260722.json \
  --resume
```

Pointer honesty: `0.1910828242 [contest-CPU]` unchanged.  MAIN landing review
is required.

STORES CONSULTED: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`,
`docs/operating_manual_craft_handoff.md`, G3 receipt/SSD atlas, v12 receipt and
frozen-score caches, `src/tac/margin_saliency_map.py`, #141 saliency and
recalibration memos, per-stratum BEV custody, G1 worldsheet measurements,
pointer/lane/task/subagent/operator ledgers, and both live inboxes.
