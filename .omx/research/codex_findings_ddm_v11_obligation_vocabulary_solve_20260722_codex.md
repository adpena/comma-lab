# DDM V11 scorer-obligation vocabulary solve — advisory finding

**Lane:** `lane_ddm_v11_obligation_vocabulary_solve_20260722`

**Tasks:** #603 / #613 on master #578

**Evidence:** `[macOS-CPU frozen-scorer advisory]`

**Authority:** `research_only=true`, `score_claim=false`, `d_seg_claim=false`, `d_pose_claim=false`

**Pointer:** `0.1910828242 [contest-CPU]` **UNCHANGED**

**MAIN landing review:** **REQUIRED**

## Outcome first

| window | atomic obligations raw / bounded | measured bundles / atoms | admitted bundles / atoms | exact bytes base → final | d_seg base → final | official YUV6 d_pose base → final | advisory objective base → final |
|---|---:|---:|---:|---:|---:|---:|---:|
| n64 `[448,512)` | 2,340 / 2,340 | 24 / 384 | **1 / 16** | 52,204 → **52,523** | .045286496480 → **.045230627060** | 159.104827981350 → **159.102320151146** | 44.451356696756 → **44.445667803122** |
| n256 `[344,600)` | 9,339 / 4,096 | 32 / 509 | **3 / 48** | 72,933 → **73,508** | .040169219176 → **.040152808030** | 157.798907948748 → **157.797341378821** | 43.789395685794 → **43.787940257364** |
| n600 `[0,600)` | 22,032 / 4,096 | 32 / 492 | **0 / 0** | 102,105 → **102,105** | .034502249824 → **.034502249824** | 163.039648911962 → **163.039648911962** | 43.896380982393 → **43.896380982393** |

The n64 positive is one 16-atom Lane dash-phase bundle for +319 bytes. The n256 positives are three
16-atom Lane-width bundles for +575 bytes total. Every admitted bundle improves the exact joint
objective; in these measured positives both d_seg and d_pose happen to improve. The admission law
itself prices pose worsening instead of vetoing it.

No row approaches `d_seg <= 0.00116`: the final gaps are 39.0x (n64), 34.6x (n256), and 29.7x
(n600). The 200,000-byte n600 budget ceiling was genuinely probed, but only 492 of the 4,096 bounded
atoms entered the 32 measured bundles and 3,604 remained unmeasured. Therefore the preregistered
wrong-worldsheet/formulation falsifier is **NOT TRIGGERED**. The durable verdict is
`ADVISORY_BOUNDED_WATERFILL_ABOVE_TARGET_UNMEASURED_OBLIGATIONS_REMAIN`, scoped to this exact
generated-inventory cap and measured bundle ordering. No carrier, predictor, or post-solve
correction family is closed.

## What became executable

- V5 and V6 now bind a real full `[0,600)` `fixed_ar1_hold24` archive. The V6 control is 100,056
  bytes and SHA-bound; the V11 empty correction envelope is 102,105 bytes.
- The V9 lineage now has a strict V11 archive schema with full Lane `c0-c3`, width `c4-c5`, and
  stored-xi-keyed dash phase `c7`; Road/Undrivable compact parabolic shearlet displacement atoms;
  and Movable birth/death shape moments plus a compact Fourier-free curvelet lobe and Pose6 transport.
- Correction packets have typed records, canonical ordering, CRC, parse/re-encode equality, one ZIP
  home per byte, strict address/window validation, and receiver no-op refusal. No pixel stream, RGB
  patch, scorer weights, or GT table enters an archive.
- Encoder-only target/predicted argmax error fields are ranked by rank-4 head flip distance, margin
  band, and Fisher curvature. Candidate ranking is proposal authority only; exact receiver replay
  through frozen SegNet and official YUV6 PoseNet is the admission authority.
- Atomic obligations are grouped by canonical scorer batch **and family**, capped at 16 atoms. This
  preserves batch reuse without aliasing a useful local correction with unrelated harmful strata.
- Candidate and budget checkpoints are immutable and preserved. Full n600 replay retains only
  argmax cells after each batch; one RGB scorer batch is resident at a time.

## Final residual decomposition

| window | Road | Lane | Undrivable | Movable | MyCar | boundary codim-1 | cell interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| n64 | .093118909613 | .496906799990 | .002658610997 | .999979319017 | .000095864440 | .470455178371 | .032996222878 |
| n256 | .082493501720 | .457585404801 | .004583978552 | .993207304160 | .001104240940 | .458539631362 | .028775778134 |
| n600 | .071837475662 | .424611121005 | .005010949479 | .988264941023 | .002402219760 | .427522828370 | .024043482406 |

Movable remains almost entirely wrong and Lane remains the next-largest target-class debt. Boundary
error is also an order of magnitude larger than interior error. This decomposition is consistent
with a high-EV structured search, but the current top-32 measurement cap is too narrow to decide
whether the remaining obligation grammar can pay. It does **not** prove that the v6 predictor's
worldsheet is the binding limitation.

## Round-1 adversarial findings

1. A fresh lint pass found that absolute source-pair IDs were batched against an undefined
   `pair_start`. The function now receives the typed window origin; a regression locks the n64
   `[448,512)` relative batch geometry.
2. The first n64 smoke collapsed every family in a scorer batch into one 128-atom proposal. All four
   mixed bundles failed, but that result was not a vocabulary verdict. The bundler was replaced with
   family-specific, 16-atom proposals before the ladder was accepted.
3. The first n600 receipt equated an archive that failed to spend its available budget with a
   near-200KB exhausted-vocabulary probe. Round 3 separates budget-ceiling coverage from inventory
   exhaustion. Round-2 and round-3 base archive, inventory, and all ladder archive SHAs are identical;
   only the falsifier semantics changed.

Two superseded generated output directories were deleted only after the durable cleanup manifest
recorded their exact byte counts, SHA-256 tree manifests, commands, false-authority flags, and
rebuildability. They are not recoverable in place but are deterministically rebuildable; no SSD write
was performed.

## Exact blocker delta versus the #603 register

**Discharged:** bound n600 V5/V6 predictor and exact advisory Seg/Pose bridge; obligation-derived full
Lane/shape/boundary vocabulary; Fourier-free receiver semantics; joint contest-objective admission;
family-specific canonical-batch measurement; exact 0/16/48/96/144 KiB requested ladder through a
200,000-byte ceiling; n600 chunk/OOM law; SHA-bound, resumable receipts.

**Remaining:** only 32 bundles are measured per window. At n600, 3,604 bounded atoms and 17,936 raw
atoms remain outside measured bundles. Before naming a v6-successor predictor from this experiment,
the next vehicle must increase measurement coverage with resumable successive-halving or finer
family/batch water-filling while preserving the exact joint scorer and 200,000-byte ceiling. The
dominant residual strata are Movable, Lane, and codimension-1 boundaries. Contest CPU/CUDA replay,
candidate promotion, and pointer movement remain unauthorized.

## Bounded re-derivation argv

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v11_obligation_n64_20260722.json --output-directory .omx/research/ddm_v11_obligation_n64_20260722T152000Z_round2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v11_obligation_n256_20260722.json --output-directory .omx/research/ddm_v11_obligation_n256_20260722T152000Z_round2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/run_ddm_v9_carrier_compose.py --config .omx/research/configs/ddm_v11_obligation_n600_20260722.json --output-directory .omx/research/ddm_v11_obligation_n600_20260722T152000Z_round3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_entropy_priced_member.py src/tac/optimization/tests/test_direct_description_carrier_compose.py
```

Fresh V11 wall clocks were 161.108845 s, 332.506300 s, and 518.230053 s respectively. Each is under
ten minutes. Completed receipts verify their typed-config hashes and return immediately.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`
- V10 finding, session summary, receipts, DAG FEED, and canonical-equations note; settled V10 rows were
  consumed and not re-derived
- V5/V6 n64/n256 receipts and fixed-AR1 archives; fresh V5/V6 n600 receipts and exact archives
- Frozen target receipt/cache plus upstream SegNet/PoseNet custody bound by each result receipt
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/canonical_task_status.jsonl`,
  `.omx/state/subagent_progress.jsonl`
- Per-arm inbox (empty) and fleet inbox through `2026-07-21T13:15:53Z`; the 2026-07-19 Fisher,
  shearlet/curvelet, xi-factorization, and reverse-waterfill directives were consumed

`0.1910828242 [contest-CPU]` remains unchanged. Every archive is a `.not_a_candidate` advisory
receipt, not a contest score, promotion record, or execution authority.
