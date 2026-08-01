# G47 — low-distortion selected-state path audit

Date: 2026-07-26  
Lane: `lane_g47_lowdist_selected_state_path_20260726`  
Authority: read-only, original-payload, score-directed audit  
Repository HEAD at audit: `0058123af31779d83d1fc10a728389b0ce7823ec`

## Executive verdict

The forest-level score path is now narrower and more coherent:

1. We already possess an exact, public-shaped V10 receiver path that realizes the low-distortion C1/MS1 scorer state: `d_seg = 0.0001519690619574653`, `d_pose = 0.00010184327939026322`. Its distortion-only score is `0.04710980004607969`, but its counted archive is `409,526,925` bytes. The low-distortion existence proof is real; its representation is wrong.
2. We also already possess an original V9/direct-description receiver grammar at the correct order of rate: a complete n600 archive of `133,941` bytes. Its realization is wrong: `d_seg = 0.027470296224`, `d_pose = 163.061327281443`. The rate existence proof is real; its inverse realization is wrong.
3. If a freshly compiled original archive retained the measured `133,941`-byte V9 shape and realized the measured MS1 distortions, the exact score would be `0.13629561408621645`. This is arithmetic, **not a candidate or score claim**. It identifies the missing composition edge: a counted, factorized V9-semantic-to-V10-preimage program and public receiver.
4. ep725/G mechanics are a useful actuator seam but not the macro score path. The measured ep725 state has a zero-rate distortion floor of `36.03874263394379`; same-state recoding cannot cross `0.172` at any byte count. Current label-local G can alter the state, so no global impossibility theorem is claimed for the family, but it has no n600/public proof and cannot express the coupled low-distortion Y0/Y1 preimage object.

Pointer delta: **zero**. This audit does not move the frontier. It identifies the shortest original, executable route to a row capable of moving it.

## 1. Exact low-distortion custody

The selected low-distortion state is not hypothetical and is not merely a private tensor:

| Object | Exact custody |
|---|---|
| C1 archive | `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/archive.zip`; `409,526,925` bytes; SHA-256 `e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42` |
| C1 packet `0.bin` | `409,526,817` bytes; SHA-256 `aa1dbb5e2efff28cd0d31f5ee2a4b0575a248a27a431151bfcae64eb320d385b` |
| Predictor payload | `409,525,473` bytes; SHA-256 `b3a792e1d838673b9047b9bd7dea93f0946a57871d484aec017650b7c1b3846e` |
| Inflated raw | `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/inflated/0.raw`; `3,662,409,600` bytes; SHA-256 `31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b` |
| Scorer Y0 plane | `353,894,400` bytes; SHA-256 `5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566` |
| Scorer Y1 plane | `353,894,400` bytes; SHA-256 `6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc` |
| Immutable MS1 receipt | `.omx/research/ddm_ms1_min_description_lattice_solve_20260723T233549Z/receipt.json`; `8,624` bytes; SHA-256 `546a7fddb0225edb15b2254ab73e362758b7b0f244e4ff39cb7bfef25f779098` |
| Current MS1 ingest receipt | `.omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json`; `8,800` bytes; SHA-256 `1b7063a44574b0839ede08c807f348ad417be0492ac32d68634b124b9c2b1e97` |
| SENSE pair rows | `/Volumes/VertigoDataTier/pact/evidence/ddm_ms1_min_description_lattice_solve_20260723_final/sense/pair_rows.jsonl`; `1,170,365` bytes; SHA-256 `276dde04cc0d6f4f4df1bfb1c7544f997800da189d49e789d00f87e699073803` |
| Historical factorization | same evidence root, `sense/factorization.json`; `5,232` bytes; SHA-256 `1c798be26b6e8aeb4b259d9e56beedd0cd99f5e5d6b5c2c6ba59f1a0ee03b450` |

The current public-shaped decoder is `src/tac/witness_dsl/v10_production_receiver.py`. Its real path is:

`build_packet` -> `parse_packet` -> `decode_y_plane_pair` -> `realize_pair_frame1` -> archive inflate.

It reconstructs the counted exact uint8 scorer planes and applies the deterministic factor-2 realization. Therefore the honest answer to “can a current counted/public decoder realize the low-distortion state?” is:

- **Yes**, the dense V10 C1 archive and production receiver do so, at 409 MB.
- **No**, neither the compact ep725/LVPG2 public decoder nor the current G17 selected-state envelope carries this low-distortion state.

The MS1 frozen-scorer diagnostic is `d_seg = 0.0001519690619574653`, `d_pose = 0.00010184327939026322`. Therefore:

`D_MS1 = 100*d_seg + sqrt(10*d_pose) = 0.04710980004607969`.

At this fixed distortion, the strict archive ceilings are:

| Target | Largest archive below target | Score at ceiling | Next byte |
|---|---:|---:|---:|
| `< 0.172` | `187,562` B | `0.17199963701158040` | `187,563` B gives `0.17200030287053352` |
| `< 0.15` | `154,522` B | `0.14999965720042385` | `154,523` B gives `0.15000032305937698` |

These are coupled score surfaces, not arbitrary per-component gates.

## 2. What the full-lattice and capped solves actually established

### MS1: exact low distortion, wrong description objective

MS1 searched feasible lattice members and compared local conditional zlib sizes. It did **not** optimize the final archive, shared cross-pair factors, a generative program, the nonlinear contest score, or even a global entropy model.

- Standalone exact member representation: `744,608,961` B.
- Previous-frame conditional representation: `731,622,325` B.
- Savings: `12,986,636` B = `1.744088%`.
- It remained `4,734.684x` the then `154,524`-byte target.
- Pose-xi conditioning was worse: `757,559,811` B total.
- Local CVP accepted `0/1200` members in both previous-frame and pose-xi formulations.
- The historical factorization found eight numerical SVD rows, six above a one-byte floor, and distilled zero shippable factors.

This falsifies those local formulations, not the existence of a compact selected solution. The selector rewarded “locally zlib-smaller canonical member,” whereas the contest rewards “globally shortest decoder program plus video-derived statistic under coupled scorer distortion.” Those are different optimization problems.

### MS2R R2: real -28.892% bytes, but a zero-rate score failure

The distortion-convergence-capped R2 solve is also real and byte-closed:

- Receipt: `.omx/research/ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/receipt.json`; `92,594` B; SHA-256 `03cd9aabc1275c49c983631dd547e7497f8fe95804a9bfd7a24c5d61e9a81d25`.
- Archive: `/Volumes/VertigoDataTier/pact/ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/stage_checkpoints/04_candidate/archive.zip`; `291,205,400` B; SHA-256 `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e`.
- Packet: same directory, `0.bin`; `291,205,292` B; SHA-256 `daf1e1db6314e8cdbf63347afa35899e9891e3068428d42dc5a2fca235bb5295`.
- Selected exact DP allocation: 208 q4 pairs + 392 q8 pairs.
- Realized `d_seg = 0.001159998575846354`, `d_pose = 0.01663315449034709`.
- Savings from C1: `118,321,525` B = `28.8922456075385%`.
- Exact distortion-only score: `0.5238375028596528`.
- Exact score at its archive size: `194.42556029038283`.

Because `0.5238375 > 0.172`, no recoding of this exact selected state can move the current frontier, even at zero bytes. Its fifty stream races choosing RAW 50/50 show that those particular independent streams have no easy local compressor; they do not establish global program incompressibility. MS2R remains a useful teacher/control and factor-harvest source, but an H1 lossless recode of it is not the shortest score route.

## 3. The under-our-nose composition

The original direct-description line already establishes the complementary rate fact:

- `.omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes`
- Exact n600 archive size: `133,941` B.
- SHA-256: `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- Receiver: `CarrierComposeReceiverV1` in `src/tac/optimization/direct_description_carrier_compose.py`.
- Counted homes: predictor ZIP `100,099` B; worldsheet `29,878` B; realization profile `85` B; six scorer-solved templates `151` B; remaining framing/manifest.
- All archive bytes have one counted home, all five semantic roles are consumed, and the decoder has no scorer dependency.
- Its measured realization is bad: `d_seg = 0.027470296224`, `d_pose = 163.061327281443`.

A later M5R full-n600 selected endpoint is `134,211` B with `d_seg = 0.02855653550889757`, `d_pose = 163.05168806999262`; it confirms that marginal template repair did not fix the semantic-to-photometric inverse edge.

The useful synthesis is therefore:

`fresh source -> original V9 task-space grammar -> counted selected-preimage program -> generic V10 factor-2 realization -> expected video -> exact upstream scorer`.

It is **not**:

- ship or copy the historical V15 archive;
- compress the 409 MB dense planes with another generic byte codec;
- store the target argmax table;
- keep tuning ep725/G while its base lies in a 36-point distortion basin;
- run a scorer or hide video-derived data in `inflate.py`.

At `133,941` B, the V9-shaped representation has `53,621` B of headroom to the MS1 `<0.172` ceiling and `20,581` B to the MS1 `<0.15` ceiling. At `134,211` B, it has `53,351` B and `20,311` B respectively. This is enough headroom to justify one full-system build, but not enough to justify unpriced sidecars.

The exact-target CPC1 partition size (`255,288` B) reinforces the geometry: semantic task-space entropy is close to the competitive scale but still too large and is a forbidden GT table payload. It must be explained by a lawful factor/worldsheet/program grammar, not shipped.

## 4. Why ep725/G is not the score path

The exact n600 ep725 bridge receipt is:

`.omx/research/original_taskspace_inverse_witness_codec_20260725/g28_ep725_xcodec_full_n600_bridge_eval_20260726.json`

- Receipt SHA-256: `7930b14740189a5b1e02a6d1406ded975aecbcb8730a9da8c1fa18decd3509d0`.
- Archive: `81,027` B; SHA-256 `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8`.
- Raw SHA-256: `8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae`.
- `d_seg = 0.0035127170849591494`.
- `d_pose = 127.35955810546875`.
- Distortion-only score: `36.03874263394379`.
- Total score: `36.09269518733842`.

The G25 same-state archive is `80,238` B. Replacing 81,027 B by 80,238 B saves only about `0.000525` score units; the same-state distortion floor remains decisive.

G43's `EP725_LABEL_LOCAL_SEMANTIC_G` is a real first actuator and must not be dismissed as byte plumbing. It can change owned semantic cells and their camera values, so it can change d_seg and may change Pose incidentally. But under `NoTransportV2` it cannot change Y0, unowned Y1, use worldsheet/transport atoms, or carry a causal Pose6 target. It has only bounded mechanics evidence, no n600 scorer row, and no current public decoder closure. The current `G17ActuatorIRV1` closed enum contains only this ep725-local kind; it cannot name or carry a coupled low-distortion V10 preimage program.

Therefore G-only work is correctly scoped as actuator economics and possible residual repair after the macro representation exists. It is not an executable frontier plan by itself.

## 5. Exact missing type and adapter

Do not create a parallel ontology. The required new packet type should bind existing objects:

### `TaskspaceSelectedPreimageProgramV1`

A counted, deterministic packet section with:

1. a freshly compiled `CarrierComposeReceiverV1` semantic/worldsheet program (mechanism donor only; no historical archive bytes copied);
2. an explicit generic decoder ID and version;
3. counted video-derived preimage operands: shared factor/basis coefficients, pair trajectories, sparse topology/event residuals, and only the irreducible chroma/luma correction payload;
4. a coupled pair contract that decodes both uint8 scorer-plane members `(Y0, Y1)` before V10 realization;
5. exact source/config/payload hashes and byte homes;
6. double-decode raw equality and public archive parse-back receipts.

The decoder API must be real:

```python
def decode_selected_preimage_pair(
    program: TaskspaceSelectedPreimageProgramV1,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact uint8 scorer-resolution (Y0, Y1) planes."""
```

The one missing adapter is:

```python
def compile_v9_v10_selected_preimage_program(
    semantic_program: CarrierComposeReceiverV1,
    selected_preimages: Sequence[tuple[np.ndarray, np.ndarray]],
    coupled_obligations: EvaluatorObligationIR,
    config: SelectedPreimageCompileConfigV1,
) -> TaskspaceSelectedPreimageProgramV1:
    """Encode/factor a coupled low-distortion preimage solution into counted program operands."""
```

Existing donors already cover the neighboring mechanics:

- `compile_carrier_compose_archive` / `receive_carrier_compose_archive` and `CarrierComposeReceiverV1` in `src/tac/optimization/direct_description_carrier_compose.py` — compact original semantic/worldsheet grammar and receiver.
- `EvaluatorObligationIR`, `PairPreimageReceipt`, `ExplicitV10PreimageCompileResult`, and `compile_explicit_v10_preimages` in `src/tac/witness_dsl/evaluator_obligation_ir.py` — coupled evaluator and exact-preimage custody.
- `solve_constructive_projection` and `realize_factor2_and_require_hard_oracle` in `src/tac/optimization/v10_constructive_solver.py` — encode-side V10 preimage construction.
- `realize_factor2_uint8_scorer_plane`, `verify_factor2_uint8_scorer_plane`, and `repair_with_hard_oracle` in `src/tac/optimization/uint8_lattice_feasibility.py` — exact R-lattice construction and encoder-only repair.
- `build_packet`, `parse_packet`, `decode_y_plane_pair`, and `realize_pair_frame1` in `src/tac/witness_dsl/v10_production_receiver.py` — public packet/realization path.
- `CoupledWitnessState` in `src/tac/witness_dsl/coupled_witness_state.py` — coupled ownership, not a new state universe.
- `taskspace_selected_solution_compiler.py` and `taskspace_g17_actuator_ir_v1.py` — current selected-state ownership and closed-enum placement surfaces.

`compile_explicit_v10_preimages` currently validates exact caller-supplied dense uint8 planes and then emits the dense V10 packet. Its own contract intentionally does not implement semantic-class-to-RGB realization. That absent realization/factorization adapter is the bridge; another wrapper around dense planes is not.

The scorer and `repair_with_hard_oracle` stay encode-side. Decode-time work may be arbitrarily sophisticated deterministic generic computation—factor expansion, spline/raster evaluation, integer projection, nullspace/gauge fill, chroma/luma repair—but every video-derived coefficient or residual is counted. No decoder-side scorer, target table, or hidden learned payload is admissible.

## 6. The one next build and non-toy command

Build exactly one full-n600 vertical artifact: a **freshly compiled original** V9 semantic/worldsheet program plus a counted V10 selected-preimage realization program, decoded twice through the public receiver into all 1,200 expected video frames and measured by the frozen upstream scorer.

The build should land:

- `src/tac/witness_dsl/taskspace_selected_preimage_program_v1.py` with the type and two APIs above;
- a new V10 packet codec/section ID rather than mutating the frozen dense V1 packet semantics;
- `tools/run_taskspace_selected_preimage_n600.py`, driven only by a typed config;
- an SSD run root such as `/Volumes/VertigoDataTier/pact/taskspace_selected_preimage_n600_20260726/` with stage checkpoints, auto-clean certification, archive/raw hashes, receiver double-decode equality, all byte homes, exact `d_seg`, exact `d_pose`, and exact score.

After that landing, the one command is:

```bash
.venv/bin/python tools/run_taskspace_selected_preimage_n600.py \
  .omx/research/configs/taskspace_selected_preimage_n600_20260726.json
```

This command does **not exist yet**; the next build must make it executable. The run is admitted only if it:

1. uses all 600 pairs and emits the full expected video;
2. compiles fresh from the current source/encoder-only target custody rather than copying V15/C1 archive bytes;
3. performs a joint score-aware fit/factor selection against final counted archive bytes, not per-component arbitrary thresholds;
4. preserves per-stage checkpoints and deterministic double-decode equality;
5. ends in one byte-closed public archive with an exact upstream row.

The first hard stop is coupled score, not a component gate: if a produced archive's measured `(d_seg, d_pose, bytes)` cannot beat `0.172`, do not spend another unit characterizing that same basin. Inspect the residual-cost ledger, retain only score-positive factors, and pivot the realization family while preserving the V9 task-space rate grammar and V10 exact-preimage contract.

## 7. Stores consulted

- `CLAUDE.md` / `AGENTS.md` (byte-identical at audit time) and `PROGRAM.md`.
- Current lane registry, sibling checkpoints, recent directives, and top-10 project memory.
- MS1, MS2/MS2R, V10 C1, V15, M5R, G21, G39, G41, G42, G43, G45, G28/G29, and frontier-closure receipts/memos.
- The production receivers, direct-description grammar, constructive/lattice solvers, evaluator obligation IR, coupled state, selected-state compiler, G17 actuator IR, and current public ep725 inverse code.

## Final classification

- **Solved:** low-distortion V10 preimage existence and public dense realization.
- **Solved:** original V9 task-space representation at roughly the necessary rate.
- **Not solved:** a counted factorized program that maps the original semantic/worldsheet state to coupled low-distortion Y0/Y1 preimages.
- **Dominated as a macro route:** same-state ep725 recoding and MS2R lossless recoding.
- **Shortest score-directed action:** build the one missing selected-preimage program type/adapter and immediately produce one full-n600 public exact row.

