# G2g2 joint multi-chart solve — measured hard-oracle result

**UTC:** 2026-07-21T17:44:16Z

**Lane:** `lane_g2g2_joint_multichart_solve_20260721`

**Authority:** MEASURED `[macOS-CPU advisory]`; frozen CPU Torch; seed 1234; no score claim; pointer `0.1910828242 [contest-CPU]` unmoved; `MAIN_REVIEW_REQUIRED=true`.

## Verdict

`MEASURED_G2G2_RATE_BREAK_EVEN_STOP_FAMILY_OPEN`.

The real counted G2g2 receiver admitted `0/6` hash-pinned G2f-selected pairs. Every measured prefix passed G2CS1 counted-byte parse-back, receiver-derived RGB, exact factor-2 uint8 realization, and independent double decode. Every prefix failed both whole represented-field semantic exactness and the declared pose tube. Each pair therefore stopped at the first measured prefix whose incremental recovered score per byte was at or below `lambda*=25/37,545,489 = 6.658589531221714e-7`.

The result is not a chart-family negative and is not a global-optimum claim. It scopes only the bounded projected greedy/coordinate-polish/one-swap search over the 20 decoded LaneLine centerline coordinates for pairs `[0,34,37,46,22,30]`, using a linear response model for proposal and the real nonlinear receiver for every verdict. It does not close other chart parameters, a nonlinear joint solver, xi-factorized pose, a different chart basis, or any route outside the measured search path.

## D1 — bounded joint solve with actual counted bytes

For each pair the runner derived 20 addresses dynamically from five decoded LaneLines and four centerline coefficients per line. It measured full SegNet-5-logit and PoseNet-6 central secants at `+/-0.5` native pixels through the actual G2CS1 receiver, AA-SDF raster, factor-2 R operator, and frozen scorer. This emitted 40 one-row response packets per pair, 240 total. Full logits, response tensors, frames, and coverage planes stayed in memory and were not persisted.

The proposal objective was lexicographic: whole-field semantic mismatch count, exact quantized pose-tube outside debt, then negative target-versus-rival margin debt weighted by the frozen categorical Fisher geometry. The amplitude alphabet was `[-16,-8,-4,-2,-1,-0.5,+0.5,+1,+2,+4,+8,+16]` native pixels. The deterministic search used projected greedy additions, up to four coordinate-polish passes, and one selected/unselected swap pass. Actual packet price was 20 bytes at k=1 and 8 incremental bytes thereafter; the model never had admission authority.

## D2 — real receiver and hard-oracle curves

| pair | source scope | measured k | best measured prefix | best cumulative recovered S | semantic mismatches @best | pose debt @best | stop k/bytes | stop marginal S/byte | admitted |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | chart-only | 1 | k=1 / 20 B | -1.669541527 | 99,681 | 171.828315 | 1 / 20 | -0.0834770764 | false |
| 34 | chart-only | 1–4 | k=3 / 36 B | 24.245069244 | 116,714 | 84.064136 | 4 / 44 | -0.4490530752 | false |
| 37 | chart-only | 1–3 | k=2 / 28 B | 37.465500051 | 127,556 | 9.135073 | 3 / 36 | -0.4827959126 | false |
| 46 | chart-only | 1–3 | k=2 / 28 B | 1.327521170 | 136,542 | 168.684692 | 3 / 36 | -0.0251794332 | false |
| 22 | pixel/chart overlap | 1 | k=1 / 20 B | -0.904122309 | 139,821 | 158.063307 | 1 / 20 | -0.0452061155 | false |
| 30 | pixel/chart overlap | 1 | k=1 / 20 B | -0.108900710 | 136,542 | 173.104848 | 1 / 20 | -0.00544503548 | false |

The strongest direction was pair 37 at k=2: 28 actual bytes recovered 37.465500051 score units and reduced pose-tube debt to 9.135073, but 127,556 semantic mismatches remained. Pair 34 improved through k=3 before k=4 regressed. Pair 46 improved through k=2 without moving its semantic mismatch count. Pairs 0, 22, and 30 were already below rate break-even at k=1. These rows establish useful local directions; none is an admitted correction route.

All measured prefixes had `counted_bytes=true`, `receiver_RGB=true`, `factor2_uint8_exact=true`, and `double_decode=true`. All had `semantic_exact=false` and `pose_tube=false`. The full six-pair search took 1,201.898939 measured wall seconds and 126,234 model-objective evaluations.

## D3 — routes and gates

- U1 Einstein/Kolmogorov routing: `false`.
- P0 G6/#603 registration: `false`.
- n64 gate: `closed`; the six-pair prerequisite did not admit all six.
- n600: `refused`; no n64 admission authority exists.
- Correction-price coefficient: not authorized because no hard-oracle row admitted.
- Family status: open. A future attempt must change the formulation, not continue later prefixes below the measured marginal break-even on these paths.

## Custody, resumability, and verification

- Implementation commit: `23183462e6d3d3d22d5382b157f05200c5edcaa7`.
- Full SSD receipt: `/Volumes/VertigoDataTier/pact/evidence/g2g2_joint_multichart_20260721/measurement_20260721T172244Z/receipt.json`, 860,288 bytes, file SHA-256 `928d3cd74cc92ef52aa9f821229ada12fbf4c3e9dad772e8a76adffcfcfcb078`, canonical receipt SHA-256 `b54e9816c3ef6ec7a93303fce62127c0b2fa0c59f094a3182e266af5cf8157f4`.
- Config SHA-256: `778ea2311fbabf89f29836cbee364035fc47926a1b5581cd6cde037c69b11f7a`; run root: `/Volumes/VertigoDataTier/pact/evidence/g2g2_joint_multichart_20260721/measurement_20260721T172244Z/run_778ea2311fbabf89`.
- Immediate identical invocation resumed all six immutable pair stages in 5 seconds and preserved the 860,288-byte receipt byte-for-byte at the same file hash.
- SSD storage was 3.4 MiB. Automatic hygiene retained only compact G2CS1 packets, JSON stages, config, checkpoints, and receipt. No rebuildable large tensors or frames were written.
- Tests: 49 passed across the predictor and G2 measurement suites; pycompile, Ruff check, Ruff format check, and diff check passed. Both changed Python modules and both changed Python test files received two clean `review_tracker` passes before the implementation commit.
- The target lane was advanced coherently to L2 (`impl_complete`, `real_archive_empirical`, and `strict_preflight`). A whole-registry `lane_maturity.py validate` remains globally red on 110 older missing evidence paths; none names this lane, and no unrelated registry row was repaired or weakened here.

## Stores consulted

Delegated authority and live inbox; `CLAUDE.md`; `AGENTS.md`; v7.5/v8 specifications; `reports/latest.md`; lane/subagent registries; G2f and G2g receipts; #549 target-cell and pose-tube custody; openpilot LBND2 chart; #557 marginal pricing; #580/#547 realization and receiver helpers; resolved `witness_realization_lsb_regime_v1`, `separable_resize_full_kernel_direct_sum_v1`, and `realization_breakeven_bytes_v1` LawRefs.

## Pointer delta honesty

No archive was produced, no contest-CPU or contest-CUDA evaluation ran, no score was claimed, and the frontier pointer did not move. This branch requires MAIN to review the full base-to-tip diff and measured custody before merge.
