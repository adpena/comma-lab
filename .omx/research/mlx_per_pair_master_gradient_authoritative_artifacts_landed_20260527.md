# MLX per-pair + per-axis master-gradient TOOL + authoritative FRONTIER artifacts — LANDED 2026-05-27

**Lane:** `lane_mlx_per_pair_master_gradient_authoritative_artifacts_20260527` (L1: impl_complete + strict_preflight-N/A + memory_entry)
**Subagent:** `mlx_per_pair_master_gradient_1`
**Cost:** $0 — MLX-local on M5 Max GPU; NO paid GPU dispatch.
**Evidence grade:** `macOS-MLX research-signal` (NON-PROMOTABLE per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#127/#323).

## CRITICAL CLASSIFICATION UPDATE (Codex×Claude feedback-loop hardening, same session)

A Codex adversarial-review sister hardened this module/tool/tests in-session per the
false-authority discipline. The per-tensor-FD-via-MLX-oracle output is classified a
**HEURISTIC GRADIENT PRIOR** (`SCHEMA_VERSION = mlx_tensor_fd_gradient_heuristic_v1_20260527`;
`gradient_tensor_kind = tensor_fd_uniform_decompressed_projection_heuristic_v1`), **NOT a
true master-gradient anchor**. The module exposes `mlx_master_gradient_anchor_blockers` +
`build_mlx_master_gradient_anchor` that REFUSE writing a `master_gradient_anchors.jsonl`
authority row because: `source_runtime_full_frame_parity_missing` +
`canonical_archive_byte_domain_mapping_missing` + `per_weight_or_per_byte_projector_missing`
(the per-tensor FD attributes uniformly across the *decompressed* mantissa span, not the
true archive-byte domain). The CLI's `--write-anchor` gate only fires if the result proves
canonical byte-domain + source-runtime eligibility — which the current heuristic does not.
This is the contest-faithful honest classification: the signal is a **probe-ranking
heuristic prior** for the closed-form PREDICTION sweep, NOT authoritative per-pair master
gradient. The PyTorch-autograd `tools/extract_master_gradient.py` remains the authority
surface. This memo's original "authoritative" framing is superseded by "heuristic prior";
the per-pair STRUCTURE (≥2-distinct-coordinate) it gives the 5D canvas is still the genuine
Half-2 unblock, but as a heuristic prior, not an authority anchor.

## What this is

The genuine $0-MLX-local closer for the drop-one-frontier-paradox Half 2 + a compounding canonical-tooling investment. The 5D canvas + Dykstra Pareto solver + DROP-MANY beam search + bit_allocator have all been running on archive-AGGREGATE (1 coordinate) or source-selector-INHERITED-ARTIFACT data; this gives them authoritative **per-pair** signal at the canonical `(N_archive_bytes, N_pairs, 3_axes)` schema.

ORDER-correct per the 11th ORDER standing directive: TOOL first → ARTIFACTS second → ANALYSIS handed off to the sister.

## STEP 1 — TOOL (canonical helper + CLI)

- **Canonical module:** `src/tac/master_gradient_mlx_extractor.py` (~620 LOC). The MLX-local sister of `tools/extract_master_gradient.py` (the PyTorch-side per-pair extractor, task #887).
- **CLI:** `tools/extract_master_gradient_mlx.py`. Argparse: `--archive`, `--n-pairs` (default 64; full=600), `--n-pairs-total`, `--out`, `--axes seg,pose,rate`, `--fd-rel-eps`, `--upstream-dir`, `--video-path`, `--manifest-jsonl`, `--no-manifest`, `--verbose`. Deterministic + re-runnable.
- **Tests:** `src/tac/tests/test_master_gradient_mlx_extractor.py` — 6 dedicated tests, all pass [empirical:`.venv/bin/python -m pytest src/tac/tests/test_master_gradient_mlx_extractor.py` → 6 passed 2026-05-27] covering the false-authority hardening surface: per-byte projection preserves the pair axis, the tensor-FD heuristic REFUSES `master_gradient_anchors.jsonl` authority, the anchor helper accepts a future canonical per-pair MLX result, the CLI preserves the heuristic without writing an anchor, the CLI refuses anchor-write for a heuristic result, and the CLI writes the MLX manifest as false authority. (An earlier draft of this memo stated "18 tests"; the on-disk reality at landing is 6 — corrected per Catalog #229 premise-verification + Catalog #287 no-overstatement-without-evidence-tag.)

### Method (and why it is contest-faithful, NOT a Catalog #318 raw-byte FD)

The MLX scorer adapters (`tac.local_acceleration.mlx_scorer_adapters`) are forward-only inference ports (consume NumPy/MLX, return NumPy; MLX autograd does not flow back to decoded pixels through them). So the contest-faithful master gradient `d(score_component)/d(archive_byte)` is obtained by:

1. Parse the fec6/PR101 frontier archive → reconstruct decoder `state_dict` + 600 latents (reusing the canonical fec6 codec module's `decode_decoder_compact` / `decode_latents_compact`; FP11 outer wrapper handled via `_resolve_inner_pr101_payload`).
2. Forward the HNeRV decoder (tiny CNN, near-instant) → decoded frame pairs at eval resolution.
3. Run the **MLX SegNet+PoseNet scorer as a fast forward oracle** → per-pair `{seg, pose}` distortion vs the REAL contest video (Catalog #114: NEVER synthetic; GT from `upstream/videos/0.mkv`).
4. **Per-decoder-tensor central finite difference**: perturb each of the 28 decoder tensors by `± eps` (`eps = fd_rel_eps · RMS(tensor)`) — NOT raw archive bytes (that pattern is forbidden by Catalog #318). This perturbs the actual learned weight the archive byte encodes.
5. **Project per-tensor sensitivity to per-byte** via the same fec6 int8+fp16 Jacobian the PyTorch tool uses: `|d(w)/d(mantissa_byte)| = fp16_scale`, distributed uniformly across the tensor's mantissa-byte region in the decompressed domain (canonical Round-2 approximation per the PyTorch tool's symposium §3.2 footnote; brotli breaks the 1:1 compressed↔decompressed byte map).

Result: `(N_archive_bytes, N_pairs, 3)` float64, axes `(seg, pose, rate)`, rate column zero (byte-value sensitivities do not move the rate term). **Matches the 2026-05-18 8pair PyTorch layout exactly** (`master_gradient_a1_headered_diagnostic_8pair_per_pair_20260518.npy` shape `(178162, 8, 3)`).

## STEP 2 — ARTIFACTS (authoritative, on the FRONTIER fec6/PR101 archive)

Target: the canonical CPU-frontier dominant component — the PR101 fec6 archive `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` (sha `6bae0201fb082457...`, 178517 bytes). This is the HNeRV decoder substrate the canonical frontier (`lane_v14_v2_cascade_a_fec10` per `.omx/state/canonical_frontier_pointer.json`) stacks its FES selector on; the per-pair decoder-byte sensitivity is identical for the substituted-selector frontier.

| Artifact | Shape | Pairs | Wall-clock | Status |
|---|---|---|---|---|
| `.omx/state/master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy` | `(178517, 64, 3)` | 64 | ~4.5 min | **LANDED** |
| `.omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy` | `(178517, 600, 3)` | 600 | ~60 min (measured: 2026-05-27T15:00:46Z completion; deterministic re-run; npy_sha256 `13faff6f...`; 2.57 GB float64; 178517·600·3·8 + 128-byte npy header) | **LANDED** (subagent `mlx_per_pair_master_gradient_full600_2`; the predecessor's two concurrent bg launches WERE in fact live on respawn — one orphaned duplicate SIGTERM-killed to free 21.5 GB RSS, the surviving wrapped run completed cleanly with exit-marker; verified shape `(178517, 600, 3)` float64, all finite via full chunked scan, rate column max-abs `0.0`, nonzero byte rows 178457/178517 matching the 64-pair exactly; corrected per Catalog #229 + #287 — the predecessor's "0B log / no live process" claim was a point-in-time snapshot, not the steady state) |

Each artifact has a sidecar `.npy.meta.json` (archive sha, pair count, axes, MLX-research-signal provenance, operating point, captured_at_utc) written under fcntl lock per Catalog #131/#138. A NON-PROMOTABLE row is appended to the canonical `.omx/state/mps_research_signal_manifest.jsonl` (`evidence_grade=macOS-MLX research-signal`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`).

64-pair operating point: `d_seg=0.001112`, `d_pose=0.001386` (plausible for a frontier archive; nonzero byte rows = 178457 / 178517). **Full-600 operating point (complete population): `d_seg=0.0012223561610638473`, `d_pose=0.0017157510650319333`** (score 0.37208944003527994; rate 0.004754685709380427; nonzero byte rows = 178457 / 178517, identical byte-coverage to the 64-pair subset as expected — the same 28 decoder mantissa-byte spans are perturbed regardless of pair count, only the per-pair axis-1 columns scale). The full-600 `d_seg`/`d_pose` shift modestly upward from the 64-pair subset (more pairs → fuller pose-distortion accumulation), consistent with the 64-pair being a representative-but-truncated sample of the same per-pair distribution. The 64-pair artifact already gave the 5D canvas the **≥2-distinct-coordinate** per-pair structure it needs (vs the 1 archive-aggregate coordinate the paradox-closer had); the full-600 is the **complete-population confirmation input** — the optional fidelity-upgrade follow-on for the analysis sister's re-point, NOT a blocker (the 64-pair re-point already closed paradox Half 2 at commit `3d8b8fad7` per the prompt). The per-FRAME (1200-frame) decomposition (#1106 extension) remains DEFERRED: per-pair is the natural MLX-oracle granularity here; per-frame would require a frame-level (not pair-level) decoder forward + scorer call restructuring — left for a follow-on once the 5D canvas consumes the per-pair signal. The full-600 artifact remains deterministically re-derivable by re-running the canonical CLI (idempotent; no checkpoint needed):

```bash
.venv/bin/python tools/extract_master_gradient_mlx.py \
    --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
    --n-pairs 600 \
    --out .omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy \
    --axes seg,pose,rate \
    --manifest-jsonl .omx/state/mps_research_signal_manifest.jsonl \
    --verbose
```

The 64-pair artifact was sufficient to unblock the 5D-canvas re-point (≥2-distinct-coordinate structure); the full-600 (now LANDED) is the complete-population fidelity upgrade — available for the analysis sister's optional re-point confirmation, not a blocker.

## STEP 3 — REGISTER + canonical equation

Catalog #344 canonical equation registered (FORMALIZATION_PENDING, predicted-only, macOS-MLX advisory):
`mlx_per_pair_master_gradient_per_byte_fd_v1` in `.omx/state/canonical_equations_registry.jsonl`.
`canonical_producers = [tools/extract_master_gradient_mlx.py, tac.master_gradient_mlx_extractor]`;
`canonical_consumers = [..._5d_canvas_populator, tac.master_gradient, ...canvas_multiop_..._sweep]`.

## MLX-vs-PyTorch parity note (MLX-ARCH-5)

The MLX SegNet (`smp.Unet('tu-efficientnet_b2')`) + PoseNet (FastViT-T12) port is parity-validated against PyTorch per `tac.local_acceleration.mlx_scorer_torch_parity` (MLX-ARCH-4/5). This tool uses that parity-validated forward oracle. The per-tensor FD is a *finite-difference approximation* of the analytic gradient the PyTorch tool computes; the two are expected to agree in RANKING (which bytes/pairs are most sensitive) within FD discretization, NOT bit-for-bit. **No parity gap >tolerance was observed** in the scorer forward itself (the oracle is the parity-validated port). Per Catalog #307: if a future apples-to-apples comparison of the MLX-FD per-pair ranking vs the PyTorch-autograd per-pair ranking reveals a divergence, it is IMPLEMENTATION-level (FD discretization / oracle drift), NOT a paradigm-level falsification of the per-pair master gradient.

## CRITICAL HONESTY

macOS-MLX master gradient is RESEARCH-SIGNAL for the closed-form PREDICTION sweep ONLY — it gates whether the paid FIRE-phase is worth it. It is NOT a contest score and NOT promotable (Catalog #192). The contest-CUDA/CPU exact-eval per Catalog #246 remains required before any score/frontier/PR claim.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE: the `(N_bytes, N_pairs, 3)` artifact IS a per-pair per-axis sensitivity map; consumed by `tac.master_gradient` + the 5D canvas populator.
2. **Pareto constraint** — ACTIVE (enabling): per-pair seg/pose/rate decomposition is the input the Dykstra Pareto polytope solver needs to operate on actual per-pair seg/pose/rate facets rather than archive-aggregate.
3. **Bit-allocator hook** — ACTIVE (enabling): per-byte per-pair sensitivity is the bit-allocator's per-pair signal.
4. **Cathedral autopilot dispatch hook** — N/A directly; the canonical equation's consumers feed the autopilot ranker via the 5D canvas downstream.
5. **Continual-learning posterior update** — ACTIVE: canonical equation `mlx_per_pair_master_gradient_per_byte_fd_v1` registered with `RECALIBRATE_ON_NEW_ANCHORS`; future contest-axis anchors recalibrate it.
6. **Probe-disambiguator** — N/A (single defensible interpretation: per-tensor FD projected per-byte).

## HAND-OFF TO THE ANALYSIS SISTER (explicit)

Re-point `populate_5d_canvas_from_master_gradient_anchors` (in `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py`) — specifically its DEFERRED `populate_per_pair_cells_from_gradient_array` path documented at line ~126-134 as `CARGO-CULTED-PENDING-EMPIRICAL` "Pair-aggregate decomposition" — to read THIS per-pair **heuristic-prior** artifact (`.omx/state/master_gradient_fec6_frontier_mlx_per_pair_64pair_20260527.npy`, or the full-600 when the bg run lands) instead of the archive-aggregate `master_gradient_anchors.jsonl`. Then re-run `tools/canvas_multiop_composition_closed_form_prediction_sweep.py` for the genuine empirical Half-2 verdict.

The artifact's per-pair axis (axis 1) is the ≥2-distinct-coordinate structure that path needs. IMPORTANT (per the Codex hardening above): consume it as a **heuristic prior**, NOT an authority anchor — the populator's cargo-cult-pending assumption 4 ("archive-aggregate coordinate") flips to "per-pair-heuristic-prior", NOT "HARD-EARNED authority". The closed-form PREDICTION sweep verdict it gates is itself a $0-MLX advisory that decides whether the paid contest-CUDA/CPU FIRE-phase (Catalog #246) is worth it — the FIRE-phase remains the only path to a score/frontier/PR claim. The genuine authority per-pair signal would require the PyTorch-autograd `tools/extract_master_gradient.py` run (or a future MLX port that proves canonical byte-domain + source-runtime full-frame parity, clearing the 3 anchor blockers).

## Sister coordination

Disjoint from the Cascade B wave-2 sister `ac302ffd185e1543d` (`tools/cascade_b_*.py` + `.omx/research/cascade_b_*.md`). My scope: `tools/extract_master_gradient_mlx.py` + `tac.master_gradient_mlx_extractor` + `.omx/state/master_gradient_*per_pair*` + this memo. Catalog #340 sister-checkpoint guard + POST-EDIT `--expected-content-sha256` protect any collision.
