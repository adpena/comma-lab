# Retroactive sweep for `lane_mamba2_ssd_mlx_port_tri_backend_20260530` per Catalog #348

**Sweep timestamp**: 2026-05-30T19:47:00Z

## (1) Bug-class symptom signature

This is a NEW canonical helper landing (Mamba-2 SSD tri-backend MLX port);
the bug class it extincts is "orphan Mamba-2 SSD MLX impl" — every sister
class-shift substrate (Z8 hierarchical predictive coding, Z7-Mamba-2,
DreamerV3 latent-dynamics, Wyner-Ziv side-info) previously had to
per-substrate rediscover Mamba-2 SSD on Apple Silicon because no
canonical helper existed.

Symptom signature:
- Z8 `mamba2_adapter.py` currently wraps `Mamba2Predictor` (Mamba-1 S6
  per its own docstring) and PINS `backend='reference_torch'` — the
  Mamba-2 SSD form is NOT available for fast MLX-LOCAL iteration.
- Z7-Mamba-2 substrate uses `mamba_ssm.Mamba2` (CUDA-only) with
  `reference_torch` fallback (~10x slower at language scale per its
  own warning).
- `tac.optimization.mamba2_predictor.Mamba2Predictor` is Mamba-1 S6
  reference (per Wave 4 fidelity audit landed 2026-05-29); Mamba-2 SSD
  reference does NOT exist in repo before this landing.

## (2) Pre-fix window

Pre-2026-05-30 the apparatus lacked any canonical Mamba-2 SSD MLX
backend. OSS state per WebSearch 2026-05-30:
- `state-spaces/mamba` — CUDA Triton only
- `alxndrTL/mamba.py` — Mamba-1 only on MLX
- `beebopkim/mamba.py_mlx` — Mamba-1 only on MLX
- `purohit10saurabh/mamba-ssm-macos` — Mamba-1+2 with MPS (not MLX)

The pre-fix window extends from Mamba-2 paper publication (Dao+Gu 2024,
arxiv 2405.21060) to this landing 2026-05-30.

## (3) Historical-KILL/DEFER/FALSIFY search results

Searched canonical surfaces for historical KILL / DEFER / FALSIFY verdicts
on Mamba-2 MLX work:

* `git log --oneline --all | head -200 | grep -i mamba`: 5 sister Z7-Mamba-2
  landings (scaffold + full-landing + score-aware trainer wiring + recipe
  blocker cleanup + Wave 4 Dao-Gu fidelity audit). All AT or AFTER the
  canonical Mamba-1 S6 sister landing; no Mamba-2 SSD MLX KILL/DEFER/FALSIFY
  verdicts.
* `.omx/research/*killed*.md` + `*falsified*.md` + `*deferred*.md`: ZERO
  results matching Mamba-2 MLX.
* `.omx/state/probe_outcomes.jsonl`: no blocking probe outcomes on `mamba2`
  / `mamba-2` / `ssd` substrate ids.
* Canonical equations registry: equation `predictive_coding_with_hierarchical_mamba2_ssd_dreamerv3_wyner_ziv_v1`
  (sister of Z8 binding-contract design memo) has 0 anchors and is in
  DEFER status per Z8 build_progress.py M12a thresholding.

Per CLAUDE.md "Forbidden premature KILL": no historical verdict is
invalidated by this landing. Z7-Mamba-2 substrate-engineering scaffolds
remain valid sister substrate work; this canonical helper is the shared
infrastructure they can now consume rather than per-substrate
rediscovering.

## (4) Per-finding RE-EVAL priority assignment

Per-finding op-routables surfaced (also enumerated in the landing memo):

| Priority | Op-routable | Rationale |
|----------|-------------|-----------|
| HIGH | Z8 `mamba2_adapter.py` rewire to canonical helper via MLX backend | Unblocks Z8 M12a paired-CUDA RATIFICATION threshold sub-0.189 per Z8 build_progress.py canonical roadmap; class-shift substrate per operator 2026-05-30 emphasis. |
| HIGH | Z7-Mamba-2 substrate architecture.py rewire | Removes ~10x slowdown from `reference_torch` fallback; enables MLX-LOCAL fast iteration on M5 Max. |
| MEDIUM | Triton chunked-scan PyTorch optimization (Dao+Gu 2024 §4 Algorithm 1) | Paid-Modal-A100 / Vast.ai 4090 throughput; sister wave operator-routable. |
| MEDIUM | Metal kernel chunked-scan MLX optimization | M5 Max throughput; pending MLX `mlx.fast` API maturity. |
| LOW | state-spaces/mamba #669 gibberish-bug investigation | External bug surfaces at language scale; our contest-scale structural-invariant tests pass; document boundary in future Z7-Mamba-2 sister work. |
| LOW | Cathedral consumer `mamba2_ssd_dispatch_consumer` landing | Per Catalog #335 canonical contract; surfaces SSD-grammar candidates to autopilot ranker. |

## Per-Catalog #348 contract compliance

✅ **Bug-class symptom signature**: documented
✅ **Pre-fix window**: documented
✅ **Historical-KILL/DEFER/FALSIFY search results**: ZERO historical verdicts invalidated
✅ **Per-finding RE-EVAL priority assignment**: 6 op-routables enumerated

## Cross-references

* Sister landing memo: `.omx/research/mamba2_ssd_mlx_port_tri_backend_canonical_helper_landed_20260530.md`
* Sister canonical equation: `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1`
  in `.omx/state/canonical_equations_registry.jsonl`
* Sister lane: `lane_mamba2_ssd_mlx_port_tri_backend_20260530` L1
