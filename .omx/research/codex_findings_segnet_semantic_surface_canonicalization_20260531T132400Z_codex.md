# Codex Findings - SegNet Semantic Surface Canonicalization

Date: 2026-05-31T13:24:00Z
Axis: `[analysis; SegNet source-vs-candidate semantic bridge]`
Authority: false-authority planning signal only; no score, promotion, rank/kill, budget, or exact-dispatch authority.

## Finding

The SegNet semantic bridge had the right mathematical object in the JSON artifact
and CLI output, but the executable per-pixel repair surface lived as a
tool-local helper. That made the surface contract easier to orphan: repair
queues, MLX LoRA/DoRA adapter training, deterministic postfilters, and future
review tests could all reinterpret or reimplement the NPZ contents instead of
calling one TAC primitive.

The hardened contract is now:

- `build_segnet_semantic_bridge(...)` emits aggregate/semantic JSON.
- `build_segnet_semantic_surface_arrays(...)` emits the exact per-pixel arrays:
  source argmax, candidate argmax, source top2, source/candidate margins,
  boundary mask, wrong mask, Crammer-Singer hinge map, and sample ids.
- `write_segnet_semantic_surface_npz(...)` writes the canonical repair/adaptation
  NPZ and returns bytes, sha256, array names, and shapes.
- `tools/build_segnet_semantic_bridge.py` delegates to TAC instead of owning the
  surface grammar.

## Why This Matters For Score-Lowering Automation

The repair/action-functional loop needs a byte- and pixel-grounded object, not
an aggregate memo. The NPZ surface is the bridge from real contest-space SegNet
behavior into:

- deterministic boundary repair/postfilter materializers;
- MLX LoRA/DoRA boundary adapter training rows;
- posterior acquisition by class, boundary, region, frame, batch, and full-video
  scope;
- receiver/exact gates that can require the same surface without duplicate
  readiness readers.

The bridge remains advisory until archive bytes, runtime consumption proof, and
exact CPU/CUDA evidence exist.

## Adversarial Review Result

Two failure modes were pinned:

1. Tool-local surface writing is an orphan-signal risk. Fixed by moving it into
   `tac.analysis.segnet_semantic_bridge`.
2. Fleet-adaptable bridge requests must not enqueue contest-fixed postfilter
   work. Existing mode blockers are now covered by a repair-queue regression:
   incompatible backlog rows are ignored before queue rows are built.

## Verification

- `ruff check src/tac/analysis/segnet_semantic_bridge.py tools/build_segnet_semantic_bridge.py src/tac/tests/test_segnet_semantic_bridge.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_segnet_semantic_bridge.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`
