# RATE-ROUTE-HEVC-NVDEC-BITSTREAM-PROBE-20260518

Status: `research_only=true`; compact routing directive extracted from `rate_attack_novel_vectors_29_deep_research_20260518.md`.
No `.omx/state` mutation in this turn.

## Vectors

Primary: `H1,H2,H9`.
Support: `Y5,H4,H7`.

## Premise

Hardware-codec-as-deterministic-byte-derivation is only score-relevant if a compact charged bitstream replaces larger archive payload bytes and deterministic full-frame decode survives the contest runtime. NVDEC/DALI/NVENC alone are speed/search tools. T4 support makes H.264/HEVC the first realistic CUDA hardware-codec probe; AV1/VVC remain grammar/software paths until runtime closure is proven.

`prediction_only DeltaS=[-0.010,-0.002]`; high risk.

## Minimal smoke

Required outputs:

- Archive-byte accounting including bitstream, headers, decoder glue, and manifests.
- Decoded full-frame hashes for each axis/runtime tested.
- Runtime dependency manifest and runtime tree SHA.
- Component deltas from official evaluator path.
- Separate modes for H.264, HEVC, software x265/ffmpeg, and NVENC-produced streams.

Kill if:

- Decode requires unshipped external state, network installs, scorer access, or hidden driver state.
- Decoded frame hashes drift across target-equivalent runs.
- Runtime glue/header cost exceeds payload savings.
- Result is only wall-clock faster with no charged-byte reduction.

## 6-hook wire-in

- Sensitivity map: connect codec ROI/QP to hard-pair and boundary masks.
- Pareto: require component deltas and full-frame replay.
- Bit allocator: count all codec and decoder bytes.
- Cathedral autopilot: fail closed until deterministic decode manifest exists.
- Continual learning: update only after archive/runtime/frame hashes land.
- Probe-disambiguator: H.264, HEVC, software, and NVENC modes remain separate.
