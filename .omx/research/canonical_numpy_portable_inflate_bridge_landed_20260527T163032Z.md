# Canonical numpy-portable inflate BRIDGE — landed

**UTC:** 2026-05-27T16:30:32Z
**Lane:** `lane_canonical_numpy_portable_inflate_bridge_20260527` (L1)
**Operator directive:** 8th MLX-first + numpy-portable standing directive (2026-05-26) —
*training MLX-first on M5 Max; INFLATE numpy-portable (no torch/mlx dep)*. This bridge is
the decode-side foundation that lets ANY trained substrate (PyTorch OR MLX) decode
framework-free at inflate time.
**Cost:** $0 (CPU/MLX-local; no paid dispatch).
**Deliverable:** `src/tac/substrates/_shared/numpy_portable_inflate.py` + comprehensive tests.

## The contract (what the two sister subagents build against)

Module `src/tac/substrates/_shared/numpy_portable_inflate.py` — fully typed, docstring'd,
fail-closed, OSS-grade. Public API:

### 1. torch-AGNOSTIC state_dict serialization
- `pack_state_dict_numpy(state_dict, *, dtype="fp16") -> bytes` — serializes a torch / MLX /
  numpy state_dict to a self-describing `{key: ndarray}` blob (`NPSD` grammar: magic +
  version + shared dtype-code + per-entry key/shape/raw-data). NO pickle, NO
  `torch._utils._rebuild_tensor` refs. Round-trips byte-stably. dtype names: fp16/fp32/int8/
  uint8/int16/int32/fp64/bool. Caller's archive grammar owns brotli (composable; no
  double-compression).
- `unpack_state_dict_numpy(blob) -> dict[str, np.ndarray]` — pure-numpy inverse, ZERO
  framework import, fail-closed on every malformation (bad magic / version / truncation /
  oversized counts / trailing bytes) via `NumpyPortableStateDictError`.
- `_as_numpy` duck-types torch (`.detach().cpu().numpy()`) / MLX (`np.asarray`) / numpy so
  the bridge module itself imports neither torch nor mlx.

### 2. canonical decode primitives (stable decode-side API)
Re-exported from the torch/MLX-free `tac.local_acceleration.pr95_hnerv_numpy_reference`:
`to_float32`, `linear`, `conv2d_nhwc`, `bilinear_upsample_2x_nhwc`, `sigmoid`, `sin`,
`mean`, `kahan_mean`. NEW decode-only primitives added in the bridge (the prompt's contract
names + the class-shift decoder needs):
- `conv2d_numpy` — stable alias for `conv2d_nhwc`.
- `bilinear_resize_nhwc(x, *, target_h, target_w, align_corners=False)` — generalized
  arbitrary-target resize (torch-free decode sister of the MLX helper; `align_corners`
  both branches; ≤1e-5 parity vs `F.interpolate`).
- `pixel_shuffle_2x_nhwc` — PyTorch `nn.PixelShuffle(2)` in NHWC (4× channel → 2× spatial,
  exact `(rh, rw)` row-major interleave; byte-stable vs torch). For boost_nerv / nirvana /
  atw convolutional decoders.
- `film_modulate_numpy(h, gamma, beta)` — FiLM `gamma*h + beta` (coin_plus_plus pattern).
- `gru_cell_numpy(x, h_prev, *, weight_ih, weight_hh, bias_ih, bias_hh)` — exact PyTorch
  `nn.GRUCell` (reset-inside-candidate; (r,z,n) gate order). For z5 autoregression.
- `tanh`, `relu`, `gelu` (tanh-approx) activations.
- `DECODE_PRIMITIVES` registry (16 callables) for introspection.

### 3. AST portability verifier
- `find_forbidden_framework_imports(source) -> [(lineno, framework)]` — pure AST walk; detects
  `import torch` / `from mlx import ...` / submodule imports; ignores relative imports; no
  substring false positives.
- `assert_inflate_is_numpy_portable(inflate_path)` — raises `InflateNotNumpyPortableError`
  on any forbidden framework (`torch`/`mlx`/`tensorflow`/`jax`/`jaxlib`). The canonical
  fail-closed check; consolidates the harness-stubbed gate to one place.

### 4. canonical numpy-portable runtime emitter
- `write_numpy_portable_contest_runtime(submission_dir, *, substrate_pkg_name, repo_root, ...)`
  — torch-FREE sister of `pact_nerv_full_main.write_contest_runtime`. Vendors ONLY the
  substrate's `archive.py` + `inflate.py` (NOT a torch `architecture.py`) + this bridge +
  the numpy primitive reference. Emits the Catalog #146 3-arg `inflate.sh` + a numpy-only
  `inflate.py` shim. `verify_portable=True` (default) runs `assert_inflate_is_numpy_portable`
  on EVERY emitted `.py` and fails closed if any forbidden import slipped in.

### raw-output writer
- `write_rgb_pair_to_raw_numpy(fh, rgb_0, rgb_1, *, input_range)` — numpy-native NHWC raw
  `.raw` lowering (torch-free sister of `inflate_runtime.write_rgb_pair_to_raw`) +
  `CONTEST_RAW_BYTES_PER_VIDEO` constant (Catalog #367 byte-count contract; 3,662,409,600).

## Primitives added vs the canonical reference

`pr95_hnerv_numpy_reference.py` was NOT modified (per sister-safety; the prompt said add new
functions only if editing it, and I did not need to). All NEW primitives
(`pixel_shuffle_2x_nhwc`, `film_modulate_numpy`, `gru_cell_numpy`, `tanh`, `relu`, `gelu`,
`bilinear_resize_nhwc`, `conv2d_numpy`, `write_rgb_pair_to_raw_numpy`) live IN the bridge
module, keeping the reference module untouched so the MLX-harness sister that imports it is
unaffected.

## Empirical parity (the proof — torch is the oracle)

All within FD tolerance vs torch reference:
- `linear` vs `F.linear`: ≤ 1e-5
- `conv2d_numpy` (NHWC) vs `F.conv2d` (NCHW transpose boundary): ≤ 1e-4
- `bilinear_resize_nhwc` vs `F.interpolate` (align_corners False AND True): ≤ 1e-5
- `pixel_shuffle_2x_nhwc` vs `F.pixel_shuffle`: ≤ 1e-6 (byte-stable element interleave)
- `sigmoid` / `tanh` vs torch: ≤ 1e-6
- `gru_cell_numpy` vs `nn.GRUCell`: ≤ 1e-5 (exact gate split verified)
- pack/unpack vs a real torch `state_dict`: exact (fp16 storage) + byte-stable + order-preserving
- end-to-end FiLM coord-MLP (fp16-stored weights → numpy decode) vs torch fp32 forward: ≤ 2e-2
  (the fp16-roundtrip parity bound the audit documents)

## Test pass count

**66 passed** (49 new bridge tests + 17 coin_plus_plus regression — the proof-of-pattern
inflate stays numpy-portable). Ruff clean on both new files.

## Migration path

### (a) the 4 class-shift blockers (boost_nerv, nirvana, atw_codec_v1, z5)
Per the audit's per-decoder port spec, each ports the SAME way coin_plus_plus did, now
against the canonical bridge instead of bespoke helpers:
1. `archive.py`: replace `brotli(pickle(torch_tensors))` with
   `brotli(pack_state_dict_numpy(sd, dtype="fp16"))`; add `parse_archive_numpy` that calls
   `unpack_state_dict_numpy`. Make `archive.py` itself torch-free (drop the module-scope
   `import torch`; wrap the torch-side parser behind a lazy import if still needed for
   training parity — OR move it to a sibling `archive_torch.py`).
2. `inflate.py`: rewrite numpy-only importing decode primitives FROM the bridge.
   - boost_nerv / nirvana / atw: `conv2d_numpy` + `pixel_shuffle_2x_nhwc` +
     `bilinear_resize_nhwc` + `linear` + `sin` + `sigmoid`; nirvana adds patch-stitch
     (numpy reshape/concatenate); boost_nerv adds the boosting-residual accumulation loop;
     atw adds `z = z_residual + film_modulate_numpy(...)`-style WZ-head reconstruction +
     `write_rgb_pair_to_raw_numpy` for the contest `.raw` output.
   - z5: add `gru_cell_numpy` for the autoregressive `z_t = predictor(z_{t-1}, ego) +
     residual[t]` rollout + `write_rgb_pair_to_raw_numpy`.
3. Verify via `assert_inflate_is_numpy_portable(inflate.py)` + a parity test (numpy decode
   vs torch forward) like coin_plus_plus's.
4. Emit the contest runtime via `write_numpy_portable_contest_runtime(...)`.

Recommended order (cleanest → hardest): atw_codec_v1 → nirvana → boost_nerv → z5.

### (b) the 18 PACT-NeRV inflates
The PACT-NeRV family uses `pact_nerv_full_main.write_contest_runtime`, which vendors a
torch `architecture.py`. Migration: (1) bump each `archive.py` to store weights via
`pack_state_dict_numpy`; (2) rewrite each `inflate.py` to decode via the bridge primitives
(the NeRV decode ops — conv/pixel-shuffle/bilinear/linear/sin/sigmoid — are all present);
(3) switch the runtime emitter from `write_contest_runtime` to
`write_numpy_portable_contest_runtime`; (4) `assert_inflate_is_numpy_portable` + parity test.

### coin_plus_plus consolidation (the proof-of-pattern, mid-migration)
coin_plus_plus already ships a torch-free numpy inflate (parity 9e-6) but its `archive.py`
still carries a module-scope `import torch` for the training-side `parse_archive`. When next
touched, migrate its bespoke `_serialize_numpy_state_dict` / `_deserialize_numpy_state_dict`
to `pack_state_dict_numpy` / `unpack_state_dict_numpy` (identical grammar; the bridge is the
canonical home) and make `archive.py` torch-free so the FULL tree (not just inflate.py)
emits portable via `write_numpy_portable_contest_runtime`.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map:** N/A — decode-side portability foundation; does not change
  per-tensor importance or score sensitivity.
- **hook #2 Pareto constraint:** N/A — no rate/distortion knob; numpy decode produces
  fp16-roundtrip-identical frames (parity 9e-6..2e-2), Pareto position unchanged.
- **hook #3 bit-allocator:** N/A — archive byte budget unchanged (the numpy blob is the same
  fp16 payload; the caller's grammar owns compression).
- **hook #4 cathedral autopilot dispatch:** ACTIVE — removing the torch-at-inflate
  shippability blocker is a PRECONDITION for any MLX-first / class-shift substrate to become
  a contest-PR candidate the autopilot can route to dispatch. The bridge is the canonical
  surface that unblocks the inflate-portability axis for the 4 class-shift blockers + the 18
  PACT-NeRV inflates.
- **hook #5 continual-learning posterior:** N/A — $0 local engineering; no empirical score
  anchor produced.
- **hook #6 probe-disambiguator:** ACTIVE — the numpy-vs-torch parity tests (≤1e-5 for fp32
  ops; exact for pack/unpack) ARE the canonical disambiguator proving a port preserves the
  trained decoder's behavior, distinguishing a faithful numpy port from a divergent
  re-implementation.

## Sister-safety (Catalog #314/#340)

Two concurrent sisters CONSUMING this contract: (a) MLX-harness-elevation
(`_shared/mlx_score_aware_full_main.py` + 4 MLX-first substrate dirs); (b)
substrate-inflate-migration (the 4 class-shift inflate paths). This work touched ONLY
`src/tac/substrates/_shared/numpy_portable_inflate.py` + its test + this memo. Did NOT edit
the sisters' substrate files, `_shared/mlx_score_aware_full_main.py`, or
`pr95_hnerv_numpy_reference.py` (added zero functions there — all new primitives live in the
bridge). The contract matches what the sisters build against: serialization (pack/unpack),
primitives (the 16 in DECODE_PRIMITIVES + raw writer), AST verifier, and the runtime emitter.

## CLAUDE.md compliance

HNeRV parity L4 (the EMITTED inflate.py is a thin numpy-only shim ≤ 200 LOC; deps numpy +
brotli + PIL) + Catalog #146 (3-arg inflate.sh + set -euo pipefail) + #205 (numpy is
device-free; no MPS fork) + #295 (PYTHONPATH self-containment; emitter vendors the bridge +
primitives) + #367 (CONTEST_RAW_BYTES_PER_VIDEO byte-count contract) + #369 (decode consumes
real trained weights). Deterministic + byte-stable; no /tmp; no scorer load. Production-
hardened: fully typed, fail-closed on malformed blobs with named errors, separation of
concerns (serialization / primitives / verification / emission are distinct surfaces).
