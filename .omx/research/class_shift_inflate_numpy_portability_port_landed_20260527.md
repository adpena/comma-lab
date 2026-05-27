# Class-shift PyTorch substrates — INFLATE numpy-portability PORT landed (4 substrates)

**UTC:** 2026-05-27
**Lane context:** follow-up to `class_shift_inflate_numpy_portability_audit_20260527T161212Z.md`
(the audit + coin_plus_plus proof-of-pattern). This memo records the 4 remaining
class-shift PyTorch substrates ported to numpy-portability via the LANDED canonical
bridge (`980808776` / `tac.substrates._shared.numpy_portable_inflate`).
**Cost:** $0 (local-CPU port + parity tests; no paid dispatch).
**Operator directive:** 8th MLX-first + numpy-portable standing directive (2026-05-26).

## Per-substrate result table

| Substrate | Status | Parse parity (weights) | Decode parity (numpy vs torch) | inflate.py portable | Schema | Commit |
|---|---|---|---|---|---|---|
| **atw_codec_v1** | PORTED | exact 0.0 | rgb max-diff **6.03e-05** | YES | ATW1 v1 (blob already torch-free; +numpy parse) | `29a91cddb` |
| **nirvana** | PORTED | exact 0.0 | patch-decode+stitch max-diff **5.96e-08** | YES | NRV1 v1→**v2** (pack_state_dict_numpy replaces pickle) | `4ce4d9faa` |
| **boost_nerv** | PORTED | exact 0.0 | post-boosting max-diff **5.36e-07** | YES | BSV1 v1→**v2** (pack_state_dict_numpy replaces pickle) | `90f294397` |
| **z5_predictive_coding_world_model** | PORTED | exact 0.0 | post-rollout decode rgb max-diff **3.16e-05** | YES | Z5PCWM1 v1 (blob already torch-free; +numpy parse) | `8e9fcd17e` |

All four decode parities are well under the 1e-4 target. All four `inflate.py`
pass `assert_inflate_is_numpy_portable` (zero torch/mlx import). Parse roundtrip is
byte-exact (0.0) because the numpy deserializer reads the SAME fp16 bytes the torch
serializer wrote.

## What was done per substrate

Each substrate followed the coin_plus_plus bridge pattern:

1. **archive.py** — added a torch-free numpy state_dict deserializer + a
   `parse_archive_numpy(blob) -> <Substrate>ArchiveNumpy` returning `np.ndarray`
   weights/latents. The torch-side `parse_archive` is preserved for training.
   - **atw_codec_v1 / z5**: the existing blob was already a hand-rolled
     length-prefixed fp16 serializer (no pickle), so only a numpy reader +
     `parse_archive_numpy` were added; **schema version unchanged** (the bytes
     are already torch-free).
   - **nirvana / boost_nerv**: the existing blob was `brotli(pickle(torch))` —
     a torch-tensor pickle CANNOT be unpickled without torch. Migrated the
     serializer to the canonical `pack_state_dict_numpy`/`unpack_state_dict_numpy`
     bridge (fp16 `{key: ndarray}`, no pickle); **bumped schema v1→v2**. The
     torch-side `_deserialize_state_dict` now wraps the numpy arrays in
     `torch.from_numpy` so training-side parse parity is preserved.
2. **inflate.py** — rewritten numpy-only (0 torch/mlx imports), consuming the
   REAL trained weights (Catalog #369; no synthetic frame base) via the bridge
   decode primitives:
   - **atw_codec_v1**: NeRV decoder (initial_proj + Conv/PixelShuffle(2)/ReLU
     blocks + final conv + bilinear resize + sigmoid) + WZ side-info head
     reconstruction `z = z_residual + fc2(relu(fc1(class_prior)))`. Contest
     `.raw` output via `write_rgb_pair_to_raw_numpy` (Catalog #367 byte assert).
   - **nirvana**: per-patch decode (combined_embed + depthwise+pointwise+sin+
     PixelShuffle blocks + 1×1 RGB heads + sigmoid) + spatial stitch. PNG output.
   - **boost_nerv**: base NeRV decode + iterative boosting residual chain
     (`residual = clamp(tanh(conv2(relu(conv1([rgb; z_grid])))), -gain, gain)`;
     `rgb = clamp(rgb + residual, 0, 1)` per round). PNG output.
   - **z5**: autoregressive MLP predictor rollout (`z_t = tanh(z_to_hidden(z) +
     ego_to_hidden(ego))` -> (n-1)×(Linear+GELU) -> hidden_to_z; `z_t += r_t`)
     + NeRV decoder. Contest `.raw` output (Catalog #367 byte assert).
3. **tests** — new `test_*_numpy_inflate.py` per substrate (5-6 tests each):
   no-torch-import AST check, torch-free-parse AST check, numpy-vs-torch parse
   exactness, numpy-vs-torch decode parity, operational-consumption no-op
   detector (Catalog #220), PNG/raw write smoke. All existing tests pass.

## Bridge primitives used (no reimplementation)

`pack_state_dict_numpy` / `unpack_state_dict_numpy` (nirvana, boost_nerv) +
`conv2d_numpy` / `bilinear_resize_nhwc` / `pixel_shuffle_2x_nhwc` / `linear` /
`sigmoid` / `relu` / `tanh` / `gelu` / `to_float32` / `write_rgb_pair_to_raw_numpy`
/ `assert_inflate_is_numpy_portable` — all from
`tac.substrates._shared.numpy_portable_inflate`.

### Substrate-unique primitives added in-inflate (NOT in the bridge)

- **Depthwise 3×3 conv (groups=C)** — nirvana + boost_nerv `_DepthSepConv`. The
  bridge `conv2d_nhwc` is cross-channel only; a per-channel depthwise helper
  (`_depthwise_conv3x3_nhwc`) was implemented inline per substrate (~17 LOC).
  Candidate for promotion to the bridge if a 3rd depthwise consumer appears.
- **Conv weight NCHW→NHWC transpose** — every torch Conv2d weight is
  `(C_out, C_in, kH, kW)`; the bridge expects `(C_out, kH, kW, C_in)`, so each
  inflate transposes `(0,2,3,1)` at decode (faithful copy of torch keys preserved).

## Honest calibration notes

- **Parity surface is the DECODER OUTPUT** ([0,1] RGB pre-camera-resize), matching
  the audit's coin_plus_plus 9e-6 measurement. The final contest `.raw` lowering
  resizes EVAL_HW (384×512) → CAMERA_HW (874×1164); the torch helper used bicubic
  and the bridge numpy helper uses bilinear, so the post-resize raw bytes will
  differ by the bicubic-vs-bilinear interpolation gap. This is a SEPARATE lowering
  step, not a decoder-fidelity gap. If exact raw-byte parity vs a torch baseline is
  ever required, the bridge's `write_rgb_pair_to_raw_numpy` would need a bicubic
  numpy resize variant — left as a follow-up (not in scope; the decoder port is faithful).
- **z5 latent rollout max-diff (3.4e-3) > decode rgb max-diff (3.16e-05)**: the
  fp16 weight roundtrip + GELU tanh-approx accumulate over the 3-step autoregressive
  rollout, but the decoder + sigmoid attenuate it to 3.16e-05 at the RGB output. Both
  are within the audit's tolerance band. The "GRU" in the audit was imprecise — the
  z5 predictor is an MLP (tanh fusion + GELU layers), ported faithfully.
- **z5 backward compat**: kept the `device` kwarg (ignored; numpy is device-free per
  Catalog #205) + `_read_single_member_archive_bytes` so the 47 existing z5 tests
  pass unchanged.

## Catalog / discipline compliance

- **Catalog #205** — no `select_inflate_device` device-fork; numpy is device-free
  (MPS structurally impossible).
- **Catalog #295** — archive parser + bridge vendored into submission tree
  (PYTHONPATH self-contained via `write_numpy_portable_contest_runtime`).
- **Catalog #369** — every inflate consumes the REAL trained weights, not a
  synthetic frame base (verified by numpy-vs-torch decode parity).
- **Catalog #367** — atw + z5 raw-output paths assert the contest byte count for
  num_pairs=600.
- **Catalog #220** — operational-consumption no-op detectors prove the
  distinguishing feature (WZ head / patch embeddings / boosting heads / predictor)
  actually changes the decode.
- **Catalog #157 / #174 / #206** — all 4 commits via canonical serializer with
  POST-EDIT `--expected-content-sha256`; review-gate satisfied (2 clean passes per
  entity).
- **HNeRV parity L4** — all 4 inflate.py numpy/PIL/brotli-only, no torch/mlx.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map:** N/A — portability hardening; no per-tensor importance change.
- **hook #2 Pareto constraint:** N/A — numpy archive is byte-equivalent fp16 payload; Pareto position unchanged.
- **hook #3 bit-allocator:** N/A — archive byte budget unchanged (nirvana/boost_nerv v2 blob is the same fp16 payload).
- **hook #4 cathedral autopilot dispatch:** ACTIVE — removing torch-at-inflate is a precondition for any of these 4 to be a contest-PR candidate; all 4 are now inflate-portability-shippable (other promotion gates still apply; lanes remain research_only L0/L1).
- **hook #5 continual-learning posterior:** N/A — $0 local engineering; no contest-CUDA/CPU eval anchor.
- **hook #6 probe-disambiguator:** ACTIVE — the numpy-vs-torch parity (≤6e-05 decode) IS the disambiguator proving a faithful port vs a divergent re-implementation.

## Sister-safety (Catalog #314/#340)

Touched ONLY the 4 named substrates' `archive.py` + `inflate.py` + `__init__.py`
+ a new test per substrate. Did NOT touch `_shared/numpy_portable_inflate.py`,
`pr95_hnerv*`, `*pact_nerv*`, the MLX-first dirs
(atw_v2_cooperative_receiver_v2 / mdl_ibps_j / faiss_ivf_pq_residual / coin_pp),
or `coin_plus_plus` (done). A codex review companion landed 3 hardening commits
(`09d5de0db` / `68182efc3` / `f4fb63d60`) adding contest-safe `_read_single_member`
+ `_raw_output_path` helpers on top; those are complementary and preserve portability.

## Status

4/4 PORTED. 99 substrate tests pass (atw 5 + nirvana 17 + boost_nerv ~20 + z5 52).
All 4 class-shift PyTorch substrates are now numpy-portable at inflate time, closing
the torch-at-inflate shippability blocker the audit identified.
