# Class-shift PyTorch substrates — INFLATE numpy-portability audit + coin_plus_plus port

**UTC:** 2026-05-27T16:12:12Z
**Lane:** `lane_class_shift_inflate_numpy_portability_audit_20260527`
**Operator directive:** 8th MLX-first + numpy-portable standing directive (2026-05-26) —
*"every substrate is MLX-first at training time AND numpy-portable at inflate time (no
torch/mlx dep in inflate)"*. FORBIDDEN at inflate time per the directive: `torch` /
`mlx` / `tensorflow` / `jax`.
**Scope:** the 5 just-landed class-shift PyTorch substrates (commits `77f102d3c` /
`d3383b4a5` / `30ea2077d`): `boost_nerv`, `coin_plus_plus`, `nirvana`,
`z5_predictive_coding_world_model`, `atw_codec_v1`.
**Cost:** $0 (local-CPU audit + numpy port + parity tests; no paid dispatch).

## Per-substrate verdict table

| Substrate | Pre-audit verdict | Decode framework dep | Archive weight storage | Catalog #369 (real weights) | Status |
|---|---|---|---|---|---|
| **coin_plus_plus** | BLOCKER-torch-at-decode | `import torch` (inflate + vendored archive/architecture) | `brotli(pickle(torch_tensors))` (needs torch to unpickle) | SATISFIED (real coord-MLP state_dict + modulations) | **FIXED → PORTABLE** |
| boost_nerv | BLOCKER-torch-at-decode | `import torch` (inflate + archive + architecture) | `brotli(pickle(torch_tensors))` | SATISFIED (real PixelShuffle decoder) | BLOCKER (port spec below) |
| nirvana | BLOCKER-torch-at-decode | `import torch` (inflate + archive + architecture) | `brotli(pickle(torch_tensors))` | SATISFIED (real patch-decode + stitch) | BLOCKER (port spec below) |
| z5_predictive_coding_world_model | BLOCKER-torch-at-decode | `import torch` (inflate + archive + architecture) | `brotli` + `frombuffer` + `torch.from_numpy` (mixed; no pickle) | SATISFIED (real enc/dec/predictor + autoregression) | BLOCKER (port spec below) |
| atw_codec_v1 | BLOCKER-torch-at-decode | `import torch` (inflate + archive + architecture) | `brotli(pickle(torch_tensors))` | SATISFIED (real enc/dec + WZ head + class-prior table) | BLOCKER (port spec below) |

### Verdict rationale (honest calibration)

All 5 substrates' shipped inflate runtimes `import torch` at module scope. The
`write_contest_runtime` canonical helper (`src/tac/substrates/_shared/pact_nerv_full_main.py:396`)
vendors `architecture.py` + `archive.py` + `inflate.py` into the submission tree —
and **all three vendored modules `import torch`**, so the shipped runtime tree carries
a torch dependency. The helper docstring even *claims* "numpy/PIL-portable (no MLX dep)"
but that claim was false for these 5 (the vendored modules carry torch).

The exact blocker is structural, not cosmetic: 3 of 5 store the decoder state_dict as
`brotli(pickle(torch_tensors))`. A torch-tensor pickle embeds
`torch._utils._rebuild_tensor_v2` GLOBAL refs and **cannot be unpickled without torch
installed** — so even a numpy-only inflate could not PARSE the archive. The directive's
bridge contract (step 2: `np.savez_compressed` numpy-loadable weights) is therefore a
structural requirement, not just an inflate-side rewrite.

**Calibration note (apples-to-apples):** the legacy canonical A1/PR101 lineage
(`submissions/a1/inflate.py`) ALSO `import torch` and is the accepted frontier — but
A1/PR101 PREDATE the 2026-05-26 directive. These 5 substrates landed 2026-05-27,
AFTER the directive, so they are bound by it. The lanes are all
`research_only=true / dispatch_enabled=false` L0/L1 scaffolds (no paid dispatch
eligible), so torch-at-inflate is not a live-dispatch hazard today — it is a
forward-engineering SHIPPABILITY blocker that must close before any of these can
be a contest-PR candidate.

## coin_plus_plus — FIXED (worked proof-of-pattern)

Chosen as the worked example because it is the cleanest decoder (pure FiLM-modulated
coord-MLP: `linear` + `sin` + `sigmoid`; no Conv2d / PixelShuffle / interpolate).

### What landed (4 files)

1. **`archive.py`** — replaced `brotli(pickle(torch_tensors))` with a **torch-free
   numpy-native state_dict serialization** (`{key: fp16 ndarray}` length-prefixed blob,
   brotli-compressed). Schema bumped `CPP1_SCHEMA_VERSION 1 → 2`. Added:
   - `_serialize_numpy_state_dict` / `_deserialize_numpy_state_dict` (torch-free)
   - `_serialize_state_dict` now converts torch→numpy then calls the numpy serializer
   - `_deserialize_state_dict` (torch-side) wraps numpy arrays in `torch.from_numpy`
   - `parse_archive_numpy(blob) -> CoinplusplusArchiveNumpy` — the torch-free parse the
     shipped inflate calls. Zero torch code-refs (AST-verified).
2. **`inflate.py`** — rewritten numpy-only (108 non-blank LOC; ≤200 L4 budget). NO torch
   import. Pure-numpy FiLM coord-MLP forward: `_build_coord_grid` (linspace/meshgrid) +
   `_linear` (x @ W.T + b, fp32 accum) + `np.sin` + sigmoid. Consumes the REAL trained
   `base_mlp_state_dict` + per-pair `modulations` (Catalog #369 satisfied; no synthetic
   frame base). No `select_inflate_device` device-fork because numpy is device-free
   (MPS structurally impossible per Catalog #205; CPU/CUDA agnostic by construction).
   Runtime tree: numpy + brotli + PIL.
3. **`__init__.py`** — export `parse_archive_numpy` + `CoinplusplusArchiveNumpy`.
4. **`tests/test_coin_plus_plus_numpy_inflate.py`** — 6 new tests.

### Empirical parity (the proof)

- numpy inflate forward vs torch `model.forward`: **max abs diff 0.000009** (fp16-roundtrip
  + matmul-order noise; near byte-identical)
- torch `parse_archive` vs numpy `parse_archive_numpy` modulations: **0.0** (exact)
- torch vs numpy weights: **0.0** (exact)
- AST: inflate.py has 0 torch/mlx imports; numpy parse path has 0 torch code-refs

### Test result

**17/17 pass** (11 existing — including the CPP1 encode/inflate roundtrip, which now
exercises the v2 numpy-native serialization — + 6 new numpy-portability). No external
caller of `coin_plus_plus.inflate_one_video` exists (signature change is safe). Trainer
pack/parse roundtrip + `load_state_dict` verified end-to-end. Catalog #295 self-containment
clean. Catalog #369 introduced 0 new violations (the 1 live violation is the pre-existing
`cascade_c_prime` anchor, out of scope).

## Per-substrate port spec for the remaining 4

All 4 follow the SAME bridge pattern coin_plus_plus established. The decode ops are all
present in the canonical numpy reference `src/tac/local_acceleration/pr95_hnerv_numpy_reference.py`
(`linear`, `conv2d_nhwc`, `bilinear_upsample_2x_nhwc`, `sigmoid`, `sin`) — PixelShuffle is
a numpy reshape. Each is real substrate-engineering (the convolutional decoders are heavier
than the coord-MLP), so they are queued as follow-ups, NOT mechanical find-replace.

**Shared bridge step (all 4):** change `archive.py` state_dict storage from
`brotli(pickle(torch_tensors))` to the numpy-native `{key: fp16 ndarray}` blob
(`_serialize_numpy_state_dict` / `_deserialize_numpy_state_dict` are reusable — promote
to `_shared/numpy_state_dict_blob.py` when the 2nd substrate ports), bump schema version,
add `parse_archive_numpy`. z5 already stores latents/residuals/ego-motion via `frombuffer`
(torch-free) but its three sub-module state_dicts still need the numpy-native bridge.

| Substrate | Decoder ops to port | Numpy reference primitives needed | Extra work |
|---|---|---|---|
| `boost_nerv` | 7×Conv2d, 4×PixelShuffle, 3×Linear, sin, sigmoid, interpolate + **boosting-chain residual loop** | `conv2d_nhwc`, `bilinear_upsample_2x_nhwc`, numpy `pixel_shuffle` reshape, `linear`, `sin`, `sigmoid` | per-round boosting gain clamp + residual accumulation in numpy |
| `nirvana` | 5×Conv2d, 4×PixelShuffle, 2×Linear, sin, sigmoid, interpolate + **patch-decode + stitch** | same as boost_nerv | per-patch decode + spatial stitch (reshape/concatenate) |
| `atw_codec_v1` | 5×Linear, 4×PixelShuffle, 3×Conv2d, sin, sigmoid, interpolate + **WZ side-info head + class-prior reconstruction** | same conv set + `linear` | `z = z_residual + wz_head(class_prior_table[pair])` in numpy; raw-output `.raw` write (already torch-free helper `write_rgb_pair_to_raw` — needs numpy variant) |
| `z5_predictive_coding_world_model` | 7×Linear, 3×PixelShuffle, 3×Conv2d, sigmoid, interpolate + **GRU recurrent predictor + autoregressive rollout** | same conv set + numpy GRU cell (`sigmoid`/`tanh` gates + `linear`) | MOST complex: autoregressive `z_t = predictor(z_{t-1}, ego) + residual[t]` loop in numpy; numpy GRU cell port |

**Recommended order (cleanest → hardest):** `atw_codec_v1` (linear-heavy + WZ head) →
`nirvana` (patch stitch) → `boost_nerv` (boosting loop) → `z5` (GRU autoregression).
Each ports independently per UNIQUE-AND-COMPLETE-PER-METHOD (the shared numpy-state_dict
blob helper serves; the per-decoder forward is substrate-unique).

## Raw-output vs PNG note

`coin_plus_plus`/`boost_nerv`/`nirvana` write per-frame PNGs (eval-resolution). `atw_codec_v1`
+ `z5` write contest `.raw` (1164×874×1200×3 = 3,662,409,600 bytes) via the torch-based
`write_rgb_pair_to_raw` helper in `_shared/inflate_runtime.py` — that helper itself
`import torch` and must get a numpy variant (`write_rgb_pair_to_raw_numpy`) before atw_codec_v1/z5
can ship numpy-portable. Catalog #367 (inflate emits expected frame count or fail-closed)
applies to the raw-output pair — the numpy variant must preserve the byte-count assertion.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map:** N/A — this is a portability/shippability hardening of the
  decode runtime; it does not change per-tensor importance or score sensitivity.
- **hook #2 Pareto constraint:** N/A — no rate/distortion knob changes; the numpy archive
  produces byte-identical-to-fp16-roundtrip frames (parity 9e-6), so the Pareto position
  is unchanged.
- **hook #3 bit-allocator:** N/A — archive byte budget is unchanged (numpy-native blob is
  the same fp16 payload, brotli-compressed; the wire grammar's 6 sections are preserved).
- **hook #4 cathedral autopilot dispatch:** ACTIVE — removing the torch-at-inflate
  shippability blocker is a precondition for any of these 5 substrates to become a
  contest-PR candidate the autopilot can route to dispatch. coin_plus_plus is now
  dispatch-shippable on the inflate-portability axis (other promotion gates still apply).
- **hook #5 continual-learning posterior:** N/A — no empirical score anchor produced
  (this is $0 local engineering; no contest-CUDA/CPU eval).
- **hook #6 probe-disambiguator:** ACTIVE — the numpy-vs-torch parity check (max abs diff
  9e-6) IS the canonical disambiguator proving the port preserves the trained decoder's
  behavior, distinguishing a faithful numpy port from a divergent re-implementation.

## Sister-safety (Catalog #314/#340)

Two sisters declared at dispatch: (a) PACT-NeRV REVIEW (read-only, writes only a memo);
(b) MLX-harness subagent owns `_shared/mlx_score_aware_full_main.py` + 6 MLX-first
substrate dirs (dreamer_v3_rssm, z8, mdl_ibps_j, atw_v2_cooperative_receiver_v2,
coin_pp_mlx, faiss_ivf_pq_residual). This work touched ONLY the 5 named class-shift
PyTorch substrates' files — and in practice only `coin_plus_plus/{archive,inflate,__init__}.py`
+ its test + this memo. Did NOT touch `*pact_nerv*`, master-gradient files, the 6
MLX-first substrate dirs, or `_shared/inflate_runtime.py` (the raw-output helper numpy
variant is left as a follow-up for the atw_codec_v1/z5 ports). No overlap with sister
checkpoints.
