<!-- SPDX-License-Identifier: MIT -->
# Z5 Rao-Ballard hierarchical predictive coding + Hinton-distilled scorer surrogate substrate

**Status:** L1 SCAFFOLD with 2/3 canonical empirical anchors landed at MLX-LOCAL
research-signal grade. PROCEED_WITH_REVISIONS per the per-substrate symposium memo
`.omx/research/council_t2_z5_rao_ballard_hinton_distilled_per_substrate_symposium_20260528.md`
pending anchor 3/3 identity-predictor disambiguator probe per Catalog #308.

## Distinguishing primitive

Rao + Ballard 1999 hierarchical predictive coding with EXPLICIT 2-level latent
split and a parameterized predictor that maps higher-level latent + ego-motion
focus-of-expansion (FoE) prior to the lower-level latent reconstruction target.
Per pair:

```
z_low_t      = self.low_latents[t]            (per-pair learnable)
z_high_t     = self.high_latents[t]           (per-pair learnable)
ego_motion_t = self.ego_vecs[t]               (per-pair learnable)
z_low_pred   = predictor(z_high_t, ego_motion_t)
residual_t   = z_low_t - z_low_pred           (training: minimized)
rgb_0, rgb_1 = decoder(z_low_t)               (FiLM-PixelShuffle decoder)
```

Distinct from sister cooperative-receiver-paradigm-class substrates:

- `tac.substrates.time_traveler_l5_z6` (Z6-v2): single-level FiLM-ego-motion
  conditioning; no explicit level-1 predictor
- `tac.substrates.time_traveler_l5_z7_mamba2` (Z7-Mamba-2): selective state-space
  recurrence; no explicit hierarchical level boundary
- THIS substrate (Z5): EXPLICIT 2-level Rao-Ballard hierarchy with separate
  `z_low` + `z_high` per-pair latents + parameterized predictor mapping
  `(z_high_t, ego_motion_t) -> z_low_pred`

## Archive grammar (Z5RB1 monolithic single-file `0.bin`)

Per CLAUDE.md HNeRV parity discipline L3 (monolithic single-file) + L4 (≤200 LOC
substrate-engineering inflate waiver). Header total 37 bytes + 6 deterministic
blob sections:

```
MAGIC(4)             b"Z5RB"
VERSION(1)           u8       schema version (currently 1)
LOW_LATENT_DIM(2)    u16      cfg.low_latent_dim (e.g. 24)
HIGH_LATENT_DIM(2)   u16      cfg.high_latent_dim (e.g. 16)
EGO_DIM(2)           u16      cfg.ego_dim (e.g. 6)
NUM_PAIRS(2)         u16      cfg.num_pairs (e.g. 600)
DECODER_BLOB_LEN(4)  u32      brotli(q=9) decoder state_dict fp16 bytes len
PREDICTOR_BLOB_LEN(4) u32     brotli(q=9) predictor state_dict fp16 bytes len
LOW_LAT_BLOB_LEN(4)  u32      brotli(q=9) low_latents fp16 bytes len
HIGH_LAT_BLOB_LEN(4) u32      brotli(q=9) high_latents fp16 bytes len
EGO_BLOB_LEN(4)      u32      brotli(q=9) ego_vecs fp16 bytes len
META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes len
[DECODER_BLOB][PREDICTOR_BLOB][LOW_LAT_BLOB][HIGH_LAT_BLOB][EGO_BLOB][META_BLOB]
```

Inflate runtime: see `tac.substrates.time_traveler_l5_z5.inflate.inflate_one_video`
(181 LOC ≤ 200 substrate-engineering waiver budget per HNeRV parity L4).

## Contest 3-arg `inflate.sh` contract per Catalog #146

```
./inflate.sh <archive_dir> <output_dir> <file_list>
```

Set the `PYTHON` environment variable to point at the desired python
interpreter (default: `python3`).

## Self-containment per Catalog #295

This checked-in `inflate.py` is a FAIL-CLOSED TEMPLATE following the NSCS01
canonical pattern: it raises `RuntimeError` BEFORE any `sys.path.insert`
unless the sibling `submission_dir/src/` exists (vendored by the trainer's
`_write_runtime` per CLAUDE.md HNeRV parity L4 + L9 + Catalog #295).

The trainer-emitted `submission_dir/inflate.py` vendors all of:

- `src/tac/substrates/time_traveler_l5_z5/{architecture,archive,inflate}.py`
- `src/tac/substrates/_shared/inflate_runtime.py` (canonical `select_inflate_device`
  per Catalog #205 + `write_rgb_pair_to_raw` per Catalog #146)

into `submission_dir/src/tac/...` so the runtime is fully self-contained per
HNeRV parity discipline L9 (clean-env dependency closure tested BEFORE dispatch).

## Runtime dependencies

- `torch` (decoder + predictor inference; per Catalog #205 canonical device select)
- `brotli` (Z5RB1 blob decompression; per HNeRV parity L9 ≤ 2 ext deps)
- `numpy` (stdlib-adjacent; serialization helpers per Catalog #295 numpy-portable
  pattern)

## Canonical citations

- **literature**: Rao + Ballard 1999 *"Predictive coding in the visual cortex"*
  (Nature Neuroscience 2(1):79-87); Atick + Redlich 1990 *"Towards a Theory of
  Early Visual Processing"* (Neural Computation 2:308-320); Hinton + Vinyals +
  Dean 2014 *"Distilling the Knowledge in a Neural Network"* (NIPS Workshop);
  Friston 2010 *"The free-energy principle: a unified brain theory?"* (Nature
  Reviews Neuroscience 11:127-138)
- **canonical equation**: `z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`
  per `tac.canonical_equations` registry
- **per-substrate symposium**:
  `.omx/research/council_t2_z5_rao_ballard_hinton_distilled_per_substrate_symposium_20260528.md`
- **canonical anti-patterns acknowledged**: `distinguishing_primitive_indistinguishability_at_underconverged_config_v1`
  + `cross_paradigm_stacking_additive_compounding_without_dykstra_feasibility_v1`
  + `mps_drift_architecture_class_dependent_v1` per `tac.canonical_anti_patterns`

## Reproducibility (per Wave N+22 + N+28 anchors)

| Anchor | Run | Archive SHA256 | Bytes | Pose-axis reduction | Wall-clock |
|---|---|---|---|---|---|
| 1/3 | 50ep/600pair pure reconstruction MLX-LOCAL | `ceb614f6c0d2784fb756ab9c127bab8d5f009ac882726cc27043a6a6055f74ca` | 214,630 | n/a (no Hinton-scorer bound) | 1.022 s on M5 Max |
| 2/3 | 600ep/600pair Hinton-distilled MLX-LOCAL | `3000ca91126a82aacbb3e54bb5eb791f6feb7d1a5f5ec358604b32d815f823fe` | 216,154 | 3.22x (107.53 → 33.38) | 25.62 s on M5 Max |
| 3/3 | identity-predictor disambiguator (PENDING per Catalog #308) | TBD | TBD | TBD | ~14-25 min MLX-LOCAL |

Both landed anchors are `[macOS-MLX research-signal]` per CLAUDE.md "MLX
portable-local-substrate authority" 8th non-negotiable + Catalog #192 / #317 /
#341 non-promotable. Paired CPU + CUDA T4 RATIFICATION per Catalog #246 is the
reactivation criterion for contest-axis score claim.
