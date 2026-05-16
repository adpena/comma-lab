# SPDX-License-Identifier: MIT
"""tac.substrates.nscs06_v8_path_b_wavelet — wavelet-residual L1 SCAFFOLD.

Per ``.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md``
(commit 963549469): v7 Path A landed 58.89 [diagnostic_cpu] inside the predicted
[40, 65] band. v8 Path B is the next-cargo-cult-target step: UNWIND cargo-cult
#3 (spatial-independent-CDF via DB4 wavelet decorrelation) + cargo-cult #6
(band-prediction-without-distortion-model via Catalog #296 structural Dykstra-
feasibility check). Predicted band [15, 25] FEASIBLE per Catalog #296 polytope
[0.40, 28.76] at seg_budget=0.06 / pose_budget=50 / archive=600 KB.

UNIQUE-AND-COMPLETE-PER-METHOD architecture: ONE coherent ~700-900 LOC packet
that binds DB4 depth-2 separable 2D DWT (per Mallat 1989 + Daubechies 1992
Ch. 6) + per-subband Mallat hierarchical arithmetic coding + Wyner-Ziv
side-information coding of frame_1 against frame_0 (Wyner-Ziv 1976) +
WLV2 monolithic archive grammar + ≤310 LOC numpy+Pillow+pywt inflate. NO
training loop, NO neural primitives, NO torch at inflate.

Cargo-cults addressed by v8 vs v7:

* **#3 UNWOUND** — DB4 depth-2 separable 2D DWT decorrelates 5-10× per
  Mallat 1989 + 40 years subsequent natural-image evidence. v7's per-pixel
  per-class CDF is replaced by per-subband per-class Laplacian-prior CDFs.
* **#6 UNWOUND** — predicted band derived from stacked first-principles
  bounds (MacKay MDL + Wyner-Ziv 1976 + Tao MSE-energy) AND structurally
  validated via Catalog #296 Dykstra-feasibility check (polytope [0.40,
  28.76] strictly contains predicted [15, 25]).
* **#5 PRESERVED-WAIVED** — NO-neural-at-medal-band; v8 targets [15, 25],
  not medal-band. Medal-band requires Path C hybrid-neural.

Canonical-vs-unique decision per layer (per CLAUDE.md
``UNIQUE-AND-COMPLETE-PER-METHOD operating mode`` + design memo Section 12):

| Layer                                  | Decision      | Rationale |
|----------------------------------------|---------------|-----------|
| DB4 wavelet DWT/IDWT                   | UNIQUE        | substrate-distinguishing core (pywt-backed) |
| Per-subband Mallat hierarchical arith  | UNIQUE        | substrate-distinguishing entropy mechanism |
| Wyner-Ziv temporal residual coding     | UNIQUE        | replaces v7's pose-warp approximation |
| WLV2 archive grammar                   | UNIQUE        | new wire format; distinct from CH06 v2 |
| ArithmeticCoder primitive              | ADOPT v7      | reuse the byte-stable Witten-Neal-Cleary impl |
| ClassConditionalCDF primitive          | ADOPT v7      | reuse the per-class CDF dataclass + builder |
| Inflate runtime                        | UNIQUE        | new DWT+Wyner-Ziv-merge logic; numpy+Pillow+pywt |
| Auth-eval routing (Catalog #226)       | ADOPT canonical | universal bug-class protection |
| Inflate device selection (Catalog #205)| ADOPT canonical | universal MPS-fallback-trap protection |
| Trainer skeleton helpers               | ADOPT canonical | pin_seeds / decode_real_pairs / etc. |
| SubstrateContract decoration           | ADOPT canonical | required by META layer per Catalog #241/#242 |
| Modal/CUDA env block                   | ADOPT canonical | Catalog #244 auto-emit |
| Lane registry pre-registration        | ADOPT canonical | Catalog #126 |
| Score-aware loss helper                | FORK (N/A)    | no training loop ⇒ no eval_roundtrip ⇒ no canonical loss |
| EMA shadow weights                     | FORK (N/A)    | no learnable weights |

The 14 layers above realize the design memo Section 12 table verbatim.

Catalog #124 archive-grammar 8 fields:

    archive_grammar:            WLV2 monolithic single-file 0.bin (Section 9 of memo)
    parser_section_manifest:    parse_archive() -> (header + 14 body blobs)
    inflate_runtime_loc_budget: <=310 LOC numpy+Pillow+pywt (substrate_engineering exception per L7)
    runtime_dep_closure:        numpy, Pillow, pywavelets
    export_format:              custom (WLV2: monolithic per-subband + Wyner-Ziv residual; no fp16/brotli/torch)
    score_aware_loss:           custom (NO TRAINING; closed-form per-(subband, class) Laplacian-prior allocation)
    bolt_on_loc_budget:         ~2200 LOC (substrate_engineering exception per L7)
    no_op_detector_planned:     Catalog #139 byte-mutation smoke + Catalog #272 distinguishing-feature

Observability surface (per CLAUDE.md "Observability surface" + Catalog #305):

1. **Per-layer inspection.** Each compress stage (DWT decomposition / Wyner-Ziv
   residual / per-subband arith encoding / WLV2 packing) emits per-subband
   coefficient counts + arith-stream byte lengths into the compress-stage
   stage_log JSON consumed by ``provenance.json`` per the trainer.
2. **Per-signal decomposition.** Archive blob lengths recorded per stream
   (GRAY_F0 / GRAY_F1RES / CB_F0 / CB_F1RES / CR_F0 / CR_F1RES / CLS) so the
   rate-axis decomposition is byte-addressable per Catalog #166 + #245.
3. **Run-to-run diff.** Byte-stable: sorted-keys JSON meta + deterministic
   DB4 filter coefficients + fixed-precision arith coder = identical 0.bin
   sha256 across runs at same (seed, commit_sha, upstream_snapshot_sha256).
4. **Post-hoc query interface.** ``provenance.json`` + ``contest_auth_eval_<axis>.json``
   + WLV2 header offsets enable structured per-stream inspection without
   re-running compress.
5. **Cite-chain.** Catalog #245 modal_call_id_ledger + Catalog #166 HEAD
   parity ledger preserved via canonical Modal dispatch path.
6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 +
   #272 + #105 distinguishing-feature contract: mutation at any wavelet
   stream offset MUST change output frames; smoke validates this.

Catalog #272 distinguishing-feature contract:

    distinguishing_feature_name:      wavelet_subband_arith_streams_with_wyner_ziv_temporal_residual
    distinguishing_bytes_path:        WLV2 body segments GRAY_F0_STREAMS_BLOB through CLS_STREAMS_BLOB
    inflate_consumer_function:        inflate.py::inflate_one_video lines decode_per_pair_subband_streams + idwt2_db4_depth2 + add_subbands
    byte_mutation_smoke_passes:       TBD (post-smoke verification per Catalog #139)

CLAUDE.md compliance:
- No silent device defaults (compress side runs scorer on cuda via canonical helper)
- No scorer loading at inflate (inflate.py has ZERO torch / scorer imports)
- No /tmp paths (compress writes under args.output_dir)
- No KILL verdicts in the scaffold (DEFER-pending-empirical-anchor only)
- Apples-to-apples axis labels on every score claim
- Forbidden empirical-claim-without-evidence-tag honored: predicted band tagged
  [prediction; first-principles + Dykstra-feasibility-validated; MEDIUM VARIANCE]
"""

from .archive import (
    WLV2_HEADER_FMT,
    WLV2_HEADER_SIZE,
    WLV2_MAGIC,
    WLV2_SCHEMA_VERSION,
    WaveletResidualArchive,
    pack_archive,
    parse_archive,
)
from .inflate import inflate_one_video, main_cli
from .wavelet_codec import (
    DB4_DECOMP_HI,
    DB4_DECOMP_LO,
    DB4_RECON_HI,
    DB4_RECON_LO,
    DWT_LEVEL,
    NUM_SUBBANDS,
    PER_SUBBAND_QUANT_STEPS,
    SUBBAND_LABELS,
    build_per_subband_laplacian_priors,
    dequantize_subband,
    dwt2_db4_depth2,
    encode_subband_arith,
    idwt2_db4_depth2,
    laplacian_cdf_uint16,
    quantize_subband,
)
from .wyner_ziv_temporal import (
    compute_wyner_ziv_residual,
    reconstruct_frame1_from_frame0_and_residual,
)

__all__ = [
    "DB4_DECOMP_HI",
    "DB4_DECOMP_LO",
    "DB4_RECON_HI",
    "DB4_RECON_LO",
    "DWT_LEVEL",
    "NUM_SUBBANDS",
    "PER_SUBBAND_QUANT_STEPS",
    "SUBBAND_LABELS",
    "WLV2_HEADER_FMT",
    "WLV2_HEADER_SIZE",
    "WLV2_MAGIC",
    "WLV2_SCHEMA_VERSION",
    "WaveletResidualArchive",
    "build_per_subband_laplacian_priors",
    "compute_wyner_ziv_residual",
    "dequantize_subband",
    "dwt2_db4_depth2",
    "encode_subband_arith",
    "idwt2_db4_depth2",
    "inflate_one_video",
    "laplacian_cdf_uint16",
    "main_cli",
    "pack_archive",
    "parse_archive",
    "quantize_subband",
    "reconstruct_frame1_from_frame0_and_residual",
]
