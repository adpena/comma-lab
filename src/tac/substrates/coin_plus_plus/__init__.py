# SPDX-License-Identifier: MIT
"""tac.substrates.coin_plus_plus — COIN++ (substrate L0 SKETCH).

Modulation-based parameter-efficient implicit neural representation.
Operator 5-tier fit-ranking verdict **MODERATE FIT ⭐⭐⭐**: COIN++ ships a
SHARED base network (large, trained once) + a tiny PER-PAIR modulation
vector. The base network is the same across all pairs; the per-pair
modulation is what gets shipped in the archive's latent slot.

Literature anchor: Dupont et al. ICML 2022 "COIN++: Neural Compression
across Modalities" (arXiv:2201.12904 — canonical literature reference
per BUILD task #1090). Canonical OSS repo: github.com/EmilienDupont/coinpp
(literature anchor; not vendored).

Hypothesis (per operator's per-variant fit verdict): the COIN++ paradigm
of "small per-pair modulation + shared base network" is a structurally
different rate-tradeoff than per-pair latent + shared decoder. The base
network is amortized over all 600 pairs; the per-pair modulation can be
extremely small (e.g., 64 floats per pair, ~76 KB total for 600 pairs).
The MODERATE FIT score reflects that this paradigm has been tested at
small scale and works, but driving video at 384x512 may require larger
modulations than COIN++'s native 32-128 dim modulations to capture the
spatial complexity.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair modulation vector m in R^MOD_DIM (default 64)
       |
       v
    Shared base coordinate-MLP F_phi:
        Input: (x, y, t) - normalized pixel coord + frame_index (0 or 1)
        Output: rgb in [0, 1]^3
        Phi parameters: SHARED across all pairs (frozen at inflate-time after
        the base network is trained; modulated by m via FiLM-style scale+shift
        on hidden layers)
       |
       v
    For each pixel (x, y) in [0, H) x [0, W):
        rgb_t(x, y) = F_phi_mod_m(x, y, t)
       |
       v
    Stack into rgb_0, rgb_1: (B, 3, H, W)

The L0 SCAFFOLD uses a small coord-MLP base + MOD_DIM=64 modulations
shipped via int8 + brotli. The forward pass is INHERENTLY O(H*W) coordinate
sampling — this is a known cost vs spatial-grid PixelShuffle decoders.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

The L0 scaffold ships:
- Substrate architecture (shared coord-MLP + per-pair modulation)
- Archive grammar (CPP1 magic + MOD_DIM in header)
- Inflate runtime (≤200 LOC for the coord-MLP forward path)
- Score-aware loss helper routing
- Test coverage for the canonical archive grammar

Council design memo:
    `.omx/research/coin_plus_plus_l0_scaffold_design_20260520T184500Z.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 23-byte header) |
| L4 inflate <= 200 LOC, <= 2 deps | PASS (target ~140 LOC for coord-MLP forward) |
| L5 full RGB renderer | PASS (NOT a mask codec; coord-MLP renders RGB) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~700 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (CPP1)
    parser_section_manifest:   parse_archive() -> 6 sections (header + base_mlp_blob
                               + modulation_blob + meta_blob + implicit
                               "shared_base_mlp_weights" + "per_pair_modulations" subsets)
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed shared base MLP state_dict
                               + int8 modulations + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~700 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- modulation paradigm = HARD-EARNED (Dupont ICML 2022 + Perez 2017 FiLM
  paper; modulating shared base networks is well-validated).
- MOD_DIM=64 = CARGO-CULTED (chosen for L0 sanity; sweep at L1 is critical
  since the per-pair modulation IS the latent rate).
- coord-MLP base architecture (vs CNN base) = HARD-EARNED-vs-CARGO-CULTED
  TENSION: coord-MLPs are the COIN/COIN++ canonical choice; but driving
  video may benefit from CNN base with per-pair modulation. Both directions
  worth exploring at L1.
- FiLM-style scale+shift modulation = HARD-EARNED (Perez 2017).
- int8 modulation quantization = CARGO-CULTED (chosen for tight rate; int16
  alternative needs empirical sweep).
- frame index as input dim = HARD-EARNED (standard for video-INR).

Operator 5-tier fit ranking citation:
    "COIN++ (Dupont ICML 2022) — MODERATE FIT ⭐⭐⭐ — modulation-based
     parameter-efficient INR. Shared base + tiny per-pair modulation.
     Different rate-tradeoff than NeRV-family; per-pair cost is small."
"""

from .architecture import (
    CoinplusplusConfig,
    CoinplusplusSubstrate,
)
from .archive import (
    CoinplusplusArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import CoinplusplusScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "CoinplusplusArchive",
    "CoinplusplusConfig",
    "CoinplusplusScoreAwareLoss",
    "CoinplusplusSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
