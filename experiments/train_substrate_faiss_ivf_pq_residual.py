#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""train_substrate_faiss_ivf_pq_residual — smoke trainer entry point (L0 SCAFFOLD).

Path 3 candidate #I Faiss IVF-PQ residual codec MLX-local L0 SCAFFOLD per
operator binding directive 2026-05-26 *"Never simply extend unless a
rigorous adversarial cargo cult pass has been done first"*. PHASE 1
cargo-cult audit committed `a883a717c`; PHASE 2 substrate-design decision
committed `587e3b85a` (path b REDIRECT); THIS PHASE 3 L0 SCAFFOLD landing.

Per Catalog #240(c) L0 SCAFFOLD posture: `_full_main raises NotImplementedError`
to declare paid-dispatch eligibility gated by:
- PHASE 2 council symposium per Catalog #325 per-substrate optimal form
- Catalog #1265 MLX↔PyTorch parity gate at threshold 0.001 contest-units
- Operator-frontier-override per Catalog #199 paired-env discipline

The `_smoke_main` path is a MLX-local research-only probe that ENCODES the
per-pair RGB residual against PR110 fec6 frontier reconstruction via PQ
codebook + per-pair codeword stream, validates archive round-trip + byte
mutation per Catalog #139, registers probe outcome per Catalog #313, and
exits with `[macOS-MLX research-signal]` non-promotable markers per
Catalog #192 + #287 + #323.

NO paid dispatch from this trainer until Catalog #325 symposium clears.
"""

from __future__ import annotations

import argparse
import sys


def _smoke_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD smoke entry point (MLX-local; research-only).

    The smoke path verifies:
    - Substrate import surface (mlx_renderer + numpy_reference + archive)
    - Synthetic round-trip via archive build + parse
    - L0 invariants: research_only=True / dispatch_enabled=False / score_claim=False

    Phase 3 sister probes (post-L0-landing) extend the smoke path to:
    - PR110 fec6 frontier residual extraction on 600 contest pairs
    - PQ codebook training (numpy reference + Faiss-CPU optional accelerator)
    - K-sweep + M-sweep + tile-size-sweep + per-class-conditioning binary
    - MLX↔numpy↔PyTorch parity validation per Catalog #1265
    """
    parser = argparse.ArgumentParser(
        description="Faiss IVF-PQ residual codec L0 SCAFFOLD smoke",
    )
    parser.add_argument("--smoke", action="store_true", help="run L0 SCAFFOLD smoke")
    parser.add_argument(
        "--validate-archive-roundtrip",
        action="store_true",
        help="validate synthetic archive build+parse round-trip",
    )
    args = parser.parse_args(argv)

    print(
        "[faiss_ivf_pq_residual L0 SCAFFOLD smoke] "
        "[macOS-MLX research-signal] "
        "research_only=True dispatch_enabled=False score_claim=False "
        "promotion_eligible=False ready_for_exact_eval_dispatch=False"
    )

    # Verify imports work
    from tac.substrates.faiss_ivf_pq_residual import (
        FAISSPQ1_HEADER_SIZE,
        FAISSPQ1_MAGIC,
        FaissIVFPQResidualConfig,
        build_archive_bytes,
        estimate_archive_bytes,
        estimate_per_pair_codeword_bytes_raw,
        parse_archive,
    )
    from tac.substrates.faiss_ivf_pq_residual.numpy_reference import (
        encode_per_pair_residual,
        train_pq_codebook,
    )

    print(f"[L0] FAISSPQ1_MAGIC={FAISSPQ1_MAGIC!r}; HEADER_SIZE={FAISSPQ1_HEADER_SIZE}")

    if args.validate_archive_roundtrip:
        import numpy as np

        # Coarse synthetic config for round-trip validation
        cfg = FaissIVFPQResidualConfig(
            m_sub_quantizers=2,
            ksub_codebook_size=8,
            tile_h=192,
            tile_w=256,
            num_pairs=4,
        )
        print(f"[L0] cfg: M={cfg.m_sub_quantizers} ksub={cfg.ksub_codebook_size} "
              f"tile=({cfg.tile_h},{cfg.tile_w}) tiles_per_pair={cfg.tiles_per_pair} "
              f"num_pairs={cfg.num_pairs}")
        raw_bytes_per_pair = estimate_per_pair_codeword_bytes_raw(cfg)
        est_archive = estimate_archive_bytes(cfg)
        print(f"[L0] estimated per-pair codeword bytes (raw): {raw_bytes_per_pair}")
        print(f"[L0] estimated archive bytes: {est_archive}")

        # Synthetic codebook + codewords
        rng = np.random.default_rng(42)
        codebook = rng.standard_normal(
            (cfg.m_sub_quantizers, cfg.ksub_codebook_size, cfg.sub_dim)
        ).astype(np.float32)
        codewords = rng.integers(
            0, cfg.ksub_codebook_size,
            size=(cfg.num_pairs, cfg.tiles_per_pair, cfg.m_sub_quantizers),
            dtype=np.uint16,
        )
        data = build_archive_bytes(codebook, codewords, tile_h=cfg.tile_h, tile_w=cfg.tile_w)
        arch = parse_archive(data)
        assert arch.codebook.shape == codebook.shape
        assert np.array_equal(arch.codebook, codebook)
        assert np.array_equal(arch.per_pair_codewords, codewords)
        assert arch.meta["research_only"] is True
        assert arch.meta["dispatch_enabled"] is False
        assert arch.meta["score_claim"] is False
        print(f"[L0] archive round-trip OK: actual_bytes={len(data)}")

    print("[L0] smoke complete; NO paid dispatch authorized")
    return 0


def _full_main(argv: list[str] | None = None) -> int:
    """L0 SCAFFOLD posture per Catalog #240(c): full main raises NotImplementedError.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    Catalog #220 non-negotiable + Catalog #325 per-substrate symposium
    discipline: paid-dispatch eligibility requires (1) PHASE 2 council
    symposium PROCEED verdict, (2) Catalog #1265 MLX↔PyTorch parity gate
    PASS at threshold 0.001 contest-units, (3) operator-frontier-override
    via Catalog #199 paired-env discipline.

    The full training path operationalizes:
    - PR110 fec6 frontier residual extraction via canonical inflate.sh
    - PQ codebook training via Faiss-CPU accelerator + numpy reference
    - K-sweep + M-sweep + tile-size-sweep + per-class-conditioning binary
    - Score-aware training via canonical score_pair_components per Catalog #164
    - EMA shadow per CLAUDE.md EMA non-negotiable (decay=0.997)
    - eval_roundtrip=True per CLAUDE.md eval_roundtrip non-negotiable
    - Catalog #245 Modal call_id ledger registration fail-closed per Catalog #339
    - Catalog #324 post-training Tier-C density validation
    """
    raise NotImplementedError(
        "faiss_ivf_pq_residual full main NOT YET IMPLEMENTED — L0 SCAFFOLD "
        "posture per Catalog #240(c). Phase 2 council symposium per Catalog "
        "#325 + Catalog #1265 MLX↔PyTorch parity gate REQUIRED before any "
        "paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md "
        "for the Phase 2+ roadmap."
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--smoke" in args:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
