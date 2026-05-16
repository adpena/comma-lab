# SPDX-License-Identifier: MIT
"""STC clean-source v2 substrate package — canonical reactivation of Lane STC.

UNIQUE-AND-COMPLETE-PER-METHOD substrate per CLAUDE.md "UNIQUE-AND-COMPLETE-
PER-METHOD operating mode" non-negotiable + the 2026-05-16 batched-reactivation
design memo (commit ``18ed278af``) +
``.omx/research/resurrection_audit_20260516.md`` Tier 1 item #2.

The original Lane STC clean-source was tagged FALSIFIED 2026-04-29 on MPS-PROXY
evidence — explicitly NOT valid per CLAUDE.md "MPS auth eval is NOISE". STATUS
REVISION 2026-04-29 PM downgraded to UNDETERMINED-pending-CUDA. The $0.20 Modal
T4 CUDA re-run was queued but never executed. v2 packages the substrate per the
canonical META layer (Catalog #241/#242 SubstrateContract) so the operator can
fire the $0.20 disambiguator with the full dispatch optimization protocol
(Catalog #270) gating.

Canonical-vs-unique decision per layer (per the 2026-05-16 design memo
Section 2.2.8):

| Layer                              | Decision        | Rationale |
|------------------------------------|-----------------|-----------|
| Codec primitives (STC)             | UNIQUE FORK     | Filler & Pevny 2010 STC IS the substrate-class-shift; no canonical equivalent |
| Mask source                        | ADOPT canonical | The original kill came from NOT adopting CUDA SegNet argmax |
| Archive grammar (STCB)             | UNIQUE FORK     | STC bytes replace masks.mkv slot; new wire format |
| Inflate runtime (STC decoder)      | UNIQUE FORK     | Pure-byte parsing; NO torch / NO scorer per strict-scorer-rule |
| Auth-eval gate                     | ADOPT canonical | substrate-agnostic ``gate_auth_eval_call`` |
| Hardware detection                 | ADOPT canonical | substrate-agnostic ``detect_hardware_substrate`` |
| Modal dispatch                     | ADOPT canonical | ``modal_train_lane.py`` |
| Recipe schema                      | ADOPT canonical | 36-field SubstrateContract per Catalog #241/#242 |
| Scorer load                        | ADOPT canonical | ``load_differentiable_scorers`` (compress-side only) |
| Score-aware loss                   | FORK (N/A)      | STC is a codec; no gradient path; no training loop |
| eval_roundtrip in loop             | FORK (N/A)      | no training loop |
| EMA shadow weights                 | FORK (N/A)      | no learnable weights |

Observability surface (per the 2026-05-16 design memo Section 2.2.12):

  * Per-config audit JSONL: ``.omx/state/lane_stc_clean_source_v2_audit.jsonl``
    with ``boundary_fraction, stcb_bytes_raw, stcb_bytes_brotli, av1_bytes_baseline,
    encode_wall_clock_seconds, mask_source_device, mask_source_sha256``.
  * Modal call_id ledger: registered per Catalog #245.
  * Premise verification PV-1: MPS argmax masks != CUDA argmax masks (verified
    by sha256 of argmax outputs at known frame indices on both devices).
  * Score-axis tag: ``[contest-CUDA T4]`` for the CUDA re-run; ``[MPS-PROXY]``
    for any legacy MPS results.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY": the
trainer ships ``research_only=true`` until the $0.20 CUDA re-run lands a
sub-1MB STCB byte count; v2 is then promoted to L2 with a CUDA-axis anchor.
"""
from __future__ import annotations

from tac.substrates.stc_v2.archive import (
    STC_V2_MAGIC,
    STC_V2_VERSION,
    build_stc_v2_archive_bytes,
    parse_stc_v2_archive,
)
from tac.substrates.stc_v2.codec import (
    STCB_MAGIC,
    decode_stc_v2_masks,
    encode_stc_v2_masks,
)
from tac.substrates.stc_v2.inflate import inflate_one_video

__all__ = [
    "STCB_MAGIC",
    "STC_V2_MAGIC",
    "STC_V2_VERSION",
    "build_stc_v2_archive_bytes",
    "decode_stc_v2_masks",
    "encode_stc_v2_masks",
    "inflate_one_video",
    "parse_stc_v2_archive",
]
