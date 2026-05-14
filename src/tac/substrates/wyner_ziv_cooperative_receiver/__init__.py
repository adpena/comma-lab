"""Wyner-Ziv cooperative-receiver substrate (alien-tech N3).

Per ``feedback_expert_team_signal_processing_alien_tech_landed_20260513.md``
(Bell Labs / NSA / Lincoln Lab / JPL alien-tech expert team), Wyner-Ziv is
the **#1 ranked technique with predicted ΔS -0.05** — the largest single-
substrate ΔS prediction in the entire alien-tech catalog.

The contest IS Wyner-Ziv compression: the SegNet + PoseNet scorer is the
KNOWN side-information at the receiver. The encoder transmits only the
conditional bits ``H(source | scorer_weights + architecture + preprocessing)``
needed to reconstruct frames that the scorer accepts as low-distortion;
generic source bits (``H(source)``) are wasted because the scorer discards
most of them at preprocessing.

Mathematical contract per Slepian-Wolf-Wyner-Ziv 1973-1976 + DISCUS 2003
(Pradhan-Ramchandran):

    R_WZ(D) = inf { I(X; U) - I(Y; U) : E[d(X, X_hat(U, Y))] <= D }

where ``X`` is the source frame pair, ``Y`` is the side-information (the
scorer's sensitivity at every pixel given the *other* frame in the pair +
the contest preprocessing), ``U`` is an auxiliary random variable, and
``X_hat`` is the receiver's reconstruction.  The DISCUS construction
realizes ``R_WZ(D)`` via Slepian-Wolf cosets: bin source frames into
cosets of a structured code; transmit only the (small) coset index; the
receiver disambiguates by picking the coset member nearest to the
side-information predictor.

Distinction from Atick-Redlich (sister cooperative-receiver primitive in
``tac.substrates.time_traveler_l5_autonomy``):

* **Atick-Redlich 1990** — *efficient-coding theorem*: the encoder
  maximizes ``MI(B; S(B))`` (bits transmitted vs scorer output). This is
  a forward-pass MI objective on the encoder. It does not use side
  information at the decoder explicitly; the encoder is "scorer-aware"
  but the decoder is generic.
* **Wyner-Ziv 1976** — *side-information at the decoder*: the encoder
  ASSUMES the decoder has access to ``Y`` (here: scorer + scorer
  preprocessing of paired frames) and only transmits the ``H(X | Y)``
  bits the decoder cannot reconstruct from ``Y`` alone. The decoder
  computes ``Y`` from the same scorer the encoder targeted.

The two are complementary, NOT redundant: Atick-Redlich shapes the
encoder loss; Wyner-Ziv shapes the archive grammar. The Time-Traveler
substrate uses Atick-Redlich (cooperative-receiver loss only); this
substrate uses Wyner-Ziv (cooperative-receiver loss + Slepian-Wolf
binning of source frames into cosets so the archive transmits the coset
index, not the full source).

**Predicted contest-CPU score band: 0.140-0.150** ``[wyner-ziv-prediction]``
(NOT a score claim — score authority requires CUDA + CPU paired auth eval
on 1:1 contest-CI hardware per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"). Predicted ΔS of -0.05
relative to PR101 baseline 0.193 lands in the 0.140-0.150 range; clears
the sub-0.188 gate per ``feedback_solver_stack_wire_in_sweep_landed_20260513``
autopilot dispatch ranking.

Catalog #124 STRICT archive-grammar 8 fields (declared inline so the AST
walker observes them):

- ``archive_grammar``: monolithic single-file ``0.bin`` per HNeRV parity L3
- ``parser_section_manifest``: WZ1 header + 4 length-prefixed sections
  (renderer state_dict + per-pair coset indices + side-info predictor +
  meta JSON)
- ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
  (full renderer + Slepian-Wolf coset disambiguation + side-info-aware
  reconstruction)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 <= 2 deps)
- ``export_format``: WZ1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``WynerZivCooperativeReceiverLoss`` runs
  eval-roundtrip + scorer-conditional rate term per Wyner-Ziv ``R_WZ(D)``
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7); the full renderer + DISCUS coset coder is substrate engineering
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes; archive
  payload is structurally consumed by every section of inflate.py

Cross-references
----------------
- Master memo: ``~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_expert_team_signal_processing_alien_tech_landed_20260513.md``
  (N3 Wyner-Ziv entry — top-ranked alien-tech compression lineage)
- Sister substrate (Atick-Redlich cooperative-receiver loss only):
  ``tac.substrates.time_traveler_l5_autonomy``
- Sister primitive (in flight at ``tac.codec.cooperative_receiver``):
  separate canonical implementation; this substrate imports the primitive
  if it lands first, otherwise uses a local stub with a TODO marker.
- Canonical scorer-input contract: ``tac.substrates.score_aware_common``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``

Lane: ``lane_wyner_ziv_cooperative_receiver_substrate_20260513``
"""

from tac.substrates.wyner_ziv_cooperative_receiver.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_COSET_INDEX_BITS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    SideInfoPredictor,
    WynerZivConfig,
    WynerZivSubstrate,
)
from tac.substrates.wyner_ziv_cooperative_receiver.archive import (
    WZ1_MAGIC,
    WZ1_SCHEMA_VERSION,
    WynerZivArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.wyner_ziv_cooperative_receiver.score_aware_loss import (
    WynerZivCooperativeReceiverLoss,
    WynerZivLossWeights,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_COSET_INDEX_BITS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "SideInfoPredictor",
    "WZ1_MAGIC",
    "WZ1_SCHEMA_VERSION",
    "WynerZivArchive",
    "WynerZivConfig",
    "WynerZivCooperativeReceiverLoss",
    "WynerZivLossWeights",
    "WynerZivSubstrate",
    "pack_archive",
    "parse_archive",
]
