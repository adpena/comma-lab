#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""T9 — cross-archive substrate composition (Phase 2 building block).

Source-of-finding
─────────────────
``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md``
prescribed T9 as a per-section codec dispatch within the contest archive's
monolithic ``0.bin`` / ``x`` member. Each section can come from a
``[contest-CUDA]``-anchored substrate via a 5-byte header indicating
per-section codec choice.

The substrate-vs-codec meta-pattern
──────────────────────────────────
``feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`` warns
that HStack composition on score-naive substrates is dominated by the
substrate's training discipline; therefore every section MUST come from a
``[contest-CUDA]`` or ``[contest-CPU GHA]`` anchored substrate.

What this tool does
──────────────────
1. **Inventory**: read each anchored substrate's archive ZIP, reconstruct
   the ``x`` / ``0.bin`` byte layout from the substrate's own
   ``parse_archive`` (or equivalent) function, and emit a per-substrate
   section table: ``(section_name, byte_offset, byte_length, codec,
   sha256)``.

2. **Composability matrix**: mark every cross-substrate (section_name,
   src_substrate, dst_substrate) cell as one of:
     - ``compatible``     — same byte format, swappable in-place
     - ``codec_mismatch`` — same logical content, different codec (needs
                            transcode pass)
     - ``substrate_tied`` — semantics depend on companion section (e.g.
                            PR101 decoder weights are co-trained with PR101
                            latent; cannot swap independently)

3. **Build smoke composition**: assemble a candidate that swaps ONLY the
   ``compatible``-marked sections from a "best-of" donor substrate into a
   target (anchor) substrate. Validates the inflate.sh roundtrip locally
   on the host's CPU.

4. **Manifest + custody**: write
   ``experiments/results/cross_archive_substrate_composition_<UTC>/{
        archive.zip, submission_dir/, build_manifest.json,
        composability_matrix.json, byte_layout_inventory.json,
        rebuild_command.txt
     }``.

5. **GHA dispatch hook**: if the composition's local inflate.sh roundtrip
   passes AND ``--gha-dispatch`` is set, emit a one-shot dispatch command
   for ``tools/dispatch_cpu_eval_via_github_actions.py``.

Substrate-aware byte layouts (verified on 2026-05-09 archives)
─────────────────────────────────────────────────────────────
A1 (`x`, 178162 B) and PR101 (`x`, 178158 B) share the
``hnerv_ft_microcodec`` byte format:
  [0:162164] decoder Brotli streams
  [162164:177551] latent LZMA blob
  [177551:end] sidecar Brotli (variable; ~607-611 B)

PR103 (`x`, 178123 B) uses ``hnerv_lc_ac`` with an 8-section layout:
  [sca | br | hists_b | merged_ac | mins_scales | lo_b | hi_hist_b | wrp_b]
This is a DIFFERENT codec entirely (constriction RangeDecoder + brotli
hybrid). The decoder schema also differs (rgb_0 + rgb_1 split heads).

PR107 (`0.bin`, 178284 B) uses a third byte format (PR107 inflate tree
not present in this repo's intake clones; flagged ``substrate_tied`` for
all sections until the deconstruction lands).

Conclusion (T9 honest scope)
────────────────────────────
The cross-archive sections of A1 / PR101 / PR103 / PR107 are NOT
trivially swappable; they are substrate-tied (different decoder schemas
+ different latent codecs). The only truly compatible cross-substrate
swap currently anchored is **A1 sidecar ↔ PR101 sidecar** (same byte
format; sidecar is independent of decoder/latent training). Because the
local PR101 row is still an advisory public-comment intake, that swap remains
inventory-only until exact PR101 custody lands.

This tool emits the inventory + composability matrix unconditionally
(useful as a typed atom for the meta-Lagrangian planner) and builds the
A1 ⊕ PR101-sidecar smoke composition only when ``--build`` is set.
Other compositions are marked DEFERRED-pending-decoder-wiring with
specific reactivation criteria.

Tagged: ``[T9-substrate-composition]`` ``score_claim=false``
        ``ready_for_exact_eval_dispatch=false``
        ``promotion_eligible=false``
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXACT_DISPATCH_CONTRACT_BLOCKER = (
    "archive_bound_candidate_contract_required_before_exact_dispatch"
)
SMOKE_INFLATE_AUTHORITY_BLOCKER = "smoke_inflate_is_not_exact_eval_authority"


def _is_under_repo_root(path: Path) -> bool:
    """Whether ``path`` lives under the repo root (so relative_to() is safe).

    Tests that pass a tmp_path output root will return False; the manifest
    fields then carry the absolute path instead, which is acceptable for
    test artifacts that are NEVER persisted into git.
    """
    try:
        Path(path).resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# Anchored-substrate registry
# ---------------------------------------------------------------------------
#
# Each entry MUST point at a ``[contest-CPU GHA]`` or ``[contest-CUDA]``
# anchored archive. Per CLAUDE.md substrate-vs-codec meta-pattern:
# composing on score-naive substrates is forbidden — only anchored.

@dataclasses.dataclass
class SubstrateEntry:
    name: str
    archive_relpath: str
    submission_relpath: str          # path to submissions/<name>/inflate.{py,sh}
    anchored_score: float            # contest-CPU or contest-CUDA score
    anchored_score_tag: str          # "[contest-CUDA]" / "[contest-CPU GHA]"
    member_name: str                 # ZIP member ("x" or "0.bin")
    section_layout: str              # "hnerv_ft_microcodec" / "hnerv_lc_ac" / ...


SUBSTRATE_REGISTRY: tuple[SubstrateEntry, ...] = (
    SubstrateEntry(
        name="A1",
        archive_relpath=(
            "experiments/results/track1_phase_a1_score_gradient_latentalign_"
            "importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/"
            "finetuned_archive/archive.zip"
        ),
        submission_relpath=(
            "experiments/results/track1_phase_a1_score_gradient_"
            "latentalign_importpathfix_lr2e6_20260509T012628Z_modal/"
            "harvested_artifacts/finetuned_archive/submission_dir"
        ),
        anchored_score=0.19284757743677347,
        anchored_score_tag="[contest-CPU GHA]",
        member_name="x",
        section_layout="a1_finetuned_with_section_header",
    ),
    SubstrateEntry(
        name="PR101",
        archive_relpath=(
            "experiments/results/public_pr101_hnerv_ft_microcodec_intake_"
            "20260504_codex/archive.zip"
        ),
        submission_relpath=(
            "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
            "source/submissions/hnerv_ft_microcodec"
        ),
        # PR101 public CUDA: 0.193 (host-claimed); contest-CUDA replay
        # not yet anchored on this exact archive bytes — flagged advisory.
        anchored_score=0.193,
        anchored_score_tag="[advisory: public PR101 host-claimed]",
        member_name="x",
        section_layout="hnerv_ft_microcodec",
    ),
    SubstrateEntry(
        name="PR103",
        archive_relpath=(
            "experiments/results/public_pr103_intake_20260504_codex/archive.zip"
        ),
        submission_relpath=(
            "experiments/results/public_pr103_intake_20260504_codex/"
            "source/submissions/hnerv_lc_ac"
        ),
        anchored_score=0.195,
        anchored_score_tag="[advisory: public PR103 host-claimed silver]",
        member_name="x",
        section_layout="hnerv_lc_ac",
    ),
    SubstrateEntry(
        name="PR107",
        archive_relpath=(
            "experiments/results/pr107_cpu_eval_20260508/eval_work/submission/archive.zip"
        ),
        submission_relpath="",  # submission tree not in repo as intake clone
        anchored_score=0.19663589,
        anchored_score_tag="[contest-CPU GHA]",
        member_name="0.bin",
        section_layout="pr107_apogee_unknown",
    ),
)


# ---------------------------------------------------------------------------
# Section layout parsers (one per substrate format)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Section:
    name: str
    offset: int
    length: int
    codec: str
    sha256: str
    notes: str = ""


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_hnerv_ft_microcodec_layout(blob: bytes) -> list[Section]:
    """PR101 layout: 3 fixed sections (no length prefix).

    Reference: experiments/results/public_pr101_hnerv_ft_microcodec_intake_
    20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py
    constants DECODER_BLOB_LEN = 162_164, LATENT_BLOB_LEN = 15_387.
    Sidecar is the variable tail.
    """
    DECODER_LEN = 162_164
    LATENT_LEN = 15_387
    if len(blob) < DECODER_LEN + LATENT_LEN:
        raise ValueError(
            f"hnerv_ft_microcodec blob too small: {len(blob)} < "
            f"{DECODER_LEN + LATENT_LEN}"
        )
    decoder_b = blob[:DECODER_LEN]
    latent_b = blob[DECODER_LEN : DECODER_LEN + LATENT_LEN]
    sidecar_b = blob[DECODER_LEN + LATENT_LEN :]
    return [
        Section("decoder", 0, DECODER_LEN, "brotli_streams_int8",
                _sha256(decoder_b),
                "Concatenated Brotli streams of q-bytes + fp16 scale per tensor."),
        Section("latent", DECODER_LEN, LATENT_LEN, "lzma_temporal_delta",
                _sha256(latent_b),
                "LZMA(fp16 min/scale per dim + centered temporal-delta uint8 codes)."),
        Section("sidecar", DECODER_LEN + LATENT_LEN, len(sidecar_b),
                "brotli_per_pair_corrections",
                _sha256(sidecar_b),
                "Brotli((u8 dim, i8 delta_x100) per frame pair)."),
    ]


def parse_a1_finetuned_layout(blob: bytes) -> list[Section]:
    """A1 (score-gradient fine-tuned) layout: 4-byte length prefix + 3 sections.

    Reference: A1's own ``submission_dir/inflate.py::parse_a1_finetuned_archive``.
    Wire format:
      [0:4]                                  uint32 LE = decoder_section_total
      [4:section_total]                      decoder_blob (PR101 split-Brotli)
      [section_total:section_total+15387]    latent_blob (PR101 ORIGINAL)
      [section_total+15387:]                 sidecar_blob (PR101 ORIGINAL)
    """
    import struct
    LATENT_LEN = 15_387
    if len(blob) < 4:
        raise ValueError("A1 archive too short for 4-byte length header")
    section_total = struct.unpack_from("<I", blob, 0)[0]
    if section_total < 4 or section_total > len(blob):
        raise ValueError(f"bad A1 decoder_section_total {section_total}")
    decoder_b = blob[4:section_total]
    latent_b = blob[section_total : section_total + LATENT_LEN]
    sidecar_b = blob[section_total + LATENT_LEN :]
    if len(latent_b) != LATENT_LEN:
        raise ValueError(
            f"A1 latent_blob truncated: got {len(latent_b)}, expected {LATENT_LEN}"
        )
    return [
        Section("a1_section_header", 0, 4, "raw_uint32_le_section_total",
                _sha256(blob[:4]),
                "uint32 LE indicating decoder_section_total bytes (variable)."),
        Section("decoder", 4, section_total - 4, "brotli_streams_int8",
                _sha256(decoder_b),
                "Concatenated Brotli streams (PR101 split-Brotli, A1 fine-tuned)."),
        Section("latent", section_total, LATENT_LEN, "lzma_temporal_delta",
                _sha256(latent_b),
                "LZMA(fp16 min/scale per dim + centered temporal-delta uint8 codes)."),
        Section("sidecar", section_total + LATENT_LEN, len(sidecar_b),
                "brotli_per_pair_corrections",
                _sha256(sidecar_b),
                "Brotli((u8 dim, i8 delta_x100) per frame pair)."),
    ]


def parse_hnerv_lc_ac_layout(blob: bytes) -> list[Section]:
    """PR103 layout: 8 fixed-length sections via ``parse_archive``.

    Reference: experiments/results/public_pr103_intake_20260504_codex/
    source/submissions/hnerv_lc_ac/inflate.py constants.
    """
    SCA_LEN = 28 * 2          # N_TENSORS * 2 fp16 scales
    BR_LEN = 7097
    HIST_LEN = 895
    MERGED_AC_LEN = 153856
    LATENT_META_LEN = 28 * 4  # LATENT_DIM * 4 (fp16 mins + fp16 scales)
    LO_LEN = 15537
    HI_HIST_LEN = 15
    head_len = (
        SCA_LEN + BR_LEN + HIST_LEN + MERGED_AC_LEN
        + LATENT_META_LEN + LO_LEN + HI_HIST_LEN
    )
    if len(blob) < head_len:
        raise ValueError(
            f"hnerv_lc_ac blob too small: {len(blob)} < {head_len}"
        )
    o = 0
    sca = blob[o : o + SCA_LEN]
    o += SCA_LEN
    br = blob[o : o + BR_LEN]
    o += BR_LEN
    hists = blob[o : o + HIST_LEN]
    o += HIST_LEN
    merged_ac = blob[o : o + MERGED_AC_LEN]
    o += MERGED_AC_LEN
    mins_scales = blob[o : o + LATENT_META_LEN]
    o += LATENT_META_LEN
    lo_b = blob[o : o + LO_LEN]
    o += LO_LEN
    hi_hist = blob[o : o + HI_HIST_LEN]
    o += HI_HIST_LEN
    wrp = blob[o:]
    return [
        Section("scales", 0, SCA_LEN, "raw_fp16",
                _sha256(sca), "Per-tensor fp16 scales."),
        Section("br_concat", SCA_LEN, BR_LEN, "brotli_int8_concat",
                _sha256(br), "Brotli of concatenated non-AC tensor bytes."),
        Section("hists", SCA_LEN + BR_LEN, HIST_LEN, "brotli_u8_histograms",
                _sha256(hists), "Brotli of per-AC-tensor 256-bin histograms."),
        Section("merged_ac",
                SCA_LEN + BR_LEN + HIST_LEN, MERGED_AC_LEN,
                "constriction_range_coded",
                _sha256(merged_ac),
                "constriction RangeCoder over AC-tensors + hi latent bytes."),
        Section("latent_meta",
                SCA_LEN + BR_LEN + HIST_LEN + MERGED_AC_LEN, LATENT_META_LEN,
                "raw_fp16_min_scale",
                _sha256(mins_scales),
                "Per-dim fp16 mins + fp16 scales."),
        Section("latent_lo",
                SCA_LEN + BR_LEN + HIST_LEN + MERGED_AC_LEN + LATENT_META_LEN,
                LO_LEN, "brotli_u8_lo_byte_temporal",
                _sha256(lo_b),
                "Brotli of low-byte temporal-delta zig-zag codes."),
        Section("hi_hist",
                SCA_LEN + BR_LEN + HIST_LEN + MERGED_AC_LEN + LATENT_META_LEN
                + LO_LEN, HI_HIST_LEN, "brotli_u16_histogram",
                _sha256(hi_hist),
                "Brotli of high-byte AC histogram."),
        Section("wrp",
                SCA_LEN + BR_LEN + HIST_LEN + MERGED_AC_LEN + LATENT_META_LEN
                + LO_LEN + HI_HIST_LEN,
                len(wrp), "brotli_per_pair_wrap",
                _sha256(wrp), "Brotli per-pair wrap corrections."),
    ]


def parse_unknown_layout(blob: bytes) -> list[Section]:
    """Fallback for substrates without a registered parser (e.g. PR107)."""
    return [
        Section(
            "monolithic_unknown", 0, len(blob), "unknown",
            _sha256(blob),
            "Layout parser not registered; treat as opaque substrate-tied blob.",
        )
    ]


SECTION_PARSERS = {
    "hnerv_ft_microcodec": parse_hnerv_ft_microcodec_layout,
    "a1_finetuned_with_section_header": parse_a1_finetuned_layout,
    "hnerv_lc_ac": parse_hnerv_lc_ac_layout,
    "pr107_apogee_unknown": parse_unknown_layout,
}


def _is_exact_evidence_tag(tag: str) -> bool:
    """Return True only for local exact-eval custody tags.

    Public host comments and rounded leaderboard claims are useful inventory
    context, but they are not eligible donors for an archive composition build
    until the exact archive/runtime pair has local contest-CUDA or Linux
    contest-CPU custody.
    """
    return tag.startswith("[contest-CUDA]") or tag.startswith("[contest-CPU")  # CUSTODY_VALIDATOR_OK: inventory-only eligibility label; composition promotion still requires exact archive/runtime custody


# ---------------------------------------------------------------------------
# Inventory + composability matrix
# ---------------------------------------------------------------------------

def inventory_substrate(entry: SubstrateEntry) -> dict:
    """Open the substrate archive, parse its sections, return a dict."""
    archive_path = REPO_ROOT / entry.archive_relpath
    if not archive_path.exists():
        return {
            "name": entry.name,
            "available": False,
            "missing_path": str(archive_path),
        }
    archive_bytes = archive_path.read_bytes()
    archive_sha = _sha256(archive_bytes)
    with zipfile.ZipFile(archive_path) as z:
        if entry.member_name not in z.namelist():
            return {
                "name": entry.name,
                "available": False,
                "missing_member": entry.member_name,
            }
        blob = z.read(entry.member_name)
    parser = SECTION_PARSERS.get(entry.section_layout, parse_unknown_layout)
    sections = parser(blob)
    return {
        "name": entry.name,
        "available": True,
        "archive_relpath": entry.archive_relpath,
        "archive_size_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "anchored_score": entry.anchored_score,
        "anchored_score_tag": entry.anchored_score_tag,
        "exact_evidence_eligible": _is_exact_evidence_tag(entry.anchored_score_tag),
        "member_name": entry.member_name,
        "member_size_bytes": len(blob),
        "member_sha256": _sha256(blob),
        "section_layout": entry.section_layout,
        "sections": [dataclasses.asdict(s) for s in sections],
    }


def composability_matrix(inventory: list[dict]) -> dict:
    """Cross-substrate composability table.

    For every (section_name, src_substrate, dst_substrate) cell, classify:
      - `compatible`     — same layout family, same codec, same byte length
      - `codec_mismatch` — same layout family, same byte length, different codec
      - `length_mismatch` — same layout family, same codec, different lengths
      - `substrate_tied` — different layout families OR section is co-trained
    """
    cells: list[dict] = []
    for src in inventory:
        if not src.get("available"):
            continue
        for dst in inventory:
            if not dst.get("available") or dst["name"] == src["name"]:
                continue
            same_layout = src["section_layout"] == dst["section_layout"]
            for src_sec in src.get("sections", []):
                # Find dst section with the same name.
                dst_sec = next(
                    (s for s in dst.get("sections", []) if s["name"] == src_sec["name"]),
                    None,
                )
                if not same_layout or dst_sec is None:
                    verdict = "substrate_tied"
                    reason = (
                        "Different section_layout families OR section name"
                        " absent in dst; cross-substrate swap requires shared"
                        " parse_archive contract."
                    )
                elif src_sec["codec"] != dst_sec["codec"]:
                    verdict = "codec_mismatch"
                    reason = (
                        f"src codec={src_sec['codec']} != dst codec="
                        f"{dst_sec['codec']}; would require transcode pass."
                    )
                elif src_sec["length"] != dst_sec["length"] and src_sec["name"] != "sidecar":
                    # Sidecar is the variable-length tail in
                    # ``hnerv_ft_microcodec`` and is parsed as ``blob[OFFSET:]``
                    # in the runtime inflate.py — swapping a different-length
                    # sidecar is byte-level safe IF the donor sidecar is a
                    # valid Brotli stream of (u8 dim, i8 delta_x100) per pair.
                    # Other sections are length-fixed; size mismatch breaks
                    # the layout offsets.
                    verdict = "length_mismatch"
                    reason = (
                        f"src length={src_sec['length']} != dst length="
                        f"{dst_sec['length']}; layout offsets would shift,"
                        " requires runtime tree update."
                    )
                else:
                    # Decoder + latent are co-trained: even if byte-formatted
                    # identically, swapping them across substrates breaks the
                    # learned coupling.
                    if src_sec["name"] in {"decoder", "latent"}:
                        verdict = "substrate_tied"
                        reason = (
                            "Decoder weights and latent codes are co-trained;"
                            " cross-substrate swap breaks the learned coupling"
                            " (per CLAUDE.md substrate-vs-codec meta-pattern)."
                        )
                    else:
                        verdict = "compatible"
                        reason = (
                            "Same layout family, same codec; sidecar is the"
                            " variable-length tail and is parsed as blob[OFFSET:]"
                            " in inflate.py — donor sidecar is byte-level safe"
                            " to swap as long as it is a valid Brotli stream."
                        )
                cells.append({
                    "section": src_sec["name"],
                    "src": src["name"],
                    "dst": dst["name"],
                    "verdict": verdict,
                    "reason": reason,
                    "src_sha256": src_sec["sha256"],
                    "dst_sha256": dst_sec["sha256"] if dst_sec else None,
                    "src_length": src_sec["length"],
                    "dst_length": dst_sec["length"] if dst_sec else None,
                })
    return {
        "verdict_legend": {
            "compatible": "Same layout / codec / length / not co-trained — swap is byte-level safe.",
            "codec_mismatch": "Same logical content but different codec — needs transcode pass.",
            "length_mismatch": "Same codec but different byte length — runtime tree must update.",
            "substrate_tied": "Layout family differs OR section is co-trained; swap breaks semantics.",
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Build smoke composition + inflate roundtrip validation
# ---------------------------------------------------------------------------

def _select_smoke_composition(inventory: list[dict], matrix: dict) -> dict | None:
    """Pick the cheapest, most-compatible smoke composition.

    Strategy: anchor on A1 (highest-scoring `[contest-CPU GHA]` substrate
    that is also our score-gradient retrain), and look for any single
    non-decoder, non-latent section that is `compatible` with an exact-eval
    donor substrate. Advisory public-comment rows remain inventory-only.
    The sidecar / wrp section is the canonical candidate once an exact donor
    exists.
    """
    anchor = next((s for s in inventory if s.get("name") == "A1" and s.get("available")), None)
    if anchor is None:
        return None
    exact_eligible = {
        s["name"] for s in inventory
        if s.get("available") and s.get("exact_evidence_eligible")
    }
    candidates = [
        c for c in matrix["cells"]
        if c["verdict"] == "compatible" and c["dst"] == anchor["name"]
        and c["src"] in exact_eligible and c["dst"] in exact_eligible
    ]
    if not candidates:
        return None
    # Pick the donor that gives the smallest section-byte change (closest to
    # anchor → most likely to be roundtrip-safe in the substrate's existing
    # decoder).
    candidates.sort(key=lambda c: abs(c["src_length"] - c["dst_length"]))
    pick = candidates[0]
    return {
        "anchor": anchor["name"],
        "donor": pick["src"],
        "swap_section": pick["section"],
        "src_sha256": pick["src_sha256"],
        "dst_sha256": pick["dst_sha256"],
        "src_length": pick["src_length"],
        "dst_length": pick["dst_length"],
    }


def _assemble_swapped_blob(
    anchor_inv: dict,
    donor_inv: dict,
    swap_section: str,
    repo_root: Path,
) -> bytes:
    """Construct the new monolithic ``x`` blob with one section substituted.

    All other sections are copied verbatim from the anchor's archive.
    """
    anchor_archive_path = repo_root / anchor_inv["archive_relpath"]
    donor_archive_path = repo_root / donor_inv["archive_relpath"]
    with zipfile.ZipFile(anchor_archive_path) as z:
        anchor_blob = z.read(anchor_inv["member_name"])
    with zipfile.ZipFile(donor_archive_path) as z:
        donor_blob = z.read(donor_inv["member_name"])
    # Find the section offsets (anchor and donor must agree because we
    # only attempt this when the layout family matches).
    a_sec = next(s for s in anchor_inv["sections"] if s["name"] == swap_section)
    d_sec = next(s for s in donor_inv["sections"] if s["name"] == swap_section)
    # Sidecar is the variable-length tail; other sections are length-fixed.
    if a_sec["length"] != d_sec["length"] and swap_section != "sidecar":
        raise RuntimeError(
            f"swap section length mismatch on fixed-length section "
            f"{swap_section!r}: anchor {a_sec['length']} vs donor {d_sec['length']}"
        )
    out = bytearray(anchor_blob)
    donor_section_bytes = donor_blob[d_sec["offset"] : d_sec["offset"] + d_sec["length"]]
    # bytearray slice assignment with a different-length RHS rewrites the
    # tail; this is correct for the variable-length sidecar at the end of
    # the blob, and would be incorrect for a fixed-length interior section
    # (guarded above).
    out[a_sec["offset"] : a_sec["offset"] + a_sec["length"]] = donor_section_bytes
    return bytes(out)


def _assemble_archive(
    blob: bytes,
    member_name: str,
    out_path: Path,
) -> None:
    """Write the swapped blob into a fresh deterministic ZIP."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use ZipInfo + writestr with fixed timestamp for determinism (per
    # check_archive_builders_use_deterministic_zip).
    zi = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_STORED  # uncompressed; matches anchor archive
    with zipfile.ZipFile(out_path, "w") as z:
        z.writestr(zi, blob)


def _smoke_inflate(
    submission_dir: Path,
    archive_path: Path,
) -> dict:
    """Run the submission's inflate.py on the swapped archive, capture
    return code + decoded byte count + any stderr.

    This is a LOCAL CPU smoke test; it validates only that the byte
    layout is parseable + the decoder runs without crashing. It is NOT a
    contest-CPU score (which requires upstream/evaluate.py on the GHA
    runner with the canonical 600-pair video).
    """
    inflate_py = submission_dir / "inflate.py"
    if not inflate_py.exists():
        return {"smoke_ok": False, "reason": f"no inflate.py at {inflate_py}"}
    work = Path(tempfile.mkdtemp(prefix="t9_smoke_"))
    raw_out = work / "decoded.raw"
    try:
        result = subprocess.run(
            [
                sys.executable, str(inflate_py),
                str(archive_path), str(raw_out),
            ],
            capture_output=True, text=True, timeout=600,
        )
        ok = (result.returncode == 0 and raw_out.exists())
        return {
            "smoke_ok": ok,
            "returncode": result.returncode,
            "decoded_bytes": raw_out.stat().st_size if raw_out.exists() else 0,
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"smoke_ok": False, "reason": "inflate.py timed out at 600s"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "T9 cross-archive substrate composition (Phase 2 building block). "
            "Inventories anchored substrates, emits composability matrix, and "
            "optionally builds + smoke-validates a single-section swap."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments" / "results"
        / f"cross_archive_substrate_composition_{time.strftime('%Y%m%dT%H%M%SZ')}",
        help="output directory for build_manifest.json + archive.zip",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="build a smoke composition (anchor=A1, swap=A1's sidecar with"
             " a donor substrate's compatible sidecar)",
    )
    parser.add_argument(
        "--smoke-inflate", action="store_true",
        help="if --build, run the submission's inflate.py on the swapped"
             " archive locally and record the result",
    )
    parser.add_argument(
        "--gha-dispatch", action="store_true",
        help="if --build and --smoke-inflate pass, emit the dispatch command"
             " for tools/dispatch_cpu_eval_via_github_actions.py (does NOT"
             " execute it; operator must opt-in to the dispatch + spend)",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[T9] writing to {args.output_root}")

    # Stage 1: inventory.
    inventory = [inventory_substrate(e) for e in SUBSTRATE_REGISTRY]
    inv_path = args.output_root / "byte_layout_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2, sort_keys=True))
    print(f"[T9] inventory → {inv_path.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root}")
    for inv in inventory:
        if inv.get("available"):
            print(f"  - {inv['name']}: {inv['member_name']} = "
                  f"{inv['member_size_bytes']} B; "
                  f"{len(inv['sections'])} sections; "
                  f"{inv['anchored_score']} {inv['anchored_score_tag']}")
        else:
            print(f"  - {inv['name']}: UNAVAILABLE "
                  f"({inv.get('missing_path') or inv.get('missing_member')})")

    # Stage 2: composability matrix.
    matrix = composability_matrix(inventory)
    mat_path = args.output_root / "composability_matrix.json"
    mat_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    counts: dict[str, int] = {}
    for c in matrix["cells"]:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    print(f"[T9] composability matrix → {mat_path.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root}: {counts}")

    build_manifest = {
        "tool": "build_cross_archive_substrate_composition.py",
        "phase": "Phase 2 building block (T9)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "from_state_hash": _sha256(
            json.dumps(inventory, sort_keys=True).encode()
            + json.dumps(matrix, sort_keys=True).encode()
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "smoke_inflate_passed": False,
        "archive_bound_candidate_contract_required": True,
        "evidence_grade": "byte-anchor-and-composability-only",
        "lane_tag": "[T9-substrate-composition]",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime",
        "next_required_actions": [
            "Run --build --smoke-inflate to validate the A1-anchor sidecar swap.",
            (
                "If smoke passes, build an archive-bound candidate contract "
                "and exact-ready bridge before any CPU/CUDA dispatch."
            ),
            "Operator approves dispatch only after exact readiness signs custody.",
        ],
        "dispatch_blockers": [
            EXACT_DISPATCH_CONTRACT_BLOCKER,
            SMOKE_INFLATE_AUTHORITY_BLOCKER,
            (
                "public_frontier_advisory_rows_inventory_only: PR101/PR103"
                " host-claimed rows are not exact local contest-CUDA or"
                " Linux contest-CPU custody, so they are excluded from"
                " buildable donor selection until exact replay lands."
            ),
            (
                "decoder_section_substrate_tied: cross-substrate decoder swap"
                " breaks learned coupling — DEFERRED-pending decoder retrain"
                " on shared latent."
            ),
            (
                "latent_section_substrate_tied: same as decoder; co-trained."
            ),
            (
                "PR103 hnerv_lc_ac uses incompatible byte format vs A1/PR101"
                " hnerv_ft_microcodec — DEFERRED-pending byte-format adapter."
            ),
            (
                "PR107 layout parser not registered — DEFERRED-pending"
                " deconstruction (intake clone absent in repo)."
            ),
            "reactivation_required_before_new_dispatch",
        ],
        "measured_config_status": "phase2_building_block_inventory_only",
        "reactivation_criteria": [
            "Decoder swap: dispatch a joint decoder+latent retrain that"
            " unifies A1 and PR101 substrates under a shared inflate runtime.",
            "PR103 cross-format: build an hnerv_lc_ac → hnerv_ft_microcodec"
            " transcoder that re-codes the AC-coded weight stream into the"
            " brotli-streamed format A1's decoder consumes.",
            "PR107 deconstruction: add the PR107 intake clone + parser entry"
            " to SECTION_PARSERS so its sections become composability-table-visible.",
        ],
        "inventory_path": str(inv_path.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root),
        "composability_matrix_path": str(mat_path.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root),
        "verdict_counts": counts,
    }

    smoke_composition: dict | None = None
    archive_out: Path | None = None

    # Stage 3: optional build smoke composition.
    if args.build:
        smoke_composition = _select_smoke_composition(inventory, matrix)
        if smoke_composition is None:
            print("[T9] --build requested but no `compatible` cross-substrate"
                  " cell exists in the matrix; skipping build (this is the"
                  " expected outcome under the substrate-vs-codec meta-pattern"
                  " when only decoder/latent sections are available).")
            build_manifest["smoke_composition"] = None
            build_manifest["next_required_actions"].insert(
                0,
                "BUILD SKIPPED: no compatible cross-substrate cell. The"
                " matrix-only inventory is still the actionable Phase 2 atom.",
            )
        else:
            anchor_inv = next(i for i in inventory if i["name"] == smoke_composition["anchor"])
            donor_inv = next(i for i in inventory if i["name"] == smoke_composition["donor"])
            print(f"[T9] building smoke composition: anchor={smoke_composition['anchor']} "
                  f"donor={smoke_composition['donor']} swap={smoke_composition['swap_section']}")
            blob = _assemble_swapped_blob(
                anchor_inv, donor_inv, smoke_composition["swap_section"], REPO_ROOT,
            )
            archive_out = args.output_root / "archive.zip"
            _assemble_archive(blob, anchor_inv["member_name"], archive_out)
            archive_sha = _sha256(archive_out.read_bytes())
            archive_size = archive_out.stat().st_size
            print(f"[T9] composed archive: {archive_out.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root} "
                  f"({archive_size} B sha256={archive_sha[:16]}...)")
            smoke_composition.update({
                "archive_path": str(archive_out.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root),
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive_size,
                "anchor_archive_sha256": anchor_inv["archive_sha256"],
                "donor_archive_sha256": donor_inv["archive_sha256"],
            })
            build_manifest["smoke_composition"] = smoke_composition

            # Optionally smoke-test inflate.
            if args.smoke_inflate:
                anchor_entry = next(e for e in SUBSTRATE_REGISTRY
                                    if e.name == smoke_composition["anchor"])
                submission_dir = REPO_ROOT / anchor_entry.submission_relpath
                smoke_result = _smoke_inflate(submission_dir, archive_out)
                build_manifest["smoke_inflate"] = smoke_result
                print(f"[T9] smoke inflate: {smoke_result}")
                if smoke_result.get("smoke_ok"):
                    build_manifest["smoke_inflate_passed"] = True
                    build_manifest["ready_for_exact_eval_dispatch"] = False
                    build_manifest["next_required_actions"][0] = (
                        "Smoke inflate PASSED — build archive-bound candidate"
                        " contract plus exact-ready bridge before CPU/CUDA dispatch."
                    )

            # Optionally record the GHA dispatch command as blocked until the
            # archive-bound contract and exact readiness bridge sign custody.
            if args.gha_dispatch and build_manifest.get("smoke_inflate_passed"):
                blocked_gha_cmd = (
                    ".venv/bin/python tools/dispatch_cpu_eval_via_github_actions.py "
                    f"--archive-path {archive_out.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root} "
                    f"--archive-sha {archive_sha} "
                    f"--submission-name t9_a1_anchor_{smoke_composition['donor'].lower()}_"
                    f"sidecar_swap_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())} "
                    f"--output-dir experiments/results/t9_smoke_eval_$(date -u +%Y%m%dT%H%M%SZ)"
                )
                build_manifest["blocked_gha_dispatch_command"] = blocked_gha_cmd
                build_manifest["gha_dispatch_blocked_reason"] = (
                    EXACT_DISPATCH_CONTRACT_BLOCKER
                )
                print(
                    "[T9] GHA dispatch blocked until archive-bound exact "
                    f"readiness: {blocked_gha_cmd}"
                )

    # Stage 4: write build_manifest + rebuild_command.
    manifest_path = args.output_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True))
    rebuild_cmd_path = args.output_root / "rebuild_command.txt"
    rebuild_cmd_path.write_text(
        "# Rebuild the T9 inventory + composability matrix.\n"
        "# Add --build / --smoke-inflate / --gha-dispatch to opt into\n"
        "# the smoke-composition flow.\n"
        ".venv/bin/python tools/build_cross_archive_substrate_composition.py\n"
    )
    print(f"[T9] manifest → {manifest_path.relative_to(REPO_ROOT) if _is_under_repo_root(args.output_root) else args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
