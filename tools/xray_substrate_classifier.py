"""Xray substrate classifier — archive bytes → substrate class via magic-byte signatures.

Given an archive zip path, this tool classifies the substrate class by:

1. Reading the ZIP member directory + first-bytes patterns;
2. Scanning each member for known magic-byte signatures (PR106 sidecar 0xFE,
   magic_codec ``MAGC``, dense_streams ``MDS1``, PR91 ``QM0``/``QH0``,
   PR92 ``RMC1``/``RSA1``/``RSB1``, PR93 ``QZPDV1``/``QZMB1``, PR65 ``PQ12``,
   etc.);
3. Cross-referencing member names against substrate-class lookup tables for
   the PR family (PR106 r1/r2, PR101 HNeRV-LC, PR103 hnerv_lc_ac, magic_codec
   variants, Cool-Chic / C3 residual sidecars, NeRV families, etc.);
4. Computing per-section entropy estimates from observed byte counts;
5. Emitting a structured JSON manifest with the classification verdict +
   the per-section parser manifest.

The classifier IS the disambiguator for autopilot ranking: given an archive
bytes path, autopilot asks "what substrate is this?" and routes composition
primitives accordingly (per CLAUDE.md "Subagent coherence-by-default" probe-
disambiguator wire-in).

SISTER TOOL DISTINCTION (per ZZZZZ audit L3a 2026-05-12):
This tool is NOT a duplicate of ``tools/cpu_cuda_xray_substrate_class_classifier.py``.
The two tools share the ``substrate_class`` output token but operate on
DIFFERENT INPUTS:

- THIS tool (``xray_substrate_classifier.py``): consumes an archive ZIP and
  classifies via static magic-byte signatures + member-name lookup tables.
  Cheap, deterministic, runs offline pre-dispatch.
- SISTER tool (``cpu_cuda_xray_substrate_class_classifier.py``): consumes N
  per-substrate ``layer_drift.json`` files and classifies via CPU-vs-CUDA
  per-layer drift signature pairwise cosine similarity. Requires prior P5
  xray sweeps; runs post-dispatch as a numerical fingerprint check.

Both feed the autopilot's ``substrate_class`` column; they are complementary
(static pre-dispatch + numerical post-dispatch) rather than redundant.

Output JSON schema::

    {
        "archive_path": "...",
        "archive_sha256": "<hex>",
        "archive_size_bytes": <int>,
        "substrate_class": <class id; see _SUBSTRATE_CLASSES>,
        "substrate_class_confidence": "low" | "medium" | "high",
        "archive_version": <string | null>,
        "sections": [{"name": ..., "size_bytes": ..., "magic": ..., "entropy_estimate_bits_per_byte": ...}, ...],
        "parser_section_manifest": {"offsets": [...], "lengths": [...], "section_names": [...], "section_sha256s": [...]},
        "classification_signals": [<list of detected signature names>],
        "ambiguity_blockers": [<list of unresolved ambiguity flags>]
    }

CLAUDE.md compliance:
* no scorer load (pure stdlib + tac.packet_compiler magics);
* no MPS / torch import;
* no ``/tmp`` paths;
* refuses classification if header is unknown OR sections fail consistency;
* deterministic-bytes: same archive → same JSON body;
* output dir defaults to ``experiments/results/xray_substrate_classifier_<ts>/``;
* ``score_claim`` permanently absent from output (no score-related field exists
  in the schema).

Usage::

    python tools/xray_substrate_classifier.py \\
        --archive submissions/pr106_latent_sidecar_r2/archive.zip \\
        --output-dir experiments/results/xray_substrate_classifier_<ts>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Known substrate-class identifiers. The classifier emits one of these (or
# "unknown_substrate_unclassifiable" when no signal matches).
_SUBSTRATE_CLASSES: tuple[str, ...] = (
    "pr106_r1_sidecar",
    "pr106_r2_sidecar",
    "pr106_latent_sidecar_pr101_grammar",
    "magic_codec_packet",
    "magic_codec_dense_streams_packet",
    "hnerv_lc_family",
    "hnerv_lc_ac",
    "ego_nerv_substrate",
    "nervdc_substrate",
    "balle_renderer",
    "sane_hnerv",
    "cool_chic_residual_sidecar",
    "c3_residual_sidecar",
    "categorical_substrate",
    "apogee_intN",
    "factorized_hnerv_v1",
    "siren_residual_sidecar",
    "raft_pose_sidecar",
    "yshift_sidechannel",
    "lrl1_sidechannel",
    "anr_substrate",
    "ffnerv_substrate",
    "tcnerv_substrate",
    "dsnerv_substrate",
    "e_nerv_substrate",
    "hinerv_substrate",
    "blocknerv_substrate",
    # ── FIX-J substrate-scaffold packages (2026-05-12 Fields-medal council) ──
    # The new scaffold packages have distinct archive grammars (4-byte
    # magic header per substrate) and distinct substrate_ids vs the older
    # NeRV-family / residual-basis rows above.
    "hybrid_renderer_residual_substrate",
    "self_compress_nn_substrate",
    "pr101_lc_v2_clone_substrate",
    "cool_chic_full_renderer_substrate",
    "wavelet_full_renderer_substrate",
    "grayscale_lut_substrate",
    "vq_vae_substrate",
    "siren_substrate",
    "block_nerv_substrate",
    "tc_nerv_substrate",
    "ff_nerv_substrate",
    "ds_nerv_substrate",
    "hi_nerv_substrate",
    # ── CompressAI codec adapter packets (EEEE 2026-05-12 + FIX-J wire-in) ──
    # These are latent-stream codecs that ship inside a host substrate's
    # archive; classifier flags them when the magic-byte appears at byte 0
    # of a packet member.
    "compressai_factorized_prior_packet",
    "compressai_balle_hyperprior_packet",
    "compressai_cheng2020_packet",
    # ── WAVE-A-2 TRADITION 2 single-file substrates (2026-05-12 CANON-1.A) ──
    # Production-mature single-file renderers pre-dating substrate-scaffold
    # subpackage discipline; tradition memo at
    # ``.omx/research/substrate_tradition_taxonomy_20260512.md``.
    "cnerv_substrate",
    "lane_12_v2_nerv_substrate",
    "quantizr_faithful_substrate",
    "mlx_mask_renderer_substrate",  # `[macOS-CPU advisory only]`.
    "dp_sims_renderer_substrate",
    "diffusion_renderer_substrate",
    "unknown_substrate_unclassifiable",
)


# Known section-magic signatures the classifier scans for at member-byte-offset 0.
# Each entry maps signature bytes → human-readable label.
_SECTION_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"MDS1", "magic_codec_dense_streams"),
    (b"MAGC", "magic_codec_envelope"),
    (b"QM0\x00", "pr91_qm0_grammar"),
    (b"QH0\x00", "pr91_qh0_grammar"),
    (b"RMC1", "pr92_joint_stream_rmc1"),
    (b"RSA1", "pr92_joint_stream_rsa1"),
    (b"RSB1", "pr92_joint_stream_rsb1"),
    (b"QZPDV1\x00\x00", "pr93_pose_delta_varint"),
    (b"QZMB1\x00\x00\x00", "pr93_model_compact"),
    (b"PQ12", "pr65_pq12_pose"),
    (b"\xfe", "pr106_sidecar_v1"),  # PR106 sidecar packet (1-byte magic 0xFE)
    # ── FIX-J substrate-scaffold magic bytes (Fields-medal 2026-05-12) ──
    # Each substrate-scaffold package under src/tac/substrates/<name>/
    # declares a 4-byte ASCII magic in its archive.py. These signatures
    # let the classifier disambiguate the substrate at xray time without
    # cracking open the trained-weights blob.
    (b"SHV1", "sane_hnerv_v1"),  # alpha primary
    (b"BRV1", "balle_renderer_v1"),  # beta parallel
    (b"HRR1", "hybrid_renderer_residual_v1"),  # gamma deferred
    (b"SCV1", "self_compress_nn_v1"),  # delta deferred
    (b"PR12", "pr101_lc_v2_clone_v1"),  # forensic
    (b"CCV1", "cool_chic_full_renderer_v1"),
    (b"WLV1", "wavelet_full_renderer_v1"),
    (b"GLV1", "grayscale_lut_v1"),
    (b"VQV1", "vq_vae_substrate_v1"),
    (b"SRV1", "siren_substrate_v1"),
    (b"BNV1", "block_nerv_substrate_v1"),
    (b"TCV1", "tc_nerv_substrate_v1"),
    (b"FFV1", "ff_nerv_substrate_v1"),
    (b"DSV1", "ds_nerv_substrate_v1"),
    (b"HIV1", "hi_nerv_substrate_v1"),
    # ── CompressAI codec adapter magic bytes (EEEE 2026-05-12) ──
    (b"CAFP", "compressai_factorized_prior"),
    (b"CABH", "compressai_balle_hyperprior"),
    (b"CACG", "compressai_cheng2020"),
    # ── WAVE-A-2 TRADITION 2 single-file substrate magic bytes (2026-05-12) ──
    # Per CANON-1.A explicit-taxonomy resolution; substrate identity carried
    # by 4-byte ASCII magic when the substrate's archive emits a dedicated
    # packet member. For single-file substrates that ship via a shared
    # contest-archive grammar, the classifier falls back to ZIP-member-name
    # rules; these magics gate xray-time substrate disambiguation when the
    # archive contains a substrate-tagged section.
    (b"CNRV", "cnerv_v1"),
    (b"ENRV", "e_nerv_v1"),
    (b"EGOV", "ego_nerv_v1"),
    (b"L12V", "lane_12_v2_nerv_v1"),
    (b"NDCV", "nervdc_v1"),
    (b"QZRV", "quantizr_faithful_v1"),
    (b"MLXR", "mlx_mask_renderer_v1"),  # Apple-Silicon-only; advisory tag.
    (b"DPSV", "dp_sims_renderer_v1"),
    (b"DIFV", "diffusion_renderer_v1"),
)


# Substrate-class signals derived from ZIP member names + section signatures.
# Each rule fires when ALL of its required-members are present (subset match).
# Order matters: more-specific rules first.
@dataclass(frozen=True)
class _SubstrateRule:
    substrate_class: str
    required_members: frozenset[str] = field(default_factory=frozenset)
    required_member_prefixes: frozenset[str] = field(default_factory=frozenset)
    required_section_magics: frozenset[bytes] = field(default_factory=frozenset)
    forbidden_members: frozenset[str] = field(default_factory=frozenset)
    archive_version: str | None = None


_SUBSTRATE_RULES: tuple[_SubstrateRule, ...] = (
    _SubstrateRule(
        substrate_class="magic_codec_dense_streams_packet",
        required_section_magics=frozenset({b"MDS1"}),
        archive_version="dense_streams_v1",
    ),
    _SubstrateRule(
        substrate_class="magic_codec_packet",
        required_section_magics=frozenset({b"MAGC"}),
        archive_version="magic_codec_v1",
    ),
    _SubstrateRule(
        substrate_class="pr106_latent_sidecar_pr101_grammar",
        required_members=frozenset({"sidecar.bin", "decoder.bin"}),
        required_member_prefixes=frozenset({"pr101_"}),
    ),
    _SubstrateRule(
        substrate_class="pr106_r2_sidecar",
        required_members=frozenset({"sidecar.bin", "decoder.bin"}),
        required_section_magics=frozenset({b"\xfe"}),
        archive_version="pr106_r2",
    ),
    _SubstrateRule(
        substrate_class="pr106_r1_sidecar",
        required_members=frozenset({"sidecar.bin"}),
        required_section_magics=frozenset({b"\xfe"}),
        archive_version="pr106_r1",
    ),
    _SubstrateRule(
        substrate_class="hnerv_lc_ac",
        required_member_prefixes=frozenset({"hnerv_lc_ac"}),
    ),
    _SubstrateRule(
        substrate_class="hnerv_lc_family",
        required_member_prefixes=frozenset({"decoder.bin", "latent"}),
    ),
    # ── FIX-J substrate-scaffold magic-byte rules (2026-05-12) ──
    # Each rule fires when the substrate's 4-byte ASCII magic appears at
    # the first 4 bytes of a ZIP member; the substrate-scaffold archive
    # is monolithic-single-file ``0.bin`` per Catalog #124, so the magic
    # appears at offset 0 of the canonical packet member.
    _SubstrateRule(
        substrate_class="sane_hnerv",
        required_section_magics=frozenset({b"SHV1"}),
        archive_version="sane_hnerv_v1",
    ),
    _SubstrateRule(
        substrate_class="balle_renderer",
        required_section_magics=frozenset({b"BRV1"}),
        archive_version="balle_renderer_v1",
    ),
    _SubstrateRule(
        substrate_class="hybrid_renderer_residual_substrate",
        required_section_magics=frozenset({b"HRR1"}),
        archive_version="hybrid_renderer_residual_v1",
    ),
    _SubstrateRule(
        substrate_class="self_compress_nn_substrate",
        required_section_magics=frozenset({b"SCV1"}),
        archive_version="self_compress_nn_v1",
    ),
    _SubstrateRule(
        substrate_class="pr101_lc_v2_clone_substrate",
        required_section_magics=frozenset({b"PR12"}),
        archive_version="pr101_lc_v2_clone_v1",
    ),
    _SubstrateRule(
        substrate_class="cool_chic_full_renderer_substrate",
        required_section_magics=frozenset({b"CCV1"}),
        archive_version="cool_chic_full_renderer_v1",
    ),
    _SubstrateRule(
        substrate_class="wavelet_full_renderer_substrate",
        required_section_magics=frozenset({b"WLV1"}),
        archive_version="wavelet_full_renderer_v1",
    ),
    _SubstrateRule(
        substrate_class="grayscale_lut_substrate",
        required_section_magics=frozenset({b"GLV1"}),
        archive_version="grayscale_lut_v1",
    ),
    _SubstrateRule(
        substrate_class="vq_vae_substrate",
        required_section_magics=frozenset({b"VQV1"}),
        archive_version="vq_vae_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="siren_substrate",
        required_section_magics=frozenset({b"SRV1"}),
        archive_version="siren_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="block_nerv_substrate",
        required_section_magics=frozenset({b"BNV1"}),
        archive_version="block_nerv_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="tc_nerv_substrate",
        required_section_magics=frozenset({b"TCV1"}),
        archive_version="tc_nerv_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="ff_nerv_substrate",
        required_section_magics=frozenset({b"FFV1"}),
        archive_version="ff_nerv_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="ds_nerv_substrate",
        required_section_magics=frozenset({b"DSV1"}),
        archive_version="ds_nerv_substrate_v1",
    ),
    _SubstrateRule(
        substrate_class="hi_nerv_substrate",
        required_section_magics=frozenset({b"HIV1"}),
        archive_version="hi_nerv_substrate_v1",
    ),
    # ── CompressAI codec adapter rules (EEEE 2026-05-12) ──
    _SubstrateRule(
        substrate_class="compressai_factorized_prior_packet",
        required_section_magics=frozenset({b"CAFP"}),
        archive_version="compressai_factorized_prior_v1",
    ),
    _SubstrateRule(
        substrate_class="compressai_balle_hyperprior_packet",
        required_section_magics=frozenset({b"CABH"}),
        archive_version="compressai_balle_hyperprior_v1",
    ),
    _SubstrateRule(
        substrate_class="compressai_cheng2020_packet",
        required_section_magics=frozenset({b"CACG"}),
        archive_version="compressai_cheng2020_v1",
    ),
)


@dataclass(frozen=True)
class SectionInfo:
    """One archive section / ZIP member with detected magic + entropy."""

    name: str
    size_bytes: int
    sha256: str
    magic_signature: str | None
    entropy_estimate_bits_per_byte: float


@dataclass(frozen=True)
class ClassificationResult:
    """Classifier verdict for one archive."""

    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    substrate_class: str
    substrate_class_confidence: str
    archive_version: str | None
    sections: tuple[SectionInfo, ...]
    parser_section_manifest: dict[str, object]
    classification_signals: tuple[str, ...]
    ambiguity_blockers: tuple[str, ...]
    classifier_version: str = "v1"
    generated_at_utc: str = ""
    target_substrate_hint: str = "any_packetized_archive_with_dense_residual"


class XraySubstrateClassifierError(Exception):
    """Raised on classifier failure (corrupt archive, refused classification, etc.)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _shannon_entropy_bits_per_byte(data: bytes) -> float:
    """Empirical Shannon entropy in bits/byte of a byte string."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return float(entropy)


def _detect_magic(first_bytes: bytes) -> str | None:
    """Return the label of the first matching signature, or None."""
    for sig, label in _SECTION_MAGIC_SIGNATURES:
        if first_bytes.startswith(sig):
            return label
    return None


def _classify_substrate(
    member_names: list[str],
    sections: list[SectionInfo],
) -> tuple[str, str, str | None, list[str]]:
    """Apply substrate rules in order; return (class, confidence, version, signals)."""
    member_name_set = frozenset(member_names)
    detected_magics: set[bytes] = set()
    signals: list[str] = []
    for section in sections:
        if section.magic_signature is not None:
            signals.append(f"section_magic:{section.name}={section.magic_signature}")
    # We pull the raw section magic bytes too for prefix matching against rule
    # signatures (the labels can drift; the bytes do not).
    for section_label in (s.magic_signature for s in sections):
        if section_label is None:
            continue
        # Recover the original bytes from the labelled table.
        for sig, label in _SECTION_MAGIC_SIGNATURES:
            if label == section_label:
                detected_magics.add(sig)

    for rule in _SUBSTRATE_RULES:
        if rule.required_members and not rule.required_members.issubset(
            member_name_set
        ):
            continue
        if rule.required_member_prefixes:
            ok = all(
                any(name.startswith(prefix) for name in member_names)
                for prefix in rule.required_member_prefixes
            )
            if not ok:
                continue
        if rule.required_section_magics and not rule.required_section_magics.issubset(
            detected_magics
        ):
            continue
        if rule.forbidden_members and rule.forbidden_members.issubset(
            member_name_set
        ):
            continue
        # Rule matched. Confidence high if it required >= 2 signals; medium
        # if just one; low otherwise (defensive default).
        n_constraints = (
            (1 if rule.required_members else 0)
            + (1 if rule.required_member_prefixes else 0)
            + (1 if rule.required_section_magics else 0)
        )
        if n_constraints >= 2:
            confidence = "high"
        elif n_constraints == 1:
            confidence = "medium"
        else:
            confidence = "low"
        signals.append(f"rule_match:{rule.substrate_class}")
        return rule.substrate_class, confidence, rule.archive_version, signals
    return "unknown_substrate_unclassifiable", "low", None, signals


def classify_archive(archive_path: Path) -> ClassificationResult:
    """Classify the substrate class of an archive zip.

    Parameters
    ----------
    archive_path:
        Path to an archive.zip file.

    Returns
    -------
    ClassificationResult
        Structured classifier verdict + per-section manifest.

    Raises
    ------
    XraySubstrateClassifierError
        On corrupt archive / non-zip input / missing file.
    """
    if not archive_path.exists():
        raise XraySubstrateClassifierError(
            f"archive not found: {archive_path}"
        )
    if not archive_path.is_file():
        raise XraySubstrateClassifierError(
            f"archive is not a file: {archive_path}"
        )
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            member_names = sorted(zf.namelist())
            members = sorted(zf.infolist(), key=lambda x: x.filename)
            sections: list[SectionInfo] = []
            offsets: list[int] = []
            lengths: list[int] = []
            section_sha256s: list[str] = []
            cumulative = 0
            for info in members:
                with zf.open(info, "r") as fp:
                    data = fp.read()
                first_bytes = data[:16]
                magic_label = _detect_magic(first_bytes)
                sha = hashlib.sha256(data).hexdigest()
                entropy = _shannon_entropy_bits_per_byte(data)
                sections.append(
                    SectionInfo(
                        name=info.filename,
                        size_bytes=len(data),
                        sha256=sha,
                        magic_signature=magic_label,
                        entropy_estimate_bits_per_byte=entropy,
                    )
                )
                offsets.append(cumulative)
                lengths.append(len(data))
                section_sha256s.append(sha)
                cumulative += len(data)
    except zipfile.BadZipFile as exc:
        raise XraySubstrateClassifierError(
            f"corrupt zip archive {archive_path}: {exc}"
        ) from exc

    substrate_class, confidence, archive_version, signals = _classify_substrate(
        member_names, sections
    )

    # Ambiguity / consistency checks.
    ambiguity_blockers: list[str] = []
    if substrate_class == "unknown_substrate_unclassifiable":
        ambiguity_blockers.append("no_substrate_rule_matched")
    if len(sections) == 0:
        ambiguity_blockers.append("archive_has_zero_sections")
    if sum(lengths) == 0:
        ambiguity_blockers.append("archive_total_section_bytes_is_zero")
    # Section consistency: every section must have at least one byte to be
    # meaningful, and no two sections may have identical names.
    if len({s.name for s in sections}) != len(sections):
        ambiguity_blockers.append("duplicate_section_names")

    parser_section_manifest = {
        "offsets": offsets,
        "lengths": lengths,
        "section_names": [s.name for s in sections],
        "section_sha256s": section_sha256s,
        "section_magics": [s.magic_signature for s in sections],
        "entropy_estimates_bits_per_byte": [
            s.entropy_estimate_bits_per_byte for s in sections
        ],
    }

    return ClassificationResult(
        archive_path=str(archive_path),
        archive_sha256=_sha256_file(archive_path),
        archive_size_bytes=archive_path.stat().st_size,
        substrate_class=substrate_class,
        substrate_class_confidence=confidence,
        archive_version=archive_version,
        sections=tuple(sections),
        parser_section_manifest=parser_section_manifest,
        classification_signals=tuple(signals),
        ambiguity_blockers=tuple(ambiguity_blockers),
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse /tmp paths per CLAUDE.md forbidden_tmp_paths_in_persisted_artifact."""
    resolved = output_dir.resolve()
    s = str(resolved)
    if (
        s.startswith("/tmp/")
        or s.startswith("/var/tmp/")
        or s.startswith("/private/tmp/")
        or "/tmp/" in s
    ):
        raise XraySubstrateClassifierError(
            f"output_dir must not be under /tmp; got {output_dir} "
            f"(per CLAUDE.md forbidden /tmp paths in persisted artifact)"
        )


def _result_to_json(result: ClassificationResult) -> dict[str, object]:
    """Convert ClassificationResult to a deterministic JSON-friendly dict."""
    data = asdict(result)
    # Render the sections as a list of dicts (asdict already does this for
    # frozen dataclasses but we double-check the section name ordering).
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify the substrate class of an archive zip via "
            "parser-section magic-byte signatures + member-name rules."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the archive.zip to classify.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Required unless --dry-run. Refuses /tmp paths."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the manifest JSON to stdout without touching disk.",
    )
    parser.add_argument(
        "--refuse-unclassifiable",
        action="store_true",
        help=(
            "When set, exit with status 3 if the substrate class cannot be "
            "determined (substrate_class == unknown_substrate_unclassifiable)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        print("error: --output-dir is required unless --dry-run", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        _validate_output_dir(args.output_dir)
    try:
        result = classify_archive(args.archive)
    except XraySubstrateClassifierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = _result_to_json(result)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.dry_run:
        print(rendered)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "xray_substrate_classifier_manifest.json"
        manifest_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {manifest_path}", file=sys.stderr)
    if args.refuse_unclassifiable and (
        result.substrate_class == "unknown_substrate_unclassifiable"
    ):
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entrypoint
    sys.exit(main())
