#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_ep2 — compile a REAL ``archive.zip`` for each burn-4 endpoint candidate.

WHY: every burn-4 composed-S figure in the campaign is priced on
``counted_bytes_ledger``'s ``total_counted_bytes`` — a per-stream PRICE MODEL, not the bytes
``upstream/evaluate.py:63`` actually stats.  The rate caveat that has to travel with every one of
those numbers ("counted-payload ledger, NOT archive.zip bytes") exists precisely because nobody
had closed the gap for these checkpoints.  This driver closes it: it compiles each candidate
through ``tac.optimization.ddm_tr1_runtime.compile_archive_from_checkpoint`` -- the eg1/E1
receiver-closed TR1 exporter, which is purpose-built for this vehicle (``variant=lotto``,
``token_temporal_mode=shared_base``) and already handles the QA24 ``token_cell_mask`` burn-4
carries -- and reports the REAL ZIP size beside the ledger estimate.

The exporter self-verifies on the way through (both asserts are inside the library, not here):
``parse_packet -> reemit_packet`` must be byte-identical, and ``parse_archive -> reemit_archive``
must be byte-identical.  A compile that returns at all has therefore already passed parse-back.
This driver adds the checks the library cannot do for itself:

  * an INDEPENDENT re-parse of the emitted bytes (``parse_archive``) plus a per-section ledger,
    so the section decomposition is visible rather than asserted;
  * DETERMINISM: compile twice, require identical SHA-256 (the same check
    ``tools/rehearse_ddm_tr1_runtime.py`` makes; a nondeterministic archive cannot be a row);
  * the ledger-vs-real BYTE GAP, in bytes and in S units, per candidate.

This is an EXPORT + PARSE-BACK step only.  It renders nothing and scores nothing.  It does NOT
produce an evaluator row: only ``upstream/evaluate.py`` on these exact bytes is a score.
``score_claim=False``; pointer 0.1910828242 [contest-CPU] UNMOVED.

Usage:
  ddm_ep2_compile_candidate_archives.py --candidate LABEL=<ckpt> [...] --out-dir <dir>
                                        [--ledger-bytes LABEL=N ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _kv(spec: str, cast=str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected LABEL=VALUE, got {spec!r}")
    label, _, raw = spec.partition("=")
    if not label.strip():
        raise argparse.ArgumentTypeError(f"empty LABEL in {spec!r}")
    return label.strip(), cast(raw.strip())


def _candidate(spec: str):
    label, raw = _kv(spec)
    p = Path(raw).expanduser()
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"--candidate {label}: no such checkpoint: {p}")
    return label, p


def compile_one(label: str, ckpt: Path, out_dir: Path,
                ledger_bytes: int | None) -> dict:
    from tac.contest_score import rate_term
    from tac.optimization.ddm_tr1_runtime import (
        compile_archive_from_checkpoint,
        parse_archive,
        reemit_archive,
        section_ledger,
    )

    row: dict = {"label": label, "checkpoint": str(ckpt)}
    try:
        built = compile_archive_from_checkpoint(ckpt)
        # DETERMINISM: a second independent compile must produce the identical bytes.
        again = compile_archive_from_checkpoint(ckpt)
        row["deterministic_recompile"] = bool(again.archive_sha256 == built.archive_sha256)

        # INDEPENDENT parse-back of the emitted bytes (the library asserted it internally;
        # re-doing it here from the on-disk artifact is the re-derive-don't-recognize step).
        parsed = parse_archive(built.archive_bytes)
        row["parse_back_reemit_byte_identical"] = bool(
            reemit_archive(parsed) == built.archive_bytes)
        row["sections"] = section_ledger(parsed.packet)

        dest = out_dir / label / "archive.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(built.archive_bytes)
        stat_bytes = dest.stat().st_size  # what evaluate.py:63 would stat

        row.update({
            "archive_path": str(dest),
            "archive_sha256": built.archive_sha256,
            "archive_bytes_in_memory": len(built.archive_bytes),
            "archive_bytes_stat": stat_bytes,
            "packet_sha256": built.packet_sha256,
            "packet_bytes": len(built.packet_bytes),
            "checkpoint_sha256": built.checkpoint.sha256,
            "checkpoint_bytes": built.checkpoint.bytes,
            "config_hash": built.checkpoint.config_hash,
            "rate_term_s_real_archive": rate_term(stat_bytes),
        })
        if ledger_bytes is not None:
            row["counted_ledger_bytes"] = int(ledger_bytes)
            row["rate_term_s_counted_ledger"] = rate_term(int(ledger_bytes))
            row["byte_gap_real_minus_ledger"] = stat_bytes - int(ledger_bytes)
            row["rate_s_gap_real_minus_ledger"] = (
                rate_term(stat_bytes) - rate_term(int(ledger_bytes)))
        row["ok"] = True
    except Exception as exc:
        row["ok"] = False
        row["error"] = repr(exc)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=_candidate, action="append", required=True,
                    metavar="LABEL=CKPT")
    ap.add_argument("--ledger-bytes", type=lambda s: _kv(s, int), action="append",
                    default=[], metavar="LABEL=N",
                    help="the campaign counted_bytes_ledger total for that label, for the gap")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    ledger = dict(args.ledger_bytes)
    rows = [compile_one(label, ckpt, args.out_dir, ledger.get(label))
            for label, ckpt in args.candidate]
    for r in rows:
        print(f"[ddm_ep2-compile] {r['label']}: "
              + (r.get("error") or
                 f"archive={r['archive_bytes_stat']}B sha={r['archive_sha256'][:12]} "
                 f"rate_S={r['rate_term_s_real_archive']:.6f} "
                 f"det={r['deterministic_recompile']} "
                 f"parseback={r['parse_back_reemit_byte_identical']}"),
              file=sys.stderr, flush=True)

    out = {
        "schema": "ddm_ep2_candidate_archives.v1",
        "exporter": "tac.optimization.ddm_tr1_runtime.compile_archive_from_checkpoint "
                    "(eg1/E1 receiver-closed TR1 grammar; lotto + shared_base + QA24 cell mask)",
        "what_this_is": "EXPORT + PARSE-BACK ONLY. Real archive.zip bytes and their rate term. "
                        "Nothing here is rendered or scored; only upstream/evaluate.py on these "
                        "exact bytes is a score.",
        "candidates": rows,
        "all_ok": all(r.get("ok") for r in rows),
        "all_deterministic": all(r.get("deterministic_recompile") for r in rows if r.get("ok")),
        "all_parse_back_byte_identical": all(
            r.get("parse_back_reemit_byte_identical") for r in rows if r.get("ok")),
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[byte-closed export; NOT an evaluator row]",
    }
    payload = json.dumps(out, indent=2, sort_keys=True) + "\n"
    dest = args.output_json or (args.out_dir / "ddm_ep2_candidate_archives.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(dest)
    print(payload, end="")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
