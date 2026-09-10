#!/usr/bin/env python3
"""ddm_rp1 rebase -- re-verify an acceptance on the base it will actually SHIP against.

THE DEFECT THIS EXISTS TO REPAIR
--------------------------------
``ddm_rp1_sizing.py`` takes its ranking from ``--rank-dir`` and its BASE FIELD from
``rp1.load_live_field()``.  The first is a flag; the second is a source constant pinned to
sj1's pass-4 npz.  Round 2 ranked on ``field_move40.u8`` and accepted on the pass-4 field,
so every "argmax identical on all 196,608 cells" verdict the n600 shards emitted was
measured against a body the pointer no longer ships.  The shard receipts say so in the open
-- ``tokens_changed_vs_pristine`` reads 9,209 + the shard's own edits, and 9,209 is pass-4's
count, not move 40's.  Nothing warned, because a flag and a constant can disagree silently.

WHAT SURVIVES THE MISMATCH, AND WHY
-----------------------------------
The rank masks out every position where the live field differs from the pristine base
(``sj1_edit_mask = target != base``, computed on the OVERRIDE field), so no proposal ever
lands on a banked token.  At every proposed position the two bases therefore carry the SAME
symbol, and the edit ``(pos, sym -> best)`` is the identical edit on either body.  That is
CHECKED here per edit, never assumed: an edit whose base symbol is not the ``sym`` the rank
recorded is refused.

What does NOT survive is the CONTEXT.  The renderer's receptive field is 9 token cells, so
a banked edit anywhere near a proposal changes both the base argmax and the edited argmax.
So the joint neutrality is re-measured here, on the shipping base, with the same cumulative
bisection the acceptance used -- a set that fails is narrowed rather than dropped whole.

THE OUTPUT CARRIES ALL 600 PLANES
---------------------------------
A pair merely ABSENT from an edits npz reverts to the pristine base and silently undoes
every edit the pointer banked.  So the emitted field is written from the BASE field for
every untouched pair, and the token count is checked against the accepted edit count.

``[macOS-CPU advisory, jg1 instrument, DALI GT lineage]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1
import ddm_rp1_rate_rank as rp1
import ddm_rp1_sizing as sizing

N_PAIRS = jg1.N_PAIRS
EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_base_field(path: Path, expect_sha: str) -> np.ndarray:
    """The field the candidate will be built on, bound by its own sha."""
    observed = sha256_file(path)
    if observed != expect_sha:
        raise rp1.Rp1Error(
            f"base field {path} has sha {observed}, expected {expect_sha}; a rebase onto "
            "an unverified body is the defect this module repairs, not a repair of it"
        )
    expected_bytes = N_PAIRS * EVAL_H * EVAL_W
    if Path(path).stat().st_size != expected_bytes:
        raise rp1.Rp1Error(
            f"base field {path} is {Path(path).stat().st_size} B, expected {expected_bytes}"
        )
    return np.memmap(
        path, dtype=np.uint8, mode="r", shape=(N_PAIRS, EVAL_H, EVAL_W)
    )


def load_accepted(paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    """pair -> accepted edits, each carrying the ``sym`` the rank measured it against.

    ``accepted`` rows carry only ``pos``/``best``; the ``sym`` that binds the edit to a
    base symbol lives in the ``tested`` list, so the two are joined here rather than the
    base symbol being taken on trust.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pair = int(row["pair"])
            accepted = row.get("accepted") or []
            if not accepted:
                continue
            by_key = {
                (int(t["pos"]), int(t["best"])): t for t in row.get("tested", [])
            }
            edits: list[dict[str, Any]] = []
            for entry in accepted:
                key = (int(entry["pos"]), int(entry["best"]))
                tested = by_key.get(key)
                if tested is None:
                    raise rp1.Rp1Error(
                        f"pair {pair}: accepted edit {key} has no tested row, so the "
                        "base symbol it was ranked against is unknown"
                    )
                edits.append(
                    {
                        "pos": key[0],
                        "best": key[1],
                        "sym": int(tested["sym"]),
                        "saving_bits": float(tested["saving_bits"]),
                    }
                )
            if pair in out:
                raise rp1.Rp1Error(f"pair {pair} appears in two row files")
            out[pair] = edits
    return out


def cmd_rebase(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pointer = rp1.verify_pointer(expect_sha=args.expect_pointer_sha)
    if not pointer["matches_expected"]:
        raise rp1.Rp1Error(
            f"pointer file reads {pointer['archive_sha256']}, this rebase targets "
            f"{args.expect_pointer_sha}"
        )

    base_field = load_base_field(Path(args.base_field_u8), args.expect_field_sha256)
    accepted = load_accepted([Path(p) for p in args.rows])

    # The instrument also carries the field the acceptance ACTUALLY ran on, which is what
    # makes the exposure a measurement here rather than an estimate.
    inst = sizing.Instrument(threads=args.threads)
    prior_field = inst.live_field
    inst.live_field = np.asarray(base_field)

    pairs = sorted(accepted)
    if args.shards > 1:
        pairs = pairs[args.shard :: args.shards]

    rows_path = out / f"REBASE_ROWS_{args.shard}.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    elif rows_path.exists():
        rows_path.write_text("")

    started = time.perf_counter()
    for n, pair in enumerate(pairs):
        if pair in done:
            continue
        pair_started = time.perf_counter()
        plane = np.array(base_field[pair], dtype=np.uint8)
        flat = plane.reshape(-1)
        prior_plane = np.asarray(prior_field[pair])
        context_cells_differ = int((plane != prior_plane).sum())

        edits = accepted[pair]
        mismatched = [
            e for e in edits if int(flat[int(e["pos"])]) != int(e["sym"])
        ]
        if mismatched:
            raise rp1.Rp1Error(
                f"pair {pair}: {len(mismatched)} accepted edits sit on positions whose "
                "base symbol is not the symbol the rank priced them against; these are "
                "not the same edits on this body"
            )

        base_argmax = inst.argmax_batch([plane], pair)[0]
        proposals = [(int(e["pos"]), int(e["best"])) for e in edits]
        kept, stats = sizing.accept_by_bisection(
            inst, pair, plane, base_argmax, proposals, max_verifies=args.max_verifies
        )
        kept_set = set(kept)
        bits_kept = float(
            sum(e["saving_bits"] for e in edits if (e["pos"], e["best"]) in kept_set)
        )
        bits_all = float(sum(e["saving_bits"] for e in edits))
        row = {
            "pair": pair,
            # The base this row was VERIFIED on travels with the row, so a merge cannot
            # assemble one field out of rows certified against two different bodies --
            # which is the exact genus of defect this module exists to repair.
            "base_field_sha256": args.expect_field_sha256,
            "context_cells_differ_vs_acceptance_base": context_cells_differ,
            "proposed": len(edits),
            "kept": len(kept),
            "accepted": [{"pos": p, "best": b} for p, b in kept],
            "transferred_whole": len(kept) == len(edits),
            "saving_bits_first_order_kept": bits_kept,
            "saving_bits_first_order_proposed": bits_all,
            "verifies": stats["verifies"],
            "hit_cap": stats["hit_cap"],
            "seconds": time.perf_counter() - pair_started,
        }
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if (n + 1) % 10 == 0 or n + 1 == len(pairs):
            print(
                json.dumps(
                    {
                        "done": n + 1,
                        "of": len(pairs),
                        "pair": pair,
                        "kept": len(kept),
                        "of_proposed": len(edits),
                        "s_per_pair": (time.perf_counter() - started) / (n + 1),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"shard": args.shard, "pairs": len(pairs), "rows": str(rows_path)}))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge rebase shards into ONE 600-plane field plus its receipt."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_field = load_base_field(Path(args.base_field_u8), args.expect_field_sha256)

    rows: dict[int, dict[str, Any]] = {}
    for path in args.rows:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pair = int(row["pair"])
            if pair in rows:
                raise rp1.Rp1Error(f"pair {pair} appears in two rebase shards")
            if row.get("base_field_sha256") != args.expect_field_sha256:
                raise rp1.Rp1Error(
                    f"pair {pair} was verified against base "
                    f"{row.get('base_field_sha256')}, but this merge builds on "
                    f"{args.expect_field_sha256}; refusing to assemble one field out of "
                    "rows certified against two bodies"
                )
            rows[pair] = row

    # A MISSING SHARD IS INVISIBLE OTHERWISE.  Merging four shards of five produces a
    # perfectly valid field carrying four fifths of the edits, which then gets priced and
    # sealed as if it were the whole pass.  So the pairs that OWED a rebase row are
    # recomputed from the acceptance itself and every one is required to have produced one.
    if args.sizing_rows:
        owed = set(load_accepted([Path(p) for p in args.sizing_rows]))
        missing = sorted(owed - set(rows))
        extra = sorted(set(rows) - owed)
        if missing or extra:
            raise rp1.Rp1Error(
                f"{len(missing)} pairs with accepted edits have no rebase row "
                f"(first: {missing[:8]}) and {len(extra)} rebase rows have no accepted "
                f"edits (first: {extra[:8]}); a partial merge ships a field that prices "
                "as if it were the whole pass"
            )

    field = np.array(base_field, dtype=np.uint8)
    tokens = 0
    for pair, row in rows.items():
        flat = field[pair].reshape(-1)
        for edit in row["accepted"]:
            flat[int(edit["pos"])] = np.uint8(int(edit["best"]))
        tokens += len(row["accepted"])
    changed = int((field != np.asarray(base_field)).sum())
    if changed != tokens:
        raise rp1.Rp1Error(
            f"rebased field differs from its base at {changed} tokens but {tokens} edits "
            "were kept; an edit collided with another or with the base"
        )
    field_path = out / "field_rebased.npz"
    np.savez_compressed(field_path, **{str(p): field[p] for p in range(N_PAIRS)})

    proposed = sum(r["proposed"] for r in rows.values())
    whole = sum(1 for r in rows.values() if r["transferred_whole"])
    exposed = [r for r in rows.values() if r["context_cells_differ_vs_acceptance_base"]]
    exposed_kept = sum(r["kept"] for r in exposed)
    exposed_proposed = sum(r["proposed"] for r in exposed)
    clean = [r for r in rows.values() if not r["context_cells_differ_vs_acceptance_base"]]
    clean_kept = sum(r["kept"] for r in clean)
    clean_proposed = sum(r["proposed"] for r in clean)
    verdict = {
        "schema": "ddm_rp1_rebase.v1",
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
        "priced_by": "first_order_ranking_only -- the bytes are a REAL encode's job",
        "base_field_u8": str(args.base_field_u8),
        "base_field_sha256": args.expect_field_sha256,
        "pairs_with_edits": len(rows),
        "pairs_transferred_whole": whole,
        "tokens_proposed": proposed,
        "tokens_kept": tokens,
        "token_transfer_fraction": (tokens / proposed) if proposed else 0.0,
        "saving_bits_first_order_kept": float(
            sum(r["saving_bits_first_order_kept"] for r in rows.values())
        ),
        "saving_bits_first_order_proposed": float(
            sum(r["saving_bits_first_order_proposed"] for r in rows.values())
        ),
        "exposure": {
            "note": (
                "pairs whose base plane differs from the plane the acceptance ran on; "
                "the CONTROL group is the pairs where the two bases agree, whose "
                "transfer fraction should be 1.0 if the instrument is sound"
            ),
            "pairs_context_changed": len(exposed),
            "pairs_context_identical": len(clean),
            "context_changed_tokens_proposed": exposed_proposed,
            "context_changed_tokens_kept": exposed_kept,
            "context_changed_transfer_fraction": (
                exposed_kept / exposed_proposed if exposed_proposed else None
            ),
            "context_identical_tokens_proposed": clean_proposed,
            "context_identical_tokens_kept": clean_kept,
            "context_identical_transfer_fraction": (
                clean_kept / clean_proposed if clean_proposed else None
            ),
        },
        "hit_cap_pairs": sorted(p for p, r in rows.items() if r["hit_cap"]),
        "field_rebased": {
            "path": str(field_path),
            "sha256": sha256_file(field_path),
            "planes": N_PAIRS,
            "tokens_changed_vs_base": changed,
        },
    }
    (out / "REBASE.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    (out / "kept_pairs.json").write_text(
        json.dumps(sorted(p for p, r in rows.items() if r["accepted"]))
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


def cmd_verify_field(args: argparse.Namespace) -> int:
    """Prove the raw base field and a named npz field are the SAME 600 planes.

    The base overlay this arm reuses was rendered from an npz; the rebase edits a raw u8.
    Two files in two formats cannot be compared by sha, so if they were ever allowed to
    drift the pose leg would be measured on renders of one body and the rate leg on the
    other -- a mismatch with exactly the shape of the one this module already repairs.
    """
    base_field = load_base_field(Path(args.base_field_u8), args.expect_field_sha256)
    npz_sha = sha256_file(Path(args.field_npz))
    if args.expect_npz_sha256 and npz_sha != args.expect_npz_sha256:
        raise rp1.Rp1Error(
            f"npz field sha {npz_sha} != expected {args.expect_npz_sha256}"
        )
    differing: list[int] = []
    cells = 0
    with np.load(args.field_npz, allow_pickle=False) as blob:
        if sorted(int(k) for k in blob.files) != list(range(N_PAIRS)):
            raise rp1.Rp1Error(
                f"npz field carries {len(blob.files)} planes whose keys are not exactly "
                f"0..{N_PAIRS - 1}; a partial or re-keyed field is not this body"
            )
        for key in blob.files:
            pair = int(key)
            plane = np.asarray(blob[key], dtype=np.uint8)
            delta = int((plane != np.asarray(base_field[pair])).sum())
            if delta:
                differing.append(pair)
                cells += delta
    verdict = {
        "schema": "ddm_rp1_field_identity.v1",
        "axis": "[scorer-free EXACT byte comparison]",
        "score_claim": False,
        "base_field_u8": str(args.base_field_u8),
        "base_field_sha256": args.expect_field_sha256,
        "field_npz": str(args.field_npz),
        "field_npz_sha256": npz_sha,
        "pairs_differing": differing,
        "cells_differing": cells,
        "identical": not differing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if not verdict["identical"]:
        raise rp1.Rp1Error(
            f"the raw base field and {args.field_npz} differ at {cells} cells over "
            f"{len(differing)} pairs; they are not the same body"
        )
    return 0


def _pointer_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-field-u8", required=True)
    parser.add_argument("--expect-field-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    rebase = sub.add_parser("rebase", help="re-verify an acceptance on a named base field")
    rebase.add_argument("--rows", nargs="+", required=True)
    rebase.add_argument("--out-dir", required=True)
    rebase.add_argument("--expect-pointer-sha", required=True)
    rebase.add_argument("--threads", type=int, default=2)
    rebase.add_argument("--shards", type=int, default=1)
    rebase.add_argument("--shard", type=int, default=0)
    rebase.add_argument("--max-verifies", type=int, default=400)
    rebase.add_argument("--resume", action="store_true")
    _pointer_flags(rebase)
    rebase.set_defaults(func=cmd_rebase)

    merge = sub.add_parser("merge", help="merge rebase shards into one 600-plane field")
    merge.add_argument("--rows", nargs="+", required=True)
    merge.add_argument("--out-dir", required=True)
    merge.add_argument(
        "--sizing-rows",
        nargs="*",
        default=[],
        help="the acceptance's own SIZING_ROWS.jsonl files; when given, every pair that "
        "carried an accepted edit must have produced a rebase row, so a shard that died "
        "cannot be merged into a field that prices as if it were the whole pass",
    )
    _pointer_flags(merge)
    merge.set_defaults(func=cmd_merge)

    verify = sub.add_parser(
        "verify-field", help="prove a raw u8 field and an npz field are the same planes"
    )
    verify.add_argument("--field-npz", required=True)
    verify.add_argument("--expect-npz-sha256", default=None)
    verify.add_argument("--out", required=True)
    _pointer_flags(verify)
    verify.set_defaults(func=cmd_verify_field)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
