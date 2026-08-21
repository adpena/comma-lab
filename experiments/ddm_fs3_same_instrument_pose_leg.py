#!/usr/bin/env python3
"""ddm_fs3 -- price the pose and seg legs as a SAME-INSTRUMENT delta, or refuse.

WHY
---
The mirror candidate's advisory returned ``d_pose = 0.00055125``.  That number is
NOT the pose leg.  The compose-stage figure it would naively be compared against
(``8.20845e-06``) came off a different instrument -- a direct frozen-scorer forward
rather than a full ``inflate.sh`` + ``upstream/evaluate.py`` pass with the CPU GT
lineage -- and the units x level x aggregation law forbids differencing across
them ([[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

The measurable leg is ``d(candidate) - d(base)`` where BOTH sides come from the
same instrument tuple.  This module computes that delta and **REFUSES** unless the
forward-relevant fields of ``instrument_tuple`` are identical on both receipts.

THE ONE FIELD ALLOWED TO DIFFER, AND WHY
----------------------------------------
``instrument_tuple.code.pact_commit`` records the repo HEAD at run time.  Two
receipts fired minutes apart in a live tree will differ there even when nothing on
the forward path moved -- sister arms land commits continuously.  So a strict
whole-tuple equality would refuse every delta this repo can actually produce.

The cure is not to relax the check but to make it PRECISE. **Two fields differ on
every honest delta this vehicle can produce, and each gets its own proof.**

**HARNESS fields -- must be byte-identical, no exceptions:**

* ``upstream_snapshot_sha256``    (the scorer + evaluate.py snapshot)
* ``upstream_evaluate_py.sha256`` (the scoring script itself)
* ``inflate_script_sha256``       (the decode entry point)
* device, torch version, GT lineage, sample count

**``runtime_files_sha256`` -- ALWAYS differs, and is TREATMENT not instrument.**
A candidate runtime tree EMBEDS its own ``archive.zip`` and the receiver pin
derived from it, so this hash cannot match between a base and a candidate.
Refusing on it would refuse every real delta; waving it through would let a
genuine receiver change ride in disguised as payload.  So it is proved
STRUCTURALLY instead: ``runtime_trees_differ_only_by_payload`` requires the two
trees to differ in NOTHING except ``archive.zip`` and ``inflate.py``, and requires
``inflate.py`` to be AST-identical once its two pin constants are normalised.  Passing
the trees is MANDATORY -- omitting them is a refusal, not a skip.

**``pact_commit`` -- may differ ONLY with ``--commit-delta-proof``**, a git range
whose changed files this module re-checks against the forward-path prefixes
itself.  A range touching ``upstream/``, the runtime, or the eval harness is
REFUSED.  The caller supplies evidence, never a conclusion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

#: Any changed path under these prefixes can move the forward, so a commit range
#: touching them cannot be waived.
FORWARD_PATH_PREFIXES = (
    "upstream/",
    "runtime-rs/",
    "experiments/contest_auth_eval.py",
    "src/tac/scorer",
    "src/tac/gt_lineage",
    "src/tac/frame_utils",
    "src/tac/differentiable_eval_roundtrip",
)

S_PER_ARCHIVE_BYTE = 25.0 / 37_545_489


class PoseLegError(RuntimeError):
    """Fail-closed error."""


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def forward_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    """The HARNESS fields -- the scorer forward, which must be identical.

    ``runtime_files_sha256`` is deliberately NOT here.  A candidate runtime tree
    EMBEDS its own ``archive.zip`` and the receiver pin derived from it, so that
    hash differs between base and candidate BY CONSTRUCTION -- it is the TREATMENT,
    not the instrument.  Refusing on it would refuse every delta this vehicle can
    produce, and waving it through unexamined would let a genuine receiver change
    ride in disguised as payload.

    So it is checked SEPARATELY and STRUCTURALLY by ``runtime_trees_differ_only_by_payload``:
    the two trees must differ in NOTHING except ``archive.zip`` and ``inflate.py``,
    and ``inflate.py`` must differ ONLY in its two pin constants.
    """
    tup = receipt.get("instrument_tuple") or {}
    prov = receipt.get("provenance") or {}
    return {
        "upstream_snapshot_sha256": _dig(tup, "code", "upstream_snapshot_sha256"),
        "upstream_evaluate_py_sha256": _dig(
            tup, "code", "upstream_evaluate_py", "sha256"
        ),
        "inflate_script_sha256": _dig(tup, "code", "inflate_script_sha256"),
        "device": prov.get("device"),
        "torch_version": prov.get("torch_version"),
        "gt_lineage": _dig(receipt, "gt_lineage", "lineage"),
        "n_samples": receipt.get("n_samples"),
    }


#: Relative paths -- matched EXACTLY, not by basename.  ``lib/inflate.py`` is not
#: a payload file just because ``inflate.py`` is (rv17 wave-3 W3-F17).
PAYLOAD_FILES = {"archive.zip", "inflate.py"}
PIN_CONSTANTS = ("ARCHIVE_SHA256", "ARCHIVE_BYTES")


def _file_hashes(root: Path) -> dict[str, str]:
    import hashlib

    out: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name.startswith("._"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        out[str(path.relative_to(root))] = digest.hexdigest()
    return out


def _pin_free_ast(path: Path) -> str:
    """``inflate.py``'s AST with the pin constants normalised.

    Compared as an AST, not as text.  Text equality is the WRONG notion of "the
    same receiver": it fails on whitespace that cannot change behaviour, and this
    check found exactly that -- two blank lines eaten by a re-pin regex whose
    ``\\s*$`` matched newlines under ``re.M``.  Refusing a delta over blank lines
    would be a false refusal; accepting a real code change would be a false pass.
    The AST is the object that decides which one a diff is.

    The normaliser REFUSES unless the pin's right-hand side is a single literal
    (rv17 wave-3 W3-F16).  It replaces ``node.value`` -- the WHOLE right-hand
    expression -- so an arbitrary expression bound to a pin name would be erased
    along with whatever it does.  ``ARCHIVE_SHA256 = (globals().__setitem__(
    "DECODE_MODE", "ALT_RECEIVER"), "abc")[1]`` normalises to exactly the same
    dump as a plain literal, and the comparator would report ``PAYLOAD_ONLY`` for
    two files with different executed semantics.  A literal cannot do that, so
    requiring a literal closes the hole at its root instead of blacklisting the
    expression forms anyone has thought of so far.
    """
    import ast as _ast

    tree = _ast.parse(path.read_text())
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Assign):
            continue
        pinned = [
            t.id
            for t in node.targets
            if isinstance(t, _ast.Name) and t.id in PIN_CONSTANTS
        ]
        if not pinned:
            continue
        value = node.value
        # ``type(...) in`` and not isinstance: bool subclasses int, and a pin is
        # never a bool.  The measured pins are ARCHIVE_SHA256 (str) and
        # ARCHIVE_BYTES (int).
        if not isinstance(value, _ast.Constant) or type(value.value) not in (str, int):
            raise PoseLegError(
                f"REFUSING to normalise {path}: pin {'/'.join(pinned)} is assigned "
                f"a {type(value).__name__}, not a plain str/int literal. "
                "Normalisation replaces the ENTIRE right-hand expression, so a "
                "non-literal RHS would be erased together with any behaviour it "
                "carries, and this comparator would then call two different "
                "receivers identical."
            )
        node.value = _ast.Constant(value="<PIN>")
    return _ast.dump(tree)


def runtime_trees_differ_only_by_payload(base: Path, cand: Path) -> dict[str, Any]:
    """Prove the runtime_files_sha difference is PAYLOAD and not a receiver change.

    Three fail-opens closed after rv17 wave-3 (W3-F16/F17/F18).  Each returned
    this module's STRONGEST verdict, ``PAYLOAD_ONLY``, on input that had not been
    scrutinised at all:

    * an empty or nonexistent tree made every set comparison vacuously true, so a
      typo'd ``--base-runtime`` read as proof (F18);
    * ``unexpected`` filtered on BASENAME while the AST check keyed on the exact
      path ``"inflate.py"``, so a differing ``lib/inflate.py`` satisfied the
      filter AND missed the AST check -- zero scrutiny, no adversary required, a
      vendored or backup copy is enough (F17);
    * the pin normaliser erased arbitrary right-hand expressions (F16, cured in
      ``_pin_free_ast``).

    Both path tests now key on the exact relative path, and EVERY differing file
    whose basename is ``inflate.py`` is AST-checked, so "the check never ran"
    can no longer coexist with "an inflate.py differs".
    """
    for label, root in (("--base-runtime", base), ("--candidate-runtime", cand)):
        if not root.is_dir():
            raise PoseLegError(
                f"REFUSING: {label} is not a directory: {root}. A missing tree "
                "compares equal to everything and would return PAYLOAD_ONLY."
            )

    hb, hc = _file_hashes(base), _file_hashes(cand)
    for label, root, hashes in (
        ("--base-runtime", base, hb),
        ("--candidate-runtime", cand, hc),
    ):
        if not hashes:
            raise PoseLegError(
                f"REFUSING: {label} contains no files: {root}. An empty file set "
                "makes every comparison below vacuously true, so this module "
                "would report PAYLOAD_ONLY -- its strongest verdict -- having "
                "examined nothing."
            )

    only_base = sorted(set(hb) - set(hc))
    only_cand = sorted(set(hc) - set(hb))
    differing = sorted(k for k in set(hb) & set(hc) if hb[k] != hc[k])
    # EXACT relative path, not basename: a differing ``lib/inflate.py`` is NOT a
    # payload file just because something called ``inflate.py`` is.
    unexpected = [k for k in differing if k not in PAYLOAD_FILES]

    # Every differing inflate.py, wherever it sits -- keyed the same way the
    # filter above is, so the two can no longer disagree about what a path is.
    inflate_paths = [k for k in differing if Path(k).name == "inflate.py"]
    inflate_body_identical: bool | None = None
    if inflate_paths:
        inflate_body_identical = all(
            _pin_free_ast(base / rel) == _pin_free_ast(cand / rel)
            for rel in inflate_paths
        )

    ok = (
        not only_base
        and not only_cand
        and not unexpected
        # `is True`, never `is not False`: None means the check did not run. It
        # is only legitimate when nothing named inflate.py differs at all.
        and (not inflate_paths or inflate_body_identical is True)
    )

    # OBSERVABILITY ONLY -- deliberately NOT a refusal here (ddm_fc3, task #1179).
    # "Was a treatment applied at all?" is the CALLER's question, and refusing it
    # here would break the pre-registered control that comparing a tree with
    # itself must still PASS.  We surface the facts this function already
    # computed so the caller can decide from the SAME hash pass rather than
    # re-hashing and risking two instruments that disagree.
    archive_paths = sorted(
        k for k in set(hb) & set(hc) if Path(k).name == "archive.zip"
    )
    archive_differs: bool | None = (
        any(hb[k] != hc[k] for k in archive_paths) if archive_paths else None
    )
    return {
        "files_only_in_base": only_base,
        "files_only_in_candidate": only_cand,
        "differing_files": differing,
        "unexpected_differing_files": unexpected,
        "inflate_py_AST_identical_once_pin_normalised": inflate_body_identical,
        "inflate_py_paths_ast_checked": inflate_paths,
        "files_compared": len(set(hb) & set(hc)),
        # Observability, not a refusal: comparing a tree with ITSELF is honestly
        # "identical", but it proves no treatment was applied. The caller owns
        # that question; this flag stops the vacuity from being invisible.
        "base_and_candidate_are_the_same_directory": base.resolve() == cand.resolve(),
        # Same boundary as the flag above: facts, not a verdict. ``None`` means no
        # archive.zip exists at a path present in BOTH trees -- which the caller
        # must treat as fail-closed, never as "nothing to check".
        "archive_zip_relpaths_in_both": archive_paths,
        "archive_zip_differs": archive_differs,
        "verdict": "PAYLOAD_ONLY" if ok else "RECEIVER_CHANGED",
        "why_this_check_exists": (
            "runtime_files_sha256 differs between any base and candidate because each "
            "tree embeds its own archive and the pin derived from it. That is the "
            "TREATMENT. This check proves nothing ELSE moved, so the hash difference "
            "cannot hide a receiver change."
        ),
    }


def assert_treatment_was_applied(
    tree_check: dict[str, Any], *, allow_identical_archive: bool = False
) -> dict[str, Any]:
    """CALLER-layer check: refuse a candidate-vs-base delta when no treatment exists.

    The division of labour, made explicit because ddm_fc2 deliberately left this gap open
    (rv17 W3-F18 note 1).  ``runtime_trees_differ_only_by_payload`` answers exactly one
    question -- *do these two trees differ ONLY by their payload?* -- and comparing a tree
    with itself is an honest ``PAYLOAD_ONLY`` for that question, which is why a
    pre-registered control requires it to keep PASSING there.  It is NOT an answer to *was
    a treatment applied at all?*, and that second question belongs here, at the layer that
    consumes the verdict to compute a candidate-vs-base seg/pose/rate delta.

    Two refusals, both fail-closed:

    * **no ``archive.zip`` in both trees** -> refuse.  A missing payload file is the
      ``VACUITY==PASS`` shape: the comparator would find nothing to disagree about and
      report its strongest verdict having compared no payload.  ``None`` here means "not
      found", never "nothing to check".
    * **byte-identical ``archive.zip``** -> refuse unless the caller explicitly opted in.
      Identical payload means the two receipts describe the SAME shipped bytes, so any
      nonzero leg computed from them is an instrument artifact, not a treatment effect.

    ``allow_identical_archive`` exists for the deliberate null control (running the pair
    against itself to show the legs come out at zero).  It is off by default so the null
    case must be *declared*, never stumbled into.
    """

    relpaths = tree_check.get("archive_zip_relpaths_in_both") or []
    differs = tree_check.get("archive_zip_differs")
    if not relpaths:
        raise PoseLegError(
            "REFUSING: no archive.zip exists at a path present in BOTH runtime trees "
            f"(base_only={tree_check.get('files_only_in_base')}, "
            f"candidate_only={tree_check.get('files_only_in_candidate')}). The payload is "
            "the treatment; with no payload to compare, a PAYLOAD_ONLY verdict has "
            "examined nothing that matters and the legs below measure nothing."
        )
    if differs is not True and not allow_identical_archive:
        raise PoseLegError(
            f"REFUSING: base and candidate archive.zip are BYTE-IDENTICAL at {relpaths} "
            f"(same_directory={tree_check.get('base_and_candidate_are_the_same_directory')}). "
            "The two receipts therefore describe the SAME shipped bytes, so any nonzero "
            "seg/pose/rate leg is an instrument artifact rather than a treatment effect. "
            "Pass --allow-identical-archive if this is a deliberate null control."
        )
    return {
        "archive_zip_relpaths_in_both": relpaths,
        "archive_zip_differs": differs,
        "allow_identical_archive": bool(allow_identical_archive),
        "why_this_check_exists": (
            "The comparator answers 'do these trees differ ONLY by payload?' and must keep "
            "passing when handed the same tree twice. 'Was a treatment applied at all?' is "
            "this caller's question, and it is asked HERE (ddm_fc3, task #1179)."
        ),
    }


def commit_delta_touches_forward(commit_range: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", commit_range],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [p for p in out if any(p.startswith(pre) for pre in FORWARD_PATH_PREFIXES)]


def run(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base_receipt).read_text())
    cand = json.loads(Path(args.candidate_receipt).read_text())
    fb, fc = forward_fields(base), forward_fields(cand)

    mismatched = {k: (fb[k], fc[k]) for k in fb if fb[k] != fc[k]}
    missing = [k for k, v in fc.items() if v is None]
    if missing:
        raise PoseLegError(
            f"candidate receipt is missing forward-determining fields {missing}; "
            "a delta across an unidentified instrument is not a measurement"
        )
    if mismatched:
        raise PoseLegError(
            f"REFUSING: forward-determining fields differ between the two receipts: "
            f"{mismatched}. A delta across a mismatched tuple is an instrument "
            "artifact, not a finding."
        )

    tree_check: dict[str, Any] = {"checked": False}
    if args.base_runtime and args.candidate_runtime:
        tree_check = runtime_trees_differ_only_by_payload(
            Path(args.base_runtime), Path(args.candidate_runtime)
        )
        tree_check["checked"] = True
        if tree_check["verdict"] != "PAYLOAD_ONLY":
            raise PoseLegError(
                "REFUSING: the two runtime trees differ by more than their payload. "
                f"unexpected differing files={tree_check['unexpected_differing_files']}, "
                f"only_in_base={tree_check['files_only_in_base']}, "
                f"only_in_candidate={tree_check['files_only_in_candidate']}, "
                f"inflate_py_AST_identical_once_pin_normalised="
                f"{tree_check['inflate_py_AST_identical_once_pin_normalised']}. "
                "A receiver change cannot ride in as payload."
            )
        # The comparator proved nothing ELSE moved. Now prove the payload DID.
        tree_check["treatment_check"] = assert_treatment_was_applied(
            tree_check, allow_identical_archive=args.allow_identical_archive
        )
    else:
        raise PoseLegError(
            "REFUSING: --base-runtime and --candidate-runtime are required. "
            "runtime_files_sha256 ALWAYS differs between base and candidate (each "
            "tree embeds its own archive and derived pin), so the only honest way "
            "to clear it is to prove structurally that nothing ELSE moved."
        )

    commit_base = _dig(base, "instrument_tuple", "code", "pact_commit")
    commit_cand = _dig(cand, "instrument_tuple", "code", "pact_commit")
    commit_note: dict[str, Any] = {
        "base_pact_commit": commit_base,
        "candidate_pact_commit": commit_cand,
        "identical": commit_base == commit_cand,
    }
    if commit_base != commit_cand:
        if not args.commit_delta_proof:
            raise PoseLegError(
                f"pact_commit differs ({commit_base} vs {commit_cand}) and no "
                "--commit-delta-proof was supplied. Pass the git range so this "
                "module can re-check it against the forward path itself."
            )
        offenders = commit_delta_touches_forward(args.commit_delta_proof)
        if offenders:
            raise PoseLegError(
                f"REFUSING: the commit range {args.commit_delta_proof} changes "
                f"forward-path files {offenders}; the two receipts are not the "
                "same instrument."
            )
        commit_note["commit_delta_proof"] = args.commit_delta_proof
        commit_note["forward_path_files_changed"] = []
        commit_note["verdict"] = (
            "pact_commit differs but the range touches NO forward-path file; "
            "re-checked by this module, not asserted by the caller"
        )

    d_seg_b, d_seg_c = base["avg_segnet_dist"], cand["avg_segnet_dist"]
    d_pose_b, d_pose_c = base["avg_posenet_dist"], cand["avg_posenet_dist"]
    bytes_b = _dig(base, "provenance", "archive_size_bytes")
    bytes_c = _dig(cand, "provenance", "archive_size_bytes")

    seg_leg = 100.0 * (d_seg_c - d_seg_b)
    pose_leg = (10.0 * d_pose_c) ** 0.5 - (10.0 * d_pose_b) ** 0.5
    rate_leg = (bytes_c - bytes_b) * S_PER_ARCHIVE_BYTE
    net = seg_leg + pose_leg + rate_leg

    report = {
        "schema": "ddm_fs3_same_instrument_legs.v1",
        "arm": "ddm_fs3",
        "task": 1176,
        "axis": "[cpu_env_mismatch_advisory] both sides -- SAME instrument, so the DELTA is the claim, never either level",
        "score_claim": False,
        "promotion_eligible": False,
        "instrument_match": {
            "harness_fields_identical": fc,
            "verdict": "IDENTICAL",
            "pact_commit": commit_note,
            "runtime_tree_payload_vs_receiver": tree_check,
        },
        "base": {
            "receipt": str(args.base_receipt),
            "archive_bytes": bytes_b,
            "d_seg": d_seg_b,
            "d_pose": d_pose_b,
        },
        "candidate": {
            "receipt": str(args.candidate_receipt),
            "archive_bytes": bytes_c,
            "d_seg": d_seg_c,
            "d_pose": d_pose_c,
        },
        "legs_same_instrument": {
            "seg": seg_leg,
            "pose": pose_leg,
            "rate": rate_leg,
            "net": net,
        },
        "multiple_of_bar": abs(net) / 3.5e-6,
        "clears_admission_bar": net < -3.5e-6,
        "caveat": (
            "these legs are measured on the PYAV/env-mismatch advisory axis. The "
            "delta is same-instrument and therefore legitimate, but its TRANSFER to "
            "the contest-CUDA axis is a separate question (ddm_pi2 measured 1.4425x "
            "on d_seg and +1.4061e-04 additive on d_pose across that lineage gap)."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("harness match: IDENTICAL on every forward-determining field")
    print(
        f"  runtime trees: {tree_check['verdict']} "
        f"(differ only in {tree_check['differing_files']}; inflate.py identical once "
        f"the pin is normalised: {tree_check['inflate_py_AST_identical_once_pin_normalised']})"
    )
    if not commit_note["identical"]:
        print(f"  pact_commit differs; range {args.commit_delta_proof} touches no forward file")
    print(f"  base      {bytes_b:,} B  d_seg {d_seg_b:.8f}  d_pose {d_pose_b:.8f}")
    print(f"  candidate {bytes_c:,} B  d_seg {d_seg_c:.8f}  d_pose {d_pose_c:.8f}")
    print(f"\n  seg  {seg_leg:+.6e}\n  pose {pose_leg:+.6e}\n  rate {rate_leg:+.6e}")
    print(
        f"  NET  {net:+.6e}  ({report['multiple_of_bar']:.2f}x bar)  "
        f"clears={report['clears_admission_bar']}"
    )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--base-runtime", default=None)
    parser.add_argument("--candidate-runtime", default=None)
    parser.add_argument(
        "--commit-delta-proof",
        default=None,
        help="git range re-checked by this module against the forward path",
    )
    parser.add_argument(
        "--allow-identical-archive",
        action="store_true",
        help=(
            "declare a deliberate NULL control: permit base and candidate archive.zip to be "
            "byte-identical. Off by default, so a run with no treatment must be declared "
            "rather than stumbled into (ddm_fc3, task #1179)."
        ),
    )
    parser.add_argument(
        "--out", default="/Volumes/APDataStore/pact/ddm_fs3/FS3_SAME_INSTRUMENT_LEGS.json"
    )
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
