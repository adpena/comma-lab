#!/usr/bin/env python3
"""ddm_fx5 -- build the E1 19-member receiver from the rc2 composed runtime.

WHAT THIS DOES, AND WHY IT IS A PATCH RATHER THAN A REWRITE.

``ddm_fx2`` raced the probability-MODEL axis on the n600 token field and measured
its best architecture, **E1**, at **-797.42 B** against the live rr4 law.  It
shipped the smaller **D1** build at **-710.84 B** instead, and said exactly why:
serial decode timing projected E1's parse-back margin at **29 s** against the
1,800 s contest budget, which is not a margin.  Its own memo named the door::

    "it becomes the pick the moment somebody measures real T4 headroom
     instead of projecting it."

``ddm_rc2`` measured it.  On the shipping object, contest-CUDA T4 n600, the
harness adjudicated **498.476 s charged <= 822 s cold-cache ceiling** -- **323.5 s
of slack**.  ``ddm_dx1`` had refused the same remainder for adding "+89 s to a
body already decode-REFUSED"; against 323.5 s of measured slack that same +89 s
lands at 587.5 s, **234.5 s under the ceiling**.  So the refusing precondition is
gone and the banked row is the pick.

E1 and D1 differ in ONE dimension: the member set, 19 against 13.  The mixer
context, the count buckets, the learning rate, the SSE switch and ma1's
within-miss sector are all IDENTICAL.  So this builder is a PATCH over the rc2
receiver, not a new receiver, and every anchor it rewrites is asserted before it
is replaced.  A rewrite would have re-derived thirteen members that are already
proven on a fired T4 row; a patch keeps that proof and puts the whole delta on
the six new members.

THE SIX NEW MEMBERS (``ddm_fx2`` race receipt ``E1_compose_19x_homogctx.json``,
retained at ``/Volumes/APDataStore/pact/ddm_fx2/race/``)::

    13  homog_boundary_surprise      ((cls*HOM + homog)*BND + boundary)*U_BINS + ubin
    14  spatial4_boundary            (cls*SP4 + spatial4)*BND + boundary
    15  homog_spatial4               (cls*HOM + homog)*SP4 + spatial4
    16  spatial4_temporal            ((cls*2 + agree1)*2 + agree2)*SP4 + spatial4
    17  homog_surprise_fast256       homog_surprise rule, count_limit 256
    18  spatial4_surprise_fast256    spatial4_surprise rule, count_limit 256

Members 17 and 18 reuse rules the C already implements, so only FOUR new rule
cases are added.  Every rule is transcribed from ``fx2_family_specs()`` in the
receiver's own ``fx2_model_axis_corrector.py`` -- the same source the Python
reference reads at runtime -- so the C and the Python cannot disagree about a
rule by transcription drift.  ``ddm_fx5_parity_e1.py`` is what PROVES they do
not, on real decoded state.

RULE 118.  Every member is generic receiver code reading only ALREADY-DECODED
symbols.  Nothing here is transmitted, learned-and-shipped, or video-derived:
the archive carries zero extra bytes for the six new members.  That is the whole
point -- the win is -86.58 B of TOKEN STREAM at +0 B of counted payload.

EXACTNESS.  The C touches no transcendental.  Every added rule is integer index
arithmetic (multiply and add on int64), which is exact, so the ``ddm_rr2``
refusal class (S = 27.83 from one libm ULP desynchronising the arithmetic
decoder) cannot be reached through this patch.  The existing AST gate in the
Python sources is untouched and still binds.

FAIL-CLOSED.  Every source file is pinned by sha256 BEFORE patching and every
anchor string must appear EXACTLY ONCE.  A base tree that is not the one this
builder was written against is refused by name rather than silently patched into
a body nobody measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

#: The rc2 composed runtime -- the SIXTEENTH-MOVE body, fired and confirmed at
#: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4, n600].
DEFAULT_BASE = Path("/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed")

#: Pinned so a base that is not the fired body is refused rather than patched.
BASE_ARCHIVE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
BASE_ARCHIVE_BYTES = 180_456

#: sha256 of EVERY file this builder rewrites, in the base tree.  Measured
#: 2026-08-21 from the fired rc2 runtime; the first two are also receiver pins in
#: ``CANDIDATE_SEAL_rc2_composed.json``, so they are cross-checkable there.
#:
#: All three are pinned deliberately.  An anchor string can match on content the
#: builder was never written against -- the anchors here are short and generic
#: enough that a drifted file could still take the patch and produce a tree whose
#: 19 members were never the 19 that ddm_fx2 raced.  The sha is what makes the
#: patch an assertion about a KNOWN body rather than a text substitution.
BASE_FILE_PINS: dict[str, str] = {
    "runtime/f26_corrector_native.c": (
        "01a6e9557f9692156d6b1a1a325c8ee1350a28900550251fd19cdead8578c986"
    ),
    "runtime/native_free_corrector.py": (
        "70d2073b4235ba6eeb436c1a48203deb4ba92fc2d94882f80e94d1adaa4b57ef"
    ),
    "runtime/fx2_model_axis_corrector.py": (
        "3cbddcf85e82d7a17e3f19e649a8af1901ea62fd5d91a7ca0d13f1f7edbcec79"
    ),
}

#: The E1 member set, VERBATIM from the fx2 race receipt's ``families`` list.
E1_FAMILIES: tuple[str, ...] = (
    # ddm_fx1's eleven, unchanged.
    "shipped_joint",
    "temporal_spatial",
    "surprise_only",
    "spatial_surprise",
    "spatial_boundary",
    "run_surprise",
    "boundary_surprise",
    "temporal_surprise",
    "shipped_fast256",
    "shipped_fast4096",
    "surprise_fast256",
    # The two members ddm_fx2's widened causal template unlocked (D1 stops here).
    "spatial4_surprise",
    "homog_surprise",
    # The six E1 adds.
    "homog_boundary_surprise",
    "spatial4_boundary",
    "homog_spatial4",
    "spatial4_temporal",
    "homog_surprise_fast256",
    "spatial4_surprise_fast256",
)

D1_FAMILIES: tuple[str, ...] = E1_FAMILIES[:13]


class Fx5BuildError(SystemExit):
    """The base tree is not the one this builder was written against."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, anchor: str, replacement: str, what: str) -> str:
    """Replace ``anchor`` exactly once, or refuse.

    A patch that silently matched zero times would produce a tree that looks
    built and behaves like the base -- the inert-flag failure class.  A patch
    that matched twice would corrupt an unrelated site.  Both are refused.
    """
    count = text.count(anchor)
    if count != 1:
        raise Fx5BuildError(
            f"REFUSING: anchor for {what!r} appears {count} time(s), expected exactly 1. "
            "The base tree is not the one this builder was written against; re-derive "
            "the anchor against the actual base rather than loosening this check."
        )
    return text.replace(anchor, replacement)


# --------------------------------------------------------------------------------------
# The C patch -- four new rule cases and four extended tables.
# --------------------------------------------------------------------------------------

C_ENUM_ANCHOR = """    RULE_SPATIAL4_SURPRISE,
    RULE_HOMOG_SURPRISE
};"""

C_ENUM_PATCH = """    RULE_SPATIAL4_SURPRISE,
    RULE_HOMOG_SURPRISE,
    /* ddm_fx5: the four rules E1's six new members need.  17 and 18 reuse
     * RULE_HOMOG_SURPRISE / RULE_SPATIAL4_SURPRISE at count_limit 256. */
    RULE_HOMOG_BOUNDARY_SURPRISE,
    RULE_SPATIAL4_BOUNDARY,
    RULE_HOMOG_SPATIAL4,
    RULE_SPATIAL4_TEMPORAL
};"""

C_NFAM_ANCHOR = """/* the live family set, in SHIPPED_CONFIG order */
#define N_FAMILIES 13"""

C_NFAM_PATCH = """/* the live family set, in SHIPPED_CONFIG order.
 * ddm_fx5: 13 -> 19.  ddm_fx2 raced this member set as E1 and measured -797.42 B
 * against the live rr4 law, 86.58 B beyond the D1 build rc2 ships; it withheld E1
 * only because serial timing PROJECTED a 29 s parse-back margin.  ddm_rc2 then
 * MEASURED the real T4 wall at 498.476 s against an 822 s ceiling (323.5 s slack),
 * so the withholding precondition is discharged.  N_MIXER_CONTEXTS is unchanged:
 * E1 and D1 share the mixer context exactly. */
#define N_FAMILIES 19"""

C_RULE_TABLE_ANCHOR = """    RULE_SPATIAL4_SURPRISE,  /* spatial4_surprise  */
    RULE_HOMOG_SURPRISE      /* homog_surprise     */
};"""

C_RULE_TABLE_PATCH = """    RULE_SPATIAL4_SURPRISE,  /* spatial4_surprise  */
    RULE_HOMOG_SURPRISE,     /* homog_surprise     */
    /* ddm_fx5: E1's six. */
    RULE_HOMOG_BOUNDARY_SURPRISE, /* homog_boundary_surprise   */
    RULE_SPATIAL4_BOUNDARY,       /* spatial4_boundary         */
    RULE_HOMOG_SPATIAL4,          /* homog_spatial4            */
    RULE_SPATIAL4_TEMPORAL,       /* spatial4_temporal         */
    RULE_HOMOG_SURPRISE,          /* homog_surprise_fast256    */
    RULE_SPATIAL4_SURPRISE        /* spatial4_surprise_fast256 */
};"""

C_SIZE_TABLE_ANCHOR = """    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS
};"""

C_SIZE_TABLE_PATCH = """    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS,
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,
    /* ddm_fx5: E1's six, transcribed from fx2_family_specs(). */
    NUM_CLASSES * HOMOGENEITY_LEVELS * BOUNDARY_LEVELS * U_BINS, /* 8000 */
    NUM_CLASSES * SPATIAL4_LEVELS * BOUNDARY_LEVELS,             /*  150 */
    NUM_CLASSES * HOMOGENEITY_LEVELS * SPATIAL4_LEVELS,          /*  150 */
    NUM_CLASSES * 2 * 2 * SPATIAL4_LEVELS,                       /*  120 */
    NUM_CLASSES * HOMOGENEITY_LEVELS * U_BINS,                   /* 1600 */
    NUM_CLASSES * SPATIAL4_LEVELS * U_BINS                       /* 1920 */
};"""

C_COUNT_LIMIT_ANCHOR = """static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0
};"""

C_COUNT_LIMIT_PATCH = """static const int64_t FAMILY_COUNT_LIMIT[N_FAMILIES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 256, 4096, 256, 0, 0,
    /* ddm_fx5: E1's six -- the two ``_fast256`` members carry the recency window. */
    0, 0, 0, 0, 256, 256
};"""

C_IS_JOINT_ANCHOR = """static const int FAMILY_IS_SHIPPED_JOINT[N_FAMILIES] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};"""

C_IS_JOINT_PATCH = """static const int FAMILY_IS_SHIPPED_JOINT[N_FAMILIES] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /* ddm_fx5: E1's six all start at weight 0, so the mixture still BEGINS at the
     * incumbent law and the learner must earn every byte away from it. */
    0, 0, 0, 0, 0, 0
};"""

C_RULE_INDEX_ANCHOR = """    case RULE_HOMOG_SURPRISE:
        return (cls * HOMOGENEITY_LEVELS + homog) * U_BINS + ubin;
    default:
        return 0;
    }
}"""

C_RULE_INDEX_PATCH = """    case RULE_HOMOG_SURPRISE:
        return (cls * HOMOGENEITY_LEVELS + homog) * U_BINS + ubin;
    /* ddm_fx5: E1's four new rules.  Each is transcribed from the matching closure
     * in ``fx2_model_axis_corrector.fx2_family_specs`` and is pure int64 index
     * arithmetic -- exact on every conforming platform, no transcendental. */
    case RULE_HOMOG_BOUNDARY_SURPRISE: {
        int64_t head = (cls * HOMOGENEITY_LEVELS + homog) * BOUNDARY_LEVELS + boundary;
        return head * U_BINS + ubin;
    }
    case RULE_SPATIAL4_BOUNDARY:
        return (cls * SPATIAL4_LEVELS + spatial4) * BOUNDARY_LEVELS + boundary;
    case RULE_HOMOG_SPATIAL4:
        return (cls * HOMOGENEITY_LEVELS + homog) * SPATIAL4_LEVELS + spatial4;
    case RULE_SPATIAL4_TEMPORAL: {
        int64_t head = (cls * 2 + agree1) * 2 + agree2;
        return head * SPATIAL4_LEVELS + spatial4;
    }
    default:
        return 0;
    }
}"""

C_DOC_ANCHOR = """ * the mixer MEMBER-outer / position-inner: for each of the 13 members it materialises the"""
C_DOC_PATCH = """ * the mixer MEMBER-outer / position-inner: for each of the 19 members it materialises the"""


def patch_c(text: str) -> str:
    text = replace_once(text, C_NFAM_ANCHOR, C_NFAM_PATCH, "N_FAMILIES")
    text = replace_once(text, C_ENUM_ANCHOR, C_ENUM_PATCH, "rule enum")
    text = replace_once(text, C_RULE_TABLE_ANCHOR, C_RULE_TABLE_PATCH, "FAMILY_RULE")
    text = replace_once(text, C_SIZE_TABLE_ANCHOR, C_SIZE_TABLE_PATCH, "FAMILY_SIZE")
    text = replace_once(
        text, C_COUNT_LIMIT_ANCHOR, C_COUNT_LIMIT_PATCH, "FAMILY_COUNT_LIMIT"
    )
    text = replace_once(
        text, C_IS_JOINT_ANCHOR, C_IS_JOINT_PATCH, "FAMILY_IS_SHIPPED_JOINT"
    )
    text = replace_once(
        text, C_RULE_INDEX_ANCHOR, C_RULE_INDEX_PATCH, "family_rule_index"
    )
    text = replace_once(text, C_DOC_ANCHOR, C_DOC_PATCH, "member-count docstring")
    return text


# --------------------------------------------------------------------------------------
# The Python patches -- the frozen config the C guard checks itself against.
# --------------------------------------------------------------------------------------

PY_FX2_ANCHOR = """        # The two members the widened causal template unlocks.
        "spatial4_surprise",
        "homog_surprise",
    ),"""

PY_FX2_PATCH = """        # The two members the widened causal template unlocks.
        "spatial4_surprise",
        "homog_surprise",
        # ddm_fx5: the six that take D1's -710.84 B to E1's -797.42 B.  ddm_fx2
        # measured this exact set and withheld it on a PROJECTED 29 s decode
        # margin; ddm_rc2 MEASURED 323.5 s of real T4 slack, so it ships.
        "homog_boundary_surprise",
        "spatial4_boundary",
        "homog_spatial4",
        "spatial4_temporal",
        "homog_surprise_fast256",
        "spatial4_surprise_fast256",
    ),"""

PY_NATIVE_ANCHOR = """        "spatial4_surprise",
        "homog_surprise",
    ),
    "mixer_context": "cls_boundary_agree_homog_ubin8","""

PY_NATIVE_PATCH = """        "spatial4_surprise",
        "homog_surprise",
        # ddm_fx5: E1's six, compiled into f26_corrector_native.c alongside these.
        "homog_boundary_surprise",
        "spatial4_boundary",
        "homog_spatial4",
        "spatial4_temporal",
        "homog_surprise_fast256",
        "spatial4_surprise_fast256",
    ),
    "mixer_context": "cls_boundary_agree_homog_ubin8","""


def build(base: Path, out: Path, *, force: bool = False) -> dict:
    base = Path(base)
    out = Path(out)

    archive = base / "archive.zip"
    if not archive.is_file():
        raise Fx5BuildError(f"REFUSING: no archive.zip under base tree {base}")
    archive_sha = sha256_file(archive)
    if archive_sha != BASE_ARCHIVE_SHA256:
        raise Fx5BuildError(
            f"REFUSING: base archive sha {archive_sha} != the fired rc2 body "
            f"{BASE_ARCHIVE_SHA256}. This builder patches the SIXTEENTH-MOVE body; "
            "pointing it at another body would produce a candidate whose base nobody "
            "measured."
        )
    if archive.stat().st_size != BASE_ARCHIVE_BYTES:
        raise Fx5BuildError("REFUSING: base archive byte count is not 180,456")

    for relative, pin in BASE_FILE_PINS.items():
        actual = sha256_file(base / relative)
        if actual != pin:
            raise Fx5BuildError(
                f"REFUSING: {relative} sha {actual} != pinned {pin}. The receiver this "
                "builder patches has moved; re-derive the anchors against it."
            )

    if out.exists():
        if not force:
            raise Fx5BuildError(f"REFUSING: {out} exists; pass --force to rebuild it")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        base, out, ignore=shutil.ignore_patterns("__pycache__", "._*", "*.pyc")
    )

    patched: list[dict[str, object]] = []

    targets = (
        ("runtime/f26_corrector_native.c", patch_c),
        (
            "runtime/fx2_model_axis_corrector.py",
            lambda t: replace_once(t, PY_FX2_ANCHOR, PY_FX2_PATCH, "SHIPPED_CONFIG"),
        ),
        (
            "runtime/native_free_corrector.py",
            lambda t: replace_once(
                t, PY_NATIVE_ANCHOR, PY_NATIVE_PATCH, "EXPECTED_SHIPPED_CONFIG"
            ),
        ),
    )
    for relative, patcher in targets:
        path = out / relative
        before = path.read_text()
        after = patcher(before)
        if after == before:
            raise Fx5BuildError(f"REFUSING: patch for {relative} changed nothing")
        path.write_text(after)
        patched.append(
            {
                "relative_path": relative,
                "sha256_before": hashlib.sha256(before.encode()).hexdigest(),
                "sha256_after": hashlib.sha256(after.encode()).hexdigest(),
                "bytes_before": len(before.encode()),
                "bytes_after": len(after.encode()),
            }
        )

    manifest = {
        "schema": "ddm_fx5_e1_runtime_build.v1",
        "built_by": "ddm_fx5",
        "base_runtime": str(base),
        "base_archive_sha256": archive_sha,
        "base_archive_bytes": BASE_ARCHIVE_BYTES,
        "out_runtime": str(out),
        "member_count_base": len(D1_FAMILIES),
        "member_count_candidate": len(E1_FAMILIES),
        "families_candidate": list(E1_FAMILIES),
        "mixer_context": "cls_boundary_agree_homog_ubin8",
        "counted_bytes_added_by_the_new_members": 0,
        "rule_118": (
            "every new member reads only already-decoded symbols; nothing is "
            "transmitted, learned-and-shipped, or video-derived"
        ),
        "fx2_race_receipt": (
            "/Volumes/APDataStore/pact/ddm_fx2/race/E1_compose_19x_homogctx.json"
        ),
        "fx2_measured_code_bytes_vs_live": -797.4238121195522,
        "d1_measured_code_bytes_vs_live": -710.84,
        "projected_token_delta_bytes": -86.58,
        "patched_files": patched,
        "axis": "[build artifact -- no measurement, no score claim]",
        "score_claim": False,
    }
    (out / "FX5_BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


#: ``inflate.py`` pins the archive it was promoted with, so the receiver refuses
#: to decode bytes it was not built for.  That guard is correct and it FIRED on
#: the first candidate decode; the pin is therefore repinned deliberately, by
#: this builder, against the archive actually on disk -- never hand-typed.
INFLATE_PIN_TEMPLATE = 'ARCHIVE_SHA256 = "{sha}"\nARCHIVE_BYTES = {bytes_:_}'


def repin_inflate(runtime: Path) -> dict:
    """Rewrite ``inflate.py``'s archive pin to the archive sitting beside it.

    The values are MEASURED from disk inside this function.  There is
    deliberately no flag to pass a sha: a repin that accepts a typed digest can
    be pointed at bytes nobody has, which is the whole failure the pin exists to
    stop.
    """
    runtime = Path(runtime)
    archive = runtime / "archive.zip"
    if not archive.is_file():
        raise Fx5BuildError(f"REFUSING: no archive.zip beside {runtime}/inflate.py")
    sha = sha256_file(archive)
    size = archive.stat().st_size

    path = runtime / "inflate.py"
    text = path.read_text()
    old_sha = next(
        (
            line.split('"')[1]
            for line in text.splitlines()
            if line.startswith("ARCHIVE_SHA256 = ")
        ),
        None,
    )
    old_bytes = next(
        (
            line.split("= ")[1].strip()
            for line in text.splitlines()
            if line.startswith("ARCHIVE_BYTES = ")
        ),
        None,
    )
    if old_sha is None or old_bytes is None:
        raise Fx5BuildError("REFUSING: inflate.py carries no ARCHIVE_SHA256/ARCHIVE_BYTES pin")
    anchor = INFLATE_PIN_TEMPLATE.format(sha=old_sha, bytes_=int(old_bytes.replace("_", "")))
    replacement = INFLATE_PIN_TEMPLATE.format(sha=sha, bytes_=size)
    text = replace_once(text, anchor, replacement, "inflate.py archive pin")
    path.write_text(text)
    return {
        "inflate_py": str(path),
        "archive_sha256_before": old_sha,
        "archive_sha256_after": sha,
        "archive_bytes_before": int(old_bytes.replace("_", "")),
        "archive_bytes_after": size,
        "inflate_py_sha256_after": hashlib.sha256(text.encode()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--out")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--repin",
        help=(
            "repin <RUNTIME>/inflate.py to <RUNTIME>/archive.zip and exit; run this "
            "AFTER the candidate archive has been spliced into the tree"
        ),
    )
    args = parser.parse_args(argv)
    if args.repin:
        print(json.dumps(repin_inflate(Path(args.repin)), indent=2, sort_keys=True))
        return 0
    if not args.out:
        parser.error("--out is required unless --repin is given")
    manifest = build(Path(args.base), Path(args.out), force=args.force)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
