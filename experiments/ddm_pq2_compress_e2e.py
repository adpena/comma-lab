#!/usr/bin/env python3
"""End-to-end compression entry point: rebuild the submission archive and prove its bytes.

WHAT THIS IS.  One sanitized command that takes the retained training product and
returns the exact ``archive.zip`` that was evaluated, then refuses to exit 0 unless
the rebuilt bytes hash to the pinned sha256.  It orchestrates the two instruments
that actually produced the candidate -- it does not reimplement them, so what this
script proves is what the shipped pipeline does, not a parallel reconstruction of it.

THE STAGES.  Three phases -- A, B, C -- across four subcommands.  The count is
stated because an earlier revision of this docstring said "three stages" and then
listed four headings, which is the kind of small incoherence that makes a reader
stop trusting the larger claims.

  ``provenance``  (stage A, documented, no compute)  Emits the training lineage that
      produced the checkpoint: the stage scripts, their arguments, the corrector
      module, and the input manifest with every sha256.  Stage A is DOCUMENTED
      RATHER THAN RE-RUN, and this script says so plainly: reproducing the
      checkpoint from raw video is days of GPU compute.  What is verifiable here
      is everything downstream of the checkpoint, which is the part that
      determines the archive bytes.

  ``encode`` + ``build``  (stage B, EXACT and verifiable)  Replays the shipped
      decode order, encodes the token field under the free decode-time corrector,
      splices the new token stream into the member, and repacks the archive.  The
      seven non-token sections are carried through byte-identically by
      construction; only the token stream is re-encoded.  ``--resume`` continues
      bit-faithfully from a retained frame-boundary checkpoint, which is what
      makes a cheap end-to-end verification possible.

  ``verify``  (stage B gate)  Hashes the rebuilt archive and compares it against the
      expected sha256 and byte count.  Fail-closed: a mismatch exits non-zero.

      WHICH CANDIDATE IS EXPECTED IS RESOLVED AT RUN TIME, never latched in this file.
      In priority order: ``--expected-archive-sha256`` + ``--expected-archive-bytes``;
      or ``--candidate-runtime <dir>``, measured from a sealed tree whose receiver pin
      must already agree with its own archive; or, by default, the canonical frontier
      pointer read now.  A hardcoded expectation here previously refused every non-rr4
      candidate BY CONSTRUCTION — the one entry point built to byte-close candidates
      could not close the two that were waiting.  The rebuild RECIPE (corrector module,
      token stream, base archive) is separate and still describes rr4 by default; a
      different candidate supplies ``--recipe-json``, and asking for candidate X under
      candidate Y's recipe is refused up front rather than deep inside the rebuild.

  ``decode``  (stage C)  Runs the shipped receiver over the rebuilt archive and
      checks that the decoded token field reproduces its pinned sha256.  This is
      the distortion proof: an identical decoded field cannot move d_seg or
      d_pose, so the score delta is a pure rate delta.

      HONEST LIMITATION.  Stage C drives our internal receiver-closure harness,
      which still resolves one source root from local custody, so it is a
      REPOSITORY proof rather than a judge-runnable one.  A judge does not need
      it: the shipped ``inflate.sh`` is the real decode path, takes no private
      input, and is what the evaluator runs.  Stage C exists to prove the
      decoded field is unchanged, not to be the decode entry point.

WHAT THIS ENTRY POINT EXPRESSES -- AND WHAT IT DOES NOT.  Read this before
concluding that re-running it would reproduce whatever archive we currently ship.

  EXPRESSIBLE.  A candidate whose build chain is (i) a re-encode of the TOKEN
      stream under a decode-time probability corrector, optionally followed by
      (ii) a CONTAINER repack declared through ``SPLIT_RECIPE_KEYS``.  The other
      seven parsed sections are carried through verbatim.  That is the rr4 and
      sz1 shape, and for those candidates this script is a genuine byte-close.

  NOT EXPRESSIBLE.  A candidate whose chain also RE-DECIDES CONTENT -- semantic
      re-quantization, seg token edits, edit admission, or a pose-carrier
      re-solve.  Those stages write sections this script copies verbatim, so no
      recipe can make a token-only rebuild produce their bytes.  Such candidates
      are named in ``NOT_EXPRESSIBLE`` and are REFUSED BY NAME, with their real
      builders cited, rather than being met with "pass ``--recipe-json``" -- an
      answer that is true for a missing recipe and false for a missing stage.

  This matters at the default invocation.  With no flags the expected archive is
      read from the canonical frontier pointer, i.e. whatever we currently ship.
      If that candidate is not expressible, the honest response is a typed
      refusal that says which stages are missing, not a generic mismatch.

NO PRIVATE PATHS.  This file contains no filesystem layout.  Every input root is
supplied by the caller through ``--inputs-json`` (or the matching environment
variables) and is validated by sha256 before any stage runs.  Run
``--emit-inputs-template`` to print the schema.

DETERMINISM.  The pipeline carries no RNG: the corrector is integer/rational
arithmetic and the range coder is exact.  The seed below is recorded and pushed
into the environment so that any future stochastic stage inherits it rather than
silently defaulting.  Determinism is checked by measurement -- ``build`` writes a
second archive from the same member and asserts the two are byte-identical.

AXIS.  ``[macOS-CPU advisory / scorer-free EXACT byte measurement]``.  This script
measures BYTES and byte-identity only.  It makes no score claim: the score on the
rebuilt bytes is whatever the contest evaluator returns on the contest hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import (
    CONSISTENT,
    ArchiveIdentity,
    SealContractError,
    check_pin_consistency,
    measure_archive_identity,
    read_frontier_archive_identity,
)

ENCODER = REPO / "experiments" / "ddm_rr2_encoder_byteclose.py"
RECEIVER = REPO / "experiments" / "ddm_rr2_receiver_close.py"

# THE REBUILD RECIPE, not an admission bar.  These values describe the PIPELINE that
# produces the rr4 candidate: which corrector runs, which token stream the rebuild must
# reproduce, which base archive it splices into.  They were measured from the retained
# build receipt and they are assertions about that rebuild.
#
# They are deliberately NOT the expected-archive identity any more.  Latching that identity
# here is what made this script refuse every non-rr4 candidate by construction -- fx1's
# 180,601 B row and sa1's ladder could not be byte-closed through the one entry point built
# to byte-close candidates.  The expected identity is now RESOLVED AT RUN TIME (see
# `resolve_expected_archive`), and a caller rebuilding a different candidate must supply
# that candidate's own recipe rather than silently asserting rr4's numbers over other bytes.
# ``rc64_source_sha256`` NAMES THE ENCODER ROLE, NOT THE SHIPPED MEMBER.  FOUR distinct
# bodies wear the file name ``rc64_backend.c``, and conflating them cost two arms a
# byte-close each.  The count is MEASURED, not remembered: ``ddm_rv14f`` hashed all 241
# copies across the three custody roots on 2026-08-19 and found four contents
# (``reverse_engineering/rc64_backend_role_registry.json``).  An earlier revision of this
# comment said "two", which is the same undercount that makes a filename search look
# conclusive when it is not:
#
#   ENCODER role      12,222 B  5c75e2c7…  1 copy.  Encoder + decoder.  THE PIN below.
#                                          ``ddm_rr2_encoder_byteclose`` appends the
#                                          2,603 B checkpoint/resume extension and
#                                          compiles the 14,825 B result at build time.
#   SHIPPED role       5,638 B  05839d14…  237 copies.  DECODER ONLY -- the member every
#                                          archive carries at runtime/entropy/.  It
#                                          exports no encoder symbol, so it can never
#                                          drive the encode stage.
#   CHECKPOINT-EXT    14,825 B  1941923a…  2 copies.  The encoder WITH the extension
#                                          already appended; also sits under the plain
#                                          name.  The encoder body is exactly recoverable
#                                          from it by removing the extension.
#   FOREIGN INTAKE    22,179 B  b249b77b…  1 copy.  A PR #138 ``opal_v1`` intake body.
#                                          NOT OURS and never a candidate for any pin.
#
# The last row is why the count matters: a search keyed on the file name can reach a
# third party's source, and pinning it would silently build against foreign code.
#
# THE PIN IS CORRECT AND THE FILE EXISTS.  ddm_ma1's memo (2026-08-19 §7) reported the
# opposite -- "158 copies, 2 distinct contents, neither matches ... clearing this pin is
# the first owed step" -- after hashing every ``rc64_backend.c`` it could find.  That scan
# missed the encoder body, which has been at
# ``<VertigoDataTier>/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/``
# since 2026-08-10, and ddm_fx2's own byte-close chain USED it successfully on 2026-08-17
# (180,450 B, archive-repeat byte-identical).  The blocker was never a stale sha; it was
# that nothing named the ROLE or the location, so a search keyed on the file name found
# only the shipped decoder and concluded the pin was unclearable.  Redundantly, the
# encoder body is also exactly recoverable from the pipeline's own retained
# ``rc64_backend_checkpoint.c`` by removing the extension.  A recipe MAY additionally pin
# the shipped receiver member through ``rc64_shipped_member_sha256`` (see
# RECEIVER_RECIPE_KEYS) so both halves of the coder are custodied by name.
RR4_RECIPE: dict[str, object] = {
    "name": "rr4",
    "archive_sha256": "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956",
    "archive_bytes": 181_161,
    "token_sha256": "6c3757bd52a18d3c38e9120d56293f03c7aefd111fb9ee655b19d055e8d06b14",
    "token_bytes": 110_512,
    "decoded_field_sha256": "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52",
    "corrector_module": "ddm_rr4_free_corrector_v2",
    "base_archive_sha256": "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e",
    "rc64_source_sha256": "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6",
}

RECIPE_KEYS = tuple(RR4_RECIPE)

# OPTIONAL receiver-role key.  When a recipe declares it, the shipped decoder member
# becomes a VERIFIED INPUT alongside the encoder source, so the rebuild pins both halves
# of the coder by role instead of pinning one and inheriting the other by accident.
# Absent, behaviour is exactly as before -- rr4's recipe does not declare it.
RECEIVER_RECIPE_KEYS = ("rc64_shipped_member_sha256",)

# OPTIONAL split-stage keys, all-or-none.  A COMPOSED candidate is a token re-encode
# FOLLOWED BY a container repack that this script's token-only rebuild cannot express
# (see the "WHY NOT ddm_pq2_compress_e2e.py" note in ddm_sz1_build_split_archive.py).
# When a recipe declares these, the build stage produces the PRE-split archive
# (asserted against ``pre_split_archive_sha256``), the split stage runs the canonical
# repack builder -- which carries its own base pin, deterministic-repack proof,
# token-stream-verbatim proof and decode-bit-identity proof -- and the final assertion
# is against the COMPOSED candidate named in ``archive_sha256``.
SPLIT_RECIPE_KEYS = (
    "pre_split_archive_sha256",
    "pre_split_archive_bytes",
    "split_stage_script",
    "split_stage_base",
    "split_stage_profile",
)

#: The SHIPPED receiver member, named as a constant so a recipe author pins the measured
#: value instead of retyping a sha from a comment.  Role and count are from the measured
#: registry cited above; this is the DECODER-only body, never the encoder pin.
RC64_SHIPPED_MEMBER_SHA256 = (
    "05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996"
)
RC64_ROLE_REGISTRY = "reverse_engineering/rc64_backend_role_registry.json"


class RebuildNotExpressible(SystemExit):
    """The requested candidate's build chain is outside this entry point's grammar."""


#: Candidates this script CANNOT rebuild, keyed by archive sha256, with the reason
#: stated as MISSING STAGES rather than as a missing recipe.
#:
#: WHY A REGISTRY AND NOT A GENERIC ERROR.  The cross-pin guard below already refuses a
#: recipe/candidate mismatch, and its advice -- "pass ``--recipe-json``" -- is correct
#: for a candidate this grammar can express.  For a candidate it cannot, that advice is
#: an over-promise: a reader would write a recipe, watch the rebuild fail deep inside
#: the encode stage, and blame the algorithm.  Worse, the packet's accounting lists this
#: entry point as ``ours-original``, so a reviewer may reasonably try the default
#: invocation first.  Naming the gap is the fail-closed answer; silence is the fake one.
NOT_EXPRESSIBLE: dict[str, dict[str, object]] = {
    "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080": {
        "name": "rc2_composed_rider_native_port",
        "archive_bytes": 180_456,
        "missing_stages": [
            "every stage the jg5 body needs (see the jg5 entry below) -- this "
            "candidate's decoded state is byte-identical to jg5's, so it inherits "
            "the whole missing chain rather than replacing it",
            "the RR5 lossless re-encode of the carrier body under an adaptive "
            "arithmetic basis, which sets reserved header flag 0x08 and is what "
            "makes this archive 169 B smaller than jg5's "
            "(runtime/rr5_arith_basis.py on the receiver side)",
        ],
        "receipt": ".omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md",
    },
    "f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e": {
        "name": "jg5_joint_waterfill_455",
        "archive_bytes": 180_625,
        "missing_stages": [
            "seg token edit solve over 573 pairs -- writes the SEMANTIC stream, which "
            "this script copies verbatim (experiments/ddm_jg3_joint_solve.py)",
            "splice of those edits into the br1 body, producing the jg4 candidate body",
            "joint edit-admission waterfill swept over a Lagrange multiplier on pose "
            "damage, 455 of 573 edits admitted "
            "(experiments/ddm_jg5_pose_resolve_on_edited_renders.py)",
            "pose-carrier re-solve against the candidate's OWN renders and the archive "
            "rebuild that re-encodes the carrier stream "
            "(experiments/ddm_up3_carrier_splice.py::build_archive, damped Gauss-Newton "
            "from experiments/ddm_br1_pose_basis_reorientation.py::gn_solve_pair)",
        ],
        "receipt": ".omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md",
    },
    "35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3": {
        "name": "ck1_composed_row_prune",
        "archive_bytes": 177_182,
        "missing_stages": [
            "SM3R mode-6 row-pruned, mixed-depth semantic re-quantization -- re-writes "
            "the semantic section this script copies verbatim "
            "(experiments/ddm_ck1_build_composed_archive.py)",
            "in-compile frame-0 pose compensation folded into the existing Rice-coded "
            "lattice (experiments/ddm_ck1_pose_resolve_kneeA.py)",
        ],
        "receipt": (
            ".omx/research/ddm_pq1_submission_packet_prep_20260815/"
            "BORROWED_SUBSTRATE_ACCOUNTING.md (section 8)"
        ),
    },
}


def refuse_if_not_expressible(expected: ArchiveIdentity) -> None:
    """Refuse a candidate outside this grammar, by name, before any stage runs."""
    entry = NOT_EXPRESSIBLE.get(expected.sha256)
    if entry is None:
        return
    stages = "\n".join(f"    - {stage}" for stage in entry["missing_stages"])
    raise RebuildNotExpressible(
        f"REFUSING: candidate {entry['name']} ({expected.sha256[:16]}…, "
        f"{int(entry['archive_bytes']):,} B, resolved from {expected.source}) is NOT "
        "expressible by this entry point.\n"
        "This script rebuilds the TOKEN stream (optionally plus a declared container "
        "repack) and carries the other seven sections through verbatim. This candidate's "
        "chain also re-decides content in sections this script copies:\n"
        f"{stages}\n"
        "No --recipe-json can close that gap: the missing stages are missing STAGES, not "
        "a missing recipe. Rebuild it with the builders named above.\n"
        f"Receipt: {entry['receipt']}\n"
        "To exercise this script on a candidate it CAN rebuild, pass that candidate's "
        "--expected-archive-sha256/--expected-archive-bytes with its --recipe-json."
    )


def load_recipe(recipe_json: Path | None) -> dict[str, object]:
    """Load the rebuild recipe: rr4's by default, a caller-supplied one otherwise."""
    if recipe_json is None:
        return dict(RR4_RECIPE)
    document = json.loads(Path(recipe_json).read_text())
    recipe = document.get("recipe", document)
    unknown = sorted(
        set(recipe) - set(RECIPE_KEYS) - set(SPLIT_RECIPE_KEYS) - set(RECEIVER_RECIPE_KEYS)
    )
    if unknown:
        raise SystemExit(
            f"recipe carries unknown key(s): {unknown}; expected any of "
            f"{list(RECIPE_KEYS) + list(SPLIT_RECIPE_KEYS) + list(RECEIVER_RECIPE_KEYS)}"
        )
    missing = [key for key in RECIPE_KEYS if key not in recipe]
    if missing:
        raise SystemExit(f"recipe is missing required key(s): {missing}")
    return dict(recipe)


def split_spec(recipe: dict[str, object]) -> dict[str, object] | None:
    """Return the split-stage declaration, or None.  All-or-none, fail-closed."""
    present = [k for k in SPLIT_RECIPE_KEYS if recipe.get(k) not in (None, "")]
    if not present:
        return None
    missing = [k for k in SPLIT_RECIPE_KEYS if k not in present]
    if missing:
        raise SystemExit(
            f"split-stage recipe keys are all-or-none; present {present}, missing {missing}"
        )
    return {k: recipe[k] for k in SPLIT_RECIPE_KEYS}


def input_spec(recipe: dict[str, object]) -> dict[str, dict[str, object]]:
    """Every input the rebuild consumes, with the env var the stage scripts read it from.

    ``sha256`` is verified before any stage runs; ``directory`` entries are verified through
    their named member file.  The expected shas come from the RECIPE, so a caller rebuilding
    a different candidate declares its inputs instead of inheriting rr4's.
    """
    spec: dict[str, dict[str, object]] = {
        "prepared_dir": {
            "env": "TAC_PQ2_PREPARED_DIR",
            "kind": "directory",
            "member": "archive.zip",
            "sha256": recipe["base_archive_sha256"],
            "role": "base archive whose seven non-token sections are carried through unchanged",
        },
        "hm1_dir": {
            "env": "TAC_PQ2_HM1_DIR",
            "kind": "directory",
            "member": "group_index.u8",
            "sha256": None,
            "role": "retained pre-correction HPAC logits, boundary buckets and group index",
        },
        "tokens_file": {
            "env": "TAC_PQ2_TOKENS_FILE",
            "kind": "file",
            "sha256": recipe["decoded_field_sha256"],
            "role": "decoded token field the encoder replays; the distortion anchor",
        },
        "rc64_source": {
            "env": "TAC_PQ2_RC64_SOURCE",
            "kind": "file",
            "sha256": recipe["rc64_source_sha256"],
            "role": (
                "RC64 ENCODER-ROLE C source (encoder + decoder, 12,222 B); the build stage "
                "appends the checkpoint/resume extension and compiles it. NOT the shipped "
                "runtime/entropy member -- see rc64_shipped_member below"
            ),
        },
    }
    # PRESENT-BUT-EMPTY MUST REFUSE, never silently skip.  A recipe author who declares
    # the receiver pin and leaves it blank would otherwise get a verification that quietly
    # checks nothing -- the silent-instrument failure this whole edit exists to end.
    if "rc64_shipped_member_sha256" in recipe:
        shipped = recipe["rc64_shipped_member_sha256"]
        if not isinstance(shipped, str) or not _is_sha256(shipped):
            raise SystemExit(
                "recipe declares rc64_shipped_member_sha256 but its value is not a "
                f"64-character lowercase hex sha256: {shipped!r}. Remove the key or pin "
                "the real shipped runtime/entropy/rc64_backend.c sha."
            )
        spec["rc64_shipped_member"] = {
            "env": "TAC_PQ2_RC64_SHIPPED_MEMBER",
            "kind": "file",
            "sha256": shipped,
            "role": (
                "RC64 SHIPPED receiver member (decoder only, 5,638 B) carried at "
                "runtime/entropy/rc64_backend.c; pinned so both coder roles are custodied "
                "by name rather than one being inherited by accident"
            ),
        }
    return spec


def _is_sha256(value: str) -> bool:
    """A sha256 pin is 64 lowercase hex characters; anything else is not a pin."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_inputs_template(spec: dict[str, dict[str, object]]) -> int:
    """Print the input manifest schema without revealing any local layout."""
    template = {
        name: {
            "path": f"<absolute path to the {entry['kind']}>",
            "role": entry["role"],
            "expected_sha256": entry["sha256"],
            "environment_variable": entry["env"],
        }
        for name, entry in spec.items()
    }
    print(json.dumps({"inputs": template}, indent=2, sort_keys=True))
    return 0


def resolve_expected_archive(
    expected_sha256: str,
    expected_bytes: int | None,
    candidate_runtime: Path | None,
) -> ArchiveIdentity:
    """Resolve WHICH candidate this rebuild must reproduce, at run time. Never a latched literal.

    Three sources, in priority order, each of which names the candidate explicitly:

      1. ``--expected-archive-sha256`` + ``--expected-archive-bytes`` — the operator says it.
      2. ``--candidate-runtime <dir>`` — measured from a sealed candidate tree, whose own
         receiver pin must agree with its archive first (archive+runtime are ONE object).
      3. the canonical frontier pointer — "whatever we currently ship", read now, so the bar
         moves when the pointer moves instead of rotting inside this file.
    """
    if expected_sha256 or expected_bytes is not None:
        if not expected_sha256 or expected_bytes is None:
            raise SystemExit(
                "--expected-archive-sha256 and --expected-archive-bytes must be supplied together; "
                "half an identity cannot verify a rebuild"
            )
        return ArchiveIdentity(sha256=expected_sha256.lower(), bytes=int(expected_bytes), source="cli")

    if candidate_runtime is not None:
        seal = check_pin_consistency(candidate_runtime)
        if seal.verdict != CONSISTENT:
            raise SystemExit(
                f"--candidate-runtime {candidate_runtime} is not a sealed tree: {seal.summary()}"
            )
        identity = measure_archive_identity(candidate_runtime / "archive.zip")
        return ArchiveIdentity(
            sha256=identity.sha256, bytes=identity.bytes, source=f"candidate_runtime:{candidate_runtime}"
        )

    try:
        return read_frontier_archive_identity()
    except SealContractError as exc:
        raise SystemExit(
            f"could not derive the expected archive from the canonical frontier pointer: {exc}\n"
            "Pass --expected-archive-sha256/--expected-archive-bytes or --candidate-runtime."
        ) from exc


def resolve_inputs(spec: dict[str, dict[str, object]], inputs_json: Path | None) -> dict[str, Path]:
    """Resolve every input root from the manifest, falling back to the environment.

    Fail-closed on a missing input: a rebuild that silently skips an input would
    produce different bytes and blame the algorithm.
    """
    supplied: dict[str, str] = {}
    if inputs_json is not None:
        document = json.loads(inputs_json.read_text())
        section = document.get("inputs", document)
        for name, entry in section.items():
            if name in spec:
                supplied[name] = entry["path"] if isinstance(entry, dict) else str(entry)

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name, entry in spec.items():
        raw = supplied.get(name) or os.environ.get(entry["env"])
        if not raw:
            missing.append(f"{name} (--inputs-json entry or ${entry['env']})")
            continue
        resolved[name] = Path(raw)
    if missing:
        raise SystemExit(
            "missing required inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun --emit-inputs-template for the schema."
        )
    return resolved


def verify_inputs(spec: dict[str, dict[str, object]], resolved: dict[str, Path]) -> list[dict[str, object]]:
    """Hash each input and refuse to proceed on a mismatch or an absent file.

    REPORTS ITS DENOMINATOR.  A verification loop over an empty or partly-unpinned spec
    passes silently, and a silent pass reads exactly like a real one.  So this prints
    ``verified N/M inputs, P pinned`` and refuses an empty spec outright: vacuity is a
    refusal here, never a green.
    """
    if not spec:
        raise SystemExit(
            "input spec is empty: there is nothing to verify, and an empty verification "
            "is not a passing one. Check the loaded recipe."
        )
    manifest: list[dict[str, object]] = []
    problems: list[str] = []
    for name, entry in spec.items():
        root = resolved[name]
        target = root / entry["member"] if entry["kind"] == "directory" else root
        if not target.is_file():
            problems.append(f"{name}: not found -> {target}")
            continue
        measured = sha256_file(target)
        expected = entry["sha256"]
        ok = expected is None or measured == expected
        if not ok:
            problems.append(f"{name}: sha256 {measured} != expected {expected}")
        manifest.append(
            {
                "input": name,
                "verified_member": target.name,
                "bytes": target.stat().st_size,
                "sha256": measured,
                "expected_sha256": expected,
                "sha256_matches": ok,
                "role": entry["role"],
            }
        )
    if problems:
        raise SystemExit("input verification failed:\n  " + "\n  ".join(problems))
    pinned = sum(1 for entry in spec.values() if entry["sha256"] is not None)
    print(
        f"[pq2] verified {len(manifest)}/{len(spec)} inputs, {pinned} of them sha256-pinned "
        f"({len(spec) - pinned} present-but-unpinned by recipe design)",
        flush=True,
    )
    return manifest


def stage_environment(
    spec: dict[str, dict[str, object]],
    recipe: dict[str, object],
    resolved: dict[str, Path],
    seed: int,
) -> dict[str, str]:
    """Build the child environment: input roots, corrector selection, and the seed."""
    environment = dict(os.environ)
    for name, entry in spec.items():
        environment[entry["env"]] = str(resolved[name])
    environment["TAC_RR2_CORRECTOR_MODULE"] = str(recipe["corrector_module"])
    environment["PYTHONHASHSEED"] = str(seed)
    environment["TAC_PQ2_SEED"] = str(seed)
    # Stage scripts run with experiments/ as sys.path[0]; correctors that import
    # through the ``experiments.`` package (fx2's mixer imports fx1's) need the
    # repo root importable too.  Guaranteeing it HERE keeps every pinned source
    # file byte-identical instead of editing a frozen corrector's import lines.
    environment["PYTHONPATH"] = str(REPO) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return environment


def run_stage(argv: list[str], environment: dict[str, str], label: str) -> dict[str, object]:
    """Run one real stage script and fail closed on a non-zero return code."""
    print(f"[pq2] {label}: {' '.join(argv)}", flush=True)
    started = time.time()
    completed = subprocess.run(argv, env=environment, cwd=str(REPO), check=False)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with return code {completed.returncode}")
    return {"stage": label, "argv": argv, "returncode": 0, "elapsed_seconds": elapsed}


def verify_archive(store: Path, expected: ArchiveIdentity, recipe: dict[str, object]) -> dict[str, object]:
    """Assert the rebuilt archive is byte-identical to the candidate resolved at run time."""
    archive = store / "retained" / "archive.zip"
    if not archive.is_file():
        raise SystemExit(f"no rebuilt archive at {archive}; run the build stage first")
    measured_sha = sha256_file(archive)
    measured_bytes = archive.stat().st_size
    sha_ok = measured_sha == expected.sha256
    bytes_ok = measured_bytes == expected.bytes

    repeat = store / "work" / "archive.repeat.zip"
    repeat_sha = sha256_file(repeat) if repeat.is_file() else None
    deterministic = repeat_sha == measured_sha if repeat_sha is not None else None

    token = store / "retained" / "token_stream.bin"
    token_sha = sha256_file(token) if token.is_file() else None

    result = {
        "archive_path": str(archive),
        "archive_sha256": measured_sha,
        "archive_bytes": measured_bytes,
        "expected_archive_sha256": expected.sha256,
        "expected_archive_bytes": expected.bytes,
        "expected_archive_source": expected.source,
        "sha256_matches": sha_ok,
        "bytes_match": bytes_ok,
        "determinism_repeat_sha256": repeat_sha,
        "determinism_repeat_byte_identical": deterministic,
        "token_stream_sha256": token_sha,
        "token_stream_matches": None if token_sha is None else token_sha == recipe["token_sha256"],
    }
    if not (sha_ok and bytes_ok):
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(
            "ARCHIVE VERIFICATION FAILED: rebuilt bytes do not match the pinned candidate"
        )
    print(
        f"[pq2] VERIFIED archive sha256={measured_sha} bytes={measured_bytes:,}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("provenance", "encode", "build", "verify", "decode", "all"),
        help="'all' runs provenance, encode, build and verify (not decode)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="working/output root; retained/archive.zip is written under it",
    )
    parser.add_argument(
        "--inputs-json",
        type=Path,
        default=None,
        help="input manifest; see --emit-inputs-template",
    )
    parser.add_argument(
        "--emit-inputs-template",
        action="store_true",
        help="print the input manifest schema and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the encode bit-faithfully from a retained frame checkpoint",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="write the run receipt here (default: <store>/RESULT_pq2_e2e.json)",
    )
    parser.add_argument(
        "--expected-archive-sha256",
        default="",
        help="the candidate this rebuild must reproduce; with --expected-archive-bytes",
    )
    parser.add_argument(
        "--expected-archive-bytes",
        type=int,
        default=None,
        help="byte count of the candidate this rebuild must reproduce",
    )
    parser.add_argument(
        "--candidate-runtime",
        type=Path,
        default=None,
        help="derive the expected archive from a sealed candidate runtime tree "
        "(its receiver pin must already agree with its archive)",
    )
    parser.add_argument(
        "--recipe-json",
        type=Path,
        default=None,
        help="rebuild recipe for a NON-rr4 candidate (corrector module, token/base/decoded "
        "shas); required when the expected archive is not rr4's",
    )
    args = parser.parse_args()

    recipe = load_recipe(args.recipe_json)
    spec = input_spec(recipe)

    if args.emit_inputs_template:
        return emit_inputs_template(spec)

    if args.store is None:
        parser.error("--store is required (or pass --emit-inputs-template)")

    expected = resolve_expected_archive(
        args.expected_archive_sha256, args.expected_archive_bytes, args.candidate_runtime
    )

    # GRAMMAR GUARD, checked BEFORE the cross-pin guard. If the candidate is outside this
    # script's grammar the honest message names the missing STAGES; the cross-pin guard's
    # "pass --recipe-json" would be an over-promise, and reaching it first would hide the
    # real reason behind a fixable-looking one.
    refuse_if_not_expressible(expected)

    # CROSS-PIN GUARD. The recipe describes the pipeline; the expected identity names the
    # product. Reproducing candidate X while still asserting rr4's token stream, decoded
    # field and corrector is the wrong-object error in its most expensive form -- it would
    # fail deep inside the rebuild and blame the algorithm. Refuse it up front, by name.
    if expected.sha256 != recipe["archive_sha256"]:
        raise SystemExit(
            f"REFUSING a recipe/candidate mismatch: the expected archive is "
            f"{expected.sha256[:16]}… ({expected.bytes:,} B, from {expected.source}) but the loaded "
            f"rebuild recipe is {recipe['name']!r}, which produces {str(recipe['archive_sha256'])[:16]}… "
            f"({int(recipe['archive_bytes']):,} B).\n"
            "A different candidate needs its own recipe: pass --recipe-json, or point "
            "--expected-archive-sha256/--expected-archive-bytes at the candidate this recipe builds."
        )

    split = split_spec(recipe)
    if split is None:
        build_expected = expected
    else:
        # With a split stage, the token-only build produces the PRE-split archive;
        # the recipe's ``archive_sha256`` names the COMPOSED product the split emits.
        build_expected = ArchiveIdentity(
            sha256=str(split["pre_split_archive_sha256"]).lower(),
            bytes=int(split["pre_split_archive_bytes"]),
            source="recipe:pre_split",
        )

    store = args.store
    store.mkdir(parents=True, exist_ok=True)
    resolved = resolve_inputs(spec, args.inputs_json)
    input_manifest = verify_inputs(spec, resolved)
    environment = stage_environment(spec, recipe, resolved, args.seed)

    receipt: dict[str, object] = {
        "schema": "pact.pq2.compress_e2e.v1",
        "stage_requested": args.stage,
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "seed": args.seed,
        "corrector_module": recipe["corrector_module"],
        "recipe_name": recipe["name"],
        "expected_archive": expected.to_dict(),
        "python_version": sys.version,
        "store": str(store),
        "input_manifest": input_manifest,
        "stages_run": [],
    }

    if args.stage in ("provenance", "all"):
        receipt["provenance"] = {
            "stage_a_status": "DOCUMENTED_NOT_RE_RUN",
            "stage_a_note": (
                "Reproducing the checkpoint from raw video is multi-day GPU compute. "
                "Stage A records the lineage; stages B and C are exactly verifiable "
                "from the retained checkpoint and are what determine the archive bytes."
            ),
            "encoder_script": str(ENCODER.relative_to(REPO)),
            "receiver_script": str(RECEIVER.relative_to(REPO)),
            "expected_archive_sha256": expected.sha256,
            "expected_archive_bytes": expected.bytes,
            "expected_archive_source": expected.source,
            "expected_token_bytes": recipe["token_bytes"],
            "expected_decoded_field_sha256": recipe["decoded_field_sha256"],
            "sections_rebuilt": "token_stream only; seven sections carried byte-identically",
        }
        print(json.dumps(receipt["provenance"], indent=2, sort_keys=True))

    if args.stage in ("encode", "all"):
        argv = [
            sys.executable,
            str(ENCODER),
            "--stage",
            "encode",
            "--store",
            str(store),
        ]
        if args.resume:
            argv.append("--resume")
        receipt["stages_run"].append(run_stage(argv, environment, "encode"))

    if args.stage in ("build", "all"):
        argv = [sys.executable, str(ENCODER), "--stage", "build", "--store", str(store)]
        receipt["stages_run"].append(run_stage(argv, environment, "build"))

    if args.stage in ("verify", "build", "all"):
        receipt["verification"] = verify_archive(store, build_expected, recipe)

    if split is not None and args.stage in ("verify", "build", "all"):
        script = REPO / "experiments" / str(split["split_stage_script"])
        if not script.is_file():
            raise SystemExit(f"split-stage script not found: {script}")
        out_root = store / "split_stage"
        if args.stage in ("build", "all"):
            argv = [
                sys.executable,
                str(script),
                "--base",
                str(split["split_stage_base"]),
                "--profile",
                str(split["split_stage_profile"]),
                "--out-root",
                str(out_root),
            ]
            receipt["stages_run"].append(run_stage(argv, environment, "split"))
        final = (
            out_root
            / f"archives/{split['split_stage_base']}__{split['split_stage_profile']}"
            / "archive.zip"
        )
        if not final.is_file():
            raise SystemExit(f"no split-stage archive at {final}; run the build stage first")
        final_sha = sha256_file(final)
        final_bytes = final.stat().st_size
        report_path = final.parent / "BUILD_REPORT.json"
        receipt["split_verification"] = {
            "archive_path": str(final),
            "archive_sha256": final_sha,
            "archive_bytes": final_bytes,
            "expected_archive_sha256": expected.sha256,
            "expected_archive_bytes": expected.bytes,
            "sha256_matches": final_sha == expected.sha256,
            "bytes_match": final_bytes == expected.bytes,
            "build_report": json.loads(report_path.read_text()) if report_path.is_file() else None,
        }
        if not (final_sha == expected.sha256 and final_bytes == expected.bytes):
            print(json.dumps(receipt["split_verification"], indent=2, sort_keys=True))
            raise SystemExit(
                "SPLIT-STAGE VERIFICATION FAILED: composed bytes do not match the pinned candidate"
            )
        print(
            f"[pq2] VERIFIED composed archive sha256={final_sha} bytes={final_bytes:,}",
            flush=True,
        )

    if args.stage == "decode":
        for sub in ("build", "parseback"):
            argv = [
                sys.executable,
                str(RECEIVER),
                "--stage",
                sub,
                "--store",
                str(store),
            ]
            receipt["stages_run"].append(run_stage(argv, environment, f"receiver-{sub}"))
        receipt["decode_note"] = (
            "receiver parse-back writes RESULT_receiver_parseback.json; the decoded "
            f"token field must hash to {recipe['decoded_field_sha256']}"
        )

    destination = args.receipt or (store / "RESULT_pq2_e2e.json")
    partial = destination.with_suffix(".partial")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    os.replace(partial, destination)
    print(f"[pq2] receipt -> {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
