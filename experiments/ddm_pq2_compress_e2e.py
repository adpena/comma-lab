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

      The shipping AFR1 candidate is also expressible through the ordered
      ``AFR1_CHAIN`` registry below.  Its five post-rc2 lossless stages run their
      real landed tools, retain every intermediate archive, and refuse unless two
      complete runs both reproduce the pinned 180,002-byte archive exactly.

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
import shutil
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

# The shipping lossless chain.  ``gb1`` is intentionally typed as a fork: the
# pointer row is ``groupbin8_surprise``, while lb1 consumes the separately retained
# ``jt21`` bank emitted by the same full-n600 collection.  Flattening those two
# outputs into a single additive chain would misstate the measured lineage.
AFR1_CHAIN: dict[str, object] = {
    "name": "afr1_from_retained_rc2",
    "archive_sha256": "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25",
    "archive_bytes": 180_002,
    "base_sha256": "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080",
    "base_bytes": 180_456,
    "stages": (
        {
            "name": "fx5",
            "tool": "experiments/ddm_fx5_build_e1_runtime.py",
            "input_sha256": "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080",
            "output_sha256": "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841",
            "output_bytes": 180_386,
            "receipt": ".omx/research/ddm_fx5_composed_rate_candidate_20260821.md",
        },
        {
            "name": "dx2",
            "tool": "experiments/ddm_dx2_cabac_receiver_fold.py",
            "input_sha256": "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841",
            "output_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
            "output_bytes": 180_368,
            "receipt": ".omx/research/ddm_dx2_cabac_receiver_fold_20260821.md",
        },
        {
            "name": "gb1",
            "tool": "experiments/ddm_gb1_groupbin8_conditioning.py",
            "input_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
            "output_sha256": "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
            "output_bytes": 180_192,
            "branch_outputs": (
                {
                    "name": "gb1_pointer",
                    "sha256": "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4",
                    "bytes": 180_215,
                },
                {
                    "name": "jt21_bank_consumed_by_lb1",
                    "sha256": "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
                    "bytes": 180_192,
                },
            ),
            "receipt": ".omx/research/ddm_gb1_groupbin8_conditioning_20260824.md",
        },
        {
            "name": "lb1",
            "tool": "experiments/ddm_lb1_banked_lossless_joint_collect.py",
            "input_sha256": "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3",
            "output_sha256": "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9",
            "output_bytes": 180_083,
            "receipt": ".omx/research/ddm_lb1_banked_lossless_joint_collect_20260829.md",
        },
        {
            "name": "afr1",
            "tool": "experiments/ddm_afr1_tile48_receiver_identity.py",
            "input_sha256": "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9",
            "output_sha256": "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25",
            "output_bytes": 180_002,
            "receipt": ".omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md",
        },
    ),
}

CHAIN_RECIPES = {str(AFR1_CHAIN["archive_sha256"]): AFR1_CHAIN}

# Inputs are paths supplied by ``--inputs-json``; only roles and content pins live
# here.  Reference archives are used solely to name the first differing offset on
# a refusal.  A stage output is always produced by a real tool before it is compared.
AFR1_CHAIN_INPUTS: dict[str, dict[str, object]] = {
    "rc2_runtime": {
        "env": "TAC_CE1_RC2_RUNTIME",
        "kind": "directory",
        "member": "archive.zip",
        "sha256": AFR1_CHAIN["base_sha256"],
        "role": "retained rc2 runtime at the legal Stage-A boundary",
    },
    "tokens_file": {
        "env": "TAC_CE1_TOKENS_FILE",
        "kind": "file",
        "sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
        "role": "full 600-frame decoded token field replayed by every RC64 stage",
    },
    "rc64_source": {
        "env": "TAC_CE1_RC64_SOURCE",
        "kind": "file",
        "sha256": "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6",
        "role": "RC64 encoder-bearing source, pinned by role",
    },
    "rc64_shipped_member": {
        "env": "TAC_CE1_RC64_SHIPPED_MEMBER",
        "kind": "file",
        "sha256": RC64_SHIPPED_MEMBER_SHA256,
        "role": "RC64 decoder-only shipped member, pinned separately by role",
    },
    "lb1_base_runtime": {
        "env": "TAC_CE1_LB1_BASE_RUNTIME",
        "kind": "directory",
        "member": "runtime/fx2_model_axis_corrector.py",
        "sha256": "06cc74279e485e2d73558b2ea5ec9a5c68606e685231c10bbf5ef1bac5c2f296",
        "role": "jt21 21-family Python runtime produced by the lb1 prepare tool",
    },
    "gb1_joint_runtime": {
        "env": "TAC_CE1_GB1_JOINT_RUNTIME",
        "kind": "directory",
        "member": "runtime/fx2_model_axis_corrector.py",
        "sha256": "06cc74279e485e2d73558b2ea5ec9a5c68606e685231c10bbf5ef1bac5c2f296",
        "role": (
            "admitted 21-family runtime carrying both groupbin8_surprise and "
            "cls_groupbin8; the exact jt21 bank mechanism"
        ),
    },
    "lb1_candidate_runtime": {
        "env": "TAC_CE1_LB1_CANDIDATE_RUNTIME",
        "kind": "directory",
        "member": "runtime/fx2_model_axis_corrector.py",
        "sha256": "460490e427e54d89f0a074d785cb8bd7678df509215be1d2a37a6f5a6f617a75",
        "role": "lb1 22-family Python runtime produced by the lb1 prepare tool",
    },
    "afr1_base_runtime": {
        "env": "TAC_CE1_AFR1_BASE_RUNTIME",
        "kind": "directory",
        "member": "archive.zip",
        "sha256": "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9",
        "role": "pin-consistent lb1 runtime used for the AFR1 null control",
    },
    "afr1_source_runtime": {
        "env": "TAC_CE1_AFR1_SOURCE_RUNTIME",
        "kind": "directory",
        "member": "runtime/fx2_model_axis_corrector.py",
        "sha256": "6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064",
        "role": (
            "AFR1 identity-tool runtime supplying the receipt-pinned tile48_groupbin8 "
            "Python corrector; the chain binds it freshly to each run's lb1 archive"
        ),
    },
    **{
        f"reference_{name}": {
            "env": f"TAC_CE1_REFERENCE_{name.upper()}",
            "kind": "file",
            "sha256": sha,
            "role": f"receipt-pinned {name} archive used only for byte-diff diagnostics",
        }
        for name, sha in (
            ("fx5", "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"),
            ("dx2", "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"),
            ("gb1", "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"),
            ("jt21", "ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3"),
            ("lb1", "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9"),
            ("afr1", "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"),
        )
    },
}


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


class ChainStageError(RuntimeError):
    """One real chain stage produced bytes other than its receipt pin."""

    def __init__(self, message: str, blocker: dict[str, object]):
        super().__init__(message)
        self.blocker = blocker


def chain_input_spec() -> dict[str, dict[str, object]]:
    """Return a copy so callers/tests cannot mutate the shipping chain contract."""
    return {name: dict(entry) for name, entry in AFR1_CHAIN_INPUTS.items()}


def file_fact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ChainStageError(
            f"required stage output is absent: {path}",
            {"status": "REFUSED", "reason": "OUTPUT_ABSENT", "path": str(path)},
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def first_differing_offset(left: Path, right: Path) -> int | None:
    """Return the first byte offset that differs, including a length-only mismatch."""
    offset = 0
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a = first.read(1 << 20)
            b = second.read(1 << 20)
            common = min(len(a), len(b))
            for index in range(common):
                if a[index] != b[index]:
                    return offset + index
            if len(a) != len(b):
                return offset + common
            if not a:
                return None
            offset += len(a)


def assert_chain_output(
    *,
    stage: dict[str, object],
    output: Path,
    reference: Path,
) -> dict[str, object]:
    """Hash a produced archive and name the exact divergence on refusal."""
    observed = file_fact(output)
    expected_sha = str(stage["output_sha256"])
    expected_bytes = int(stage["output_bytes"])
    if observed["sha256"] != expected_sha or observed["bytes"] != expected_bytes:
        blocker = {
            "status": "REFUSED",
            "stage": stage["name"],
            "reason": "ARCHIVE_PIN_DIVERGENCE",
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
            "observed": observed,
            "reference": file_fact(reference),
            "first_differing_offset": first_differing_offset(output, reference),
        }
        raise ChainStageError(
            f"{stage['name']} produced {observed['bytes']}/{observed['sha256']}, "
            f"expected {expected_bytes}/{expected_sha}",
            blocker,
        )
    return observed


def retain_archive(source: Path, destination: Path) -> dict[str, object]:
    """Retain a generated archive; this is custody, never a stage substitute."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if source.read_bytes() != destination.read_bytes():
        raise ChainStageError(
            f"retained copy differs from generated archive: {source} -> {destination}",
            {
                "status": "REFUSED",
                "reason": "RETENTION_COPY_DIVERGENCE",
                "source": str(source),
                "destination": str(destination),
                "first_differing_offset": first_differing_offset(source, destination),
            },
        )
    return file_fact(destination)


def jg2_argv(
    *,
    stage: str,
    store: Path,
    runtime: Path,
    pointer: Path,
    pointer_sha256: str,
    tokens: Path,
    tag: str | None = None,
    resume: bool = False,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO / "experiments/ddm_jg2_tail_reencode.py"),
        "--stage",
        stage,
        "--store",
        str(store),
        "--runtime-root",
        str(runtime),
        "--pointer-archive",
        str(pointer),
        "--expect-pointer-sha256",
        pointer_sha256,
        "--tokens",
        str(tokens),
        "--frames",
        "600",
        "--checkpoint-every",
        "25",
    ]
    if tag is not None:
        argv.extend(("--tag", tag))
    if resume:
        argv.append("--resume")
    return argv


def assert_runtime_archive(runtime: Path, expected: Path, label: str) -> None:
    candidate = runtime / "archive.zip"
    if not candidate.is_file() or candidate.read_bytes() != expected.read_bytes():
        raise ChainStageError(
            f"{label} runtime is not bound to the preceding stage archive",
            {
                "status": "REFUSED",
                "stage": label,
                "reason": "RUNTIME_ARCHIVE_HANDOFF_DIVERGENCE",
                "runtime_archive": file_fact(candidate) if candidate.is_file() else None,
                "preceding_archive": file_fact(expected),
                "first_differing_offset": (
                    first_differing_offset(candidate, expected) if candidate.is_file() else None
                ),
            },
        )


def execute_afr1_chain_run(
    *,
    run_number: int,
    store: Path,
    inputs: dict[str, Path],
    environment: dict[str, str],
    resume: bool,
) -> dict[str, object]:
    """Run the five receipt-pinned stages once, in order, with full-n600 encodes."""
    run_name = f"run_{run_number}"
    work = store / "work" / run_name
    retained = store / "retained" / run_name
    if work.exists() and not resume:
        raise ChainStageError(
            f"chain work root already exists: {work}; pass --resume or use a fresh store",
            {"status": "REFUSED", "reason": "WORK_ROOT_EXISTS", "path": str(work)},
        )
    work.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)
    tokens = inputs["tokens_file"]
    stages = {str(row["name"]): dict(row) for row in AFR1_CHAIN["stages"]}
    commands: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    started = time.time()

    def run(argv: list[str], label: str) -> None:
        try:
            outcome = run_stage(argv, environment, f"{run_name}:{label}")
        except SystemExit as error:
            raise ChainStageError(
                f"{label} did not complete: {error}",
                {
                    "status": "REFUSED",
                    "stage": label,
                    "reason": "STAGE_TOOL_FAILURE",
                    "error": str(error),
                    "argv": argv,
                },
            ) from error
        commands.append(outcome)

    def resume_stage(label: str, artifact: Path) -> None:
        commands.append(
            {
                "stage": f"{run_name}:{label}",
                "returncode": 0,
                "elapsed_seconds": 0.0,
                "resumed": True,
                "artifact": file_fact(artifact),
            }
        )

    # FX5: the named tool builds the exact 19-member runtime; jg2 is the landed
    # physical archive writer for that runtime.
    fx5_store = work / "fx5"
    fx5_runtime = fx5_store / "runtime_fx5"
    rc2_archive = inputs["rc2_runtime"] / "archive.zip"
    if resume and fx5_runtime.is_dir():
        resume_stage("fx5-build-runtime", fx5_runtime / "runtime/fx2_model_axis_corrector.py")
        temporary_archive = fx5_runtime / "archive.zip.resume-partial"
        shutil.copy2(rc2_archive, temporary_archive)
        os.replace(temporary_archive, fx5_runtime / "archive.zip")
        assert_runtime_archive(fx5_runtime, rc2_archive, "fx5-resume-input")
    else:
        run(
            [
                sys.executable,
                str(REPO / str(stages["fx5"]["tool"])),
                "--base",
                str(inputs["rc2_runtime"]),
                "--out",
                str(fx5_runtime),
            ],
            "fx5-build-runtime",
        )
    run(
        jg2_argv(
            stage="control",
            store=fx5_store,
            runtime=inputs["rc2_runtime"],
            pointer=rc2_archive,
            pointer_sha256=str(AFR1_CHAIN["base_sha256"]),
            tokens=tokens,
            resume=resume,
        ),
        "fx5-control",
    )
    run(
        jg2_argv(
            stage="encode",
            store=fx5_store,
            runtime=fx5_runtime,
            pointer=rc2_archive,
            pointer_sha256=str(AFR1_CHAIN["base_sha256"]),
            tokens=tokens,
            tag="fx5_e1_19member",
            resume=resume,
        ),
        "fx5-encode",
    )
    fx5_archive = fx5_store / "retained/candidate_fx5_e1_19member.zip"
    fx5_fact = assert_chain_output(
        stage=stages["fx5"], output=fx5_archive, reference=inputs["reference_fx5"]
    )
    retain_archive(fx5_archive, retained / "01_fx5.zip")
    shutil.copy2(fx5_archive, fx5_runtime / "archive.zip")
    run(
        [
            sys.executable,
            str(REPO / str(stages["fx5"]["tool"])),
            "--repin",
            str(fx5_runtime),
        ],
        "fx5-repin",
    )
    outputs.append({"stage": "fx5", "archive": fx5_fact})

    # DX2's builder is itself the complete exact fold and retains its own repeat.
    dx2_store = work / "dx2"
    dx2_archive = dx2_store / "retained/candidate_dx2_cabac.zip"
    if resume and dx2_archive.is_file():
        assert_chain_output(
            stage=stages["dx2"], output=dx2_archive, reference=inputs["reference_dx2"]
        )
        resume_stage("dx2-fold", dx2_archive)
    else:
        run(
            [
                sys.executable,
                str(REPO / str(stages["dx2"]["tool"])),
                "--out-dir",
                str(dx2_store),
            ],
            "dx2-fold",
        )
    dx2_fact = assert_chain_output(
        stage=stages["dx2"], output=dx2_archive, reference=inputs["reference_dx2"]
    )
    retain_archive(dx2_archive, retained / "02_dx2.zip")
    outputs.append({"stage": "dx2", "archive": dx2_fact})

    # GB1 is one measured collection with two outputs.  The pointer archive is
    # retained, while the jt21 bank is the typed handoff consumed by lb1.
    gb1_store = work / "gb1"
    gb1_measurement = gb1_store / "measurement"
    gb1_pointer_runtime = gb1_store / "runtime_groupbin8_surprise"
    jt21_runtime = gb1_store / "runtime_cls_groupbin8"
    gb1_tool = str(REPO / str(stages["gb1"]["tool"]))
    for member, destination in (
        ("groupbin8_surprise", gb1_pointer_runtime),
        ("cls_groupbin8", jt21_runtime),
    ):
        if resume and destination.is_dir():
            resume_stage(
                f"gb1-patch-{member}", destination / "runtime/fx2_model_axis_corrector.py"
            )
        else:
            run(
                [
                    sys.executable,
                    gb1_tool,
                    "--stage",
                    "patch",
                    "--out",
                    str(gb1_measurement),
                    "--member",
                    member,
                    "--destination",
                    str(destination),
                ],
                f"gb1-patch-{member}",
            )
    dx2_runtime = dx2_store / "candidate_runtime_dx2"
    run(
        jg2_argv(
            stage="control",
            store=gb1_store,
            runtime=dx2_runtime,
            pointer=dx2_archive,
            pointer_sha256=str(stages["gb1"]["input_sha256"]),
            tokens=tokens,
            resume=resume,
        ),
        "gb1-control",
    )
    assert_runtime_archive(inputs["gb1_joint_runtime"], dx2_archive, "gb1-joint21")
    for tag, runtime in (
        ("gb1_groupbin8_surprise", gb1_pointer_runtime),
        ("gb1_joint21", inputs["gb1_joint_runtime"]),
    ):
        run(
            jg2_argv(
                stage="encode",
                store=gb1_store,
                runtime=runtime,
                pointer=dx2_archive,
                pointer_sha256=str(stages["gb1"]["input_sha256"]),
                tokens=tokens,
                tag=tag,
                resume=resume,
            ),
            f"gb1-encode-{tag}",
        )
    gb1_archive = gb1_store / "retained/candidate_gb1_groupbin8_surprise.zip"
    jt21_archive = gb1_store / "retained/candidate_gb1_joint21.zip"
    gb1_branch = dict(stages["gb1"])
    gb1_branch.update(
        output_sha256="ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4",
        output_bytes=180_215,
    )
    gb1_fact = assert_chain_output(
        stage=gb1_branch, output=gb1_archive, reference=inputs["reference_gb1"]
    )
    jt21_fact = assert_chain_output(
        stage=stages["gb1"], output=jt21_archive, reference=inputs["reference_jt21"]
    )
    retain_archive(gb1_archive, retained / "03a_gb1_pointer.zip")
    retain_archive(jt21_archive, retained / "03b_jt21_bank.zip")
    outputs.append(
        {
            "stage": "gb1",
            "pointer_archive": gb1_fact,
            "consumed_jt21_bank": jt21_fact,
        }
    )

    # LB1's named tool re-derives the bank and prepares both runtimes.  The full
    # control and joint encode are then rerun into this chain's own retained store.
    lb1_store = work / "lb1"
    run(
        [
            sys.executable,
            str(REPO / str(stages["lb1"]["tool"])),
            "--stage",
            "prepare",
        ],
        "lb1-prepare",
    )
    assert_runtime_archive(inputs["lb1_base_runtime"], inputs["reference_jt21"], "lb1-base")
    assert_runtime_archive(
        inputs["lb1_candidate_runtime"], inputs["reference_jt21"], "lb1-candidate"
    )
    run(
        jg2_argv(
            stage="control",
            store=lb1_store,
            runtime=inputs["lb1_base_runtime"],
            pointer=jt21_archive,
            pointer_sha256=str(stages["lb1"]["input_sha256"]),
            tokens=tokens,
            resume=resume,
        ),
        "lb1-control",
    )
    run(
        jg2_argv(
            stage="encode",
            store=lb1_store,
            runtime=inputs["lb1_candidate_runtime"],
            pointer=jt21_archive,
            pointer_sha256=str(stages["lb1"]["input_sha256"]),
            tokens=tokens,
            tag="lb1_joint22_patch192",
            resume=resume,
        ),
        "lb1-encode",
    )
    lb1_archive = lb1_store / "retained/candidate_lb1_joint22_patch192.zip"
    lb1_fact = assert_chain_output(
        stage=stages["lb1"], output=lb1_archive, reference=inputs["reference_lb1"]
    )
    retain_archive(lb1_archive, retained / "04_lb1.zip")
    outputs.append({"stage": "lb1", "archive": lb1_fact})

    # AFR1's tool revalidates the retained physical control and native binding.
    # jg2 then reruns the full Python-authority archive mechanism on this run's lb1 bytes.
    afr1_store = work / "afr1"
    afr1_tool = str(REPO / str(stages["afr1"]["tool"]))
    for substage in ("control", "port", "parity", "byte-close"):
        run([sys.executable, afr1_tool, "--stage", substage], f"afr1-{substage}")
    assert_runtime_archive(inputs["afr1_base_runtime"], inputs["reference_lb1"], "afr1-base")
    afr1_encoding_runtime = afr1_store / "runtime_tile48_groupbin8"
    if afr1_encoding_runtime.exists():
        if not resume:
            raise ChainStageError(
                f"AFR1 encoding runtime already exists: {afr1_encoding_runtime}",
                {
                    "status": "REFUSED",
                    "stage": "afr1",
                    "reason": "RUNTIME_EXISTS_WITHOUT_RESUME",
                    "path": str(afr1_encoding_runtime),
                },
            )
    else:
        temporary = afr1_encoding_runtime.with_name(afr1_encoding_runtime.name + ".partial")
        shutil.copytree(inputs["afr1_base_runtime"], temporary, copy_function=shutil.copy2)
        source_corrector = inputs["afr1_source_runtime"] / "runtime/fx2_model_axis_corrector.py"
        shutil.copy2(source_corrector, temporary / "runtime/fx2_model_axis_corrector.py")
        os.replace(temporary, afr1_encoding_runtime)
    assert_runtime_archive(afr1_encoding_runtime, lb1_archive, "afr1-encoding-runtime")
    encoding_corrector = file_fact(
        afr1_encoding_runtime / "runtime/fx2_model_axis_corrector.py"
    )
    if encoding_corrector["sha256"] != AFR1_CHAIN_INPUTS["afr1_source_runtime"]["sha256"]:
        raise ChainStageError(
            "fresh AFR1 encoding runtime does not carry the receipt-pinned corrector",
            {
                "status": "REFUSED",
                "stage": "afr1",
                "reason": "CORRECTOR_BRIDGE_DIVERGENCE",
                "observed": encoding_corrector,
            },
        )
    run(
        jg2_argv(
            stage="control",
            store=afr1_store,
            runtime=inputs["afr1_base_runtime"],
            pointer=lb1_archive,
            pointer_sha256=str(stages["afr1"]["input_sha256"]),
            tokens=tokens,
            resume=resume,
        ),
        "afr1-control-full-n600",
    )
    run(
        jg2_argv(
            stage="encode",
            store=afr1_store,
            runtime=afr1_encoding_runtime,
            pointer=lb1_archive,
            pointer_sha256=str(stages["afr1"]["input_sha256"]),
            tokens=tokens,
            tag="afr1_tile48_groupbin8_ce1",
            resume=resume,
        ),
        "afr1-encode-full-n600",
    )
    afr1_archive = afr1_store / "retained/candidate_afr1_tile48_groupbin8_ce1.zip"
    afr1_fact = assert_chain_output(
        stage=stages["afr1"], output=afr1_archive, reference=inputs["reference_afr1"]
    )
    final_retained = retained / "05_afr1.zip"
    retain_archive(afr1_archive, final_retained)
    outputs.append({"stage": "afr1", "archive": afr1_fact})

    return {
        "run": run_name,
        "status": "PASS",
        "elapsed_seconds": time.time() - started,
        "commands": commands,
        "stage_outputs": outputs,
        "final_archive": file_fact(final_retained),
    }


def retention_manifest(retained: Path) -> dict[str, object]:
    entries = [
        {
            "relative_path": str(path.relative_to(retained)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(retained.rglob("*"))
        if path.is_file()
    ]
    return {
        "schema": "ddm_ce1_retention_manifest.v1",
        "root": str(retained),
        "entry_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "entries": entries,
    }


def run_afr1_chain(
    *,
    store: Path,
    resolved: dict[str, Path],
    input_manifest: list[dict[str, object]],
    seed: int,
    repeats: int,
    resume: bool,
) -> dict[str, object]:
    if repeats != 2:
        raise SystemExit("AFR1 admission requires exactly two complete chain runs")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < 8 << 30:
        raise SystemExit(f"storage preflight refused: {free} B free, require at least 8 GiB")
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["TAC_CE1_SEED"] = str(seed)
    environment["TAC_JG2_RC64_SOURCE"] = str(resolved["rc64_source"])
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt: dict[str, object] = {
        "schema": "pact.pq2.afr1_chain.v1",
        "status": "RUNNING",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "seed": seed,
        "git_head_at_run": git_head,
        "runner": file_fact(Path(__file__).resolve()),
        "storage_preflight": {"free_bytes": free, "minimum_bytes": 8 << 30, "status": "PASS"},
        "chain": AFR1_CHAIN,
        "input_manifest": input_manifest,
        "runs": [],
    }
    destination = store / "RESULT_pq2_e2e.json"

    def persist() -> None:
        partial = destination.with_suffix(".partial")
        partial.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        os.replace(partial, destination)

    persist()
    try:
        for run_number in range(1, repeats + 1):
            result = execute_afr1_chain_run(
                run_number=run_number,
                store=store,
                inputs=resolved,
                environment=environment,
                resume=resume,
            )
            receipt["runs"].append(result)
            persist()
        first = Path(receipt["runs"][0]["final_archive"]["path"])
        second = Path(receipt["runs"][1]["final_archive"]["path"])
        deterministic = first.read_bytes() == second.read_bytes()
        receipt["determinism_repeat"] = {
            "byte_identical": deterministic,
            "first": file_fact(first),
            "second": file_fact(second),
            "first_differing_offset": None if deterministic else first_differing_offset(first, second),
        }
        if not deterministic:
            raise ChainStageError(
                "the two complete AFR1 chain runs differ",
                {
                    "status": "REFUSED",
                    "stage": "determinism_repeat",
                    **receipt["determinism_repeat"],
                },
            )
        manifest = retention_manifest(store / "retained")
        manifest_path = store / "RETENTION_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        receipt["retention_manifest"] = file_fact(manifest_path)
        receipt["status"] = "PASS"
        persist()
        return receipt
    except ChainStageError as error:
        receipt["status"] = "REFUSED"
        receipt["blocker"] = error.blocker
        receipt["error"] = str(error)
        persist()
        raise SystemExit(f"AFR1 CHAIN REFUSED: {error}") from error


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
        choices=("provenance", "encode", "build", "verify", "decode", "chain", "all"),
        help="'all' selects the registered chain when the expected archive has one",
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
        "--chain",
        choices=("afr1",),
        default=None,
        help="select a registered chain explicitly (also selects its inputs template)",
    )
    parser.add_argument(
        "--chain-repeats",
        type=int,
        default=2,
        help="complete chain runs; AFR1 admission requires exactly 2",
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

    if args.emit_inputs_template:
        recipe = load_recipe(args.recipe_json)
        spec = chain_input_spec() if args.chain == "afr1" else input_spec(recipe)
        return emit_inputs_template(spec)

    if args.store is None:
        parser.error("--store is required (or pass --emit-inputs-template)")

    expected = resolve_expected_archive(
        args.expected_archive_sha256, args.expected_archive_bytes, args.candidate_runtime
    )

    chain = CHAIN_RECIPES.get(expected.sha256)
    if args.chain == "afr1" and expected.sha256 != AFR1_CHAIN["archive_sha256"]:
        raise SystemExit(
            f"--chain afr1 requires expected archive {AFR1_CHAIN['archive_sha256']}, "
            f"not {expected.sha256}"
        )
    if chain is not None:
        if args.recipe_json is not None:
            raise SystemExit("a registered chain and --recipe-json are mutually exclusive")
        if args.stage not in ("all", "chain"):
            raise SystemExit("the AFR1 chain must run with --stage chain or --stage all")
        spec = chain_input_spec()
        resolved = resolve_inputs(spec, args.inputs_json)
        input_manifest = verify_inputs(spec, resolved)
        result = run_afr1_chain(
            store=args.store,
            resolved=resolved,
            input_manifest=input_manifest,
            seed=args.seed,
            repeats=args.chain_repeats,
            resume=args.resume,
        )
        print(
            f"[pq2] AFR1 CHAIN {result['status']} -> {args.store / 'RESULT_pq2_e2e.json'}",
            flush=True,
        )
        return 0

    recipe = load_recipe(args.recipe_json)
    spec = input_spec(recipe)

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
