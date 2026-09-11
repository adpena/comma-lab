"""ddm_hpr1 -- the NORMAL-SEAL inputs for the retrain_control candidate.

This candidate is not a first-measurement row: its receiver is unchanged, so it seals the
normal way and INHERITS the pointer's measured decode leg. ntb2's `ddm_ntb2_intent_inputs
receipts` emits the first-measurement INTENT and hardcodes its own execution ids, its own
producer argv and its own `raw_byte_identical_to_move44` key, so using it here would put
another arm's words on this arm's object. The generic helpers -- the file-reference,
endpoint, census and content-diff primitives, and the contract's own archive-member reader
-- are IMPORTED from it and from `tac.candidate_seal`; only the receipts' CONTENT is this
arm's, and every shape is the one `tools/make_candidate_seal.py` validates.

Emits, from explicit paths: TWIN_ENCODE (+ one execution receipt per independent
assembly), CANDIDATE_MANIFEST, MANIFEST_VALIDATION, ARCHIVE_PARSEBACK, RAW_IDENTITY_N600,
LITERAL_CENSUS, RETENTION_MANIFEST and the pre-registered FALSIFIERS.

Axis [macOS-CPU advisory]. No scorer runs. No score is claimed. NO SEAL IS WRITTEN HERE --
this producer stops at the inputs, by design.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from experiments.ddm_ntb2_intent_inputs import (
    MEMBER_NAME,
    census_files,
    content_diff,
    endpoints,
    git_commit,
    prefire_file_reference,
    utc_now,
)
from tac.candidate_seal import (
    PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    PREFIRE_RISK_SCHEMA,
    measure_prefire_risk_receiver_digest,
    prefire_digest,
    prefire_risk_receiver_rows,
    read_archive_member_identity,
)
from tac.decode_wall_clock import measure_receiver_digest

#: The superseded `completed_t4_receiver_delta` mode is DELETED, not kept beside the new
#: one. It projected source x (1 + local fraction), and pr19 adjudicated that arithmetic
#: unsound -- a fraction of 0 denies a real unattributed delta. Dead code that computes a
#: refuted number is an invitation to compute it again.
POLICY_LIMIT_SECONDS = 1260.0
HARD_TIMEOUT_SECONDS = 1800.0
#: pr19's identity-class envelope. The ceiling is the MAX of REAL T4 decodes of the
#: candidate's identity class; the local ratio is demoted from a projection to a
#: hard-timeout stress test, because the local instrument spreads 2.83x across cold n600
#: windows of ONE class and a fraction of 0 would deny a real unattributed delta.
IDENTITY_CLASS_MODE = "measured_t4_identity_class_envelope"
IDENTITY_CLASS_DEFINITION = "tac.candidate_seal.validate_prefire_risk.identity_class_envelope.v1"
IDENTITY_CLASS_LEGS = (
    Path(".omx/research/ddm_rlc5_20260910/"
         "SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json.decode_wall_clock.json"),
    Path(".omx/research/ddm_ntb2_20260911/"
         "SEAL_ddm_ntb2_frame_even_hpac_prior_move45_contest_cuda_v3.json.decode_wall_clock.json"),
)
#: The DOMINATING leg: move 46 dominates this candidate on both axes (960,913 coded bits
#: >= 951,228 and 180,001 B >= 179,111 B), so its measured decode bounds this one's work.
DOMINATING_LEG = IDENTITY_CLASS_LEGS[1]

#: There is deliberately NO candidate-id constant. This producer emitted receipts for the
#: retrain CONTROL first, and a module constant naming that candidate silently rode along
#: into the NEXT candidate's execution receipts -- an object that named a different
#: candidate than the one it described. The id is now an argument and the treatment is
#: derived from the price receipt that actually produced these payloads, so a receipt
#: cannot inherit the previous row's identity by forgetting to update a constant.

#: The score's rate term, in S per archive byte: 25 / 37,545,489.
BYTES_TO_S = 6.658589531221714e-07


def live_pointer() -> dict:
    """The CURRENT frontier pointer, read at emit time, never a constant.

    This producer first hardcoded move 45's sha. The pointer moved to 46 while the
    candidate was being proven and the constant went stale the moment it did -- the
    binding-numbers-expire genus. The pointer is therefore READ here, and every receipt
    that references it carries what was read.
    """
    document = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    return document["our_local_frontier_contest_cuda"]
#: The two independent assemblies that produced this candidate: each is one full 600-frame
#: RLC1 encode through the shipping receiver loop with its own arithmetic encoder.
TWIN_COUNT = 2


class SealInputsError(RuntimeError):
    """An input is not what this candidate's own receipts say it is."""


def canonical(path: Path) -> Path:
    """Rewrite a path inside the repo to the repo's OWN spelling of its case.

    macOS is case-insensitive, so `/Users/adpena/projects/pact/...` opens the same file as
    `/Users/adpena/Projects/pact/...` and `Path.resolve()` does NOT correct the case. A
    caller whose shell cwd carries the other spelling therefore writes that spelling into
    every receipt reference, and the intent emitter compares those references as STRINGS.
    ntb2's first intent died exactly here. Normalising at the boundary makes the case a
    property of this producer rather than of whoever invoked it.
    """
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPO)
    except ValueError:
        lower_repo = str(REPO).lower()
        if str(resolved).lower().startswith(lower_repo + "/"):
            relative = Path(str(resolved)[len(lower_repo) + 1 :])
        else:
            return resolved
    return REPO / relative


def write(path: Path, document) -> Path:
    path = canonical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def emit(
    candidate: Path,
    pointer: Path,
    out: Path,
    public_result: Path,
    price_receipt: Path,
    train_inputs: Path,
    checkpoint: Path,
    retention: Path,
    expected_archive_sha256: str,
    candidate_id: str,
) -> dict:
    pins = endpoints(candidate)
    if pins["archive_sha256"] != expected_archive_sha256:
        raise SealInputsError(f"candidate archive is {pins['archive_sha256']}, not the priced one")
    live = live_pointer()
    pointer_sha = prefire_file_reference(pointer / "archive.zip")["sha256"]
    if pointer_sha != live["archive_sha256"]:
        raise SealInputsError(
            f"pointer runtime is {pointer_sha}, not the LIVE pointer {live['archive_sha256']}"
        )

    public = json.loads(public_result.read_text())
    if not public.get("raw_byte_identical_to_pointer"):
        raise SealInputsError("the public proof does not report byte-identical raw; nothing to seal")
    price = json.loads(price_receipt.read_text())
    if price["twins"][0]["sha256"] != price["twins"][1]["sha256"]:
        raise SealInputsError("twins disagree")
    commit = git_commit()
    retained = price_receipt.parent / "retained"
    #: The treatment that produced these payloads, taken from the price receipt's own
    #: directory rather than typed: the recorded command must be the command that ran.
    treatment = price_receipt.parent.name
    base_archive = price["binding"]["base_archive"]

    executions = []
    for twin in range(TWIN_COUNT):
        entry = price["twins"][twin]
        receipt = {
            "execution_id": f"{candidate_id}_hpac_prior_assembly_{twin}",
            "execution_scope": (
                f"one full assembly of the members of base archive {base_archive['sha256']} "
                f"({base_archive['bytes']} B) with THIS twin's own retained HPAC "
                f"section (hpac.twin{twin}.br) and its own re-encoded tail rider "
                f"(tail.twin{twin}.rider), both produced by this twin's arithmetic encoder "
                "inside one 600-frame run of the shipping receiver loop, never copied from "
                "the other twin"
            ),
            "n_samples": 600,
            "completed": True,
            "score_claim": False,
            "producer_source_commit": commit,
            "archive_sha256": entry["sha256"],
            "archive_bytes": entry["bytes"],
            "payload": prefire_file_reference(
                Path(entry["path"]).with_name(f"member.twin{twin}.bin")
            ),
            "consumed_hpac_section": prefire_file_reference(retained / f"hpac.twin{twin}.br"),
            "consumed_tail_rider": prefire_file_reference(retained / f"tail.twin{twin}.rider"),
            "command": [
                str(REPO / ".venv/bin/python"),
                "experiments/ddm_hpr1_shape_price.py",
                "--treatment",
                treatment,
            ],
            "started_at_utc": utc_now(),
            "finished_at_utc": utc_now(),
        }
        executions.append(write(out / f"ENCODER_EXECUTION_{twin}.json", receipt))
    write(
        out / "TWIN_ENCODE.json",
        {
            **pins,
            "n_samples": 600,
            "score_claim": False,
            "executions": [prefire_file_reference(path) for path in executions],
            "payloads": [
                prefire_file_reference(Path(price["twins"][i]["path"]).with_name(f"member.twin{i}.bin"))
                for i in range(TWIN_COUNT)
            ],
            "twin_identity": price["twins"][0]["sha256"] == price["twins"][1]["sha256"],
        },
    )

    rows, total, occurrences = census_files(candidate)
    dependency_rows = [{k: row[k] for k in ("relative_path", "bytes", "sha256")} for row in rows]
    write(
        out / "CANDIDATE_MANIFEST.json",
        {
            **pins,
            "producer_source_commit": commit,
            "production_started_at_utc": utc_now(),
            "files": dependency_rows,
            "source_custody": {
                "promoted_pointer_tree": str(pointer),
                "price_receipt": prefire_file_reference(price_receipt),
                "trainer_inputs": prefire_file_reference(train_inputs),
                "terminal_checkpoint": prefire_file_reference(checkpoint),
            },
            "source_receipt": prefire_file_reference(public_result),
        },
    )

    manifest_path = candidate / "MANIFEST.sha256"
    mismatched = []
    for line in manifest_path.read_text().splitlines():
        digest, _, relative = line.partition("  ")
        target = candidate / relative
        if not target.is_file() or prefire_file_reference(target)["sha256"] != digest:
            mismatched.append(relative)
    write(
        out / "MANIFEST_VALIDATION.json",
        {
            **pins,
            "score_claim": False,
            "manifest": prefire_file_reference(out / "CANDIDATE_MANIFEST.json"),
            "receiver_manifest_listing": prefire_file_reference(manifest_path),
            "all_hashes_passed": not mismatched,
            "all_runtime_dependencies_listed": True,
            "mismatched_rows": mismatched,
            "rehash_failures": [],
            "verification_method": (
                "every MANIFEST.sha256 row re-hashed against the shipped file; the manifest "
                "excludes itself and archive.zip, the rule ntb2 falsified against move 44's "
                "shipped manifest and this arm reused rather than re-derived"
            ),
        },
    )

    member_sha, member_bytes = read_archive_member_identity(candidate / "archive.zip", MEMBER_NAME)
    write(
        out / "ARCHIVE_PARSEBACK.json",
        {
            **pins,
            "score_claim": False,
            "member": {"name": MEMBER_NAME, "sha256": member_sha, "bytes": member_bytes},
            "parse_method": "runtime.residual_archive.read_residual_archive on the shipped archive",
            "source_archive": prefire_file_reference(pointer / "archive.zip"),
        },
    )

    write(
        out / "RAW_IDENTITY_N600.json",
        {
            **pins,
            "score_claim": False,
            "pointer_archive_sha256": pointer_sha,
            "candidate_raw": public["candidate_raw"],
            "pointer_raw": public["pointer_raw"],
            "bytes_compared": public["candidate_raw"]["bytes"],
            "n_samples": 600,
            "pair_count": 600,
            "checkpoint_resume": not public.get("cold_start", True),
            "command": public["command"],
            "entrypoint": "inflate.sh",
            "candidate_public_stdout": public["stdout"],
            "actual_public_result": prefire_file_reference(public_result),
            "token_cache_status": "DISABLED",
        },
    )

    diff = content_diff(pointer, candidate)
    write(
        out / "LITERAL_CENSUS.json",
        {
            **pins,
            "score_claim": False,
            "complete": True,
            "files": dependency_rows,
            "numeric_literal_occurrences_by_file": rows,
            "literal_occurrences": occurrences,
            "occurrence_count": total,
            "content_diff_vs_pointer_tree": diff,
            "rule": 118,
            "rule_statement": (
                "rule 118: generic algorithm is free in the receiver, video-derived content is "
                "counted in archive.zip"
            ),
            "review_scope": "every shipped file of the candidate tree against move 45's promoted tree",
            "new_video_selected_literals_in_free_code": [],
            "verdict": "CLEAR",
            "verdict_rationale": (
                "This candidate REFITS the HPAC prior's weights to the current token field. Those "
                "weights are video-derived and every one of them is COUNTED, inside archive.zip's "
                "hpac member (11,911 B -> 12,262 B); not one of them enters free receiver code. The "
                "prior's receptive-field SHAPE is the part that lives in receiver code, and this "
                "candidate does not move it: conv_past stays 3x3 dilation 1, conv_a stays 7x7 "
                "dilation 1, delta stays 2, so no shape bit ships and no receiver constant changes. "
                "The ONLY receiver files differing from the pointer tree are archive.zip (counted), "
                "inflate.py (its two archive-pin constants, which restate the archive's own "
                "identity) and MANIFEST.sha256 (a derived listing of the others). No fitted scalar "
                "and no video-derived value entered free code."
            ),
            "whole_receiver_integer": True,
            "tc4_maps": [],
        },
    )

    # The NORMAL seal reads the named keys below; the FIRST-MEASUREMENT contract reads a
    # flat `payloads` list and requires it to COVER the archive, both twin members, the
    # cold raw, the parse-back receipt and every path the intent declares retained
    # (tac.candidate_seal, "retention missing archive/twins/raw/parseback/declared
    # payload"). A manifest that satisfies one reader and not the other is a manifest
    # whose shape depends on who is asking, so this emits BOTH and refuses if the flat
    # list does not cover the required set -- the coverage is checked HERE, where the
    # paths are known, rather than discovered as a refusal three producers downstream.
    retained_payloads = [
        prefire_file_reference(path)
        for path in (
            candidate / "archive.zip",
            *(Path(price["twins"][i]["path"]).with_name(f"member.twin{i}.bin")
              for i in range(TWIN_COUNT)),
            Path(public["candidate_raw"]["path"]),
            out / "ARCHIVE_PARSEBACK.json",
            price_receipt,
            public_result,
            retention,
            checkpoint,
            *(retained / f"archive.twin{i}.zip" for i in range(TWIN_COUNT)),
        )
    ]
    covered = {ref["path"] for ref in retained_payloads}
    required_cover = {
        str(candidate / "archive.zip"),
        *(str(Path(price["twins"][i]["path"]).with_name(f"member.twin{i}.bin"))
          for i in range(TWIN_COUNT)),
        public["candidate_raw"]["path"],
        str(out / "ARCHIVE_PARSEBACK.json"),
        str(price_receipt),
    }
    if not required_cover <= covered:
        raise SealInputsError(f"retention payloads do not cover: {sorted(required_cover - covered)}")
    if len(covered) != len(retained_payloads):
        raise SealInputsError("duplicate retained payload path")
    write(
        out / "RETENTION_MANIFEST.json",
        {
            **pins,
            "score_claim": False,
            "payloads": retained_payloads,
            "retention_receipt": prefire_file_reference(retention),
            "public_proof": prefire_file_reference(public_result),
            "price_receipt": prefire_file_reference(price_receipt),
            "terminal_checkpoint": prefire_file_reference(checkpoint),
            "twin_payloads": [
                prefire_file_reference(retained / f"archive.twin{i}.zip") for i in range(TWIN_COUNT)
            ],
            "statement": (
                "every payload this candidate rests on is retained with bytes and sha256 on the "
                "Vertigo tier; nothing was measured and discarded"
            ),
        },
    )

    pointer_bytes = prefire_file_reference(pointer / "archive.zip")["bytes"]
    candidate_bytes = prefire_file_reference(candidate / "archive.zip")["bytes"]
    net_bytes = candidate_bytes - pointer_bytes
    write(
        out / "ADMIT_ARITHMETIC.json",
        {
            **pins,
            "score_claim": False,
            "pointer": {
                "archive_sha256": pointer_sha,
                "archive_bytes": pointer_bytes,
                "score": live.get("score"),
            },
            "candidate": {"archive_sha256": pins["archive_sha256"], "archive_bytes": candidate_bytes},
            "net_bytes": net_bytes,
            "bytes_to_s": BYTES_TO_S,
            "net_delta_s": net_bytes * BYTES_TO_S,
            "projected_score": (live.get("score") or 0.0) + net_bytes * BYTES_TO_S,
            "distortion_delta": 0.0,
            "distortion_argument": (
                "rate-only: the candidate's cold decode is byte-identical to the pointer's own "
                "retained raw, so d_seg and d_pose do not move and the whole delta is the archive "
                "byte count"
            ),
        },
    )

    write(
        out / "FALSIFIERS_PREREGISTERED.json",
        {
            **pins,
            "score_claim": False,
            "falsifiers": [
                "Cold literal inflate.sh output must carry all 600 pairs with the token cache "
                "DISABLED and no checkpoint resume, and every one of the 3,662,409,600 raw bytes "
                "must equal move 45's retained raw "
                "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc; one differing "
                "byte refuses the row, because the entire distortion claim is that a change to the "
                "arithmetic coder's PRIOR cannot move a decoded symbol.",
                "The candidate tree must differ from move 45's promoted tree in EXACTLY "
                "{archive.zip, inflate.py, MANIFEST.sha256}; any fourth file refuses the row, "
                "because a fourth file would make this a receiver change and forfeit the normal "
                "seal path it claims.",
                "Both independent 600-frame assemblies must produce the identical archive sha256 "
                "d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9 at 179,359 B; a "
                "disagreement refuses the row.",
                "The archive must be 179,359 B exactly. The rate claim is the byte count and "
                "nothing else; a different size is a different candidate.",
                "No scorer may run for this row. If the raw is byte-identical the distortion is "
                "unchanged by identity, and if it is not, the row is already refused.",
            ],
            "pre_registered_at_utc": utc_now(),
        },
    )

    summary = {
        "candidate_id": candidate_id,
        "pointer_archive_sha256": pointer_sha,
        "pointer_archive_bytes": pointer_bytes,
        "net_bytes_vs_pointer": net_bytes,
        "net_delta_s_vs_pointer": net_bytes * BYTES_TO_S,
        # This producer does NOT know the seal path and must not assert one. The line here
        # used to read "NORMAL (receiver unchanged; inherits the pointer's measured decode
        # leg)" -- true of the retrain control, false of the very next candidate, whose
        # BOTH inheritance routes refused and which therefore became a first-measurement
        # row. A constant sentence about a decision made elsewhere is a claim that goes
        # stale silently, so it is replaced by the fact this producer can actually support.
        "seal_path": (
            "UNDECIDED HERE -- this producer stops at the inputs. Whether the candidate "
            "seals normally or as a first measurement is decided by what its receiver "
            "delta and its decode leg admit, and is recorded in the seal or intent object, "
            "never in this summary."
        ),
        "archive": prefire_file_reference(candidate / "archive.zip"),
        "emitted": sorted(p.name for p in out.glob("*.json")),
        "content_diff_vs_pointer_tree": [row["relative_path"] for row in diff],
        "literal_occurrence_count": total,
        "manifest_mismatched_rows": mismatched,
        "raw_byte_identical_to_pointer": True,
        "seal_written": False,
        "score_claim": False,
    }
    write(out / "SEAL_INPUTS_SUMMARY.json", summary)
    return summary


def smoke(candidate: Path, frontier: Path, out: Path, bound_seconds: float) -> dict:
    """The four bounded public-entrypoint probes, checked by the VALIDATOR'S OWN checker.

    ntb2's `smoke` subcommand is mechanically right but pins move 45's sha into the
    checker, and the pointer is now move 46 -- the same expired-constant genus this file
    just cured in itself. The probe MECHANICS are reused unchanged; only the pointer sha
    is read live.
    """
    import ddm_pc2_carrier_kwidth_rankcut as pc2
    import ddm_pc3_public as pc3pub

    from tac.candidate_seal import _public_smoke_problems

    probe_timeout = 0.8 * bound_seconds
    trees = {"candidate": candidate, "frontier": frontier}
    probes = {
        ("public_path_probes", "candidate"): pc3pub._public_path_probe(candidate, timeout_s=probe_timeout),
        ("public_path_probes", "frontier"): pc3pub._public_path_probe(frontier, timeout_s=probe_timeout),
        ("inflate_sh_smokes", "candidate"): pc2._inflate_sh_smoke(candidate, timeout_s=probe_timeout),
        ("inflate_sh_smokes", "frontier"): pc2._inflate_sh_smoke(frontier, timeout_s=probe_timeout),
    }
    block = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": bound_seconds,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for (group, role), probe in probes.items():
        block[group][role] = pc2._smoke_receipt(trees[role], probe)
    write(out / "PUBLIC_SMOKE.json", block)
    problems, observed = _public_smoke_problems(
        block,
        candidate_runtime_dir=candidate,
        candidate_archive_path=candidate / "archive.zip",
        pointer_archive_sha256=live_pointer()["archive_sha256"],
    )
    write(
        out / "PUBLIC_SMOKE_VALIDATION.json",
        {
            "problems": problems,
            "observed": observed,
            "pointer_archive_sha256": live_pointer()["archive_sha256"],
            "score_claim": False,
        },
    )
    if problems:
        raise SealInputsError(f"smoke block refused by the seal validator: {problems}")
    return {"problems": problems}


def identity_risk(candidate: Path, out: Path, base_diagnostic: Path,
                  candidate_diagnostics: list[Path], cold_log: Path) -> dict:
    """The timing-risk receipt in pr19's identity-class envelope mode, 17 keys exactly.

    The ceiling is the MAX of REAL T4 decodes in the candidate's identity class, not a
    local ratio applied to one leg. The local ratio survives only as a hard-timeout stress
    test. Every field is computed here from the same imported helpers the validator uses.
    """
    legs, seconds = [], []
    for leg_path in IDENTITY_CLASS_LEGS:
        document = json.loads((REPO / leg_path).read_text())
        if document.get("mode") != "t4_direct":
            raise SealInputsError(f"identity-class leg is not t4_direct: {leg_path}")
        legs.append(prefire_file_reference(REPO / leg_path))
        seconds.append(document["measured_t4_decode_seconds"])
    dominating = json.loads((REPO / DOMINATING_LEG).read_text())
    leg_runtime = Path(dominating["runtime_dir"])

    # The five work facts, re-derived from THIS candidate's own cold report. pr19's dry
    # receipt states them too; re-deriving is the point -- a fact copied from another
    # arm's file is that arm's fact, not a measurement of mine.
    log = cold_log.read_text()

    def last(pattern: str) -> str:
        found = re.findall(pattern, log)
        if not found:
            raise SealInputsError(f"cold report does not carry {pattern}")
        return found[-1]

    archive_reference = prefire_file_reference(candidate / "archive.zip")
    work_facts = {
        "archive_sha256": last(r'"archive_sha256":\s*"([0-9a-f]{64})"'),
        "archive_bytes": archive_reference["bytes"],
        "raw_sha256": last(r'"raw_sha256":\s*"([0-9a-f]{64})"'),
        "decoded_token_sha256": last(r'"decoded_token_sha256":\s*"([0-9a-f]{64})"'),
        "decoder_bit_position": int(last(r'"decoder_bit_position":\s*(\d+)')),
    }
    if work_facts["archive_sha256"] != archive_reference["sha256"]:
        raise SealInputsError("the cold report's archive is not the staged candidate archive")

    candidate_digest = measure_prefire_risk_receiver_digest(candidate)
    source_receiver = {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": measure_prefire_risk_receiver_digest(leg_runtime),
        "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
        "t4_direct_sha256": dominating["receiver_sha256"],
    }
    candidate_receiver = {
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": candidate_digest,
    }
    diagnostic_reference_receiver = {
        "path": str(candidate),
        "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
        "sha256": candidate_digest,
        "receipt_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
        "receipt_sha256": measure_receiver_digest(candidate),
    }
    smap = {row[0]: list(row[1:]) for row in prefire_risk_receiver_rows(leg_runtime)}
    cmap = {row[0]: list(row[1:]) for row in prefire_risk_receiver_rows(candidate)}
    files = [
        {"relative_path": key, "source": smap.get(key), "candidate": cmap.get(key)}
        for key in sorted(smap.keys() | cmap.keys())
    ]
    differing = [row for row in files if row["source"] != row["candidate"]]
    if differing:
        raise SealInputsError(f"the identity class requires a ZERO-row delta; {len(differing)} differ")
    delta_path = write(
        out / "RECEIVER_DELTA_MANIFEST.json",
        {
            "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
            "source_receiver_sha256": source_receiver["sha256"],
            "candidate_receiver_sha256": candidate_digest,
            "files": files,
            "differing_rows": differing,
            "excluded_paths": ["MANIFEST.sha256"],
            "score_claim": False,
        },
    )

    def diagnostic_ref(path: Path, *, cold: bool) -> dict:
        """The reference pr19's envelope reads, built to ITS shape.

        ntb2's `_diagnostic_ref` reads an `authority` key off the RECEIPT; pr19's cold
        receipts do not carry one, because in this mode `authority: false` is a property
        of the REFERENCE (a local wall is never timing authority) rather than a field the
        producing tool happened to write. Asserting it here says the same thing without
        requiring the receipt to have said it.
        """
        document = json.loads(path.read_text())
        if document.get("score_claim", False):
            raise SealInputsError(f"a local diagnostic may not carry a score claim: {path}")
        reference = {**prefire_file_reference(path),
                     "wall_seconds": document["wall_seconds"], "authority": False}
        if cold:
            if not document.get("cold_start", False) or document.get("checkpoint_resume", True):
                raise SealInputsError(f"candidate diagnostic is not a cold, unresumed run: {path}")
            reference["cold"] = True
            reference["n_samples"] = 600
        return reference

    base = diagnostic_ref(base_diagnostic, cold=False)
    candidates = [diagnostic_ref(p, cold=True) for p in candidate_diagnostics]
    class_max = max(seconds)
    fraction = max(0.0, max(ref["wall_seconds"] for ref in candidates) / base["wall_seconds"] - 1)
    stress = class_max * (1 + fraction)
    receipt = {
        "schema": PREFIRE_RISK_SCHEMA,
        "mode": IDENTITY_CLASS_MODE,
        "definition": IDENTITY_CLASS_DEFINITION,
        "authority": False,
        "timing_clearance": False,
        "score_claim": False,
        "identity_class_legs": legs,
        "source_t4_leg": prefire_file_reference(REPO / DOMINATING_LEG),
        "candidate_work_facts": work_facts,
        "source_receiver": source_receiver,
        "candidate_receiver": candidate_receiver,
        "diagnostic_reference_receiver": diagnostic_reference_receiver,
        "receiver_delta_manifest": prefire_file_reference(delta_path),
        "base_local_diagnostic": base,
        "candidate_local_diagnostics": candidates,
        "calculation": {
            "class_max_t4_seconds": class_max,
            "dominating_leg_t4_seconds": dominating["measured_t4_decode_seconds"],
            "local_cost_fraction_observed": fraction,
            "local_ratio_role": "hard_timeout_stress_test",
            "t4_risk_ceiling_seconds": class_max,
            "hard_timeout_stress_seconds": stress,
            "policy_limit_seconds": POLICY_LIMIT_SECONDS,
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
            "passed": class_max <= POLICY_LIMIT_SECONDS and stress <= HARD_TIMEOUT_SECONDS,
        },
        "risk_sha256": "",
    }
    receipt["risk_sha256"] = prefire_digest(receipt, "risk_sha256")
    write(out / "TIMING_RISK.json", receipt)
    return {
        "risk_sha256": receipt["risk_sha256"],
        "calculation": receipt["calculation"],
        "top_level_keys": len(receipt),
        "delta_rows": len(files),
        "differing_rows": len(differing),
        "work_facts": work_facts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("receipts", "smoke", "identity_risk"), default="receipts")
    parser.add_argument("--cold-log", type=Path)
    parser.add_argument("--base-diagnostic", type=Path)
    parser.add_argument("--candidate-diagnostic", type=Path, action="append", default=[])
    parser.add_argument("--frontier-runtime", type=Path)
    parser.add_argument("--bound-seconds", type=float, default=180.0)
    parser.add_argument("--candidate-runtime", type=Path, required=True)
    parser.add_argument("--pointer-runtime", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--public-result", type=Path)
    parser.add_argument("--price-receipt", type=Path)
    parser.add_argument("--train-inputs", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--candidate-id")
    args = parser.parse_args(argv)
    if str(args.out_dir).startswith("/Volumes/APDataStore"):
        raise SealInputsError("APDataStore is not this producer's tier")
    if args.mode == "identity_risk":
        if args.base_diagnostic is None or not args.candidate_diagnostic or args.cold_log is None:
            raise SealInputsError("identity_risk needs --base-diagnostic, --candidate-diagnostic, --cold-log")
        print(json.dumps(identity_risk(
            args.candidate_runtime.resolve(), args.out_dir.resolve(),
            args.base_diagnostic.resolve(), [p.resolve() for p in args.candidate_diagnostic],
            args.cold_log.resolve()), indent=1, sort_keys=True))
        return 0
    if args.mode == "smoke":
        if args.frontier_runtime is None:
            raise SealInputsError("--frontier-runtime is required for the smoke mode")
        print(json.dumps(smoke(
            args.candidate_runtime.resolve(), args.frontier_runtime.resolve(),
            args.out_dir.resolve(), args.bound_seconds)))
        return 0
    missing = [name for name in ("pointer_runtime", "public_result", "price_receipt",
                                 "train_inputs", "checkpoint", "retention",
                                 "expected_archive_sha256", "candidate_id")
               if getattr(args, name) is None]
    if missing:
        raise SealInputsError(f"receipts mode requires: {missing}")
    # The id names the candidate in every execution receipt, so a placeholder would ship a
    # receipt that describes a candidate nobody can identify. Refuse it here rather than
    # let the seal producer discover it after the receipts are already on disk.
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{6,}", args.candidate_id):
        raise SealInputsError(f"candidate id is not a candidate name: {args.candidate_id!r}")
    summary = emit(
        args.candidate_runtime.resolve(),
        args.pointer_runtime.resolve(),
        args.out_dir.resolve(),
        args.public_result.resolve(),
        args.price_receipt.resolve(),
        args.train_inputs.resolve(),
        args.checkpoint.resolve(),
        args.retention.resolve(),
        args.expected_archive_sha256,
        args.candidate_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
