#!/usr/bin/env python3
"""fire_modal_auth_eval.py — the ONE deterministic path from candidate runtime to a contest row.

Operator binding 2026-08-17: "Every mistake due to doing things by hand should be
fixed and deterministic and no longer ad hoc or manual." This tool replaces the
hand-assembled Modal auth-eval dispatch, which failed FOUR separate ways on the
rr4 fire of 2026-08-17, each failure a paid round-trip or a dead watcher:

  F1  AppleDouble litter (._*, .DS_Store) on an ExFAT runtime tree tripped the
      hidden-file validator remotely.                     -> stage SANITIZE
  F2  A payload-custody dir (retained/) inside the tree tripped the secret-name
      validator remotely.                                 -> stage VALIDATE (local, $0)
  F3  --expected-runtime-tree-sha256 was omitted by hand. -> pinned 'auto' always
  F4  A phantom-active Modal claim (call long completed) blocked the single-flight
      guard; the terminal-close needed args discovered by argparse error.
                                                          -> stage CLAIMS
  F5  The harvest poller was hand-armed with INVENTED flags and died instantly,
      leaving the paid call unwatched; the prior watcher was a background bash
      loop the harness SIGURG-kills at ~3 min.            -> stage ARM-POLLER
                                                            (detached, real argparse)

Sister incident (rr2, same day): a HAND-ASSEMBLED staged_* tree missing the
receiver's decoder guard was fired instead of the proved candidate_runtime —
fired bytes were never the proved bytes (S 27.83 vs projected 0.1585). This tool
therefore takes the runtime dir ITSELF, computes every sha from the actual bytes
(no hand-typed pins), and refuses on any mismatch with an optional external pin.

Stages (all local until DISPATCH):
  1 SANITIZE   strip macOS metadata litter (._* / .DS_Store) — filesystem
               artifacts, never receiver content. Anything else hidden refuses.
  2 VALIDATE   walk the tree with the SAME validate_runtime_upload_file the
               remote path uses, so every refusal surfaces locally in one pass.
               Secret-name hits print the exact relocation command and refuse
               (payload custody moves stay a deliberate act, never automatic).
  3 PIN        archive sha256/size computed from the file; optional
               --require-archive-sha asserts against a sealed fire-order pin.
  4 CLAIMS     reconcile; auto-terminal-close ONLY the exact provable-phantom
               condition (active Modal claim while the call-id ledger has zero
               live rows), recording the automation in the claim notes. The claim
               PRE-STAGED for this very dispatch (same lane_id + instance_job_id,
               written within SELF_CLAIM_MAX_AGE_HOURS) is EXEMPT: it matches the
               phantom trigger by construction, and closing it made the tool
               self-defeat its own --claim-policy require_active (rc2 r1).
  5 DISPATCH   the proven `modal run --detach ...::main` invocation with a fixed
               flag template (never hand-typed), tree sha pinned 'auto'.
  6 ARM-POLLER read call_id from the spawn record (refuse-loud if absent) and
               arm tools/modal_harvest_poller.py DETACHED via
               tools/launch_detached_process.py with a done-receipt.
  7 MANIFEST   write FIRE_MANIFEST.json (stages, shas, claim actions, poller).

Contest-axis note: F26_TOKEN_DECODER-style env overrides are deliberately NOT
exposed — --inflate-env demotes the row to a diagnostic axis; decoder guards
belong IN the fired tree (the rr4 lesson).

Axis selection (--axis cuda|cpu, added 2026-08-18 for task #1105): the contest
ranks a CPU axis the CUDA axis cannot stand in for, and pq1 sealed a contest-CPU
fire-order that this tool could not execute — its own `canonical_chain_gap` reads
"tools/fire_modal_auth_eval.py ... is CUDA-only. There is no CPU-axis equivalent.
[so] a CPU row must currently be hand-assembled, which is the exact hazard the
hand-assembled-dispatch law names as an error factory." The cure is a selector on
THIS tool, never a second dispatcher: --axis picks the worker entrypoint
(experiments/modal_auth_eval{,_cpu}.py::main), the [contest-CPU]/[contest-CUDA]
evidence tag, and the poller deadline from one table. Every stage above — sanitize,
local validate, computed sha pins, the archive/receiver seal, claim reconciliation,
the auto-armed detached poller, the manifest — applies identically on both axes.
Paired-axis semantics are unchanged: the worker still demands --pair-group-id or an
explicit --single-axis-waiver-reason, on CPU exactly as on CUDA.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.candidate_seal import (  # noqa: E402
    MISMATCH,
    PIN_ABSENT,
    check_pin_consistency,
    repin_receiver,
    validate_seal,
)
from tac.deploy.modal.auth_eval import validate_runtime_upload_file  # noqa: E402
from tac.deploy.modal.paired_dispatch import (  # noqa: E402
    PAIRED_AUTH_EVAL_CPU_WRAPPER,
    PAIRED_AUTH_EVAL_CUDA_WRAPPER,
    PAIRED_AUTH_EVAL_ENTRYPOINT,
)
from tac.modal_source_snapshot import (  # noqa: E402
    SNAPSHOT_ROOT_REL,
    assert_python_source_resolves,
    build_snapshot,
    dispatch_env,
    prune_snapshots,
    verify_dispatch_paths,
)

VENV_PY = str(REPO / ".venv" / "bin" / "python")
MODAL_BIN = str(REPO / ".venv" / "bin" / "modal")
LITTER_BASENAMES = {".DS_Store"}
LITTER_PREFIXES = ("._",)

# launch_detached_process.py refuses --done-receipt outside this shape. We compose the
# receipt from --instance-job-id, whose CONVENTION is "modal:<name>" — and ':' is not in
# the allowed set, so the compose silently produced an unarmable name and the watcher
# never started (fired 2026-08-18 on ddm_sa3; the call ran with no closer).
# Bound to the launcher's own _RECEIPT_NAME by test, not by hope.
_DONE_RECEIPT_ALLOWED = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

# How recently a claim must have been written to count as "pre-staged for THIS fire"
# and be exempted from the phantom auto-closer. Pre-staging is a step of the same
# operator action as the fire, so the true gap is minutes; 6h is loose enough to
# survive a slow compose/seal/validate pass and tight enough that a claim abandoned
# yesterday under the same lane+job is still treated as the phantom it is.
SELF_CLAIM_MAX_AGE_HOURS = 6.0


def _done_receipt_name(instance_job_id: str) -> str:
    """Sanitize an instance-job-id into a launcher-legal done-receipt name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_job_id).lstrip("_.-")
    if not cleaned:
        cleaned = "auth_eval"
    receipt = f"{cleaned}_harvest"[:128]
    if not _DONE_RECEIPT_ALLOWED.fullmatch(receipt):  # pragma: no cover - belt and braces
        raise ValueError(f"could not build a legal done-receipt from {instance_job_id!r}")
    return receipt

# The two contest score axes. CPU and CUDA are SEPARATE evidence spaces (neither is
# ever inferred from the other), so the axis picks the worker entrypoint, the evidence
# tag, and the watch deadline together — one table, never three hand-kept lists.
#
# The wrapper paths come from tac.deploy.modal.paired_dispatch so "which file is the CPU
# worker" has ONE definition. Its paired_auth_eval_axis_command() command SHAPE is
# deliberately NOT reused: it always emits --pair-group-id and has no
# single_axis_waiver_reason parameter, so it cannot express the waived single-axis fire
# that cured t1h r1/r2; it emits the runtime-tree pin only conditionally, weakening
# failure F3's always-'auto'; and it appends --gpu, which would move the proven rr4/fx1
# argv. Sharing the constants keeps one source of truth; sharing the shape would be a
# mechanism reduction.
AXES: dict[str, dict] = {
    "cuda": {
        "entrypoint": f"{PAIRED_AUTH_EVAL_CUDA_WRAPPER}::{PAIRED_AUTH_EVAL_ENTRYPOINT}",
        "evidence_axis_tag": "[contest-CUDA]",
        "score_axis": "contest_cuda",
        # T4 decode+score inside the CUDA worker's @app.function(timeout=4800).
        "poller_deadline_s": 2400.0,
    },
    "cpu": {
        "entrypoint": f"{PAIRED_AUTH_EVAL_CPU_WRAPPER}::{PAIRED_AUTH_EVAL_ENTRYPOINT}",
        "evidence_axis_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        # The CPU worker is @app.function(cpu=8.0, memory=16 GiB, timeout=9000) and a
        # 600-sample CPU scorer pass runs far longer than the T4 one. The watcher must
        # OUTLIVE the worker or a paid call goes unwatched (failure F5); 2400 s — the
        # CUDA default — would abandon every CPU row mid-flight.
        "poller_deadline_s": 9600.0,
    },
}


def axis_spec(axis: str) -> dict:
    """Resolve the axis contract. Fail-closed: an unknown axis is never a row."""

    key = str(axis or "").strip().lower()
    spec = AXES.get(key)
    if spec is None:
        raise ValueError(
            f"unknown auth-eval axis {axis!r}; evidence tagging is fail-closed — "
            f"choose one of {sorted(AXES)}"
        )
    return {"axis": key, **spec}


def write_fire_manifest(out_dir: Path, manifest: dict) -> Path:
    """Write FIRE_MANIFEST.json, REFUSING an axis-untagged row.

    A row whose axis is unrecorded cannot be read as [contest-CPU] or [contest-CUDA]
    evidence later, and the apples-to-apples discipline forbids inferring one axis
    from the other. So the manifest carries its axis tag or it is not written.
    """

    for key in ("axis", "evidence_axis_tag", "stage5_entrypoint"):
        if not str(manifest.get(key) or "").strip():
            raise ValueError(
                f"refusing to write an axis-untagged FIRE_MANIFEST: missing {key!r}. "
                "Every Modal auth-eval row carries its contest axis tag or it is not evidence."
            )
    known_tags = {spec["evidence_axis_tag"] for spec in AXES.values()}
    tag = manifest["evidence_axis_tag"]
    if tag not in known_tags:
        raise ValueError(
            f"refusing unknown evidence axis tag {tag!r}; expected one of {sorted(known_tags)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # A refusal receipt from an earlier attempt into this same dir would otherwise sit
    # beside a real manifest and make a fire that DID take read as refused. It is
    # ARCHIVED, never unlinked: the prior refusal is the record of why a re-fire exists.
    archive_prior_refusal(out_dir)
    path = out_dir / "FIRE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def archive_prior_refusal(out_dir: Path) -> str | None:
    """Move an existing FIRE_REFUSED.json aside so a re-fire needs no new directory.

    THE DEFECT (2026-09-04, ddm_fs2). The first fs2 fire was refused by Modal
    ("source modified during build process"). Re-firing into the same --output-dir
    would have left a stale refusal receipt beside a real manifest, and the rule
    "presence of FIRE_REFUSED.json means NO call exists" would then be a LIE about the
    second fire. MAIN's workaround was a whole new directory (`..._r2`), which splits
    the custody of one candidate across two paths for a bookkeeping reason.

    Archiving keeps both facts: the re-fire proceeds in place, and the refusal survives
    under ``refusals/<utc>.json`` where the next reader of this candidate will find it.
    Deleting it would be signal loss; leaving it in place would be a false refusal.
    """

    receipt = out_dir / "FIRE_REFUSED.json"
    if not receipt.is_file():
        return None
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = out_dir / "refusals"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{stamp}.json"
    receipt.replace(target)
    print(f"PRIOR REFUSAL ARCHIVED: {target}")
    return str(target)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_litter(runtime_dir: Path, *, apply: bool = True) -> list[str]:
    """macOS metadata litter. Returns relative paths removed (or, with
    ``apply=False``, the paths that WOULD be removed).

    A --dry-run must not mutate the tree it rehearses: these fires are aimed at
    SEALED runtime dirs whose hashes another agent already recorded, and deleting a
    file changes the runtime FILES digest the seal names. The dry-run therefore
    reports the litter and validate_tree skips it, modelling the real path exactly
    without touching a single byte.
    """

    removed: list[str] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name in LITTER_BASENAMES or any(name.startswith(p) for p in LITTER_PREFIXES):
            rel = str(path.relative_to(runtime_dir))
            if apply:
                path.unlink()
            removed.append(rel)
    return removed


def validate_tree(runtime_dir: Path, *, skip: frozenset[str] = frozenset()) -> list[str]:
    """Run the remote upload validators locally. Returns refusal messages.

    ``skip`` holds relative paths the sanitize stage has removed (or, on a dry run,
    would remove) so both modes validate the same effective tree.
    """

    refusals: list[str] = []
    for path in sorted(runtime_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(runtime_dir))
        if rel in skip:
            continue
        try:
            validate_runtime_upload_file(path, rel)
        except ValueError as exc:
            refusals.append(str(exc))
    return refusals


def refuse(out_dir: Path, rc: int, reason: str, manifest: dict) -> int:
    """Record a refusal ON DISK, then return its exit code.

    The t1h r1 fire returned rc=5 and the nonzero exit was swallowed by a ``| tail`` on
    the invocation, so a refused fire read as a successful one and the ladder cost two
    more attempts ("never pipe a fire command"). A tool cannot stop a caller from piping,
    but it can refuse to be silent: every refusal writes FIRE_REFUSED.json next to where
    the manifest would have gone, so the evidence survives whatever the shell does with
    the exit status. Presence of FIRE_REFUSED.json means NO call exists — never assume one.
    """

    manifest = {**manifest, "refused": True, "refusal_rc": rc, "refusal_reason": reason}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "FIRE_REFUSED.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"REFUSED (rc={rc}): {out_dir / 'FIRE_REFUSED.json'}")
    except (OSError, TypeError, ValueError) as exc:
        # Bookkeeping must NEVER mask the refusal it is recording. A manifest that will
        # not serialize is a bug to see later; the refusal is the fact to see now.
        print(f"REFUSED (rc={rc}); could not write refusal receipt: {exc}")
    return rc


#: The seal's contest axis -> this tool's --axis. ``advisory`` is deliberately absent: an
#: advisory seal is a real seal for a local row, but a paid Modal call always produces
#: contest-axis evidence, so firing one from an advisory seal would mislabel the row.
SEAL_AXIS_TO_FIRE_AXIS = {"contest_cuda": "cuda", "contest_cpu": "cpu"}

#: Flags whose value the seal already carries. Supplying one alongside --seal is the
#: hand-typed duplicate the error-factory law forbids: two sources of the same truth are
#: two chances to disagree, and the one that disagrees silently is the one that fires.
SEAL_OWNED_FLAGS = ("--runtime-dir", "--archive", "--require-archive-sha", "--axis")


def refuse_seal(seal_path: Path, out_dir: Path, rc: int, reason: str, manifest: dict, detail: dict) -> int:
    """Refuse a sealed fire LOUDLY, on stderr and on disk, twice.

    The t1h r1 lesson is that a refusal which exists only as an exit code is a refusal a
    pipe can erase. So a seal refusal writes a receipt NEXT TO THE SEAL (where the next
    reader of that candidate will look) as well as the usual FIRE_REFUSED.json in the output
    dir, and it writes the human line to stderr, which survives ``| tail`` on stdout.
    """

    receipt = {
        "schema": "candidate_seal_fire_refusal.v1",
        "seal_path": str(seal_path),
        "refusal_rc": rc,
        "refusal_reason": reason,
        "seal_validation": detail,
    }
    try:
        seal_receipt = seal_path.with_name(seal_path.name + ".REFUSED.json")
        seal_receipt.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"SEAL REFUSAL RECEIPT: {seal_receipt}", file=sys.stderr)
    except OSError as exc:  # pragma: no cover - custody volume unwritable
        print(f"SEAL REFUSED; could not write receipt beside the seal: {exc}", file=sys.stderr)
    print(f"SEAL REFUSED (rc={rc}): {reason}", file=sys.stderr)
    return refuse(out_dir, rc, reason, {**manifest, "seal_refusal": receipt})


def build_dispatch_argv(
    *,
    spec: dict,
    archive: Path,
    runtime_dir: Path,
    archive_sha: str,
    out_dir: Path,
    lane_id: str,
    instance_job_id: str,
    claim_agent: str,
    single_axis_waiver_reason: str = "",
    pair_group_id: str = "",
    claim_policy: str = "",
) -> list[str]:
    """Compose the fixed dispatch template. Identical on both axes but the entrypoint.

    Every flag emitted here is declared by BOTH local_entrypoint signatures — the CPU
    worker simply omits the CUDA-only knobs (gpu / scorer-device / inflate-device /
    inflate-env), none of which this template emits. That parity is asserted against the
    real signatures in tools/tests/test_fire_modal_auth_eval_axis.py, so never-invent-flags
    is executed rather than eyeballed (pq1 verified it by hand; hand-verification is the
    step this determinizes).

    --expected-runtime-tree-sha256 is pinned 'auto' on BOTH axes (failure F3). That is not
    a convenience: the projected and remote tree hashes are environment-coupled and
    structurally disagree (the r9m deadlock), so both workers accept only ''/'auto'/the
    runtime FILES digest and REFUSE any other value. Custody is carried by the transport
    zip sha256 plus that FILES digest.
    """

    cmd = [
        MODAL_BIN, "run", "--detach", spec["entrypoint"],
        "--archive", str(archive),
        "--submission-dir", str(runtime_dir),
        "--inflate-sh", "inflate.sh",
        "--expected-archive-sha256", archive_sha,
        "--expected-runtime-tree-sha256", "auto",
        "--output-dir", str(out_dir),
        "--detach", "--provider-detach-ack",
        "--lane-id", lane_id,
        "--instance-job-id", instance_job_id,
        "--claim-agent", claim_agent,
    ]
    if single_axis_waiver_reason:
        cmd += ["--single-axis-waiver-reason", single_axis_waiver_reason]
    if pair_group_id:
        cmd += ["--pair-group-id", pair_group_id]
    if claim_policy:
        cmd += ["--claim-policy", claim_policy]
    return cmd


def reconcile_claims(
    auto_close_agent: str,
    dry_run: bool,
    *,
    self_lane_id: str = "",
    self_instance_job_id: str = "",
    self_claim_max_age_hours: float = SELF_CLAIM_MAX_AGE_HOURS,
    claims_path: Path | str | None = None,
    modal_ledger: Path | str | None = None,
) -> dict:
    """Reconcile; terminal-close the provable-phantom condition ONLY, never our own claim.

    The auto-closer exists for failure F4: an active Modal claim left behind by a call
    that long since finished blocks the single-flight guard. Its trigger is "active claim
    + ZERO live ledger call_ids".

    A claim PRE-STAGED for the dispatch this very invocation is about to make satisfies
    that trigger EXACTLY — it is active, and its call does not exist yet, so the ledger
    has no live row for it. The closer therefore terminal-closed the claim the fire was
    about to consume, and the worker's `--claim-policy require_active` then refused with
    "newest matching claim is terminal: ... status=stale_superseded_reconciled_no_live_call".
    Measured on the rc2 T4 fire, 2026-08-20:
    /Volumes/APDataStore/pact/ddm_rc2/t4_row_r1/FIRE_REFUSED.json records
    stage4_claims.closed = [lane_ddm_rc2_composed_cuda_20260820 / modal:ddm_rc2_composed_cuda_r1]
    and refusal_rc 5 on that same lane+job. The tool self-defeated its own claim policy;
    the standing workaround was `--claim-policy open`, i.e. disarming the guard.

    The cure is an identity + freshness exemption. A claim is OURS when its lane_id AND
    instance_job_id both equal the ones this invocation will dispatch with, and it was
    written within `self_claim_max_age_hours`. Identity alone is not enough: the same
    lane+job reused after an abandoned fire hours earlier IS a real phantom and must still
    close, which is why the reconciler now reports each claim's timestamp.

    Reads structured JSON rather than scraping the human table — the previous text scrape
    could not see a timestamp at all, so the exemption was not expressible.
    """

    cmd = [VENV_PY, str(REPO / "tools" / "claim_lane_dispatch.py"), "reconcile", "--format", "json"]
    if claims_path is not None:
        cmd += ["--claims-path", str(claims_path)]
    if modal_ledger is not None:
        cmd += ["--modal-ledger", str(modal_ledger)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)  # subprocess-no-check-OK: success judged by JSON-parse of stdout below; unparseable reconcile closes NOTHING (fail-safe)
    action: dict = {"closed": [], "exempt": []}
    try:
        report = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        # Fail SAFE, never silent: an unparseable reconcile closes NOTHING (closing on a
        # guess is how the rc2 refusal happened) and the raw output is preserved.
        action["reconcile_unparseable"] = True
        action["reconcile_output_tail"] = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
        return action

    live = list(report.get("live_modal_call_ids") or [])
    claims = list(report.get("active_modal_claims") or [])
    action["live_modal_call_ids"] = live
    action["reconcile_output_tail"] = [str(p) for p in (report.get("problems") or [])][-4:]
    # Provable phantom requires ZERO live ledger call_ids alongside >=1 active claim.
    if live:
        return action

    for row in claims:
        lane = str(row.get("lane_id") or "")
        job = str(row.get("job") or "")
        if not lane or not job:
            continue
        age = row.get("age_hours")
        is_self = bool(self_lane_id) and bool(self_instance_job_id) and (
            lane == self_lane_id and job == self_instance_job_id
        )
        fresh = isinstance(age, (int, float)) and float(age) <= float(self_claim_max_age_hours)
        if is_self and fresh:
            action["exempt"].append(
                {
                    "lane": lane,
                    "job": job,
                    "age_hours": age,
                    "reason": "pre-staged for THIS dispatch (lane+job identity, within "
                    f"{self_claim_max_age_hours}h) — closing it would self-defeat "
                    "--claim-policy require_active",
                }
            )
            continue
        if dry_run:
            action["closed"].append({"lane": lane, "job": job, "dry_run": True})
            continue
        close = subprocess.run(
            [
                VENV_PY,
                str(REPO / "tools" / "claim_lane_dispatch.py"),
                "claim",
                "--lane-id", lane,
                "--platform", "modal",
                "--instance-job-id", job,
                "--agent", auto_close_agent,
                "--force",
                "--status", "stale_superseded_reconciled_no_live_call",
                "--notes",
                "Auto terminal-close by tools/fire_modal_auth_eval.py: reconcile "
                "reported an active Modal claim with ZERO live ledger call_ids "
                "(the provable-phantom condition). Deterministic per operator "
                "2026-08-17 no-manual binding.",
            ]
            + (["--claims-path", str(claims_path)] if claims_path is not None else []),
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        action["closed"].append({"lane": lane, "job": job, "rc": close.returncode})
    return action


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seal",
        default=None,
        help="a candidate seal (tools/make_candidate_seal.py). Validated FIRST against disk; "
        "the runtime dir, archive, archive-sha pin and axis are DERIVED from it, never "
        "re-typed. Mutually exclusive with the flags it owns.",
    )
    ap.add_argument("--runtime-dir", default=None, help="required unless --seal supplies it")
    ap.add_argument("--archive", default=None, help="default: <runtime-dir>/archive.zip")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lane-id", required=True)
    ap.add_argument("--instance-job-id", required=True)
    ap.add_argument("--claim-agent", default="MAIN")
    ap.add_argument(
        "--axis",
        choices=sorted(AXES),
        # None, not "cuda": the resolved default is still cuda (backward compatible), but a
        # sentinel is the only way to tell "the caller asked for cuda" from "the caller said
        # nothing", and --seal must refuse the former while accepting the latter.
        default=None,
        help="contest score axis (default: cuda): cuda -> T4 CUDA worker, cpu -> the Linux x86_64 "
        "contest-CPU worker (experiments/modal_auth_eval_cpu.py). Picks the entrypoint, "
        "the evidence tag, and the poller deadline together.",
    )
    ap.add_argument("--single-axis-waiver-reason", default="")
    ap.add_argument("--pair-group-id", default="")
    ap.add_argument(
        "--claim-policy",
        default="",
        choices=("", "open", "require_active"),
        help="forwarded to the worker entrypoint when set; require_active refuses to "
        "dispatch unless a live claim already covers this lane",
    )
    ap.add_argument("--require-archive-sha", default="", help="sealed fire-order pin; mismatch refuses")
    ap.add_argument(
        "--repin-receiver",
        action="store_true",
        help="re-pin the staged receiver's ARCHIVE_SHA256/ARCHIVE_BYTES from the staged archive "
        "(the compose-time staging step; without it a mismatched tree refuses)",
    )
    ap.add_argument(
        "--poller-deadline-s",
        type=float,
        default=None,
        help="default: axis-derived (cuda 2400 s, cpu 9600 s — the CPU worker's own "
        "timeout is 9000 s and the watcher must outlive it)",
    )
    ap.add_argument(
        "--no-source-snapshot",
        action="store_true",
        help="dispatch straight from the live working tree (the pre-2026-09-04 behaviour). "
        "The default snapshots every mounted source tree first so a concurrent edit "
        "cannot abort or contaminate the image build; turn it off only to diagnose the "
        "snapshot itself, never to save 9 seconds.",
    )
    ap.add_argument(
        "--snapshot-retain-days",
        type=float,
        default=3.0,
        help="prune source snapshots older than this at fire time (default 3). Clones cost "
        "~0 bytes when made and diverge as the tree changes, so they are cheap but not free.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # The worker's pairing validation treats these as mutually exclusive; passing both
    # refuses REMOTELY after the image build (the ck1 cpu_row_r1 rc=5, 2026-08-19).
    # A worker-side rule the firer can check must be checked here, before any subprocess.
    if args.single_axis_waiver_reason and args.pair_group_id:
        ap.error(
            "--pair-group-id and --single-axis-waiver-reason are mutually exclusive "
            "(the worker refuses both-set after the image build): pass --pair-group-id "
            "when both legs fire as a registered pair, or the waiver alone when the "
            "other leg was bought in a separate invocation."
        )
    # NEITHER-set is equally a guaranteed worker refusal ("paired-by-default"), but that
    # one arrives AFTER the image build with its message buried under the mount tree —
    # the br1 triple-refusal (2026-08-19, task #1152) spent two blind rc=5 rounds on it.
    # A worker-side rule the firer can check must be checked here, before any subprocess.
    if not args.single_axis_waiver_reason and not args.pair_group_id:
        ap.error(
            "the worker is paired-by-default: pass --pair-group-id (shared by the "
            "CPU/CUDA sibling jobs) for a paired fire, or --single-axis-waiver-reason "
            "with a substantive rationale for a single-axis row."
        )

    # The dispatch subprocess runs with cwd=REPO, so a relative --output-dir must
    # resolve against REPO here too or the spawn-record check reads the wrong dir.
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    # A re-fire into a directory that already refused must NOT need a fresh directory
    # (the fs2 `_r2` split). Archive the prior receipt first so this fire's own refusal —
    # or its manifest — is the only top-level verdict in the dir, and the old one survives.
    out_dir.mkdir(parents=True, exist_ok=True)
    archived_refusal = archive_prior_refusal(out_dir)

    # ---- STAGE 0 SEAL --------------------------------------------------------------
    # Runs before every other stage and before any subprocess: a refused seal must cost
    # nothing, least of all a paid call. Everything it derives is measured from disk by
    # tac.candidate_seal.validate_seal, so a pin that drifted since the seal was written
    # (the ps1u r1 class) refuses here instead of failing remotely on the meter.
    seal_manifest: dict = {}
    require_archive_sha = args.require_archive_sha
    resolved_axis = args.axis
    if args.seal:
        seal_path = Path(args.seal).resolve()
        supplied = [flag for flag in SEAL_OWNED_FLAGS if getattr(args, flag[2:].replace("-", "_"))]
        if supplied:
            reason = (
                f"--seal owns {', '.join(SEAL_OWNED_FLAGS)}; {', '.join(supplied)} was also supplied by hand. "
                "Two sources for one truth is the hand-assembly hazard — pass the seal alone."
            )
            print(f"FATAL: {reason}", file=sys.stderr)
            return refuse_seal(seal_path, out_dir, 7, reason, {"seal_path": str(seal_path)}, {})
        if args.repin_receiver:
            reason = (
                "--repin-receiver rewrites the staged receiver, which would invalidate the very seal "
                "being consumed. Re-pin at compose time, then seal the result."
            )
            print(f"FATAL: {reason}", file=sys.stderr)
            return refuse_seal(seal_path, out_dir, 7, reason, {"seal_path": str(seal_path)}, {})

        verdict = validate_seal(seal_path)
        seal_manifest = {"seal_path": str(seal_path), "seal_validation": verdict.to_dict()}
        print(f"SEAL: {verdict.summary()}")
        if not verdict.ok:
            return refuse_seal(
                seal_path, out_dir, 7, f"seal invalid: {verdict.verdict}", seal_manifest, verdict.to_dict()
            )

        document = json.loads(seal_path.read_text())
        seal_axis = str(document.get("axis") or "")
        if seal_axis not in SEAL_AXIS_TO_FIRE_AXIS:
            reason = (
                f"seal declares axis {seal_axis!r}; a paid Modal auth-eval row is contest evidence and "
                f"can only be fired from {sorted(SEAL_AXIS_TO_FIRE_AXIS)}"
            )
            print(f"FATAL: {reason}", file=sys.stderr)
            return refuse_seal(seal_path, out_dir, 7, reason, seal_manifest, verdict.to_dict())
        resolved_axis = SEAL_AXIS_TO_FIRE_AXIS[seal_axis]
        args.runtime_dir = document["runtime"]["path"]
        args.archive = document["archive"]["path"]
        require_archive_sha = document["archive"]["sha256"]
        seal_manifest["seal_candidate_id"] = document.get("candidate_id")
        seal_manifest["seal_sha256"] = document.get("seal_sha256")
        seal_manifest["seal_admit_bar"] = document.get("admit_bar")

    if not args.runtime_dir:
        print("FATAL: --runtime-dir is required unless --seal supplies it", file=sys.stderr)
        return refuse(out_dir, 2, "--runtime-dir is required unless --seal supplies it", {})

    spec = axis_spec(resolved_axis or "cuda")
    poller_deadline_s = (
        float(args.poller_deadline_s)
        if args.poller_deadline_s is not None
        else float(spec["poller_deadline_s"])
    )

    runtime_dir = Path(args.runtime_dir).resolve()
    archive = Path(args.archive) if args.archive else runtime_dir / "archive.zip"
    manifest: dict = {
        **seal_manifest,
        "schema": "fire_modal_auth_eval.v2",
        "archived_prior_refusal": archived_refusal,
        # Self-describing: a dry-run manifest carries no call_id, and a reader must never
        # have to infer from an absence whether a fire actually took.
        "dry_run": bool(args.dry_run),
        "runtime_dir": str(runtime_dir),
        "axis": spec["axis"],
        "evidence_axis_tag": spec["evidence_axis_tag"],
        "score_axis": spec["score_axis"],
    }

    if not runtime_dir.is_dir():
        print(f"FATAL: runtime dir missing: {runtime_dir}")
        return refuse(out_dir, 2, f"runtime dir missing: {runtime_dir}", manifest)
    if not (runtime_dir / "inflate.sh").is_file():
        print(f"FATAL: {runtime_dir}/inflate.sh missing — not a receiver runtime tree")
        return refuse(out_dir, 2, "inflate.sh missing — not a receiver runtime tree", manifest)
    if not archive.is_file():
        print(f"FATAL: archive missing: {archive}")
        return refuse(out_dir, 2, f"archive missing: {archive}", manifest)

    litter = sanitize_litter(runtime_dir, apply=not args.dry_run)
    manifest["stage1_sanitized"] = litter
    manifest["stage1_applied"] = not args.dry_run
    if litter:
        verb = "would remove" if args.dry_run else "removed"
        print(f"SANITIZE: {verb} {len(litter)} metadata-litter file(s)")

    refusals = validate_tree(runtime_dir, skip=frozenset(litter))
    manifest["stage2_refusals"] = refusals
    if refusals:
        print("VALIDATE: the remote upload path WOULD refuse this tree. Fix locally first:")
        for r in refusals:
            print(f"  - {r}")
        print(
            "Payload-custody dirs (e.g. retained/) must be RELOCATED, never deleted "
            "(ALWAYS KEEP THE PAYLOAD): mv <dir> <runtime-dir>_retained_custody/"
        )
        return refuse(out_dir, 3, "local validators would refuse this tree remotely", manifest)

    archive_sha = _sha256(archive)
    archive_bytes = archive.stat().st_size
    manifest["stage3_archive"] = {"path": str(archive), "sha256": archive_sha, "bytes": archive_bytes}
    print(f"PIN: archive {archive_bytes} B sha {archive_sha[:16]}…")
    if require_archive_sha and require_archive_sha.lower() != archive_sha:
        print(
            "FATAL: archive sha does not match the sealed fire-order pin "
            f"(pinned {require_archive_sha[:16]}…, actual {archive_sha[:16]}…)"
        )
        return refuse(
            out_dir,
            4,
            f"archive sha {archive_sha} does not match sealed pin {require_archive_sha}",
            manifest,
        )

    # Stage 3b SEAL — archive and runtime are ONE sealed object (the r3 law). The staged
    # receiver's own pin must name the staged archive's exact bytes. Nothing here is compared
    # against a remembered candidate: both sides are measured from this tree, at this moment,
    # so it holds for rr4, fx1, sa1 and every successor without editing a literal.
    seal = check_pin_consistency(runtime_dir, archive_path=archive)
    if seal.verdict == MISMATCH and args.repin_receiver:
        repin = repin_receiver(runtime_dir, archive_path=archive)
        manifest["stage3b_repin"] = repin.to_dict()
        print(
            f"SEAL: re-pinned {seal.receiver_path.name} "
            f"{str(repin.old_sha256)[:16]}…/{repin.old_bytes:,} B -> "
            f"{repin.new_sha256[:16]}…/{repin.new_bytes:,} B"
        )
        seal = check_pin_consistency(runtime_dir, archive_path=archive)
        # On ExFAT/FAT volumes macOS re-creates AppleDouble ._ litter the instant the
        # repin writes inflate.py; the upload path refuses hidden files, so sanitize
        # must re-run after the LAST mutation of the tree (the sz1 r1 rc=5 refusal).
        post_repin_litter = sanitize_litter(runtime_dir, apply=not args.dry_run)
        manifest["stage3b_post_repin_sanitized"] = post_repin_litter
        if post_repin_litter:
            verb = "would remove" if args.dry_run else "removed"
            print(f"SANITIZE(post-repin): {verb} {len(post_repin_litter)} metadata-litter file(s)")
    manifest["stage3b_seal"] = seal.to_dict()
    print(f"SEAL: {seal.summary()}")
    if seal.verdict == MISMATCH:
        print(
            "FATAL: the staged receiver pins a DIFFERENT candidate than the staged archive. "
            "This tree would be refused remotely at decode time, after the meter started. "
            "Re-run with --repin-receiver to re-pin at compose time, or stage the archive "
            "this receiver names."
        )
        return refuse(
            out_dir, 6, "staged receiver pins a DIFFERENT candidate than the staged archive", manifest
        )
    if seal.verdict == PIN_ABSENT:
        print(
            "WARNING: this receiver carries no archive pin, so the seal is weaker than the "
            "rr4 lineage's and the pin check is vacuous for this tree. Proceeding, loudly."
        )

    manifest["stage4_claims"] = reconcile_claims(
        args.claim_agent,
        args.dry_run,
        self_lane_id=args.lane_id,
        self_instance_job_id=args.instance_job_id,
    )

    cmd = build_dispatch_argv(
        spec=spec,
        archive=archive,
        runtime_dir=runtime_dir,
        archive_sha=archive_sha,
        out_dir=out_dir,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
        claim_agent=args.claim_agent,
        single_axis_waiver_reason=args.single_axis_waiver_reason,
        pair_group_id=args.pair_group_id,
        claim_policy=args.claim_policy,
    )
    manifest["stage5_entrypoint"] = spec["entrypoint"]
    manifest["stage5_dispatch_argv"] = cmd
    manifest["stage6_poller_deadline_s"] = poller_deadline_s

    # ---- STAGE 4b SNAPSHOT ---------------------------------------------------------
    # The mounts read the LIVE working tree. A concurrent edit during the image build
    # aborts the dispatch ("source modified during build process", ddm_fs2 2026-09-04,
    # MAIN's own `ruff format`) or — worse — bakes half-old bytes into a paid image.
    # Cloning every mounted tree first makes the fire read an immutable instant.
    # Verified fail-closed: an incomplete snapshot REFUSES rather than firing a short
    # mount, because a short mount spends the meter and returns a crash.
    # THE CWD STAYS THE REPO ROOT. The worker's local half resolves its dispatch claim
    # from Path.cwd() and tac.deploy.claims then spawns a RELATIVE `.venv/bin/python
    # tools/claim_lane_dispatch.py`. MEASURED 2026-09-04 (ps2 t4_custody, rc=5): firing
    # from the snapshot dir raised FileNotFoundError on that interpreter — and had it
    # resolved, the claim row would have landed in the snapshot's own .omx/state and been
    # lost, with the single-flight guard then reading an empty ledger. The snapshot is
    # therefore injected through PACT_MODAL_SOURCE_ROOT, which changes only what the image
    # MOUNTS. The mounts in the refused fire's log prove the snapshot half already worked.
    dispatch_cwd = REPO
    dispatch_env_vars: dict | None = None
    if args.no_source_snapshot:
        manifest["stage4b_source_snapshot"] = {
            "enabled": False,
            "consequence": "the image build reads the LIVE working tree; a concurrent edit can abort it",
        }
        print("SNAPSHOT: DISABLED — the image build will read the live working tree")
    else:
        entry_rel = spec["entrypoint"].split("::", 1)[0]
        pruned = prune_snapshots(
            REPO / SNAPSHOT_ROOT_REL, retain_days=float(args.snapshot_retain_days)
        )
        snap = build_snapshot(
            source_root=REPO,
            entrypoint=REPO / entry_rel,
            label=args.instance_job_id,
        )
        probe_failures = (
            assert_python_source_resolves(
                snap.root,
                snap.mounts.python_source_modules,
                python_executable=VENV_PY,
                entrypoint=snap.entrypoint,
            )
            if snap.complete
            else ["snapshot incomplete; import probe not attempted"]
        )
        snap_record = snap.to_dict()
        snap_record["pruned_older_snapshots"] = pruned
        snap_record["python_source_probe_failures"] = probe_failures
        manifest["stage4b_source_snapshot"] = snap_record
        print(
            f"SNAPSHOT: {snap.file_count:,} files / {snap.total_bytes:,} B in "
            f"{snap.elapsed_s:.1f}s (clonefile={snap.clonefile_used}) "
            f"digest {snap.files_digest[:16]}…"
        )
        if not snap.complete or probe_failures:
            for reason in [*snap.verify_failures, *snap.missing_in_source, *probe_failures]:
                print(f"  - {reason}")
            return refuse(
                out_dir,
                8,
                "source snapshot is not provably complete; refusing to fire a short mount",
                manifest,
            )
        dispatch_env_vars = dispatch_env(snap.root, entrypoint=snap.entrypoint)

    # Before the meter: every relative path the dispatch and its local half will spawn
    # must resolve from the cwd we are about to use. The ps2 rc=5 refusal is what this
    # costs when it is discovered remotely instead of here.
    path_problems = verify_dispatch_paths(dispatch_cwd, cmd)
    manifest["stage4c_dispatch_path_check"] = {"cwd": str(dispatch_cwd), "problems": path_problems}
    if path_problems:
        for problem in path_problems:
            print(f"  - {problem}")
        return refuse(
            out_dir, 9, "dispatch paths do not resolve from the dispatch cwd", manifest
        )

    if args.dry_run:
        print(f"AXIS: {spec['axis']} {spec['evidence_axis_tag']} -> {spec['entrypoint']}")
        print(f"DRY-RUN: would dispatch from cwd={dispatch_cwd}:\n  " + " ".join(cmd))
        print(write_fire_manifest(out_dir, manifest))
        return 0

    disp = subprocess.run(cmd, cwd=dispatch_cwd, env=dispatch_env_vars, capture_output=True, text=True)  # subprocess-no-check-OK: success judged by the spawn-record effect check below (refuse rc=5 when absent)
    sys.stdout.write(disp.stdout[-2000:])
    sys.stderr.write(disp.stderr[-2000:])
    # Persist the FULL dispatch output: the 2000-char echo above is a courtesy tail, and
    # modal's mount tree fills it — the br1 incident (#1152) hid a one-line entrypoint
    # refusal behind it for two rounds. The files, not the echo, are the diagnostic record.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dispatch_stdout.log").write_text(disp.stdout)
    (out_dir / "dispatch_stderr.log").write_text(disp.stderr)
    refusal_lines = [
        ln
        for stream in (disp.stdout, disp.stderr)
        for ln in stream.splitlines()
        if ln.startswith(("FATAL", "REFUSED", "Traceback", "Error:"))
        or "PreflightError" in ln
        or "refuse" in ln.lower()
    ][:20]
    spawn_path = out_dir / "modal_auth_eval_spawn.json"
    if not spawn_path.is_file():
        manifest["entrypoint_refusal_lines"] = refusal_lines
        for ln in refusal_lines:
            print(f"ENTRYPOINT SAID: {ln}")
        print("FATAL: dispatch produced no spawn record — the fire DID NOT take. Do not assume a call exists.")
        print(f"Full dispatch output: {out_dir / 'dispatch_stdout.log'} / dispatch_stderr.log")
        return refuse(out_dir, 5, "dispatch produced no spawn record — the fire DID NOT take", manifest)
    call_id = json.loads(spawn_path.read_text()).get("call_id", "")
    if not call_id:
        print("FATAL: spawn record carries no call_id — refuse-loud, investigate before refiring.")
        return refuse(out_dir, 5, "spawn record carries no call_id", manifest)
    manifest["stage5_call_id"] = call_id
    print(f"DISPATCHED: call {call_id}")

    receipt = _done_receipt_name(args.instance_job_id)
    arm = subprocess.run(  # subprocess-no-check-OK: armed-or-not adjudicated from stdout JSON below; failure recorded in manifest, non-fatal by design (job already live)
        [
            VENV_PY, str(REPO / "tools" / "launch_detached_process.py"),
            "--output-dir", str(out_dir),
            "--purpose", f"auto-armed harvest poller for {call_id} (fire_modal_auth_eval)",
            "--done-receipt", receipt,
            "--", VENV_PY, str(REPO / "tools" / "modal_harvest_poller.py"),
            "--call-id", call_id,
            "--output-dir", str(out_dir),
            "--deadline-s", str(poller_deadline_s),
            # Round-11 F1(b) class cure: the poller writes a scalar anchor
            # mirror under experiments/results so tac.frontier_scan can SEE a
            # row whose --output-dir is an SSD custody path it never globs.
            # The lane and label are passed here because the poller cannot know
            # them; the P0 payload still stays at --output-dir.
            "--lane-id", args.lane_id,
            "--mirror-label", args.instance_job_id,
            # The terminal claim row must key on the SAME instance/job id the spawn rows
            # used, or the compliance checker's dispatch_claim_prior_active_row sees a
            # lone terminal row with no active predecessor and reds. Without this the
            # poller keyed the row on the call_id and MAIN appended a second row by hand
            # (and mistyped a sha doing it, 2026-09-04).
            "--instance-job-id", args.instance_job_id,
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    try:
        arm_info = json.loads(arm.stdout)
        manifest["stage6_poller"] = {"pid": arm_info.get("pid"), "done_receipt": receipt}
        print(f"POLLER ARMED: pid {arm_info.get('pid')} receipt {receipt}")
    except (json.JSONDecodeError, ValueError):
        manifest["stage6_poller"] = {
            "error": arm.stdout[-400:] + arm.stderr[-400:],
            "attempted_done_receipt": receipt,
            "poller_armed": False,
        }
        # The job IS live on Modal; refusing here would strand it worse. Make the
        # unarmed state DISCOVERABLE instead of only printed — a scrollback warning
        # is not a state anyone can query later.
        (out_dir / "POLLER_UNARMED.json").write_text(
            json.dumps(
                {
                    "schema": "fire_modal_auth_eval_poller_unarmed.v1",
                    "call_id": call_id,
                    "attempted_done_receipt": receipt,
                    "instance_job_id": args.instance_job_id,
                    "arm_stdout_tail": arm.stdout[-400:],
                    "arm_stderr_tail": arm.stderr[-400:],
                    "consequence": "NO watcher will close this call; harvest it by hand or arm one manually",
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print("=" * 72)
        print("POLLER NOT ARMED — the call is LIVE with NO watcher.")
        print(f"  call_id: {call_id}")
        print(f"  marker:  {out_dir / 'POLLER_UNARMED.json'}")
        print("  ARM ONE MANUALLY before ending the turn, or the row never closes.")
        print("=" * 72)

    print(f"MANIFEST: {write_fire_manifest(out_dir, manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
