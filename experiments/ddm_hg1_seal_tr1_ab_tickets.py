"""ddm_hg1 -- emit the ring-0 margin-hinge A/B as launcher-loadable TR1 sealed tickets.

The ddm_hg1 seal was first written in a private ``ddm_hg1_tr1_sealed_ab.v1`` shape, which is
correct in content and UNFIREABLE: ``tools/launch_tr1_run.py`` refuses any schema other than
``ddm_tb1_tr1_sealed_ticket.v1``.  This tool re-emits the same A/B as two tickets that the
governed launcher accepts, one per arm, so no one is tempted to fire raw argv around the
governor (memory ``concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806``).

The launcher's G1 gate RECOMPILES the argv from the ticket's own levers through
``TR1RendererProgramV1.compile_trainer_argv()`` and refuses on any drift.  So the argv here is
COMPILED, never hand-assembled -- a hand-written argv that merely looks right is exactly what
G1 exists to catch.

Matched-control discipline: both arms share one base lever set, taken from the sealed lv1 T3
ticket, and differ in the SEG FORM LEVER ALONE.  The lv1 base bundled the seg form together
with the seg trunk weights in a single ``tr1_seg_ce`` lever; that bundle is split here into
``hg1_seg_trunk_weights`` (identical on both arms) and the per-arm form lever, so the arms
cannot silently differ in ``--class-weight-lane`` or ``--w-seg`` while appearing matched.

Arm B's form lever is the real DSL factory ``lever_hg1_ring0_margin_hinge``, so the derived
``--margin-target`` is resolved through the registered ``margin_band_satisficing_threshold_v1``
LawRef at seal time rather than typed in as a literal.

``scope_laws`` is deliberately EMPTY: every registered scope law is a jd1/jd3 pose-retreat
policy, and none governs a seg-form A/B.  Declaring one to make a gate run would be inventing
scope.  ``ticket_hash`` is still emitted truthfully via the canonical ``ticket_payload_hash``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

#: The sealed lv1 T3 ticket the A/B inherits its base from (proven, already-fired config).
BASE_TICKET = REPO / ".omx/research/configs/ddm_lv1_t3_long_burn_lotto_v2_20260728.json"

#: Bulk root. VertigoDataTier is at 893 MiB and must NOT be targeted.
OUT_ROOT = "/Volumes/APDataStore/pact/ddm_hg1_ring0_hinge_ab_20260816"

#: The lv1 lever whose overrides bundle the seg FORM with the seg TRUNK WEIGHTS.
BASE_SEG_LEVER = "tr1_seg_ce"

#: Flags in that bundle that are trunk weights, not form -- identical on both arms.
TRUNK_WEIGHT_FLAGS = ("--class-weight-lane", "--w-seg")


class Hg1SealError(RuntimeError):
    """Fail-closed error for seal-construction violations."""


def _load_base() -> dict[str, Any]:
    base = json.loads(BASE_TICKET.read_text())
    if base.get("schema") != "ddm_tb1_tr1_sealed_ticket.v1":
        raise Hg1SealError(f"base ticket schema is {base.get('schema')!r}")
    return base


def _split_base_levers(base: dict[str, Any]) -> tuple[list[dict], dict[str, str]]:
    """Return (levers without the seg bundle, the seg bundle's overrides)."""
    kept: list[dict] = []
    seg_overrides: dict[str, str] | None = None
    for row in base["levers"]:
        if row["name"] == BASE_SEG_LEVER:
            seg_overrides = dict(row["overrides"])
            continue
        kept.append({"name": row["name"], "overrides": dict(row["overrides"]),
                     "notes": row.get("notes", "")})
    if seg_overrides is None:
        raise Hg1SealError(
            f"{BASE_TICKET.name} no longer carries a {BASE_SEG_LEVER!r} lever; the arms would "
            "not be matched on the seg trunk. Re-derive the split before sealing."
        )
    return kept, seg_overrides


def build_arms() -> dict[str, dict[str, Any]]:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.hg1_ring0_margin_hinge_levers_20260816 import (
        lever_hg1_ring0_margin_hinge,
    )
    from tac.witness_dsl.spec_tr1_renderer_20260728 import TR1RendererProgramV1

    base = _load_base()
    kept, seg_overrides = _split_base_levers(base)

    trunk = {f: seg_overrides[f] for f in TRUNK_WEIGHT_FLAGS if f in seg_overrides}
    if len(trunk) != len(TRUNK_WEIGHT_FLAGS):
        raise Hg1SealError(
            f"base seg lever is missing trunk weights: expected {TRUNK_WEIGHT_FLAGS}, "
            f"found {sorted(trunk)}"
        )
    form_ce = {f: v for f, v in seg_overrides.items() if f not in trunk}

    trunk_lever = {
        "name": "hg1_seg_trunk_weights",
        "overrides": trunk,
        "notes": ("MATCHED on both arms: the seg trunk weights lifted out of the lv1 "
                  "tr1_seg_ce bundle so the A/B differs in the seg FORM alone."),
    }
    hinge = lever_hg1_ring0_margin_hinge()

    arms = {
        "arm_a_control_ce": [*kept, trunk_lever, {
            "name": "hg1_seg_form_ce_control",
            "overrides": form_ce,
            "notes": ("MATCHED CONTROL: the lv1 seg form (ce trunk, knee -> tau_softplus) at "
                      "the trainer's own --margin-target default. Everything else is identical "
                      "to arm_b."),
        }],
        "arm_b_hinge": [*kept, trunk_lever, {
            "name": hinge.name,
            "overrides": {f: str(v) for f, v in hinge.overrides.items()},
            "notes": hinge.notes,
        }],
    }

    out: dict[str, dict[str, Any]] = {}
    for arm, levers in arms.items():
        out_dir = f"{OUT_ROOT}/{arm}"
        prog = TR1RendererProgramV1(
            levers=tuple(Lever(name=d["name"], overrides=dict(d["overrides"]),
                               notes=d.get("notes", "")) for d in levers),
            num_pairs=int(_flag(base["argv"], "--num-pairs") or 600),
            out_dir=out_dir,
            seed=int(_flag(base["argv"], "--seed") or 0),
            gt_cache=_flag(base["argv"], "--gt-cache"),
            resume_from=None,
            full_confirm="--full-confirm" in base["argv"],
            scope_laws=(),
        )
        argv = prog.compile_trainer_argv()   # fail-closed never-invent-flags
        ticket: dict[str, Any] = {
            "schema": "ddm_tb1_tr1_sealed_ticket.v1",
            "trainer": base["trainer"],
            "argv": argv,
            "levers": levers,
            "score_claim": False,
            "hg1_arm": arm,
            "hg1_out_dir": out_dir,
            "hg1_base_ticket": str(BASE_TICKET.relative_to(REPO)),
            "hg1_falsifiers": _FALSIFIERS,
            "hg1_owed_before_any_claim": _OWED,
            "hg1_serial_order": (
                "The launcher's G4 gate admits ONE n600 job at a time, so the arms CANNOT run "
                "concurrently. Fire arm_a_control_ce, let it finish, then fire arm_b_hinge. "
                "Both are fresh-start (margin_hinge is a START-ONLY seg form) and must be "
                "judged at the SEG ASYMPTOTE against each other -- never against the warm hv1 "
                "incumbent (wd3: the fresh-vs-warm floor is 2.5x)."
            ),
            "sealed_sha256": hashlib.sha256(
                json.dumps(argv, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        from tac.witness_dsl.scope_laws import ticket_payload_hash
        ticket["ticket_hash"] = ticket_payload_hash(ticket)
        out[arm] = ticket
    return out


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


_FALSIFIERS = {
    "primary": ("arm_b realized seg recovery below 25% of the ddm_hg1 re-derived ladder (half "
                "of the measured 49.211% at delta=+0.1) at the seg asymptote => the DERIVED "
                "upper bound is not reachable by this term; report the routing honestly and do "
                "NOT re-tune into a result."),
    "inert": ("hinge active-pixel fraction ~0 => INERT, run confounded, verdict inadmissible. "
              "MEASURED expected support at the sealed target: 1.91%-2.74% of ring-0."),
    "global_push": ("hinge active fraction approaching the whole frame => not a hinge but a "
                    "global margin push that will fight rate. The trainer default 1.0 sits "
                    "here (53.2% of ring-0)."),
    "pose": ("arm_b d_pose worse than arm_a beyond the run's own pose noise floor => the seg "
             "gain is bought with pose and must be re-priced at the 6.03x pose marginal "
             "before any claim."),
}

_OWED = [
    "byte-closed archive + [contest-CUDA] n600 exact eval; advisory MLX/CPU rows are never a score.",
    "pose leg MEASURED, not assumed -- and never quoted from the advisory instrument, which "
    "ddm_rn1 measured 18.2x optimistic on pose (its seg half is sound at 2.5%).",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / ".omx/research/configs")
    args = ap.parse_args(argv)

    arms = build_arms()
    a, b = arms["arm_a_control_ce"], arms["arm_b_hinge"]

    # The arms must differ in the seg form alone. Prove it here rather than assert it in prose.
    da = [x for x in a["argv"] if x not in b["argv"]]
    db = [x for x in b["argv"] if x not in a["argv"]]
    print("arm_a-only argv tokens:", da)
    print("arm_b-only argv tokens:", db)

    for arm, ticket in arms.items():
        p = args.out_dir / f"ddm_hg1_tr1_ticket_{arm}_20260816.json"
        p.write_text(json.dumps(ticket, indent=2, sort_keys=True) + "\n")
        print(f"wrote {p}  ticket_hash={ticket['ticket_hash'][:16]}  "
              f"sealed_sha256={ticket['sealed_sha256'][:16]}  argv={len(ticket['argv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
