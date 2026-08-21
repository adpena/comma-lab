#!/usr/bin/env python3
"""ddm_fs3 leg 3 -- re-screen jg3 at the MEASURED token price, trap disarmed and proved.

WHY A SHIM AND NOT A CONSTANT EDIT
----------------------------------
``ddm_fs3``'s census found 38 shipping pairs whose configuration re-selects once
tokens are priced at what they actually cost.  Materialising those configurations
means re-running ``ddm_jg3``'s per-pair sweep so the winner IS the reopened
configuration and its ``accepted`` coordinate list gets emitted -- jg3 emits that
list only for the winner, which is why the reopened configurations were not
recoverable from the retained ledgers.

Making the reopened configuration win means changing the price jg3 charges.  That
price is ``RATE_PRIOR_BITS_PER_TOKEN`` and it is read in FOUR places in TWO
different ways, which is the trap this arm named in its first round:

===========================================  ==========================  ==========
site                                         how it reads the constant   reassign?
===========================================  ==========================  ==========
``:410`` LogitPrice.bits_for fallback        module global, at call      **works**
``:807`` the configuration sweep's cost      module global, at call      **works**
``:972`` break_even_yield                    module global, at call      **works**
``:902`` ``project(..., bits_per_token=X)``  **DEFAULT ARG, at import**  **SILENT NO-OP**
===========================================  ==========================  ==========

Python binds default arguments at DEFINITION time.  So the obvious move --
``jg3.RATE_PRIOR_BITS_PER_TOKEN = new`` -- changes three of the four sites and
leaves the fourth quietly reporting a projection computed at the OLD price, in a
run whose whole point is the new one.  That is the same class as the trap
``ddm_fs2_jg5_on_candidate.py`` exists for, and it is exactly the sort of
half-applied change that produces a confidently wrong number.

So this shim does BOTH: it reassigns the global AND rebinds ``project`` with a new
default, and then it **PROVES** the disarm by introspecting the live module rather
than trusting that it worked:

* every module-global read site is re-read from the live module and must equal the
  new price;
* ``project.__kwdefaults__['bits_per_token']`` must equal the new price;
* the module source is re-scanned for any OTHER ``= RATE_PRIOR_BITS_PER_TOKEN``
  default-argument binding, and an unaccounted one is a REFUSAL, so a future jg3
  edit that adds a fifth site cannot slip past this shim silently.

WHAT MUST NOT CHANGE
--------------------
The per-site inner gate at ``:695`` prices each candidate move with the
``LogitPrice`` ranker, not with this constant, and it is deliberately LEFT ALONE.
That gate builds the candidate SITE POOL; re-pricing it would change the pool and
therefore change every sweep entry, which would destroy the control below.

THE CONTROL
-----------
With the site pool unchanged, the re-screen's sweep entries must REPRODUCE jg3's
retained sweep entries exactly -- same ``(tokens, repaired)`` at every
``(separation, keep_fraction)``.  If they do, the solver is the same solver, only
the argmin moved, and the emitted ``accepted`` list for the reopened configuration
is trustworthy.  If they do not, the re-screen is measuring a different object and
says so.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CENSUS = "/Volumes/APDataStore/pact/ddm_fs3/FS3_REOPEN_CENSUS.json"


class Fs3RescreenError(RuntimeError):
    """Fail-closed error."""


def audit_constant_sites(module_path: Path, name: str) -> dict[str, Any]:
    """Find every read of ``name`` in the module source, split by binding class.

    A default-argument binding is evaluated at import and is IMMUNE to module
    reassignment.  Every other read is a runtime global lookup.  The audit is done
    on the AST rather than by grep so a rename or a new call site cannot hide.
    """
    tree = ast.parse(module_path.read_text())
    default_arg_sites: list[dict[str, Any]] = []
    global_read_sites: list[dict[str, Any]] = []

    def is_target(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    default_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for default in list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]:
                for sub in ast.walk(default):
                    if is_target(sub):
                        default_nodes.add(id(sub))
                        default_arg_sites.append(
                            {"function": node.name, "line": sub.lineno}
                        )

    for node in ast.walk(tree):
        if is_target(node) and id(node) not in default_nodes:
            # The definition itself (`NAME = 4.1379`) is a Store, not a read.
            if isinstance(getattr(node, "ctx", None), ast.Store):
                continue
            global_read_sites.append({"line": node.lineno})

    return {
        "constant": name,
        "default_argument_bindings_immune_to_reassignment": default_arg_sites,
        "module_global_reads_cured_by_reassignment": global_read_sites,
    }


def disarm(jg3: Any, module_path: Path, new_price: float) -> dict[str, Any]:
    """Reassign the global, rebind every default-arg binding, and PROVE both."""
    audit = audit_constant_sites(module_path, "RATE_PRIOR_BITS_PER_TOKEN")
    old = float(jg3.RATE_PRIOR_BITS_PER_TOKEN)

    known = {"project"}
    unaccounted = [
        site
        for site in audit["default_argument_bindings_immune_to_reassignment"]
        if site["function"] not in known
    ]
    if unaccounted:
        raise Fs3RescreenError(
            "REFUSING: jg3 has default-argument bindings of "
            f"RATE_PRIOR_BITS_PER_TOKEN this shim does not know how to disarm: "
            f"{unaccounted}. A silently un-disarmed site is the whole reason this "
            "shim exists."
        )

    jg3.RATE_PRIOR_BITS_PER_TOKEN = new_price
    if jg3.project.__kwdefaults__ is None:
        raise Fs3RescreenError("jg3.project has no keyword defaults to disarm")
    jg3.project.__kwdefaults__["bits_per_token"] = new_price

    # PROVE it, from the live module, rather than assuming the writes took.
    proof = {
        "old_price": old,
        "new_price": new_price,
        "module_global_now": float(jg3.RATE_PRIOR_BITS_PER_TOKEN),
        "project_kwdefault_now": float(jg3.project.__kwdefaults__["bits_per_token"]),
        "break_even_yield_now": float(jg3.break_even_yield()),
        "break_even_yield_expected": new_price / jg3.BITS_PER_SEG_CELL,
    }
    failures = []
    if proof["module_global_now"] != new_price:
        failures.append("module global did not take")
    if proof["project_kwdefault_now"] != new_price:
        failures.append("project's keyword default did not take (the :902 trap)")
    if abs(proof["break_even_yield_now"] - proof["break_even_yield_expected"]) > 1e-12:
        failures.append("break_even_yield still reads the old price")
    if failures:
        raise Fs3RescreenError(f"DISARM FAILED: {failures}; proof={proof}")

    proof["verdict"] = "DISARMED_AND_PROVED"
    proof["audit"] = audit
    proof["left_alone"] = {
        "site": "ddm_jg3_joint_solve.py:695 per-site inner gate",
        "prices_with": "LogitPrice log2(p_old/p_new), not this constant",
        "why": (
            "that gate builds the candidate SITE POOL; re-pricing it would change "
            "every sweep entry and destroy the reproduction control"
        ),
    }
    return proof


def run(args: argparse.Namespace) -> int:
    census = json.loads(Path(args.census).read_text())
    price = float(census["composition_at_the_measured_marginal_price"][
        "pairs_scored_at_bits_per_token"
    ])
    identified_at = float(census["composition_at_the_measured_marginal_price"][
        "pairs_identified_at_bits_per_token"
    ])
    rows = [
        r
        for r in census["census_by_price"]["MEASURED_real_full_set"]["rows"]
        if r["ships"]
    ]
    pairs = sorted(r["pair"] for r in rows)
    if args.limit:
        pairs = pairs[: args.limit]
    if args.shard_count > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.shard_count == args.shard]

    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo))
    import ddm_jg3_joint_solve as jg3

    proof = disarm(jg3, repo / "ddm_jg3_joint_solve.py", price)
    out_dir = Path(args.store)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"DISARM_PROOF_shard{args.shard}.json").write_text(
        json.dumps(
            {
                "schema": "ddm_fs3_jg3_disarm_proof.v1",
                "arm": "ddm_fs3",
                "census": {"path": str(args.census)},
                "pairs_identified_at_bits_per_token": identified_at,
                "pairs_scored_at_bits_per_token": price,
                "pairs": pairs,
                "disarm": proof,
            },
            indent=2,
        )
    )
    print(
        f"[fs3] price {proof['old_price']} -> {proof['new_price']} "
        f"({proof['verdict']}); break_even_yield "
        f"{jg3.break_even_yield():.6f}; {len(pairs)} pairs",
        flush=True,
    )
    if args.dry_run:
        return 0

    argv = [
        "solve",
        "--store",
        str(args.store),
        "--tag",
        args.tag,
        "--pair-list",
        ",".join(str(p) for p in pairs),
    ]
    if args.resume:
        argv.append("--resume")
    argv += args.passthrough
    return int(jg3.main(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", default=DEFAULT_CENSUS)
    parser.add_argument("--store", required=True)
    parser.add_argument("--tag", default="fs3_reopen38")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.passthrough and args.passthrough[0] == "--":
        args.passthrough = args.passthrough[1:]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
