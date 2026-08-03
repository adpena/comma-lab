#!/usr/bin/env python3
"""ddm_mt1 — enumerate every discrete choice point on the live v4d chain, then triage it.

WHY THIS EXISTS
---------------
`ddm_bs2` reported a denominator of **84** discrete choice points over 10 live-chain files and
measured 5.  `ddm_lg2` independently opened 13 files and returned **64** rows.  Neither published
its per-row table, so neither denominator is re-checkable and neither is a substrate a successor
can triage against.  This module RE-DERIVES the inventory from source with a deterministic AST
scan, so the denominator is reproducible and every row carries its own site.

It does NOT re-use bs2's or lg2's row numbering; it reports its own denominator and reconciles.

THE TRIAGE RULE (validated on BOTH signs before use here)
---------------------------------------------------------
* ``ddm_pw1`` (POSITIVE): occupancy piled AT A BOUND => freeing pays.  Measured 0.9639878 ->
  0.9476091 by removing two saturated bounds (``dim0`` bracket, ``BETA_MAGS`` top entry).
* ``FEED-pb2`` (NEGATIVE): a menu that is a DIRECTION resolved by a myopic probe got WORSE when
  freed.  "Free the menu" is not automatically an improvement.
* ``ddm_dc1`` (NULL): ``st_grid``'s 7/11 dead codewords bought nothing in FORMAT because ``s_t``
  is exactly multiplicatively degenerate with the shipped translation triple (rel diff 4.539e-16,
  n600).  A menu that imposes no limit cannot be freed.

Hence four classes: AT_A_BOUND / DIRECTION / DEGENERATE / NO_OCCUPANCY_DATA.

SCOPE
-----
The live chain is established from ``experiments/stage_v4d_realized_gate.sh:41-44`` (the 5 files
actually staged into the eval submission = the DECODE set) plus the 5 encode/solve modules that
determine the archive bytes.  Ten files, named in ``LIVE_CHAIN``.  Any claim from this module is
scoped to that list and never repo-wide.

Axis: [macOS-CPU $0 static scan].  score_claim=false.  No scorer job, no training, no dispatch.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --- the live chain, DERIVED from stage_v4d_realized_gate.sh:41-44 + the build/solve producers ---
LIVE_CHAIN: tuple[tuple[str, str], ...] = (
    # (path, role)  role in {decode, encode}
    ("experiments/inflate_runner_v4d.py", "decode"),
    ("src/tac/optimization/pfs1_warp_receiver.py", "decode"),
    ("experiments/ddm_r7_token_coder.py", "decode"),
    ("src/tac/optimization/ddm_tr1_runtime.py", "decode"),
    ("src/tac/optimization/repair_entropy_coder_runtime_adapters.py", "decode"),
    ("experiments/ddm_v4d_resolve.py", "encode"),
    ("experiments/ddm_v4c_resolve.py", "encode"),
    ("experiments/ddm_v4d_build_composed_archive.py", "encode"),
    ("experiments/ddm_pfs1_ep_warp_pose_solve.py", "encode"),
    ("experiments/train_tr1_partition_renderer_mlx.py", "encode"),
)

KIND_MENU = "discrete_menu"
KIND_BOOL = "boolean_flag"
KIND_ACCEPT = "accept_reject_rule"
KIND_MODE = "mode_string"

# argparse actions that create a boolean
_BOOL_ACTIONS = {"store_true", "store_false", "BooleanOptionalAction"}


# --- exclusion classes.  NOTHING is dropped silently; every candidate carries its reason. -----
# Per the vacuity law: an empty or filtered scope is VACUOUS, never clean.  The denominator of
# each exclusion class is reported alongside the in-scope count.
EXC_LOCAL_STATE = "local_state_var"  # `accepted = False` inside a loop: state, not a choice
EXC_DUNDER = "dunder_export_list"  # `__all__`
EXC_ENTRYPOINT = "entrypoint"  # `__name__ == "__main__"`
EXC_INTEGRITY = "integrity_guard"  # fail-closed structural validation: no numeric DOF
EXC_DISPATCH_ARM = "dispatch_arm_of_counted_menu"  # `codec == "smevr"`: one arm of a menu row
EXC_ACCUMULATOR = "accumulator_not_admissible_set"  # `counts = [1, 1]`, `beta_counts = [0,0,0]`

# tokens whose presence in a comparison marks it a structural integrity check rather than a
# tunable acceptance threshold
_INTEGRITY_TOKENS = (
    "sha", "digest", "magic", "schema", "len(", ".size", ".shape", ".ndim", ".dtype",
    "SECTION_", "ARCHIVE_MEMBERS", "PACKET_", "HEADER", "_MAGIC", "sorted(", "set(",
    "section_count", "cursor", "offset", "_encode_name", "reemit_", "config_hash",
    "num_pairs", "expected", "observed", "names !=", "version !=",
)


@dataclass
class ChoicePoint:
    """One discrete choice point: a site where the code picks from a finite admissible set."""

    row_id: str = ""
    kind: str = ""
    file: str = ""
    line: int = 0
    role: str = ""
    name: str = ""
    admissible: list[object] = field(default_factory=list)
    cardinality: int | None = None
    default: object = None
    snippet: str = ""
    exclusion: str = ""  # "" == in scope

    def key(self) -> str:
        return f"{self.file}:{self.line}:{self.name}"


def _literal(node: ast.AST):
    """Return the python literal for ``node`` or the sentinel ``NotImplemented``."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return NotImplemented


def _is_homogeneous_literal_seq(val) -> bool:
    if not isinstance(val, (tuple, list, frozenset, set)):
        return False
    seq = list(val)
    if len(seq) < 2:
        return False
    kinds = {type(x) for x in seq}
    # bools are ints; treat {int,float} as one numeric kind
    if kinds <= {int, float, bool}:
        return True
    return kinds == {str}


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, path: str, role: str, src: str) -> None:
        self.path = path
        self.role = role
        self.lines = src.splitlines()
        self.rows: list[ChoicePoint] = []
        self._seen: set[str] = set()
        self._depth = 0  # >0 == inside a function/class body
        self._str_menu_members: set[str] = set()

    def visit_FunctionDef(self, node):  # noqa: N802, ANN001
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    # ---- helpers -------------------------------------------------------
    def _snippet(self, line: int) -> str:
        idx = line - 1
        if 0 <= idx < len(self.lines):
            return self.lines[idx].strip()[:160]
        return ""

    def _add(self, cp: ChoicePoint) -> None:
        if cp.key() in self._seen:
            return
        self._seen.add(cp.key())
        cp.file = self.path
        cp.role = self.role
        cp.snippet = self._snippet(cp.line)
        self.rows.append(cp)

    # ---- module-level constant menus ------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        val = _literal(node.value)
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name):
                continue
            nm = tgt.id
            if val is NotImplemented:
                continue
            if _is_homogeneous_literal_seq(val):
                seq = list(val)
                is_str = all(isinstance(x, str) for x in seq)
                kind = KIND_MODE if is_str else KIND_MENU
                exc = ""
                if nm == "__all__":
                    exc = EXC_DUNDER
                elif self._depth > 0:
                    exc = EXC_LOCAL_STATE
                elif len(set(seq)) < 2:
                    # `counts = [1, 1]` / `beta_counts = [0, 0, 0]`: an accumulator seeded to a
                    # constant, not an admissible set.  DOF is 0.
                    exc = EXC_ACCUMULATOR
                if is_str and not exc:
                    self._str_menu_members |= {s for s in seq}
                self._add(
                    ChoicePoint(
                        kind=kind,
                        line=node.lineno,
                        name=nm,
                        admissible=[_jsonable(x) for x in seq],
                        cardinality=len(seq),
                        exclusion=exc,
                    )
                )
            elif isinstance(val, bool):
                self._add(
                    ChoicePoint(
                        kind=KIND_BOOL,
                        line=node.lineno,
                        name=nm,
                        admissible=[True, False],
                        cardinality=2,
                        default=val,
                        exclusion=EXC_LOCAL_STATE if self._depth > 0 else "",
                    )
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        # dataclass fields:  flag: bool = False
        # module Finals:     AUTO_CODECS: Final = ("smevr", "brotli11")
        #                    CODEC_IDS: Final = {"smevr": 3, ...}   <- a dict-keyed menu
        if isinstance(node.target, ast.Name) and node.value is not None:
            nm = node.target.id
            val = _literal(node.value)
            if isinstance(val, bool):
                self._add(
                    ChoicePoint(
                        kind=KIND_BOOL,
                        line=node.lineno,
                        name=nm,
                        admissible=[True, False],
                        cardinality=2,
                        default=val,
                        exclusion=EXC_LOCAL_STATE if self._depth > 0 else "",
                    )
                )
            else:
                keys = list(val) if isinstance(val, dict) else val
                if _is_homogeneous_literal_seq(keys):
                    seq = list(keys)
                    is_str = all(isinstance(x, str) for x in seq)
                    exc = EXC_LOCAL_STATE if self._depth > 0 else (
                        EXC_ACCUMULATOR if len(set(seq)) < 2 else ""
                    )
                    if is_str and not exc:
                        self._str_menu_members |= set(seq)
                    self._add(
                        ChoicePoint(
                            kind=KIND_MODE if is_str else KIND_MENU,
                            line=node.lineno,
                            name=nm,
                            admissible=[_jsonable(x) for x in seq],
                            cardinality=len(seq),
                            exclusion=exc,
                        )
                    )
        self.generic_visit(node)

    # ---- argparse + accept/reject ---------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        fn = node.func
        attr = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
        if attr == "add_argument" and node.args:
            self._argparse(node)
        self.generic_visit(node)

    def _argparse(self, node: ast.Call) -> None:
        flag = _literal(node.args[0])
        if not isinstance(flag, str):
            return
        name = flag.lstrip("-").replace("-", "_")
        action = _literal(_kw(node, "action")) if _kw(node, "action") is not None else None
        choices_node = _kw(node, "choices")
        choices = _literal(choices_node) if choices_node is not None else NotImplemented
        default_node = _kw(node, "default")
        default = _literal(default_node) if default_node is not None else None
        if isinstance(action, str) and action in _BOOL_ACTIONS:
            self._add(
                ChoicePoint(
                    kind=KIND_BOOL,
                    line=node.lineno,
                    name=name,
                    admissible=[True, False],
                    cardinality=2,
                    default=(action == "store_false"),
                )
            )
            return
        if choices is not NotImplemented and _is_homogeneous_literal_seq(choices):
            seq = list(choices)
            kind = KIND_MODE if all(isinstance(x, str) for x in seq) else KIND_MENU
            self._add(
                ChoicePoint(
                    kind=kind,
                    line=node.lineno,
                    name=name,
                    admissible=[_jsonable(x) for x in seq],
                    cardinality=len(seq),
                    default=_jsonable(default),
                )
            )

    # ---- accept / reject rules ------------------------------------------
    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        # An accept/reject rule = a comparison whose body is a control-flow exit
        # (break / continue / return / raise) or a lone assignment of the "best" slot.
        body = node.body
        exits = any(isinstance(s, (ast.Break, ast.Continue, ast.Return, ast.Raise)) for s in body)
        if exits and isinstance(node.test, ast.Compare):
            desc = _compare_desc(node.test)
            if desc is not None:
                raises = any(isinstance(s, ast.Raise) for s in body)
                self._add(
                    ChoicePoint(
                        kind=KIND_ACCEPT,
                        line=node.lineno,
                        name=desc,
                        admissible=["accept", "reject"],
                        cardinality=2,
                        default="raises" if raises else "flows",
                    )
                )
        self.generic_visit(node)


def _compare_desc(cmp: ast.Compare) -> str | None:
    """Render ``a < 0.5`` style tests; skip identity/None/membership tests."""
    if len(cmp.ops) != 1 or len(cmp.comparators) != 1:
        return None
    op = cmp.ops[0]
    sym = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }.get(type(op))
    if sym is None:
        return None
    try:
        left = ast.unparse(cmp.left)
        right = ast.unparse(cmp.comparators[0])
    except Exception:
        return None
    if right in {"None", "True", "False"} or left in {"None"}:
        return None
    return f"{left} {sym} {right}"[:120]


def _jsonable(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def _classify_accept(cp: ChoicePoint, str_menu_members: set[str]) -> str:
    """Assign an exclusion class to an accept/reject candidate, or '' if it is in scope."""
    nm = cp.name
    if nm.startswith("__name__ =="):
        return EXC_ENTRYPOINT
    # `codec == "smevr"` -> one arm of an already-counted string menu, not its own choice point
    if " == '" in nm or ' == "' in nm:
        rhs = nm.split(" == ", 1)[1].strip().strip("'\"")
        if rhs in str_menu_members:
            return EXC_DISPATCH_ARM
    # fail-closed structural validation: the test has no numeric degree of freedom.  It either
    # passes or the program dies; no setting of it trades score.
    if cp.default == "raises" and any(t in nm for t in _INTEGRITY_TOKENS):
        return EXC_INTEGRITY
    return ""


def scan(paths: tuple[tuple[str, str], ...] = LIVE_CHAIN) -> tuple[list[ChoicePoint], dict]:
    rows: list[ChoicePoint] = []
    missing: list[str] = []
    per_file: dict[str, int] = {}
    for rel, role in paths:
        p = REPO / rel
        if not p.is_file():
            missing.append(rel)
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        sc = _Scanner(rel, role, src)
        sc.visit(ast.parse(src, filename=str(p)))
        members = sc._str_menu_members
        for cp in sc.rows:
            if cp.kind == KIND_ACCEPT and not cp.exclusion:
                cp.exclusion = _classify_accept(cp, members)
        rows.extend(sc.rows)
        per_file[rel] = len(sc.rows)
    rows.sort(key=lambda r: (r.file, r.line, r.name))
    letters = {}
    for r in rows:
        letters.setdefault(r.file, chr(ord("A") + len(letters)))
    counters: dict[str, int] = {}
    for r in rows:
        L = letters[r.file]
        counters[L] = counters.get(L, 0) + 1
        r.row_id = f"{L}{counters[L]}"
    in_scope = [r for r in rows if not r.exclusion]
    excluded = [r for r in rows if r.exclusion]
    exc_counts: dict[str, int] = {}
    for r in excluded:
        exc_counts[r.exclusion] = exc_counts.get(r.exclusion, 0) + 1
    denom = {
        "files_requested": len(paths),
        "files_scanned": len(per_file),
        "files_missing": missing,
        "candidates_raw": len(rows),
        "excluded_total": len(excluded),
        "excluded_by_class": exc_counts,
        "in_scope_total": len(in_scope),
        "in_scope_by_kind": {
            k: sum(1 for r in in_scope if r.kind == k)
            for k in (KIND_MENU, KIND_BOOL, KIND_ACCEPT, KIND_MODE)
        },
        "raw_by_file": per_file,
        "in_scope_by_file": {
            f: sum(1 for r in in_scope if r.file == f) for f in per_file
        },
        "file_letters": letters,
    }
    return rows, denom


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, default=None, help="write the full row table here")
    ap.add_argument("--kind", default=None, help="filter to one kind")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows, denom = scan()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "schema": "ddm_mt1_choice_point_inventory.v1",
                    "axis": "[macOS-CPU $0 static scan] NON-PROMOTABLE",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "pointer_moved": False,
                    "scope": [rel for rel, _ in LIVE_CHAIN],
                    "denominator": denom,
                    "rows": [asdict(r) for r in rows],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(denom, indent=1))
        for r in rows:
            if args.kind and r.kind != args.kind:
                continue
            card = r.cardinality if r.cardinality is not None else "-"
            adm = str(r.admissible)[:64]
            print(f"{r.row_id:5s} {r.kind:19s} {r.file}:{r.line:<5d} {r.name[:42]:44s} K={card} {adm}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
