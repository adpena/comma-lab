# SPDX-License-Identifier: MIT
"""SAFE-COMPILE: a determinism-first, bit-certified ``mx.compile`` layer (#252).

Operator 2026-07-08: *"Can we engineer our own mx.compile using fixed point or
other techniques for fidelity and engineering determinism"*.

Why this exists
---------------
``mx.compile`` recovers the per-launch fusion speedup MLX otherwise loses to
lazy-eval overhead, but it was **measured-excluded** from the witness R operator
(``.omx/research/v7_compute_exploitation_audit_20260708.md`` lever #3): fusion
introduces fp-contraction (fused multiply-add = ONE rounding instead of two) that
flips the uint8-STE argmax knife-edge (fwd Δ~4.8e-3 MEASURED 2026-07-03) and thus
d_seg. A blanket ``mx.compile`` is therefore a NO-FAKE hazard.

The engineering answer is NOT "never compile" — it is "compile ONLY the subgraphs
where fusion provably cannot change a single output bit, and PROVE it per region
with certificates, not vibes." This module is that layer:

1. A **safe-subgraph partitioner** (an explicit allowlist API — we nominate the
   elementwise-heavy hot regions; the v2 auto graph-walker is a scope note) that
   classifies a region SAFE-TO-COMPILE (pure elementwise, no reduction /
   matmul / conv) vs UNSAFE (everything else — stays on the uncompiled path, or,
   for reductions, routes to the #348 fixed-order primitive).
2. A **per-region bit-certification harness**: (a) bit-equality vs the uncompiled
   reference on real witness tensors (max|Δ| must be EXACTLY 0), (b) cross-process
   determinism via the #348 probe pattern (N separate processes, identical hash),
   (c) measured wall-clock delta. A region ships ENABLED only with all three
   recorded in a machine-readable manifest; a failing region is recorded FAILED
   with the first-diverging output (the fp-contraction fingerprint).
3. A **fixed-point / fixed-order option** for reduction regions (route to the
   deterministic accumulation primitive instead of ``mx.compile``).
4. A **default-OFF trainer hook** (``--safe-compile-regions``): byte-identical when
   OFF; when ON, only manifest-certified regions activate. A v7.1 lever, NOT a
   launch-gating flip (register in the DSL lever factory + deferral ledger).

Authority note (NO-FAKE, CLAUDE.md "MPS auth eval is NOISE" / "surrogate ≠ authority")
-------------------------------------------------------------------------------------
A bit-identity certificate is a valid on-device FACT ([macOS-MLX research-signal]);
it is NEVER a score. The d_seg/d_pose verdict stays the frozen numpy/torch-CPU
scorer on the byte-closed archive. This layer buys SPEED (score-neutral by
construction: a certified region is bit-identical to the uncompiled path) — nothing
here moves the pointer (0.19110). Only a byte-closed ``upstream/evaluate.py`` n600
exact row does.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = [
    "SCHEMA_VERSION",
    "RegionClass",
    "ELEMENTWISE_SAFE_OPS",
    "REDUCTION_OPS",
    "CONTRACTION_OPS",
    "classify_op_kinds",
    "SafeCompileRegion",
    "register_region",
    "get_region",
    "REGION_BUILDERS",
    "RegionCertificate",
    "CertificationManifest",
    "certify_region",
    "certify_fixed_point_reduction",
    "safe_compile",
    "maybe_safe_compile",
    "resolve_enabled_regions",
    "fixed_order_reduce_mlx",
    "route_reduction_to_fixed_point",
    # v2 (auto-discovery + trust closure)
    "ast_op_kinds",
    "trace_named_ops",
    "CandidateRow",
    "discover_candidate",
    "discover_candidates",
    "host_fingerprint",
    "fingerprint_matches",
    "manifest_fingerprint_ok",
    "emit_kernel_candidates",
    "install_safe_compiled_regions",
    "STEP_REDUCTION_SITES",
    "HOSC_BETA_ANNEAL_RANGE",
]

SCHEMA_VERSION = "mlx_safe_compile.manifest.v2"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1. PARTITIONER — op-kind classification.
#
# The classifier is a HEURISTIC pre-filter over a region's DECLARED op kinds; it
# is NOT a proof (the certification harness is the arbiter). It exists so we do
# not even attempt to compile a region that structurally cannot be bit-safe
# (any reduction / matmul / conv reorders accumulation and/or contracts FMAs).
# ---------------------------------------------------------------------------

# Pure elementwise ops whose fusion CANNOT change accumulation order (no reduction).
# NOTE: presence of BOTH ``mul`` and ``add`` in a chain is FMA-contraction-eligible
# (``a*b + c`` can fuse to one rounding) — still classified SAFE_CANDIDATE, but the
# ``fma_contraction_possible`` flag warns the certifier this region is the most
# likely to FAIL empirical bit-equality. Unary-only / add-only / mul-only chains are
# the genuinely low-risk SAFE regions.
ELEMENTWISE_SAFE_OPS = frozenset(
    {
        "sin", "cos", "tan", "tanh", "sinh", "cosh", "exp", "log", "log1p",
        "sigmoid", "relu", "gelu", "silu", "abs", "neg", "sign", "sqrt", "rsqrt",
        "reciprocal", "square", "clip", "clamp", "maximum", "minimum",
        "add", "subtract", "sub", "multiply", "mul", "divide", "div",
        "power", "erf", "floor", "ceil", "round", "where", "astype", "stop_gradient",
    }
)

# Structural / layout ops: rearrange memory, do NO arithmetic => fusion cannot
# change a bit (no accumulation reorder, no FMA). Treated SAFE alongside pure
# elementwise. ``take``/``gather`` are DELIBERATELY EXCLUDED — their VJP is the
# #348 dup-index scatter-add hazard; a region containing them stays UNSAFE_UNKNOWN.
STRUCTURAL_SAFE_OPS = frozenset(
    {"concatenate", "concat", "reshape", "broadcast_to", "expand_dims", "squeeze",
     "transpose", "swapaxes", "moveaxis", "stack", "split", "flatten"}
)

# Reductions collapse an axis => accumulation-order sensitive under fusion.
REDUCTION_OPS = frozenset(
    {"sum", "mean", "prod", "max", "min", "logsumexp", "softmax", "logaddexp",
     "cumsum", "norm", "var", "std", "argmax", "argmin"}
)

# Contractions / index-scatter => reorder accumulation and/or use atomics.
CONTRACTION_OPS = frozenset(
    {"matmul", "@", "dot", "conv2d", "conv1d", "conv", "einsum",
     "scatter_add", "at_add", "take_grad", "gather_grad"}
)


class RegionClass(str, Enum):
    """Coarse compile-eligibility verdict for a nominated region."""

    SAFE_ELEMENTWISE = "safe_elementwise"        # compile-eligible (still must certify)
    UNSAFE_REDUCTION = "unsafe_reduction"        # route to fixed-point instead
    UNSAFE_CONTRACTION = "unsafe_contraction"    # matmul/conv/scatter — not compile-eligible
    UNSAFE_UNKNOWN = "unsafe_unknown"            # op kind not in any allowlist


@dataclass(frozen=True)
class OpKindVerdict:
    region_class: RegionClass
    reason: str
    fma_contraction_possible: bool = False

    @property
    def compile_eligible(self) -> bool:
        return self.region_class is RegionClass.SAFE_ELEMENTWISE

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_class": self.region_class.value,
            "reason": self.reason,
            "fma_contraction_possible": self.fma_contraction_possible,
            "compile_eligible": self.compile_eligible,
        }


def classify_op_kinds(op_kinds: "list[str] | frozenset[str] | set[str]") -> OpKindVerdict:
    """Classify a region by the set of op kinds it (declares it) contains.

    Falling-rule order (first match wins): contraction > reduction > unknown-op >
    elementwise. This is deliberately conservative — a single reduction/contraction
    op makes the whole region ineligible for ``mx.compile`` fusion.
    """

    kinds = {str(k).strip().lower() for k in op_kinds if str(k).strip()}
    if not kinds:
        return OpKindVerdict(RegionClass.UNSAFE_UNKNOWN, "empty op-kind set (nothing to classify)")
    contraction = sorted(kinds & CONTRACTION_OPS)
    if contraction:
        return OpKindVerdict(
            RegionClass.UNSAFE_CONTRACTION,
            f"contains contraction op(s) {contraction} (accumulation reorder / atomics)",
        )
    reduction = sorted(kinds & REDUCTION_OPS)
    if reduction:
        return OpKindVerdict(
            RegionClass.UNSAFE_REDUCTION,
            f"contains reduction op(s) {reduction} (accumulation-order sensitive; "
            f"route to fixed-order primitive, not mx.compile)",
        )
    unknown = sorted(kinds - ELEMENTWISE_SAFE_OPS - STRUCTURAL_SAFE_OPS)
    if unknown:
        return OpKindVerdict(
            RegionClass.UNSAFE_UNKNOWN,
            f"contains op(s) not in the elementwise/structural allowlist {unknown}",
        )
    fma = ("mul" in kinds or "multiply" in kinds) and ("add" in kinds or "sub" in kinds
                                                        or "subtract" in kinds)
    reason = "all ops elementwise (no reduction/contraction)"
    if fma:
        reason += "; mul+add present => FMA-contraction-eligible (bit-equality NOT guaranteed)"
    return OpKindVerdict(RegionClass.SAFE_ELEMENTWISE, reason, fma_contraction_possible=fma)


# ---------------------------------------------------------------------------
# Region registry — the practical allowlist. A region is a pure callable plus a
# builder that (re)constructs it + seeded example inputs. The builder must be
# importable + deterministic so a SEPARATE process can reconstruct the region for
# the cross-process determinism certificate (mirrors the #348 probe's run_op).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafeCompileRegion:
    """A nominated compile-candidate region.

    ``build`` returns ``(fn, make_inputs)`` where ``fn(*inputs)`` is the pure MLX
    callable and ``make_inputs(seed) -> tuple`` produces one seeded input tuple.
    ``op_kinds`` is the DECLARED op-kind set (drives the partitioner).
    """

    region_id: str
    build: Callable[[], "tuple[Callable[..., Any], Callable[[int], tuple]]"]
    op_kinds: frozenset[str]
    notes: str = ""

    def classify(self) -> OpKindVerdict:
        return classify_op_kinds(self.op_kinds)


REGION_BUILDERS: dict[str, SafeCompileRegion] = {}


def register_region(region: SafeCompileRegion) -> SafeCompileRegion:
    """Register a nominated region under its ``region_id`` (idempotent overwrite)."""

    REGION_BUILDERS[region.region_id] = region
    return region


def get_region(region_id: str) -> SafeCompileRegion:
    if region_id not in REGION_BUILDERS:
        raise KeyError(
            f"region {region_id!r} not registered; known: {sorted(REGION_BUILDERS)}"
        )
    return REGION_BUILDERS[region_id]


# ---------------------------------------------------------------------------
# 1b. AUTO-DISCOVERY (the v2 partitioner) — systematic op-kind extraction.
#
# The v1 partitioner needed a HAND-DECLARED ``op_kinds`` set per region. v2
# extracts the op-kind set from the candidate itself via TWO complementary
# mechanisms whose UNION feeds ``classify_op_kinds``:
#
#   (i)  AST scan of the function source — catches the operator-overload ops
#        (``@`` matmul, ``*`` mul, ``+`` add, ``-`` sub, ``/`` div) that a
#        runtime trace CANNOT see (they dispatch to C-extension array dunders),
#        plus named ``mx.<op>`` / ``nn.<op>`` calls. This is the SAFETY net: a
#        hidden ``@`` matmul is what would misclassify a region SAFE.
#   (ii) A runtime TRACE shim over the mx namespace — CONFIRMS which named
#        ``mx.<op>`` calls actually execute at real-shaped inputs (a function
#        may import an op it never calls on a given branch).
#
# Neither alone is sufficient (AST over-reports dead branches; the trace
# under-reports operator overloads); the UNION is conservative-correct.
# ---------------------------------------------------------------------------

# AST BinOp node -> op-kind name. These are the operator-overload ops a runtime
# mx-namespace trace cannot observe (they go through ``mx.array.__mul__`` etc.).
_AST_BINOP_KIND = {
    "Add": "add", "Sub": "subtract", "Mult": "mul", "Div": "divide",
    "MatMult": "matmul", "Pow": "power", "Mod": "mod",
}


def ast_op_kinds(fn: Callable[..., Any]) -> "frozenset[str]":
    """Extract op-kind names from a function's SOURCE via AST.

    Captures: BinOp operators (``@`` -> matmul, ``*`` -> mul, ``+`` -> add, ...)
    AND ``mx.<name>`` / ``nn.<name>`` / bare ``<name>(...)`` call targets whose
    name is in a known op vocabulary. Returns the UNION. On any source/parse
    failure returns the empty set (the trace + declared kinds still apply).

    This is deliberately over-inclusive on the CONTRACTION/REDUCTION side (better
    to over-flag a region UNSAFE than to miss a hidden matmul)."""

    import ast
    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return frozenset()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return frozenset()

    known = ELEMENTWISE_SAFE_OPS | STRUCTURAL_SAFE_OPS | REDUCTION_OPS | CONTRACTION_OPS
    kinds: set[str] = set()

    class _V(ast.NodeVisitor):
        def visit_BinOp(self, node: "ast.BinOp") -> None:
            k = _AST_BINOP_KIND.get(type(node.op).__name__)
            if k:
                kinds.add(k)
            self.generic_visit(node)

        def visit_Call(self, node: "ast.Call") -> None:
            name = None
            f = node.func
            if isinstance(f, ast.Attribute):
                name = f.attr
            elif isinstance(f, ast.Name):
                name = f.id
            if name:
                nl = name.lower()
                if nl in known:
                    kinds.add(nl)
                # nn.Linear / nn.Conv* layer construction or a *matmul-bearing*
                # layer call is a contraction; flag conservatively.
                if nl in {"linear", "conv1d", "conv2d", "conv3d", "conv"}:
                    kinds.add("matmul" if nl == "linear" else "conv")
            self.generic_visit(node)

    _V().visit(tree)
    return frozenset(kinds)


class _OpTraceShim:
    """Context manager that wraps named ``mlx.core`` callables to record which
    ones are invoked during a traced call. Only the op vocabulary is wrapped;
    operator overloads (``*``/``+``/``@``) are invisible here (AST covers them)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self._orig: dict[str, Any] = {}

    def __enter__(self) -> "_OpTraceShim":
        import mlx.core as mx

        vocab = ELEMENTWISE_SAFE_OPS | STRUCTURAL_SAFE_OPS | REDUCTION_OPS | {"take"}
        for name in vocab:
            orig = getattr(mx, name, None)
            if orig is None or not callable(orig):
                continue
            self._orig[name] = orig

            def _make(nm: str, fn: Any) -> Any:
                def _wrapped(*a: Any, **k: Any) -> Any:
                    self.seen.add(nm)
                    return fn(*a, **k)
                return _wrapped

            try:
                setattr(mx, name, _make(name, orig))
            except (AttributeError, TypeError):  # pragma: no cover - immutable attr
                self._orig.pop(name, None)
        return self

    def __exit__(self, *exc: Any) -> None:
        import mlx.core as mx

        for name, orig in self._orig.items():
            try:
                setattr(mx, name, orig)
            except (AttributeError, TypeError):  # pragma: no cover
                pass


def trace_named_ops(
    build: "Callable[[], tuple[Callable[..., Any], Callable[[int], tuple]]]",
    *,
    seed: int = 0,
    device: str = "cpu",
) -> "frozenset[str]":
    """Run a region's ``fn`` ONCE under the trace shim and return the set of named
    ``mx.<op>`` calls it executed. ``device`` defaults to CPU so discovery never
    contends with a live GPU training run (op-kind identity is device-agnostic)."""

    import mlx.core as mx

    prev = mx.default_device()
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    try:
        fn, make_inputs = build()
        args = make_inputs(seed)
        with _OpTraceShim() as shim:
            out = fn(*args)
            mx.eval(out)
        return frozenset(shim.seen)
    finally:
        mx.set_default_device(prev)


@dataclass(frozen=True)
class CandidateRow:
    """One row of the auto-discovery candidate table."""

    region_id: str
    ast_ops: frozenset[str]
    traced_ops: frozenset[str]
    declared_ops: frozenset[str]
    verdict: OpKindVerdict
    notes: str = ""

    @property
    def union_ops(self) -> "frozenset[str]":
        return self.ast_ops | self.traced_ops | self.declared_ops

    @property
    def compile_eligible(self) -> bool:
        return self.verdict.compile_eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "ast_ops": sorted(self.ast_ops),
            "traced_ops": sorted(self.traced_ops),
            "declared_ops": sorted(self.declared_ops),
            "union_ops": sorted(self.union_ops),
            "region_class": self.verdict.region_class.value,
            "compile_eligible": self.compile_eligible,
            "fma_contraction_possible": self.verdict.fma_contraction_possible,
            "reason": self.verdict.reason,
            "notes": self.notes,
        }


def discover_candidate(region_id: str, *, trace: bool = True,
                       device: str = "cpu") -> CandidateRow:
    """Classify ONE registered region by the UNION of {AST-scanned, traced,
    declared} op kinds. ``trace=False`` (or no MLX) uses AST + declared only."""

    region = get_region(region_id)
    declared = frozenset(str(k).lower() for k in region.op_kinds)
    try:
        fn, _ = region.build()
        ast_ops = ast_op_kinds(fn)
    except Exception:  # pragma: no cover - defensive (no mlx / bad builder)
        ast_ops = frozenset()
    traced: frozenset[str] = frozenset()
    if trace:
        try:
            traced = trace_named_ops(region.build, device=device)
        except Exception:  # pragma: no cover - MLX absent / trace failure
            traced = frozenset()
    union = ast_ops | traced | declared
    verdict = classify_op_kinds(union if union else declared)
    return CandidateRow(region_id, ast_ops, traced, declared, verdict, notes=region.notes)


def discover_candidates(region_ids: "list[str] | None" = None, *, trace: bool = True,
                        device: str = "cpu") -> list[CandidateRow]:
    """Sweep the registered regions (or a subset) into the candidate table.

    The v2 replacement for hand-nomination: run this over the witness hot-path
    candidate inventory to get each region's SAFE / UNSAFE / MIXED verdict + why,
    then certify every SAFE row through the v1 harness."""

    ids = sorted(REGION_BUILDERS) if region_ids is None else list(region_ids)
    return [discover_candidate(rid, trace=trace, device=device) for rid in ids]


# ---------------------------------------------------------------------------
# 2. CERTIFICATION HARNESS
# ---------------------------------------------------------------------------


@dataclass
class RegionCertificate:
    """Per-region evidence bundle. A region ships ENABLED only when
    ``verdict == "CERTIFIED"`` (bit_equal AND determinism n/n)."""

    region_id: str
    region_class: str
    verdict: str  # CERTIFIED | FAILED | ERROR
    bit_equal: bool
    max_abs_delta: float
    determinism_n_pass: int
    determinism_n: int
    speedup: float                       # uncompiled_ms / compiled_ms (>1 == faster)
    uncompiled_ms: float
    compiled_ms: float
    input_coverage: int                  # number of distinct real inputs bit-checked
    input_shapes: list[str] = field(default_factory=list)
    reference: str = "uncompiled_reference"
    fma_contraction_possible: bool = False
    first_divergence: dict[str, Any] | None = None  # fp-contraction fingerprint on failure
    fixed_point_routed: bool = False
    reason: str = ""
    input_source: str = "synthetic"      # synthetic | real_checkpoint

    @property
    def cross_process_deterministic(self) -> bool:
        return self.determinism_n >= 2 and self.determinism_n_pass == self.determinism_n

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_class": self.region_class,
            "verdict": self.verdict,
            "bit_equal": self.bit_equal,
            "max_abs_delta": self.max_abs_delta,
            "determinism": f"{self.determinism_n_pass}/{self.determinism_n}",
            "cross_process_deterministic": self.cross_process_deterministic,
            "speedup": round(self.speedup, 4),
            "uncompiled_ms": round(self.uncompiled_ms, 4),
            "compiled_ms": round(self.compiled_ms, 4),
            "input_coverage": self.input_coverage,
            "input_shapes": self.input_shapes,
            "input_source": self.input_source,
            "reference": self.reference,
            "fma_contraction_possible": self.fma_contraction_possible,
            "first_divergence": self.first_divergence,
            "fixed_point_routed": self.fixed_point_routed,
            "reason": self.reason,
        }


def _flatten(tree: Any) -> list[Any]:
    """Flatten an MLX pytree (array / tuple / list / dict) to a list of arrays."""

    if isinstance(tree, (tuple, list)):
        out: list[Any] = []
        for t in tree:
            out.extend(_flatten(t))
        return out
    if isinstance(tree, dict):
        out = []
        for k in sorted(tree):
            out.extend(_flatten(tree[k]))
        return out
    return [tree]


def _hash_outputs(outs: Any) -> str:
    import numpy as np

    m = hashlib.sha256()
    for a in _flatten(outs):
        m.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    return m.hexdigest()[:16]


def _bit_equality_and_timing(
    fn: Callable[..., Any],
    make_inputs: Callable[[int], tuple],
    *,
    n_inputs: int,
    reps: int,
) -> dict[str, Any]:
    """Compare compiled vs uncompiled on ``n_inputs`` seeded inputs (max|Δ| must be
    EXACTLY 0) and time both over ``reps`` reps. Runs in-process."""

    import mlx.core as mx
    import numpy as np

    compiled = mx.compile(fn)
    max_delta = 0.0
    first_div: dict[str, Any] | None = None
    shapes: list[str] = []
    for i in range(n_inputs):
        args = make_inputs(1000 + i)
        ou = fn(*args)
        oc = compiled(*args)
        mx.eval(ou, oc)
        if i == 0:
            shapes = [str(tuple(np.asarray(a).shape)) for a in _flatten(args)]
        lu, lc = _flatten(ou), _flatten(oc)
        for leaf_idx, (a, b) in enumerate(zip(lu, lc)):
            an, bn = np.asarray(a), np.asarray(b)
            d = float(np.max(np.abs(an - bn))) if an.size else 0.0
            if d > max_delta:
                max_delta = d
                if d > 0.0 and first_div is None:
                    flat = np.abs(an - bn).ravel()
                    pos = int(np.argmax(flat))
                    first_div = {
                        "input_index": i, "output_leaf": leaf_idx,
                        "flat_position": pos, "abs_delta": d,
                        "uncompiled": float(an.ravel()[pos]),
                        "compiled": float(bn.ravel()[pos]),
                    }
    # timing (warm both graphs first)
    warm = make_inputs(7)
    mx.eval(fn(*warm), compiled(*warm))

    def _time(g: Callable[..., Any]) -> float:
        args = make_inputs(11)
        t0 = time.perf_counter()
        for _ in range(reps):
            mx.eval(g(*args))
        return (time.perf_counter() - t0) * 1000.0 / reps

    uncompiled_ms = _time(fn)
    compiled_ms = _time(compiled)
    return {
        "bit_equal": max_delta == 0.0,
        "max_abs_delta": max_delta,
        "first_divergence": first_div,
        "input_shapes": shapes,
        "uncompiled_ms": uncompiled_ms,
        "compiled_ms": compiled_ms,
        "speedup": (uncompiled_ms / compiled_ms) if compiled_ms > 0 else 0.0,
    }


def _cross_process_determinism(region_id: str, *, n: int, device: str,
                               timeout_s: int = 300) -> tuple[int, int]:
    """Spawn ``n`` separate processes that each rebuild + COMPILE the region and hash
    its output on a fixed seed. Returns ``(n_pass, n)`` where n_pass == n iff all
    hashes are identical (cross-process bit-identical). Mirrors the #348 probe."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO, "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    hashes: list[str] = []
    for _ in range(n):
        p = subprocess.run(
            [sys.executable, "-m", "tac.mlx_safe_compile", "--certify-child", region_id, device],
            env=env, capture_output=True, text=True, timeout=timeout_s,
        )
        if p.returncode != 0:
            return 0, n
        try:
            row = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0, n
        hashes.append(row["hash"])
    n_pass = n if (len(hashes) == n and len(set(hashes)) == 1) else 0
    return n_pass, n


def certify_region(
    region_id: str,
    *,
    n_inputs: int = 32,
    n_determinism: int = 5,
    reps: int = 40,
    device: str = "gpu",
    cross_process: bool = True,
    input_source: str = "synthetic",
) -> RegionCertificate:
    """Full 3-certificate gate for a compile-candidate region.

    (a) bit-equality vs uncompiled on ``n_inputs`` real inputs (max|Δ| == 0 EXACTLY);
    (b) cross-process determinism over ``n_determinism`` processes (skippable);
    (c) measured wall-clock delta. Verdict CERTIFIED iff (a) AND (b) pass.

    Reduction/contraction/unknown regions are NOT compile-eligible: the verdict is
    FAILED with the classifier reason (reductions should route via
    ``certify_fixed_point_reduction`` instead). This makes a known-unsafe region
    (e.g. a ``sum``/``logsumexp`` reduction) fail certification deterministically,
    without depending on flaky empirical divergence.
    """

    region = get_region(region_id)
    verdict_class = region.classify()
    base = dict(
        region_id=region_id, region_class=verdict_class.region_class.value,
        fma_contraction_possible=verdict_class.fma_contraction_possible,
        input_source=input_source,
    )
    if not verdict_class.compile_eligible:
        return RegionCertificate(
            verdict="FAILED", bit_equal=False, max_abs_delta=float("nan"),
            determinism_n_pass=0, determinism_n=0, speedup=0.0,
            uncompiled_ms=0.0, compiled_ms=0.0, input_coverage=0,
            reason=f"not compile-eligible: {verdict_class.reason}", **base,
        )
    try:
        import mlx.core as mx

        # BUGFIX (v2): pin the in-process bit-equality + timing to the REQUESTED device so it
        # matches the cross-process determinism device. v1 measured bit-equality on the process
        # DEFAULT device regardless of ``device=`` (which only reached the child procs) — a
        # certify_region('cpu') could measure GPU bit-equality. fp-contraction is device-specific
        # (MEASURED: hosc compiled-vs-uncompiled == 0 on GPU but 5.96e-8 on CPU), so a mismatched
        # device silently mis-certifies. Set + restore the default device around the measurement.
        _prev_dev = mx.default_device()
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        try:
            fn, make_inputs = region.build()
            eq = _bit_equality_and_timing(fn, make_inputs, n_inputs=n_inputs, reps=reps)
        finally:
            mx.set_default_device(_prev_dev)
    except Exception as exc:  # pragma: no cover - defensive (bad builder / no mlx)
        return RegionCertificate(
            verdict="ERROR", bit_equal=False, max_abs_delta=float("nan"),
            determinism_n_pass=0, determinism_n=0, speedup=0.0,
            uncompiled_ms=0.0, compiled_ms=0.0, input_coverage=0,
            reason=f"harness error: {type(exc).__name__}: {exc}", **base,
        )

    n_pass, n_det = (0, 0)
    if cross_process and eq["bit_equal"]:
        n_pass, n_det = _cross_process_determinism(region_id, n=n_determinism, device=device)

    bit_equal = bool(eq["bit_equal"])
    det_ok = (n_det >= 2 and n_pass == n_det) if cross_process else True
    verdict = "CERTIFIED" if (bit_equal and det_ok) else "FAILED"
    reason = "all certificates pass" if verdict == "CERTIFIED" else (
        "bit-equality FAILED (fp-contraction detected)" if not bit_equal
        else "cross-process determinism FAILED"
    )
    return RegionCertificate(
        verdict=verdict, bit_equal=bit_equal, max_abs_delta=eq["max_abs_delta"],
        determinism_n_pass=n_pass, determinism_n=n_det, speedup=eq["speedup"],
        uncompiled_ms=eq["uncompiled_ms"], compiled_ms=eq["compiled_ms"],
        input_coverage=n_inputs, input_shapes=eq["input_shapes"],
        first_divergence=eq["first_divergence"], reason=reason, **base,
    )


# ---------------------------------------------------------------------------
# 3. FIXED-POINT / FIXED-ORDER option for reductions (route away from mx.compile)
# ---------------------------------------------------------------------------


def fixed_order_reduce_mlx(x, axis: int = -1):
    """Deterministic fixed-order sum along ``axis`` (sequential left-fold).

    ``mx.sum`` is already cross-process bit-identical on MLX (#348 probe:
    ``sum_reduce`` / ``mean_axis`` PASS), but its accumulation ORDER is a kernel
    detail; a reduction region that must be certified against a STABLE order routes
    here. The left-fold pins the order explicitly (the MLX analogue of the #348
    ``kahan_compensated_sum`` reference in ``deterministic_primitives``). O(N) launches
    — for large N prefer ``mx.sum`` after certifying it cross-process on the target.
    """

    import mlx.core as mx

    n = x.shape[axis]
    if n == 0:
        return mx.sum(x, axis=axis)
    acc = mx.take(x, mx.array([0]), axis=axis).squeeze(axis)
    for i in range(1, n):
        acc = acc + mx.take(x, mx.array([i]), axis=axis).squeeze(axis)
    return acc


def route_reduction_to_fixed_point(region_id: str) -> SafeCompileRegion:
    """Return a fixed-order-reduce version of a reduction region for certification.

    The returned region replaces the region's reduction with ``fixed_order_reduce_mlx``;
    it is certified by ``certify_fixed_point_reduction`` (determinism-only — it does not
    go through ``mx.compile``)."""

    region = get_region(region_id)
    vc = region.classify()
    if vc.region_class is not RegionClass.UNSAFE_REDUCTION:
        raise ValueError(
            f"region {region_id!r} is {vc.region_class.value}, not a reduction; "
            "fixed-point routing applies to reductions only"
        )
    return region


def certify_fixed_point_reduction(
    region_id: str,
    *,
    n_determinism: int = 5,
    device: str = "gpu",
    cross_process: bool = True,
) -> RegionCertificate:
    """Certify a reduction region's FIXED-ORDER form for cross-process determinism.

    No ``mx.compile`` and no compiled-vs-uncompiled bit-equality (a reduction is not
    compile-eligible); the guarantee is that the FIXED-ORDER reduction is deterministic
    across processes. ``bit_equal`` is set True by construction (the fixed order is the
    reference), ``fixed_point_routed=True``."""

    route_reduction_to_fixed_point(region_id)  # validates it IS a reduction (raises otherwise)
    n_pass, n_det = (0, 0)
    if cross_process:
        n_pass, n_det = _cross_process_determinism(region_id, n=n_determinism, device=device)
    det_ok = (n_det >= 2 and n_pass == n_det) if cross_process else True
    return RegionCertificate(
        region_id=region_id, region_class=RegionClass.UNSAFE_REDUCTION.value,
        verdict="CERTIFIED" if det_ok else "FAILED",
        bit_equal=True, max_abs_delta=0.0,
        determinism_n_pass=n_pass, determinism_n=n_det, speedup=1.0,
        uncompiled_ms=0.0, compiled_ms=0.0, input_coverage=0,
        fixed_point_routed=True, reference="fixed_order_reduce_mlx",
        reason="fixed-order reduction (determinism-only certificate)",
        input_source="fixed_point",
    )


# ---------------------------------------------------------------------------
# 3b. PER-CHIP TRUST CLOSURE — a certificate is bit-identity evidence for the
# SPECIFIC {chip, macOS build, MLX version} it was measured on. fp-contraction
# is a per-chip Metal-codegen property; a manifest carried to a different host
# is STALE and MUST fail closed (recertify), exactly like the ``--fused-r-kernel``
# per-chip parity gate.
# ---------------------------------------------------------------------------


def host_fingerprint() -> dict[str, str]:
    """The {chip, macOS build, MLX version} triple that scopes a certificate.

    Best-effort: any unavailable field is "" (never raises)."""

    chip = macos_build = mlx_version = ""
    try:
        import subprocess as _sp

        chip = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                       capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # pragma: no cover - non-mac / sysctl absent
        pass
    try:
        import subprocess as _sp

        macos_build = _sp.run(["sw_vers", "-buildVersion"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # pragma: no cover
        pass
    try:
        import mlx.core as _mx

        mlx_version = str(getattr(_mx, "__version__", ""))
    except Exception:  # pragma: no cover - mlx absent
        pass
    return {"chip": chip, "macos_build": macos_build, "mlx_version": mlx_version}


def fingerprint_matches(recorded: dict[str, str] | None,
                        host: dict[str, str] | None = None) -> bool:
    """True iff every NON-EMPTY recorded field equals the host's. An empty
    recorded fingerprint (legacy v1 manifest) matches by construction (nothing to
    contradict); a recorded field that the host cannot read ("") does not veto."""

    if not recorded:
        return True
    h = host if host is not None else host_fingerprint()
    for key in ("chip", "macos_build", "mlx_version"):
        want = str(recorded.get(key, "")).strip()
        got = str(h.get(key, "")).strip()
        if want and got and want != got:
            return False
    return True


def manifest_fingerprint_ok(manifest: "CertificationManifest | None",
                            host: dict[str, str] | None = None) -> tuple[bool, str]:
    """(ok, reason) for a manifest's fingerprint vs the host. A missing manifest or
    empty fingerprint is OK (nothing to contradict — legacy/back-compat)."""

    if manifest is None:
        return True, "no manifest"
    fp = getattr(manifest, "fingerprint", None) or {}
    if not fp:
        return True, "manifest carries no fingerprint (legacy v1 / unscoped)"
    h = host if host is not None else host_fingerprint()
    if fingerprint_matches(fp, h):
        return True, "fingerprint matches host"
    return False, (
        f"STALE certificate: manifest fingerprint {fp} != host {h} — recertify on "
        "this host (fp-contraction is per-chip; fail-closed)"
    )


# ---------------------------------------------------------------------------
# 4. MANIFEST (machine-readable evidence)
# ---------------------------------------------------------------------------


@dataclass
class CertificationManifest:
    """Machine-readable record of every region's certificate. THE evidence surface —
    a region activates in the trainer only if its manifest row is CERTIFIED AND the
    manifest's host fingerprint matches the current host."""

    schema_version: str = SCHEMA_VERSION
    device: str = "gpu"
    created_utc: str = ""
    fingerprint: dict[str, str] = field(default_factory=dict)
    rows: dict[str, RegionCertificate] = field(default_factory=dict)

    def add(self, cert: RegionCertificate) -> "CertificationManifest":
        self.rows[cert.region_id] = cert
        return self

    def certified_ids(self) -> list[str]:
        return sorted(r for r, c in self.rows.items() if c.verdict == "CERTIFIED")

    def is_certified(self, region_id: str) -> bool:
        c = self.rows.get(region_id)
        return c is not None and c.verdict == "CERTIFIED"

    def failed_ids(self) -> list[str]:
        return sorted(r for r, c in self.rows.items() if c.verdict in ("FAILED", "ERROR"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device": self.device,
            "created_utc": self.created_utc,
            "fingerprint": dict(self.fingerprint),
            "rows": {rid: c.as_dict() for rid, c in self.rows.items()},
            "certified_ids": self.certified_ids(),
            "kernel_candidates": emit_kernel_candidates(self),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.as_dict(), fh, indent=1, sort_keys=True)
        os.replace(tmp, path)  # atomic
        return path

    @classmethod
    def load(cls, path: str) -> "CertificationManifest":
        with open(path) as fh:
            raw = json.load(fh)
        man = cls(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            device=raw.get("device", "gpu"),
            created_utc=raw.get("created_utc", ""),
            fingerprint=dict(raw.get("fingerprint", {}) or {}),
        )
        for rid, row in raw.get("rows", {}).items():
            det = str(row.get("determinism", "0/0")).split("/")
            man.rows[rid] = RegionCertificate(
                region_id=rid, region_class=row.get("region_class", ""),
                verdict=row.get("verdict", "FAILED"),
                bit_equal=bool(row.get("bit_equal", False)),
                max_abs_delta=float(row.get("max_abs_delta", float("nan"))),
                determinism_n_pass=int(det[0]), determinism_n=int(det[1]) if len(det) > 1 else 0,
                speedup=float(row.get("speedup", 0.0)),
                uncompiled_ms=float(row.get("uncompiled_ms", 0.0)),
                compiled_ms=float(row.get("compiled_ms", 0.0)),
                input_coverage=int(row.get("input_coverage", 0)),
                input_shapes=list(row.get("input_shapes", [])),
                input_source=row.get("input_source", "synthetic"),
                reference=row.get("reference", "uncompiled_reference"),
                fma_contraction_possible=bool(row.get("fma_contraction_possible", False)),
                first_divergence=row.get("first_divergence"),
                fixed_point_routed=bool(row.get("fixed_point_routed", False)),
                reason=row.get("reason", ""),
            )
        return man


# ---------------------------------------------------------------------------
# 5. ACTIVATION API — default-OFF, byte-identical unless certified + enabled.
# ---------------------------------------------------------------------------


def safe_compile(
    fn: Callable[..., Any],
    *,
    region_id: str,
    manifest: CertificationManifest | None = None,
    enabled: bool = False,
):
    """Return ``mx.compile(fn)`` iff (``enabled`` AND ``manifest`` certifies
    ``region_id``); else return ``fn`` unchanged (byte-identical default).

    This is the single activation gate: no certificate => no compile, ever."""

    if not enabled:
        return fn
    if manifest is None or not manifest.is_certified(region_id):
        return fn
    import mlx.core as mx

    return mx.compile(fn)


def resolve_enabled_regions(
    spec: str | None,
    manifest: CertificationManifest | None,
    *,
    enforce_fingerprint: bool = True,
    host: dict[str, str] | None = None,
    run_device: str | None = None,
) -> frozenset[str]:
    """Resolve the ``--safe-compile-regions`` spec to the SET of region ids to enable.

    ``None`` / "" / "none"/"off" => empty (default OFF). "all-certified" => every
    CERTIFIED row in the manifest. Otherwise a comma-separated id list, intersected
    with the CERTIFIED rows (an id that is not certified is silently NOT enabled —
    fail-closed, never activate an uncertified region).

    PER-CHIP TRUST (v2): if the manifest carries a host fingerprint that does NOT
    match this host, the manifest is STALE and NO region activates (empty set,
    fail-closed) — a certificate is bit-identity evidence only for the chip it was
    measured on. ``enforce_fingerprint=False`` disables the check (tests / same-host
    replay where the caller has already validated freshness)."""

    if not spec or spec.strip().lower() in {"none", "off", ""}:
        return frozenset()
    if enforce_fingerprint and manifest is not None:
        ok, _reason = manifest_fingerprint_ok(manifest, host)
        if not ok:
            return frozenset()  # stale certificate -> fail closed
    # Device match: fp-contraction is device-specific (hosc: 0 on GPU, 6e-8 on CPU). A GPU run
    # MUST NOT activate a CPU-measured certificate (or vice versa) — fail closed on mismatch.
    if run_device is not None and manifest is not None and manifest.device:
        if str(manifest.device).strip().lower() != str(run_device).strip().lower():
            return frozenset()
    certified = frozenset(manifest.certified_ids()) if manifest is not None else frozenset()
    if spec.strip().lower() in {"all", "all-certified", "all_certified"}:
        return certified
    requested = frozenset(s.strip() for s in spec.split(",") if s.strip())
    return requested & certified


def emit_kernel_candidates(manifest: "CertificationManifest") -> list[dict[str, Any]]:
    """The failed-certification -> Metal-kernel-candidate pipeline (D16 feed).

    A region that FAILS bit-equality is an fp-contraction site the compiler cannot
    make safe on this chip — i.e. exactly a place where a hand-written fixed-order
    Metal kernel could recover both determinism AND the fused speedup. This turns
    every FAILED row into a ranked kernel candidate so the day a region fails
    somewhere, its replacement is auto-named (dormant while there are 0 failures).

    Rank key (descending value): a genuine fp-contraction (bit_equal False, finite
    delta) OUTRANKS a harness error; larger measured speedup potential outranks
    smaller (the fusion win a kernel would recover); FMA-eligible outranks not."""

    cands: list[dict[str, Any]] = []
    for rid, c in manifest.rows.items():
        if c.verdict not in ("FAILED", "ERROR"):
            continue
        if c.verdict == "FAILED" and not c.bit_equal and c.region_class == RegionClass.SAFE_ELEMENTWISE.value:
            kind = "fp_contraction"  # the real kernel opportunity
        elif c.verdict == "ERROR":
            kind = "harness_error"
        else:
            kind = "other_failed"  # reduction/contraction refused pre-compile: not a kernel target
        delta = c.max_abs_delta if c.max_abs_delta == c.max_abs_delta else 0.0  # NaN-guard
        speedup_potential = c.speedup if c.speedup and c.speedup > 0 else 1.0
        cands.append({
            "region_id": rid,
            "candidate_kind": kind,
            "max_abs_delta": delta,
            "speedup_potential": round(speedup_potential, 4),
            "fma_contraction_possible": c.fma_contraction_possible,
            "first_divergence": c.first_divergence,
            "rationale": (
                "fp-contraction on a compile-eligible elementwise region -> hand-write a "
                "fixed-order Metal kernel to recover determinism + fusion speedup"
                if kind == "fp_contraction" else
                ("harness error -> fix builder / inputs before kernel work"
                 if kind == "harness_error" else
                 "not a kernel target (route reduction to fixed-order primitive)")
            ),
        })
    # rank: fp_contraction first, then by speedup_potential desc, then delta desc, then FMA
    order = {"fp_contraction": 0, "harness_error": 1, "other_failed": 2}
    cands.sort(key=lambda d: (order.get(d["candidate_kind"], 3),
                              -d["speedup_potential"], -d["max_abs_delta"],
                              0 if d["fma_contraction_possible"] else 1))
    for rank, d in enumerate(cands):
        d["rank"] = rank
    return cands


def maybe_safe_compile(
    fn: Callable[..., Any],
    *,
    region_id: str,
    enabled_regions: frozenset[str],
    manifest: CertificationManifest | None = None,
):
    """Trainer-facing convenience: compile ``fn`` iff ``region_id`` is in the resolved
    ``enabled_regions`` (which are already CERTIFIED ∩ requested). Byte-identical when
    ``region_id`` not enabled."""

    if region_id not in enabled_regions:
        return fn
    return safe_compile(fn, region_id=region_id, manifest=manifest, enabled=True)


# ---------------------------------------------------------------------------
# 5b. HOT-LOOP WIRING — install compiled elementwise hooks onto the witness model.
#
# The model's ``_act`` (hosc = tanh(beta*sin(omega*u))) is a pure elementwise
# region, wired via an installable slot: ``model._compiled_act`` is None by
# default (=> the plain path runs => BYTE-IDENTICAL, which is exactly what the
# live default-OFF run does). ``install_safe_compiled_regions`` sets the slot to
# ``mx.compile(pure_hosc)`` ONLY when ``hosc_activation`` is in the resolved
# ``enabled_regions`` (which are already CERTIFIED ∩ requested ∩ fingerprint-fresh).
# Activation is therefore a FLAG-FLIP, not a code change; wiring is COMPLETE.
# ---------------------------------------------------------------------------

# The (region_id -> model attribute) map for the wired hot-loop regions. The
# film/compose tails are MIXED (their elementwise tail rides a Linear matmul) —
# reported by discovery as v3 sub-chain opportunities, NOT wired in v2.
WIRED_MODEL_REGIONS = {"hosc_activation": "_compiled_act"}


def install_safe_compiled_regions(
    model: Any,
    enabled_regions: "frozenset[str]",
    manifest: CertificationManifest | None = None,
) -> list[str]:
    """Install compiled hooks for every wired region in ``enabled_regions`` onto
    ``model``. Returns the list of region ids actually installed. Byte-identical
    no-op when ``enabled_regions`` is empty (the default-OFF live path).

    ``hosc_activation`` -> ``model._compiled_act = mx.compile(lambda u,beta,omega:
    tanh(beta*sin(omega*u)))`` (beta/omega passed as traced mx.array inputs so the
    per-epoch beta anneal never bakes a constant / forces a recompile). The model's
    ``_act`` consults ``_compiled_act`` and falls back to the plain path when None."""

    installed: list[str] = []
    if not enabled_regions:
        return installed
    if "hosc_activation" in enabled_regions:
        compiled = safe_compile(
            _pure_hosc, region_id="hosc_activation", manifest=manifest, enabled=True)
        if compiled is not _pure_hosc:  # certified + enabled -> a real compiled fn
            setattr(model, "_compiled_act", compiled)
            installed.append("hosc_activation")
    return installed


def _pure_hosc(u: Any, beta: Any, omega: Any) -> Any:
    """The certified ``hosc_activation`` region as a standalone pure fn (matches the
    ``_hosc_activation`` builder exactly). beta/omega are mx.array scalars."""

    import mlx.core as mx

    return mx.tanh(beta * mx.sin(omega * u))


# ---------------------------------------------------------------------------
# 5c. REDUCTION-SITE INVENTORY — the step's reduction sites + routing decision.
#
# Per the #348 cross-process determinism probe: native ``mx.sum`` / ``mx.mean`` /
# ``mx.logsumexp`` are ALREADY cross-process bit-identical on MLX, so the default
# routing is NATIVE (no fixed-order rewrite needed). A site routes to the #348
# fixed-order primitive ONLY if a future host's determinism certificate FAILS for
# that native reduction. This table is the machine-readable record of that audit.
# ---------------------------------------------------------------------------

STEP_REDUCTION_SITES = (
    {"site": "ce_loss", "ops": ("logsumexp", "sum", "mean"),
     "routing": "native", "reason": "#348 probe: logsumexp/sum/mean cross-process bit-identical; "
     "fixed-order fallback available via certify_fixed_point_reduction('ce_reduction')"},
    {"site": "eikonal_loss", "ops": ("mean", "square"),
     "routing": "native", "reason": "#348: mean bit-identical; elementwise square is compile-safe"},
    {"site": "length_loss", "ops": ("sum", "mean"),
     "routing": "native", "reason": "#348: sum/mean bit-identical cross-process"},
    {"site": "R_operator_reductions", "ops": ("interpolation", "mean"),
     "routing": "native_fused_r", "reason": "R roundtrip routes through the #348 --fused-r-kernel "
     "(fixed-order VJP); NOT an mx.compile target (v7 audit lever #3 EXCLUDED it)"},
)


# ---------------------------------------------------------------------------
# REAL-WITNESS CERTIFICATION COVERAGE (#252 v7 GPU bit-cert).
#
# The v2 memo flagged the toy certification coverage — (256, 96) shapes at a
# SINGLE beta=2.5 — as a robustness gap: fp-contraction is a per-VALUE knife-edge
# event, so ``n=32`` seeded inputs at one beta can PASS while unsampled inputs
# diverge (exactly what happened on CPU: hosc bit-equal at n=32/(256,96) toy
# coverage in v1, yet a clean 5.96e-8 CPU failure surfaced under the v2 device
# bugfix). This block replaces the toy coverage with the REAL witness geometry:
#
#   * shape: the 384x512 render grid (196_608 points) x hidden_dim=96 (the real
#     ``--hidden-dim`` trunk width) — the actual per-frame pre-activation shape ``_act``
#     sees. Element count faithful; dtype float32.
#   * value distribution: engineered ADVERSARIAL coverage of the fp-contraction
#     knife-edges (dense linspace across sin's steep zero-crossings + saturation
#     probes that drive beta*sin onto tanh's rails), not just a narrow normal.
#   * beta: SWEPT across the v7 anneal range [1.0, 10.0] (``--hosc-beta`` 4.0 ->
#     ``--hosc-beta-end`` up to 10) — certify across the range, not one point.
#
# Elementwise bit-identity is a per-element property, so a certificate that holds
# across this coverage is much stronger evidence than the toy one. These generators
# run ONLY during certification/discovery — never in the training hot loop (the
# trainer's ``_act`` calls the compiled fn on its own live activations).
# ---------------------------------------------------------------------------

_CERT_GRID_POINTS = 384 * 512      # 196_608 real render points ("384x512 grids")
_CERT_HIDDEN = 96                  # --hidden-dim default (real trunk width)
HOSC_BETA_ANNEAL_RANGE = (1.0, 10.0)  # the v7 hosc beta anneal span certified across
_HOSC_OMEGA = 1.0                  # --hosc-omega default


def _real_coverage_array(rng, *, points: int | None = None, channels: int = _CERT_HIDDEN,
                         knife_edge: bool = True):
    """A real-witness-shaped array with ADVERSARIAL value coverage of the
    fp-contraction knife-edges.

    Shape ``(points, channels)`` reflects the 384x512 render grid at hidden_dim.
    ``knife_edge`` overlays (col 0) a dense linspace across ``[-4pi, 4pi]`` (many
    sin zero-crossings) and (first rows) large-magnitude saturation probes so the
    certification samples exactly where fused-vs-unfused rounding is most likely
    to diverge — not merely the bulk of a narrow normal."""

    import mlx.core as mx
    import numpy as np

    n = int(points if points is not None else _CERT_GRID_POINTS)
    c = int(channels)
    a = (rng.standard_normal((n, c)) * 3.0).astype(np.float32)
    if knife_edge and n > 0:
        a[:, 0] = np.linspace(-4.0 * np.pi, 4.0 * np.pi, n, dtype=np.float32)
        k = min(64, n)
        a[:k] = (rng.standard_normal((k, c)) * 15.0).astype(np.float32)  # saturation probes
    return mx.array(a)


def _sweep_hosc_beta(seed: int) -> float:
    """Map a certification seed deterministically onto the v7 hosc beta anneal
    range ``[1.0, 10.0]``.

    The bit-equality harness sweeps seeds ``1000 .. 1000+n_inputs-1``; ``seed % 32``
    over a complete residue block (the canonical ``--n-inputs 32``) is a PERMUTATION
    of ``0..31``, so the 32-input cert tiles ``[1.0, 10.0]`` UNIFORMLY with BOTH
    endpoints hit (beta=1.0 and the step-native beta=10.0 — the hardest saturation
    case). This is stronger than a random hash: the anneal endpoints are guaranteed
    in the sweep, not left to chance."""

    lo, hi = HOSC_BETA_ANNEAL_RANGE
    frac = (int(seed) % 32) / 31.0
    return float(lo + (hi - lo) * frac)


# ---------------------------------------------------------------------------
# Canonical nominated regions from the witness hot loop (the practical allowlist).
# These mirror the real trainer's elementwise-heavy candidates; real trainer
# regions register their own builders (with real checkpoint weights) at wire-in.
# make_inputs use the REAL witness coverage above (grid shape + beta sweep).
# ---------------------------------------------------------------------------


def _register_canonical_regions() -> None:
    """Register the canonical witness hot-loop candidate regions.

    Builders are lazy (import mlx only when built) so importing this module never
    requires MLX. Each ``make_inputs(seed)`` is deterministic in ``seed``."""

    def _hosc_activation():
        import mlx.core as mx
        import numpy as np

        def fn(u, beta, omega):
            return mx.tanh(beta * mx.sin(omega * u))  # unary+mul chain (no add) => low-risk

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            u = _real_coverage_array(rng)                       # real 384x512 grid coverage
            beta = _sweep_hosc_beta(seed)                       # sweep v7 anneal range [1,10]
            return (u, mx.array(np.float32(beta)), mx.array(np.float32(_HOSC_OMEGA)))

        return fn, make_inputs

    def _sigmoid_scale():
        import mlx.core as mx
        import numpy as np

        def fn(base, tex):
            return mx.sigmoid(base + tex) * 255.0  # add then unary then mul

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            base = _real_coverage_array(rng, channels=3)        # (P, 3) RGB-logit coverage
            tex = _real_coverage_array(rng, channels=3, knife_edge=False)
            return (base, tex)

        return fn, make_inputs

    def _film_modulate():
        import mlx.core as mx
        import numpy as np

        def fn(h, scale, shift):
            return h * scale + shift  # mul-ADD => FMA-contraction-eligible (bit-equality NOT guaranteed)

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            h = _real_coverage_array(rng)                       # (P, hidden) real trunk activation
            scale = mx.array((1.0 + rng.standard_normal((_CERT_HIDDEN,)) * 0.1).astype(np.float32))
            shift = mx.array((rng.standard_normal((_CERT_HIDDEN,)) * 0.1).astype(np.float32))
            return (h, scale, shift)

        return fn, make_inputs

    def _ce_reduction():
        import mlx.core as mx
        import numpy as np

        def fn(logits, tgt_oh):
            logsum = mx.logsumexp(logits, axis=-1)  # REDUCTION => not compile-eligible
            tgt = mx.sum(logits * tgt_oh, axis=-1)
            return mx.mean(logsum - tgt)

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            logits = mx.array(rng.standard_normal((256, 5)).astype(np.float32))
            tgt = rng.integers(0, 5, size=(256,))
            tgt_oh = mx.array(np.eye(5, dtype=np.float32)[tgt])
            return (logits, tgt_oh)

        return fn, make_inputs

    def _siren_activation():
        import mlx.core as mx
        import numpy as np

        def fn(u, omega):
            return mx.sin(omega * u)  # unary+mul (no add) => low FMA risk

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            u = _real_coverage_array(rng)
            return (u, mx.array(np.float32(30.0)))

        return fn, make_inputs

    def _finer_activation():
        import mlx.core as mx
        import numpy as np

        def fn(u, omega):
            return mx.sin(omega * (mx.abs(u) + 1.0) * u)  # abs+add+mul => FMA-eligible

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            u = _real_coverage_array(rng)
            return (u, mx.array(np.float32(30.0)))

        return fn, make_inputs

    def _wire_activation():
        import mlx.core as mx
        import numpy as np

        def fn(u, omega, scale):
            return mx.sin(omega * u) * mx.exp(-0.5 * mx.square(scale * u))

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            u = _real_coverage_array(rng)
            return (u, mx.array(np.float32(20.0)), mx.array(np.float32(10.0)))

        return fn, make_inputs

    def _film_affine_tail():
        import mlx.core as mx
        import numpy as np

        def fn(x, scale, shift):
            return x * scale + shift  # elementwise tail AFTER the Linear (matmul lives outside)

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            x = _real_coverage_array(rng)
            scale = mx.array((1.0 + rng.standard_normal((_CERT_HIDDEN,)) * 0.1).astype(np.float32))
            shift = mx.array((rng.standard_normal((_CERT_HIDDEN,)) * 0.1).astype(np.float32))
            return (x, scale, shift)

        return fn, make_inputs

    def _compose_rgb_tail():
        import mlx.core as mx
        import numpy as np

        def fn(z):
            return mx.sigmoid(z) * 255.0  # elementwise tail AFTER self.out (matmul outside)

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            z = _real_coverage_array(rng, channels=3)
            return (z,)

        return fn, make_inputs

    def _achromatic_luma():
        import mlx.core as mx
        import numpy as np

        def fn(rgb):
            luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
            return mx.concatenate([luma, luma, luma], axis=-1)

        def make_inputs(seed: int) -> tuple:
            rng = np.random.default_rng(seed)
            rgb = mx.array((rng.random((_CERT_GRID_POINTS, 3)) * 255.0).astype(np.float32))
            return (rgb,)

        return fn, make_inputs

    register_region(SafeCompileRegion(
        "hosc_activation", _hosc_activation,
        frozenset({"tanh", "sin", "mul"}),
        notes="tanh(beta*sin(omega*u)) — step-native activation; unary+mul (no add) => low FMA risk",
    ))
    register_region(SafeCompileRegion(
        "sigmoid_scale", _sigmoid_scale,
        frozenset({"sigmoid", "add", "mul"}),
        notes="sigmoid(base+tex)*255 — compose_rgb tail; add+mul present (FMA-eligible)",
    ))
    register_region(SafeCompileRegion(
        "film_modulate", _film_modulate,
        frozenset({"mul", "add"}),
        notes="h*scale+shift — FiLM affine; canonical FMA-contraction candidate (expect FAIL)",
    ))
    register_region(SafeCompileRegion(
        "ce_reduction", _ce_reduction,
        frozenset({"logsumexp", "mul", "sum", "subtract", "mean"}),
        notes="CE loss elementwise+reduction — NOT compile-eligible (route sum to fixed-order)",
    ))
    register_region(SafeCompileRegion(
        "siren_activation", _siren_activation, frozenset({"sin", "mul"}),
        notes="sin(omega*u) — SIREN activation; unary+mul => low FMA risk (WIRED-eligible)",
    ))
    register_region(SafeCompileRegion(
        "finer_activation", _finer_activation, frozenset({"sin", "abs", "add", "mul"}),
        notes="sin(omega*(|u|+1)*u) — FINER activation; add+mul => FMA-eligible",
    ))
    register_region(SafeCompileRegion(
        "wire_activation", _wire_activation, frozenset({"sin", "exp", "square", "mul"}),
        notes="sin(omega*u)*exp(-.5*(s*u)^2) — WIRE real-Gabor; unary+mul => low FMA risk",
    ))
    register_region(SafeCompileRegion(
        "film_affine_tail", _film_affine_tail, frozenset({"mul", "add"}),
        notes="x*scale+shift — the elementwise TAIL of the FiLM stage (Linear matmul lives "
        "OUTSIDE); MIXED in call_batch => v3 sub-chain split, not wired in v2",
    ))
    register_region(SafeCompileRegion(
        "compose_rgb_tail", _compose_rgb_tail, frozenset({"sigmoid", "mul"}),
        notes="sigmoid(z)*255 — the elementwise TAIL of compose_rgb (self.out matmul OUTSIDE); "
        "MIXED in call_batch => v3 sub-chain split, not wired in v2",
    ))
    register_region(SafeCompileRegion(
        "achromatic_luma", _achromatic_luma, frozenset({"mul", "add", "concatenate"}),
        notes="BT.601 luma replicate (chroma-OFF control arm); mul+add => FMA-eligible",
    ))


_register_canonical_regions()


# ---------------------------------------------------------------------------
# Cross-process child entrypoint (mirrors the #348 probe's --child).
# ---------------------------------------------------------------------------


def _certify_child(region_id: str, device: str) -> None:
    import mlx.core as mx

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    region = get_region(region_id)
    fn, make_inputs = region.build()
    vc = region.classify()
    if vc.region_class is RegionClass.UNSAFE_REDUCTION:
        # fixed-order reduction determinism child (no compile)
        args = make_inputs(2024)
        out = fn(*args)  # the region's own reduction; determinism is the claim
        mx.eval(out)
    else:
        compiled = mx.compile(fn)
        args = make_inputs(2024)
        out = compiled(*args)
        mx.eval(out)
    print(json.dumps({"region_id": region_id, "device": device, "hash": _hash_outputs(out)}))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="SAFE-COMPILE certification harness (#252)")
    ap.add_argument("--certify-child", nargs=2, metavar=("REGION_ID", "DEVICE"), default=None,
                    help="internal: rebuild+compile a region in this process and print its hash")
    ap.add_argument("--certify", nargs="*", default=None,
                    help="certify region ids (default: all registered)")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--n-inputs", type=int, default=32)
    ap.add_argument("--n-determinism", type=int, default=5)
    ap.add_argument("--reps", type=int, default=40,
                    help="timing reps per region (timing is NOT a certificate — a small value keeps "
                    "the cert a good GPU citizen beside a live run; bit-equality is contention-invariant)")
    ap.add_argument("--no-cross-process", action="store_true",
                    help="skip the cross-process determinism certificate (in-process only)")
    ap.add_argument("--discover", action="store_true",
                    help="run the v2 auto-discovery sweep (AST + trace) and print the candidate "
                    "table (region, op-kinds, SAFE/UNSAFE/MIXED verdict); no certification")
    ap.add_argument("--out", default=None, help="write the manifest JSON here")
    args = ap.parse_args(argv)

    if args.certify_child:
        _certify_child(args.certify_child[0], args.certify_child[1])
        return 0

    if args.discover:
        # AST always; trace only when MLX is importable (device=cpu => no GPU contention).
        try:
            import mlx.core  # noqa: F401
            _trace = True
        except Exception:
            _trace = False
        for row in discover_candidates(trace=_trace, device="cpu"):
            print(json.dumps(row.as_dict()), flush=True)
        return 0

    ids = args.certify if args.certify else sorted(REGION_BUILDERS)
    from datetime import datetime, timezone

    man = CertificationManifest(device=args.device,
                                created_utc=datetime.now(timezone.utc).isoformat(),
                                fingerprint=host_fingerprint())
    for rid in ids:
        region = get_region(rid)
        if region.classify().region_class is RegionClass.UNSAFE_REDUCTION:
            cert = certify_fixed_point_reduction(
                rid, n_determinism=args.n_determinism, device=args.device,
                cross_process=not args.no_cross_process)
        else:
            cert = certify_region(
                rid, n_inputs=args.n_inputs, n_determinism=args.n_determinism,
                reps=args.reps, device=args.device, cross_process=not args.no_cross_process)
        man.add(cert)
        print(json.dumps(cert.as_dict()), flush=True)
    if args.out:
        man.save(args.out)
        print(f"[manifest] wrote {args.out} ({len(man.certified_ids())} certified)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
