# SPDX-License-Identifier: MIT
"""spec_v10_structured_init_defaults — seeded statics as the v10 compile-path DEFAULT (P1).

SPEC (canonical): ``.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md`` §2 P1.
Charter (naming SoT): ``.omx/research/vehicle_naming_resolution_v10_capstone_20260717.md`` —
*"cold start on a fully seeded Kolmogorov optimal program."*

THE POINT (SPEC P1 / Train-least doctrine).  v10 spends NO epoch learning what a mask already
states.  hood (MyCar) and sky/horizon (Undrivable) are BORN from the measured GT masks; lane is
BORN from openpilot polynomial priors + per-dash anchors; the hood texture is BORN from the
measured static hood-tex seed.  In v9-and-earlier these were OPT-IN levers (default-off flags,
the orphaned-signal failure mode).  In v10 they are the DEFAULT of the compile path — this module
holds that default AS SPEC-MODULE CODE.

CONTAINMENT + FAIL-CLOSED (SPEC §8, unchanged).  This module is $0/pure: it declares the default
composition and probes (read-only) whether each default's machinery + seed artifact is present.
It NEVER launches and NEVER emits a launchable argv.  It COMPOSES with
``spec_v10_capstone.spec_v10_status`` at the boundary fold via one call
(``structured_init_blockers``) — it does NOT edit that module (clean merge), and it does NOT
change the fail-closed compile: a missing machinery import or required seed artifact becomes a
typed blocker, exactly as the spec skeleton's post-merge/gate probes do.

Value-provenance: every constant is a ``(value, provenance, cite)`` triple (MEASURED / DERIVED /
ANCHOR), never bare (SPEC §9; constants-are-poison).  Class indices are NEVER hardcoded — each
default names the class-SELF-DETECTING entry point (comma10k canonical order is NOT the luma
sort; re-deriving it bit us 3x — CLAUDE.md scorer section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StructuredInitDefault:
    """One seeded-static that is a v10 birth DEFAULT (not an opt-in flag).

    ``machinery_module``   : the in-tree class-SELF-DETECTING component (import-probed).
    ``self_detect_entry``  : the callable name that detects the class from the data (NEVER a
                             hardcoded class index).
    ``seed_artifact``      : the COUNTED seed the init reads (None ⇒ born from GT masks at
                             fold time, no shipped artifact); required ⇒ a fail-closed blocker
                             when absent.
    ``default_on``         : True ⇒ ON at v10 birth (the whole point of P1).
    ``provenance``/``cite``: value-provenance triple for the measured basis.
    """

    key: str
    human: str
    machinery_module: str
    self_detect_entry: str
    seed_artifact: str | None
    seed_required: bool
    default_on: bool
    provenance: str
    cite: str


# ── The v10 structured-init DEFAULT set (SPEC P1; each realized with EXISTING machinery). ──
_DEFAULTS: tuple[StructuredInitDefault, ...] = (
    StructuredInitDefault(
        key="hood_static",
        human="hood (MyCar) born static from the measured GT masks (single canonical mask)",
        machinery_module="tac.boundary_math.hood_static_component",
        self_detect_entry="identify_static_hood_class",
        seed_artifact=None,  # born from GT masks at fold time; no shipped artifact
        seed_required=False,
        default_on=True,
        provenance="MEASURED",
        cite="hood static IoU 0.993/0.994 (naming SoT P1; CLAUDE.md scorer section #139 static core)",
    ),
    StructuredInitDefault(
        key="sky_undrivable_static",
        human="sky/horizon (Undrivable) born static from the measured GT masks (top region)",
        machinery_module="tac.boundary_math.hood_static_component",
        self_detect_entry="identify_static_hood_class",  # same detector distinguishes top vs bottom static
        seed_artifact=None,
        seed_required=False,
        default_on=True,
        provenance="MEASURED",
        cite="Undrivable top-region static IoU 0.976 (naming SoT P1; hood_static_component docstring rows [0,218])",
    ),
    StructuredInitDefault(
        key="lane_polynomial",
        human="lane (class-1) born from openpilot polynomial priors + rasterized SDF band",
        machinery_module="tac.boundary_math.lane_sdf_component",
        self_detect_entry="build_structured_lane_sdf",
        seed_artifact=None,  # polynomial+homography rasterizer is FREE (rule 118); fit at fold time
        seed_required=False,
        default_on=True,
        provenance="DERIVED",
        cite="openpilot road-frame IPM + deg-3 lane curve (lane_sdf_component; CLAUDE.md rate-lever table)",
    ),
    StructuredInitDefault(
        key="per_dash_anchors",
        human="lane dashes born from per-dash anchors (comb REFUTED; anchors are the sufficient stat)",
        machinery_module="tac.boundary_math.lane_sdf_component",
        self_detect_entry="fit_lane_line",  # dash fit lives on the fitted lane line
        seed_artifact=None,  # anchors derived from GT / v9c2 terminal at fold time (COUNTED at byte-close)
        seed_required=False,
        default_on=True,
        provenance="MEASURED",
        cite="spacing CV 0.41-0.78 ⇒ per-dash anchors (hard_frame_mechanism_atlas_20260716; ~0.9-1.8KB DERIVED)",
    ),
    StructuredInitDefault(
        key="hood_tex_seed",
        human="hood texture born from the measured static hood-tex seed (1,759 counted bytes)",
        machinery_module="tac.boundary_math.hood_static_component",
        self_detect_entry="build_static_hood_sdf",
        seed_artifact="experiments/results/necessity_dseg_calibration_20260715/hood_tex_seed.npz",
        seed_required=True,  # the ONE decisive static texture buy (SPEC row 3); a real counted artifact
        default_on=True,
        provenance="MEASURED",
        cite="d_seg 0.04538->0.01328 at eps=0, min-S 1.613 (necessity_dseg_calibration_20260715.md)",
    ),
)


def structured_init_defaults() -> tuple[StructuredInitDefault, ...]:
    """The canonical v10 structured-init DEFAULT set (SPEC P1).  All ``default_on`` by design —
    seeded statics are birth defaults, not opt-in flags."""
    return _DEFAULTS


def _probe_import(modname: str) -> tuple[bool, str]:
    import importlib.util
    try:
        ok = importlib.util.find_spec(modname) is not None
    except (ImportError, ValueError):
        ok = False
    return ok, ("" if ok else f"machinery module {modname!r} not importable")


def _probe_self_detect(modname: str, entry: str) -> tuple[bool, str]:
    """Import-probe that the class-SELF-DETECTING entry point actually exists (no hardcoded
    class index is ever emitted — the detector is the contract)."""
    import importlib
    try:
        mod = importlib.import_module(modname)
    except Exception as exc:  # pragma: no cover - env import failure
        return False, f"cannot import {modname!r}: {exc!r}"
    if not hasattr(mod, entry):
        return False, f"self-detect entry {modname}.{entry} absent (class index would be hardcoded)"
    return True, ""


@dataclass
class StructuredInitStatus:
    """Read-only readiness surface for the v10 structured-init defaults ($0)."""

    machinery_present: dict[str, bool] = field(default_factory=dict)
    self_detect_present: dict[str, bool] = field(default_factory=dict)
    seed_present: dict[str, bool] = field(default_factory=dict)
    blockers: list[dict[str, str]] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return not self.blockers


def structured_init_status(repo_root: str | Path = ".") -> StructuredInitStatus:
    """Probe (read-only, $0) whether every v10 structured-init default is realizable: its
    class-self-detecting machinery imports AND its required seed artifact exists.  Missing pieces
    become typed blockers (fail-closed, exactly like the spec skeleton's gate probes)."""
    root = Path(repo_root)
    st = StructuredInitStatus()
    for d in _DEFAULTS:
        mod_ok, mod_why = _probe_import(d.machinery_module)
        st.machinery_present[d.key] = mod_ok
        if mod_ok:
            sd_ok, sd_why = _probe_self_detect(d.machinery_module, d.self_detect_entry)
        else:
            sd_ok, sd_why = False, mod_why
        st.self_detect_present[d.key] = sd_ok
        if not sd_ok:
            st.blockers.append({
                "id": f"structured_init_machinery:{d.key}",
                "detail": f"{d.human}: {sd_why}",
            })
        if d.seed_artifact is not None:
            present = (root / d.seed_artifact).exists()
            st.seed_present[d.key] = present
            if d.seed_required and not present:
                st.blockers.append({
                    "id": f"structured_init_seed:{d.key}",
                    "detail": (f"{d.human}: required seed {d.seed_artifact} absent "
                               f"(MEASURED {d.cite}); confirmed at harvest/fold."),
                })
    return st


def structured_init_blockers(repo_root: str | Path = ".") -> list[dict[str, str]]:
    """The one-call fold surface (SPEC §8 boundary fold): the blockers the v10 compile path adds
    so seeded statics are the DEFAULT.  Intended composition (a single line at fold time)::

        # in spec_v10_capstone.spec_v10_status(...), after the gate/seed probes:
        from tac.witness_dsl.spec_v10_structured_init_defaults import structured_init_blockers
        report.blockers.extend(structured_init_blockers(repo_root))

    This keeps ``compile_v10_capstone_launch_config`` fail-closed and never edits that module on
    this branch (clean boundary merge)."""
    return structured_init_status(repo_root).blockers


__all__ = [
    "StructuredInitDefault",
    "StructuredInitStatus",
    "structured_init_defaults",
    "structured_init_status",
    "structured_init_blockers",
]
