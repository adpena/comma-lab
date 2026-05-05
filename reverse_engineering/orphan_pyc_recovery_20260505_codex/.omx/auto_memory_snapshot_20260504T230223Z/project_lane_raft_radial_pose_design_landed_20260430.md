---
name: Lane RAFT/radial pose — Level 0 → Level 1 SCAFFOLD landed
description: 2026-04-30. Phase 3 Lane 18 (RAFT/radial 6-DoF pose decomposition) advanced from Level 0 (sketch only — existing src/tac/raft_pose.py was single-DOF Lane FL only) to Level 1 (SCAFFOLD). 25/25 synthetic tests passing. Council design doc landed. Two operating modes: Mode A (compress-time prior, fully compliant, default) + Mode B (inflate-time recompute, env-gated, non-compliant pending human approval per CLAUDE.md "Strict scorer rule"). impl_complete gate satisfied.
type: project
authoritative_for: lane_raft_radial_pose_level1_scaffold
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR

Lane RAFT/radial extends `src/tac/raft_pose.py` (Lane FL — single-DOF longitudinal pose dim 0) to **full 6-DoF radial-basis decomposition** of optical flow per Longuet-Higgins 1981 + Mallat-style basis projection. Two operating modes:

**Mode A (compress-time prior, default — SAFE):** RAFT runs at compress time only. Produces a 6-DoF pose initialization that the existing pose-TTO loop refines. Pose stream still shipped (~50 KB on Lane G v3). Net effect: faster TTO convergence, slightly better optimum, ~0 byte change. **Predicted band: -0.001 to -0.003 score [prediction]**.

**Mode B (inflate-time recompute, env-gated, NON-COMPLIANT pending human approval):** RAFT runs at inflate time. Pose stream eliminated. Saves ~50 KB net (~ -0.04 score) IF disagreement vs contest pose ≤ 1e-4 average. **Predicted band: -0.000 to -0.005 [prediction]** if disagreement OK; **-0.05 to -0.20 score REGRESSION** if not. Contrarian VETO until human approval per CLAUDE.md "Strict scorer rule".

## Files landed

- `src/tac/raft_radial_pose.py` — 423 LOC; required-keyword args; Mode B refuses to instantiate without `inflate_compliance_marker` parameter; runtime banner emission required for Mode B
- `src/tac/tests/test_raft_radial_pose.py` — 25 tests; covers basis builder (canonical 6-DoF Longuet-Higgins basis), decomposition (pure translation / pure roll recover correct alpha), calibration (LSQ identity recovery + affine fit), disagreement metrics + kill threshold, Mode A end-to-end + Mode A-only enforcement, compliance banner emission
- `.omx/research/council_lane_raft_radial_pose_design_20260430.md` — full council deliberation; Hotz LEAD + Yousfi + Fridrich + Mallat + Karpathy + Quantizr + Selfcomp + **Contrarian VETO on Mode B without explicit human approval** seats

## Council verdict

**Adopt: Two-mode implementation. Mode A unconditionally shippable. Mode B implemented but env-gated and tagged `[non-compliant, requires compliance ruling]` until human approval.**

Hotz raised the dependency check: torchvision RAFT-Large is ~5MB on disk; only viable at inflate time if it's preinstalled in the contest scorer environment. Otherwise Mode B is dead on arrival.

Yousfi quantified the viability gate: Mode B works only if average per-frame RAFT-radial vs contest-pose disagreement < 1e-4. Above that, the pose-distortion regression kills the rate savings.

Mallat noted that Phase 3 should consume the residual flow (after the 6-DoF projection) via wavelet basis (Lane 11 intersection).

## Tests

```
PYTHONPATH=src .venv/bin/python -m pytest src/tac/tests/test_raft_radial_pose.py
25 passed in 0.11s
```

All tests run on synthetic flow fields with known basis coefficients — no GPU, no real RAFT inference. Real-anchor disagreement measurement is Phase C (Level 2 — out of scope for this scaffold).

## Lane registry status

```
lane_raft_radial_pose: level 0 → 1
gates: impl_complete=true; remaining 6 gates false
```

## Strict-scorer-rule discipline

Mode B's `RaftRadialPoseConfig.__post_init__` HARD-RAISES `ValueError("Mode B requires inflate_compliance_marker (human-signed approval per CLAUDE.md 'Strict scorer rule')")` if instantiated without the marker. Belt-and-suspenders: `emit_inflate_compliance_banner` raises again if the marker is missing at the banner step. Both checks land tested in Phase B.

## Phase ordering ahead

- Phase B (Level 1) ✅ THIS COMMIT
- Phase C (Level 2 prep) — empirical RAFT-vs-contest-pose disagreement measurement on Lane G v3 anchor; tag `[empirical:reports/raft_radial_disagreement.json]`
- Phase D (Level 2) — Mode A wired into compress.sh as TTO initializer
- Phase E (Mode B prep) — implement Mode B BEHIND env-gate; runtime banner; HUMAN APPROVAL gate
- Phase F (Level 3 path) — STRICT preflight check + 3-clean-pass adversarial review

## Cross-references

- CLAUDE.md "Strict scorer rule" (the binding gate on Mode B)
- CLAUDE.md "Auth eval EVERYWHERE"
- `feedback_production_hardened_standard_definition_20260430.md`
- `.omx/research/council_lane_raft_radial_pose_design_20260430.md`
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 18 RAFT/radial pose"
- `src/tac/raft_pose.py` (existing Lane FL single-DOF scaffold being extended)
- `src/tac/depth_motion.py` (sibling depth-flow lane)
- Teed & Deng 2020 RAFT (arXiv 2003.12039); Longuet-Higgins 1981; Mallat 2009
