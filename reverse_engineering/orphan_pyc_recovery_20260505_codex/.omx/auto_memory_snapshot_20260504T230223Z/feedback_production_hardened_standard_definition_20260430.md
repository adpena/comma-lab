---
name: USER STANDARD — Full Production Hardened + Recursive Adversarial Reviewed (NON-NEGOTIABLE)
description: 2026-04-30. User explicitly raised the bar: "full production hardened recursive adversarial reviewed is our non-negotiable standard for all". Defines exactly what "sketch" / "scaffold" / "scaffolded" / "full production hardened" mean and the gap closure required for every Phase 1/1.5/2/3 lane. Sets the discipline going forward: NO lane ships in scaffold or sketch state.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The 4 maturity levels (rigorously defined going forward)

### Level 0 — SKETCH
- Design idea documented in memory ONLY (e.g. project_phases_2_3_4_design...)
- ZERO code, ZERO tests
- Examples in current state: Phase 3 lanes "Bit-level archive optimization", "MDL/Bayesian", "RAFT/radial pose", "Decoder systems rewrite", "Multi-pass compress Phase 3 extension"

### Level 1 — SCAFFOLD
- Minimal Python module exists (`src/tac/<lane>.py`)
- Basic unit tests pass on SYNTHETIC data only (e.g. tests/test_lane_X.py with 5-15 tests)
- NO integration into production paths (archive build, inflate_renderer.py, remote_lane scripts)
- NO real-archive empirical measurement
- Examples in current state: Lane 12 NeRV, Lane 17 IMP, Lane 19 logit-margin, Lane 20 Ballé, Joint-ADMM coordinator (V1 wraps), Lane 8 multi-pass MVP

### Level 2 — INTEGRATION
- Wired into actual archive build path (compress_archive.py / build_submission_archive)
- Wired into inflate path (inflate_renderer.py with new magic-byte handler if applicable)
- Wired into remote_lane_*.sh dispatch script
- Real-archive empirical measurement on Lane G v3 anchor (or equivalent baseline) — tagged [empirical:<test path>]
- STRICT preflight check that covers the bug class (or warn-only with promotion path)
- Examples in current state (post-tonight): Lane Ω-W-V2 (40.98% empirical + #272 inflate handler firing), Lane PD-V2 (18.6% empirical, #277 bolt-on integration in flight), Lane LCT (#277 in flight), Lane J-NWC (end-to-end inflate dispatch + producer + Check 89-warn registered)

### Level 3 — FULL PRODUCTION HARDENED + RECURSIVE ADVERSARIAL REVIEWED ⭐ USER STANDARD
- All Level 2 PLUS:
- Contest-CUDA validation: actual `[contest-CUDA]` score on Vast.ai 4090 / Modal A100 / equivalent — NOT just `[empirical]` byte counts or `[contest-CPU advisory]`
- 3-clean-pass adversarial review (per CLAUDE.md "Recursive adversarial review protocol") — counter at 3/3 with rotating perspectives
- All STRICT preflight checks pass NATURALLY (no PREFLIGHT_HOOK_ENABLED=0 workarounds)
- Memory entry documenting empirical result + cross-refs to council reports
- Deploy script + watchdog + heartbeat + harvest path documented
- NO outstanding Round-N council findings
- Examples in current state: Lane G v3 = 1.05 [contest-CUDA] is the ONLY currently Level 3 lane

## Per-lane current maturity audit (2026-04-30 ~02:30 CDT)

### Phase 1 (8 lanes)
| # | Lane | Current Level | Gap to Level 3 |
|---|---|---|---|
| 1 | SC++/q_faithful | 1.5 (firing now) | post-result harvest + archive integration + 3-clean review |
| 2 | Pose-delta + PD-V2 | 2 (#277 in flight) | contest-CUDA validation in stacked archive + 3-clean review |
| 3 | STC clean-source | 1.5 (firing now) | post-result harvest + archive integration + 3-clean review |
| 4 | Ω-W water-fill V2 | 2 (#272 in flight) | contest-CUDA result lands in next ~30min |
| 5 | LCT | 2 (#277 integrating) | contest-CUDA validation in stacked archive + 3-clean review |
| 6 | GP rerun | KILLED per Council #271 | gap: PROPER REPLACEMENT needed (B-spline / DCT / non-polynomial fit) at user's standard |
| 7 | PSD | 1 (script + watchdog landed but never dispatched) | dispatch-or-kill-with-rationale decision needed |
| 8 | Multi-pass inflate | 1 (MVP control flow, GPU inner-step deferred) | GPU inner-step implementation + integration + 3-clean review |

### Phase 1.5 (Council E stacking architecture)
| Lane | Current Level | Gap to Level 3 |
|---|---|---|
| Joint-ADMM | 1 (28 tests synthetic) + Round 11 Nesterov fix in flight | real-archive integration + Lane 10 V2 dispatch + 3-clean review |
| Multi-pass inflate | 1 | same as Phase 1 Lane 8 |
| J-NWC | 2 (end-to-end inflate dispatch wired) | shared-corpus codec build + amortization measurement + contest-CUDA validation |

### Phase 2 (5 ACCELERATE lanes per Council E)
| Lane | Current Level | Gap to Level 3 |
|---|---|---|
| 10 — ADMM real-codec wrap | 1 | Round 11 #276 fix → real-archive dispatch → contest-CUDA → 3-clean review |
| 12 — NeRV mask codec | 1 | CUDA train pass on 1200-frame mask + byte measurement vs AV1 421KB + integration into archive build → contest-CUDA → 3-clean review |
| 17 — IMP 10-cycle | 1 | CUDA full 10-cycle on Lane G v3 + integration → contest-CUDA → 3-clean review |
| 19 — SegNet logit-margin | 1 | A/B vs standard CE on Lane G v3 + integration → contest-CUDA → 3-clean review |
| 20 — Ballé hyperprior | 1 | train ScalePriorMLP on Lane G v3 qint stream + amortization measurement + integration → contest-CUDA → 3-clean review |

### Phase 3 (Council E sketches)
| Lane | Current Level | Gap to Level 3 |
|---|---|---|
| Sensitivity-map module | 0 → 1 (#275 in flight) | Level 0 lift to Level 1 in flight; then needs CUDA dispatch + integration |
| Bit-level archive optimization | 0 (sketch only) | full Level 0→3 cycle |
| MDL/Bayesian (MacKay) | 0 (sketch only) | full Level 0→3 cycle |
| RAFT/radial pose | 0 (sketch only) | full Level 0→3 cycle |
| Decoder systems rewrite | 0 (sketch only, PAUSED per Council E) | DEFERRED — paused state acceptable per Council E |
| RL/PufferLib bandit | 0 (sketch only, Council #271 PILOT-WITH-CAP $5) | full Level 0→3 cycle if pilot proves out |

## What "Full Production Hardened" actually requires per lane

For each lane to graduate from Level 1/2 to Level 3:
1. **Implementation completion** — production code, no `NotImplementedError` placeholders, all CLI flags wired, all error paths covered
2. **Real-archive empirical measurement** — actual bytes saved or score delta on Lane G v3 anchor (or equivalent baseline)
3. **Contest-CUDA validation** — `[contest-CUDA]` score from Vast.ai 4090 or Modal A100 (NOT [contest-CPU advisory], NEVER MPS)
4. **STRICT preflight check** covering the lane's bug class (or warn-only with documented STRICT-promotion path)
5. **3-clean-pass adversarial review** with rotating perspectives per CLAUDE.md (Round N+ counter at 3/3 with no new bugs found)
6. **Memory entry** documenting empirical result + dispatch metadata + cross-refs
7. **Deployment runbook** — remote_lane script + heartbeat + watchdog + harvest path

## How many Level-3 lanes do we have right now?

**ONE**: Lane G v3 = 1.05 [contest-CUDA].

That's it. Everything else is Level 0/1/2.

## What needs to happen tonight + tomorrow

To advance lanes from Level 1/2 → Level 3, we need:
- **Tonight's 6 GPU dispatches** to land successfully + harvest results = potential graduations:
  - Lane Ω-W-V2 (Level 2 → Level 3 if [contest-CUDA] result lands) — ~30min
  - Lane SC++ retry (Level 1 → Level 2 or Level 3 depending on result)
  - Lane SA-v2 retry (same)
  - Lane SO retry (same)
  - Lane HM-S (band-calibration anchor)
  - Lane STC clean-source
- **Phase 2 ACCELERATE lanes** (10/12/17/19/20) need real-archive integration work (Phase 2 swarm)
- **Phase 3 sketches** need at least Level 0 → Level 1 lifts
- **3-clean-pass adversarial review** must run after each landing to confirm Level 3 status

## Cross-refs

- CLAUDE.md "Recursive adversarial review protocol" (the 3-clean-pass discipline)
- CLAUDE.md "Council conduct" (mathematical/empirical arguments only, no conservative bias)
- project_session_state_checkpoint_20260430.md (today's snapshot)
- Council E reprioritization (.omx/research/council_grand_battleplan_round5_20260429.md)
- Council #271 strategic design (.omx/research/council_strategic_design_decisions_20260430.md)
- council_chain_integrity_audit_20260430.md (Ω-W-V2 stack alternative + 33% prior hit-rate)
