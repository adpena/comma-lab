# Deterministic optimizer directive: RESTORE 3 disfavored frameworks for curiosity-driven evaluation
# Date: 2026-05-18
# Audience: in-flight subagent acb41f8d3f7f0a3ea
# SUPPLEMENTS prior directives (commits 1694726b4 + 691393849)

## Operator directive (verbatim 2026-05-18)

> *"restore the three frameworks because i'm curious how they will do"*

The prior design-constraint directive filtered the 12-framework list down to a TIER 1+2+3 set (8 kept) + 2 explicit DISFAVORED + 4 implicit demotions. Operator wants 3 RESTORED for curiosity-driven evaluation. NOT to elevate them to the canonical-hybrid recommendation; just to give them fair empirical analysis in the deliverable.

## 3 RESTORED frameworks

### RESTORED #1: Algebraic geometry / Gröbner basis (was: explicit DISFAVORED for performance)

**Why curiosity-worthy**: Solves KKT polynomial system EXHAUSTIVELY → enumerates ALL stationary points (not just local minima). For the contest scorer's specific structure, the polynomial system may have ELEGANT decomposition into irreducible components that reveals GLOBAL optima the canonical hybrid would miss.

**Evaluation questions**:
- What's the KKT polynomial system for `score = 100·d_seg + sqrt(10·d_pose) + 25·rate`?
- Does the system admit a tractable Gröbner basis decomposition under any term order (lex / grlex / grevlex)?
- For the 178417-byte fec6 archive, how many monomials would the Buchberger algorithm encounter?
- Could a TROPICAL Gröbner basis variant (max-plus polynomial extensions) handle the argmax-piecewise-constant d_seg structure tractably?
- Hybrid: use Gröbner for SMALL subproblems (per-pair / per-tensor) + canonical hybrid for full archive?

**Honest assessment**: likely fails performance constraint on full archive scale, but may reveal global-optimum structure on per-pair / per-tensor subproblems that informs the canonical hybrid's initialization.

### RESTORED #2: Game-theoretic minimax / Nash equilibrium (was: explicit DISFAVORED for elegance)

**Why curiosity-worthy**: Frames optimization as 2-player game (codec player minimizes; scorer player maximizes within frozen weights). Nash equilibrium IS the optimal codec by definition. Particularly relevant for ADVERSARIAL-CODEC TRAINING — pre-compute archives robust to any scorer-side adversarial perturbation.

**Evaluation questions**:
- Is the scorer-vs-codec game finitely-representable as a matrix game (polynomial-time Nash via LP)?
- If continuous: does the game have a UNIQUE Nash equilibrium? (Required for deterministic solver.)
- Does any existing public-PR archive correspond to a (provable / empirical) Nash equilibrium?
- Could mixed strategies over CODEC CONFIGURATIONS (not weights) yield robust archives?
- Relationship to ADVERSARIAL-INPUT-ROBUST optimization (Madry-style L∞ ball)?

**Honest assessment**: overcomplicated for the static scorer (the scorer doesn't actually play; it's a fixed function). But: (a) game-theoretic framing inherits ROBUSTNESS GUARANTEES the canonical hybrid lacks; (b) if we ever build adversarial-codec, this becomes the canonical framework. Worth scoping.

### RESTORED #3: Submodular optimization with Lovász extension (was: implicit demotion)

**Why curiosity-worthy**: IF archive_bytes-vs-score has SUBMODULAR structure (diminishing returns as more bytes added), Lovász extension gives a CONVEX continuous relaxation solvable in POLYNOMIAL TIME via matroid intersection. This would be the STRONGEST theoretical guarantee in the entire framework lineup — global optimality with provable complexity.

**Evaluation questions**:
- Is `score(archive_bytes)` submodular as a set function on byte-subsets?
- Empirically: does diminishing-returns hold across the 178417 bytes of fec6 archive? (Per-byte sensitivity should fall faster than linear in cumulative-bytes if submodular.)
- If not submodular globally, is there a partition of bytes (per-section / per-pair / per-class) where submodularity holds locally?
- Lovász extension specifically: convex envelope of submodular f, with subgradient `∇L_f(x)_i = f(perm_x(1..i)) - f(perm_x(1..i-1))` — does this match per-byte master gradient?
- Connection to MATROID structure on the archive byte set?

**Honest assessment**: BIGGEST POTENTIAL payoff (polynomial-time global optimality is theoretically definitive) but BIGGEST PRECONDITION RISK (requires empirical submodularity verification, which we haven't done). Worth a $0 in-context empirical check on the existing master-gradient fp64 data.

## Evaluation deliverable

For each of the 3 restored frameworks, the subagent should produce a section in DELIVERABLE 1 with:

| Field | Content |
|---|---|
| Mathematical structure match | How well does the framework match the contest scorer's specific structure? |
| Performance estimate | Best-case / worst-case wall-clock on M5 Max |
| Implementation complexity | LOC estimate for canonical-helper integration |
| Hybrid composability | Could this framework be a sub-solver in the canonical hybrid? |
| Empirical preconditions | What needs to be empirically verified (e.g., submodularity check)? |
| Verdict | KEEP IN HYBRID / KEEP AS STANDALONE / EXPERIMENTAL / DEFER |

## INSTRUCTION TO acb41f8d3f7f0a3ea

INCORPORATE the 3 restored frameworks into DELIVERABLE 1's framework comparative analysis. Treat them with the same empirical-evaluation discipline as the TIER 1-3 frameworks. Operator wants curiosity-driven evaluation; honest verdicts welcomed (including "implementation infeasible at this scale" if that's the empirical answer).

The CRITICAL ADDITION: an empirical SUBMODULARITY CHECK on the existing master-gradient fp64 data — does score(archive_bytes) exhibit diminishing returns? If YES, submodular framework MAY be the global-optimality answer.

Acknowledge this directive in your next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated restore_three_disfavored_frameworks_directive_20260518 (Groebner + game-theoretic + submodular Lovász)"`.

— Main-Claude (relayed on behalf of operator 2026-05-18)
