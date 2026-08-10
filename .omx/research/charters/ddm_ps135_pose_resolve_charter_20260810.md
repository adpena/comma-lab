# ddm_ps135 — THE POSE RE-SOLVE: close the 95.1% axis with the public machinery + our engines

## Mission (operator 2026-08-10: "laser focused on frontier score lowering... micro to macro")

Our gap to the custodied bar is **95.1% POSE**: lc2 d_pose 2.332e-5 vs PR135's 6.88e-6 →
+0.006974 S. The closing mechanism is PUBLIC + GRANTED and now in custody: PR133's
quantize-then-compensate — coarsen pose-carrier atoms, then JOINTLY RE-SOLVE the int12
coefficients against exact PoseNet with full-n600 Jacobian coordinate passes. pi136's matched
control is the load-bearing receipt: CBQ WITHOUT the re-solve worsens pose ~29×; WITH it, pose
lands BELOW baseline. The compensation is everything. AND their search was TRUNCATED: both
matched 8-pass arms were STILL ACCEPTING MOVES at stop (pi136) — the caps-genus law
(`caps_genus_trajectory_stopping_20260805`) applied to THEIR vehicle: they stopped at a cap,
not at convergence.

## Two legs (A = the score-mover on OUR vehicle; B = the continuation on theirs)

**LEG A — re-solve OUR lc2 pose carrier (the −0.0070 S candidate).** Our lc2 base (archive sha
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`, 187,226 B, PR130-family
carrier, CPU-DECODABLE unlike PR135) has coefficients that were NEVER re-solved against PoseNet
— exactly PR130's condition before PR133 improved it 2.6×. Run the joint coordinate re-solve on
OUR carrier: exact CPU-torch PoseNet authority (deterministic; PIN batch shape + threads per the
batch-shape-instrument law), full n600, Jacobian-linearized passes with periodic exact refresh
(their 8-pass structure), CONTINUE PAST 8 passes until a real convergence test passes (the
#850/#935 uncap lesson — never stop at an inherited cap while still descending). Target: d_pose
2.332e-5 → ≤7e-6. Byte-neutral-or-better (int12 stays int12; optional atom bit-drops ONLY with
compensation, per the 29× control).

**LEG B — continue THEIR truncated search on the PR135 shipped coefficients.** Same machinery,
warm-started from their shipped int12 values (extract via their own runtime parse). Their
d_pose 6.88e-6 with a still-descending search ⇒ headroom below the bar's pose term. Bounded:
stop at convergence or 3 dry passes. (Frames on their base decode CUDA-only — leg B's objective
runs through the CARRIER→pose-warp→PoseNet path; if that path cannot be reproduced CPU-side
from their runtime code, record the typed blocker and deliver leg A alone. Do NOT fake a proxy.)

## Sources (all granted off the shelf; verify at source, never from memory)

- Encode-side solver: hunt `joint_pose_solve` in `/Volumes/VertigoDataTier/pact/
  pr135_intake_20260810/experiment_book/` (231 files) + pr135_src; PR133 custody at `.../pr133/`.
  If the solver code is absent, IMPLEMENT from the PR133 mechanism description + OUR fd-family
  GN/CG engine (#740, in-house off the shelf) — same math, ours already built.
- fd135's L3 coefficient-displacement map (`.omx/research/ddm_fd135_*`) = named input showing
  how far their re-solve moved coefficients from PR130's; consume when it lands, do NOT block.
- Authority: CPU-torch PoseNet fp32 = the verdict instrument; MLX = speed-only research signal
  with CPU spot-check; NEVER a score claim. n600 ONLY.

## Deliverables

1. Re-solved lc2 coefficient set + byte-closed lc2-container candidate (parse-back receipt,
   bytes ≤ 187,226) + local exact-decode advisory d_pose/d_seg n600 (our CPU chain) → MAIN
   fires the ONE Modal CUDA row (single-flight; never dispatch Modal yourself).
2. Convergence CURVE (d_pose vs pass number, both legs) — the micro-to-macro bridge artifact:
   where the marginal gain per pass dies, in S units per hour.
3. ALWAYS KEEP THE PAYLOAD: every coefficient set + candidate archive persisted to
   /Volumes/VertigoDataTier/pact/ddm_ps135_20260810/ with sha256+bytes.
4. borrowed_substrate_accounting (NO-FAKE #7 honesty-half): mechanism = PR133/codexblack;
   engine implementation ours-or-theirs as actually used; base = our lc2 (itself PR130-lineage).
5. Durable memo `.omx/research/ddm_ps135_pose_resolve_20260810.md`, serializer commit
   (post-edit --expected-content-sha256, tags [no-triality] [p0-ledger-ok]). Checkpoint per protocol.

## OPTIMAL FORM

Reference: PR133's realized form (8 exact full-600 Jacobian coordinate passes, int12 domain,
joint vs PoseNet) — our run must MATCH-or-EXCEED that form (uncapped continuation = exceed;
lattice-native int12 moves per the true-domain triple #974). SCOPE reduction allowed: leg B
deliverable may honestly degrade to a typed blocker; MECHANISM reduction FORBIDDEN (no
surrogate-only objective, no subset pairs, no skipping the exact-refresh passes). Pins: lc2 sha
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`; PR135 sha
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`; pi136 memo
`.omx/research/ddm_pi136_leaderboard_breadth_intake_20260810.md` (commit b591fb1a4e).
PRIOR-LAW PREDICTION (from PR130→PR133 realized 2.6× pose improvement on the SAME carrier
family): leg A lands d_pose in [6e-6, 1.1e-5] from our 2.332e-5, i.e. ΔS −0.0047 to −0.0070;
if leg A moves pose by <20% the mechanism transfer failed and the residual is OUR-carrier-
specific — decompose per-pair before any family verdict (one-defect law).

## Falsifier

Leg A converged (3 dry passes at uncapped budget) with d_pose > 1.8e-5 → the re-solve mechanism
does NOT transfer to our carrier at this formulation; record per-pair decomposition + the exact
divergence from PR133's setup, INSTRUMENT-scoped (m94), never a family kill.
