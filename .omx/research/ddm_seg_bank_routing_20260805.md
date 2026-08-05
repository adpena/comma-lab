# SEG-BANK ROUTING (2026-08-05, operator-corrected) — seg-solve → CONSTRAIN → JOINT descent (pose trained down)

Operator verbatim: *"Because of the order of operations, I thought solving seg first was
most important and then properly constraining and then doing joint descent such that pose
is trained down. But it looks like you just ditched pretty much all of the seg stuff you
tested overnight."* CONFIRMED — disposition + re-routing below. Supersedes the
"elimination proof" framing entirely (audit F4/F5 partially corrected it; this completes it).

## The pipeline (the doctrine, m09/#383, restated as stages)
1. **SEG-SOLVE/TRAIN** — drive d_seg down by any admissible mechanism (burn, head-solve,
   phase field, correction stacks). Pose erosion HERE is expected and is NOT a verdict.
2. **CONSTRAIN** — hold the seg gains: Q3 pose-null projection where it holds (soft — sq1
   08-03: pose-neutral ~4% float, breaks at pixel-granular integer actuators; so guard
   terms + trust regions, not exact projection alone) · lg1 lane-guard/protection masks ·
   j11 seg-null/pose-null split · seg-hold loss on the conditioned trunk.
3. **JOINT DESCENT** — pose TRAINED down on the constrained seg-solved base (#383
   conditioning-gate pattern: pose descent engages once the trunk is conditioned). This is
   the ONLY family measured to cross the photometric wall (m09 clarification: post-hoc/
   stored pose is dead on frames never shaped for pose; joint-descent VALUES in
   sidecar-shaped bytes is the live mechanism). The terminal 6-eq analytic solve is the
   cheap FALLBACK/finisher (sl1 is measuring how far it gets on the corrected bases), not
   the primary.

## The seg bank (stage-1 outputs, NONE dead except grammar-targets)
- sq2 uncapped head-solve endpoint: η 0.9113, net −0.11679 S (n32, 50-step cw1 point) —
  THE largest measured seg mover. Status: BANKED, awaiting stages 2-3.
- et1 regional phase field: reach 41.84% @ 46,247 B, realized η 0.4875 (2.86× bar),
  seg-LIVE. Status: BANKED, awaiting stages 2-3 (its "pose-BLOCKED" = unconstrained).
- et1 block16: seg gain real; R8 measured pre-constraint. BANKED.
- #897 v19 joint-priced correction stack (88→3, −0.0144): already joint-priced; composes.
- rz1/lc1 grammar labels: dead as TARGETS (seg-axis harm, all 32 pairs) — conditioning-only.
- TP1/QA24 burn line: the live stage-1 trunk (burning).

## The MISSING BUILD (why every pose read was post-hoc)
TR1 trains seg-only (`compute_pose=False`); tk1's cheapdct4 hook is ACCOUNTING-only
("full in-loop consumption = owed design decision"). The #383 joint pose-finish window
does NOT exist on this vehicle. Every R8 number on a corrected base was therefore
necessarily post-hoc — the measured-dead family for pose. The kills were artifacts of the
missing stage-3 machinery. BUILD = ddm_jd1 (queued rank-1): in-loop pose loss +
#383-pattern conditioning gate + stage-2 constrain hooks on the TR1 trainer; sealed
ticket; fires a joint-finish window from the seg bank at the TP1 boundary.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
