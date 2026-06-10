# STANDING GOAL — v3 (2026-06-10) — AGGRESSIVE · BOLD · VISIONARY · ORIGINAL

The active goal. Supersedes v1/v2 (fuses v2's threshold discipline + the innovation mandate). Read at
every session wake alongside the dashboard. This is the operating law; nothing in it is aspirational.

## THE VISION
We are not tuning a video codec to shave bytes off a leaderboard. We are building — and being the
first to demonstrate — a **proof-carrying evaluator-equivalent program compiler**: a system that,
given a frozen oracle (`upstream/evaluate.py`) and a frozen input, synthesizes the SHORTEST executable
witness whose output lands in the oracle's equivalence cell. The comma contest is instance one. The
whole field is stuck at 0.19-0.20 because everyone is solving a HARDER problem than the score asks
(compressing pixels) instead of the actual problem (landing in the SegNet-argmax × PoseNet-tube cell
at minimum description length). Our edge is that we have MEASURED the oracle's geometry — the
80.67%-invisible pixel subspace, the argmax-cell boundary (cone/flip-map), the per-axis sensitivity
atlas — and no competitor has. The win we are after is not −0.00003; it is a **class shift** that makes
the leaderboard cluster obsolete and is unquestionably original.

## THE THRESHOLD LADDER (the goal is UNSATISFIED until crossed — letter AND spirit)
  T_hold  = current pointer (`.omx/state/canonical_frontier_pointer.json`; never hardcode)
  T_1     = sub-0.19   REQUIRED near-gate
  T_2     = sub-0.17   the real target
  T_3     = sub-0.15   the operator's bold stretch — pursue it as the default aim, not a maybe
  T_floor = the information-theoretic floor (derive it; the lower-bound ledger's best bound)
Holding a frontier is NEVER the result. A unit with S unchanged and above the next threshold is
INCOMPLETE unless it (a) lands an exact-eval row that lowers S, (b) PROVES (committed derivation) the
lever cannot lower S and immediately starts the next-ranked lever, or (c) hits a real wall
(budget-exhausted / measurement-running) and ends by LAUNCHING that measurement. Aim at T_3 by
default; settle for less only when the floor ledger proves T_3 unreachable on the current lever.

## THE INNOVATION GATE (binding on every SUBMISSION; operator 2026-06-10)
A contest submission must be GENUINELY ORIGINAL/INNOVATIVE **and** meaningfully below frontier — never
a noise-margin absorb-recode of a competitor's method (the recoded-R3 hold fails this: −2.6e-5, within
contest reporting precision, built from PR#112's codec). The "competitive OR innovative" statement
must be UNQUESTIONABLE: a class-shift carrier, a novel coding/training method, or original analysis
that no leaderboard entry has. Defensive holds are BANKED for readiness; they are not what we ship.
The offensive levers (ranked, `innovation_mandate_and_original_directions_20260610.md`):
  A. Evaluator-equivalence quotient compiler (the V6 thesis realized — new problem formulation)
  B. Score-native decomposition carrier (store the measured quantities, not pixels — the clean shift)
  C. Fresh-init score-aware NAS/training, null-space-primary objective (the long-training class shift;
     NOT continuation of the memorized point — that degrades, proven)
  D. Inverse-steganalysis-native coding (STC/UNIWARD at the argmax-cell boundary — the contest's own
     theory, principled past the 1.525 B/flip naive sidecar floor)
  E. Generative/implicit micro-prior carrier (novel, higher-risk)
  F. Information-theoretic floor derivation (original analysis; sets T_floor, proves headroom)
  G. Engineered deterministic corrections — ZERO-byte distortion lever (inflate.py decode-time field
     DERIVED from flip-map/atlas/cone, no scorer load, 0 archive bytes; PR95-L28 proves it; the cleanest
     constant-byte move toward sub-0.15; reuse engineered_corrections{,_v2,_readiness}.py — must be a
     real solve, NOT a per-pixel search dressed as a correction, per NO-FAKE class 6)
  H. Super-cheap small postfilters — low-byte distortion lever (≤few-KB learned residual at inflate,
     score-aware eval_roundtrip+diff-YUV6, MUST pay rent; reuse train_postfilter_on_renderer.py +
     modal_hdm8_postfilter_sweep.py; stack AFTER G)
Continuously research + propose directions NOT YET conceived; the menu is open, not closed.

## AUTONOMY CONTRACT — DECIDE, DON'T DEFER
At any fork the EVIDENCE decides, ACT and report; do NOT pause for operator input. Operator input is
reserved for EXACTLY three things: (1) the final contest-PR submission click; (2) spend beyond the
standing budget; (3) a genuine values/strategy ambiguity the evidence cannot resolve. A message ending
in "want me to…?" about a decidable action is a GOAL VIOLATION — launch it. Saturate 3 (burst 4)
bounded subagents on distinct levers; never idle a slot while S > T_1.

## THE ORACLE + AUTHORITY (binding)
A result exists only as a typed row {archive_sha256, d_seg, d_pose, bytes, score, authority_tier,
metric_family, surface}. Authority: [contest-CPU]/[contest-CUDA] exact 600-sample evaluate.py on 1:1
hardware > local CPU-torch exact-scorer advisory > MLX research-signal > telemetry proxy. Rank/admit
ONLY from the top rung (MPS corrupts 95.5% of orderings; foreign-host FP reverses 1e-5 orderings;
wrong GT decode = 100× phantom — GT only via frame_utils.yuv420_to_rgb). PSNR never a verdict. Compose
in DISTORTION domain. Lossless recode is axis-invariant (prove parity). Recompute score from
components (the rounded field lies). Every unit updates the two-sided scoreboard: UPPER (best exact S
vs next threshold) + LOWER (info-theoretic floor).

## MATHEMATICAL POSTURE + CRUX DISCIPLINE
Everything derived/measured/learned, never convention; constants carry provenance
{DERIVED|MEASURED|LEARNED|ARBITRARY}. Exploit the measured evaluator geometry. Hunt the CRUX and
recurse (cruxes nest + interact; architecture × objective dual: J_scorer·J_renderer ≠ 0 in rewarded
directions). Ask WHY to mechanism; seek patterns, then patterns-of-patterns. A negative that localizes
a crux + redirects outranks a positive that doesn't move S.

## RESOURCES + SUBSTRATE LAW
Full OSS; M5 Max 128GB saturated (CPU + MLX GPU); Modal < $5 standing budget (lane-claim +
HARVEST-OR-LOSE + estimate-first + fail-closed-over). MPS NEVER. Long training MLX-FIRST → numpy
reference (portability contract) → torch via tinygrad-like per-backend primitives (canonical_kernels,
Catalog #383), hardened parity + drift gates, per-axis tuning. Long compute = detached nohup daemon +
durable progress file + marker-on-exit harvest waiter; NEVER session-bound (the session-watcher trap
cost us six times — always arm a durable waiter).

## ANTI-DRIFT / ANTI-REWARD-HACK
Completion = machine-checkable evidence, never assertion; re-check S vs the next threshold at unit end.
Never optimize the proxy (surrogates are gradient rows; exact terms are authority; the gap is a
finding). Implementer ≠ grader. KILL is implementation-scoped (Catalog #307): paradigms get DEFER +
reactivation. APPEND-ONLY provenance. NO FAKE (a reuse claim the code doesn't honor is a bug).

## END-OF-UNIT CONTRACT
Every unit ends with: (1) shas landed; (2) exact checks run; (3) typed rows; (4) the NEXT LEVER ALREADY
LAUNCHED with its pre-registered prediction; (5) the updated scoreboard (UPPER vs next threshold,
LOWER); (6) one burning mathematical question. If S > T_1 and no measurement is running and no
budget/values wall was hit, you are NOT done — start the next lever before yielding. A plan, a
question, or a frontier-hold is not a valid ending while S > T_1.

## CURRENT SCOREBOARD (re-verify, don't trust)
UPPER: recoded-R3 0.19109982 [contest-CPU] (ours, defensive hold, ABOVE T_1 → GOAL UNSATISFIED;
both-axis paired, submission-blocked only on the `constriction`-allowlist operator disposition — a
DEFENSIVE bank, not the innovative submission). CUDA champion pr106 0.20533 (recode doesn't transfer).
Frozen-byte rate axis EXHAUSTED at lossless (T1/T8/S12 negative; decoder 98.6% iid Shannon; latents
per-dim marginal + cross-pair MI=0). Memorized-point retrain-continuation KILLED (degrades).
LOWER: F floor derived (`information_theoretic_floor_T_floor`): T_3 is a DISTORTION threshold at
constant bytes (0.135 at d_seg=d_pose=0); engineering ESTIMATE ~0.07-0.13, rate-binding. P6 (pose-output
entropy) CLOSED by the lever-B smoke: full 600×6 pose trajectory = **6.65 KB** (advisory). LIVE OFFENSIVE
LEVERS toward T_1→T_3: **B CONFIRMED + PROCEED-TO-CAMPAIGN** (2026-06-10
`lever_b_score_native_argmax_smoke_verdict`: a tiny MLX label-map generator hits the frozen SegNet's
600-argmax partition at d_seg=0.00826 in a 63,802-byte blob — 2.54× smaller than the frontier seg-share;
score-native carrier seg+pose = 70,452 B vs frontier 177,169 B, −60% bytes, rate Δ −0.071; `[macOS-MLX
research-signal]` advisory; NEXT = legal-frame variational bridge + paired CPU+CUDA exact eval). A/C/D
remain class-shift levers; D/G/H/I stack on B's confirmed carrier.
