# ddm_fd135 — PR135 COMPLETE recursive-fractal decomposition + OUR-UNIDENTIFIED-EDGES crosswalk

## Operator mandates (2026-08-10, verbatim)

"download and completely, recursively, fractally, understand and decompose." +
"We may have some things that they haven't identified yet. You have full research
online ability and also access to our entire corpus."

## Custody already on disk (ALL READ-ONLY intake; certify-or-block on cleanup)

- `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip` —
  186,724 B, sha `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`,
  one stored member `p` 186,624 B (payload sha16 `66da2921780038ca`).
- `.../pr133/archive.zip` — 190,212 B, member `p` 190,112 B.
- `.../pr135_src/` — PR135 head branch clone (submission runtime, 4,598 lines).
- `.../experiment_book/` — the author's FULL experimental record: 231 files,
  CHECKPOINT_01→02E memos + tests naming every surface (carrier_cbq, hpac_lora,
  frame0_selector, joint_pose_solve, residual_calibration, entropy_audit,
  margin_allocation_audit, renderer_weight_codec, rc64, coder_overhead, ...).
- A paired Modal replay of the PR135 archive is IN FLIGHT (MAIN,
  lane_pr135_replay_20260810) — do NOT re-dispatch; consume its rows when they land.

## Deliverable 1 — FRACTAL SECTION MAP of member `p` (per the fractal-audit standard)

Use THEIR OWN runtime off the shelf (pr135_src inflate.py/runtime/*) to parse `p`
level by level, loop-until-dry:
- L0: model_word + selector bits → codec routing (fixed-schema/WANS1/CAP1/RC64).
- L1: per-section byte spans {renderer weights · token/carrier streams · int12
  coefficients · frame-0 edit stream · selector K=8 stream · pose atoms} —
  MEASURED offsets + lengths + per-section entropy vs its own memoryless bound
  (the #996 protocol applied to THEIR base).
- L2: per-section DIFF vs PR130's 191,052 B section map (we hold PR130's anatomy
  from pi1/eh1) — WHERE did each of the −4,328 B come from; which of their
  "small lossless changes in many areas" overlap our lc2 ANS recode (−3,826 B on
  PR130's sections) and which compose.
- L3: coefficient-level: extract the int12 carrier coefficients + the 4-bit
  atoms 2/5/9 + frame-0 edits; quantify how far the coefficients moved from
  PR130's (the Jacobian re-solve's realized displacement field).

## Deliverable 2 — ExperimentBook COMPLETE read

Every CHECKPOINT memo + test surface: what they TRIED and dropped (dead ends =
both-ways signal), what `hpac_lora` and `margin_allocation_audit` are, their
measured tables, their stopping reasons. Rank {SHIPPED / TRIED-DROPPED w/ their
reason / NEVER-TRIED}.

## Deliverable 3 — OUR UNIDENTIFIED EDGES (the operator's hypothesis)

Cross OUR ENTIRE CORPUS against their NEVER-TRIED column. Named candidates to
adjudicate first (add any the corpus surfaces):
- lc2 same-state ANS token recode (our payload 178 B smaller than F26's).
- #580 resize-nullity projector (80.67% real-linear kernel DOF) — did they
  exploit R's null space at all? pi135 saw no sign of it.
- #401 blind-coordinate fill (230,904 camera px/frame invisible to both scorers).
- #139 hood-static clamp + self-detecting components.
- pk2 pose-carrier representation attack (23,384 B carrier) vs their carrier.
- #869 adaptive per-cell quantization map (the toolbox law: adaptive · aware ·
  sub-int16) vs their uniform atom bit-drops.
- Our gauge/ker(A) family (#519/#580) + the m91 per-EDGE decomposition
  (Road↔Lane = 49.2% of flips) — their seg is UNTOUCHED (0.00029639 ≈ PR130's);
  NOBODY has moved seg on this base. If any of our seg levers apply, that is
  the axis with zero competition.
- Online research: anything published since their 08-06 submission that attacks
  the same carrier family.
Output: ranked edge table {EDGE / evidence sha / projected ΔS on the PR135 base /
$0-or-cheap falsifier / consumer}.

## Ground rules

Full research + internal-leverage authority (standing sol-ultra clauses).
Borrowed_substrate_accounting on every transfer. Every byte figure MEASURED from
the custodied bytes. Axis labels everywhere; no score claims. Durable memo
`.omx/research/ddm_fd135_fractal_decomposition_20260810.md` + serializer commit
(post-edit --expected-content-sha256, tags [no-triality] [p0-ledger-ok]).
Checkpoint per protocol. Payloads extracted from `p` PERSIST to the intake tier
(ALWAYS KEEP THE PAYLOAD).

## OPTIMAL FORM

Reference: the #996 coder-axis protocol (per-section vs own memoryless bound) +
pi1's bit-anatomy standard + the fractal-audit loop-until-dry law. SCOPE = one
archive + one repo, NO mechanism reduction. Pins: archive sha
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`, PR130 bar sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, our anchor
sha `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.
PRIOR-LAW PREDICTION: the never-tried column will contain the R-null-space and
blind-coordinate families (no trace in their file list), and seg will be
untouched across their entire book — if they DID try seg and dropped it, their
stopping reason is the single most valuable paragraph in the repo.
