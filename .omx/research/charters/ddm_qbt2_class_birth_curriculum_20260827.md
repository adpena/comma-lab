# ddm_qbt2_class_birth_curriculum — cure the measured 2-of-5 class-birth wall on the QBFLOW vehicle: prototype-color init + CE birth stage BEFORE the margin law; build + validate + hand MAIN the r3 fire order; NO launch from the arm

## MANDATE

Operator standing GO (08-21 "whatever it takes... frontier score lowering"). The
qbt1 R1+R2 verdict (`.omx/research/ddm_qbt1_r1_r2_qbflow_verdict_20260827.md`
@ ba52f6cdee) measured the QBFLOW family state: pose realization PROVEN
(119.84 → 4.5e-4 through the real render→R→uint8→PoseNet path — first carrier
ever) · rate PROVEN through training (~107.5 KB archives repeat-identical
across 973 re-encodes, under the 137,986 B cap) · seg BLOCKED at CLASS BIRTH:
the realized field is a TWO-CLASS output (Undrivable 59.42% + MyCar 40.58%
predicted shares; Road/Lane/Movable NEVER predicted, 100% within-class error,
frozen realized flip 0.2504 for 4,670 steps while the native interface
converged to 0.0174). This arm builds the measured cure and hands MAIN a
sealed r3 fire order. NO training launch, NO Metal claim, NO Modal from the
arm.

## SCOPE

1. **Recall-first (m122), the binding laws**: the #315/#686 derived-schedule
   law (CE births, tau/margin sharpens — event-triggered hand-off, exit CE on
   all-classes-born-and-stable, never a step-count constant) · m194/v14
   paint-not-partition + v14's margin-optimal prototype colors solved
   closed-form from the frozen SegNet head · #208 rare-class-protected
   structured init (Lane/Movable seeded, never zero) · m143 (the qbt1 miss:
   the w96b aligned-law receipts were measured on BORN fields; from-scratch
   birth needs CE first — do not repeat the transfer) · tv1/tv2 τ-inversion
   (#1253: pose lives in interiors — the pose–seg interior conflict is the
   INFERRED mechanism this build must discriminate, not assume).
2. **Build A — $0 prototype-color init**: solve, closed-form from the frozen
   SegNet head (no training), per-class RGB prototypes that maximize argmax
   margin through the real preprocess (resize→normalize per
   upstream/modules.py semantics; cite v14's construction, do not re-derive
   blind). Re-initialize ONLY the interior head's output mapping so each
   native class region paints its class prototype at birth; boundary latents
   and pose-relevant structure untouched. Emit a $0 receipt BEFORE any
   training: render the 32 r1 pairs from the re-initialized state →
   R → uint8 → SegNet argmax → per-class predicted shares (the §4
   decomposition table re-run). GATE: ≥4 of 5 classes present with
   within-class error < 60% at init. If the closed-form init alone cannot
   evoke Road through the frozen scorer, STOP and report — that is the
   FORMULATION falsifier firing early, worth knowing for $0.
3. **Build B — CE birth stage (stage-03a)**: extend
   `experiments/ddm_qbt1_qbflow_trainer.py` with a birth stage that runs
   per-pixel CE on the REALIZED scorer logits (same real path, same chunking,
   same checkpoints/resume discipline) BEFORE the existing joint margin+pose
   stage. Event-triggered exit per #315: all 5 classes present in the
   realized argmax with per-class within-class error below a derived
   threshold AND stable across 2 consecutive verdicts — never a hardcoded
   step count. Pose loss stays ACTIVE during birth (the pose leg is the
   family's proven asset; protect it — record per-stage pose_mse so any
   birth-stage pose damage is measured, which also discriminates the
   INFERRED interior-conflict mechanism). The margin law then takes over
   exactly as built (τ schedule unchanged from the validator's pins).
4. **Config + validation**: extend the run-config schema minimally
   (birth-stage fields additive, legacy configs still validate); EMA decay
   still resolved through `resolve_ema_law(total_steps)`; chunk ≤30 assert
   unchanged; resume identity test covers a resume ACROSS the stage-03a/03
   boundary. **Storage preflight two-landing leg 2** (the r2 ENOSPC lesson):
   the compile-launch-request memory/storage projection must model ON-DISK
   bytes = (per-checkpoint logical bytes + n_files × fs_cluster_size) ×
   (steps / checkpoint_every) at the REAL step count, and the fire order must
   fail closed against live df. Also default the re-encode retention layout
   to one tar (or single npz) per checkpoint — the 128 KB-cluster ExFAT
   amplification (~8×) is now a measured constant of this custody tier.
5. **Deliver the sealed r3 fire order**: config (n32, same seeded stratified
   lineage, same 116 GiB memory ceiling m79, walltime + storage projections
   at the real schedule) + the $0 init receipt + the fire-order checklist.
   MAIN reviews the .py (2 visible passes), claims lanes, fires on Metal.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO launch/Metal/Modal/full-n600 scorer from the arm;
  CPU-scorer use permitted only for the $0 init receipt + bounded n≤4 smoke.
- Custody root `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`
  (cite live df at write time — AP was 100% full 08-27 and is being repacked;
  fail closed if free space < the projected r3 demand). ALWAYS KEEP THE
  PAYLOAD; retention layout per SCOPE-4.
- Serializer commits w/ post-edit `--expected-content-sha256`; ALL `.py` = 2
  genuine visible review passes.
- The frozen QBF1 ABI is IMMUTABLE input (any needed change = typed REFUSAL).
  The archive format, coder, and §5 gate are unchanged — this arm touches
  INIT and CURRICULUM only.
- Axis honesty: all rows [macOS frozen-scorer advisory], score_claim=false;
  n-scope honesty per m88 (same seeded stratified n32, never prefix).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends)

- qbt1 r2: the margin-law-from-step-0 configuration is CLOSED at INSTANCE
  scope (4,670 flat steps) — do not re-run it hoping for late birth.
- v14's 1,700× Movable projection loss: fixed flat paint through the paint
  path without margin-optimal colors is measured-dead; the init must use the
  frozen-head-derived prototypes, not hand-picked colors.
- The witness-line CE-stage receipts (#177/#302): CE alone plateaus after
  birth — the hand-off to the margin law is required, in that order.
- Single-seed caveat #1251 binds every r3 number.

## OPTIMAL FORM

- Reference form + provenance pins: qbt1 trainer + verdict (@ 755f31e9ef +
  ba52f6cdee) — EXTEND, do not fork · v14 realization-fidelity memo (the
  prototype-color construction) · #315 event-triggered hand-off spec ·
  no2 §5 gate (@ d0fe0168b5) unchanged · gb1 sha ba1f3830…88a3e4 (control
  object). SCOPE reductions (legal): n32; $0 init receipt on the 32 r1
  pairs. MECHANISM reductions FORBIDDEN: real R+uint8+frozen-scorer loop in
  CE and in the init receipt; real coder re-encode per checkpoint; the
  event-triggered exit must read the REALIZED argmax, never the native field.
- **PRIOR-LAW PREDICTION (falsifiable):** the #315 law + v14 cure predict the
  $0 prototype init alone lifts ≥4 classes into the realized argmax, and the
  CE stage then births all 5 within a bounded window, after which the margin
  law resumes descent BELOW 0.2504 for the first time. FALSIFIER: prototype
  init + CE still cannot birth Road at n32 → the QBFLOW seg leg closes at
  FORMULATION scope and the no2 family table closes on distortion; the memo
  must then name what the pose+rate proven legs transfer to.

## DELIVERABLE

`.omx/research/ddm_qbt2_class_birth_curriculum_20260827.md` — the $0 init
receipt (per-class table, before/after) + trainer-extension inventory (files,
tests, 2-pass review receipts) + bounded-smoke receipts (CE stage runs +
resume-across-boundary identity) + the corrected storage projection + the
sealed r3 fire order for MAIN, OR the early FORMULATION falsifier report.
Commit via the serializer. End with the own-vehicle frontier line (gb1 — S
0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]).
