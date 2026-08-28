# ddm_qbt2b_inherited_palette_birth — corrected class-birth curriculum (qbt2 NEXT_IF_RESUMED route a): FP1's real-path CE-TRAINED palette as an honestly-labeled inherited init + a data-dependent QBFLOW readout fit + the CE birth stage; NO launch from the arm

## MANDATE

Operator standing GO (08-21 "whatever it takes... frontier score lowering"). The qbt2
type falsifier (`.omx/research/ddm_qbt2_class_birth_curriculum_20260827.md` @
d638d0c5ae) correctly REFUSED the prior charter's "closed-form RGB from the frozen
head" construction — no such object exists in the corpus (FP1's palette is
Adam/CE-trained; the genuine closed-form bank lives in a 4-dim feature quotient with
no RGB inverse). The empirical question — can Road/Lane/Movable be BORN on the QBFLOW
vehicle — remains UNTESTED. This charter is MAIN's corrected charter per qbt2's
NEXT_IF_RESUMED, choosing option (a): FP1's palette as a provenance-labeled,
VIDEO-DERIVED, CE-trained inherited initialization, plus a separately-defined
data-dependent fit of QBFLOW's renderer readout, then the CE birth stage. Erratum
trail: qbt1 verdict memo carries the appended correction (269566a151); memory
`charter-provenance-claims-need-primary-implementation-trace` binds the authoring
discipline this charter now follows (file::symbol pins below, not memo labels).
NO training launch, NO Metal claim, NO Modal, NO full-n600 scorer from the arm.

## SCOPE

1. **Recall-first (m122), the binding record**: qbt2's falsifier receipt
   (@ d638d0c5ae) — its 6 DEAD-ENDS bind verbatim (no closed-form relabeling; no
   rank-4-quotient-as-RGB; no head-only solves passed off as real-path); its 4
   LIVE-HYPOTHESES are this arm's work items. FP1 primary implementation:
   `experiments/ddm_fp1_class_field_projection.py::solve_prototypes` (sha
   c2b22289480538bb8cb4db50a7a4e0b1d947e6c65245e09ad591d024b71fadf4) + receipt
   `.omx/research/ddm_fp1_class_field_projection_20260731.md` (sha b594de4b…, the
   "32-pair/100-step CE" row). The #315/#686 law (CE births, margin sharpens,
   event-triggered hand-off). #208 rare-class-protected init. tv1/tv2 interior-conflict
   (#1253) — discriminated by per-stage pose telemetry, not assumed. Scorer semantics
   per qbt2's source-verified statement: SegNet preprocess = last-frame selection +
   bilinear resize, NO normalization; the nonlinear EfficientNet-B2 body maps RGB to
   the 144-channel terminal-head domain.
2. **Build A′ — inherited-palette init, honestly labeled**:
   (i) Locate FP1's trained (5,3) palette VALUES in FP1's artifacts and sha-pin them;
   if absent from disk, re-run `solve_prototypes` at its documented config
   (32-pair/100-step CE through the full frozen CPU SegNet + R, $0-class CPU) and
   RETAIN the payload (the P0 payload law — FP1's original values may only exist as
   a memo row). Provenance label everywhere: VIDEO-DERIVED, CE-TRAINED, INHERITED —
   never "closed-form", never "solved from the head".
   (ii) Fit ONLY the VALUES of QBFLOW's existing RGB readout
   (`render_out_w`/`render_out_b` — shapes unchanged, QBF1 ABI untouched) so pixels
   in each native-class region render ≈ palette[c]: a least-squares/regression fit on
   observed render states from the r1 checkpoint, LABELED as a data-dependent fit.
   If the linear readout cannot separate the 5 classes regionally from render_state,
   report the residual matrix honestly — do not force it; that is itself the
   measurement (qbt2 live-hypothesis 3).
   (iii) **$0 init receipt**: render the 32 r1 pairs from the re-initialized state →
   R → uint8 → frozen SegNet argmax → the qbt1 §4 per-class table (predicted share /
   within-class error) before vs after. INIT GATE: ≥4/5 classes present with
   within-class error <60%. An init-gate MISS does NOT close anything by itself —
   FP1's own palette was BORN through 100 CE steps, so Build B proceeds regardless
   unless the readout fit is degenerate (rank-deficient / residual ≈ variance), in
   which case STOP and report the typed blocker.
3. **Build B — CE birth stage 03a** (unchanged from qbt2's design): extend
   `experiments/ddm_qbt1_qbflow_trainer.py` with per-pixel CE on the REALIZED scorer
   logits (same real render→R→uint8→SegNet path, same chunking/checkpoint/resume
   discipline) BEFORE the existing joint margin+pose stage. Event-triggered exit per
   #315: all 5 classes present in the realized argmax with per-class within-class
   error below a derived threshold AND stable across 2 consecutive verdicts — never
   a hardcoded step count. Pose loss ACTIVE during birth with per-stage pose_mse
   telemetry (protects the family's proven pose leg AND discriminates the
   interior-conflict mechanism). The margin law then takes over exactly as built.
4. **Config + validation + storage leg 2 (REQUIRED, carried from qbt2)**: additive
   run-config schema (legacy configs still validate); EMA via
   `resolve_ema_law(total_steps)`; chunk ≤30 assert unchanged; resume identity test
   ACROSS the 03a/03 boundary. Storage two-landing leg 2: the compile-launch-request
   projection must model ON-DISK bytes = (per-checkpoint logical bytes +
   n_files × fs_cluster_size) × (steps / checkpoint_every) at the REAL step count,
   fail closed vs live df; re-encode retention defaults to one tar (or single npz)
   per checkpoint (the ExFAT 128 KB-cluster ~8× amplification is a measured constant
   of the AP custody tier).
5. **Deliver the sealed r3 fire order for MAIN**: config (n32, same seeded stratified
   lineage, 116 GiB ceiling m79, walltime + ON-DISK storage projections at the real
   schedule vs live AP df — ~61.5 GiB free at qbt2's 03:02Z read, re-read at write) +
   the palette custody receipt + fit residuals + the $0 init table + the fire-order
   checklist. MAIN reviews the .py (2 visible passes), claims lanes, fires on Metal.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO launch/Metal/Modal/full-n600 scorer from the arm;
  CPU-scorer use permitted only for the palette re-derivation (if needed), the $0
  init receipt, and a bounded n≤4 smoke.
- Custody root `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` (live
  df at write time; fail closed if free < projected demand). ALWAYS KEEP THE PAYLOAD
  (palette values, fit residuals, every receipt frame).
- Serializer commits w/ post-edit `--expected-content-sha256`; ALL `.py` = 2 genuine
  visible review passes.
- The frozen QBF1 ABI is IMMUTABLE: re-VALUING `render_out_w`/`render_out_b` is
  permitted (same tensors, same shapes — an init, not a schema change); any shape or
  section change is a typed REFUSAL. Archive format, coder, §5 gate unchanged.
- Axis honesty: all rows [macOS frozen-scorer advisory], score_claim=false; n-scope
  honesty per m88 (same seeded stratified n32, never prefix).
- NO provenance-upgrading language anywhere (the fit is a FIT; the palette is
  CE-TRAINED INHERITED) — the qbt2 falsifier's NO-FAKE boundary binds.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends)

- qbt2's 6 DEAD-ENDS verbatim (d638d0c5ae §DEAD-ENDS) — especially: FP1's palette
  may not be called closed-form; rank-4 prototypes are not RGB; margin-law-from-
  step-0 is closed at INSTANCE scope (4,670 flat steps).
- v14's 1,700× projection loss under fixed flat paint — the init must be the
  scorer-evoked palette, not hand-picked colors.
- CE alone plateaus after birth (#177/#302 witness-line receipts) — the hand-off to
  the margin law is required, in that order.
- #1251 single-seed caveat binds every r3 number.

## OPTIMAL FORM

- Reference form + provenance pins (file::symbol, per the provenance-trace law):
  `experiments/ddm_fp1_class_field_projection.py::solve_prototypes` (@ c2b22289…) ·
  qbt2 falsifier receipt (@ d638d0c5ae) · qbt1 trainer
  `experiments/ddm_qbt1_qbflow_trainer.py` (@ fa5251ea…, EXTEND not fork) · qbt1
  verdict + erratum (@ 269566a151) · no2 §5 gate (@ d0fe0168b5) · gb1 sha
  ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4 (control object).
  SCOPE reductions (legal): n32; $0 receipt on the 32 r1 pairs. MECHANISM reductions
  FORBIDDEN: real R+uint8+frozen-scorer loop in the init receipt AND the CE stage;
  real coder re-encode per checkpoint; the event exit reads the REALIZED argmax.
- **PRIOR-LAW PREDICTION (falsifiable):** FP1's palette evoked all classes through
  the SAME frozen scorer in its own context; predict the inherited init + readout
  fit lifts ≥4/5 classes into the realized argmax at init, OR the CE stage births
  all 5 within a bounded window — after which the margin law descends below 0.2504
  for the first time on this vehicle. FALSIFIER: inherited-palette init + readout
  fit + CE still cannot birth Road at n32 → the QBFLOW seg leg closes at FORMULATION
  scope (CE-birth-first curriculum with a real-path inherited palette on the QBF1
  renderer) and the no2 family table closes on distortion; the memo must name where
  the proven pose (119.84→4.5e-4) and rate (~107.5 KB held) legs transfer.

## DELIVERABLE

`.omx/research/ddm_qbt2b_inherited_palette_birth_20260828.md` — palette custody
(values + sha + provenance label) + readout-fit residual receipt + the $0 init
per-class table (before/after) + trainer-extension inventory (files, tests, 2-pass
review receipts) + bounded-smoke receipts (CE stage + resume-across-boundary
identity) + the corrected ON-DISK storage projection + the sealed r3 fire order for
MAIN, OR the honest typed blocker / FORMULATION report. Commit via the serializer.
End with the own-vehicle frontier line (gb1 — S 0.14811799921260607 @ 180,215 B
[contest-CUDA T4 n600]).
