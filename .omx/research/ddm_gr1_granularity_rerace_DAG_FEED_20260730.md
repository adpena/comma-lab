# FEED — ddm_gr1 token-granularity re-race (2026-07-30)

PARENT: co9 granularity_race_duty ARMED→DUE (band re-parent onto tb1 0.00389 burn endpoint).
POINTER 0.1910828242 [contest-CPU] UNMOVED. Axis [macOS-CPU advisory]. score_claim=false.

## NODES

- gr1.grid: pfs1 D1 token field = [600×24×32×4] mod-16 (L16) SMEVR, 557,253 B = 97.8% of the
  569,996 B archive. The token field IS the rate.
- gr1.method: archive-faithful re-quant of the mod-16 residual (steps {1,2,4,drop}) → real SMEVR
  bytes; realized d_seg via ckpt-model render + frozen CPU SegNet vs GT. Gates PASS: ckpt↔archive
  codes 99.997%; baseline injection d_seg n600 0.0038892 vs evaluate.py 0.00389011 (Δ1.9e-6).
- gr1.token_DOMINATED: token-granular sensitivity-ordered {drop, nested-rung} — EVERY candidate
  worse than the current point on realized seg+rate (best tok_drop27 +0.086); B/flip 0.04–0.51 all
  below water 1.273. STRICTLY dominated by cell-drop. Scope INSTANCE/FORMULATION.
- gr1.QA11_overturned: QA11 "continuous log-bit dominates the rung ladder" (a $0 gradient
  prediction) OVERTURNED through the real coder+render. Mechanisms: (a) zero-GRADIENT tokens flip
  under finite drop (S2 caveat realized); (b) SMEVR conditions on per-cell temporal mode → the
  CELL, not the token, is the efficient coding+coarsening unit.
- gr1.QA07_dominated: nested-rung {L16,L8,L4,base} DOMINATED by clean drop-to-base at BOTH
  granularities (cell_rung_a 354,946@0.004681 vs cell_drop50 359,221@0.003947). Intermediate
  precision never pays.
- gr1.cell_frontier: cell-granular drop-to-base IS the RD frontier (confirms wr1's unit). Knee
  cell_drop50 = 359,221 B @ realized n600 d_seg 0.004310 → seg+rate 0.6702 = **−0.098 vs the
  current 0.7685**. Byte-closed `a6398e44…` (tokens `305a2be9…`). Better seg+rate knee than wr1
  Knee-A (−0.098 vs −0.032); same family, refined knee.
- gr1.pose_caveat: cell_drop50 ordering is seg-only / pose-blind; cell-drop freezes far-field →
  pose damage (co9 R1.1; wr1 +0.185). Composed decision cell = base + pose re-solve (ck1
  recoverable) = v4b/v4c, not this rung.
- gr1.finer_gated: ≥48×64 finer grid needs the QA24 re-burn (no-retrain cannot synthesize token
  values). Measured signal: token(finer-than-cell) is DOMINATED → finer is a coder gamble; COARSER
  is validated (cell-drop of low-|g| half near-free) → a from-birth coarser burn (QA24) is the
  promising re-burn. QA08 reopen (≥48×64) NOT met by a no-retrain win.

## LEDGER FLIPS
- QA07 → MEASURED DOMINATED (INSTANCE): nested-rung dominated by drop at both granularities.
- QA08 → stays CEILING-PRICED-CLOSED; the ≥48×64 reopen rider is NOT triggered (no-retrain finer
  impossible; token-finer dominated). Coarser-not-finer is the live direction.
- QA11 → the continuous-log-bit-dominates law re-scoped: INSTANCE-OVERTURNED through the real
  coder+render (gradient ≠ finite-drop cost; cell is the unit).
- QA24 → SIGNAL STRENGTHENED for the COARSER re-burn (post-hoc cell-drop −0.098 is a lower bound a
  from-birth coarser solve-init can exceed); still BLOCKED/HELD (4h re-burn, gated).

## HAND-OFF
v4b/v4c composed gate should consume a CELL-drop base (cell_drop50 `a6398e44` OR re-run the
|g|-sum-ordered cell-drop knee), NOT a token-granular base. Pose must be re-solved on the dropped
base (ck1-style). This rung supplies the RATE base + the measured −0.098 seg+rate; pose+seg-solve
are the P3v2/fd1 arms.
