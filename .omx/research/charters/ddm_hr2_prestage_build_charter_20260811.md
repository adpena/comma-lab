# ddm_hr2 — realization-stage PRE-STAGE BUILD: close the camera-uint8 P0 gap + the SAFE-TO-PREPARE table

## Mission

Execute EXACTLY the SAFE-TO-PREPARE table from hr1's spec
(`.omx/research/ddm_hr1_realization_engineering_20260811.md` §Pre-staging assessment) plus the
P0 helper-gap closure hr1 named — so the realization stage's only implementation blocker is gone
before the ps135 terminal. Scorer-FREE, light-RSS, no model/scorer imports beyond what each item's
boundary allows.

## Ordered work (hr1's own table + the P0 gap)

1. **THE P0 GAP (hr1 §Binding receiver contract):** extend `tac.differentiable_eval_roundtrip`
   with a TYPED camera-uint8 ordering mode — bicubic-up → clamp/round-to-uint8 AT CAMERA
   RESOLUTION (forward exact, Uint8STE backward) → bilinear-down — alongside the existing mode
   (default UNCHANGED, byte-identical when the new mode is off; both-direction tests). Note the
   PR130/135 receiver's bilinear-lift discrepancy in the docstring; the mode takes the lift
   kernel as a typed parameter (bicubic|bilinear) so the incumbent-comparator receiver is
   representable too. Review-gate discipline: 2-pass review marks; 15+ tests incl. the
   #855/#903 controls (real-frame parity, per-tensor grad max-rel-err vs CPU torch, never
   loss-scalar-only).
2. **Camera-uint8 round-trip PIXEL positive control** on retained real RGB frames — training
   hard-forward camera bytes MUST equal the public-receiver camera bytes; every output retained
   (payload law). Pixel equality only; NO SegNet/PoseNet conclusions.
3. **Typed schemas/program factories** for the 4-arm race + compile/no-consumer tests (no raw
   flags — DSL-compile per #506).
4. **Checkpoint/resume/payload-manifest schemas** + atomic-write tests (apparatus tests, never
   mechanism evidence).
5. **Content-hash/path binder** for HY1 objects, terminal-base placeholders, HPAC sources
   (streaming reads; public intake read-only).
6. **Shape-only per-arm memory-configuration compiler** that emits REFUSE until a fresh
   real-config memory probe exists.

## Boundaries

Scorer-FREE (no SegNet/PoseNet forward anywhere). No Modal. No renderer/scorer weight loading
except as item 2's boundary allows (pixel path only). RSS per item ≤ hr1's projections (record
actual). No V0-V5 ladder, no HPAC retraining, no arms B-D, no terminal-base decode — those WAIT.
Serializer commits (post-edit --expected-content-sha256, [no-triality] [p0-ledger-ok],
--no-co-author). Durable memo `.omx/research/ddm_hr2_prestage_build_20260811.md` w/
NEXT_IF_RESUMED.

## OPTIMAL FORM

Pins: hr1 memo `.omx/research/ddm_hr1_realization_engineering_20260811.md` (its table IS the
scope) · `src/tac/differentiable_eval_roundtrip.py` current sha (record before edit) · hy1
solved-token/stream shas per hr1's binding table. SCOPE = all 6 items, no sampling; item-1 tests
on REAL retained frames only. PRIOR-LAW PREDICTION (derived): the new camera-uint8 mode will
show a NONZERO argmax difference vs the legacy mode on real frames (the uint8 cliff placement is
measurably material — sister of #532's uint8-breaks-exactness Δ=62.74) — if the two modes are
argmax-identical on all tested frames, the P0 gap was immaterial for this vehicle and the spec's
blocker downgrades honestly. FALSIFIER on item 2: any camera-byte inequality vs the public
receiver = the mode is WRONG, fix before landing, never ship a control that fails.
