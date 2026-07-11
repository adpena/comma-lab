# Appearance-phase d_seg endgame — build-legs disposition (2026-07-10)

Standalone leg-record for the two builds that landed this turn (kept out of the DAG because a
sister Fable agent held an uncommitted DAG append — no absorption). Their DAG FEEDs are already
committed by the build agents; this records the EQUATIONS + CONSUMER leg dispositions the
drift-detector asks for. `[no-triality]` (measurement pending, not a fake registration) +
`[consumers-generic]` (verified).

## T1 `phase_advection_consistency` (TRAIN side, #424) — commits 597f2e304 / 3fb2c7672 / d6d6869ee
- **DSL** ✓ — `PhaseAdvectionConsistency` Lever, auto-discovered by `lever_registry` (AST),
  `completeness().unmapped=[]` for phase-advect; flag `--seg-phase-advect-*` + Lever landed
  TOGETHER (flag-deferral cleared; never-invent-flags verified on the real parser).
- **DAG** ✓ — `FEED-phase-advect-build`.
- **equations = PENDING-A/B** — T1 is a BUILD, not a measured finding. The two candidates
  `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` +
  `transform_chain_phase_noise_partition_v1` stay UNREGISTERED per NO-FAKE until the ON-vs-OFF
  n600 A/B measures them. The flicker floor `0.005318` is already registered as Law-5.
- **consumer leg = [consumers-generic] VERIFIED** — `schedule_readback` renders via
  `describe()`/`to_display_dict()` soft-detect; `producer_bridge` iterates `activation_report()`
  by lever name; NO per-lever-class hardcoding in any consumer; the costate controller already
  carries T1 duty rows (via the a198a7f0 curriculum+costate landing).
- **SEAL owed** — round-1 review found+fixed 2 bugs (provider shape + the CRITICAL silent-no-op
  gate-flip omission = "lever never fires" class) → counter reset → 3-clean-pass SEAL owed
  before the operator fires the arm. LAUNCH = operator-GO.

## Phase-residual carrier (STORE side, #425) — commit 70055d352
- `src/tac/boundary_math/phase_residual_carrier.py` — predictive coder over the scored-frame
  (stride-2 f1) sub-pixel tie sequence; closed-loop ξ-transport predictor (decoder-visible fp16
  ξ, no drift); stores the quantized residual per GROUND class {Road,Lane,Undrivable}; one
  entropy stage; NO-FAKE bit-identity self-check. Reuses T1's `phase_primitives` + the pose ξ
  (#140/R1) — no separate ξ stored. `--phase-carrier` selectable mode in
  `levelset_byte_close_and_eval.py`, default-off byte-identical; `--phase-carrier` row in the
  #406 apply-pass.
- **DSL** = N/A / carrier-mode (per #140 pose-codec disposition, recorded so the detector sees
  the leg). **DAG** ✓ `FEED-phase-carrier-build`. **equations** = `phase_residual_carrier_bytes_vs_dseg`
  flagged MEASURED-not-registered (registers WITH the through-R A/B). **costate** =
  `phase_residual_carrier_359` registered UNMEASURED/duty-to-measure, confirmed surfacing in
  `duty_to_measure_ranked`. **task** `phase_residual_carrier_store_half_359` in_progress.
- **Honest cached-n6 findings (recovered-d_seg NOT claimed, OWED_through_R_n600_AB):**
  (1) ~1780 B/pair full-GROUND-annulus @q=1/64 — naive per-pixel store too expensive at n600 →
  needs the witness's smoother `_signed` tie + a coarser per-segment rep + the T1-amortized
  small residual (OWED refinement). (2) ξ-amortization currently neutral/slightly-negative on
  cached-margin ties — consistent with the `lane_groundframe_xi_transport_no_collapse` prior.

**Pointer 0.19108282 UNMOVED** — both builds move the score only when the through-R n600 A/B
byte-closes. The endgame is BUILT (train + store + curriculum + costate); the measurement is owed.
