# Factorized costate-organ upgrades A+B+C — landing + measured rows (2026-07-17)

**Pointer-delta honesty first: the exact frontier pointer is UNMOVED (contest-CPU 0.19108).
Everything below is MEANS — apparatus (score-neutral SENSE/DECIDE upgrades to the #247/#426
costate organ), built to sharpen the next exact row's aim.**

Axis of every number here: `[macOS-CPU advisory] NON-PROMOTABLE`, `score_claim=false`.
Live run touched: `experiments/results/levelset_n600_witness_20260717T113932Z/` — READ-ONLY
(only npz/log reads; all writes land in `.omx/state` / `.omx/research`).

## Operator additions (2026-07-17) folded in

* **ADDITION 1 — per-class split**: module B now emits the realization-vs-gradient
  classification + sub-LSB fraction SEPARATELY per GT class (5-class table below); the digest
  regime line surfaces the Lane/Movable cells. THE decision number (Lane sub-LSB 0.281 → MIXED
  → majority-recoverable) is in the measured table below.
* **ADDITION 2(a) — standing telemetry NOW**: the per-class split lands as a standing row in
  `.omx/state/witness_realization_regime.jsonl` each time B runs and is surfaced in the costate
  digest (read-only, no manual pull; works on the sacred live run). Run-dir artifacts are NOT
  written (the run dir is SACRED); `.omx/state` is the durable queryable surface.
* **ADDITION 2(b) — native verdict field (QUEUED, NOT applied)**: the trainer edit carrying
  `realization_regime_by_class` alongside `d_seg_by_class` in every verdict row is specified,
  ready-to-apply, in `.omx/research/QUEUED_EDIT_native_verdict_perclass_realization_regime_20260717.md`.
  It is score-neutral observability (defaults ON, cadence-gated). NOT applied because the
  trainer file is imported by the LIVE run — **flagged for merge at the post-c2 boundary.**

## What landed (3 modules + 1 tool + digest wiring + 2 canonical equations + 62 tests)

* **Shared core** `src/tac/witness_control/factorized_features.py` — the EXACT shared resize
  operator A as a closed-form tap table (torch-bilinear align_corners=False convention;
  VERIFIED against live `F.interpolate` to 2.5e-14 max-abs; blind-pixel perturbation → output
  bitwise unchanged; closed-form zero-weight fraction 0.22696926 vs canonical MEASURED
  0.226969, |Δ|=2.6e-7) · sha256-verified frozen-SegNet loader · canonical EMA-npz loader
  (consumes `build_witness_showcase._load_witness`) · the margin SNAPSHOT (real decode
  through R via `decode_levelset_torch` → frozen SegNet logits → exact pairwise margins at
  witness-vs-GT flips against bit-exact `gt_n600.npz` lstars).
* **A** `src/tac/witness_control/factorized_duty_ranking.py` — closed-form first-order duty
  marginal `Δd_seg(ℓ,ε) = (1/N)#{p: m_p ≤ ε·κ·‖w_pair‖·align(u_ℓ,p)}` from the rank-4 head
  law + DSL lever class-directions + the ker(A) zero-marginal theorem (κ=0 exactly on
  pure-ker camera support). Surfaced in the digest as an ALTERNATIVE beside the statistical
  duty line (never a replacement). Self-calibrated ε = the live snapshot's median
  feature-space flip distance (no new constants).
* **B** `src/tac/witness_control/realization_regime.py` — realization-vs-gradient regime:
  per sampled remaining-flip pixel, full-chain VJP to the camera tensor (differentiable
  resize bitwise-parity-asserted against `segnet.preprocess_input`; margin-equality guard vs
  the snapshot); `a_max = m·max|g|/‖g‖²`; sub-LSB iff `a_max < 0.5`
  (`realization_necessity_preimage_per_stratum_v1` convention). Regime cuts 0.5 / 0.25
  (majority / supermajority-open) are a stated DERIVED convention; the fraction is primary.
* **C** `tools/costate_live_ingest.py` — external read-only per-verdict ingest: canonical
  `witness_run_monitor.classify_line` surface filter + parsed-stage binding; computes the
  visible/blind (range(A)/ker(A)) residual-energy split + per-pair flip-distance histograms;
  appends a FEED-426-organ block (`continual_costate.append_trajectory_record`, run_ref
  suffixed `#factorized-ep<E>` so organ tournament records are never clobbered — tested on
  the real ledger semantics) + a compact row to
  `.omx/state/witness_factorized_snapshot.jsonl` (what the digest/A consume). Idempotent
  (state file + run_ref dedup backstop).
* Digest: `tools/costate_digest.py::section_factorized_sense` (reads ONLY the ledgers;
  SessionStart <5s budget preserved).
* Equations leg: `witness_realization_lsb_regime_v1` +
  `factorized_duty_marginal_projected_v1`
  (`src/tac/canonical_equations/{witness_realization_lsb_regime,factorized_duty_marginal}_20260717.py`),
  registered append-only in the registry.

## MEASURED rows (2026-07-17/18, live c2 run)

**B — the operator's live "run ~16h to ep1400 vs cut to terminal solve" input** (rolling EMA
shadow `__epoch=900`, 24 stride-25 pairs, 305 stratified VJP pixels over 18,094 flips,
sample d_seg 0.003835; deterministic — bit-identical across independent runs):

* **AGGREGATE**: `sub_lsb_frac_mass_weighted = 0.4346` → **regime MIXED**;
  `terminal_solve_admissible = False` at the majority convention. The majority of the
  remaining flip mass still has ≥LSB min-norm amplitude paths → the terminal SOLVE is not
  yet the forced move.

* **PER-CLASS split (GT class = the UNDER-SERVED class; operator ADDITION 1 — the number that
  decides ~16h to ep1400):**

  | GT class | sub-LSB frac | regime | remaining flips | interpretation |
  |---|---|---|---|---|
  | Road | 0.466 | mixed | 7,226 | (Road is near-solved at 0.66% d_seg; these are its residual edges) |
  | **Lane** | **0.281** | **MIXED** | **5,807** | **72% GRADIENT-limited → RECOVERABLE by more training; ~16h to ep1400 IS worth it for Lane** |
  | Undrivable | 0.647 | realization_limited | 2,401 | majority sub-LSB — amplitude training won't clear it |
  | **Movable** | **0.533** | **realization_limited** | **1,765** | **majority sub-LSB → NOT a dynamics story: the unborn-island REPRESENTATION gap (birth lever, not more epochs)** |
  | MyCar | 0.413 | mixed | 895 | (MyCar solved at 0.06%; residual) |

  **THE decision number: Lane's realization-limited fraction is 0.281 — Lane (the dominant
  remaining d_seg, 22.5% within-class) is MAJORITY gradient-limited (72% recoverable).** The
  ~28% sub-LSB Lane pocket ≈ the irreducible GT-flicker floor (night-wet, degraded markings,
  low-persistence dashes) the phase line already names. Movable's 0.533 realization-limited
  reading independently confirms the operator's framing: Movable is a representation/unborn-
  island gap, NOT amplitude-recoverable — routing to a birth lever, not more training.
  (Lane was also kicked by the ep726 Muon-start; this VJP is on the ep900 EMA, post-kick.)

**A — measured ranking** (12 stride-50-pair snapshot beside the ep900 verdict; self-calibrated
ε=0.0593): `lane_edge`/`thin_lane` 4.20e-4 first-order d_seg marginal > `horizon_margin`/
`chroma_boundary` 3.30e-4 > `persistence` 3.06e-4 — the closed-form ranking independently
reproduces the campaign's lane-first physics from the live margin field.

**C — first ingested factorized trajectory row** (verdict ep900): witness-vs-GT camera
residual energy is 77.4% scorer-VISIBLE / 22.6% in ker(A) — ≈ the blind-support fraction
itself, i.e. the residual is spatially unstructured w.r.t. A and the witness is NOT yet
exploiting the free ker(A) subspace (a rate-side design input, cf. the null-subspace P0).

## Honesty / limits (travel with the numbers)

* Subset-labeled: stride-25/stride-50 pairs + n=125 VJP pixels (the parent necessity law's
  own subset conventions); NOT n600-exhaustive rows; regime read is advisory decision
  support, never a verdict/score.
* Sub-LSB of the MIN-NORM move is necessary-side: dithered wider-support moves can still
  realize sub-min-norm-LSB margin changes.
* A's crossing test is pairwise first-order (third-class interception unmodeled).
* C binds the rolling EMA npz current at ingest time; `ema_epoch` is recorded next to the
  verdict epoch (recorded, not assumed).
* MEASURED wart found + pinned in tests: full verdict rows classify as `confound_alarm` in
  `witness_run_monitor.classify_line` (benign `"frozen_epoch": false` substring outranks
  the verdict category). This tool binds on parsed stage; the monitor-vocabulary exclusion
  fix (`"frozen_epoch": ?false`) is QUEUED for the post-run boundary (not landed mid-run —
  the module feeds live Monitor pipelines).

## Triality

DAG: this memo is the durable leg (FEED-style; graph-memory indexes it). DSL: no new trainer
flags — these are organ SENSE surfaces, not levers (nothing to fold; the lever set consumed
IS `lambda_net.LEVER_FEATURE_MAP`). Equations: the two registered laws above carry the
anchors. Ledgers: `.omx/state/witness_realization_regime.jsonl` ·
`.omx/state/witness_factorized_snapshot.jsonl` ·
`.omx/research/costate_organ_trajectory_ledger.md` (FEED-426-organ factorized block).
