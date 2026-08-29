# ddm_fcd2_distortion_legs_execute — execute fcd1's two MAIN-owned distortion fire-orders (trigger MET: r10 terminal 16:50Z; task #1319, owning memo `ddm_fcd1_field_for_coder_diagonal_20260829.md`)

## MANDATE

Operator 2026-08-29 (standing GO, verbatim): *"do whatever it takes and work for as long as it
takes autonomously with full authority and standing go to accomplish frontier score lowering."*
Routed finding: fcd1 (`ddm_fcd1_field_for_coder_diagonal_20260829.md`, commit 6df3b4ea9b) banked
a receiver-closed −3,756 B rate opening (union 176,436 B sha c45ab4e6…, rate-only ΔS
−0.0025009662 vs the jt21 bank 180,192 B sha ec0dd68f…) with distortion deliberately unmeasured
behind the one-full-n600-scorer-job fleet contract. That trigger is NOW MET: qbt2b r10 exited
rc=0 at 2026-08-29T16:50:55Z (keeper receipt + safe_run status=ok), its n32 chase is STOPPED by
the pre-registered trajectory rule (e(30k→40k)=−0.6022 > −0.85), and the Metal + scorer surfaces
are free. This arm EXECUTES fcd1's two QUEUED-WITH-A-FIRE-ORDER rows exactly as written, then
either hands MAIN a sealed dual-axis fire-order or closes per the folded orders. If the union
admits, projected S ≈ 0.1456 — the largest single move since jg5.

## SCOPE

1. **Fire-order 1 — fresh Schur chain on the union body** via
   `experiments/ddm_fcd1_incompile_schur.py` (the fcd1-built wrapper; subcommands
   decode/instrument/publish; instrument delegates jg5 modes {control, baseline, gn, refine,
   waterfill, close}; grep BOTH argparse surfaces before emitting any flag — never invent).
   NO re-decode needed: raws are RETAINED at
   `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/decode/{base_jt21,union}/inflated/0.raw`
   (base sha 7246a4ff…, union sha 042fad94…); staged pin-checked runtimes at
   `runtimes/{base_jt21,candidate_union,...}` (hd1 (sha,bytes) pin preflight enforced by the
   wrapper — a pin refusal is information, fix the pin per #1237, never bypass). Chain: baseline
   on the jt21 base (same-instrument n600 control, REQUIRED by publish) + baseline on the union →
   GN solve → close ×2 (byte-identical repeat REQUIRED) → `publish` with its in-code pose gate
   `d_pose_after <= d_pose_base + pose_band` (the gate also raises under `python -O` — fcd1
   built it in-compile per the qs4/qs5 law; a publish refusal is a MEASURED verdict, record it).
2. **Fire-order 2 — n600 frozen scorers on base AND union** (fires only on repeat-identical
   publish): realized d_seg + d_pose through the real R/uint8/scorer path on both bodies,
   recompute S FROM COMPONENTS (#877 — never the rounded display), apply the ±3.5e-6 canonical
   admit band to net ΔS = Δseg + Δpose + Δrate (rate leg already exact: −0.0025009662).
3. **Folded orders** (execute as written in the fcd1 memo): union not admitted but realized
   refusal ≤5× → process batch0, batch2, then batch1 through the SAME receiver/Schur/scorer gate
   (never infer batches from the union — scorer costs need not compose additively); any row
   admitted beyond the band → write the canonical candidate seal (`tools/make_candidate_seal.py`,
   dual-axis; single-axis needs a waiver reason per #1152) + fire-order into the consumer store —
   **MAIN retains Modal dispatch under single-flight**.
4. **Typed exit**: per-row table {realized d_seg, compensated d_pose w/ repeat, net ΔS,
   admit/refuse}; admitted → seal + READY row for MAIN; all refused >5× → the #1295 family
   closure with verdict_scope per the ladder (the fcd1 prior-law falsifier: re-selection refusal
   >5× across ≥3 independent batches refutes the pose-free re-selection hypothesis — count it
   plainly if it lands).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes
  (this arm should need few-to-zero code edits — the tooling is built; prefer running it).
- ALWAYS KEEP THE PAYLOAD; all receipts/artifacts to
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/` (the existing consumer store —
  do NOT fork a new store; fcd1's NEXT_IF_RESUMED names it).
- Axis honesty: every local row `[macOS-CPU frozen-scorer advisory]`, score_claim=false,
  promotable=false; ONLY the MAIN-fired T4 row is authority.
- Do NOT touch the r10 run dir (terminal, adjudication owned by MAIN) or its claims. Claim your
  own scorer lane via `tools/claim_lane_dispatch.py` before the n600 legs.
- Compensation must be the FRESH publish-gated solve — carrying qs2/qs4/qs5 numbers onto this
  object refuses the build (`ddm_qs4_collateral_suppression_20260813.md`, task #1039).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- fcd1 DEAD-ENDS (all in `ddm_fcd1_field_for_coder_diagonal_20260829.md`): same-move
  compensation dead at 45.18×; entropy/average/additive-credit pricing dead (real joint
  re-encodes only); carried compensation forbidden; B/H token labels ≠ realized SegNet flips
  (the exact wrong-object inference this arm exists to replace with measurement); stale native
  corrector refuses jt21 (use the Python `FreeCorrector` path, proven under budget ×2).
- qs5 (`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md`, 38553ad124): in-compile
  compensation PROVEN (d_pose below base, repeat identical) — reach, not transfer; this arm
  measures transfer to the union object.
- dg2 (`ddm_dg2_diagonal_distortion_verdict_20260824.md`): the uncompensated diagonal refused
  686× w/ pose 93.3% — the decomposition that shaped fcd1's re-selection.

## OPTIMAL FORM

- Family exemplar: the fcd1 landing itself — reference receipts `BYTE_ONLY_RESULT.json` +
  `PREPARE.json` + `DECODE.json` pair in the consumer store (commit 6df3b4ea9b), and the jg5
  Gauss-Newton machinery it wraps (the proven up2/jg5 pose-resolve lineage). This arm runs that
  landed form at full scope — no new mechanism.
- SCOPE reductions: NONE on admission claims (n600 realized only). The Schur GN solve may use
  the tool's own convergence controls; MECHANISM reductions FORBIDDEN (no surrogate scorers, no
  MLX pose authority — MLX-PoseNet drift 0.55% rel, CPU-torch is the advisory instrument).
- **PRIOR-LAW PREDICTION (falsifiable):** fcd1's screening arithmetic (1.403 exact
  label-benefits/saved-byte > the 0.785 realized-flip/B breakeven) predicts the union's realized
  seg leg is NET-NEGATIVE-or-neutral in ΔS, and qs5's proven compensation predicts the pose leg
  closes within band. FALSIFIER: realized seg ΔS > +0.0025 (eats the whole rate credit) — that
  refutes label→realized transfer at this scale and closes the family per the fcd1 falsifier;
  count it plainly.

## DELIVERABLE

`.omx/research/ddm_fcd2_distortion_legs_execute_20260829.md` — typed rows: (1) Schur publish
receipt {d_pose_base, d_pose_after, repeat-identity, gate verdict}; (2) per-body n600 scorer
table + S recomputed from components; (3) the admit/refuse adjudication vs the ±3.5e-6 band w/
net ΔS arithmetic shown; (4) seal + fire-order for MAIN OR the family-closure table;
(5) NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Commit via the serializer. End with the
own-vehicle frontier line.
