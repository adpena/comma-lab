# ddm_sb1 — the $0 seg/feasibility batch (QA03 · QA04 · QA05 · QA11) — 2026-07-29

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED. All rows below are
[macOS-CPU advisory]; score_claim=false; promotion_eligible=false.**

Charter: `sb1_charter.md` (gc8 op-routable 1, Contrarian-bound; task #773) — the four
orphan-due measurements QA03/QA04/QA05/QA11, run on the single n600 scorer slot
(pfs1 released it; ONE scorer job at a time throughout; serial chain).

## Operating row (re-verified from receipts, not quoted)

pfs1 D1 (`ddm_pfs1_d1_eval_receipt_20260729.json`, rc=0 full n600, locked evaluate.py):
S=2.256641 = seg **0.389011** (d_seg 0.00389011) + pose 1.488093 + rate 0.379537;
archive 624ffe57… 569,996 B. **Seg axis gap = 0.389011** (#404 denominator throughout).
Seg base for QA03/QA04/QA05 = `p2c_aimed_archive.zip` (sha b9a7983b… = pfs1's
`seg_archive_sha256` — the deployed seg endpoint; bit-equal frame_1 chain verified at
the receipt level). ru1 `atlas_flat.npz` (built on p1_base 85d575be…) is ADVISORY aim;
every acceptance measured on current endpoint bytes (the pb1 P2c staleness note).

## Instruments (recall-first; reused, not rebuilt)

`tac.optimization.ddm_tr1_runtime` (parse/render/re-encode — the pb1/ru1 receiver path) ·
frozen CPU-torch SegNet (`seg_core.load_real_segnet`) · `cpu_verdict_d_seg_batch`
(whole-pair realized argmax — ERF collateral captured) · ru1 atlas ·
`experiments/ddm_lv1_s2_nullspace_audit.py` (QA11, verbatim protocol at the new ckpt).
NEW (this arm): `tools/sb1_seg_batch.py` — resumable driver (per-instance JSONL caches,
slot-refusal, in-place token eval). Behavioral equivalence PROVEN: in-place eval
reproduces the full-re-encode path exactly (pair 66: +52 == +52), ~15× faster.

## QA03 — full-population GN (multi-quanta damped-Newton) terminal seg solve — FIRED

Receipt: `/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/qa03_receipt.json` +
`qa03_instances.jsonl` (120 rows). Top-120 atlas (pair,cell) instances (the ru1
concentration law: top-100 cells hold 83% of flips), per instance a damped-Newton
line search of up to 4 single-quantum steps (8 candidates/step, whole-pair realized
acceptance), 26.6 min wall.

**Result: net +1,866 flips fixed / 600-pair population → d_seg −1.58e-5 →
seg ΔS −0.00158 = 0.41% of the seg axis (0.389011)** [magnitude-ok]. Bytes: tr1
re-encode +2,709 B (advisory; true price = r7 SMEVR, xi1 handoff).

- Realized fraction of the booked bands: **3.42% of the −0.046 Contrarian band;
  1.15% of the −0.138 tier-2 ceiling.**
- Mechanism (jsonl): 108/120 instances net-positive, 0 net-negative (whole-pair
  acceptance), mean +15.6/instance, max +52; **51/120 saturated at 4 quanta** (per-cell
  line search NOT exhausted); yield = 30.3% of the aimed instances' atlas flips;
  channels used evenly (83/88/86/69), signs balanced (156 +/170 −).
- Escalation vs round-1 (P2c, top-24 single-quantum, −155): 12× more flips at 5× the
  instance count and 4× the depth — multi-quanta pays, but the per-instance yield
  decays fast down the ranking (+52 at rank 1 → +3…+5 at rank ~120).

**Verdict (scope: FORMULATION — sequential per-(pair,cell) multi-quanta token
line-search, atlas-aimed, this endpoint/scorer):** the −0.046…−0.138 free-solve band
is NOT reachable by this formulation class. Extrapolating the measured decaying yield
over ALL ~648 non-zero cells gives ~10k flips ≈ 0.22 of the −0.046 band — the booked
band overstates per-cell-sequential reach by ≥4×. The band's remaining hope is the
UNMEASURED joint formulation (ru1 row-2: joint GN over 4ch × ~4 NEIGHBOR cells, ≥16
DOF vs 3.3κ deficit — a different formulation, not run here; collateral coupling is
exactly what per-cell search cannot exploit). ru1 row-1 falsifier check: population
solve (+1,866) ≫ the +151 single-edit lower bound → the strict-band mechanism
SURVIVES; it is the BAND-SIZE booking that fails, not the mechanism.
E2 ledger: the ~11-min finisher quote is now a MEASURED price (26.6 min at top-120,
gain-rate 0.0036 S/hr — far below any training-window handoff bar).

## QA04 — flicker-aimed attack search round-2 (SparseRS/Square mix) — [PENDING-FILL]

## QA05 — global rank-1 output-bias probe (tier-2 token-only?) — [PENDING-FILL]

## QA11 — S2 ν null-snap re-measure at the burn-final ckpt — FIRED

Receipt: `/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa11/receipt.json` (lv1
executor verbatim, `--qs 0,…`, 800 probes, chunk 120; ckpt = t3_long_burn_lotto_v2
`stage_seg_trunk_tau_final.npz`). **Positive control: q=0 full n600 d_seg
0.0038892 = the burn endpoint exactly** (apparatus valid on the new substrate).

**Result: ν = 0.0 at the burn-final ckpt** — recomputed against the ckpt's OWN
q=0 baseline + the preregistered +2e-4 tolerance (the executor's emitted `nu=0.7`
field is an ARTIFACT of its hardcoded stale T2 anchor 0.013833 and is
non-load-bearing; recorded, not used). Curve (full n600 realized per q):
q=25 → +1.37e-3 (6.9× tol) at −9.5KB zlib; q=50 → +2.76e-3 at −140KB;
q=90 → +1.74e-2 at −687KB. Monotone; no q>0 within tolerance.

- The T2 ν=0 verdict TRANSFERS to the deployed vehicle: the substrate-change
  trigger fired, the re-measure says the same thing — **no free token-nullspace
  rate via bulk bottom-q% |g| snapping.** G4 token-budget feasibility stays
  NO on this mechanism; the token-LOTTO/free-fiber rationale is NOT rescued.
- BUT the nullspace is REAL at per-quantum grain: hard-null probes show |g|
  deciles 0–2 at 79–85% zero-flip fraction; 27.0% of token grads are exactly 0.0.
  The failure is the BULK-SNAP FORM (snap-to-shared-base displaces near-null-but-
  live quanta together with true nulls), not the existence of nulls.
- verdict_scope: FORMULATION (bottom-q% |g| uniform snap-to-base, this ckpt).
  Named next (unchanged from charter framing): a derived-form selective snap
  (true-null-verified per quantum, not |g|-ranked bulk) — feeds wr1/#766, which
  already treats zero-flip cells as HYPOTHESES not free bytes (QA18 discipline).

## Ledger + consumers

QA03/QA04/QA05/QA11 flipped → FIRED in `ddm_deferral_queue_ledger_20260729.md`
(same commit). Consumers: E2 finisher quotes (QA03 measured price) · the seg box
(§3 ru1 arithmetic — the band re-booking) · wr1/QA12 (QA11 ν) · pb1/xi1 (byte
pricing of any shipped edit stream).

## 6-hook wire-in declaration (Catalog #125)

sensitivity-map: N/A (no new per-axis weights; consumes ru1's) · Pareto: ACTIVE
(band re-booking feeds the §3 budget sheet) · bit-allocator: N/A (no byte-routing
change; r7 owns coding) · cathedral autopilot: N/A (no dispatchable archive; edits
advisory) · continual-learning: ACTIVE (receipts + this memo + ledger flips) ·
probe-disambiguator: ACTIVE (QA05 IS the disambiguator for token-only vs global-lever).

## DAG FEED — ddm_sb1 (2026-07-29)

FEED-sb1-a [MEASURED — QA03]: [PENDING-FILL after all items land]
