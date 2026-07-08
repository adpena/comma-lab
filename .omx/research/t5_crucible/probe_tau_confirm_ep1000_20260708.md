# T5 CRUCIBLE — τ-CONFIRM FOLLOW-UP: end-checkpoint m_q re-derive (ep1000 + ep650 EMA-best) — VERDICT: 0.062 does NOT stand; the m_q = 0.10 anchor is an APPARATUS ARTIFACT (witness-margin tautology), and the "support edge" convention the law bound to does not exist on the true GT-margin axis

`[no-triality]`
review_status: fresh-eyes-measured(1)
axis: [macOS-MLX research-signal] / [macOS-numpy advisory . NON-PROMOTABLE] — 16-pair advisory
re-render (96-pair for the anchor leg); NO score claims; pointer 0.19110 UNMOVED.
run under test (READ-ONLY, no run-dir writes): `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/`
cost: renders 81s + 82s CPU (16 pairs each), peak RSS 8.93 GB (~8.3 GiB — marginally above the
~8 GiB advisory budget, noted honestly); cached legs < 10 s each.

STORES CONSULTED: `.omx/research/t5_crucible/probe_waveA_ct_schedule_20260708.md` §4 (τ-CONFIRM
PARTIAL + the stated re-render path) · `ORCHESTRATION_LEDGER.md` (req H/J/L/Q/R) ·
corpus_query "m_q ln5 tau_end derive" (provenance chain: v3 §2.2d ← seal_round2 (b) ←
`birth_death_persistence_dseg_20260630T172510Z.md` L134/L196) · `tools/birth_death_persistence_dseg.py`
(the anchor's ACTUAL apparatus, read line-by-line) · `experiments/results/witness_per_stage_attribution/summary.json`
(the anchor witness's identity + d_seg) · `tools/witness_annulus_convergence.py` +
`tools/witness_per_stage_annulus_attribution.py` (render path, REUSED not reimplemented; L194–224
= the margin-key semantics) · `src/tac/witness_annulus_metrics.py` (canonical metric-math home) ·
`experiments/results/mlx_fleet_gt_cache/{gt_n600,gt_strided_n200}.npz` · `DRAFT_OPTIMAL_STACK_v3_20260707.md`
§2.2d (pre-GO re-derive rule) · `DRAFT_OPTIMAL_STACK_v5_20260707.md` consistency row (a) ·
`seal_round2_verdict_20260707.md` row (b) (the seal that re-executed the anchor — see §4) ·
`negatives_scale_validity_review_20260707.md` (per-side m_q caveat, untouched here).

## DURABLE INSTRUMENT (req Q)

| surface | path |
|---|---|
| metric math (pure numpy, tested) | `src/tac/witness_annulus_metrics.py::flip_margin_quantiles` |
| CLI (renders via the REAL render-through-R + CPU-torch SegNet path, reused; runs on ANY run/ckpt or cached maps) | `tools/witness_tau_mq_confirm.py` |
| tests (4 new; 22/22 module suite green) | `src/tac/tests/test_witness_annulus_convergence.py` |
| artifacts (full precision) | `.omx/research/t5_crucible/artifacts/tau_mq_confirm_{end,cached}_20260708.json` + rendered maps `.omx/research/t5_crucible/artifacts/tau_mq_maps/` |

Reproduce: `.venv/bin/python tools/witness_tau_mq_confirm.py --ckpt END=<run>/levelset_ckpt_stageTau_muon_ep1000.npz --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --pairs 16`
Instrument validation: on the SAME cached maps Wave-A used, it reproduces Wave-A's §4 numbers
BIT-FOR-BIT (BEST_ep300: mass<0.10 = 0.243173, rates 0.419388 / 0.00363118, q50/q90/q99 =
0.25406/0.93914/3.36112). ruff F clean; review-gate clean (no override).

## 1. ANCHOR PROVENANCE — the m_q = 0.10 anchor is an APPARATUS ARTIFACT (the corpus finding)

Chain: v3 §2.2d ← seal_round2 (b) ← `birth_death_persistence_dseg_20260630T172510Z.md` ("per-pixel
flip-rate 0.7645 for GT-margin < 0.10, ~0.000 above ⇒ m_q = 0.10"). Re-reading the generating
script (`tools/birth_death_persistence_dseg.py` L315–L405) against the maps producer
(`witness_per_stage_annulus_attribution.py` L194–224):

- **The anchor's "GT-margin" was NOT the GT margin.** The script binned flips by
  `wz["gt_margin"]` from `maps_l7.npz` — and that npz key holds the **realized WITNESS SegNet
  margin-toward-GT** (`logit[GT] − max_{k≠GT} logit[k]`, SIGNED), not the frozen GT-cache
  top1-top2 field. MEASURED: on `maps_l7.npz` the field spans **[−5.934, +15.515]**, and its max
  over flip pixels is **−1.12e-5 ≤ 0** — a flip has negative witness margin BY DEFINITION.
- **Therefore "~0.000 flip-rate above 0.10" was a TAUTOLOGY** (margin > 0 ⟹ argmax = GT ⟹ not a
  flip), and `np.digitize(..., [0.0, 0.1, ...])` + clip folded ALL flips (negative margins) into
  the lowest bin. Bit-for-bit reproduction with the anchor's own recipe ([::4,::4] subsample, 96
  frames = 1,179,648 px exactly): bin-[0,0.10) = 6,484 px, flip-rate **0.7644972239** = the
  anchor's 0.7645 row. The 0.10 "support edge" carried NO information about the GT-margin axis.
- **Witness identity** (for the record): θ* per-stage-attribution run, l7-stage ckpt ep725,
  softmax_temp 0.1361, d_seg 0.00428719 on its 96-pair subset — comparable d_seg to mod32cap, so
  Wave-A's "state-dependent, unverified at end" hedge was the wrong axis: the discrepancy is
  **apparatus-vs-anchor, not physics and not vehicle state**. Proof: the SAME anchor witness,
  re-measured on the TRUE GT-cache margin (`gt_strided_n200.npz`), has only **0.268877** of its
  flip mass below 0.10 (q90 = 0.81806) — statistically the same shape as mod32cap.
- Confound classification (CLAUDE.md confound rule): DEFAULT-HARMFUL × SILENT ×
  MEASUREMENT-CORRUPTING — a misnamed npz key (`gt_margin` = witness margin) silently corrupted a
  registered constant's derivation, and seal_round2's [re-executed] check verified the MEMO's
  arithmetic against the MEMO, not the field semantics. The instrument now refuses the trap
  structurally: `witness_tau_mq_confirm.py` ALWAYS takes the margin axis from a GT cache
  (≥ 0, witness-independent; verified min 1.05e-4 / 1.76e-4 on gt_n600/gt_n200) and documents the
  trap in its docstring. (Rename of the npz key itself = a wider refactor across 3 tools +
  cached artifacts; left as a named follow-up, see §5.)

## 2. MEASUREMENT — m_q flip-mass quantile table (req H per-class + per-pair; req J full precision)

Flip = witness argmax ≠ GT lstar; margin axis = frozen-SegNet GT-cache top1-top2 field;
m_q(q) = GT-margin below which q of the flip mass sits; τ*(q) = m_q(q)/ln5, ln5 = 1.6094379.
END/BEST rendered fresh through the real render-through-R + CPU-torch SegNet (16 pairs, stride 37,
n600 cache, so_iters 4); ep300/anchor legs from cached maps. `[macOS advisory subset — NOT n600
evidence; this probe re-derives a CONSTANT, it kills nothing]`

| leg | ep | τ(ckpt) | d_seg(16p) | rate<0.10 | rate≥0.10 | mass<0.10 | m_q50 | m_q80 | **m_q90** | m_q95 | m_q99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| END_ep1000 | 1000 | 0.2157 | 0.00388209 | 0.417211 | 0.00273223 | 0.298149 | 0.19803 | 0.49485 | **0.74347** | 1.08245 | 2.71257 |
| BEST_ep650 (EMA) | 650 | 0.3098 | 0.00345612 | 0.400825 | 0.00235065 | 0.321744 | 0.17943 | 0.44646 | **0.65607** | 0.88845 | 2.31518 |
| BEST_ep300 (cached) | 300 | 0.8050 | 0.00478458 | 0.419388 | 0.00363118 | 0.243173 | 0.25406 | 0.63413 | **0.93914** | 1.41863 | 3.36112 |
| ANCHOR_l7 (θ* run, 96p, true GT margin) | 725 | 0.1361 | 0.00428719 | 0.409991 | 0.00314330 | 0.268877 | 0.22298 | 0.54668 | **0.81806** | 1.13970 | 2.30186 |

Implied τ*_end = m_q/ln5 per quantile (END_ep1000 / BEST_ep650):

| convention | τ*(q50) | τ*(q80) | **τ*(q90)** | τ*(q95) |
|---|---|---|---|---|
| END_ep1000 | 0.12304 | 0.30747 | **0.46194** | 0.67256 |
| BEST_ep650 | 0.11148 | 0.27740 | **0.40764** | 0.55202 |

Per-class (END_ep1000; flip-mass share / m_q90): Road 0.3727/1.09587 · Lane 0.3827/0.75007 ·
Undrivable 0.1010/0.46756 · Movable 0.0963/0.61520 · MyCar 0.0472/0.25447. Top class-pairs by
flip mass: **Lane→Road 0.3795 (m_q90 0.75045)** · **Road→Lane 0.1978 (m_q90 1.94020 — the fattest
margin tail: high-confidence-GT Road pixels painted Lane = FP-lane/dash-misregistration class)** ·
Road→Undrivable 0.0734 (0.37804) · Road→MyCar 0.0592 · Undrivable→Road 0.0581. MyCar is the only
class approaching a compact sub-0.10-scale annulus (m_q50 0.08619).

Training-direction note (advisory): the flip-margin distribution TIGHTENS toward best (q90
0.939@ep300 → 0.656@ep650-best → 0.743@ep1000) and rate≥0.10 falls 0.00363 → 0.00235 → 0.00273 —
high-margin (structural/interior) flips are being consumed, but at ~10⁻³ rate over a huge
above-edge area they still hold ~70% of flip mass at end state.

## 3. VERDICT — per v3 §2.2d's own re-derive rule — scope: INSTANCE (the constant m_q = 0.10 and everything computed from it), law form τ* = m_q/ln5 untouched

- **Pre-registered interpretation applied** (charter, not post-hoc): m_q(90) ∈ 0.3–0.5 ⇒ τ*_end
  ~0.19–0.31; m_q(90) ≈ 0.10–0.15 ⇒ 0.062 stands. **Measured m_q(90) = 0.74347 (end) / 0.65607
  (best) — above even the pre-registered upper branch.** τ*_end = 0.062 does NOT stand.
- **Which quantile convention does the original intent bind?** (from the corpus provenance): v3
  §2.2d bound the law to the SUPPORT EDGE of a compact flip support ("GT-margin < 0.10 holds all
  flip mass" — effectively q→1.0 on a compact set). §1 proves that compact support never existed
  on the true GT-margin axis — on EVERY leg measured (both vehicles, three epochs) the flip mass
  is heavy-tailed to q99 ≈ 2.3–3.4. **The support-edge convention is UNDEFINED on the real field;
  the law cannot be re-derived to a single constant without first CHOOSING a quantile convention**
  — that choice is a stack-design decision (SC-3's live-m_q row is the right owner), not a probe
  output. Honest bands per convention: **q90 → τ*_end ≈ 0.41–0.46; q80 → 0.28–0.31; q50 → 0.11–0.12.**
- **Physics caveat for the chooser** (flagged, not decided here): flips at GT-margin ≫ τ·ln5 are
  NOT τ-limited — they are structural/capacity misses (mod32cap is the islands-unborn control;
  Road→Lane FP tail at m_q90 1.9 is not a smoothing artifact). A τ-law computed on the FULL flip
  population over-demands; a boundary-jitter-restricted m_q (e.g. flips within the |m|<2 annulus,
  or per-class-pair per L's asymmetry addendum) is the physically-motivated reformulation. Note
  the control's own end τ = 0.2157 ≈ τ*(q50–q80) band — and v2's 0.2, which v3 "corrected" to
  0.062 off the artifact anchor, sits INSIDE the defensible band under the q50 convention.
- **Kill-scope discipline (req R):** this probe re-derives a CONSTANT. Nothing dies. The Maslov
  form τ* = m_q/ln5 is untouched; δ_τ = τ·ln5 width-law consistency (v5 row (a)) now inherits the
  SAME artifact input and needs the same re-bind (it "agreed" with 0.10 because both sides cited
  the same corrupted number, not because two fields converged).
- Scale tag: **scale-bound** (16-pair advisory subset, this vehicle's end state; the anchor-leg
  reproduction is 96-pair). The n600 live-m_q row (SC-3) remains the authority path.

## 4. UPSTREAM CONSEQUENCES (named, for the stack owner — no silent recalibration)

1. v3 §2.2d τ_end = 0.062 → REVOKED-as-derived; τ_end becomes a CHOICE pending SC-3's convention
   decision (defensible measured bands in §3). v5 §1.4/§5 rows citing 0.062 inherit.
2. v5 consistency row (a) ("same law from two fields, ≈0.2% agreement") → CIRCULAR: both sides
   consumed the artifact 0.10. Re-derive δ_τ width from the true field after the convention choice.
3. seal_round2 (b) [re-executed] HOLDS-verdict → superseded by §1 (append-only; the seal verified
   memo-vs-memo arithmetic, which was correct — the corruption was one level deeper, in the
   anchor's own field semantics).
4. `birth_death_persistence_dseg_20260630` — the per-pixel margin table AND the component
   "peak GT-margin" curves used the same witness-margin field (same script, same key); its
   OTHER findings (component-size flip curves, R-survival, vineyard, PH-dims) bin by size/
   topology, not margin, and are untouched. Flagged for a targeted re-read before any further
   law consumes its margin-axis rows.
5. Follow-up (named, not done here): rename the `gt_margin` key in the maps-npz schema (producer
   `witness_per_stage_annulus_attribution.py` + consumers) to `witness_margin_toward_gt`, with a
   back-compat read; until then the trap is documented in `witness_tau_mq_confirm.py`'s docstring
   and neutralized in this instrument by construction.

## 5. HONEST LIMITS

- 16-pair strided advisory subset (anchor leg 96-pair); NOT n600 evidence. 16p d_seg(ep650)
  0.003456 vs recorded n600 best 0.0033662 (subset noise visible, direction consistent).
- Peak RSS 8.93 GB marginally exceeded the ~8 GiB advisory budget (81s/ckpt; no OOM risk margin
  concern at 128 GB, no live run present).
- Per-side (signed, per-class-pair-direction) m_q per req L's asymmetry addendum: per-PAIR rows
  are in the artifact JSON; a per-DIRECTION signed split is one flag away in the instrument if
  SC-3 wants it.
