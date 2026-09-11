# ddm_ls1 — the Lane-conditioned surprise atlas on the SHIPPED move-44 field ($0, n600, measurement only) — charter, MAIN 2026-09-11

## The question (gdc4's successor pointer, gs3 Addendum 28)
Pointer move 44: S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]; rate demand at held distortion −25,899 B off the token
tail (119,909 B; non-tail 60,497 B). Twelve generator/context constructions closed at formulation scope (gb2, bnd2/3, tc2/3, eb2, bd1,
mc1, gdc1–gdc4). gdc4 (`.omx/research/ddm_gdc4_run_native_endpoint_generator_20260910.md`, sha 4cd0187b2f6d2c48…) showed the demand must come from the MODEL on the dense factorization, and tc1 measured
the mixing ceiling at ≈ 35 % of it (35-weight mixer realized −589 B = 6.09 % of a −9,011 B Miller–Madow joint oracle). Lane is 0.59 %
of area, 34.1 % of the incumbent's stream bits, and 59–65 % of boundary cost in gdc3's unrelated factorization (`.omx/research/ddm_gdc3_next_construction_against_the_geometry_law_20260910.md`, sha db4510fdae22e343…).
**Where do 25,899 B of Lane-conditioned surprise live that the HPAC prior and the free mixer do not already remove?** Measure it.

## The field (provenance law from gdc4)
Pin the SHIPPED field `subset6.u8` by the sha the move-44 encoder bound (a92e7d90…; read it from the encode receipt under
`/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/encode/` or the sj1 pass-6 store — record path/bytes/sha), NOT `pass6.u8`
(78e57545…, never shipped). Every number in your memo is on the shipped field.

## Deliverable (all MEASURED with the incumbent's real receiver-side models; no training; no Modal; no scorer)
1. **Per-symbol surprise atlas of the shipped tail.** Instrument the incumbent's decode path (RC64 + sparse HPAC prior + free adaptive
   corrector + counted mixer rider; `runtime/` of the move-44 tree, read-only copy) to emit −log2 p per coded symbol under the
   SHIPPED model, and aggregate: by class (5), by geometry class (gdc3's six: isolated/short/long × boundary-adjacent/interior),
   by frame position, by pair. Reconcile the sum to the real stream bytes (±0.5 %) — the reconciliation is the falsifier of the instrument.
2. **Lane-conditioned oracle ladder (upper bounds, each priced honestly).** For the Lane-heavy cells: (a) ideal-oracle conditional
   entropy given the true previous-row/previous-frame Lane geometry (what a perfect side channel could remove; the side channel's own
   cost bounded by tc2/tc3's measured 5.5 KB oracle map); (b) the same under only receiver-visible context (the decoded token plane,
   eb2's law); (c) the Miller–Madow joint oracle of tc1 re-run on the shipped field per class. Report each as bytes removable vs the
   25,899 B demand, per class and per geometry, with the bias correction named.
3. **Localization.** Is the removable surprise concentrated (a few hundred pairs / a band of rows / dash phases) or diffuse? Give the
   cumulative curve (bytes removable vs number of cells touched) — this decides whether a COUNTED Lane carrier (a positioned subspace,
   pc1/pc2 law) or a model change is the next object.
4. **Verdict table**: for each mechanism class — (i) bigger mixer (tc1 ceiling), (ii) receiver-visible context model (eb2 bound),
   (iii) counted Lane side-channel/carrier (cost from tc2/tc3 + pc2), (iv) born-vehicle generator (the-cross) — the measured bytes it
   can remove on this field vs its measured cost; name the ONE that clears the demand if any, or state the measured shortfall of each.
5. Memo `.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md` with the atlas tables, the oracle
   ladder, the cumulative curve, the verdict table, and the next charter you would write. Retain every instrument output on the SSD
   tier with sha. Serializer commit LAST (two review passes per .py; `REVIEW_GATE_OVERRIDE=1` for .md); rc 17 is NOT a stop.
   Checkpoint as `ddm_ls1` and mark it COMPLETE at the end.

## Boundaries
No training, no Modal, no scorer, no candidate archive; read-only on every `/Volumes/...` tree (copy the runtime to instrument it);
never edit `upstream/`, the PR tree, sealed trees; n600 only. Every number MEASURED / DERIVED / labeled; oracles are UPPER BOUNDS and say so.

## OPTIMAL FORM
- Reference form: tc1's instrumented mixer + Miller–Madow oracle (the measured ceiling) and gdc3's geometry probe as the two instruments
  to reconcile; the incumbent's real receiver models, not re-implementations. Scope reduction = SCOPE (declared; no verdict); a proxy
  model or a different coder = MECHANISM (toy-bracketed).
- Provenance pins (sha256 prefixes): gdc4 memo 4cd0187b2f6d2c48…; gdc3 memo db4510fdae22e343…; tc1 memo (grep .omx/research for ddm_tc1; record sha); the-cross
  memory; pointer move 44 commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e; shipped field
  subset6.u8 (record full sha and the encode receipt that binds it).

## Prior negatives accounted (operator 2026-08-15)
- gdc4 provenance defect: pin the shipped field by the encoder's receipt.
- gdc2/gdc3/gdc4: borrowed constants and premises (0.2909 B/mismatch; ≥90 % long-boundary) — every rate in your memo is measured on this field.
- eb2: the receiver holds no partition/pose — oracle (a) is a bound, (b) is what a receiver can use; never present (a) as reachable.
- tc1 / m164 (UNION ≠ SUM 3.705×): report joint removals, not sums of per-context removals.
- m166: −log2p is direction-dependent; average ≠ marginal — price removals as marginals under the shipped model.

Final message: the atlas headline (bits by class × geometry), the oracle ladder (bytes removable vs 25,899 B), the localization verdict,
the mechanism verdict table, the next charter, serializer rc, and the frontier line
`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: measurement charter; the atlas lands in the equations leg when it becomes a law -->
