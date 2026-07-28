# DAG FEED — ddm_op1 (openpilot + physics/geometry review for tb1) — 2026-07-28

For MAIN to fold into `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`. Pointer
`0.1910828242 [contest-CPU]` UNMOVED; $0 arm; no scorer jobs; research_only=true.

## FEED-op1-openpilot-custody-distilled-bev-vs-image-decided (2026-07-28)

- **P1 custody CONSUMED, not re-derived** (memo `.omx/research/ddm_op1_openpilot_physics_geometry_review_20260728.md`):
  #326/#327 settled geometry (1.22 m; scorer fx=400.27/fy=399.82/c=(256,192); **v_h=174 MEASURED-optimal**,
  two-horizon-roles law; ξ/SE(3) path CONFIRMED-CORRECT — do not touch); #145/#156 scene facts (EON-native
  clip; comma2k19 segment identity; pose ~1–2 DOF w/ noise floors; chroma pre-cheap); #325 carrier-scoped
  negative (smooth-curve lane carrier floor 0.002144 n96 > need; prior = 64% positional headstart;
  image-space explicit centerline NOT cheap at 65 KB/600, IoU(0,1)=0.284; supercombo $0-CPU CONFIRMED);
  #263 rows absorbed/superseded (pose→terminal 6-eq law; hood→#139 core).
- **RECALL DISCREPANCY (binding report):** the op1 charter characterized #609 as "road/lane MORE STATIC
  in BEV" — the committed receipts say the OPPOSITE for the tested chart: v2 D0 hood PASS then
  **Road 39.02 px / Lane 47.12 px p50 ruling residual, 4.3% ≤1 px, n600 → NON-static; probe verdict KILL
  (exact-G1-chart scope, 3 reactivation criteria)**. Consistent with the 07-27 pantheon canon ("atlas
  lives in the IMAGE chart; ξ predicts motion but cannot place boundaries"). No number consumed from the
  charter framing.
- **P2 DERIVED RECOMMENDATION (for tb1's T2 freeze): token grid stays in the IMAGE plane; NO BEV grid
  lane in the D∈{8,12,16} race.** Five legs, each on a MEASURED anchor: (1) #609 v2 inverted the
  staticity premise + surviving image-chart variation is d_gauge (flicker) which BEV cannot remove;
  (2) d_seg's metric is pixel-uniform ⇒ image grid is the perspective-optimal allocation (BEV cell
  density per image px grows ∝ d²; 1 m cells at 100 m = 0.049 px); (3) Lane is sub-pixel beyond ~60 m
  (60/d px) and a BEV→image resample stacks an extra erasure stage BEFORE the fd2 uint8/R wall;
  (4) only Road+Lane (23.8% of pixels) have a ground chart — sky+hood+movable don't; (5) the BEV upside
  (store-once, replay-through-ξ) is the same redundancy the all-pairs renderer already absorbs (pp1
  +57 KB gap, ee1 C10). **openpilot geometry enters tr1 as three FREE roles instead:** decode-side
  CLADE-ICPE-style conditioning features (v−174, d(v)=488.3/(v−192), dist-to-boundary — rule-118 free),
  compress-time supercombo/poly initializer, unchanged SE(3) terminal-pose path.
- **Pre-registered BEV re-entry falsifier (F1∧F2, $0):** F1 = #609-v2 reactivation (new custodied
  chart: D0 pass + Road AND Lane p50 ≤1 px, fraction ≥0.5, n600); F2 = BEV-chart token change rate
  <0.5× image-chart on Road+Lane AND ≥99% Lane token mass survives BEV→image→R. Predicted: F1 fails.
- **P3 ranked fresh sweep (8 rows, each w/ label + prior art + falsifier + tb1 consumption point);
  top-4 for pre-T2 attention:** (1) CLADE-class-adaptive modulation + geometric ICPE (T0 conditioning
  lever `renderer_conditioning`; A/B vs mini-SPADE at equal bytes); (2) row-anisotropic D foveation
  WITHOUT BEV (T1 grid-race variant; adopt iff ≥50% flip mass in rows 160–240 — $0 from gt_n600);
  (3) boundary-gated token code width, PointRend-logic-at-the-coder (T1; adopt iff ≥15% token-stream
  saving); (4) OASIS class-balancing on the seg loss (T0 lever; Lane per-class d_seg decides). Parked
  w/ falsifiers: PDE interior fill (CONJECTURE), crack-edge tokens (CONJECTURE), dirty-paper anchor
  (already SPEC §S2.4), Schur structure (engineering-only; fd1r says solve time is not binding).
- **Triality:** DAG leg = this FEED + memo. DSL leg = intentionally NONE (no lever adopted; adoption
  is the BUILD arm's job per DSL-as-SoT). Equations leg = FORMALIZATION_PENDING (no new measured law;
  P2 is a derivation over already-registered/receipted anchors). verdict_scope: review/derivation;
  the BEV family is NOT killed (exact-chart scope preserved).
- Pointer `0.1910828242 [contest-CPU]` UNMOVED — all means, no ends claimed.

- MAIN annotation (append-only): the #609-v2 **KILL** referenced in this FEED carries
  `verdict_scope: formulation — BEV-staticity in the exact G1 chart only`; BEV family alive
  behind the pre-registered F1∧F2 re-entry falsifier. Parked rows are INSTANCE-scoped
  sequencing decisions with falsifiers, not magnitude kills (see memo annotation for the
  relative-significance arithmetic). [magnitude-ok]
