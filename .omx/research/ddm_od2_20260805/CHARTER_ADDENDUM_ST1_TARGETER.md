# OD2 charter addendum — st1 surgical targeter BANKED (2026-08-05, MAIN)

Optional prior, not a scope change. If OD2's Stage-1 solve allocates per-pair/per-cell
budget, a measured targeter now exists:

- st1 Leg A selected student: **8192-bucket hashed local-context Road/Lane band targeter,
  9,718 counted B, 8,336/8,670 charter CQ1 SE3 r1 n32 flip-set hits (96.1% recall)**.
- Receipt: `.omx/research/ddm_st1_20260805/ST1_RECEIPT_20260805.md` +
  `ddm_st1_receipt.json`; artifacts under `/Volumes/VertigoDataTier/pact/ddm_st1_20260805/`.
- Use (if useful): rank pairs/cells for solve iterations by predicted flip density —
  the g3/g4 corpus made queryable at 9.7 KB. Do NOT use it as a paint mask (pe2 route
  adjudication: direct paint measured DEAD at n600, S 1.51–6.11 vs live 0.7539807).
- Ignore freely if OD2's own margin field already covers targeting; this is signal
  preservation, not an obligation.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.

## UPDATE (st2 landed, c9050f7ff1) — SUPERIOR targeter supersedes st1's as the prior

- st2 scorer-native-plus-context, 2048 buckets: **3,602 counted B, 8,656/8,670 hits
  (99.84% recall)** — beats st1 by +320 hits at −6,116 B (2.7× smaller).
- Features: cached GT top-2 margin field + Fisher trace + Road/Lane head-hyperplane
  distance (‖w_Road−w_Lane‖=3.953, lg1) + #141 gradient saliency + #725 HOPE BN stratum
  code. All compress-time only; nothing scorer-derived ships.
- Receipt: `.omx/research/ddm_st2_20260805/ST2_RECEIPT_20260805.md` + `ddm_st2_receipt.json`.
- Same usage rule: solve-budget ranking prior only, never a paint mask.
