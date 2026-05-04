# Submission Writeup Integration - 2026-05-02

Scope: paper/writeup narrative only. This ledger integrates the current contest-faithful evidence packet into the draft surfaces without editing `reports/latest.md` or the claim matrix.

## 2026-05-02 Supersession - C-067 Frontier

- [evidence:experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json] C-067 is the active exact A++ internal frontier: score `0.31561703078448233`, archive bytes `276214`, SHA-256 `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, SegNet `0.00061244`, PoseNet `0.00049637`, `600` samples, Tesla T4 CUDA.
- [evidence:experiments/results/pr67_direct_publicmask_c059_modelpose_20260502/build_manifest.json] C-067 archive anatomy is PR67 mask segment `219472` bytes plus C-059 model segment `55965` bytes and C-059 pose segment `677` bytes. The local score claim is for charged archive bytes, but the PR67 mask segment requires external-source attribution in the writeup.
- [evidence:experiments/results/lightning_batch/exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z/contest_auth_eval.adjudicated.json] C-063 remains valid A++ predecessor evidence at score `0.3156230307844823`, archive bytes `276223`, SHA-256 `83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d`, but it is superseded by C-067.
- [evidence:experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.adjudicated.json] C-059 remains valid A++ lineage evidence and submission-packet custody support, not the active frontier.
- Exact-eval runtime custody now requires `inflate_runtime_manifest.runtime_tree_sha256` in score-bearing comparisons. The same archive SHA can score differently if repo-local runtime Python changes; such comparisons are runtime-custody comparisons, not pure archive comparisons.

## Integrated Frontier

- [evidence:experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json] C-067 is the current exact A++ internal frontier: score `0.31561703078448233`, archive bytes `276214`, SHA-256 `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, SegNet `0.00061244`, PoseNet `0.00049637`, `600` samples, Tesla T4 CUDA.
- [evidence:experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/adjudication_provenance.json] C-059 adjudication records `promotion_eligible=true`, component gates passed, and a `-75` byte delta versus C-058 with unchanged measured components.
- [evidence:experiments/results/submission_packet_c059_20260502/submission_packet_manifest.json] The C-059 submission packet is metadata-only custody support. It records the archive, eval JSON, optional artifacts, validation checks, and field-supported `A++` classification, but its claim policy keeps `score_claim=false`, `ranking_claim=false`, and `promotion_claim=false`.
- [evidence:experiments/results/submission_packet_c059_20260502/submission_packet_checklist.md] Packet checks passed for archive SHA/bytes, sample count, component trace cross-check, eval-provenance device/SHA/bytes, and adjudicated JSON sample/SHA/bytes.
- C-063, C-059, C-058, and C-057 remain predecessor/context rows for explaining the PR67 comparison and micro-frontier chain, not the latest frontier.

## External And Quarantined Context

- [external] PR #67 is the contest-faithful public target and reports rounded `0.31`, `276564` bytes, PoseNet `0.00048597`, and SegNet `0.00061000`. C-067 uses the PR67 mask segment as charged archive payload, so the local score is A++ evidence for C-067 bytes while the mask source remains externally attributed.
- [invalid] PR #70 is exploit-forensics only because its reported improvement depends on moving score-affecting bytes out of `archive.zip` into script-side payload. It must stay out of ranked frontier tables.
- [external_quarantine] PR #68 and PR #69 are useful compliance-boundary lessons, not promotion evidence.

## Narrative Rules Added

- Present the method as a meta-Lagrangian atom compiler: typed mask, renderer, pose, residual, packer, and layout atoms carry charged byte cost, predicted component effect, interaction risk, rejection reason, source attribution, and archive identity when accepted.
- Treat atom-waterfill and hard-pair policies as proposal machinery. They become evidence only after selected atoms are packed into a deterministic archive and pass exact CUDA auth eval on identical bytes under recorded runtime custody.
- Preserve both positive and negative signals. Exact regressions, invalid exploit lessons, no-op controls, component-gate failures, and proxy disagreements should remain in evidence-tagged tables with revival criteria.
- Limit OSS and production-readiness claims to committed contracts and hardening: deterministic ZIP construction, payload closure, scorer-load guards, zip-slip and hidden-file exclusion, component gates, dispatch claims, JSON provenance, and reproducible eval commands.

## Negative Results And Bug-Class Hardening

- `A-negative` rows are reserved for exact measured regressions with archive custody, CUDA path, full sample count, component fields, and recomputed score. They retire only the measured config unless broader evidence exists.
- `invalid` / `external_quarantine` rows cover PR70-style script-side payload moves, PR68/PR69 boundary lessons, malformed archive metering, hidden sidecars, scorer patches, host-local dependencies, and CPU/MPS/proxy promotion attempts.
- `B` / `empirical` rows cover H100-only diagnostics, no-op packers, component-gate probes, smoke-only failures, queue anomalies, and byte-only wins. They can guide proposal filters but cannot rank.
- Every negative row should carry an artifact path, failure class, allowed use, and reactivation criterion. Repeated bug classes should become packet checks, validators, preflight checks, dispatch-claim rules, manifest requirements, or report-generator gates.
- [evidence:experiments/results/vast_harvest/qfaithful_zoom_runtime_fix_h100sxm_20260502T0920Z/contest_auth_eval.json] Q-FAITHFUL zoom-runtime closure fixed the charged geometry consumption bug, but the measured snapshot remains H100 `A-negative` at score `22.1476`; retire only that snapshot/export.
- [evidence:experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/cmg2_lossless_probe_charged_pr67_20260502T0950Z/cmg2_mask_codec_probe_manifest.json] CMG2 raw lossless probing is byte-regressive: best `340315` bytes versus the current `219472` charged mask segment. This blocks raw lossless wrapping, not predictive/lossy/scorer-weighted mask grammar.

## Next-Wave Roadmap Boundary

- Promote only charged archives: predictive/lossy mask grammar atoms, Q-FAITHFUL successor geometry, pose-basis atoms, hard-pair temporal windows, payload-efficient residuals, and packer/layout atoms must live in `archive.zip` or fixed contest code and pass exact CUDA auth eval on identical bytes.
- Use PR67/public-floor anatomy as proposal feedback, not evidence. Public PR text, proxy rows, H100 diagnostics, and byte-only savings remain non-promotable until the repo builds and evaluates a closed archive.
- The report generator should keep a `next_wave_roadmap` section separate from `frontier_summary` so predictions cannot silently become results.

## 2026-05-02 Late Integration - CMG2, CMG3, Dispatch Closure, And C-067 Matrix

- [evidence:experiments/results/lightning_batch/exact_eval_c067_cmg2_downsample2x2_t4_20260502T1000Z/contest_auth_eval.adjudicated.json] Plain CMG2 2x2 is exact Tesla T4 `A-negative scoped forensic`, not a score row: archive `194020` bytes, SHA-256 `e695829a9c45e827b8abc430b87c4871f7d563ff5b26767a6776483613fff3b1`, score `2.294741150018026`, PoseNet `0.30416307`, SegNet `0.00421524`, `600` samples, component trace `all_match=true`, `promotion_eligible=false`.
- [evidence:experiments/results/lightning_batch/exact_eval_c067_cmg2_foveated_top512_t4_20260502T1007Z/contest_auth_eval.adjudicated.json] CMG2 top512 AMR1 repair is exact Tesla T4 `A-negative scoped forensic`: archive `248074` bytes, SHA-256 `efd0da3ee2f451d7574409e4193ab2fc3fd09b5292315dd900fddea4426c6244`, score `2.1249135530811407`, PoseNet `0.24762903`, SegNet `0.00386108`, runtime tree SHA-256 `0e356bde4df817ea7c6557d67823bfaa5393ca4c38ae6fd4d3414732e6f459a0`, `promotion_eligible=false`.
- [evidence:experiments/results/lightning_batch/exact_eval_c067_cmg2_foveated_top256_t4_20260502T1007Z/contest_auth_eval.adjudicated.json] CMG2 top256 AMR1 repair is exact Tesla T4 `A-negative scoped forensic`: archive `219850` bytes, SHA-256 `e1f88079b8b36eb4326812d00d1e1a4c89a19b61778280c3924037e10fbbc664`, score `2.2229578832824526`, PoseNet `0.27912381`, SegNet `0.00405869`, same runtime tree SHA-256, `promotion_eligible=false`.
- Interpretation boundary: these rows retire only the measured nearest-neighbor CMG2 base and hand-picked AMR1 repair depths as promotion paths. They do not kill learned, predictive, row-span, or geometry-preserving mask grammars.
- [evidence:experiments/results/c067_predictive_mask_grammar_probe_20260502T1040Z/predictive_mask_grammar_probe_manifest.json] The predictive mask-grammar row-span probe is `empirical_byte_probe_only`, not score evidence. Best probe payload was `row_span_stride4_class_predictor` with `lzma6` at `63212` bytes, `-156260` bytes versus the `219472` byte charged PR67 mask segment. It excludes decoder code, archive wrapper, validator coverage, and exact CUDA eval.
- Next tranche: implement `CMG3` as a closed charged row-span archive with deterministic runtime decoder, validator allowlist, archive builder, local smoke, and exact CUDA gate. The first exact candidate should target stride4 row spans plus a small charged residual map; no roadmap row ranks until its own archive SHA/bytes and CUDA JSON exist.
- [evidence:.omx/research/shannon_floor_claim_matrix_20260430_codex.md#C-076] C-067 fixed-slice matrix hardening is an engineering guardrail only. The refreshed matrix now accepts recognized `public_pr67_qzs3_qp1_fixed_slices` payloads and records mask/model/pose slice bytes `219472/55965/677` with `score_claim=false`; it does not change C-067 as the active A++ frontier.
- Dispatch-claim closure guard: narrative and submission-pipeline materials should require terminal claim rows on completion (`completed_...`, `completed_score=...`, `completed_no_frontier`, or `failed_...`) so newer terminal rows close matching older active rows and do not leave phantom claims. This is an engineering guardrail, not score evidence.

## Edited Surfaces

- `.omx/research/contest_faithful_swarm_execution_20260502_codex.md`
- `.omx/research/submission_writeup_integration_20260502_codex.md`
- `docs/paper/04_results.md`
- `docs/runbooks/contest_faithful_submission_next_tranche_20260502.md`
- `docs/runbooks/contest_submission_pipeline_20260502.md`
- `reports/latest.md`
- `reports/writeup_working.md`
- `reports/yousfi_fridrich_observability_20260502/observability_report.md`

## 2026-05-02 Current Tranche - C-067 Observability, PMG Negative, SJ-KL Prep

- [evidence:experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md] C-067 byte accounting is an `empirical` control-plane profile, not score evidence. It confirms the unchanged-distortion sub-`0.300` crossing requires `23454` fewer bytes (`23455` buffered), with C-067 at `276214` bytes and target archive size `252759`.
- [evidence:experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png] The PNG byte-accounting profile is a visual/reporting surface only. It may appear in paper figures or observability dashboards but must not be treated as promotion evidence.
- [evidence:experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md] The charged stream breakdown is `masks.mkv=219472`, `renderer.bin=55965`, `optimized_poses.bin=677`, ZIP overhead `100`. Nested compression showed `0` best savings on the scored C-067 streams, so generic recompression is exhausted for the current byte grammar.
- [evidence:experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json] PMG atomtop4068 is L40S CUDA `A-negative scoped forensic`: archive `195762` bytes, SHA-256 `2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`, recomputed score `28.41411894150047`, PoseNet `62.34251404`, SegNet `0.03315286`, `600` samples, NVIDIA L40S. It cannot rank against C-067 or promote without T4/equivalent confirmation, and the measured PoseNet collapse is already decisive for stopping PMG row-run-only T4 promotion.
- [evidence:experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive_byte_accounting.md] PMG atomtop4068 saved `80452` bytes versus C-067 but paid catastrophic component distortion. Retire only PMG row-span/row-run residual rescue at this atom scale; do not kill charged mask grammar, learned topology, atom planning, or pose-conditioned residuals.
- [evidence:experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json] SJ-KL smoke tensor prep is `build_tensor_prep_only`, not score evidence: `4` pairs, `target_slot=0`, runtime target `robust_current JointFrameGenerator pair slot 0 / fake1`, `score_claim=false`, `promotion_eligible=false`.
- [evidence:experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json] SJ-KL full tensor prep is also `build_tensor_prep_only`: `600` pairs, `gt_pairs_btchw` shape `[600, 2, 3, 384, 512]`, `renderer_target_slot_chw` shape `[600, 3, 384, 512]`, and next builder command pinned to `--target-slot 0`.

Narrative integration:

- Keep C-067 as the only active frontier row until a newer exact CUDA archive exists.
- Add byte-accounting and observability surfaces as control-plane evidence supporting the next-action rationale, not as result rows.
- Add PMG atomtop4068 to negative-results tables with failure class `PoseNet collapse`, allowed use `mask-grammar redesign input`, and reactivation criterion `changed atom semantics with local parity plus exact CUDA archive evidence`.
- Add SJ-KL runtime integration and target-slot fix as OSS/production readiness progress: additive charged payload handling, shape-checked optional runtime path, manifest target-slot custody, and no scorer imports at inflate time. Do not claim score improvement until `sjkl.bin` is packed into a deterministic archive and exact CUDA auth eval lands.
- Next tranche should prioritize SJ-KL full tensor residual build and charged archive packaging, with local decode parity and validator coverage before any dispatch claim.
