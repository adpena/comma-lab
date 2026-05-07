# Frontier Artifact Triage - 2026-05-07 Worker

Scope: read-only triage of current untracked/ignored experiment artifacts for
exact-evaluable score-lowering signal. No GPU dispatch, lane claim, code edit,
runtime edit, archive rewrite, or dispatch-state edit was performed.

## Floor Used For This Triage

Score is lower-is-better. Public PR body/leaderboard values around `0.193` to
`0.195` remain external/public context unless an exact CUDA replay on the exact
archive bytes is present. For local exact-eval custody, the active HNeRV
rate-only anchor is now:

- Candidate: `pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z`
- Exact eval JSON:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- Archive: `experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive.zip`
- Bytes: `185578`
- SHA-256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Exact CUDA score: `0.20898105277982337`
- Evidence grade: `A++ contest T4`

For rate-only HNeRV candidates, the relevant byte floor is therefore `185578`
bytes, not the older PR106 source `186239` bytes or PR106x low-level Brotli
`186080` bytes. Any rate-only candidate above `185578` bytes is dominated for
score lowering unless it stacks onto this anchor or the anchor is invalidated.

## Candidate Triage

| Candidate path | Bytes / SHA | Evidence | Beats active floor? | Blockers | Next action |
|---|---:|---|---|---|---|
| `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json` | `185578` / `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce` | `A++ contest T4`; exact score `0.20898105277982337` | This is the active local exact floor; beats PR106 source and PR106x low-level exact baseline, but not the external public `0.193`/`0.195` body-score band. | Terminal/completed. | Use as the strict HNeRV rate anchor. Do not spend exact CUDA on rate-only candidates above `185578` bytes unless they stack with a scorer-changing candidate. |
| `experiments/results/pr103_repack_pr106_composed_op1_op2_20260507/archive.zip` | `185578` / `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce` | Same archive identity as the active exact anchor; manifest still says prediction because it predates the harvested exact eval. | Duplicate of active floor, not new signal. | Duplicate artifact identity. | Treat as de-duplicated evidence; cite the Lightning adjudicated JSON for score. |
| `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json` and `release_surface/archive.zip` | `186088` / `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7` | `empirical_archive_candidate_until_exact_cuda`; byte delta `-151` vs PR106 source; expected rate delta `-0.00010054470192144788` | No. It is `+510` bytes over the active `185578` byte floor, despite beating the older PR106 source. | Missing active lane claim and local Lightning env in field selection; entropy-frontier selector also blocks it as `not_below_active_candidate_byte_floor:185578`. | Do not dispatch for score lowering as-is. Rebase/stack onto the PR103-on-PR106 anchor, or keep only as a raw-equivalent conformance/proof row. |
| `experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/hnerv_lowlevel_exact_eval_packet.json` and `release_surface/archive.zip` | `186079` / `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2` | `empirical_archive_candidate_until_exact_cuda`; byte delta `-1` vs PR106x low-level baseline | No. It is `+501` bytes over the active rate floor and Pareto-dominated by q10 within the older PR106/PR106x comparison. | Field selection: missing active claim, KKT/ADMM proof, and pareto-dominated packet. | No exact-eval spend before a lower anchor-relative packet exists. Retain as archive-diet/saturation evidence. |
| `experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/hdm3_archive_candidate_manifest.json` and `exact_eval_static_release_surface/archive.zip` | `186066` / `5b5619628b54ccec44d51360ecb258dfe61742a581c7605c74d1ddaa5c025771` | `empirical_archive_candidate_runtime_adapter_parity_static_packet_ready`; byte delta `-14` vs PR106x low-level source | No. It is `+488` bytes over the active `185578` byte floor. | `ready_for_exact_eval_dispatch=false`; lane claim and exact CUDA missing; field selector marks strict preflight refused and entropy selector marks not below active floor. | Do not dispatch rate-only. Reuse the HDM3 runtime adapter only if rebased onto the active anchor or paired with a scorer-changing payload. |
| `experiments/results/hnerv_hdm3_entropy_packet_20260507_codex/candidate_packet.json` | No byte-closed archive | `planning_proxy_entropy_target`; field row expected delta about `-0.009032376699102255` but no candidate archive | No floor comparison possible. The signal is not exact-evaluable. | Missing candidate archive manifest, strict compliance JSON, runtime decoder contract, runtime tree parity, lane claim, exact CUDA. | Build a byte-closed archive/runtime consumer first. Until then it is an entropy target, not a dispatch candidate. |
| `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/wr01_exact_eval_packet.json` and `release_surface/archive.zip` | `186222` / `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628` | `empirical_archive_candidate_until_exact_cuda`; byte delta `-9`; changes decoded output | Byte floor: no, `+644` bytes over active floor. Score floor: unknown; no exact CUDA and decoded-output change means component drift dominates. | Missing Lightning env and active lane claim; component-response or exact-CUDA evidence missing; roadmap notes tiny byte win and decoded-output change risk. | Only exact-eval after an adversarial/component-response review shows plausible component benefit. Lower EV than rebasing rate-only work onto the current anchor. |
| `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/archive.zip` | `178258` / `c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933` | `empirical`; local decoder transducer parity, no contest score | Byte-only yes, it is below `185578`; score no, PR101 exact semantics are around `0.22635331443973267`, worse than the active exact anchor. | `ready_for_exact_eval_dispatch=false`; exact runtime parity not supplied; matching lane claim and exact CUDA missing. | Do not dispatch as a score win. Preserve as PR101 decoder conformance signal unless a stack path changes scorer behavior or combines with the active anchor. |
| `experiments/results/pr101_codecop_sweep_20260507_codex/substituted/pr101_native_op1_pr101_native_op1_auto_selectFalse_brotli_quality11/archive.zip` | `178258` / `87849d0097788c0295ad8954ef3f2e64db5a4fa504d5a8809d63c1e35ef3cf08` | `empirical`; sweep artifact | Byte-only yes; score no for the same PR101-semantics reason. | Manifest blockers: exact runtime parity not supplied, matching lane claim not supplied; another sweep row still has archive-substitution surgery pending. | Keep as byte-different PR101 packer evidence, not exact-eval priority. |
| `experiments/results/pr101_repack_pr106_20260507T152608Z_claude/manifest.json` and `archive.zip` | `185998` / `219844e77442fb352424a7f232d9a25f52e3f28d93b80fce800874ded94853aa` | `prediction`; PR101 split-Brotli-on-PR106; byte delta `-241` vs PR106 source | No. It is `+420` bytes over active rate floor. | Runtime adapter not integrated; PR106 inflate expected to fail on PR101 decoder format; runtime tree parity, inflate-output parity, strict compliance, lane claim, exact CUDA missing. | Dominated by PR103-on-PR106 for rate-only scoring. Revisit only if it unlocks a stack/runtime adapter that PR103 cannot provide. |
| `experiments/results/hnerv_pr101_schema_candidate_20260507_codex/pr101_schema_archive_candidate_manifest.json` and `pr106x_lowlevel_brotli_pr101_schema_candidate.zip` | `186044` / `a927bd957b34d18a85230ac1a784331c6b6ad9b25bde7532d646c875b7548d6e` | `empirical`; raw-equivalent f32 schema candidate; rate delta `-0.000023970922` if runtime supported and components equal | No. It is `+466` bytes over active rate floor. | Runtime tree parity, inflate-output parity, strict compliance, lane claim, exact CUDA missing. | Do not dispatch. Keep as schema/runtime-adapter research; rebase below `185578` before score-eval spend. |
| `experiments/results/hnerv_pr101_schema_candidate_20260507_codex/pr106x_lowlevel_brotli_pr101_schema_fp16_scale_probe.zip` | `185990` / `0928bf4692bd8055ba312cab2c14619c55177627318aa1ca61aeaa9648c4e4c9` | `empirical`; q-stream roundtrips but scales are not raw-equal | No. It is `+412` bytes over active rate floor and is scorer-facing. | Runtime adapter not integrated; inflate-output parity and exact CUDA missing; scale raw equality false. | Treat as substitute/scorer-facing probe, not lossless repack. Needs component-risk proof before exact eval. |
| `experiments/results/categorical_openpilot_payload_candidate_20260506_codex/readiness.json` and `archive.zip` | `169725` / `6a5bf1f3c4d6d2b6cb752979caa72a31b994825d24ee42ae90eeaa58f44abd7d` | `planning_manifest_audit` / local byte-closed payload custody; score claim false | Byte-only yes, but not a valid score candidate. No score floor beat because decode/reencode and runtime consumption are unproven. | No-op/decode-reencode identity controls not passed; full decode and byte-exact reencode not proven; runtime consumer is a charged fail-closed skeleton; exact-eval requirements missing. | Highest cross-paradigm hidden-gem blocker: recover HPM1 probability/range-state grammar and full 600-frame decode/reencode parity before replacing the skeleton runtime. |
| `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip` | `178981` / `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641` | `external_custody_exact_replay_missing`; title claims `0.19538 CPU`; archive identity matches PR100 intake artifact | Byte-only yes relative to `185578`; score no local exact evidence. | Existing PR102 intake archive appears to be the PR100 asset; exact CUDA replay missing; port-to-current-stack and no-op control missing. | Resolve release-asset identity first. Do not route as PR102 score-lowering until the correct public asset and exact CUDA replay exist. |
| `experiments/results/lane_bilevel_phase1_20260507T193228Z/phase_manifest.json` | No archive | `[CPU-prep]`; next action predicts `0.190` band only | Prediction could beat score floor if materialized, but no exact-evaluable archive exists. | Missing byte-closed archive, runtime closure, lane claim, exact CUDA. | Build the PR101 split-Brotli plus PR103 AC bolt-on archive on the chosen substrate; only then compare to the active floor. |
| `experiments/results/lane_deltaepszeta_pr106_candidate_20260507_codex/candidate_plan.json` | No archive | `[empirical]`; score claim false | No. Missing candidate archive. | Missing trained payload and missing candidate archive. | Materialize trained payload and archive before any exact-eval readiness discussion. |
| `experiments/results/lane_paradigm_chorus_20260507T192006Z/chorus.json` | No archive | `[empirical]`; many scaffold rows GREEN; score claim false | No. No score-affecting archive. | Pipeline/scaffold evidence only. | Convert GREEN operations into a byte-closed candidate row with archive SHA/bytes and runtime consumer. |
| `experiments/results/lane_per_tensor_shannon_pr106_20260507T173436Z/per_tensor_shannon.json` | No dispatch archive | `[empirical]`; roadmap says HDC1/HDC2 parity fixtures remain byte-negative versus source Brotli | No. | Deterministic decoder runtime missing; context-table overhead dominates. | Cluster/codebook-share context tables before building an exact-eval candidate. |
| `experiments/results/lane_score_gap_decomposition_20260507T191556Z/gap.json` | No archive | `[empirical:target+public-PR-claim:references]`; score claim false | No direct floor beat. | Reference/decomposition artifact only. | Use to prioritize candidate builders, not dispatch. |
| `experiments/results/field_meta_dispatch_selection_20260507_codex/selection.json` | Selection rows only | Planner/selector artifact; no dispatch | It selects no row that both beats the active anchor and is dispatch-ready. | Top selected rows are either missing claim/env/KKT, not below active floor, runtime-blocked, or proxy-only. | Regenerate after any rebased byte-closed archive lands; keep exact CUDA reserved for non-dominated rows. |
| `experiments/results/cross_paradigm_frontier_inventory_20260507_codex/inventory.json` and `experiments/results/frontier_roadmap_status_20260507_codex/status.json` | Inventory/status only | Control-plane artifact; score claim false | Confirms active anchor plus blocked planning rows. | Most rows need byte-closed archive manifests, runtime consumers, component gates, or exact CUDA. | Use as routing: current top work is not one-byte Brotli spend; it is scorer-changing/cross-paradigm closure. |

## Exact-Evaluable Interpretation

No current untracked/ignored artifact I inspected is both:

1. below the active local exact HNeRV rate floor of `185578` bytes,
2. exact-CUDA scored or ready for exact-eval dispatch, and
3. plausible as a score-lowering claim without new runtime/component evidence.

The main source of possible confusion is byte-only artifacts below `185578`
bytes, especially PR101-shaped candidates and the PR102/PR100 public archive.
Those have smaller bytes, but their local exact or semantic score evidence does
not beat the active exact anchor. They must not be promoted as score-lowering
without exact CUDA and runtime parity.

## Top Actionable Blockers

1. Active anchor landed: `pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z`
   is the A++ local floor. Update downstream selectors/scorecards to compare
   rate-only HNeRV candidates against `185578` bytes and score
   `0.20898105277982337`.
2. `pr106_q10_151byte_brotli` is static/raw-equivalent versus older PR106, but
   now dominated by the active anchor. Its unresolved blockers are lane claim,
   Lightning env, KKT/ADMM or explicit adversarial override, and most
   importantly rebase/stack below `185578`.
3. HDM3 is static-packet-ready relative to PR106x and has runtime-adapter
   parity evidence, but at `186066` bytes it is still `488` bytes above the
   active anchor. Rebase or stack it; do not spend exact CUDA as rate-only.
4. `wr01_apply_pr106x_half` is the most concrete scorer-changing archive
   packet, but it changes decoded output and carries only a `-9` byte delta.
   It needs component-response/adversarial evidence before exact CUDA.
5. The categorical/OpenPilot/HPM1 payload is the largest cross-paradigm byte
   signal, but it is not exact-evaluable until the HPM1 range/probability
   grammar and full decode/reencode/runtime-consumer parity are closed.

## Commands Used For Read-Only Triage

- `git status --short --branch`
- `git ls-files --others --exclude-standard experiments/results .omx/research .omx/state reports`
- `git ls-files --others --ignored --exclude-standard experiments/results .omx/research .omx/state reports`
- `find experiments/results .omx/research/artifacts reports -path '*20260507*' -o -path '*20260506*'`
- `rg -n -uuu "ready_for_exact_eval_dispatch|contest_auth_eval|archive_sha256|archive_size_bytes|candidate" ...`
- `.venv/bin/python` one-shot JSON/ZIP inspection commands for bytes, SHA-256,
  exact scores, ready flags, blockers, and nested candidate rows.
