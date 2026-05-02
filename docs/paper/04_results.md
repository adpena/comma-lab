# 4. Results

This section reports only evidence-tagged results. A score row can rank the system only when it names the exact archive bytes, archive SHA-256, CUDA auth-eval JSON, sample count, component values, and recomputed contest formula. Public pull requests are external context unless reproduced through our archive-custody path.

## 4.1 Current Verified Frontier

| Row | Evidence | Score | Archive bytes | SegNet | PoseNet | Archive SHA-256 | Artifact |
|---|---:|---:|---:|---:|---:|---|---|
| C-067 PR67-mask/C-059-model/C-059-pose fixed-slice frontier | `A++` | `0.31561703078448233` | `276214` | `0.00061244` | `0.00049637` | `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a` | `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json` |
| C-063 lossless Brotli repack of C-059 | `A++` superseded | `0.3156230307844823` | `276223` | `0.00061244` | `0.00049637` | `83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d` | `experiments/results/lightning_batch/exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z/contest_auth_eval.adjudicated.json` |
| C-059 QZS3 B32 mask-first QP1 fix1 predecessor | `A++` superseded | `0.3157055307844823` | `276347` | `0.00061244` | `0.00049637` | `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab` | `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.adjudicated.json` |
| C-058 QZS3/QP1 PR67-informed active-subspace byte micro-frontier | `A++` | `0.3157555307844823` | `276422` | `0.00061244` | `0.00049637` | `5145fb57be574b85639856d239420ffa35e605e32664f93e06753b120b21633f` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_active_fix2_t4_20260502T0250Z/contest_auth_eval.adjudicated.json` |
| C-057 QZS3/QP1 anisotropic basis pose continuation | `A++` | `0.3157562807844823` | `276423` | `0.00061244` | `0.00049637` | `63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z/contest_auth_eval.adjudicated.json` |
| C-056 QZS3/QP1 r8 scalar continuation | `A++` | `0.3159064496962538` | `276426` | `0.00061244` | `0.00049846` | `c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1` | `experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z/contest_auth_eval.adjudicated.json` |
| PFP16 Lane G v3 | `A++` historical | `1.043987524793892` | `686635` | `0.00400656` | `0.00346442` | `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` | `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json` |

C-067 is the current exact A++ frontier. C-063 and C-059 remain predecessor rows for the PR67 comparison narrative, lossless-repack chain, and one-byte micro-frontier chain. None of these rows is a Shannon-floor attainment claim.

C-067's archive anatomy is charged but source-attributed: PR67 mask segment `219472` bytes, C-059 model segment `55965` bytes, and C-059 pose segment `677` bytes. Because a public PR67 segment is part of the candidate, the writeup must carry external-source attribution even though the promoted claim is the local exact T4 archive eval of the charged bytes.

The C-059 submission packet at `experiments/results/submission_packet_c059_20260502/` is a metadata-only custody packet for the C-059 lineage. It confirms the archive path, SHA-256, byte size, `600` samples, Tesla T4 CUDA device, component fields, optional artifacts, and passed validation checks, but it is not a separate score source: its manifest records `score_claim=false`, `ranking_claim=false`, and `promotion_claim=false`. The active C-067 score authority is `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`.

## 4.2 Score Formula Recheck

Every row uses the contest formula:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

For C-057, C-058, C-059, C-063, and C-067, the reported rows are exact-T4 A++ evidence: `archive.zip -> inflate.sh -> upstream/evaluate.py`, CUDA device, `600` samples, component trace cross-check, archive SHA/bytes match, and adjudicated promotion eligibility. Rounded component values in prose are display values; the authoritative scores are the JSON/adjudicated rows.

## 4.3 External Public Context

| External item | Public status | Public score signal | Evidence tag | Paper use |
|---|---|---:|---|---|
| PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` | Open PR | Reports rounded `0.31`, `276564` bytes, PoseNet `0.00048597`, SegNet `0.00061000` | `external` | Target geometry and packer context |
| PR #65 `henosis_qz_n3z_r25_clean` | Open PR | Reports `284425` bytes and local score around `0.31968005` in local reverse-engineering notes | `external` | Side-channel/postprocess design motivation |
| PR #63/#64 public-floor lineage | Merged/visible public-floor context | `0.32`/`0.33` rounded band in local refresh notes | `external` plus local diagnostics when reproduced | Basin anatomy and packer transfer |

PR #67 remains the most relevant contest-faithful external source. Its public fields are still external claims, but C-067 locally evaluates a charged fixed-slice candidate that uses the PR67 mask segment with C-059 model/pose bytes. That makes C-067 local A++ evidence for the exact archive bytes and simultaneously requires PR67 source attribution for the mask segment.

## 4.4 Exploit And Boundary Evidence Quarantine

| External item | Public signal | Evidence tag | Allowed wording |
|---|---|---|---|
| PR #68 `loophole_v2` | Closed proof-of-concept moving compressed payload bytes into script-side data | `invalid` / `external_quarantine` | Demonstrates archive-metering loophole risk |
| PR #69 `houdini` | Open PR; no maintainer-filled eval report at inspection time; describes refactoring the data-flow boundary | `external_quarantine` | Unverified boundary experiment pending charged-payload audit |
| PR #70 `mask_decoder` | Reports rounded `0.19`, `57329` bytes, and states bytes were moved from archive into `inflate.py` | `invalid` / `external_quarantine` | Compliance lesson: script-side score-affecting payloads are not valid evidence |

These rows must not appear in ranking, promotion, or frontier tables. They belong in the compliance appendix because they stress the distinction between contest-visible `archive.zip` size and a stricter scientific payload-closure standard. Our own submission policy meters all score-affecting data as charged archive payload and rejects scorer patches, host-local files, hidden sidecars, malformed ZIP reliance, and script-embedded payload transfers.

## 4.5 Automated Result Surfaces

The final writeup should be generated from structured data, not hand-edited score prose:

| Section | Source rows | Evidence tags |
|---|---|---|
| `frontier_summary` | Current highest-grade claim row, currently C-067 | `A++`, `A` |
| `exact_artifact_table` | One row per exact archive eval | `A++`, `A`, `A-negative` |
| `public_external_context` | PR63/64/65/67 public anatomy and reports | `external` |
| `quarantined_exploit_context` | PR68/69/70 and any script-side payload cases | `invalid`, `external_quarantine` |
| `negative_results` | Scoped exact regressions and invalid evidence lessons | `A-negative`, `B`, `invalid`, `empirical` |
| `submission_checklist` | Payload closure and contest-compliance fields | `engineering_policy` |

Minimum columns for score-bearing rows: `row_id`, `grade`, `allowed_use`, `archive_path`, `archive_sha256`, `archive_bytes`, `eval_json_path`, `device_kind`, `gpu_model`, `n_samples`, `seg_dist`, `pose_dist`, `score_recomputed`, `inflate_runtime_manifest.runtime_tree_sha256`, `source_ledger`, source-attribution fields when external bytes are used, and `review_status`.

C-067 matrix hardening belongs in planning surfaces, not score tables. The refreshed breakthrough matrix accepts the active public fixed-slice payload contract, records PR67/C-059 slice bytes `219472/55965/677`, and keeps `score_claim=false`; C-067 remains the active A++ frontier until another closed archive passes exact CUDA adjudication.

## 4.6 Negative Results And Bug-Class Hardening

Negative evidence is part of the result set, but only with scoped evidence grades and allowed uses:

| Class | Evidence tags | Report use | Required fields |
|---|---|---|---|
| Exact measured regression | `A-negative` | Retire the measured config only; keep revival criteria | archive path, SHA/bytes, eval JSON, CUDA device, samples, components, recomputed score |
| Invalid compliance lesson | `invalid`, `external_quarantine` | Explain payload-closure and archive-metering boundaries | artifact path or PR, failure class, charged-byte issue, exclusion reason |
| Empirical/proxy blocker | `B`, `empirical` | Guide proposal filters and hardening, not ranking | command/log path, hardware, scope, promotion blocker |
| No-op or packer bug | `invalid`, `empirical` | Add validator/preflight coverage | before/after payload identity, provenance, targeted payload check |

The hardening outputs to cite are deterministic ZIP construction, payload closure, hidden-file/resource-fork exclusion, zip-slip rejection, scorer-load guards, component gates, dispatch claims, JSON adjudication, and the C-059 packet checks at `experiments/results/submission_packet_c059_20260502/submission_packet_checklist.md`. Dispatch claims must also be closed with terminal rows (`completed_...`, `completed_score=...`, `completed_no_frontier`, or `failed_...`) so completed jobs do not remain phantom active conflicts in the shared ledger.

Current scoped negatives to carry forward: Q-FAITHFUL's zoom runtime bug is fixed, but the measured snapshot remains H100 `A-negative` at score `22.1476`; CMG2 raw lossless wrapping is empirical and byte-regressive, with best lossless `340315` bytes versus the current `219472` byte charged mask segment. The exact T4 CMG2 wave is also `A-negative scoped forensic`: plain 2x2 scored `2.294741150018026` at `194020` bytes, top512 AMR1 repair scored `2.1249135530811407` at `248074` bytes, and top256 AMR1 repair scored `2.2229578832824526` at `219850` bytes. These retire only nearest-neighbor CMG2 plus hand-picked AMR1 repair as promotion paths; they do not kill learned, predictive, row-span, or geometry-preserving mask grammars.

The predictive mask-grammar row-span probe is `empirical_byte_probe_only`, not score evidence. Its best row, `row_span_stride4_class_predictor` with `lzma6`, measured `63212` bytes, `-156260` bytes versus the charged PR67 mask segment, but excludes decoder code, archive wrapper, validator coverage, and exact CUDA eval.

## 4.7 Historical Lessons

The older post-filter, renderer, PFP16, Alpha, OWV3, Q-FAITHFUL, and raw CMG2 results remain valuable as method history and negative evidence, but only with their original evidence grades. CPU/MPS/proxy scores, byte-only reports, H100 diagnostics without T4/equivalent promotion, public PR comments, and exploit submissions cannot rank or promote a result. <!-- MPS-DECISION-WAIVED: this is the rule restatement (matches CLAUDE.md non-negotiable on contest-CUDA-only score truth), not an MPS-derived decision -->

## 4.8 Method And Reproducibility Claims To Carry Forward

The current contest-faithful story is a meta-Lagrangian atom compiler, not a single lucky archive: typed mask, renderer, pose, residual, and packer changes are proposed as charged atoms with byte cost, predicted component effect, interaction risk, rejection reason, and exact archive identity once accepted. Atom-waterfill and hard-pair selection are proposal policies only; they become evidence only when selected atoms are packed into `archive.zip` and pass exact CUDA auth eval on identical bytes.

The hardening stack is part of the result: deterministic ZIP construction, payload-closure checks, hidden-file/resource-fork exclusion, zip-slip rejection, scorer-load guards, CUDA-only score truth, dispatch claims, component gates, structured JSON adjudication, and negative-result ledgers. The OSS and production-readiness claim is limited to these reproducible contracts and guardrails, not to unsupported generalization beyond the measured archives.

## 4.9 Next-Wave Roadmap

The next wave should stay inside the C-067 evidence contract: the immediate tranche is CMG3 closed row-span archive implementation plus exact CUDA gate, followed by charged pose-basis atoms, hard-pair temporal windows, predictive/lossy mask grammar atoms, Q-FAITHFUL successor geometry, payload-efficient residuals, and packer/layout atoms. None can promote from prediction, proxy, H100-only diagnostic, or byte-only evidence. A roadmap row becomes rankable only after a deterministic archive records its own SHA/bytes, payload closure, runtime tree hash, exact CUDA auth eval, component gates, and source artifact paths.
