# Latest Report - 2026-05-02 C-067 contest-faithful status

## Current Floor

**Best internal contest-CUDA artifact in the current claim matrix: C-067 PR67-mask/C-059-model/C-059-pose fixed-slice frontier = `0.31561703078448233` [A++]**.

- Evidence: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
- Archive: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Archive bytes/SHA: `276214`, `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Components: SegNet `0.00061244`, PoseNet `0.00049637`, samples `600`, device Tesla T4 CUDA
- Anatomy (charged but source-attributed): PR67 mask segment `219472` bytes + C-059 model segment `55965` bytes + C-059 pose segment `677` bytes
- External attribution required: PR #67 mask segment (see `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`)
- Authority: `.omx/research/shannon_floor_claim_matrix_20260430_codex.md` row `C-067` and `.omx/research/submission_writeup_integration_20260502_codex.md`

C-067 supersedes C-063 (`0.3156230307844823` bytes `276223` SHA `83615afd...`) and C-059 (`0.3157055307844823` bytes `276347` SHA `cf44aa7f...`) by `-0.0000060` and `-0.000088` respectively, with PoseNet/SegNet identical (improvement is pure charged-rate `-9 bytes` and `-133 bytes`). C-063, C-059, C-058, C-057 remain predecessor rows for the PR67 comparison narrative and the lossless-repack/byte-micro-frontier chain. None of these rows is a Shannon-floor attainment claim.

## Public Context

| Entry | Status | Score signal | Evidence use |
|---|---:|---:|---|
| PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` | Open external PR | Reports rounded `0.31`, bytes `276564`, PoseNet `0.00048597`, SegNet `0.00061000` | External target; mask segment used in C-067 with attribution |
| PR #65 `henosis_qz_n3z_r25_clean` | Open external PR | Reports `284425` bytes, local replay `0.31968`/`0.3600` | Side-channel correction motivation; multi-stage residual paradigm reverse-engineered |
| PR #63/#64 public-floor lineage | Merged/visible | `0.32`/`0.33` rounded band | Basin anatomy and packer transfer |
| C-067 | Internal exact A++ | Recomputed `0.31561703078448233`, bytes `276214` | **Active frontier** |
| C-063 | Internal exact A++ superseded | Recomputed `0.3156230307844823`, bytes `276223` | Predecessor in PR67 comparison chain |
| C-059 | Internal exact A++ superseded | Recomputed `0.3157055307844823`, bytes `276347` | Predecessor in lossless-repack chain |
| C-058 | Internal exact A++ superseded | Recomputed `0.3157555307844823`, bytes `276422` | Active-subspace byte micro-frontier predecessor |
| C-057 | Internal exact A++ superseded | Recomputed `0.3157562807844823`, bytes `276423` | Anisotropic-basis pose comparison anchor |
| PR #68 `loophole_v2` | Closed external PR | Proof-of-concept moving payload to script | Quarantine as `invalid`/`external_quarantine`; archive-metering loophole risk |
| PR #69 `houdini` | Open external PR | No maintainer-filled eval report at inspection time | Quarantine as `external_quarantine`; unverified boundary experiment |
| PR #70 `mask_decoder` | Open external PR | Reports rounded `0.19`, bytes `57329`, author states bytes moved into `inflate.py` | Quarantine as `invalid`/`external_quarantine`; not leaderboard-comparable |

PR #67 remains the most relevant contest-faithful external source. C-067 locally evaluates a charged fixed-slice candidate that uses the PR67 mask segment with C-059 model/pose bytes; the local exact T4 score is A++ evidence for the exact archive bytes and simultaneously requires PR67 source attribution for the mask segment. PR #68/#69/#70 are useful only for compliance hardening: they show why payload closure must meter every score-affecting byte and why our reports must reject script-side payloads, hidden sidecars, malformed ZIP reliance, or uncharged runtime data.

## Frontier Custody (last 24h)

Codex pose-manifold water-fill lineage (C-058 → C-059 → C-063 → C-067) on H100 NVL/SXM with Lightning Tesla T4 promotion. All micro-frontier improvements are pure charged-rate (PoseNet/SegNet identical, byte deltas only), reflecting the C-059-basin packer-and-layout exhaustion. Q-FAITHFUL successor work continues separately as `B`/`A-negative` evidence (zoom-runtime fix verified; measured snapshot still PoseNet-collapsed at `22.1476`); retire only that snapshot/export, not the future QAT++ or geometry-trained architecture.

CMG2 exact T4 wave landed as `A-negative scoped forensic` (plain 2x2: `2.295` at `194020` bytes; top512 AMR1 repair: `2.125` at `248074` bytes; top256 AMR1 repair: `2.223` at `219850` bytes). These retire the measured nearest-neighbor CMG2 base + hand-picked AMR1 repair only; learned/predictive/row-span/geometry-preserving mask grammars remain open. Predictive mask-grammar row-span probe is `empirical_byte_probe_only` (best `63212` charged bytes for `row_span_stride4_class_predictor`+`lzma6` on PR67 mask, `-156260` vs current `219472`-byte segment) and motivates CMG3 closed-archive implementation as the next charged-evidence step.

## Submission Pipeline

C-067 archive bytes are deploy-ready. Packet path candidate:

- archive: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip` (276,214 bytes, sha `226475de...8efb38a`)
- inflate: `submissions/exact_current/inflate.sh` + frozen upstream evaluator
- attribution: `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md` (PR #67 mask segment provenance)
- runtime custody: `inflate_runtime_manifest.runtime_tree_sha256` recorded in adjudicated JSON
- compliance audit: payload-closure check (no scorer patches, no host-local sidecars, no script-side payload movement, deterministic ZIP, hidden-file/resource-fork exclusion, zip-slip rejection, scorer-load guards, CUDA-only score truth)

## Report Pipeline Contract

Every public or judge-facing packet should be generated from structured rows with these sections:

1. `frontier_summary` - only Grade `A++`/`A` rows; current default is C-067, with C-063/C-059/C-058/C-057 retained as predecessor rows.
2. `public_external_context` - PR67/PR65/PR63/PR64 anatomy and claimed scores tagged `external`.
3. `quarantined_exploit_context` - PR68/PR69/PR70-style sidecar or rule-boundary evidence tagged `invalid` or `external_quarantine`.
4. `exact_artifact_table` - archive path, SHA, bytes, eval JSON, device, samples, component values, recomputed score, evidence tag, allowed use, and `inflate_runtime_manifest.runtime_tree_sha256` for cross-run comparisons.
5. `negative_results` - exact scoped regressions only [contest-CUDA]; no broad method kills from proxy, CPU/MPS, byte-only, or exploit evidence (these are `[advisory only]` device classes).
6. `submission_checklist` - payload closure, deterministic ZIP, no scorer patches, no sidecars, no hidden files, `archive.zip -> inflate.sh -> upstream/evaluate.py`, CUDA/T4/equivalent proof, inflate budget, review signoff.
7. `next_wave_roadmap` - CMG3 closed row-span archive + exact CUDA gate, predictive/lossy mask grammar atoms, Q-FAITHFUL successor geometry, charged pose-basis atoms, hard-pair temporal windows, payload-efficient residuals, and packer/layout atoms; all blocked from promotion until exact CUDA archive evidence exists.

## Caveats

- C-067 is the current internal exact frontier; C-063/C-059/C-058/C-057 are the predecessor rows for the PR67 comparison and lossless-repack/byte-micro-frontier chains. None is a Shannon-floor attainment claim.
- Public PRs and GitHub comments are external design signals unless we have exact archive bytes, SHA, CUDA eval JSON, component recomputation, and custody.
- PR #70's low reported score is non-comparable under our compliance policy because the public PR text says score-affecting bytes were moved from `archive.zip` into `inflate.py`.
- C-067 carries an external-source attribution requirement (PR #67 mask segment); the local score claim is A++ evidence for the charged archive bytes, but the mask source remains externally attributed per `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.
- The H100 NVL diagnostic of the same C-067 archive bytes scored `0.36295` earlier; this is a runtime-custody warning, not a contradiction. Cross-run comparisons require matching `inflate_runtime_manifest.runtime_tree_sha256`.

## Cross-references

- Paper draft: `docs/paper/04_results.md` (C-067 frontier table integrated)
- Codex writeup ledger: `.omx/research/submission_writeup_integration_20260502_codex.md`
- Working notes: `reports/writeup_working.md` (live operating point)
- Submission pipeline runbook: `docs/runbooks/contest_submission_pipeline_20260502.md`
- Reverse-engineering refs: `reports/raw/leaderboard_intel_20260501/` (PR #65/#67 inflate.py + archive.zip + line_search.py)
- Memory: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`, `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`

## Deadline

Contest deadline: **May 3 11:59 PM AOE = May 4 06:59 AM CDT (May 4 11:59 UTC)**. Approximately 48 hours from this report timestamp (2026-05-02 ~06:30 AM CDT).

## Next Queue

1. Generate the structured `paper_results.csv`, `paper_component_ablation.csv`, `paper_negative_results.csv`, and `submission_packet.md` surfaces from the claim matrix with C-067 as frontier.
2. Stand up the C-067 submission packet at `experiments/results/submission_packet_c067_20260502/` mirroring the C-059 packet schema with PR #67 mask attribution carried in the manifest.
3. Treat PR #67 as the external component target; close the remaining PoseNet/SegNet gap with charged archive atoms only.
4. Keep PR #68/#69/#70 evidence in quarantine unless a strict archive-metered rerun exists.
5. CMG3 closed row-span archive implementation + exact CUDA gate (predictive mask-grammar row-span probe at `63212` bytes is the empirical motivation, not score evidence).
</content>
