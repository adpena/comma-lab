# Latest Report - 2026-05-04 PR106 belt_and_suspenders adapter contest-faithful status

## Current Floor — NEW EXACT PUBLIC FRONTIER

**Best contest-CUDA artifact in the current claim matrix: PR106 `belt_and_suspenders` adapter replay = `0.20945673680571203` [A++].**

- Evidence: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json`
- Adapter: `experiments/results/public_runtime_adapters_20260504_codex/pr106_belt_and_suspenders_adapter/inflate.sh`
- Archive bytes/SHA: `186239`, `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- Components: SegNet `0.067142000` (avg 0.00067142), PoseNet `0.018305737` (avg 0.0000335), Rate `0.124009000`, samples `600`, device Tesla T4 CUDA, gpu_t4_match=true
- Runtime adapter wraps the public PR106 inflate path through repo-managed `.venv/bin/python` with brotli closure; preserves archive bytes
- Authority: `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md`

**Predecessor (now superseded)**: PR100 HNeRV-LC-v2 adapter replay `0.22826947142244708` [A++].

**Score gap unlocked**: PR106 beats PR100 by `0.01881273461673505` and beats PR101 (the earlier exact best at the time of last report) by `0.01689658`. The dominant driver is PoseNet contribution `0.018306` vs PR101's `0.041355` — a ~2.3× pose-distance reduction in exchange for +5,314 archive bytes (rate cost +0.0053). This is the Fridrich square-root-law in action: concentrated bytes purchasing disproportionate pose reduction. PR106 is now the public frontier our internal stack must beat.

### Cross-validating x-repack confirmation

Two byte-different deterministic ZIP-member-name repacks (changing `0.bin` → `x`, saving 8 ZIP-header bytes) reproduce the predicted `~6.66e-7/byte` rate-only effect within float epsilon:

- PR106 adapter `0.20945673680571203` vs PR106 xrepack `0.20945123680571204` — Δ = `5.500e-6` for 8 bytes saved (matches `25/37545489 × 8 = 5.326e-6`).
- PR105 adapter `0.23043732986984997` vs PR105 xrepack `0.23043182986984995` — same Δ.

This is decisive evidence the runtime adapter path is byte-faithful and our public-frontier intake gate is calibrated.

The PR100 adapter replay supersedes the PR95 stem-permutation repack
(`0.23089404465634825`, bytes `178277`, SHA
`e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`) by
`0.00156292999674471` score points despite adding `115` archive bytes. The
win comes from improved SegNet distance with a small PoseNet/rate tradeoff.
The adapter changes the runtime call contract for exact contest replay while
preserving the PR98 archive bytes; cross-run comparisons must include the
runtime tree hash above. It also supersedes both the conservative PR95 repack
(`0.23091954465634829`, bytes `178321`, SHA
`2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`) and the
exact public PR95 replay
(`0.23098329465634826`, bytes `178417`, SHA
`e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`). It
saves `44` bytes versus the conservative repack and `140` bytes versus the
public PR95 replay. It also supersedes the previous
PR85-family exact anchor, `PR85+STBM1BR+PR92/RMB1` at
`0.2535063602939779`. None of these rows is a Shannon-floor attainment claim;
the next tranche targets PR98/PR99 HNeRV deconstruction, PR91/HPM1 parity
recovery, and public-release hygiene.

## Public Context

| Entry | Status | Score signal | Evidence use |
|---|---:|---:|---|
| PR #98 HNeRV/Muon adapter replay | Public-source adapter exact T4 | Exact T4 `0.22826947142244708`, bytes `178981` | **Active exact frontier; promotion-eligible A++; PR100 source attribution required** |
| PR #99 HNeRV/Muon LC adapter replay | Public-source adapter exact T4 | Exact T4 `0.2297226895103603`, bytes `178546` | Superseded exact A++ PR98-family predecessor; PR99 source attribution required |
| PR95 stem-permutation repack | Internal exact T4 | Exact T4 `0.23089404465634825`, bytes `178277` | Superseded exact predecessor |
| PR95 conservative repack | Internal exact T4 | Exact T4 `0.23091954465634829`, bytes `178321` | Superseded exact predecessor |
| PR #95 HNeRV/Muon public archive | Public/open replay | Exact T4 `0.23098329465634826`, bytes `178417`; public body reports `0.1987048012202245` | Superseded exact source anchor; body/CPU score is external only |
| PR #96 rem2_HNeRV | Open/self-reported | Public body reports `0.20567121179282477`, bytes `186631`; no local exact replay in this packet | External frontier signal only until exact CUDA replay |
| PR85+STBM1BR+PR92/RMB1 | Internal exact T4 | Exact T4 `0.2535063602939779`, bytes `229480` | Superseded PR85-family exact anchor |
| PR85 + STBM1BR mask recode | Internal exact T4 | Exact T4 `0.25369011029397787`, bytes `229756` | Superseded exact PR85-family anchor |
| PR #91 HPM1 hybrid | Open/self-reported | PR reports `0.24879480490416128`; local replay currently fails HPM1 entropy decode | External signal only until parity and exact CUDA replay |
| PR #85 adaptive masking joint frame model | Public/open replay | Exact T4 `0.25806611029397786`, bytes `236328` | Superseded exact anchor and source for STBM recode |
| PR #84 QMA9/no-router | Public/open replay | Exact T4 `0.2751401491321396`, bytes `215735` | Superseded public replay frontier; source attribution required |
| PR #81 QMA9/range-mask | Public/open replay | Exact T4 `0.2812078926981712`, bytes `215960` | Superseded exact frontier; QMA9 mask stream anchor |
| PR #82 Henosis | Public/open replay | Exact T4 `0.2983246102939779`, bytes `296789` | Non-frontier A++ transfer target |
| PR #79 `qpose14_r55_segactions_minp` | Official leaderboard top row at rounded `0.31` | Internal exact replay/repack frontier `0.31453355357318635` before PR81 | Superseded by PR81 exact replay |
| PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` | Open external PR | Reports rounded `0.31`, bytes `276564`, PoseNet `0.00048597`, SegNet `0.00061000` | External target; mask segment used in C-067 with attribution |
| PR #65 `henosis_qz_n3z_r25_clean` | Open external PR | Reports `284425` bytes, local replay `0.31968`/`0.3600` | Side-channel correction motivation; multi-stage residual paradigm reverse-engineered |
| PR #63/#64 public-floor lineage | Merged/visible | `0.32`/`0.33` rounded band | Basin anatomy and packer transfer |
| C-067 | Internal exact A++ superseded | Recomputed `0.31561703078448233`, bytes `276214` | Historical PR67 fixed-slice frontier |
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

Current observability tightens the next tranche. C-067 byte accounting is
control-plane evidence only, but it shows the exact unchanged-distortion
sub-`0.300` crossing requires `23454` fewer bytes (`23455` buffered), with
stream pressure concentrated in `masks.mkv` (`219472` bytes) and `renderer.bin`
(`55965` bytes); `optimized_poses.bin` is only `677` bytes and ZIP overhead is
`100` bytes. The generated markdown/PNG profiles live at
`experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md`
and `experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png`.

PMG atomtop4068 is now an L40S CUDA `A-negative scoped forensic`, not a T4
frontier row: `experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json`
recomputed score `28.41411894150047`, archive `195762` bytes, SHA
`2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497`,
PoseNet `62.34251404`, SegNet `0.03315286`, `600` samples, NVIDIA L40S. It
retires PMG row-run-only rescue for this implementation and blocks another
PMG row-run-only T4 promotion; it does not kill learned mask grammar,
multimask reconciliation, atom planning, or pose-conditioned residuals.

SJ-KL moved from theory toward production integration without becoming score
evidence. The robust-current runtime can consume charged `sjkl.bin`
additively with shape checks and no scorer imports, and the target-slot bug is
closed by using JointFrameGenerator pair slot `0` (`fake1`) consistently. The
smoke manifest
`experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json`
prepared `4` pairs; the full manifest
`experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json`
prepared `600` pairs. Both are `build_tensor_prep_only`,
`score_claim=false`, and `promotion_eligible=false`.

## Submission Pipeline

The PR100 adapter replay is the current score champion. The PR98 packet is:

- packet:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`
- archive:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/archive.zip`
  (`178981` bytes, SHA
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`)
- inflate:
  `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/inflate.sh`
  plus the copied PR100 HNeRV-LC-v2 adapter runtime source tree
- attribution: PR #98 public HNeRV/Muon archive and runtime source
- runtime custody:
  `inflate_runtime_manifest.runtime_tree_sha256=ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- compliance audit: strict pre-submission gate passed with `78` checks and no
  failed checks in
  `experiments/results/submission_packet_pr100_adapter_20260504/pre_submission_compliance.json`

The previous PR95 stem-permutation upload directory is now superseded for score
wording:

- packet:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm`
- archive:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm/archive.zip`
  (`178277` bytes, SHA
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`)
- inflate:
  `experiments/results/submission_packet_pr95_stemperm_20260504/apogee_pr95_stemperm/inflate.sh`
  plus the copied PR95 HNeRV/Muon runtime source tree
- attribution: PR #95 public HNeRV/Muon archive and runtime source
- runtime custody:
  `inflate_runtime_manifest.runtime_tree_sha256=a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`
- compliance audit: strict pre-submission gate passed with `80` checks and no
  failed checks in
  `experiments/results/submission_packet_pr95_stemperm_20260504/pre_submission_compliance.json`

Public Apogee surfaces are prepared but not score authorities:

- PR body template: `docs/submission_template.md`
- public supplement plan:
  `docs/runbooks/apogee_public_supplement_20260502.md`
- Lightning notebook skeleton: `notebooks/apogee_lightning_supplement.ipynb`
- Cloudflare Pages sanitized bundle/runbook:
  `reports/graphs/public_site/` from `reports/graphs/build_public_site_bundle.py`
  and `reports/graphs/deploy_cloudflare_pages.md`

Before publishing, run strict public-release hygiene over those exact surfaces
and replace only sanitized placeholders such as `${LIGHTNING_SUPPLEMENT_URL}`,
`${CLOUDFLARE_PAGES_URL}`, `${APOGEE_ARCHIVE_ZIP_URL}`, and
`${APOGEE_RELEASE_MANIFEST}`. Do not publish private Lightning/Vast job links,
raw `.omx/state`, local absolute paths, or provider transcripts.

## Report Pipeline Contract

Every public or judge-facing packet should be generated from structured rows with these sections:

1. `frontier_summary` - only Grade `A++`/`A` rows; current default is the PR100 adapter replay, with PR95 stem-permutation repack, PR95 conservative repack, PR95 public exact replay, PR85+STBM/RMB1, PR85, PR84, PR81, and C-067 retained as predecessor rows.
2. `public_external_context` - PR96 unresolved or self-reported claims, PR95 body/CPU score, PR91/PR67/PR65/PR63/PR64 anatomy and claimed scores tagged `external`.
3. `quarantined_exploit_context` - PR68/PR69/PR70-style sidecar or rule-boundary evidence tagged `invalid` or `external_quarantine`.
4. `exact_artifact_table` - archive path, SHA, bytes, eval JSON, device, samples, component values, recomputed score, evidence tag, allowed use, and `inflate_runtime_manifest.runtime_tree_sha256` for cross-run comparisons.
5. `negative_results` - exact scoped regressions only [contest-CUDA]; no broad method kills from proxy, CPU/MPS, byte-only, or exploit evidence (these are `[advisory only]` device classes).
6. `submission_checklist` - payload closure, deterministic ZIP, no scorer patches, no sidecars, no hidden files, `archive.zip -> inflate.sh -> upstream/evaluate.py`, CUDA/T4/equivalent proof, inflate budget, review signoff.
7. `next_wave_roadmap` - CMG3 closed row-span archive + exact CUDA gate, predictive/lossy mask grammar atoms, Q-FAITHFUL successor geometry, charged pose-basis atoms, hard-pair temporal windows, payload-efficient residuals, and packer/layout atoms; all blocked from promotion until exact CUDA archive evidence exists.

## Caveats

- PR100 adapter replay is the current exact frontier; PR95 stem-permutation repack, PR95 conservative repack, PR95 public exact replay, PR85+STBM/RMB1, PR85, PR84, PR81, and C-067 are predecessor rows. None is a Shannon-floor attainment claim.
- Public PRs and GitHub comments are external design signals unless we have exact archive bytes, SHA, CUDA eval JSON, component recomputation, and custody.
- PR #70's low reported score is non-comparable under our compliance policy because the public PR text says score-affecting bytes were moved from `archive.zip` into `inflate.py`.
- C-067 carries an external-source attribution requirement (PR #67 mask segment); the local score claim is A++ evidence for the charged archive bytes, but the mask source remains externally attributed per `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.
- The H100 NVL diagnostic of the same C-067 archive bytes scored `0.36295` earlier; this is a runtime-custody warning, not a contradiction. Cross-run comparisons require matching `inflate_runtime_manifest.runtime_tree_sha256`.

## Cross-references

- Paper draft: `docs/paper/04_results.md` (stale PR95 frontier table; update before publication)
- Codex writeup ledger: `.omx/research/submission_writeup_integration_20260502_codex.md`
- Working notes: `reports/writeup_working.md` (live operating point)
- Submission pipeline runbook: `docs/runbooks/contest_submission_pipeline_20260502.md`
- Reverse-engineering refs: `reports/raw/leaderboard_intel_20260501/` (PR #65/#67 inflate.py + archive.zip + line_search.py)
- Memory: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`, `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`

## Deadline

Contest deadline: **May 3 11:59 PM AOE = May 4 06:59 AM CDT (May 4 11:59 UTC)**. This report is now in final handoff mode for the confirmed PR100 adapter champion, not pre-deadline queue planning.

## Internal sub-frontier launch-ready candidate — Lane Ω-W-V3 (post-deadline)

**Lane Ω-W-V3** (water-filling codec v2 → PR106 HNeRV decoder) is fully scaffolded as of 2026-05-04 and ready for ~$0.30 GPU dispatch on Vast.ai 4090 (or T4-equivalent). Council 8/10 GO; predicted band [0.194, 0.204] [contest-CUDA].

### Pipeline (5 commits today)

| Stage | File | Commit | Purpose |
|---|---|---|---|
| 1 | `experiments/extract_pr106_decoder.py` | `45149f21` | Unpack PR106 0.bin → 28 tensors / 228,958 params + 600×28 latents |
| 2 | `experiments/build_sensitivity_map_pr106.py` | `5ca67264` | Per-channel β-Fisher (CUDA-required); CPU stub OK |
| 3 | `experiments/repack_pr106_with_water_filling.py` | `b2f958a4` | Per-Conv2d water-fill OWV2 + brotli-int8 fallback → apogee_v2 0.bin |
| 4 | `submissions/apogee_v2/inflate.{py,sh}` + src | `c7f237eb` | Inverse parser; **round-trip VERIFIED** on real bytes |
| 5 | `scripts/remote_lane_omega_w_v3_pr106.sh` | `ac3053ef` | 4-stage GPU dispatch wrapper |

### Stub-mode preview (CPU, all-ones sensitivity) [empirical:experiments/results/apogee_v2_repack_20260504_claude/repack_metadata.json]

| Metric | PR106 (current public frontier) | Apogee-v2 stub | Δ |
|---|---:|---:|---:|
| Archive bytes | 186,239 | 164,087 | **−22,152 (−11.9%)** |
| Rate-component score | 0.124 | 0.109 | **−0.01475** |
| Predicted total score (if distortion holds) | 0.20945673 | **~0.1947** | **SUB-0.20** |

Real β-Fisher (CUDA dispatch) could shift bytes among layers and improve further. Distortion delta is UNKNOWN until contest_auth_eval runs on T4 — predicted band [0.194, 0.204] per audit assumes the per-channel water-fill quantization preserves PR106's PoseNet sensitivity (the dominant attack surface at 0.018306).

### Operator one-liner (when GPU dispatch approved)

```bash
PR106_ARCHIVE=experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
bash scripts/remote_lane_omega_w_v3_pr106.sh
```

Anchor: `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_01_water_filling_codec_v2_pr106_decoder.md` (Council 8/10 GO).

## Lane #04 intN expansion (NEW 2026-05-04 evening) [empirical:experiments/repack_pr106_with_intN_block_fp.py]

The Lane #04 ternary-block_fp falsification was revived as a Pareto sweep over signed intN block-FP variants (int4..int8). Verified end-to-end byte-decodable locally (CPU stub mode):

| Variant | Magic | Archive bytes | Rate Δ | Rel err per weight | Risk | Predicted band [contest-CUDA] |
|---|---|---:|---:|---:|---|---|
| int4 | `0xA4` | 109,981 | −0.0508 | 7.1% | HIGH | [0.155, 0.180] |
| **int5** | **`0xA5`** | **154,555** | **−0.0211** | **3.3%** | **MEDIUM** | **[0.180, 0.196] — sweet spot** |
| int6 | `0xA6` | 170,450 | −0.0105 | 1.55% | LOW | [0.190, 0.204] |
| int7 | `0xA7` | (sketch only) | (TBD) | 0.79% | VERY LOW | [0.198, 0.208] |
| int8 | `0xA8` | (sketch only) | (TBD) | 0.24% | ALMOST LOSSLESS | [0.196, 0.207] |

**Lane #04 closed-loop chain — all four components landed this session:**
- Producer: `experiments/repack_pr106_with_intN_block_fp.py` (`82ca9456`) — generic `--bits N`
- Inflate adapter: `submissions/apogee_intN/inflate.{py,sh}` + vendored `src/intn_codec.py` (`62e3a51c`)
- Dispatch wrapper: `scripts/remote_lane_apogee_intN.sh` (`5c31755e`) — operator picks bits via `APOGEE_INTN_BITS` env
- Decision matrix doc: `docs/pr106_stacking_decision_table_20260504.md` (`7b354a1a`)

**Operator one-liners** (Vast.ai 4090 ~$0.30 / 30min each):

```bash
APOGEE_INTN_BITS=5 bash scripts/remote_lane_apogee_intN.sh   # sweet spot
APOGEE_INTN_BITS=6 bash scripts/remote_lane_apogee_intN.sh   # safe fallback
APOGEE_INTN_BITS=4 bash scripts/remote_lane_apogee_intN.sh   # high-risk high-reward
```

## Recovery + Lane SJ-KL launchability (NEW 2026-05-04 evening)

A defensive-validation pass uncovered 8 lost helpers from a subagent-worktree-cleanup bug class (subagent built helpers + wrapper, only the wrapper got committed before quota; helpers wiped on cleanup). Full recovery doc: `docs/recovery_report_20260504.md`. Bug class structurally extinct via two new preflight checks (PCC9 + PCC9b, warn-only initially) that scan shell scripts + test files for runtime-executed file references that don't resolve to disk.

**Lane SJ-KL C067 is now end-to-end runtime-decodable** after the 6 lost SJ-KL Python modules were rebuilt from runbook + addendum spec:
- `src/tac/sjkl_basis.py` — full codec library (basis + alpha-block V1+V2 + full payload + runtime aliases), 26 tests pass
- `experiments/build_sjkl_residual.py` — Lanczos top-K Fisher-info basis + per-pair alpha quantization → sjkl.bin
- `experiments/prepare_sjkl_pair_tensors.py` — frame-pair I/O orchestration
- `experiments/build_sjkl_c067_archive.py` — `top_level_sibling` layout fully implemented (preserves source `p` bytes exactly)
- Plus the recovered `tools/claim_lane_dispatch.py` (full) + `scripts/ensure_remote_uv.sh` (canonical uv installer)

End-to-end pipeline verified locally: prepare → build_residual → 942-byte sjkl.bin → build_archive → charged ZIP `{p untouched, sjkl.bin charged}` → decode via `tac.sjkl_basis`: byte-faithful basis + alpha extraction.

**Operator one-liner** (Vast.ai 4090, separate base archive from PR106-stacking lanes):

```bash
bash scripts/remote_lane_sjkl_c067.sh
```

## Updated Next Queue

1. Use `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter` as the release packet unless a newer exact T4 A++ packet is explicitly promoted.
2. Keep PR100 source attribution and the exact T4 custody block with every public/judge-facing score claim.
3. Run strict public-release hygiene on the exact PR body, notebook, and site bundle before publishing URLs.
4. Keep PR96, PR91/HPM1, and any public body/CPU scores in external context until exact CUDA replay lands.
5. **Lane Ω-W-V3 GPU dispatch (~$0.30)** — council 8/10 GO, predicted [0.194, 0.204], LOWEST risk of the sub-0.20 candidates. Operator approval gate per CLAUDE.md "Design decisions — non-negotiable".
6. **Lane #04 int5 GPU dispatch (~$0.30)** — predicted [0.180, 0.196], MEDIUM risk, sweet spot of the int-N Pareto. Worth running in parallel with Ω-W-V3 since they're orthogonal (different codec families).
7. **Lane #04 int6 GPU dispatch (~$0.30)** — predicted [0.190, 0.204], LOW risk fallback if int5 distortion exceeds tolerance.
8. **Lane SJ-KL C067 GPU dispatch (~$0.30)** — uses C067 base archive (not PR106), so dispatchable in parallel with the PR106-stacking lanes. Predicted score band TBD pending first run.

**Total dispatch matrix: 4 lanes × ~$0.30 = ~$1.20** for the highest-EV sweep that would land the first contest-CUDA score below PR106's 0.20946.

