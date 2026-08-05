---
arm: ddm_ar2
title: "arXiv 2602.22432 LoBoost crosswalk"
utc: 2026-08-05
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[paper-crosswalk scorer-free]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# AR2 Receipt - LoBoost Crosswalk

## Answer First

arXiv `2602.22432` is **LoBoost: Fast Model-Native Local Conformal Prediction for Gradient-Boosted Trees** by Vagner Silva Santos, Victor Coscrato, Luben M. C. Cabezas, Rafael Izbicki, and Thiago R. Ramos. arXiv reports v1 submitted 2026-02-25 and v2 last revised 2026-08-03.

Verdict for Pact: **ADOPT only the model-native locality / calibration-size discipline as a scorer-free diagnostic.** LoBoost does not provide a new renderer, pose solver, token coder, byte stream, or score-lowering mechanism. The useful transfer is a rule: reuse the model's own induced partition, require enough local calibration mass, and refuse local conclusions when the cell is too sparse or heterogeneous.

No scorer, no launch, no archive mutation, no `upstream/evaluate.py`, and no pointer move occurred.

## Paper Read

Sources fetched:

- `https://arxiv.org/abs/2602.22432`
- `https://arxiv.org/pdf/2602.22432`
- `https://arxiv.org/html/2602.22432`

Core mechanism, paraphrased from the abstract, introduction, theory, algorithm, and experiments:

- LoBoost represents each point by the sequence of gradient-boosted-tree leaves it visits.
- Matching prefixes of those leaf paths define nested local groups.
- Residual quantiles are estimated inside those groups, with a global fallback.
- The method is post-fit: it reuses the fitted ensemble and calibration residuals rather than training an auxiliary partition model.
- The governing tradeoff is local homogeneity versus enough calibration samples. The paper states a finite-sample coverage-error term controlled by within-cell score-distribution discrepancy plus terms that shrink with local calibration count.
- The algorithm enforces a minimum local count `Nmin = max(Npart, ceil(pmin * ncal))`, splits by leaf path, merges undersized terminal regions, and computes per-region conformal quantiles.
- Experiments report LoBoost beating ICP SMIS on 7/11 datasets, tying on 2, losing on 2, and improving worst-slab coverage on all 11 datasets, with sensitivity runs showing the resolution/runtime/coverage tradeoff across `Npart` and `pmin`.

## RECALL EVIDENCE

| source / query | found | impact |
|---|---|---|
| `.omx/tmp/codex_runs/ar2_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md` | AR2 owns a paper crosswalk only: no scorer slot, no launch, no archive, write this receipt, serialize with `[no-triality] [p0-ledger-ok]`, final own line unchanged. | Kept work scorer-free and receipt-only. |
| `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Current own line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved; live fleet includes `ar2`; TP1 birth A/B is burning; PE3 reads conditioning-only; SMEVR is not the token-bulk bar. | Crosswalk is not score progress and cannot reroute the active scorer/trainer lane. |
| `rg "2602\.22432|LoBoost|leaf prefix|local conformal|boosted.*conformal" .omx/research .omx/state docs src tools` | Direct hit only in AR2 queue/hot-state records, plus one unrelated older "local conformal metric" phrase. No prior LoBoost receipt or paper-checked memo found in this scope. | Treated this as first direct LoBoost crosswalk, but not as first use of locality/verdict hygiene. |
| `rg --files .omx/research .omx/state | rg 'papers_checked|graph_memory|canonical_research_index|ledger'` | Existing paper-checked ledgers, graph memory, canonical indexes, and ledgers exist; no LoBoost-specific file appeared by name. | Used local current receipts for actual Pact facts instead of importing paper novelty blindly. |
| `tools/list_canonical_equations.py --json` | Adjacent equations observed for token/codecs, pose nullity, trajectory stopping, and Brotli cascades; no LoBoost/local-conformal Pact equation overriding live receipts. | No canonical equation change claimed. |
| `ddm_tk1_20260805/TK1_RECEIPT.md` | PE3 conditioning and cheapdct4 accounting consumers landed default-OFF; no scorer or launch; PE3 is conditioning only, not label replacement. | LoBoost's "reuse model-native structure" is already embodied as conditioning/trust-gate style, not direct target replacement. |
| `ddm_lc1_20260805/LC1_RECEIPT.md` | LC1 measured direct PE3 target labels worsening all n32 pairs, `net_fixed = -12,884`; TR1 learned carrier crowned primary. | Any LoBoost-style local partition cannot resurrect PE3 labels-as-targets. |
| `ddm_tp1_20260805/TP1_PACKET.md` and `.omx/state/main_hot_state.md` | TP1 sealed BI1 birth ON/OFF; hot state says OFF full arm is live and ON is queued; adoption needs n32-stratified delta plus R8 pose accounting at composed stage. | AR2 must not launch or modify TP1; it can only add diagnostic guidance for future reads. |
| `ddm_xo1_20260805/XO1_RECEIPT.md`, `ddm_sv2_smevr_base_rule_race_20260803.md` | XO1 semantic/control ordering worsened IX2 token bytes by `+11,561 B`; oracle flip order worsened by `+11,838 B`; SMEVR loses to shipped IX2/Brotli-Q11 by `+5,183 B`; token wins are match-structure wins, not semantic-order wins. | LoBoost prefix grouping is N-A for token bytes unless a same-coder match-structure preflight wins. |
| `ddm_p4x_lane_existence_birth_matrix_20260803.md`, `ddm_sq1_eta_seg_and_hinge_ab_20260803.md`, `ddm_sq2_20260804/SQ2_RUN_MEMO.md` | Prefix samples can be the wrong subset; SQ1 used non-prefix stratified n32 and proved pose collateral is mandatory; SQ2 improved seg eta but failed R8 pose-bank accounting. | LoBoost's minimum-cell/local-homogeneity rule maps to verdict hygiene, not promotion. |
| `ddm_p3v2_optimal_form_pose_resolve_20260729.md` plus hot-state DQ1 correction | Free-frame pose is budget-conditional and solver/terminal-pose specific; hot state routes sl1 to confirm tail pairs. | LoBoost does not change terminal pose mechanics, only possible sampling/uncertainty reporting. |
| `canonical_research_index_dseg_20260629.md`, `.omx/state/main_hot_state.md` | Margin-hinge/tau surfaces are local loss/gradient engineering; current hot state says margin-hinge -> tau_softplus build debt remains owed. | LoBoost is post-fit calibration, not a replacement loss. |

## Ranked Crosswalk

| rank | disposition | Pact surface | LoBoost transfer | honesty label | named consumer | falsifier / stop rule | cost |
|---:|---|---|---|---|---|---|---|
| 1 | ADOPT | Verdict hygiene for TR1/PE3/IX2/correction reads | Use native Pact partitions as strata: token lattice cells, PE3 component modes, TR1 support buckets, per-edge cells, and pair strata. For each local verdict, carry both local count and residual homogeneity instead of a single global mean. | CONJECTURE from paper plus MEASURED local m88/m96 need | `gr1` stratified rerace, future PE3/LC-style ceiling reads, future TR1 birth A/B read | If local strata do not reduce worst-stratum spread versus the existing stratified-random selector, or if prefix/scene-block bias reappears, fold the diagnostic. | $0, scorer-free over existing receipts/caches; no launch |
| 2 | ALREADY-EMBODIED | PE3/TR1 conditioning | The paper's strongest engineering lesson is "reuse the fitted model's own partition without training an auxiliary partition." TK1 already implements PE3 as conditioning/trust gates, and LC1 already forbids PE3 labels-as-targets. | MEASURED/Built locally | TK1 PE3 conditioning, TP1 successors | Matched n32 PE3-conditioning ON/OFF harms d_seg, pose, or rate after R8 accounting; direct labels-as-targets remain refused by LC1. | Already paid as TK1 build; next cost is ordinary TP1/TK1 A/B accounting |
| 3 | ADOPT | Local cell-size guard | Mirror LoBoost's `Nmin` idea as a Pact admission rule: no per-cell or per-mode conclusion is banked unless support is large enough and local residual spread is printed. This is not a numeric import of `Npart/pmin`; it is the structural guard. | DERIVED | m88/m96 sampling hygiene, LC1/RZ1 label ceilings, PE3 mode reads, edge-conditioned Lane reads | If applying the guard collapses all useful cells to global or positive controls fail, record VACUOUS/underpowered rather than a negative. | $0 metadata pass; no scorer |
| 4 | N-A / FOLDED for bytes | Token coder races | LoBoost leaf-prefix grouping is not a byte coder. Pact evidence says semantic/orderer quality can worsen IX2/Brotli bytes because live token economics are LZ match-structure dominated. | MEASURED local blocker | en1 context race, any XO1-like orderer | Any future LoBoost-inspired token grouping must first increase same-coder match structure and beat shipped IX2/Brotli-Q11 bytes; otherwise refuse before scorer. XO1 and SV2 are current blockers. | No action now; optional $0 byte-only match-structure preflight if a concrete candidate exists |
| 5 | N-A for pose mechanics | Terminal 6-eq pose solve and free-frame pose bound | LoBoost quantiles can describe uncertainty over pose-tail samples, but they do not provide a PoseNet inverse, 6-equation terminal solve, free-frame actuator, or joint-descent mechanism. | DERIVED from mismatch plus local pose receipts | sl1 only as sample/uncertainty annotation, not as solver design | If a row cites LoBoost as pose solution or post-hoc sidecar cure, refuse. Pose claims still require actual frozen PoseNet measurements and R8-at-composed-stage accounting. | $0 annotation only |
| 6 | ADOPT narrowly / ALREADY-EMBODIED | Correction stacks under joint pricing | LoBoost supports checking residual-score homogeneity inside model-native regions before trusting local corrections. It does not override the requirement that frame edits be jointly priced across Seg, Pose, and bytes. | MEASURED local joint-pricing blockers plus CONJECTURE diagnostic | sl1, v19-family joint-priced correction stacks, RGB/chroma boundary finisher reads | If a local correction improves seg but violates R8 pose-bank or same-archive bytes, it remains non-adoptable. SQ1/SQ2 are the current example. | $0 diagnostic over cached per-pair rows before any scorer spend |
| 7 | N-A for loss design | Margin-hinge / tau_softplus engineering | LoBoost is post-fit conformal calibration. It does not change gradients, margin-hinge wiring, tau_softplus, Muon, or the TR1 loss schedule. At most it can set per-stratum acceptance bands for an A/B read. | DERIVED | en1 margin-hinge -> tau_softplus build/read | If presented as a new loss, refuse; the real debt is wiring and matched A/B, not a paper-derived surrogate. | $0 read discipline; build debt remains elsewhere |

## What AR2 Did Not Measure

- No SegNet or PoseNet scorer forwards.
- No `upstream/evaluate.py`.
- No archive bytes, token bytes, or candidate archive.
- No n32/n120/n600 Pact measurement.
- No code edit, no trainer launch, no lane claim, no GPU/MLX job.
- No new canonical equation.

## NEXT_IF_RESUMED

```json
{
  "schema": "ddm_ar2_next_if_resumed.v1",
  "status": "PAPER_CROSSWALK_COMPLETE_NO_SCORE",
  "paper": "arXiv:2602.22432 LoBoost",
  "adopt": [
    "model_native_locality_diagnostic",
    "minimum_local_support_and_homogeneity_guard",
    "per_stratum_acceptance_bands_for_future_reads"
  ],
  "already_embodied": [
    "PE3_conditioning_only_trust_gate_style",
    "TR1_learned_carrier_primary_after_LC1"
  ],
  "refuse_as_solution_for": [
    "token_bytes_without_same_coder_match_structure_win",
    "terminal_pose_solve",
    "post_hoc_pose_sidecar",
    "margin_hinge_tau_loss_wiring",
    "joint_priced_correction_admission"
  ],
  "scorer_forwards": 0,
  "evaluate_py": false,
  "archive_mutation": false,
  "pointer_moved": false,
  "next_fire_order": "If reused, implement only a $0 cached diagnostic that reports native-cell count, residual spread, and prefix/stratified selection mode before any scorer spend."
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
