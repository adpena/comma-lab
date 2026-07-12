---
title: "Frontier overhead-stack — the NON-click rare-lever sweep on the PR110-lineage frontier payload"
authority: "[macOS-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19108282 [contest-CPU]; $0; CPU-only; no PR; Modal MODAL-HOLD"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-07-11
subagent: overhead-stack
verdict: ALL_FOUR_NON_CLICK_LEVERS_RED_OR_DONE_AT_MEASURED_OPTIMAL_FORM; SIDECAR_AT_INTEGER_GRID_BREAK_EVEN_LAW_REGISTERED
sibling_boundary: "click-polish latent search (task #399, click_polish_block_loop) is OWNED by the sibling; this agent never ran click search — only the deterministic NON-click levers below."
cross_refs:
  - .omx/research/pr128_intake_reverse_engineering_20260710.md
  - .omx/research/p_suff_task_ablation_verdict_20260619.md
  - experiments/results/frontier_exact_bitalloc_solve/rd_curve.jsonl
  - experiments/results/frontier_int5_qat_finetune/calibration_ptq_probe.json
  - src/tac/canonical_equations/click_polish_byte_neutral_slack_20260711.py
  - experiments/results/frontier_overhead_stack_run1/rows_final.json
  - experiments/results/frontier_overhead_stack_run1/rows_repair.json
  - experiments/results/frontier_overhead_stack_run1/rows_repair2.json
---

# Frontier overhead-stack — the rare NON-click levers on the 0.19108282 frontier

## TL;DR (honest headline)

**All four non-click overhead levers are RED or DONE on this exact vehicle, each with a
measurement or a cited prior measurement — none composes a winning candidate today.**
The one lever that was still open (lever 1, SIDECAR-FOLD, PR128-proven on its own stack)
was driven to per-pair-exact OPTIMAL FORM and is **RED by +2.75e-5**: the 607-byte PR101
latent sidecar sits almost exactly at integer-grid break-even — its sub-grid corrections
buy 4.16e-6 d_seg (= +0.000416 S if folded optimally to the integer grid) against the
−0.000404 rate win from deleting the section. **The composed best-stack therefore remains
the sibling's click candidate unchanged (advisory 0.19101380).** The actionable yield: the
fold becomes FREE money the moment the sibling's round-2+ click search runs ON THE FOLDED
TABLE (any search gain > 2.75e-5 — round-1 already found 7.9e-5 on 48 pairs — banks the
−607 B on top). Registered as canonical equation
`pr110_lineage_sidecar_fold_integer_grid_boundary_v1`.

All numbers `[macOS-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19108282 [contest-CPU].

## Borrowed-substrate accounting (NO-FAKE #7)

- **Substrate:** OURS — the PR110-lineage frontier archive (`pr110_payload_entropy_recode_20260610`,
  sha `b4689726…`, 177,169 B; member `x` = FP11 → CTXR: decoder 161,104 B [PR95/#101 frozen] +
  latent 15,070 B + **sidecar 607 B** + FECa selector 222 B + DQS1 tail 42 B). Decoder=PR95/#101;
  entropy=PR112 `codec_ctx` (mattneel, MIT); selector/DQS1=our PR110.
- **Mechanism (lever 1):** the sidecar-fold *idea* is PR128's (a12dongithub, rhnerv_latent_polish,
  MIT, [external unverified]). The fold-to-integer-step realization, the per-pair 4-option
  exact-selection repair (novel here), and the byte-closed FP11/CTXR splice-repack are OUR harness
  (`tac.click_polish.FrozenPacket`, `repack_archive_bytes(drop_sidecar=True)`).
- **Classification:** DEFENSIVE BANK, not innovation. Code lifted from PR128 tree: NONE (METHOD
  reference only). No unlicensed HNeRV/NeRV code touched.

## The grammar (SOURCE-VERIFIED by parsing the bytes)

`member[x]` = `FP11 | u32 src_len | CTXR(...) | u16 sel_len | FECa selector | DQS1 tail`;
`CTXR = magic | u8 ver | u24 dec_len | u24 lat_len | u24 sidecar_len | dec_sec | lat_sec | sidecar`.
The 607-B sidecar (`SIDECAR_HUFF_ENUM_LEN`, enum-rank canonical-Huffman form — already the smallest
of the 6 PR101 sidecar encodings) stores per-pair `(dim, δ·100)` latent corrections,
`δ ∈ {±1,±2,±3,±4,±5,±6,±8,±10}/100`, applied as `latents[pair,dim] += δ/100` at inflate.
**597/600 pairs carry a correction** (3 no-ops). Grid scales: 0.0151–0.0283/step (mean 0.0197),
so corrections are 0.35–5.6 grid steps (median 1.26).

## Per-lever ΔS table (all `[macOS-CPU advisory]`, n600, same harness as #399)

| lever | candidate | bytes | d_seg | d_pose | S | ΔS vs base | verdict |
|---|---|---|---|---|---|---|---|
| — base (incumbent, calibration) | sha `b4689726…` | 177,169 | 0.00055989 | 2.942e-5 | **0.19110945** | — | reproduces #399's 0.19109312 within 1.6e-5 (chunk-layout fp; same harness/GT) |
| L1a fold-naive (round-to-nearest, drop sidecar) | `candidate_lever1_fold.zip` sha `f5f4b8b2…` | 176,562 (−607) | 0.00056718 | 2.946e-5 | 0.19144690 | **+3.37e-4** | RED (naive) |
| L1b fold-opt3 ({near,zero,other} per-pair exact) | superseded | 176,562 | 0.00056416 | 2.947e-5 | 0.19114860 | +3.91e-5 | RED |
| L1c fold-opt4 (+far; OPTIMAL FORM) | `candidate_lever1_foldopt.zip` sha `3c43cd26…` | 176,562 | 0.00056405 | 2.947e-5 | 0.19113692 | **+2.75e-5** | **RED at optimal form** |
| L2 dominated-byte deletion | — (no admissible set) | — | — | — | — | — | **RED, cited** (#153: 0/28 tensors, 0/28 dims deletable; best case +0.078 ΔS_task) |
| L3 bit-alloc / re-quant | best RD point (mean-bits 6.5) | 155,409 | 0.00174 | 1.2e-4 | 0.313 | +0.12 | **RED, cited** (full KKT waterfill sweep `rd_curve.jsonl`; binding 90% of decoder at bit-floor=8 per #153; PR128 concurs: strict discrete optimum) |
| L4 lossless recode sweep | — | — | — | — | — | — | **DONE** (frontier IS the PR112 ctx recode; all sections at ~8 bits/B; sidecar already minimal-form; single ZIP_STORED member) |
| composed naive (sibling clicks + fold-naive) | `candidate_best_stack_click_plus_fold.zip` | 176,562 | 0.00056652 | 2.946e-5 | 0.19138227 | +2.73e-4 | RED — composition does not rescue the fold |
| **composed BEST (deliverable)** | = sibling `candidate_archive.zip` sha `08720866…` UNCHANGED (click round: run1 pairs 0-47, 37 clicks) | 177,169 | — | — | **0.19101380** (sibling's own n600 authority pass) | **−7.9e-5** | the only winning stack today |

## The measured law (lever 1, the durable yield)

**The PR101 607-byte sidecar on this vehicle is a nearly break-even-efficient section at the
integer-grid boundary:** deleting it saves ΔS_rate = 25·607/37,545,489 = 0.000404, while its
sub-grid (δ/100-precision) corrections are worth 0.000416 of d_seg that NO integer-grid fold can
reproduce — measured at per-pair exact optimum over 4 integer options (2,388 option-renders,
choices 439 near / 20 zero / 121 other / 17 far; selection exact by pair-locality, means decompose).
Margin: **+2.75e-5 against the fold.** Three corollaries:

1. **Fold-without-re-search is RED here** (verdict_scope: FORMULATION — deterministic fold on this
   vehicle; NOT the PR128 family verdict). PR128's fold paid because it ran inside a full
   exact-gated click search that re-optimized the whole table post-fold.
2. **Fold + click-search IS the winning composition:** run sibling round-2+ ON THE FOLDED table
   (`drop_sidecar=True` renderer + `repack(drop_sidecar=True)`); any recovered gain > 2.75e-5 banks
   the −607 B for free. Round-1 found 7.9e-5 on 48 pairs; 552 pairs remain.
3. Byte fact: the AR latent-coder length is INVARIANT to all fold variants tried (every candidate =
   176,562 B exactly = base − 607) — the entire saving is the deleted section; the moved codes are
   rate-free.

## Levers 2/3 — why no eval was spent (measurement-first, cited)

- **#153 P-SUFF** (`p_suff_task_ablation_verdict_20260619.md`, THIS vehicle, realized-through-R):
  zero deletable tensors/dims; the six big weight tensors (~90% of the decoder) at bit-floor 8;
  joint sub-int8 cut saves 1,156 B at +0.0258 ΔS_task. "The binding mass is at b8, do not spend
  the budget re-quantizing it."
- **`frontier_exact_bitalloc_solve`** (2026-06-19): the KKT reverse-waterfill (per-tensor frozen-scorer
  g_seg/g_pose sensitivities) full RD sweep, mean-bits 4.0→6.5 — best point S=0.313 ≫ 0.191.
  **`frontier_int5_qat_finetune`**: int5-LSQ 0.486 (vs absmax 0.710) — better re-quant, still
  catastrophic. PR128 independently: all 229K weight-code clicks rejected (strict discrete optimum).
  Re-running any of this would be rediscovery (the cardinal signal-loss sin).

## Artifacts

- `experiments/results/frontier_overhead_stack_run1/rows_final.json` — 3 measured n600 rows
  (base / fold-naive / composed-naive) + shas + per-row components.
- `rows_repair.json` / `rows_repair2.json` — 3-option and 4-option exact-selection results.
- `perpair_state.npz`, `repair_state.npz`, `repair2_state.npz` — per-pair d_seg/d_pose for
  base + 4 fold options (the raw measurement; resumable-protocol state).
- `candidate_lever1_fold.zip` (naive, sha `f5f4b8b2…`), `candidate_lever1_foldopt.zip`
  (optimal-form, sha `3c43cd26…`), `candidate_best_stack_click_plus_fold.zip` (composed-naive),
  `sibling_click_candidate_COPY.zip` (untouched copy of the composition base).
- NO new MODAL-HOLD entry staged: no candidate of mine beats the sibling's already-staged one
  (`click_polish_399_run1/staged_exact_eval_queue_MODAL_HOLD.json` remains THE exact-eval entry).

## Operational note (harness discipline, for future agents)

Detached background evals (nohup AND `start_new_session`) were killed silently ~5 min in (SIGURG-144
class); `pgrep -f` matched the waiter shells' own command strings, masking the death. The working
protocol — per memory L47 — is CHUNKED RESUMABLE FOREGROUND: ≤235 s work-budget per invocation,
per-slice npz state with atomic replace, rc=10 continue / rc=0 done. 6,000+ pair-renders landed
this way with zero loss.

## Pointer honesty

Pointer UNMOVED at contest-CPU **0.19108282**. This unit produced measured negatives + one durable
law + the fold-on-folded-table composition path; it did NOT lower the exact score. The only
score-moving path staged remains the sibling's MODAL-HOLD entry (operator GO).
