# PR110++ R2 — non-MPS per-mode effect table + alternative-selector candidate

**Subagent:** `pr110pp_r2_nonmps_mode_table_20260609` · UTC 2026-06-09 · $0-local CPU-exact phase.
**Lane:** `lane_pr110pp_r2_nonmps_mode_table_20260609`.
**Axis discipline:** everything below is `[macOS-CPU advisory]` / candidate-generator. No score,
promotion, rank/kill, or dispatch authority is minted here. The final paired contest-CPU eval is
operator-dispatched (R1). CUDA never touched; **MPS never used** (CPU forced via `torch.device("cpu")`).
**Provenance discipline:** every load-bearing claim cites `{file, field, observed_value, surface,
reproduce_command}`. Predecessor: `pr110pp_frontier_direct_research_20260609` (assessment
`.omx/research/pr110pp_frontier_direct_assessment_20260609.md`, commit `25fc7b5ca`).

---

## 0. What R2 executes (and the one correction to the assessment)

R2's job (per the assessment's rank-2 $0-local-only fallback, escalated to also build R1's $0 half):
1. Rebuild the per-mode effect table on **macOS-CPU (NOT MPS)** — the MPS substrate
   (`frame_exploit_*_mps600_codex/pair_component_rows.jsonl`) is non-promotable noise (PoseNet drift 23×).
2. Build **ONE alternative-selector candidate** (argmin per-pair pose mode), byte-closed, no-op-proven,
   equal-or-fewer selector bytes.
3. Emit the honest advisory delta + the **READY-FOR-R1** exact paired contest-CPU eval packet.

**Correction to the assessment's byte budget.** The assessment said the attack surface is an "879-byte
sidecar" containing the K=16 selector. The byte-exact parse of member `x` shows the 879 figure conflates
THREE distinct regions plus FP11 framing. The real grammar of member `x` (178,393 B, stored/method-0) is:

```
[FP11 magic 4B][source_len u32 = 178121][source_payload 178121B][selector_len u16 = 220][FECa selector 220B][DQS1 tail 42B]
```
- `source_payload` (178,121 B) is itself `[decoder 162,127B][latent 15,387B][PR101 latent-correction sidecar 607B]`.
- **The actual selector attack surface is the 220-byte FECa stream** at offset 178,131 in `x`
  (179 + 8 + 178121 + 2). NOT 879. The 879 = 607 (latent-correction sidecar) + 220 (FECa selector)
  + 42 (DQS1 tail) + 8 (FP11 header) + 2 (selector_len) — three orthogonal regions, only one of which
  (the 220B FECa selector) is the PR110++ per-pair mode lever.
- **The frontier selector is FECa (FEC10 hybrid adaptive-blend Markov arithmetic coder), not FEC6
  fixed-Huffman.** The assessment cited `FEC6_FIXED_K16_MODE_IDS`; the runtime dispatches `FECa` via
  `decode_fec10_hybrid_selector`. The 16-mode *vocabulary* IS `FEC6_FIXED_K16_MODE_IDS` (the codes index
  into it), but the *coder* is FECa. Reproduce: `parse_frontier.py` → `fp11_grammar.selector_magic = "FECa"`.
- There is ALSO a `DQS1` decoder-q patch tail (42B): patches one decoder q-value (storage_index 26,
  q_offset 0, delta +1) on 31 selected pairs under `pair_all_frames` policy. **Independent of the
  selector; left byte-identical in the candidate.**

Provenance: `{file: <submission>/archive.zip, field: member_x FP11 grammar, observed_value:
source_len=178121/selector_len=220/dqs1_tail=42/selector_magic=FECa, surface:
zipfile+struct.unpack_from, reproduce_command: .venv/bin/python <workdir>/parse_frontier.py}`.
Archive sha256 `b7106c9bdbb8…8997c8c` verified on disk (178,493 B).

---

## 1. The per-mode effect table (CPU-exact, non-MPS)

**Method.** For each sampled pair: render the receiver comp frames (frontier HNeRV decoder + DQS1 patch
where applicable + bicubic upscale to 874×1164 + the −1 channel postprocess + clamp/round) **with NO
selector** (identity baseline render), then apply each of the 16 frame-0 modes to frame-0 and score the
16-mode batch through the **EXACT `upstream/modules.py` DistortionNet on CPU** in one call (B=16). This
is byte-for-byte the same scorer the contest uses; the only difference from contest-CPU is the host
(macOS arm CPU vs Linux x86_64) — hence `[macOS-CPU advisory]`, not `[contest-CPU]`.

### 1a. SegNet-blindness — VERIFIED, not assumed

The 16 active modes are ALL frame-0 transforms (`frame_index==0` for every spec). SegNet scores only the
LAST frame of each pair (`upstream/modules.py:108` `x = x[:, -1, ...]`), so frame-0 transforms cannot
change the SegNet argmax. **Empirically confirmed:** `d_seg = +0.000e+00` exactly for all 16 modes vs
identity, on **all 67 sampled pairs** (`segnet_blindness_verified = true`; `nonzero_seg_findings == []`).
This is the mechanistic anchor: **the selector's only distortion lever is the PoseNet term.** d_seg=0 ⇒
the score-optimal per-pair mode is exactly `argmin pose_dist`. (Reproduce:
`build_mode_table.py --n 64` → `per_mode_summary_n64.json:segnet_blindness_verified = true`.)

> Note: the MPS source rows show the same seg-invariance *within* a pair, but a different *absolute*
> seg value (5.7983e-04 MPS vs 8.2397e-04 CPU-exact on pair 0) — MPS SegNet drift ~1.4× consistent with
> CLAUDE.md's ~2× SegNet MPS-drift note. The pose values diverge far more (§2).

### 1b. Headline (N=67 stratified sample; CPU-exact)

- `segnet_blindness_verified = true` (d_seg=0 exact for every frame0 mode on every pair).
- **64 of 67 pairs are "improvable"**: the exact-CPU argmin-pose mode ≠ the incumbent mode AND lowers
  pose. The incumbent selector (chosen by the FECa reparameterization, on the MPS substrate) is
  per-pair pose-suboptimal on the exact axis for 95.5% of sampled pairs.
- **Total sampled pose reduction (argmin vs incumbent) = 8.265771e-02** over 67 pairs.
- Incumbent per-pair pose: median 1.124e-03, max 1.436e-02 (frontier operating point, O(1e-3)).
- Reproduce: `per_mode_summary_n64.json:{n_pairs_improvable proxy via pair_summaries, total_argmin_pose_reduction}`.

### 1c. Which modes dominate pose — strongly per-pair

Two distinct rankings (the gap between them is *why a per-pair selector exists*):

- **Mean dpose_vs_identity (uniform-mode view, over 67 pairs):** `frame0_luma_bias_-1` is the only mode
  that beats identity on average (−7.69e-6); `frame0_roll_dx+0_dy+1` is catastrophic on average (+1.20e-2,
  ~200× the operating point). A uniform-mode policy would pick luma_bias_-1 or none.
- **argmin-pose usage (per-pair-best view):** `luma_bias_-4` (16), `luma_bias_-2` (13),
  `roll_dx+0_dy+1` (13), `rgb_bias_p0_p2_m2` (12) dominate. The SAME roll mode that is catastrophic on
  average is the per-pair optimum for 13 pairs (where the pair's ego-motion aligns with a 1-px vertical
  roll). This proves the pose lever is genuinely per-pair: no single mode is globally good, and the
  incumbent's frequent `none`/`blue_chroma` picks leave per-pair pose on the table.

---

## 2. R2 falsifiable test — does MPS mis-rank the modes? **YES (confirmed).**

The assessment's R2 falsifiable prediction: *"if the local exact-CPU d_pose mode ranking disagrees with
the MPS ranking, it proves the MPS substrate was mis-ranking modes (the root cause)."* Cross-check
(`crosscheck_mps_vs_cpu.py`, 67 pairs, 16 shared modes):

- **argmin-pose agreement rate = 4.48% (3/67 pairs).** The MPS substrate picks the same per-pair best
  mode as exact-CPU only 4.48% of the time.
- **mean Spearman rank-corr = 0.215** (median 0.247) — MPS per-mode pose ranking barely correlates with
  the exact-CPU ranking.
- **Verdict: "MPS substrate mis-ranks modes (R2 prediction confirmed)."**

Anchor example (pair 0, `frame0_blue_chroma_amp_1`): MPS pose = 2.30e-05 vs CPU-exact pose = 8.74e-05
(~3.8× divergence). The MPS substrate's per-mode pose values are drifted at exactly the operating point
(pose ~1e-3..1e-5) where the √(10·d_pose) marginal is largest. **This is the binding gap the assessment
named** ("the entire PR110++ menu cluster is built on a measurement substrate that cannot produce an
exact row"): the MPS substrate was mis-ranking the per-pair modes 95.5% of the time, so any selector
chosen on it is per-pair pose-suboptimal on the exact axis. The exact-CPU table here is the corrected
substrate. Reproduce: `crosscheck_mps_vs_cpu.json:{argmin_agreement_rate, mean_spearman_rank_corr}`.

---

## 3. The alternative-selector candidate (byte-closed, no-op-proven)

**Candidate dir:** `experiments/results/pr110pp_r2_nonmps_candidate_20260609/`
- `candidate_archive.zip` — sha256 `5facf0fb9b6ea558b4f589278192e3a958d84fe644090dae015fbacf914fa55d`,
  **178,493 bytes (identical to frontier — rate-neutral)**.
- `runtime/` — byte-identical inflate runtime (inflate.py sha matches frontier
  `c956bff6f1df…163dda90`); the candidate uses the SAME runtime, only archive bytes differ.

**Kind:** `subset_switch_argmin_pose_from_sample` — switch only the sampled pairs where exact-CPU
argmin-pose beats incumbent AND the FECa re-encode stays ≤ incumbent selector bytes (220). This is the
strict equal-or-fewer-byte deliverable backed entirely by exact-CPU evidence (no extrapolation).

- **Selector bytes: 220 → 220 (delta 0).** decoder/latent/latent-correction-sidecar byte-identical;
  DQS1 tail byte-identical. (`byte_closure_proof.json`.)
- **n_pairs_switched = 2** (pair 485: `none → frame0_roll_dx+0_dy+1`, exact pose gain 5.74e-3;
  pair 138: `frame0_blue_chroma_amp_3 → frame0_luma_bias_-2`, exact pose gain 1.10e-3). Total exact
  pose gain on switched = **6.842e-3**; linear pose-avg reduction over 600 = 1.14e-5.
- **No-op detector PASSED** (`noop_detector.json`): both archives inflate to 1200 frames;
  `inflated_frames_differ = true`; **5,621,484 raw bytes differ**; `consumption_proven = true`. The new
  selector bytes genuinely change the rendered frames (not a cosmetic repack).

### 3a. The binding rate-distortion finding (why only 2 switches)

The strict ≤220-byte constraint is *tightly binding* because the FECa FEC10 hybrid adaptive-blend Markov
arithmetic coder is fitted to the incumbent selection — deviations break its learned transition statistics
and cost bytes. The byte-vs-gain Pareto curve (top-k argmin switches by pose gain):

| switches | selector bytes | Δ bytes | cum exact pose gain |
|---:|---:|---:|---:|
| 0 (incumbent) | 220 | 0 | 0 |
| 1 | 221 | +1 | 1.414e-2 |
| 2 | 221 | +1 | 1.988e-2 |
| 12 | 227 | +7 | 5.523e-2 |
| 32 | 233 | +13 | 7.611e-2 |
| 64 (all improvable) | 247 | +27 | 8.266e-2 |

**The byte-neutral budget is the wrong Pareto constraint.** Switching ALL 64 improvable pairs costs only
**+27 selector bytes** = +27/37,545,489 × 25 = **+1.80e-5 in the rate term** (negligible), for a pose
reduction of 8.27e-2 (sample). Even the single top switch (+1 byte) captures 1.41e-2 pose gain. The
contest score is `100·d_seg + √(10·d_pose) + 25·bytes/37.5M`; trading a few selector bytes for large
pose reduction is overwhelmingly favorable. The strictly-compliant 2-switch candidate is the conservative
deliverable; the **+27-byte all-switch variant is the dominant lever** and is the recommended R1-PLUS
candidate (§4). Hook #2 Pareto: the 220-byte rate floor was never the binding constraint — the FECa
coder's resistance to re-selection is, and it is cheap to overcome.

---

## 4. READY-FOR-R1 packet (operator dispatch required)

**Packet:** `experiments/results/pr110pp_r2_nonmps_candidate_20260609/READY_FOR_R1.json`.

R1 is the one **operator-dispatched** half: run `upstream/evaluate.py --device cpu` on Linux-x86_64 (the
contest-CPU axis the 0.19199 lives on) for BOTH the frontier baseline and the candidate, then mint ONE
`tac.action_effect.v1` row. The candidate is byte-closed + no-op-proven; the only missing input is the
exact contest-CPU score (local macOS-CPU is advisory per CLAUDE.md, not 1:1 with the contest CI host).

Exact paired commands (real argparse verified from `tools/plan_dual_device_auth_eval.py`):
```bash
# Candidate (contest-CPU axis; requires operator dispatch auth + lane claim)
.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --archive experiments/results/pr110pp_r2_nonmps_candidate_20260609/candidate_archive.zip \
  --inflate-sh experiments/results/pr110pp_r2_nonmps_candidate_20260609/runtime/inflate.sh \
  --label pr110pp_r2_candidate_subset_switch --execute cpu \
  --lane-id lane_pr110pp_r2_nonmps_mode_table_20260609 --instance-job-id <OPERATOR_FILLS> \
  --json-out experiments/results/pr110pp_r2_nonmps_candidate_20260609/r1_candidate_plan.json

# Frontier baseline (same runtime, same contest-CPU host) — apples-to-apples
.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --archive <frontier>/archive.zip --inflate-sh <frontier>/inflate.sh \
  --label pr110pp_frontier_baseline --execute cpu \
  --lane-id lane_pr110pp_r2_nonmps_mode_table_20260609 --instance-job-id <OPERATOR_FILLS> \
  --json-out experiments/results/pr110pp_r2_nonmps_candidate_20260609/r1_frontier_plan.json
```
(`<frontier>` = the submission dir under `.omx/research/frontier_final_rate_attack_fp11_brotli_exec3_…/`.)

**Falsifiable prediction (per the assessment, sharpened):** the 2-switch candidate should land
`delta_score_total ≤ 0` on contest-CPU because d_seg is provably 0, rate is unchanged, and both switches
reduce exact macOS-CPU pose by construction. **Kill criterion:** `delta_score_total ≥ +0.0005` falsifies
"macOS-CPU per-mode pose ordering transfers to Linux-x86_64 CPU" (host FP drift reversed the ordering) and
re-routes to scoring directly on the contest-CPU host. **Win:** `delta_score_total < 0` is the first exact
PR110++ frontier win and unlocks the menu-ILP gate. Because the 2-switch gain is tiny (pose-avg −1.14e-5),
the more decisive R1 experiment is the **+27-byte all-64-switch variant** (R1-PLUS): its sample pose
reduction (8.27e-2) is ~12× larger, the rate cost is +1.8e-5 (negligible), and it is the candidate most
likely to move the exact score measurably. R2 leaves the strict deliverable built; R1-PLUS is a one-line
re-run of the candidate builder with the budget relaxed to 247 bytes (operator/R1 decision).

---

## 5. Wire-in / orphan-signal note (Catalog #125)

- Hook #6 probe-disambiguator: this table IS the non-MPS substrate the assessment's R2 called for; it
  resolves "is the incumbent K=16 selector optimal on the exact axis?" into a candidate-generator table.
- Hook #2 Pareto: CORRECTED — the 220-byte rate floor is NOT the binding PR110++ constraint. The FECa
  Markov coder's resistance to re-selection is; +27 bytes (rate +1.8e-5) buys the full distortion-optimal
  selection. The Pareto frontier here is pose-distortion-vs-selector-rate, and it is steeply favorable to
  spending a few selector bytes. The strict deliverable is rate-neutral; R1-PLUS spends +27 bytes.
- Hook #5 continual-learning: the one R1 exact row (when dispatched) reseeds the V3 ΔS-judge + recalibrates
  `pr110_opt11_multi_mode_per_pair_composition_savings_v1` (FORMALIZATION_PENDING). The exact-CPU per-mode
  table itself is the non-MPS substrate that should SUPERSEDE `frame_exploit_*_mps600/pair_component_rows.jsonl`
  as the input to the whole PR110++ menu/per-region cluster (the MPS substrate mis-ranks 95.5% of pairs).
- Hooks #1/#3/#4 N/A until a real contest-CPU row lands (R1).

## Manifest (machine-readable)

```
frontier_archive_sha256: b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c
frontier_archive_bytes: 178493
member_x_grammar: FP11[8] + source_payload[178121] + selector_len[2] + FECa_selector[220] + DQS1_tail[42]
selector_attack_surface_bytes: 220                # FECa (FEC10 hybrid adaptive-blend), NOT 879, NOT FEC6-Huffman
mode_vocabulary: FEC6_FIXED_K16_MODE_IDS (16 frame0 modes)
segnet_blindness_verified: true                   # d_seg == 0 exact for all frame0 modes, all 67 pairs
per_mode_table_substrate: exact_cpu_distortionnet # upstream/modules.py DistortionNet, device=cpu (NOT MPS)
n_pairs_sampled: 67
n_pairs_improvable: 64                             # exact-CPU argmin-pose != incumbent AND improves pose
total_sample_pose_reduction_argmin_vs_incumbent: 8.265771e-02
mps_vs_cpu_argmin_agreement_rate: 0.0448          # R2 falsifiable test: MPS mis-ranks 95.5% of pairs
mps_vs_cpu_mean_spearman: 0.215
r2_falsifiable_prediction: CONFIRMED              # MPS substrate mis-ranks modes
candidate_archive_sha256: 5facf0fb9b6ea558b4f589278192e3a958d84fe644090dae015fbacf914fa55d
candidate_archive_bytes: 178493                   # rate-neutral (equal bytes)
candidate_selector_bytes: 220                     # delta 0 vs incumbent (strict equal-or-fewer)
candidate_n_pairs_switched: 2                     # strict-220 max-gain subset (optimal under budget)
candidate_decoder_latent_dqs1_byte_identical: true
candidate_noop_consumption_proven: true           # 5,621,484 raw bytes differ
candidate_advisory_pose_avg_reduction_over_600: 1.14e-05  # [macOS-CPU advisory]; exact ΔS needs R1
rate_distortion_finding: all_64_switches = +27 selector bytes (rate +1.80e-5) for 8.27e-2 pose reduction
recommended_r1_plus_candidate: budget=247 (all-64-switch); ~12x larger pose lever, negligible rate cost
axis_tag: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false              # R1 operator dispatch required for contest-CPU axis
biggest_blocker: contest_cpu_exact_eval_is_operator_dispatched (R1); local macOS-CPU is advisory-only
```
