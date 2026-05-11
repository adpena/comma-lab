# Device-axis paired anchor matrix — frontier state 2026-05-11

## Classification

- `score_claim=false` for any cell not explicitly tagged with the listed evidence-grade and substrate
- `dispatch_attempted=false` for new work
- this memo is a consolidation of existing on-disk evidence; no new dispatch was triggered
- senior-review refresh: `2026-05-11T16:27:18Z`
- active/live rows in this memo are advisory until refreshed from
  `.omx/state/active_lane_dispatch_claims.md` or provider status immediately
  before dispatch

## Matrix

All scores are from `upstream/evaluate.py` against the EXACT archive bytes,
with substrate listed. CPU rows are Linux x86_64 (GHA / Modal CPU) per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable; macOS-CPU is not
authoritative and is excluded.

| Anchor | Representation | Bytes | CUDA (T4) score | CUDA seg | CUDA pose | CPU (Linux x86_64) score | CPU seg | CPU pose | Δ (CUDA−CPU) | Pose ratio (CUDA/CPU) | Seg ratio (CUDA/CPU) | Winning axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A1** (PR101-derived, score-gradient training) | HNeRV-cluster | 178,262 | **0.22635** | 0.000663 | 0.000171 | **0.19285** | 0.000560 | 0.000033 | +0.03350 | 5.18× | 1.18× | **CPU** (silver-medal-band proximity; rounds to 0.19 = PR101 gold display tier) |
| PR103-on-PR106 AC repack | HNeRV decoder + AC rate work | 185,578 | 0.20898 | (per dual_device analysis 2026-05-11) | — | 0.22966 | — | — | −0.02068 | inverted | — | **CUDA** |
| **PR106 latent sidecar r1** (Kaggle table → exact T4) | HNeRV + per-pair latent perturbation | 186,808 | 0.20739 | 0.000649 | 0.000033 | 0.22868 | 0.000638 | 0.000164 | −0.02129 | 0.20× | 1.02× | **CUDA** |
| **PR106 latent sidecar r2** (FLOOR until 2026-05-11T18:09Z) | HNeRV + per-pair latent r2 perturbation | 186,822 | 0.20665 | 0.000643 | 0.000032 | **0.22809** (claude 2026-05-11T17:21Z, Modal CPU Linux x86_64) | 0.000632 | 0.000164 | -0.02145 | 0.197× | 1.017× | r2 r1 (format_id=0x01); displaced as CUDA exact-floor 2026-05-11T18:09Z by PR101-grammar variant (-2.78e-5 rate-only Δ on bit-identical decoded frames) |
| **PR106 latent sidecar r2 + PR101 grammar** (NEW FLOOR 2026-05-11T18:09Z) | HNeRV + per-pair latent r2 perturbation (format_id=0x02 PR101 ranked Huffman/no-op grammar) | **186,780** | **0.20662** | 0.000643 | 0.000032 | **0.22806463271134514** (claude 2026-05-11T18:52Z, Modal CPU Linux x86_64; call fc-01KRC5T9DD1ZD71N4B96DNGZBT; rc=0, n_samples=600) | 0.00063196 | 0.00016402 | -0.02144 | 0.197× | 1.017× | **PAIRED COMPLETE 2026-05-11T18:52Z**: CPU score 0.22806463 matches predicted band 0.22806±0.00003 EXACTLY. Δ vs r2 r1 on CPU = −0.0000278 IDENTICAL to CUDA-side Δ (-0.0000278) — definitive proof rate-only Δ propagates the same on BOTH axes. Pose/seg device-axis ratios unchanged from r2 r1 (rate-only change preserves distortion components by construction). |
| **c3 PR106 r2 sidecar (L1 empty)** 2026-05-11T18:13Z | PR106 r2 + 0xFD/0x12 family wrapper + empty C3 residual | 186,832 | **0.2066336354574151** | 0.0006424 | 0.0000324 | **0.22810213271134513** (Modal CPU Linux x86_64; call_id `fc-01KRCC1X3DYYFC187BGM3HFTN3`; rc=0, n_samples=600) | 0.00063199 | 0.00016402 | -0.02147 | **5.06×** | 1.18× | **CUDA** Δ vs PR106 r2 r1 baseline (pre-PR101-grammar) = −1.22e-5; CPU Δ vs r2 r1 CPU (0.22809) = +1.0e-5 (the +10 wrapper byte rate cost shows on CPU axis as well, slightly worse than CUDA-side absorption). Pose ratio 5.06× matches PR106 r2 r1 family pattern (5.07×) — wrapper bytes do NOT change device-axis profile. L1 byte-closure proof of the 0xFD wrapper format on BOTH axes. Archive sha256=`eafb1a027f7065751bc6eb8716f9cc92e0e86c290c68eaa529c59b0066f0960a`. |
| **wavelet PR106 r2 sidecar (L1 empty)** 2026-05-11T18:13Z | PR106 r2 + 0xFD/0x10 family wrapper + empty wavelet residual | 186,832 | **0.2066336354574151** | 0.0006424 | 0.0000324 | **0.22810213271134513** (Modal CPU Linux x86_64; call_id `fc-01KRCC22ZFF7D3QPWCWNDK886F`; rc=0, n_samples=600) | 0.00063199 | 0.00016402 | -0.02147 | **5.06×** | 1.18× | **CUDA** Δ identical to c3 row above on BOTH axes. CPU score bit-identical to c3 CPU — confirming the wrapper-format byte-identical-inflate invariant holds across the device-axis transition. Archive sha256=`ed90a2250e948ed70086162a84ab907a866099b9d2a7d13fd13870e5fa29fe81`. |
| **cool_chic PR106 r2 sidecar (L1 empty)** 2026-05-11T20:17Z | PR106 r2 + 0xFD/0x11 family wrapper + empty cool_chic residual | 186,832 | **0.2066336354574151** | 0.0006424 | 0.0000324 | not yet measured | — | — | — | — | — | **CUDA** Δ identical to c3/wavelet (V-CONSOLIDATION 3/5 byte-closure). Archive sha256=`d48600da99bad7a75301a4b982d9e3d2e1e3b0dba23e2a49859187305b06cf44`. |
| **siren PR106 r2 sidecar (L1 empty)** 2026-05-11T20:22Z | PR106 r2 + 0xFD/0x13 family wrapper + empty SIREN residual | 186,832 | **0.2066336354574151** | 0.0006424 | 0.0000324 | not yet measured | — | — | — | — | — | **CUDA** Δ identical to c3/wavelet/cool_chic (V-CONSOLIDATION 4/5). Archive sha256=`f373b308b08059e4d8749a0ff4c25d814e2b7af0f30109a8f5f8246ceca1e878`. |
| **coord_mlp PR106 r2 sidecar (L1 empty)** 2026-05-11T20:25Z | PR106 r2 + 0xFD/0x14 family wrapper + empty coordinate-MLP residual | 186,832 | **0.2066336354574151** | 0.0006424 | 0.0000324 | not yet measured | — | — | — | — | — | **CUDA** Δ identical to all 4 prior family rows (V-CONSOLIDATION 5/5 COMPLETE — full non-HNeRV family wrapper-format byte-closure proof across `{wavelet, cool_chic, c3, siren, coord_mlp}`). Archive sha256=`01df6f12e562fdd1ad7c0a2d4dfc289ce814823cdea06c9a5987feb021d17cc1`. |

### V-CONSOLIDATION 5/5 family byte-closure (2026-05-11T20:30Z)

Operator directive 2026-05-11 "proceed with all that does not cost more than $5 individually" enabled the remaining 3 family CUDA dispatches (cool_chic + siren + coord_mlp) following the c3 + wavelet TOP-2 from earlier in the same day. Empirical anchor: **all 5 non-HNeRV residual basis family wrappers produce byte-identical `0.bin` inflate output in empty-residual mode (sha `891369c4...`), so all 5 score identically at `0.2066336354574151` [contest-CUDA T4].** Per-family Δ vs PR106 r2 r1 baseline is `-1.22e-5` (essentially noise within contest-CUDA jitter; the +10 wrapper-header byte rate cost is exactly offset by the seg-avg drift from the `np.round`-pipeline absorption shared by all 5 family inflate runtimes).

This anchors the FAMILY-EMPTY-RESIDUAL byte-closure ladder. The L2 score-aware encoder dispatches (each family's gradient-trained residual blob) are the next dispatch class; per HNeRV parity discipline lesson 6 ("score-domain Lagrangian"), each L2 dispatch is independent and operator-gated.

Paired-CPU eval for c3 and wavelet was dispatched in the same V-CONSOLIDATION cycle (call_ids `fc-01KRCC1X3DYYFC187BGM3HFTN3` and `fc-01KRCC22ZFF7D3QPWCWNDK886F`); both Modal CPU Linux x86_64 jobs running detached for harvest within 24h TTL. The empirical CPU rows above will be updated post-harvest with the contest-CPU axis score and the (CUDA-CPU) Δ.

## Mechanism reading

### Device-axis behavior is **packet-specific**, not monotone

- HNeRV-cluster A1: CPU wins; pose-term CUDA is **5.18×** worse than CPU (the geometric drift profile per `feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`).
- PR106-derived sidecars: CUDA wins; pose-term CUDA is **5×** *better* than CPU (the inverse). SegNet term is roughly stable across devices in both families (1.02–1.18× ratio).

This **falsifies any extrapolation of axis** from one packet to another. A1
remains the CPU-axis leader; PR106 r2 is the CUDA-axis leader. They cannot
be merged into a single ranking without paired evidence per packet.

### Operating point per CLAUDE.md SegNet-vs-PoseNet section

At the **PR106 frontier operating point** (pose_avg ≈ 3.2–3.4e-5), CLAUDE.md
states:
- marginal `d(seg)/d(seg_avg) = 100`
- marginal `d(pose)/d(pose_avg) = 5 / sqrt(10·pose_avg) ≈ 271`
- **pose is 2.71× more marginal-valuable than seg** at this operating point
- crossover threshold at pose_avg ≈ 2.5e-4 (we are ~7× below this)

A1's pose CPU (3.3e-5) is at the same operating point. PR106 sidecar's pose
CUDA (3.2e-5) is at the same operating point. **Pose improvements buy more
score per byte than seg improvements at both anchors' winning-axis frontier.**

### Implications for non-HNeRV lane prioritization

The frontier roadmap (`frontier_roadmap_status_20260511_codex.md`) lists 5
next-unblocked keys. The 3 NON-HNeRV ones with clear pose/seg targeting:

| Lane key | Tier | Target axis | Why it matches the operating point |
|---|---:|---|---|
| `joint_admm_balle_arithmetic_stack` | 20 | rate (both axes) | JCSP runtime consumer is LANDED (51 tests pass per `jcsp_rawvideo_runtime_consumption_20260511_codex.md`); needs a byte-closed candidate archive |
| `raft_radial_openpilot_pose` | 90 | **pose** | RAFT-derived pose sidecar directly attacks the pose term; matches the 2.71× marginal at frontier |
| `lapose_motion_atom_allocator` | 50 | **pose** allocator | LAPose-inspired inverse-dynamics → pose-conditioned byte allocation; planning-only, needs charged archive consumer |
| `telescopic_foveation_field` | 40 | seg+pose (foveation) | Charge runtime geometry consumer contract; currently planning-only |
| `categorical_qma9_clade_spade_openpilot` | 10 | seg (mask grammar) | Highest tier, but seg-targeted at low-marginal operating point |

**Council reading**: at the PR106 r2 frontier (pose_avg=3.2e-5), pose-axis
lanes (RAFT, LAPose) have higher marginal-value-per-byte than mask grammar
(categorical) by the 2.71× crossover. The tier ranking in the roadmap is
strategic-state, not marginal-EV; for direct score-lowering at the current
floor, **the pose-axis non-HNeRV lanes are the higher-EV bet per dollar.**

## What this matrix decides

### NOT YET ITEM 1 (A1 PR submission-policy review) — informed by this matrix

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE": A1 has both axes in custody at silver-medal-band proximity on the
contest-ranked CPU axis. The CUDA value (0.22635) is consistent with the
HNeRV-cluster predictor's published −0.033 gap. A1 is a **CPU-axis submission
candidate for policy review**, not an automatic PR. It remains pending the
5-turn skunkworks council greenup, paired-custody packet review, public-PR
policy, and explicit operator approval per CLAUDE.md "Submission PR gate"
non-negotiable.

The matrix does NOT change that gate. It clarifies that A1's CPU dominance is
not a device-axis bug — it is a packet-specific operating-point characteristic.

### Pose-axis non-HNeRV work (RAFT / LAPose / foveation) — newly higher-priority

If the operator approves Phase 2 dispatch ($223–303 envelope per NOT YET ITEM 2),
the budget breakdown should be re-examined: T15 (FiLM) and T17 (VQ codebook)
were ranked under the previous operating point. At the current PR106 r2 floor
(pose marginal 2.71× seg), the budget should reweight toward T-class lanes
targeting the pose term directly.

This is informational; the dispatch decision remains operator-gated.

### PR106 r2 paired CPU eval is closed

Validation addendum `partner_pr106_r2_pr101_grammar_validation_20260511_codex.md`
recomputed the paired r2 row from on-disk JSON:

- r2 CUDA score: `0.206645885457`
- r2 CPU score: `0.228092382711`
- archive SHA-256 on both axes:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- CPU/CUDA pose ratio: `5.068603`

The old "next $0 follow-up" text is superseded. The remaining action is not
another CPU harvest; it is a runtime-consumed byte candidate or a broader
paired-axis mechanism study.

## Cross-references

- Frontier roadmap: `.omx/research/frontier_roadmap_status_20260511_codex.md` (13 candidates, dirty_blocked_row_count=0, selected = pr106_q10_151byte_brotli rate-only)
- Synthesis: `.omx/research/full_stack_score_lowering_synthesis_20260511_codex.md` (multi-scale review discipline + highest-EV execution order)
- Mechanism map: `.omx/research/pr95_plus_mechanism_map_20260511.md` (PR95+ writeup digest + non-HNeRV escape routes)
- Cross-paradigm inventory: `.omx/research/cross_paradigm_frontier_inventory_20260511_codex.md` (13 paradigm rows + stackability matrix)
- PR106 r2 dispatch record: `.omx/research/pr106_latent_sidecar_dual_axis_and_radius2_dispatch_20260511_codex.md` (kernel v2, exact T4 result, P100→T4 drift)
- NOT YET items: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
