# ddm_pc1 — pose-carrier efficiency: basis precision, coefficient precision, generated basis, learned low rank — each WITH the re-solve (charter, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-05 ~17:05Z under the operator's standing GO. Axes: bytes
`[exact local byte arithmetic]`; d_pose `[macOS-CPU advisory, cpu_torch fp32 authority backend, n600]`; `score_claim=false` until a T4 row.

## The object (recall, MEASURED)
The frontier archive spends **22,010 B (12.2%)** on the pose carrier: a 12-dim luma BASIS (`cpr1/carrier_codec.py`: `BASIS_BITS = 5`) ≈ 12.2 KB
stored once, plus 600 × 12 signed-int12 coefficients Rice-coded to **9,829 B** (dx1; lossless recode ceiling −18 B). frame_0 of each pair is
`ddm_up2.render_frame0`: a·warp(frame_1)+b plus the carrier pattern `127.5 + CARRIER_AMPLITUDE × (basis · coefficients)`, then the fs1
selector op. The carrier steers PoseNet's 6 scored outputs per pair: d_pose 6.14e-6 → contribution 0.0078 S. The information floor of 6 targets
at the needed precision is ~50 bits/pair ≈ 3.8 KB total; the carrier is ~6× above it. jg1 measured that CUTTING rank/precision of the shipped
basis WITHOUT re-solving damages pose 104.6–822.7×; the re-solve (`ddm_jg5.refine_pair`: damped Gauss–Newton on the basis + int12 lattice with
the ±2 polish, jg5's materiality stop) is the OBJECT CHANGE that reopens every one of those cuts (composition law m148). fs2 re-solved 21 pairs
and moved the pointer. Exchange: 1 B = 6.658589531221714e-7 S; pose marginal d S/d d_pose = 5/√(10·d_pose) ≈ 638 S per unit at 6.14e-6.

## PRIOR-LAW PREDICTION (m38) — per variant, ΔS = rate saving − pose cost; ADMIT iff projected ΔS ≤ −2e-5 (10× the pure-pose projection error 3.7e-6)
| variant | rate saving (DERIVED) | predicted Δd_pose after re-solve | predicted ΔS | falsifier |
|---|---:|---:|---:|---|
| V1 basis 5→4 bits, coefficients re-solved | −2,440 B (−1.6e-3 S) | ≤ +3e-7 (+2e-4 S) | −1.4e-3 | ΔS ≥ 0 |
| V2 basis 5→3 bits | −4,880 B (−3.3e-3 S) | ≤ +1e-6 (+6e-4 S) | −2.6e-3 | ΔS ≥ 0 |
| V3 coefficients 12→10 bits (lattice coarser ×4) | −1,200 B (−8e-4 S) | ≤ +5e-7 | −5e-4 | ΔS ≥ 0 |
| V4 GENERATED basis, rank 12 (2-D DCT lowest 12 or Zernike, zero bytes; coefficients re-solved from scratch) | −12,200 B (−8.1e-3 S) | 1e-5…5e-5 (break-even at 2.3e-5) | −5e-3 … +1.3e-2 | d_pose > 2.3e-5 |
| V5 learned rank-8 basis = SVD of the 600 realized patterns, re-solved | −4,070 B basis −3,280 B coeffs (−4.9e-3 S) | ≤ +2e-6 (+1.3e-3 S) | −3.6e-3 | ΔS ≥ 0 |
V4 is THE structural test (basis becomes a free generator in inflate.py — the "compile the generator" doctrine); V1/V2/V5 are the likely
banked wins. Compose the admitted variants (they act on different bytes) and price the composition exactly.

## What to do
A. RECALL: `experiments/ddm_fs2_carrier_resolve_on_changed_pairs.py` (the solver contract, verbatim reuse of `ddm_jg5.refine_pair`),
   `experiments/ddm_jg5_pose_resolve_on_edited_renders.py`, `ddm_up2.render_frame0`, `cpr1/carrier_codec.py` (Rice, BASIS_BITS, COEFFICIENT_BITS),
   memos `ddm_jg1_joint_solve_20260819.md` (§ carrier, the damage numbers), `ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md`, `ddm_fs2_pointer_move_25_20260904.md`,
   `ddm_bp2_*`/`ddm_br1_*` (frame_0 warp facts). Shipped renders (frame_1) for all 600 pairs are retained (cl2 parse-back, 3.66 GB, sha `f86bfaf3…`,
   under `…/ddm_cl2_hpac_prior_capacity_ladder/parseback/`). `tools/subagent_checkpoint.py read --subagent-id ddm_pc1` first.
B. SVD of the 600 realized carrier patterns (basis·coefficients, 600 × pixels): energy spectrum, effective rank at 99%/99.9% — the prior for V5.
C. Each variant: rebuild the frame_0 generator with the changed basis/lattice → re-solve ALL 600 pairs with `refine_pair` (start from the shipped
   coefficients projected onto the new basis; V4 from least-squares of the realized pattern onto the DCT basis) → d_pose n600 through
   `cpu_torch` (authority) — `coreml_cpu_fp32` (bit-exact, 1.94×, `tac.ane_screening`) may drive the inner GN iterations, the FINAL d_pose is cpu_torch
   → exact bytes (basis at its bit depth + Rice re-encode of the coefficients with `carrier_codec`) → ΔS. Time it on 20 pairs first (scope, not
   mechanism), then run all 600 with multiprocessing over pairs (≤ 8 processes; another arm shares the CPUs).
D. Winner/composition: build the candidate archive through the shipped container path (cl2's stage step rewrites one section; the carrier section
   is the sister case), receiver decode identity, twin the solve (a second full re-solve from the same start must reproduce the coefficients
   bit-exactly — the solver is deterministic), `tools/make_candidate_seal.py` contest-CUDA. **Never dispatch Modal; MAIN fires.**
E. Memo `.omx/research/ddm_pc1_pose_carrier_efficiency_20260905.md` (table above with MEASURED columns · residuals · verdict_scope per negative ·
   frontier line last); law `pose_carrier_rate_per_target_bit_v1` registered with the anchors; lane `lane_ddm_pc1_pose_carrier_efficiency_20260905`;
   owed items as `## ITEM n — …` registered with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_pc1`.

## OPTIMAL FORM
Reference form = fs2's solve (jg5 `refine_pair`, full n600, cpu_torch authority, exact Rice bytes). Mechanism delta = the basis/lattice under test.
SCOPE deltas allowed: the 20-pair timing run only. A variant solved on a subset, or scored on a screening backend, or with a truncated GN is a
TOY: refuse it at the typing moment.

## Compute, memory, disk, resumability (binding)
- CPU only (no Metal — md3's cell and bd1's trainer own Metal); ≤ 8 processes; ANE/coreml screening allowed for inner iterations.
- Disk: APFS tiers only for trees (`/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/`, 29 GiB free; boot volume has room for small
  trees under `experiments/results/`); APDataStore is ExFAT — payload blobs only. KEEP THE PAYLOAD (coefficients per variant, bases, archives).
- Detached launches via `tools/launch_detached_process.py` with distinct `--done-receipt`s (`.omx/tmp/codex_runs/ddm_pc1_<stage>.done`); background
  until-loops for waits. Resumable; `tools/subagent_checkpoint.py` every ~10 tool uses. Commits ONLY via the serializer (`[no-triality] [p0-ledger-ok]`,
  post-edit shas); `.py` two review passes; NO co-author trailer; no `/tmp`; grep argparse first. Read CLAUDE.md + `docs/operating_manual_craft_handoff.md`.
  Label every number MEASURED / DERIVED / PREDICTED. End with `cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]` + any advisory candidate line.
