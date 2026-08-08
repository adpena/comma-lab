# ddm_pk1 Pose-Keystone Compatibility Findings

tags: [no-triality] [p0-ledger-ok]

## Scope and authority

- Charter: `.omx/research/ddm_pk1_20260807/CHARTER.md`
- Common contract: `.omx/tmp/codex_runs/_common_contract.md`
- Axis: `[macOS-CPU advisory]`
- Score claim: false
- Pointer moved: false
- Preferred surface used for B: ep854 cell_drop50 / CR1 surface
- Sample: seeded stratified n=120 of 600, seed 20260807, 10 temporal blocks
- Selection check: `pose_target_center_energy` subset/population ratio 1.00146657564028, seeded-random band [0.9846426635608936, 1.0164728592336898], verdict MATCHED
- No `.py` files were edited for this task; the serializer/review-tracker path was not triggered.

## Result table

| Candidate | Work done | d_pose result | d_seg collateral | Counted bytes surface | Verdict |
|---|---:|---:|---:|---:|---|
| A: PR130 semantic-pose CPR1 mechanism | Adapter/custody smoke only | Not measured on ep854 | Not measured | CPR1 carrier parsed at 23,054 B inside external 191,052 B archive | QUEUED-WITH-FIRE-ORDER for surface-fit measurement |
| B: terminal 6-eq GN pose solve | Measured n=120 | 160.49780204844654 -> 116.48953806212609 | 120/120 frame1 byte-identical; SegNet reads `x[:, -1, ...]` | measured outer archive 367,855 B; terminal packet 7,295 B | FOLDED negative for this formulation |

## Candidate A

The local PR130 CPR1 release archive matched SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
The vendored CPR1 carrier decoded and re-encoded losslessly with the expected
shape: 600 x 12 coefficients and 12 x 3 x 24 x 32 basis fields. This is a
borrowed-substrate custody/adapter smoke only. It does not realize or score the
PR130 pose stream on the ep854 banked seg/rate surface.

Borrowed-substrate accounting:

- Theirs: PR130 semantic-pose CPR1 carrier mechanism, tensor shapes, compact codec semantics, external release bytes.
- Ours: vendored loader/codec custody smoke and parse-back receipt only.
- Remaining step: fit/train the PR130 neutral-gray semantic-pose carrier against the ep854 cell_drop50 surface frames and PoseNet target cache on seeded stratified n>=120, parse back the counted CPR1-style carrier, then score d_pose and d_seg through CPU-torch real scorers.

Fire order: QUEUED-WITH-FIRE-ORDER. Run the A surface-fit measurement before any #984 promotion or rejection based on PR130-style pose.

## Candidate B

B was fully measured at the charter's instrument scope on the preferred ep854 surface. It used the existing terminal six-equation GN mechanism, zero frame0 policy, `eg1_generic_low_frequency_six_v1`, seed 20260728, amplitude_q8 512, rank 6, two relinearizations, and CPU-torch PoseNet.

The pose debt improved but stayed far outside the compatibility tube:

- Selected base mean d_pose: 160.49780204844654
- Selected final mean d_pose: 116.48953806212609
- Mean delta: -44.00826398632046
- Strict-improvement pairs: 120 / 120
- Frame1 byte-identical pairs: 120 / 120
- Pose tube threshold used by the target cache: radius 0.05, MSE 0.0025000000000000005

This is not a near miss. The remaining pose term is `sqrt(10*d_pose) = 34.13056373137222`, so B cannot inherit the ep854 seg/rate prize.

Verdict scope: FORMULATION-level negative for terminal six-equation GN, low-frequency frame0-only residual, zeros frame0 policy, on the ep854 cell_drop50 surface, seeded stratified n120 advisory. This does not kill PR130-style neutral-gray carriers, joint pose-in-loop vehicles, different terminal bases, or a different frame0 policy registered as a new formulation.

Follow-on: FOLDED into #984 only as a negative B leg. Do not route #984 through this B formulation on ep854 unless a new basis/policy is explicitly registered and measured.

## Joint arithmetic

These are advisory component arithmetic rows, not exact contest scores:

- Formula: `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489`
- CR1 endpoint d_seg used: 0.003943024
- CR1 derived endpoint bytes used for comparison: 284,248 B
- B measured outer bytes: 367,855 B
- S with ep854 derived bytes and selected base pose: 40.64574840712788
- S with B measured outer bytes and selected final pose: 34.76980567657298
- Delta S for B vs selected-base arithmetic: -5.875942730554904
- External PR130-pose-transfer arithmetic at CR1 derived bytes: 0.5988390884870703
- Delta vs current own-vehicle hot-state pointer 0.7534578126155775: -0.1546187241285072

The PR130 line is target arithmetic only. It is not a measured A result on the banked surface.

## Receipts

- A smoke: `/Volumes/VertigoDataTier/pact/ddm_pk1_20260807/pk1_candidate_A_pr130_cpr1_adapter_smoke.json`, SHA-256 `92bbee55ce460f52cebec7c928224a0e7e94575425abbc9a68b79186994bf062`
- B receipt: `/Volumes/VertigoDataTier/pact/ddm_pk1_20260807/pk1_candidate_B_terminal_gn_ep854_n120_receipt.json`, SHA-256 `1760428647f0bbdbed939f83926515e6f574b3d142ab9d318c070f3e008a3af2`
- B rows: `/Volumes/VertigoDataTier/pact/ddm_pk1_20260807/pk1_candidate_B_terminal_gn_ep854_n120_rows.jsonl`, SHA-256 `6872fa5f7f2d1ff8cdb2034e6815bda2049536947be1b08b00a4fdd6f29393ed`
- B composed outer archive: `/Volumes/VertigoDataTier/pact/ddm_pk1_20260807/pk1_candidate_B_terminal_gn_ep854_n120_outer.zip`, SHA-256 `6db71c45dd33623b350cfda2e6e4f28c8ac74c927b6cda7ad4029c59f1477039`

Recall evidence consulted: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `.omx/research/ddm_cr1_composition_row_827_20260801.md`, `.omx/research/ddm_na2_negative_audit_20260803.md`, `.omx/research/ddm_eg1_endgame_chain_20260728.md`, `.omx/research/pr130_eureka_intake_acquisition_20260806.md`, and the local PR130 lift receipts/tests.
