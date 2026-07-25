# Codex Findings — DDM PC2 PC1 Pose-Descent Smoke

`[macOS-CPU frozen-scorer advisory]` · `research_only=true` ·
`score_claim=false` · `promotion_eligible=false` · `pointer_moved=false`

## Disposition

**MEASURED:** the preregistered 16-step PC1 score-domain coordinate smoke
descended exact n600 pose debt and exact joint action in both measured
half-windows. The bounded instance verdict is
`PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE`; the fork is `PC1_DESCENT_STAGE`.
MAIN landing review is mandatory.

This does not establish convergence to `d_pose=2.94e-5`, a contest score, or a
family-level result. The PC1 active zero home begins at `d_pose=163.0453`, which
is consistent with the already-settled PC1 admission row but remains roughly
5.55 million times the target. A constant-slope projection requires about
1,216 total accepted steps and is explicitly a derivation, not a convergence
law.

## Exact receiver-closed n600 rows

All rows use the exact composition receiver, evaluator resize/uint8 path, frozen
CPU scorers, batch 32, and exact archive bytes.

| Accepted steps | `d_seg` | `d_pose` | Archive bytes | Advisory action |
|---:|---:|---:|---:|---:|
| 0 | 0.02491527133517795 | 163.04531226928225 | 139,547 | 42.96331550786856 |
| 8 | 0.02502353244357639 | 162.0079751433403 | 139,562 | 42.845496154523296 |
| 16 | 0.025105328030056422 | 160.90004329290474 | 139,570 | 42.71581437381254 |

**MEASURED 0→16:** `Δd_pose=-2.1452689763775084`,
`Δd_seg=+0.0001900566948784717`, `Δbytes=+23`, and
`ΔS=-0.24750113405601581`.

The aggregate per-step slope is `Δd_pose=-0.13407931102359427`,
`Δd_seg=+1.187854342990448e-5`, and `ΔS=-0.015468820878500988`.
The observed pose-score-progress/Seg-regression ratio is
`14.023295441931698`, clearing registered
`R*=4.1215446777965665`. The two independent half-window ratios are
`11.883810721381112` and `16.855025089164695`; both clear the gate and
both have negative exact joint deltas.

## Seg collateral is structured

The small composite Seg regression hides opposing class movements:

- Lane: `+0.010778424039186885`
- Road: `+0.0018581718000546255`
- MyCar: `+0.00012215976463440803`
- Undrivable: `-0.00011262849624899506`
- Movable: `-0.02262236146063376`

The PC1 descent stage is therefore admissible only with the holistic per-class
watch list. The Movable gain offsets Lane/Road harm; no headline composite may
erase that collateral.

## What actually ran

- Exact parent:
  ws4 `W_joint_step50_live`, 138,813 bytes, SHA-256
  `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`.
- Objective compile-refuse:
  `score_domain_loss=true`, `PoseMarginalWeightLaw=false`, static `w_pose=1`.
  The mutually exclusive raw-pose marginal law was not double-composed with
  the contest square-root term.
- Governor:
  16/16 accepted steps, 192/768 candidates, train batch 4, exact verdict batch
  32, stop `TARGET_ACCEPTED_STEPS_REACHED`.
- Accepted coordinates:
  one even-indexed knot per bit-reversal sweep; 15 rotations in `rx`/`ry` and
  one in `rz`, all at quantum 256. No luma residual opened.
- Resumability:
  17 immutable complete checkpoints (steps 0 through 16), exact n600 verdicts
  at steps 0/8/16, and 19 independently resumable chunks per exact verdict.

## Honest horizon and successor

**DERIVED:** holding the measured 0→16 `d_pose` slope constant would reach the
target after `1216.0361029942253` total accepted steps, or
`1200.0361029942253` more after this smoke. This ignores quantization,
curvature, changing knot support, and photometric saturation. It authorizes no
long launch.

The positive fork keeps a bounded PC1 descent stage alive. Depth-stratified,
object-local ξ advection remains the named `PREDICT` successor if later
receiver-closed slopes die; the #601 planar one-depth and #605 n16
single-ground-depth rows remain scoped controls, not family closure.

## Durable custody

- Compact tracked result:
  `.omx/research/ddm_pc2_pose_descent_smoke_result_20260725.json`.
- Full SSD receipt:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_pc2_pose_descent_smoke_20260725T121448Z/ddm_pc2_pose_descent_smoke_receipt.json`,
  26,844 bytes, SHA-256
  `82713bc2bc1e128a0cd12ada95101bd85bd6d47f40a8ea6e85d8f503a088b234`.
- Final exact composition:
  139,570 bytes, SHA-256
  `946cf434b96839318af63364021b2b3cbfe0c7d3dee68b0849cee8cc45312477`,
  parse-back exact.
- Exact verdict SHA-256 at steps 0/8/16:
  `98026f33ab8fa61a6069b952037f6a978913cac43a73fca5a0370f8167154e3d`,
  `11814216632afc417c067db93c2c0b203ad2227461e640ec2259ed2338128bcb`,
  and
  `308cc1c5e6c02112688ed388ec2613e863c5a18ee66f929fc657b2542f178964`.
- Canonical continual-learning row:
  `ddm_pc2_pc1_solved_plane_pose_descent_n600_20260725`, verdict `PARTIAL`
  and advisory-only, in `.omx/state/probe_outcomes.jsonl`.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
the current optimal-start card; canonical lane, subagent, dispatch, and
frontier surfaces; PC1 admission; ws4 arbitration and exact W_joint custody;
J7 governor precedent; Menu-1 typed config; photometric-wall and #601/#605
lineage; both delegated inboxes.

## MAIN landing review

MAIN must review the exact receiver semantics, the static score-domain
compile-refuse, the surprising but previously settled active-home pose scale,
the classwise Lane/Road collateral, checkpoint/resume closure, and the
instance-only scope before landing. No external task should be closed and no
pointer should move from this advisory result.

## Verification seal

- Ruff lint and format checks: pass.
- JSON parse and `git diff --check`: pass.
- Three consecutive post-edit clean passes: 36 tests each, in 227.36 s,
  219.77 s, and 221.61 s.
- Review tracker: all 51 entities across the new law, test, and runner files
  have exactly three consecutive `marked_reviewed` events.
- Lane maturity: L2 research-only; implementation, real-archive empirical,
  strict-preflight, and three-clean-review gates are true. Contest CPU/CUDA,
  memory-entry, and deploy-runbook gates remain false.
- Global lane validation remains red on 110 pre-existing missing historical
  evidence paths outside PC2; there is no PC2-specific inconsistency.
