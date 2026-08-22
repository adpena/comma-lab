# ddm_jo5_determinism_cure_reseal — cure the pair-0 Pose6 repeat divergence → reseal r8 → MAIN refires the solve

## MISSION
The jo1 joint solve FIRED under seal r7 and self-BLOCKED 1,293 s in with the typed reason
{"reason": "winner deterministic repeat differs from explored Pose6 at pair 0", "status": "BLOCKED"}
(rc=2, peak RSS 7,240 MiB ≪ 16 GiB cap — memory geometry fine; stage target_birth checkpoint
RETAINED at experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo4_certified_retention_reseal_20260821_r7_final/checkpoints/
incl. RESUME_LATEST.json). This is the entrypoint's own certified-retention determinism gate
firing CORRECTLY — the winner's PoseNet first-6 output differed between the explore pass and
the deterministic repeat. DIAGNOSE the divergence mechanism, cure it IN the entrypoint,
re-prove determinism, reseal r8 READY_TO_FIRE_UNDER_STANDING_GO resuming from the retained
checkpoint. MAIN refires. You launch NOTHING heavy (bounded ≤3-pair probes only).

## WORK ORDER (strict order)
1. REPRODUCE: run the pair-0 explore-vs-repeat comparison in isolation (bounded, minutes).
   Capture max-abs divergence and WHERE it enters (PoseNet forward? carrier solve? uint8
   realization? RNG state?).
2. BISECT the mechanism — the leading hypothesis (VERIFY, do not assume): torch-CPU
   multi-thread reduction-order nondeterminism in the PoseNet forward (config law: 1-thread
   train, memory †D). Controls: torch.set_num_threads(1) + OMP/MKL env pins +
   torch.use_deterministic_algorithms(True) on BOTH paths; also check for RNG/cache state
   leaking between explore and repeat, and for fp accumulation-order differences from
   batch-shape changes. If the mechanism is thread-order: same-thread-count repeat is NOT a
   cure (still nondeterministic vs a THIRD run) — pin to deterministic single-thread for the
   Pose6-bearing forwards, or restructure the repeat to byte-compare a same-conditions
   recompute that the retention certs also reference (the cert regeneration tuple must
   reproduce EXACTLY — this is the KEEP-THE-PAYLOAD certify-or-block guarantee, so the
   determinism fix must hold for cert regeneration too, not just the gate).
3. CURE in experiments/ddm_jo3_joint_objective_entrypoint.py (2 genuine review passes,
   serializer w/ post-edit sha). Determinism RE-PROOF: ≥3 pairs × 3 independent repeats,
   byte-identical Pose6 + camera payload shas each time — executed, receipts retained.
4. WALL-CLOCK RE-DERIVE: if thread pinning changes step time (measured 1.3895 s/step at the
   r6 probe), re-measure one real step and re-derive the schedule in the seal (never carry
   the stale figure — cross-regime constant transfer).
5. RESEAL r8 via the landed resealer: resume-from the RETAINED target_birth checkpoint
   (resumable-from-disk P0 — do NOT restart from scratch), pin TRIPLE from the working tree,
   FIRE_ORDER complete argv → READY_TO_FIRE_UNDER_STANDING_GO. Final message: mechanism
   named + cure + re-proof receipts + FIRE command.

## OPTIMAL FORM
Family reference form + receipt: the jo1 solve family reference is the jo3/jo4-landed
entrypoint chain (seal r7 compiled sha 3af9848e→r7 5ba2a9b8 lineage; one-pair probe receipt
MEMORY_PREFLIGHT.json 09e5affa…, 1.3895 s/step, grad norm 0.0012942249) + the LIVE typed
blocker receipt in train.log (safe_run status=ok exit=2 elapsed=1292.99s). The cure must
preserve the pinned solve mechanism EXACTLY — only the determinism substrate changes.
Provenance pin: experiments/ddm_jo3_joint_objective_entrypoint.py=92d2a2ab2a927d15dcdc1b97642edfdd4ceaf414113a3ad342b3423760c1f4a6
(pre-cure sha; your commit supersedes it — reseal re-pins from the working tree).
SCOPE reductions (legal): ≤3-pair determinism probes. MECHANISM reductions: NONE — weakening
the determinism gate (tolerance, skip-on-mismatch) is the exact fake the gate exists to
refuse; the cure makes the COMPUTATION deterministic, never the CHECK looser.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- The L70 MLX-GPU bit-identity wall (memo .omx/research/…#348 lineage; task-ledger #348):
  GPU accumulation is nondeterministic without fixed-order reduction — the entrypoint is
  CPU-torch, but the same reduction-order genus applies to multi-thread CPU; the proven
  cures there were fixed-point/fixed-order accumulation.
- rr2 device-scoped decode desync REFUSED S 27.83 (task #1096): probability/compute state
  differing between passes destroys candidates — same genus, cross-pass instead of
  cross-device.
- jo4's own r7 endpoint cure (memo .omx/research/ddm_jo4_certified_retention_reseal_20260821/):
  one-sided derivatives at 9 endpoint coords — do not disturb; the determinism cure is
  orthogonal.
- Config law (MEMORY †D): 1-thread train measured 2.96× slower but deterministic — the
  wall-clock tradeoff is KNOWN; re-derive the schedule honestly rather than keeping threads
  for speed at the cost of the gate.

## CONTEXT ANCHORS (memo-associated)
- Campaign sub-0.12 #1182 (memo .omx/research/ddm_r012_rate_representation_20260821.md);
  pointer fx5_e1 S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600].
- Seal chain memos: .omx/research/ddm_jo3_entrypoint_and_final_reseal_20260821.md +
  .omx/research/ddm_jo4_certified_retention_reseal_20260821/ (seal_r7).
- Run dir (sacred, do not clobber): experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo4_certified_retention_reseal_20260821_r7_final/.

## CONTRACT
upstream/ READ-ONLY; keep the payloads; serializer commits; .py = 2 genuine review passes;
final message states mechanism + cure + re-proof + r8 status + exact FIRE command.
