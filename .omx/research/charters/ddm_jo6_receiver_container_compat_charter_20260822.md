# ddm_jo6_receiver_container_compat — unblock the r8 chain: RX1M archive vs F26 receiver

## MISSION (critical path — the jo1 r8 solve is BLOCKED at receiver execution)
r8's target_birth stage is COMPLETE and retained (2.37 h, pointers under
experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/stages/01_target_birth/
— run dir otherwise SACRED). The chain is BLOCKED at run_receiver: the composed archive
(receiver_close_attempt_0000/archive.repeat.zip, 181,484 B, member census ["p"]) has a
payload whose head magic is **RX1M** (MAIN verified: WANS1/SD1M/SM3R/CPR1/HPAC ALL ABSENT
from the full 181,384 B payload), while the staged shipped receiver
(receiver_close_attempt_0000/submission — the F26/fx5 runtime) refuses at
runtime/f26_inflate.py:429-430: `parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M",
b"SM3R"))` → InflationError. Receipts: attempt_0000/inflate.log (bare-python rc=2, CURED
via shim) and attempt_0001/inflate.log (rc=1, the InflationError — the REAL defect).
DIAGNOSE which side violates the jo1/jo2/jo3 design intent and FIX at the optimal layer,
then RESUME r8 past receiver execution. This arm owns the chain until
RECEIVER_EXECUTION_POINTER.json is written and the run proceeds, or a typed blocker.

## THE FORK (adjudicate from the chain's own design docs/code, not from preference)
A. receiver_close SHOULD emit an F26-compatible archive (fx5_e1 body carrying the edited
   token field + UNTOUCHED semantic-weights section): fix the closer/builder module and
   REBUILD as a fresh receiver_close attempt (never overwrite attempt_0000). Check
   whether the builder dropped/re-wrapped the base archive's semantic section.
B. RX1M is the INTENDED jo1 container (RX1/RX2 lineage per the pq1 packet notes): then
   run_receiver stages the WRONG submission tree — it must execute an RX1M-capable
   receiver. Note this receiver must be the SHIPPING one for the eventual T4 row
   (a verification against a receiver that will not ship is the #417 fake).

## BINDING CONSTRAINTS (MAIN-derived, verified at source)
1. ENTRYPOINT SHA GUARD: verify_inventory (experiments/ddm_jo3_joint_objective_entrypoint.py:452)
   REFUSES resume if the entrypoint file changes → an entrypoint edit forces a full stage
   re-solve (~2.37 h measured — affordable ONLY if no outside-fix exists; imported modules
   are NOT sha-pinned by the inventory, prefer fixing there).
2. Run-dir custody: preserve-aside pattern only (.bak precedent:
   STORAGE_POLICY.json.r8_initial_launch.bak); never delete/overwrite retained artifacts.
3. KNOWN resume-idempotency defect: write_storage_policy (:482) bakes observed_free_bytes
   (live disk free) into the identity-gated STORAGE_POLICY.json → EVERY resume after disk
   change needs the preserve-aside dance. If you edit the entrypoint anyway, fix this in
   the SAME edit (volatile fields out of identity, or replace_metadata for policy files);
   else file as named debt.
4. Shipped receiver bytes = CUSTODY, never edit. Bare-python cure: tools/host_shims/python
   (exec-wrapper) must be on PATH for any shipped-receiver run.
5. RELAUNCH COMMAND (proven, 2 resumes executed; reuse verbatim):
   .venv/bin/python tools/spawn_durable_daemon.py --log <run>/train.log --label ddm_jo2_joint_objective_fx5 \
     --projected-gb 48 --min-free-gb 44 --rss-cap-mb 16384 --walltime-cap-s 259200 --projected-peak-gib 16.0 \
     -- env "PATH=/Users/adpena/Projects/pact/tools/host_shims:$PATH" TAC_GOVERNED_ADMISSION=1 \
     .venv/bin/python -m experiments.ddm_jo3_joint_objective_entrypoint train \
     --compiled-config .omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8/compiled_config.json \
     --expected-config-sha256 38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90 \
     --resume-from <run>/checkpoints --main-owned-dispatch-authorization
6. TWO-LANDING OWED: receiver_close wrote its pointer WITHOUT decode-validating through
   the staged receiver — the gate whose absence let r8 run 2.37 h before this surfaced.
   Land the validation at the closer path (fail-closed: no pointer until the staged
   receiver decodes the archive to a 0.raw of the expected byte count).

## OPTIMAL FORM
Family reference: the jo2/jo3 chain's own build discipline (sealed configs, typed
pointers, fail-closed verify_record). Provenance pins:
experiments/ddm_jo3_joint_objective_entrypoint.py=<compute at start; refuse if the tree
drifts mid-work> · seal_r8/compiled_config.json sha
38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90.
SCOPE reductions: none. MECHANISM reductions: NONE — a "fix" that swaps in a
non-shipping receiver for verification is the fake this charter refuses.

## CONTRACT
upstream/ READ-ONLY; serializer commits; .py = 2 genuine review passes; memo
.omx/research/ddm_jo6_receiver_container_compat_20260822.md; final message = fork verdict
(A or B, with the design-intent receipt) + fix landed + resume status + the two-landing
gate status + GESTALT-DELTA.
