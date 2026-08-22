# ddm_jo1u2_materializer_cure — diagnose + cure the jo1u T4 materializer remote failure → reseal → ONE re-fire order for MAIN (sources: .omx/research/ddm_jo1u_payload_unblock_20260821.md · commit 032e352f71ea091c1b412bca3a1e51282d1a6921 · ddm_gs3_unbridled_gestalt_20260821.md §JO1-UNBLOCK)

## MANDATE

The gs3 rank-1 critical path is BROKEN: the sealed jo1u materializer (config sha a217a527…,
fired as Modal call **fc-01M0JH3TK89HVSV03GAN8RHQWJ**, app ap-bTFhD1vNXaLd0U0ZE1e1GU) FAILED
remotely. Evidence (all MEASURED by MAIN): (1) `FunctionCall.get()` raises **RemoteError with
an EMPTY message**; (2) `modal app logs` shows NOTHING past the dispatch lines; (3) the remote
volume path `comma-auth-eval-cache-artifacts:ddm_jo1u_fx5_e1_n600_r1/` was **never created** —
the function died before its first volume write. Diagnose at source, cure, reseal, emit ONE
re-fire order. This blocks jo1's joint solve (gs3 rank-2) AND dc1's Family-B fire trigger.

## PRIME SUSPECTS (verify at source in experiments/ddm_jo1_modal_joint_objective.py + the
worker + compiled_config.json — do not assume)

1. **Local-path dependency in the container**: the compiled config pins /Volumes/APDataStore
   and /Volumes/VertigoDataTier paths (fx5 archive, runtime tree, DALI Pose6 .npy). If the
   materializer entrypoint expects them ON the container filesystem without an upload/mount
   leg, it dies at first open(). Check what the jo1u build actually ships to the container
   (Modal mounts vs volume gets vs inline bytes) vs what the config references.
2. **Custom-exception serialization masking**: a repo-defined exception class raised remotely
   cannot deserialize locally → empty RemoteError. Whatever the cure, ALSO make the silence
   loud (control-plane law): wrap the remote entrypoint so ANY exception writes a structured
   FAILURE receipt (traceback + stage + inputs-seen) to the volume BEFORE re-raising — the
   volume write is the black box recorder.
3. Config-sha gate or authorization flag mismatch inside the remote function (it re-validates
   a217a527…; a container-side path normalization could break the digest).

## SCOPE

1. Reproduce CHEAPLY first: a bounded probe entrypoint (or the materializer with chunks≤2 /
   pairs≤8) fired ONCE by you IS permitted for diagnosis ONLY (cost ≈ cents; record call id +
   lane row; single-flight: the failed claim is closed, claim your probe row properly and
   close it terminal at exit). NO full n600 fire from the arm.
4  (numbering deliberate) — the FULL fire stays MAIN-owned.
2. Fix the defect at source (2 genuine review passes), keep the jo1u contract: chunks ≤120,
   every payload retained w/ sha, deterministic repeat checks, storage preflight unchanged,
   training entrypoints stay blocked.
3. RESEAL (new compiled_config + FIRE_ORDER.json under the same seal_r1 dir lineage, version
   bumped) and emit READY_TO_FIRE for MAIN with the exact argv.
4. Two-landing: the structured remote-failure receipt (suspect 2's cure) lands regardless of
   which suspect is guilty — this failure class (empty RemoteError, no logs, no artifacts) must
   never be silent again on this dispatcher.

## HARD CONSTRAINTS

upstream/ READ-ONLY · serializer commits w/ post-edit sha · ALWAYS KEEP THE PAYLOAD · bulk →
/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/ · frozen packet
untouchable · the fx5_e1 archive bytes (sha 4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841)
are custody — read-only inputs.

## PRIOR NEGATIVE SIGNAL

The r1/r2 LOCAL refusals were ledger-state (phantom fx5 claim · pre-staged-claim self-collision,
#1167 genus) — both cured; do NOT re-diagnose those. The r3 dispatch itself was clean
(DISPATCHED receipt, dispatcher self-claim recorded). The failure is REMOTE-side only.

## OPTIMAL FORM

- Family exemplar — receipt: .omx/research/ddm_jo1u_payload_unblock_20260821.md pinned at
  commit 032e352f71ea091c1b412bca3a1e51282d1a6921 (the build this arm cures); seal receipt:
  /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r1/compiled_config.json
  sha a217a5273536a98766379d89f5834026aedbd5ccdb4b63a3e5f9165ba7a0dc40; failure receipt: the
  MAIN-measured evidence triple above (empty RemoteError · empty logs · absent volume dir).
- MECHANISM reductions FORBIDDEN in the cure (real archive decode, real T4 path); the bounded
  diagnostic probe is a declared SCOPE reduction (chunks≤2) and cannot produce the payload
  verdict — only the defect diagnosis.
- **PRIOR-LAW PREDICTION (falsifiable):** M2 (plumbing ×3 today) predicts suspect 1 —
  a local-path/upload-leg binding defect, not new physics. Counter: a genuine Modal
  runtime/serialization limitation. Count which lands.

## DELIVERABLE

.omx/research/ddm_jo1u2_materializer_cure_20260821.md — root cause (receipt-pinned) · cure
diff summary · probe receipt · new seal + FIRE_ORDER · READY_TO_FIRE|BLOCKED(named). Serializer
commit. End with the own-vehicle frontier line.
