# V10 A2 profiler review FIX12 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Two independent reviews of exact ordered FIX11 bundle
`857ca79567c906d85525e8de10ce704d7078be2635ecc501a63a2ab0cfd204cd`
reproduced one P1 operational / P0-resumability defect. A crash can occur
after the first `staging_scratch.json` record is fully written and fsynced to
its stable `.creation-prepared` path but before the atomic rename to the final
scratch name. The retry bootstrap inspected only the final name, so it derived
the stable identity from the new preflight. Initialization then found the
complete prepared record but rejected its legitimate first-attempt volatile
observations as drift.

Round 10 is `NOT_CLEAN`; the one scoped clean review of FIX11 does not seal the
surface.

## Required implementation

- Treat a complete canonical, regular, link-count-one prepared first-scratch
  record as certified pre-final bootstrap evidence when the final scratch name
  does not yet exist.
- Validate its identity hash, exact rebuild argv, stable storage projection,
  rebuildability flags, output root, and schema against the freshly derived
  request before adopting it. Stable selection drift remains fail-closed.
- Preserve the prepared record's complete first preflight and atomically finish
  its rename; do not substitute retry-time volatile observations.
- Continue treating a malformed partial prepared first-scratch record as
  certified rebuildable pre-stage scratch: preserve fail-closed link/path
  checks, then rebuild it from the newly passed current preflight.
- Independently rerun the current capacity preflight before either path. No
  prepared record grants capacity, launch, score, or promotion authority.

## Required regressions

- Interrupt after the complete prepared first-scratch write/fsync and before
  rename; then rerun storage preflight with both free-space and sampling-anchor
  drift. Recovery must preserve the first full preflight across scratch,
  progress, and certification and finish with the same identity.
- The existing malformed-partial prepared-scratch recovery remains green.
- A complete prepared record with stable tier-selection drift is rejected and
  preserved.
- Final-scratch, later prepared-file, final-directory-rename, and ordinary
  `--resume` interruption paths remain green.

## Stop rule

Do not freeze source or launch until the replacement exact-byte bundle passes
three consecutive zero-finding reviews. This fix and its tests grant no score,
promotion, Pose, factor-10, global-compression-minimum, or contest-axis
authority.
