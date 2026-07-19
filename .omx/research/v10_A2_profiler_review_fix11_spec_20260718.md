# V10 A2 profiler review FIX11 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Two independent final-seal reviews of exact ordered bundle
`34e552c7c6a0cac7fb0d22f14cbf0de70e4e8c4a881c4a01eadbef45eb84b22b`
confirmed one P1 operational / P0-resumability defect. FIX10 bound the full
creation preflight into the stage-chain identity, including the volatile
`free_bytes_before` observation. A normal retry after an interrupted
staging-to-final creation therefore derived a new identity whenever unrelated
disk activity changed free space, and refused the otherwise certified staging
directory. `--resume` could not recover it because the final root did not yet
exist.

Round 9 is `NOT_CLEAN`; no earlier scoped clean result seals this surface.

## Required implementation

- Keep creation-time SSD selection custody immutable: waterfall order,
  complete ordered existing-tier snapshot, fresh-first selection scope,
  selected output root, required capacity, local-test policy, and PASS state
  remain identity-bound.
- Exclude only preflight observations that can change solely because the
  creation transaction itself or unrelated disk activity progressed:
  `free_bytes_before` and the nearest-existing `filesystem_anchor` sampling
  path. Name those exclusions explicitly in the identity schema.
- Preserve the complete first successful creation preflight, including both
  excluded observations, byte-for-byte in staging scratch, progress, output
  certification, terminal custody, and final receipt validation.
- On a fresh retry with certified pre-final staging, bootstrap the stable
  creation-storage identity from that staging scratch. Independently rerun the
  current storage-capacity preflight before trusting or continuing any bytes.
- Validate the stored identity, scratch, certification, progress, and complete
  stage-chain root against the same immutable identity hash. Stable custody
  drift must still fail closed; observation-only drift must not change the
  identity or replace the first preflight receipt.
- On ordinary `--resume`, load only the stable identity-bound creation-storage
  anchor from the final identity. The current preflight remains a separate
  live-capacity gate and never rewrites creation evidence.

## Required regressions

- A real two-invocation final-rename interruption changes free bytes and the
  nearest-existing sampling anchor on retry, yet recovers with the same
  identity and retains the first full preflight across all custody records.
- Stable identity is invariant to the two named volatile observations and
  changes when any bound waterfall/selection field changes.
- Omitted-higher-tier coordinated sibling tampering remains rejected.
- Rewriting the stable identity changes the frame-zero hash and remains
  rejected by an existing stage chain.
- Existing sacred lower-tier resume remains valid when a higher tier appears,
  while fresh lower-tier creation remains refused.

## Stop rule

Do not freeze source or launch until the replacement exact-byte bundle passes
three consecutive zero-finding reviews. This fix, its tests, and any local
storage receipt grant no score, promotion, Pose, factor-10, or contest-axis
authority.
