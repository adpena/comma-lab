# V10 A2 profiler review FIX10 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Two independent reviews of exact bundle
`ad99ec37497e5b6a02cfb06de745c101cc70bf1a34e9b3e6fc7aa71755b1950a`
confirmed one P1 false-certification class. The generated
`existing_approved_roots` snapshot was copied among scratch, progress, and
certification records but was not part of the immutable experiment identity
that roots the stage chain. A coordinated rewrite could omit a higher tier and
regenerate internally consistent custody.

Rounds 7 and 8 are `NOT_CLEAN`; no prior scoped clean result seals this surface.

## Required implementation

- Bind the complete creation-time storage preflight record into immutable
  identity before the identity hash and frame-zero chain root are derived.
- Require scratch, progress, output certification, terminal custody, and final
  receipt validation to use exactly the identity-bound creation preflight,
  never a self-declared sibling copy.
- On resume, safely load only the identity-bound creation preflight needed to
  re-derive the expected identity, then validate the stored identity and full
  stage chain against that hash.
- Preserve P0 resumability when SSD availability changes after creation:
  a fresh run must select the first existing tier, while an existing sacred run
  may resume in its already-bound tier. Current free-space admission remains a
  fresh per-invocation check and is not substituted for creation custody.

## Required regressions

- Omitting the higher tier from scratch/progress/certification while keeping
  the original identity is rejected.
- Coordinated omission plus a rewritten identity changes the identity hash and
  is rejected by the existing stage-chain root.
- Fresh creation binds the exact preflight object into identity.
- Resume reuses that bound creation record even if free bytes or currently
  mounted higher tiers change, while relevant source-byte drift still fails.

## Stop rule

Do not freeze source or launch until a replacement exact-byte bundle passes
three consecutive zero-finding reviews. No storage receipt, static review, or
local test grants score or promotion authority.
