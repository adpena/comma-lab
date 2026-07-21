# V10 A2 profiler review FIX13 specification — 2026-07-18

`research_only=true`  
`authority_axis=[macOS-CPU advisory]`  
`score_claim=false`  
`promotion_eligible=false`  
`pointer_delta=0`

## Trigger

Two independent reviews of exact ordered FIX12 bundle
`41aa959e5a8bf4ac98dde1f8f312bf3c9bf7df082806c00cc916a8eb5138923d`
reproduced one high-severity custody defect. Canonical-object loading collapsed
filesystem read `OSError` into the same generic error as malformed/truncated
JSON. Both complete-prepared bootstrap and materialization caught that generic
error as the certified partial-write case, so a persistent/transient read
failure could unlink a complete fsynced first-preflight record and silently
substitute retry-time custody.

Round 11 is scoped `CLEAN`; rounds 12 and 13 are `NOT_CLEAN`. No seal exists.

## Required implementation

- Introduce an explicit canonical-JSON malformation error distinct from
  custody/path/read I/O errors.
- Only positively identified decoding, JSON-parse, non-object, non-canonical,
  or non-serializable canonical-content failures may enter the certified
  partial-prepared rebuild path.
- Any `lstat`, open, read, permission, device, or other `OSError` must preserve
  all bytes and fail closed. It must never trigger unlink, rewrite, adoption,
  score authority, or stable-selection substitution.
- Apply the distinction both when bootstrapping a complete prepared first
  scratch and when recovering every later prepared creation JSON
  (identity/progress/certification).
- Preserve all FIX12 behavior for readable complete prepared records and true
  malformed partial records.

## Required regressions

- Inject a read `OSError` on a complete prepared first-scratch record during
  retry. Recovery must raise, leave the prepared bytes untouched, create no
  final scratch/root, and retain no retry-substituted receipt.
- Inject a read `OSError` on a complete later prepared creation JSON.
  Materialization must raise and preserve the prepared bytes without creating
  or replacing the final file.
- Truncated/non-canonical regular link-count-one prepared records still rebuild
  through the certified partial path.
- Complete readable prepared scratch with volatile re-preflight drift still
  recovers and retains the first full preflight; stable drift still blocks.

## Stop rule

Do not freeze source or launch until the replacement exact-byte bundle passes
three consecutive zero-finding reviews. This fix and its tests grant no score,
promotion, Pose, factor-10, global-compression-minimum, or contest-axis
authority.
