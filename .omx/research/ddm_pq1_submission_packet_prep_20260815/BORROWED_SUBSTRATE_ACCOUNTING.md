# Borrowed-substrate accounting — generation 2 (rr4 re-encode candidate)

This accounting is mechanism-level and deliberately unflattering to us. The
archive is a new exact-byte composition, but it is **not** a wholly original
learned vehicle, and this table exists so that nobody has to infer which is
which. Categories are closed: `ours-original`,
`PR130-lineage-retrained-on-our-labels`, `PR130/135-byte-identical`, or
`PR135-lineage-modified`.

**The one-line honest summary.** Seven of the archive's eight sections are
byte-identical to the base candidate we inherited. This generation's own
contribution is the eighth: a decode-time probability corrector that re-encodes
the token stream losslessly, saving 1,598 bytes and changing nothing else. The
learned content — the semantic renderer, the carrier, the HPAC probability
object — is PR130/PR135 lineage and we do not claim it.

| Section or mechanism | Classification | Exact receipt | What the classification means |
|---|---|---|---|
| Semantic renderer state | `PR130/135-byte-identical` | decoded section 36,051 B, SHA-256 `b489c73567046e64…`; byte-identical to base | The learned semantic renderer descends from PR130/PR135. No originality claimed. |
| Carrier state | `PR130/135-byte-identical` | decoded section 22,242 B, SHA-256 `196f0e5136f4d6bf…`; byte-identical to base | The learned carrier descends from PR130/PR135. No originality claimed. |
| Compressed model container | `PR130/135-byte-identical` | 70,453 B, SHA-256 `e35d12371fa79747…`; byte-identical to base | Carried through unchanged from the base candidate. |
| HPAC probability object | `PR130-lineage-retrained-on-our-labels` | 17,952 B, SHA-256 `e8c0cfd73d3275ad…`; byte-identical to base | Architecture descends from PR130; this object was trained on our label field, but in THIS generation it is inherited unchanged from the base. |
| Compensation blob | `PR130/135-byte-identical` | 36 B, SHA-256 `38792b4953318117…`; byte-identical to base | Carried through unchanged. |
| Residual payload + table codes | `ours-original` (fitted earlier), inherited here | residual 100 B, SHA-256 `74775aab04c7615c…`; table codes SHA-256 `76afdc3ceda1212a…` | Fitted by earlier repository work; unchanged in this generation. |
| **RC64 token stream (the only changed section)** | **`ours-original` mechanism over borrowed probabilities** | 110,512 B, SHA-256 `6c3757bd52a18d3c…`; base was 112,110 B, SHA-256 `73a878891a31c366…` | The **free decode-time corrector** (`ddm_rr4_free_corrector_v2`, SHA-256 `96fd35aaf82c737a…`) is ours. It adjusts the probability model from already-decoded symbols, stores zero bytes, and re-encodes losslessly. The **probabilities it corrects** are the borrowed HPAC object's. So: our estimator, their base model. |
| RC64 range-coder backend, encoder side | `PR130/135-byte-identical` | compiles `rc64_backend.c` SHA-256 `5c75e2c70b89f148…` taken from the PR135 intake | Our encoder compiles the PR135 range-coder source unmodified. This is a direct borrow and is disclosed as one. |
| RC64 range-coder backend, shipped receiver | `PR135-lineage-modified` | shipped `runtime/entropy/rc64_backend.c` SHA-256 `05839d1416e68a49…`, which **differs** from the PR135 source above | The shipped decoder's coder is a modified descendant of PR135's, not a byte-identical copy. Stated explicitly so the difference is not mistaken for either full originality or a clean copy. |
| Receiver binding and archive assembly | `ours-original` | runtime tree SHA-256 `7acedb07e670e76c…`; archive `35ac2b9beb7e6fa8…`, 181,161 B | The composition, the receiver regeneration that re-pins the archive hash, the deterministic repeat, and the custody chain are repository work. They do not erase the section-level borrowing above. |
| End-to-end compression entry point | `ours-original` | `experiments/ddm_pq2_compress_e2e.py`; rebuild verified 2026-08-17 | The sanitized reproduction path and its fail-closed hash assertions are ours. |

## What we claim, stated narrowly

A **lossless entropy re-encode** of an inherited PR130/PR135-lineage archive,
using an original zero-byte decode-time probability corrector, yielding 1,598
fewer archive bytes with a provably unchanged decoded field, plus an original
CPU-capable receiver port, custody chain, and reproducible build path.

## What we do not claim

We do not claim the learned vehicle, the semantic renderer, the carrier, the
HPAC architecture, or the range-coder design. We do not claim a new distortion
result: `d_seg` and `d_pose` are unchanged from the base candidate by
construction, and we say so rather than presenting an unchanged number as an
achievement.
