# `rc64_backend.c` — four bodies wear this name

Measured 2026-08-19 by `ddm_rv14f` (rv13 F13). Machine-readable sibling:
`rc64_backend_role_registry.json`.

**Read this before searching for `rc64_backend.c` by name.** A name-keyed search that
stops at the first hit can land on any of four different bodies. That has already cost us
twice: a sealed fire order declared the pinned body nonexistent and named an unnecessary
fix, and `ddm_rc1x`'s cure — which was otherwise the best instrument work in that wave —
enumerated **two** roles when there are **four**.

## The census

Scope: `/Volumes/VertigoDataTier/pact` + `/Volumes/APDataStore/pact` +
`/Users/adpena/Projects/pact`. Method: `find -name rc64_backend.c -not -name '._*' -type f`,
then sha256 every hit. **241 files, 4 distinct contents.**

| sha256 (16) | bytes | copies | role |
|---|---:|---:|---|
| `05839d1416e68a49` | 5,638 | 237 | **shipped receiver, decoder-only** — inside every archive's `runtime/entropy/` |
| `1941923a94e4e0a1` | 14,825 | 2 | **checkpoint-extended encoder** — under the *plain* name, at `ddm_rc64p_20260810/{runtime,runtime_optimized}/` |
| `b249b77bb06a27c8` | 22,179 | 1 | foreign intake (PR138 `opal_v1`) — not ours |
| `5c75e2c70b89f148` | 12,222 | 1 | **encoder — THE PIN** — `pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/` |

## The pin, and the false blocker

`experiments/ddm_pq2_compress_e2e.py`'s default recipe pins
`rc64_source_sha256 = 5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6`.

`.omx/research/ddm_fx2_t4_sealed_fire_order_20260818.json:19-27` asserts that **no file on
either SSD matches it**. That is false. The file has been on VertigoDataTier since
2026-08-10 — eight days before the fire order declared it missing — at

    /Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c

12,222 B, sha `5c75e2c70b89f148…`. Disproved by direct hash three times independently:
`ddm_rc1x`, `ddm_rv13`, and this registry. The byte-close is **not** blocked at input
verification, and the `fix` the fire order names is unnecessary.

## Why the two-role table was one body short

`ddm_rc1x` describes the 14,825 B build as `rc64_backend_checkpoint.c`. On disk that exact
content **also** sits under the plain name `rc64_backend.c`, twice. So the
name-without-role hazard rc1x correctly diagnosed is one body worse than rc1x measured —
and the extra body is **encoder-class**, precisely the class whose absence the false
blocker turned on. A search that found `1941923a94e4e0a1` and concluded "found an encoder,
but the sha doesn't match the pin" would reproduce the false blocker exactly.

This does not weaken rc1x's verdict. The roles it names are real and its byte-close stands;
its control ladder (P1–P5 plus A–E) remains the reference for this campaign.

## The corroboration that ties the two encoders together

rc1x's P1 control claims that subtracting the 2,603 B checkpoint/resume block from the
retained 14,825 B body yields the pin. Arithmetic, verified from the filesystem rather than
from rc1x's prose:

    14,825 − 2,603 = 12,222      ← exactly the pinned encoder's size

## How to use this

- **Never key on the filename.** Key on sha256. `find -name rc64_backend.c` has 241 hits.
- **Encoder vs receiver is the distinction that matters.** The 5,638 B body is decoder-only
  and is what ships; the 12,222 B and 14,825 B bodies are encoder-class and never ship.
- **Before recording a "file not found" for a pinned sha**, hash the candidate set and say
  what scope you searched. `m53`: a negative-existence claim needs an exhaustive search or
  the honest form *"did not find in `<scope>`"*. All three published copy-counts for this
  sweep (`ma1` 158, `fx2` 252 `.c` files, `rc1x` 232) are different scopes, and none stated
  its scope beside its number.
- **Sibling sweep still open.** rc1x asked for it and it has not run: *any other
  recipe/driver pin whose file is named generically and located off-tree.*

## Sources

- `.omx/research/ddm_rv13_landing_wave_review_20260819.md` §F1, §F8, §F13
- `.omx/research/ddm_rc1x_rc64_recipe_fix_20260819.md`
- `.omx/research/ddm_fx2_t4_sealed_fire_order_20260818.json` (the false blocker; see the
  supersession note appended to it)
- `reverse_engineering/rc64_backend_role_registry.json` (machine-readable)
