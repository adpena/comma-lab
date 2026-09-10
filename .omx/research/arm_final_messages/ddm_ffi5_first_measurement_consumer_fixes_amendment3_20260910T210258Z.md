Implemented all three fixes, including MAIN’s later charter correction. **Corrected files are ready for MAIN to land; the frontier is unchanged.**

| Finding | Code | Test evidence |
|---|---|---|
| Runtime path mismatch | Content-only validation through dispatcher → worker → evaluator; tree hashes retained | Relocation passes; one-byte change refuses; real rlc5 provenance passes |
| Snapshot ledger writes | Explicit repo ledger and matching lock | Real append and lifecycle check pass with snapshot defaults |
| Authorize import failure | Repository root added to `sys.path` | Foreign-cwd subprocess passes without `PYTHONPATH` |

**Validation:** 146 required tests + 181 adjacent tests passed; one existing socket test was sandbox-blocked. Ruff clean; two corrected review passes on all five Python files.

**Serializer rc 17:** Git object writes denied. Its pre-correction bundle is **obsolete—do not land it**. The corrected [source patch](/Users/adpena/Projects/pact/.omx/research/ddm_ffi5_20260910/FINAL_SOURCE.patch), 16-file manifest, freeze draft, and [memo](/Users/adpena/Projects/pact/.omx/research/ddm_ffi5_first_measurement_consumer_fixes_amendment3_20260910.md) are retained.

rlc5’s measured digests:

- Intent runtime: `f4c6c250cb1d806d6cafbc4d6cfed82f8f59d95fe432db1925637571b624c669`
- Normal projected tree: `e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891`
- Retained worker tree: `baeb53afc8d1bb0c43bb2ce91fe1e09e845809b76fc3047bc8ed3397254155af`
- **Content digest, local = worker:** `e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf`

No Modal, authorization, timing, scorer, or SSD writes. Protected files and staged index preserved; checkpoint `ddm_ffi5` complete.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43) unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; store `ddm_ffi5_20260910/FINAL_SOURCE.patch`; trigger packet harvest:** land corrected files, verify all 16 committed manifest rows, and append amendment[3].
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; store `ddm_ffi5_20260910/PYTEST_CORRECTED.txt`; trigger unrestricted host:** rerun the sandbox-blocked socket test.

## LIVE-HYPOTHESES

- A fresh dispatch can pass the observed runtime-identity failure: retained worker provenance now passes content validation. Remote execution and completion remain untested.

## DEAD-ENDS

- Run3 used the seal digest: disproved by retained argv and request.
- Pinning the normal tree hash alone fixes retention: disproved by exact path-projection reproduction.
- Landing fallback `011a0345`: superseded by the corrected implementation.