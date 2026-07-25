# Codex findings — DDM LA1 layer assignment and context pricing

Date: 2026-07-25 UTC  
Lane: `lane_ddm_la1_layer_assignment_context_pricing_20260725`  
Authority: delegated #669(b+c), research-only, MAIN-review-required  
Evidence axis: `[macOS-CPU local lossless-byte advisory]`  
Pointer: `0.1910828242 [contest-CPU]` UNMOVED  
Score claim: false  
Promotion eligible: false

## Verdict

`CONTEXT_PRICED_OUT_INSTANCE_TYPED_MASS_LT_1_PERCENT;`
`SEVEN_HOME_LA1_ALTERNATIVE_BEATS_POST_CC3_COORDINATED_ACCOUNTING;`
`E5_RECEIVER_INTEGRATION_PENDING`

The exact real-coder race prices an LA1 alternative at **128,254 bytes**,
`-5,957 B` versus LP1's 134,211-byte seven-home accounting object. CC3's
130,789-byte coordinated total and LA1's 128,254-byte total are overlapping
alternatives, so their deltas must not be added. The prospective coordinated
best case is therefore **128,254 B**, or **-2,535 B versus 130,789 B**.

This is not yet a receiver-closed candidate. The current receiver-closed
accounting remains **130,789 B** until E5 consumes the selected frames and
proves a counted-payload/receiver-consumer bijection.

Typed receipt:
`.omx/research/ddm_la1_layer_assignment_context_pricing_20260725T112908Z/receipt.json`

Receipt SHA-256:
`7975d6fd63b86654bde84b0e922de58bd3f9ebb93f94ac6996b8614b0d38cca5`

## #669(b) residual versus decoder-derived context

Every arm pays the same 46-byte self-delimiting envelope: codec identity,
raw length, and raw SHA-256. Explicit ownership races identity,
Brotli-Q11, and stdlib raw-LZMA1. Context ownership races the landed G4
previous-byte/bit-prefix model and the CC2/CC3 Bellard mixture of four
decoder-derived KT experts. Every frame parsed back exactly and reproduced
deterministically.

Here `ΔB = context - explicit`, so a negative value selects CONTEXT.

| exact EV2 stream bucket | explicit bytes | explicit coder | context bytes | context coder | winner | ΔB |
|---|---:|---|---:|---|---|---:|
| manifest | 1,314 | Brotli-Q11 | 1,960 | Bellard-KT | RESIDUAL | +646 |
| v15 predictor ZIP outer payload | 96,563 | Brotli-Q11 | 99,821 | Bellard-KT | RESIDUAL | +3,258 |
| G1 movable worldsheet payload | 29,856 | raw explicit | 29,858 | Bellard-KT | RESIDUAL | +2 |
| receiver realization profile | 69 | raw explicit | 68 | Bellard-KT | CONTEXT | -1 |
| solved template payload | 97 | raw-LZMA1 | 98 | Bellard-KT | RESIDUAL | +1 |
| central directory | 268 | raw-LZMA1 | 284 | Bellard-KT | RESIDUAL | +16 |
| Lane seed payload | 102 | raw-LZMA1 | 88 | Bellard-KT | CONTEXT | -14 |

CONTEXT wins the 85-byte receiver-profile accounting home and the 270-byte
Lane-seed accounting home: `355 / 134,211 = 0.002645088703608497`, or
**0.264509%**. This is below the preregistered 1% threshold, so CONTEXT is
priced out at `INSTANCE(current seven-home stream geometry)` scope. The family
reopens on a scorer-recursive same-object geometry where context wins at least
1% after uniform framing.

The 270-byte Lane home is not pretended to be one contiguous payload. J2's
sealed 134,211-byte archive was reconstructed byte-exactly; the irreducible
receiver-consumed Lane program is 90 bytes. The other 180 bytes are manifest
and ZIP framing changes. LA1 counts the full 270-byte source home before
measuring the lossless 90-byte semantic re-home.

## #669(c) whole-stream layer assignment

Rate remains stream-level. No byte was allocated to any EV1/EV2 scorer cell.
The layer assignment preserves LP1's scorer-recursive home and prices the
same exact payload at every admissible candidate home. Re-tagging alone earns
no extra coding gain; unmaterialized deeper representations remain `NULL`.
L5 payloads are refused because scorer weights and ground-truth tables cannot
be shipped.

| stream | source home | semantic payload | deepest admissible layer | selected coder | re-homed bytes | measured ΔB |
|---|---:|---:|---|---|---:|---:|
| manifest | 3,345 | 3,302 | L1 program | Brotli-Q11 | 1,314 | -2,031 |
| v15 predictor ZIP outer | 100,099 | 100,056 | L2 chart | Brotli-Q11 | 96,563 | -3,536 |
| G1 movable worldsheet outer | 29,878 | 29,810 | L2 chart | raw explicit | 29,856 | -22 |
| receiver realization profile | 85 | 23 | L1 program | Bellard-KT | 68 | -17 |
| solved template outer | 151 | 86 | L4 scorer feature | raw-LZMA1 | 97 | -54 |
| central directory | 383 | 383 | L1 program | raw-LZMA1 | 268 | -115 |
| Lane seed | 270 | 90 | L2 chart | Bellard-KT | 88 | -182 |
| **total** | **134,211** | — | — | — | **128,254** | **-5,957** |

Payload-cleanliness framing was run on every row:

- generic interpreter bytes are zero-counted;
- every selected frame counts the full video-derived payload;
- no hash, table, or video statistic is moved into code;
- no manifest/container byte is declared free before E5 proves generic
  derivation and exact receiver parse-back;
- lossless frame parse-back proves the byte object, not an integrated
  distortion or contest result.

## Coordination and disposition

The lawful coordination is:

`min(130,789 post-CC3, 128,254 LA1 alternative) = 128,254 B`.

It is not:

`134,211 - 3,422 - 5,957`.

That additive expression double-counts overlapping coding changes. E5 should
consume exactly the seven LA1 selected frame identities in the receipt,
materialize a receiver-closed export, and rerun the payload-consumption
bijection. Only then may 128,254 replace 130,789 as confirmed composition
accounting.

GA1 was not rerun. Its
`DOMINATED_INSTANCE_CURRENT_LP1_COMPOSITION` disposition and four named
reopeners are carried verbatim in the typed receipt.

## Verification

- Exact C1 source: 133,941 B,
  SHA-256 `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- Exact J2 seeded reconstruction: 134,211 B,
  SHA-256 `3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd`.
- Seven source homes sum independently to 134,211 B.
- Seven selected frames sum independently to 128,254 B.
- Row deltas sum independently to -5,957 B.
- Context-winning accounting mass sums independently to 355 B.
- 29 adjacent tests pass, including 7 focused LA1 tests.
- Three fresh clean review passes, Ruff, formatting, `py_compile`, independent
  accounting re-derivation, deterministic replay, and `git diff --check` pass.
- A pre-seal review finding that allowed output outside the durable repository
  was fixed fail-closed and covered by regression test.

## MAIN landing review

MAIN must independently verify before merging:

1. the uniform 46-byte framing and exact parse-back of all five coder arms;
2. that raw-LZMA1 really uses `FORMAT_RAW`, 1 MiB dictionary, `lc=3`,
   `lp=0`, `pb=2`;
3. the 90-byte Lane semantic payload versus 270-byte source-home accounting;
4. the stream-level-only EV2 join law and the `NULL` deeper-home rows;
5. the non-additive CC3/LA1 coordination and prospective-only 128,254 B;
6. the `<1%` falsifier's `INSTANCE` scope and named reopener;
7. that score, promotion, dispatch, and pointer authority remain false.
