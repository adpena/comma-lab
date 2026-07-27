# G94 — sequential typed actuator product

Status: **implemented and exact-fixture-closed; research-only**  
Lane: `lane_g94_sequential_typed_actuator_product_20260727`  
Goal pointer: effective competitive frontier `0.172`; G94 does not move it.

## Causal product

G94 implements one fixed-order counted member:

`[base PVSA1 with exact incumbent G74 BOTH, G89 class-complete Y1 program on the same semantic P, G88 conditional Y0|combined-Y1 operand on the same base/P]`

The receiver:

1. decodes exact incumbent Y0 from PVSA1;
2. opens G89 against the same semantic P;
3. dynamically merges incumbent boundary atoms into G89's native Y1 state,
   refusing every donor-address collision;
4. renders `(exact incumbent Y0, combined native Y1)`;
5. applies G88 to Y0 only and proves combined Y1 unchanged.

The merge is generic receiver behavior. All three video-specific operands live
exactly once in the counted member.

## Wire and receiver invariants

- versioned `G94SEQ1` member with fixed section order and lengths;
- SHA-256 for every section, product CRC32, strict nested parse, and exact
  parse/re-encode equality;
- exact foreign keys from G89 and G88 to the base member and semantic P;
- exact base requirement: one G74 `BOTH` operand;
- ≤16 contiguous pairs per streaming decode;
- deterministic double decode for the preconditional and G88 transitions;
- exact STORE/DEFLATE outer race followed by strict selected-archive reopen;
- every bounded result recomputes all array hashes/change counts, binds the
  exact product member and conditioning-state hashes, and freezes its arrays;
- G83-shaped archive/member/byte-home metadata refuses a bounded proof from
  any other product state, while admission remains false.

## Conditioning custody and G95 boundary

G94 exposes:

`conditioning_state_sha256 = H(domain || base_PVSA_member_SHA || G89_program_SHA || transition_IDs)`

For the exact fixture it is:

`7ab4829d0ecf53b973629be518cc0be575cf826f8a33eceffcb13cb00d678c9b`

Changing the G89 program changes this hash. G88 V1 cannot attest that its
values were fit against that exact Y1 conditioning state. The current G88
fixture therefore has authority only as:

`STRUCTURAL_EXECUTABLE_ONLY_NO_POSE_MARGINAL_TRANSFER_UNTIL_G95_FIT_RECEIPT`

No Pose marginal or score transfers from it. G95 must emit a fit receipt
binding the exact conditioning-state hash before the conditional operand can
be treated as fitted evidence.

## Exact current-base fixture

The fixture uses the exact current G85 base:

- outer base archive: 129,392 bytes,
  `b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd`;
- compact base member: 133,363 bytes,
  `d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31`;
- semantic P:
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.

Counted homes:

| Home | Bytes | SHA-256 |
|---|---:|---|
| base PVSA1 incumbent G74 BOTH | 133,363 | `d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31` |
| G89 class-complete Y1 program | 247 | `50729be3d6803e4fa700be89addc96e2a82f88622614d42805145f38275c1ef8` |
| G88 conditional Y0\|Y1 operand | 198 | `f9170051a46ac71fbb34ca690d02578390c5f2bff16a391ec7f769dd9251e593` |

Exact product and outer race:

- member: 133,929 bytes,
  `84335c287ccee915fafce19f0258dc3fa6939095b5bdd773e09dbf7e47fd934a`;
- STORE: 134,037 bytes,
  `fed4e44ba04c088053c78f3cedf538dc79ef053879694055065766a8d9627c6b`;
- DEFLATE selected: 129,799 bytes,
  `bb8d5c4d75ff0b9c3e4d0c3d9ace7a4291b9fc7f8c818a2bdb55e14b2a313910`.

Pair-0 bounded execution proves both transitions are non-noop:

- G89 changed 12,453 Y1 channel values;
- G88 changed 840,221 Y0 channel values across 433,312 pixels;
- exact final camera SHA:
  `894430c83108c7d66b569a1e09972bb740109267be0013a1274af6158e9e79ea`;
- deterministic double decode: true.

These are structural execution facts, not a useful-distortion, rate-benefit,
Pose, candidate, or score claim. The typed fixture operands are newly built
fixtures and are not historical V15/C1 payloads.

Root adversarial substitution tests reject both a forged result-array hash and
a valid bounded proof produced by a different G89/G94 conditioning state.
Combined G92/G94 focused validation after that hardening is 23 passed; Ruff,
format, and `py_compile` are clean.

## Open authority gates

- G95 fit receipt binding exact conditioning state;
- public `inflate.sh` recursive runtime closure;
- same selected archive full-n600 `upstream/evaluate.py` Seg/Pose/rate row;
- G83 admission using that complete same-state row.

Until all gates close: `research_only=true`, `candidate_claim=false`,
`score_claim=false`, `pointer_moved=false`.
