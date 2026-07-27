# G82 — rich TSPPV2 to compact PVSA1 lowering

Date: 2026-07-27  
Lane: `lane_g82_tsppv2_to_pvsa1_compact_lowering_adapter_20260727`  
Authority: bounded local compiler-to-wire and receiver-equivalence proof  
Status: research-only; no score, candidate, public inflate, or full-n600 claim

## Outcome

G82 closes G80's named rich-IR lowering debt without changing the now-owned
G77 or G80 modules. The new additive adapter:

1. strictly parses and re-emits exact TSPPV2 bytes;
2. verifies semantic-P identity, target custody, and decoder/runtime identity
   against separately supplied external identities;
3. lowers only the exact counted G74RA1 operand into PVSA1;
4. builds and strictly reopens both STORE and DEFLATE outer encodings;
5. builds the zero-actuator PVSA1 baseline through the same container;
6. proves rich TSPPV2/G74 and compact PVSA1 produce identical pair-0 native
   camera, support, and execution-receipt bytes; and
7. emits a canonical external lowering receipt binding the rich packet,
   compact member/archive, baseline, source/runtime custody, and proof facts.

The compact member contains no TSPPV2 packet, header JSON, target hashes,
decoder hashes, Torch-version string, or complete semantic-P ZIP. The exact
52-byte G74RA1 operand is present once and is byte/SHA identical to the rich
factor body.

## Triality

DSL:

```text
lower(
  rich = TSPPV2(P_id, target_id, decoder_id, G74RA1),
  P,
  expected(P_id, target_id, decoder_id)
) -> PVSA1(P_payloads, G74RA1) + external_receipt
```

DAG:

```text
external compile/target/runtime custody
  + strict rich TSPPV2
  + exact semantic P
  -> verify identities
  -> preserve exact G74RA1 operand
  -> zero-actuator PVSA1 baseline
  -> one-actuator PVSA1
  -> strict STORE and DEFLATE reopen
  -> rich pair-0 decode == compact pair-0 decode
  -> canonical external lowering receipt
```

Equations:

```text
A_compact = body_G74RA1(TSPPV2)
SHA(A_compact) = SHA(A_rich)

B_semantic = bytes(PVSA1(P, []))
B_actuated = bytes(PVSA1(P, [A]))
Delta_A_same_container = B_actuated - B_semantic = +57

Delta_container = B_semantic - bytes(P) = -4606
```

The last two deltas are deliberately separate. The `-4,606` bytes come from
the semantic container recode. The actuator's exact marginal in that same
container is `+57` bytes.

## Exact inputs and lowering receipt

Fresh semantic P:

```text
bytes   133941
sha256  759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
```

Rich TSPPV2:

```text
bytes          3593
sha256         8bd80138198fb01d9c2a376c8918634eb0e7f08f12a15d6eb9a888dc941fb6bb
framing bytes  3541
operand bytes  52
operand sha256 5616799adc0d2ab942f37a20b070f7a0fa48119771e8f1b56c1f45e2605306ca
selector       BOTH
window         [0,600)
```

External identities:

```text
semantic identity sha256 1583276ad2f8166cf8b9119d221d0dc7894558681b63be6a03a79e7c5c0cfc28
target identity sha256   fc17dcf09682af8ae411546136ef1e6b8eb98e8c4175f455bad22008b01ab149
decoder identity sha256  db7ffbaaf5cd3a5453f367741ea750f6bd180ca92ebf87859946e0206d3fba0b
Torch runtime            2.12.1
```

The canonical external receipt produced by the adapter is 3,249 bytes,
SHA-256 `08a411af145228ff3d615a91eb395b7ada205bee55e8666cfd22ad45871dcc65`.
It is not included in the compact archive.

## Exact same-container pricing

Semantic-only compact baseline:

```text
member bytes   133306
member sha256  6208ac91c465caa8990f7d643f50c06da28c1e00ca359d4ee55005818cc12352
archive bytes  129335
archive sha256 fa173ef4f75adbe9194d3cd89b04021dabd2b9e9fd3aa87081148b6b42a26c75
```

Same container plus exact rich operand:

```text
member bytes    133363
member sha256   d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31
STORE bytes     133471
STORE sha256    144d68dae35534fcad11edb216540e3e5a5b10b88b735ab2af5b28d75b418384
DEFLATE bytes   129392
DEFLATE sha256  b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd
selected        DEFLATE
```

Both compact member and selected outer archive grow by exactly 57 bytes. This
fixture uses G77's current `BOTH` selector, so its exact hashes differ from
G80's earlier Y1 transport fixture while retaining the same exact marginal.

## Receiver equivalence

The adapter opens exact P through the bound TSPPV2/G74 path and independently
through parsed PVSA1. Pair 0 is decoded through both real native paths.

Proven equal:

- camera tensor bytes;
- changed support values;
- owned camera values; and
- serialized G74 execution receipt.

Exact evidence:

```text
camera sha256            caf69dade383564ef8123149d193052b8b5b641711fed232b25dd67b29af25db
execution receipt sha256 b56a2c418460f0c0d3957f88d647aa4a5eac6d568999a4a04b6a0abf36d4991d
```

G74's inner double decode and TSPPV2's outer double decode remain active. No
scorer, evaluator, Pose network, or submission runtime is invoked.

## Negative and compatibility proofs

G82 fails closed on:

- rich packet mutation or parse/re-emit drift;
- semantic-P length/SHA mismatch;
- external semantic, target, or decoder identity mismatch;
- operand byte/SHA drift during compact lowering;
- STORE/DEFLATE compact parse-back drift; and
- any rich packet/header/custody/runtime string appearing in the compact
  member.

The G77 and G80 source files were not edited. G82 consumes their public typed
interfaces and binds their exact source hashes in its external receipt.

## Verification

```text
G82 focused: 7 passed in 39.31s
adjacent G80 + G77 + outer codec: 31 passed in 53.64s
ruff: passed
py_compile: passed
```

The adjacent replay includes G80's zero-actuator baseline, cached bounded
receiver, strict compact parser, and outer STORE/DEFLATE codec.

## Open blockers

1. `PVSA_CONDITIONAL_Y0_GIVEN_Y1_ACTUATOR_TYPE_AND_TRANSITION_OWED`
2. `PVSA_PUBLIC_INFLATE_SH_RUNTIME_INTEGRATION_OWED`
3. `PVSA_FULL_N600_DOUBLE_DECODE_AND_UPSTREAM_EVAL_OWED`

G82 removes the lowering-receipt blocker only at bounded local authority. It
does not establish public inflate closure, a complete 600-pair actuator
population, component distances, Pose safety, exact contest CPU/CUDA score, or
frontier promotion.

## Pointer-delta honesty

The exact pointer is unchanged. G82 proves a real compiler-to-wire lowering and
an exact actuator marginal; it does not produce or evaluate a candidate.
