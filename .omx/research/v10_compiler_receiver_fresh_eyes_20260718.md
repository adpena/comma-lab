# V10 compiler / receiver — three-clean-pass receipt

Date: 2026-07-18  
Completed at: 2026-07-18T15:14:43Z  
Lane: `v10_compiler_receiver_20260718`  
Verdict scope: `V10 COMPILER / RECEIVER / BYTE-CUSTODY STRUCTURAL CERTIFICATE`  
Authority axis: `[local-CPU structural/non-score]`  
Pointer delta: `0`

## Frozen candidate

All three counted passes reviewed the identical seven-file candidate:

| file | SHA-256 |
|---|---|
| `src/tac/witness_dsl/v10_compiler_receiver.py` | `d6ede78ba220daab18a50e5124ca0c6a7f55658cfafd4def2cd35d51dda3c2c2` |
| `src/tac/tests/test_v10_compiler_receiver.py` | `1d05f697a5f08c5174e40f36e6af8da1a54a1f2d9baabd3246576bf6e10d3a5c` |
| `.omx/research/BUILD_SPEC_v10_compiler_receiver_20260718.md` | `00bf0690107aa0d87dd534df2c1476bbd479b4aee6535c54e81b5ef2ff719bdf` |
| `.omx/research/v10_compiler_receiver_20260718.md` | `ea213692eb019187004af8013a53a922915c412485cf7a46824f5acc10c4efdb` |
| `.omx/research/v10_compiler_receiver_DAG_FEED_20260718.md` | `e5701702afde308d01cc13cdf06f2387e97f6855ebf0db89d6bc87d6f6a9293a` |
| `.omx/research/v10_compiler_receiver_equation_candidates_20260718.jsonl` | `8abc87fe921ea049e835b25c9b93c6483562a47d41a7a510ff5c681e8b12af49` |
| `.omx/research/inverse_solve_completeness_matrix_20260718.md` | `4ca7d32feda64b854e7113b70826c74c356c8a0ce8b708be974f06b15e76696f` |

## Reset history

The initial review sequence was **not clean** and reset the counter to zero.
It found the wrong factor map, an overclaimed #332 audit, trusted parsed-state
input, caller-controlled route/video metadata, custom-handler authority,
one-shot completeness loss, permissive scalar/checkpoint schemas, and stale
triality claims. A second pre-audit found handler-source custody absent,
bookkeeping-only semantic changes, a false seed-reachability boundary,
exported-registry rebinding, suffix-only resume receipts, noncanonical base64,
and modulo aliases. Each finding landed as code, tests, or corrected evidence
before the final sequence began. No failed or classifier-terminated review was
counted.

## Consecutive independent clean passes

| pass | independent reviewer | result | gates |
|---|---|---|---|
| 1 | `ordinary_correctness_qa_pass1` | `CLEAN PASS 1` | 7/7 SHAs; 60 tests; Ruff F/E9/I; `py_compile`; `git diff --check` |
| 2 | `formal_correctness_qa_pass2` | `CLEAN PASS 2` | 7/7 SHAs; 60 tests; cross-process identity; full ordered resume receipts; Ruff; `py_compile`; diff-check |
| 3 | `formal_correctness_qa_pass3_replacement` | `CLEAN PASS 3` | 7/7 SHAs; 60 tests; end-to-end claim/code consistency; Ruff; `py_compile`; diff-check |

The assumption-challenge pass kept the claim at structural-reference scope:
disjoint/frozen quotient parameters do not imply a globally injective
representation or production residual optimum. No such numerical claim is
made.

## Independent local verification

- **MEASURED:** focused suite `60 passed`.
- **MEASURED:** adjacent provenance, typed-config, LawRef, and curriculum DSL
  suites `274 passed`.
- Two fresh processes emitted identical program SHA
  `74acfbf9d0f304d4f0c731d805f752dd7e46ee34e4977fca894503c569594cc1`,
  receiver-output SHA
  `beb5e37256d45b94da4c8d9346785dfdcaa0912520c6c5627515b75bdb78173c`,
  and handler-registry SHA
  `5688bc16c7c73d5e89f5d77e9960771615a65831596cc4d3aff12f590128d7cd`.
- Ruff F/E9/I, `py_compile`, and `git diff --check` were clean.

## Verdict and remaining blockers

`THREE_CLEAN_LOCAL_STRUCTURAL_PASSES`.

Factors `2` and `10` remain literal `MISSING/BLOCKED` rows with no paid
section. Strict #332 debt remains explicit and non-authorizing. Production
renderer parity, realized uint8/resize scorer interaction, byte-close archive
custody, contest-CPU and contest-CUDA replay, adoption, launch, score,
promotion, and frontier-pointer movement remain owed. MAIN landing review is
required before integration.
