# ddm_rc2 — scope corrected; byte race blocked before materialization

**UTC:** 2026-08-10T04:01:16Z  
**Owner:** codex `ddm_rc2` arm  
**Axis:** `[macOS-CPU advisory]`, scorer-free, `score_claim=false`  
**State:** `BLOCKED-STORAGE-PERMISSION`  
**Base:** PR130 CPR1, `archive.zip` 191,052 B, SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`

## Conclusion

Task #996 did **not** byte-race either family in the reference form required by the rc2 charter.
Its token incumbent is already an HPAC arithmetic/range stream and #996 raced ANS against that
stream under the **same existing HPAC probability model**. It did not build and encode a real
stream with a newly selected per-symbol adaptive context model. It did not build or encode a
BP/LDPC-syndrome stream on a PR130 residual object.

Therefore “CODER AXIS CLOSED on the PR130 base” is valid only for #996's tested candidate set:
the existing token model with Range versus ANS, generic whole-section recodes, and the reported
memoryless/order-1 comparisons. It is not a family-wide byte verdict for adaptive arithmetic or
BP/LDPC syndrome coding. The rc2 reopening was not stale.

The race nevertheless did not run. The mandatory retained-payload destination could not be
created. The common contract says to stop on an SSD permission error and forbids a symlink or
alternate-storage workaround; the rc2 charter also explicitly forbids writing this bulk to the
nearly full Vertigo tier. No candidate payload entered memory, so no payload was discarded.

## #996 scope determination

| Family or comparison | Found in #996 receipts? | Honest disposition |
|---|---:|---|
| Shipped HPAC Range versus ANS, same token model | Yes | Measured on real token bytes; ANS won 2,120 B. This does not test a new adaptive context model. |
| Generic Brotli/LZMA recodes of unchanged sections | Yes | Measured within the receipt's candidate set. |
| Memoryless and selected order-1 comparisons/bounds | Yes | Bounds or restricted comparisons, not a family-wide higher-order context verdict. |
| Reference-form adaptive arithmetic with a real context model and retained stream | No | **UNMEASURED by rc2 and not closed by #996.** |
| Reference-form BP/LDPC-syndrome stream with iterative decode | No | **UNMEASURED by rc2 and not closed by #996.** |

The source receipts are:

- `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`
  at commit `0df79dc0ace6420447a279537037efb334524d3e`.
- `.omx/research/ddm_pr130_reproduce_20260809/RATE_AXIS_LOSSLESS_RACE.md`
  at commit `0eea12ac3554ff67f0a0768881f8c5ea97b83fa3`.
- `.omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md`, read as a downstream
  interpretation rather than an independent coder race.

## Byte and timing ledger

No number in this table is an rc2 measurement. The incumbents and bounds are recalled from the
#996 receipts solely to pin the future comparison surface.

| PR130 section | Receipt incumbent or harvested value | Receipt memoryless comparison | rc2 adaptive-arithmetic bytes | rc2 BP/LDPC bytes | rc2 native seconds |
|---|---:|---:|---:|---:|---:|
| tokens, raw section 116,980 B | ANS 114,860 B | model cross-entropy 114,852 B | **NOT MEASURED** | applicability not established | **NOT MEASURED** |
| semantic, raw section 40,252 B | Brotli-q11 35,033 B | 36,805 B | **NOT MEASURED** | applicability not established | **NOT MEASURED** |
| pose, raw section 23,054 B | shipped Huffman 23,054 B; reported recode 4 B worse | 22,989 B | **NOT MEASURED** | applicability not established | **NOT MEASURED** |
| hpac/models, raw section 20,179 B | Brotli 14,962 B | 16,567 B | **NOT MEASURED** | applicability not established | **NOT MEASURED** |

The semantic and hpac incumbents beating order-0 entropy estimates rules out only the named
memoryless substitution. It does not prove that every higher-order adaptive model loses. The
pose gap of 65 B is a ceiling before model/table/framing cost, not a projected win. No projected
byte count is promoted to a coded-stream result here.

## Mandatory storage preflight and stop receipt

The charter required all materialized candidate streams under
`/Volumes/APDataStore/pact/ddm_rc2_20260810/`. The preflight attempted only to create that arm's
retention directory and a zero-byte writability probe:

```text
mkdir -p /Volumes/APDataStore/pact/ddm_rc2_20260810/retained
touch /Volumes/APDataStore/pact/ddm_rc2_20260810/retained/.write_probe

exit 1
mkdir: /Volumes/APDataStore/pact/ddm_rc2_20260810: Operation not permitted
```

Read-only capacity facts at the stop:

| mount | available 1 KiB blocks | utilization | disposition |
|---|---:|---:|---|
| `/Volumes/APDataStore` | 1,045,735,808 | 47% | Capacity sufficient, but sandbox write permission absent. |
| `/Volumes/VertigoDataTier` | 30,202,148 | 99% | Explicitly forbidden as rc2 bulk fallback. |

Nothing was created on APDataStore. No local or Vertigo bulk fallback was used. No encoder,
decoder, timing loop, candidate sweep, or scorer job was launched.

## RECALL EVIDENCE

### Sources and queries searched

- Full `.omx/research/` content search for `adaptive arithmetic`, `range coder`, `ANS`,
  `context model`, `LDPC`, `belief propagation`, `syndrome`, `STC`, `residual`, `flip`,
  `PR130`, and `#996`; the #996 receipts above were read in full.
- Canonical equations via
  `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `arithmetic`,
  `Markov`, `syndrome`, `LDPC`, and entropy/coder terms.
- `.omx/research/CANONICAL_RESEARCH_INDEX*` and
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` for the same terms.
- Task/harness ledgers and bridge snapshots for `#996`, coder ownership, and receiver ownership.
- `tools/`, `src/tac/`, and `runtime-rs/` for real encoders, decoders, reference oracles,
  golden vectors, and native parity surfaces.
- Primary literature for arithmetic model/coder separation, LDPC syndrome coding with decoder
  side information, and syndrome-trellis distortion coding.

### Findings beyond the charter seeds

1. `tools/pr101_markov1_aac_codec.py` at commit
   `82ecc2a0ce89d1de3bcb7e6e9e8cfaf7e87302b9` is a real constriction-based Markov-1
   adaptive range encoder/decoder. It is a reusable Python-oracle starting point, not a PR130
   byte result.
2. `.omx/research/ddm_pp1_direct_partition_pricing_20260728.md` reports context-adaptive
   arithmetic coding, but its full-n600 value is a closed-form model price while real range
   coding covered a subset. The charter forbids substituting that projection for a retained
   full-stream result.
3. The equation registry contains Markov-context and arithmetic-coder equations, including
   `markov_context_selector_stream_compression_savings_v1` and
   `markov_1_adaptive_aac_pays_30kb_small_sample_cost_under_high_conditional_bins_v1`.
   No PR130 LDPC/syndrome equation or real-stream closure was found in the searched registry.
4. `src/tac/codec/syndrome_trellis_codec.py` is a pedagogical Viterbi STC implementation for
   distortion-minimizing embedding. It is not an optimized LDPC belief-propagation lossless
   receiver and cannot be presented as the requested family race.
5. `runtime-rs/crates/tac-levelset-inflate/` already has an integer range decoder plus
   Python-oracle SHA parity, while `runtime-rs/crates/tac-boundary-decode/` has the required
   committed golden-vector, audit-manifest, rebuild, negative-control, and benchmark pattern.
6. The #214 receipt in `.omx/research/contest_legal_inflate_20260705.md` and the canonical DAG
   records a bit-exact parallel Python decode that reduced the measured end-gate runtime. The
   DAG says Rust was deferred, so it is precedent for native-capable decode engineering rather
   than evidence that these two rc2 decoders were already lowered.

### What recall changed

- It changed the #996 disposition from its downstream family-wide wording to a restricted-set
  closure and established that rc2 was a legitimate reopening.
- It identified a real adaptive range coder that should be adapted rather than writing a toy
  bit-price estimator.
- It ruled out relabeling the existing STC module as an LDPC/BP implementation.
- It made decoder side information an explicit applicability precondition for LDPC syndrome
  coding. Syndrome source coding of correlated sources is not honest without a receiver-owned
  correlated object; if PR130 exposes none, that section must be marked not applicable rather
  than assigned projected savings.
- It selected the existing Rust golden-vector/SHA contract. No new parity scheme is warranted.

## Reference-form execution contract if resumed

1. Re-run the APDataStore write probe. Before any encoder launch, create a per-family/per-section
   retention manifest under the required destination with deterministic seed, source SHA,
   configuration, expected output path, and atomic temporary-output path.
2. Adaptive arithmetic: adapt the real Markov/context range codec to each applicable PR130 byte
   object; make the context/state reset and transmitted model cost explicit. Persist every coded
   stream, decoder output, and model/table side payload, then verify exact source reconstruction.
3. BP/LDPC syndrome: first name the real residual/flip object and the decoder-owned side
   information. Encode actual syndrome bits with an explicitly stored or generic parity-check
   construction; persist every syndrome and any counted side payload. Iteratively decode and
   verify exact reconstruction. If no legitimate side information exists for a section, record
   `NOT-APPLICABLE` rather than manufacture a projected comparison.
4. Compare retained stream bytes against the matching incumbent and matching receipt bound.
   Only a byte winner proceeds to Rust.
5. Lower only the winning decoder. Report Python and Rust seconds as facts; never use time as
   an admissibility or family verdict.

## Conditional Rust lowering plan

- **Adaptive arithmetic winner:** extend the integer range-decoder surface in
  `runtime-rs/crates/tac-levelset-inflate/src/range_decode.rs` or a narrowly owned module in
  that crate. Generate committed golden vectors from the canonical Python oracle on the exact
  retained PR130 streams. Require decoded-output SHA-256 parity, a bit-flip negative control,
  deterministic rebuild instructions, payload manifest, embedded-constant audit, and a release
  benchmark on the real stream.
- **BP/LDPC winner:** place the native decoder next to the applicable residual grammar, with
  `runtime-rs/crates/residual-codec/` the first reuse candidate. Match the same Python-oracle
  and `assert_sha256_parity` pattern used by `tac-boundary-decode`; do not invent score-level or
  approximate parity.
- **Gate:** one-bit output divergence fails the lowering. A slow but bit-identical decoder remains
  a live engineering task; it does not reverse a byte win.

## Rule-118 boundary

The generic arithmetic/BP algorithm, deterministic finite-precision operations, and a generic
parity-check construction generated from a fixed non-video seed belong in free inflate/runtime
code. Every PR130-derived context table, learned transition model, tuned parity matrix, syndrome,
residual, reset schedule chosen from the video, or other video-derived side payload must be stored
inside and counted with `archive.zip`. The future byte ledger must include all such model and
framing costs.

## Boundaries

- **MEASURED by rc2:** the filesystem preflight only.
- **RECALLED, not re-measured:** PR130 section bytes, #996 incumbent/bound values, and prior ANS
  savings.
- **NOT MEASURED:** both chartered family streams, all rc2 decode seconds, Rust parity, composed
  archive bytes, evaluator components, and exact score.
- **Verdict scope:** storage-instance blocker plus correction of #996's tested-candidate-set
  claim. Neither coder family received a byte verdict.
- **Frontier:** PR130 remains `S = 0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`.
  rc2 did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rc2 successor after operator restores the required
  mount permission`; consumer store: `.omx/research/ddm_rc2_20260810/RC2_BLOCKER_AND_SCOPE_RECEIPT.json`;
  fire trigger: `/Volumes/APDataStore/pact/ddm_rc2_20260810/` accepts an atomic retained-payload
  write and read-back probe. Then execute the adaptive reference stream first, followed by the
  LDPC applicability check and any applicable retained stream; lower only byte winners.

## LIVE-HYPOTHESES

- A higher-order adaptive context model may find structure that the order-0/order-1 receipt
  comparisons and same-model Range-to-ANS swap did not test. It is plausible because the
  semantic and HPAC objects contain repeated structured fields, but its transmitted model and
  reset costs may erase the gain.
- The pose section's 65 B gap above its memoryless estimate may contain a small real win. It is
  plausible only if a context model plus framing costs less than that narrow ceiling.
- LDPC syndrome coding may pay on a sparse flip/residual object if the decoder already owns a
  strongly correlated predictor. It is plausible through correlation-as-channel coding, but is
  not applicable to an unchanged raw section lacking legitimate decoder side information.

## DEAD-ENDS

- Treating #996's “coder axis closed” sentence as proof that these two families lost on bytes:
  its own receipts do not contain those reference-form streams.
- Treating the shipped HPAC range coder or the same-model ANS swap as the missing adaptive-model
  race: that changes the entropy engine, not the chartered probability model.
- Treating a closed-form bit price or entropy bound as a coded payload: the charter requires a
  real retained stream and exact decode.
- Relabeling `syndrome_trellis_codec.py` as an LDPC/BP lossless decoder: it solves a different
  distortion-minimizing embedding problem and says it is pedagogical.
- Using pure-Python or projected wall-clock as a family rejection: decode time is report-only.
- Rerouting rc2 bulk to Vertigo, local disk, `/tmp`, or a symlink after APDataStore denied writes:
  both governing contracts forbid that workaround.
