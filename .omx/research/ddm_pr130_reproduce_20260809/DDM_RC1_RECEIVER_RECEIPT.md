# DDM-RC1 receiver receipt — explicit split-model grammar and causal ANS decode

Date: 2026-08-09  
Authority: scorer-free receiver and byte-closure work; `score_claim=false`  
Landing: the serializer commit containing this receipt  
Base: PR130 CPR1, 191,052 B, `S=0.172141297491896447`
`[contest-CUDA, DALI GT, n600]`

## Outcome

The owned runtime at `src/tac/pr130_runtime/fx1_runtime_tree/` is now a strict
functional superset of the custodied PR130 receiver for these wire forms:

| outer selector | model section | token section |
|---|---|---|
| model bits `00`, token bit `0` | legacy XZ | Range |
| model bits `01`, token bit `0` | three-stream Brotli | Range |
| model bits `10`, token bit `0` | three-stream raw LZMA2 | Range |
| model bits `00`, `01`, or `10`, token bit `1` | selected legacy/split form | ANS |

Bit 31 of the existing outer `u32` selects Range/ANS. Bits 29–30 select
legacy XZ/split Brotli/split raw LZMA2; selector `11` is reserved and refused.
The low 29 bits remain the model-section length. Legacy PR130 bytes are
unchanged, and the new selectors add zero bytes.

Compatibility boundary: this is a strict superset for the custodied PR130
instance and its new split forms, whose model sections are about 74 KiB. A
hypothetical legacy model section at or above `2^29` bytes would collide with
the selector bits and is intentionally outside this grammar.

Both requested legs executed on real PR130 inputs at the allowed reduced
scope of two frames. The original Range field and the new ANS field each
decoded to the same 393,216 input tokens while exercising all 190 groups per
frame, within-frame partial context, and prior-frame temporal context. The
model parser reconstructed the exact loader input and fed the real HPAC
loader. This is a `TOY-BRACKET n2/600` receiver proof, not n600 authority.

## Measured receiver proof

Pinned runtime control:

- axis: `[macOS-CPU mechanism, TOY-BRACKET n2/600]`, `score_claim=false`
- Python environment:
  `/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv`
- constriction: `0.5.0`
- model source: real split raw-LZMA2 archive
- original token codec: Range
- candidate token codec: ANS
- decoded tokens: 393,216/393,216 exact for both codecs
- decoded token SHA-256:
  `3b0ea476b3655a9c9e97217f9d2171999bb3e4bd27a079ab94123cc107d90ba1`
- reconstructed `models_raw` SHA-256:
  `62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517`
- ANS bytes: 588
- original Range decode: 2.553 s
- conditional-table materialization: 2.791 s
- ANS decode: 2.683 s
- ANS final-state check: passed through `AnsCoder.is_empty()`

The measured n2 ANS decode is 2.683 s against the 1,800 s whole-evaluation
budget. A full n600 ANS decode and render were not measured, so this does not
close the contest wall-clock gate.

The local full suite used repository constriction `0.4.2` and Brotli `1.2.0`:

```text
22 passed
ruff: all checks passed
inflate.sh: sh -n passed
git diff --check: passed
```

The cross-version fixed vector emitted little-endian bytes `6c666601` and
SHA-256
`607631237db3862296ce51b2efe95792a98f4ca4a673966769846394681e7adb`
under constriction 0.4.2 and 0.5.0, decoded exactly, and ended empty. This is
one fixed vector plus the pinned-0.5.0 real n2 proof, not full n600 wire-parity
evidence.

## Real tagged archive custody

The previously measured split archives were untagged evidence inputs. They
were regenerated as stored-ZIP Range candidates by changing only the existing
outer `u32`, preserving the member length, token bytes, archive size, and all
non-content ZIP metadata. The member CRC was deterministically regenerated for
the changed four-byte content:

| form | bytes | tagged archive SHA-256 | member `p` SHA-256 |
|---|---:|---|---|
| split Brotli + Range | 190,149 | `4c9751582937e48e22be8336dbf36cbe229207e65875fe2196694032b40aa891` | `beeb8bdf2bb51ce5bd7f09c055e0557cbecbbb071f424b797f8f118dec951c1d` |
| split raw LZMA2 + Range | 190,818 | `622cc7d8eb512d728b9e579a5d9cca73eccab3c5bf1a1495158c04ce509432c1` | `23a9476588c134b7a6b689c5d1a72a4ae4c7d82154186d29ed6b6933dd930f36` |

Custody paths and deterministic transformation metadata are recorded in
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/splitpack/`
`TAGGED_RANGE_ARCHIVES.json`, SHA-256
`c20abde33e1f18f1ad42ea658acf1c96b6dcd0df15459c5d04989d9211408ca2`.
Both archives parse with explicit dispatch, preserve the original Range tail,
reconstruct the exact `models_raw` SHA, and retain their source archive size.
Neither has been run through `upstream/evaluate.py`.

## n600 coder measurement and boundary

Axis: `[macOS-MPS table materialization + macOS-CPU entropy coding, scorer-free]`;
`score_claim=false`. The co-running n600 coder race completed with `rc=0` in 681 s:

| quantity | bytes |
|---|---:|
| real-table ideal length | 114,851.8 |
| Range | 116,980 |
| ANS | 114,860 |
| measured ANS saving | 2,120 |

Result:
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/ans_n600/`
`ans_vs_range_n600_result.json`, SHA-256
`8816f91afcc21060753a6612cda4e1b7f3b483a7aa073cbfa1b9b5d7e520d451`.

This is a real n600 serialized-length measurement under repository
constriction 0.4.2. The script saved only the lengths: it did not retain ANS
words, hash either newly encoded word stream, decode ANS, or build an archive.
The Range value equals the shipped 116,980-byte length, but byte identity was
not tested. Combining the measured 903-byte model saving and 2,120-byte token
saving gives 188,029 B and
`S=0.170128405876608123` by arithmetic only. Those are `DERIVED`,
`score_claim=false`, and not an archive or evaluator row.

## Brotli dependency decision

Brotli is REQUIRED for the best measured split-model leg. The entrypoint reads
the explicit model selector before importing receiver code. The Brotli branch
pins `Brotli==1.2.0`, installs it and `constriction==0.5.0` wheel-only into a
fresh runtime target, checks the required APIs before and after installation,
and fails closed rather than guessing another codec. Legacy XZ and split raw
LZMA2 install/check constriction only and do not require or install Brotli. NumPy and Torch
remain contest-runtime-provided dependencies that the entrypoint asserts but
never installs.

The dependency-free alternative is separately tagged raw LZMA2 decoded by the
Python standard library with `FORMAT_RAW`, `FILTER_LZMA2`, and preset
`9 | PRESET_EXTREME`. It saves 234 B from the PR130 base, versus 903 B for
Brotli, so avoiding the Brotli wheel costs 669 archive bytes. There is no
decode-time fallback: the wire selector chooses the codec, and wrong bytes
fail closed.

The six-module/four-package manifest and AST-derived closure tests pass
locally. The old FX5 Linux receipt covered the five-module Range runtime only;
a clean Linux bootstrap of this six-module Brotli/ANS closure remains owed.

## Input and borrowed-substrate custody

- intake `codec_hpac_integer.py` SHA-256:
  `70632168250cbecc40b9d6de5da5b167adeb56031368311ff936404a1ceba7e0`
- borrowed intake `inflate.py` SHA-256:
  `335369c9b3b295707f1790feb0b5b7ae288338fae350056cc4bb03aaa18f0c9e`
- real HPAC checkpoint SHA-256:
  `0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7`
- official-Ada GT cache SHA-256:
  `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`

The PR130 semantic, carrier, HPAC, and renderer vehicle remains borrowed. The
original work in this landing is the explicit receiver grammar, split-model
reconstruction, raw-LZMA2/Brotli dispatch, ANS integration and bounded
encode-side chunk helpers, dependency binding, tests, and receipt. No learned
or video-derived content was moved into free receiver code.

## RECALL EVIDENCE

Sources searched before and during implementation:

```text
rg -l -i 'stack\.AnsCoder|encode_reverse|AnsCoder' .omx/research src tests experiments .omx/state
rg -l -i 'three[- ]stream|3[- ]stream|split[- ]Brotli|split brotli|Brotli.*(fallback|self.install|dependency)|fallback.*Brotli' .omx/research src tests experiments .omx/state
rg -n -i 'ddm_rc1|receiver_recall|receiver_ans|receiver_code' .omx/state/canonical_task_status.jsonl .omx/state/lane_registry.json .omx/state/active_lane_dispatch_claims.md .omx/state/subagent_progress.jsonl .omx/research/harness_tasklist_bridge_20260803.jsonl
.venv/bin/python tools/list_canonical_equations.py --json
rg -n -i 'ddm_rc1|anscoder|encode_reverse|split.brotli|Brotli.*fallback' .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* .omx/research/*DAG* .omx/research/*SPEC* .omx/research/*design*
rg -n -i 'ddm_rc1|receiver|split|brotli|constriction|AnsCoder' /Users/adpena/.codex/memories/MEMORY.md
```

Findings beyond the charter seeds:

- The correct owned home already existed at
  `src/tac/pr130_runtime/fx1_runtime_tree/`, with dependency-closure guards.
  This changed the plan from creating a parallel receiver to extending that
  tree and its manifest/tests together.
- The real dependency-free streams are raw LZMA2, not XZ. Brotli and raw
  LZMA2 cannot be safely distinguished by stream magic. This changed the
  design from format inference to explicit three-way model dispatch.
- Existing split archives had no model-codec tag, and token tails had no
  Range/ANS tag. This caused the zero-byte high-bit grammar and regeneration
  of new tagged archive hashes.
- Earlier PR101 split parsers established exact-consumption precedents;
  canonical equation entries `pr95_family_l23_split_brotli_streams_v1`,
  `pr95_family_l37_byte_incremental_brotli_streaming_v1`, and
  `pr95_family_l42_lazy_brotli_auto_install_bootstrap_v1` were relevant. The
  12-byte self-delimiting Brotli idea conflicts with RC1's required three
  lengths and was not substituted.
- Existing VCM research settled on constriction's Range/stack primitives and
  closed a custom entropy-coder detour.
- The bounded task/lane/dispatch query found no exact `ddm_rc1` registration
  in the five named stores. This is a scoped registration absence, not a claim
  of global ownerlessness; a new heavy fire requires registration first.

## Verification surface

Tests cover legacy, tagged Brotli, and tagged raw-LZMA2 model forms; exact
model-loader bytes; Range and ANS forward reconstruction; reverse chunk-call
order and the forward-order negative; fixed-vector cross-version bytes;
int16-code PMF rehydration; mmap chunk encoding; reserved selectors; wrong
codec tags; truncation; trailing model bytes; nonempty ANS state; missing
Brotli; real full tagged ZIP custody; and the real n2 temporal/group-causal
round-trip. Dependency tests derive the import closure from runtime ASTs, bind
versions and required APIs to `inflate.sh`, and hash the current runtime tree.

## What did not move

No n600 ANS payload, split-Brotli+ANS archive, full receiver decode, Linux
bootstrap receipt, CUDA decode, or `upstream/evaluate.py` result exists. The
exact pointer therefore did not move. The live baseline remains PR130 CPR1,
191,052 B, `S=0.172141297491896447`
`[contest-CUDA, DALI GT, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN receiver/runtime arm; consumer store: a refreshed FX5-style Linux six-module closure receipt; fire trigger: this serializer commit is immutable and a clean Linux x86_64 lane is claimed.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN exact-row arm; consumer store: tagged split-Brotli+Range receiver decode and `upstream/evaluate.py` receipts keyed to archive SHA `4c9751582937e48e22be8336dbf36cbe229207e65875fe2196694032b40aa891`; fire trigger: the Linux closure passes and the contest-CUDA lane is claimed.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN ANS encode arm; consumer store: an SSD-resident atomic int16-code chunk manifest, persisted constriction-0.5.0 ANS words, and interrupted/resumed byte-identity receipt; fire trigger: task/lane registration, storage preflight, and a real-table resume positive control all pass.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: contest-CUDA exact-row arm; consumer store: tagged split-Brotli+ANS archive, exact token/model hashes, full decode wall-clock, and evaluator receipt; fire trigger: the retained n600 ANS payload decodes exactly under the committed receiver and ends empty.

## LIVE-HYPOTHESES

- Int16 logit-code spill will reproduce the full real ANS payload because PR130 quantizes to that lattice before softmax; synthetic chunk identity is proven, but real n600 identity is not.
- Full 0.4.2/0.5.0 wire compatibility is likely because the fixed vector and real n2 trajectory agree, but the unretained n600 words leave large-state compatibility untested.
- Full n600 ANS token decode may fit the 30-minute budget because n2 took 2.683 s, a linear decode-only extrapolation is about 805 s, and rendering is separate; only an n600 run can promote that estimate.

## DEAD-ENDS

- Heuristic model-codec or token-codec detection is closed: raw LZMA2/Brotli and Range/ANS do not carry reliable discriminators.
- Calling the old split archive SHAs receiver-closed is closed: they were untagged evidence inputs and have been superseded by the tagged Range SHAs above.
- Calling the n600 length result an ANS archive, byte-identity proof, or score is closed: no compressed words were retained or decoded.
- Labeling XZ fixtures as split LZMA2 is closed: the real dependency-free streams are raw LZMA2.
- Encoding ANS chunks in forward chronological call order is closed: LIFO pops the chunks backward.
- Treating the 2,080-byte n60 projection as current authority is closed: the n600 serialized-length measurement is 2,120 B.
- Building a custom arithmetic coder is closed: the recalled canonical primitive layer already settled on constriction.
- Using one-shot `lzma.decompress` as an exact wire validator is closed: raw LZMA2 accepted inner trailing junk and XZ accepted concatenated streams, so the receiver now requires one decoder EOF with no unused bytes.
- Provisioning Brotli for every wire form is closed: the entrypoint now reads the explicit model selector first and only the split-Brotli leg installs or requires the wheel.
