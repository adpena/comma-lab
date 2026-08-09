# RR3 — PR130 HPAC and shipping audit

**Verdict:** the HPAC mechanism and canonical bytes survive this audit, but the reproduced PR130
submission is **not dependency-closed in the current contest runtime tree**. `inflate.py` needs
`constriction`; neither the shipped submission nor current upstream dependencies declare or install
it. A clean empty-venv decode-entrypoint smoke fails at that import. This is a current-shipping
blocker for another PR130-derived evaluation, not a retroactive rejection of the already completed
PR130 contest-CUDA row.

All new execution here was `[byte-only scorer-free, macOS CPU]`. No CUDA, Metal/MPS, scorer, or exact
evaluation was used. The base remains PR130 CPR1 `S=0.172141297491896447`
`[contest-CUDA, DALI GT, n600]`, archive 191,052 B, SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

Machine-readable receipt: `RR3_HPAC_SHIPPING_RECEIPT.json`.

## 1. Contest-critical dependency verdict

### Answer

**Yes, decode requires `constriction`.** The import is unconditional at `code/inflate.py:13`; the
real token decoder constructs `constriction.stream.queue.RangeDecoder` and categorical models at
`:563-568`. This is not a training-only dependency.

The dependency is not closed on the audited current surface:

- Shipped `code/inflate.sh` has 14 lines and directly invokes `python inflate.py`; it has no import
  preflight, self-install, version pin, or fail-closed bootstrap.
- The 68-file reproduction repository has no `pyproject.toml`, requirements file, or lockfile.
- PR130 commit `9a77b6ad660d6310ab54436757ab07a9bbc9f3e1` changed 13 submission paths and zero dependency
  paths.
- Current `upstream/pyproject.toml` declares 11 runtime dependencies; `constriction` is not one of
  them. Current `.github/workflows/eval.yml` runs `uv sync --group "$UV_GROUP"`, then the evaluator.

The required clean-environment test was fired after a storage-waterfall preflight selected
`/Volumes/VertigoDataTier/pact/ddm_rr3_hpac_shipping_20260809` with 86,309,560,320 free bytes and no
blocker. The venv contained only `pip==26.1.2`, had `site.ENABLE_USER_SITE=false`, and
`find_spec("constriction") is None`. Running the shipped `inflate.sh` with that venv at the front of
`PATH` returned **rc=1 in 0.03 s**:

```text
ModuleNotFoundError: No module named 'constriction'
```

The test deliberately points at an absent archive; that path is never reached because the
unconditional import fails first. Denominator: 1/1 clean empty-venv decode-entrypoint attempts
failed for missing `constriction`.

After certifying the result in the machine-readable receipt, I deleted the success-only scratch
venv and empty decode-output directory; the storage plan and all command/result facts remain.

### Historical positive evidence does not close the current bootstrap

There is strong evidence that the decoder works and is fast enough once the package is present:

- the official PR comment records a successful challenge evaluation over all 600 samples;
- retained source evidence records two full 600-pair GPU runs using `constriction==0.4.2`;
- an independent RTX 5070 run of the unmodified wrapper completed in 142.098 s with
  `constriction==0.5.0`.

Those facts falsify “the library cannot decode the stream” and establish historical total-runtime
feasibility. They do **not** identify a declaration or self-install path in the submitted files or
the current upstream dependency tree. I could not reconstruct from the retained snapshot how the
historical GitHub evaluation acquired the package.

The current workflow's `timeout-minutes: 30` covers the whole job, not only inflate. Therefore a
future bootstrap must be timed as part of the full current job. No such self-install exists to time
today. The rate denominator is also not the archived constant: current `upstream/evaluate.py:64`
sums every file under `uncompressed_dir.rglob('*')`; 37,545,489 B is the measured historical dataset
instance.

**Verdict scope:** current reproduced PR130 submission tree plus current upstream evaluator
dependency tree. **Falsifier:** a clean current-upstream `uv sync` followed by the shipped decode
entrypoint succeeds without host packages and records the resolved `constriction` version, or an
owned runtime patch adds a pinned fail-closed bootstrap and the same bare-venv plus full-job test
passes inside 30 minutes.

## 2. Element-by-element audit

| Element | Result | Evidence and correction | Verdict scope / falsifier |
|---|---|---|---|
| Float32 exactness guard | **PASS for the canonical configuration; MAIN's broad wording is corrected.** | `hpac_integer.py:186-187` directly checks only the 64-term 1x1 head bound: `1,065,024 < 2^24`. It is not, by itself, a proof for every layer. I separately bounded all 9 canonical affine modules: maximum conservative bound is conv-A at 2,629,537; 9/9 are below 16,777,216. Power-of-two exponent scaling does not increase magnitude because exponents are clamped at `<=0`. | Canonical `channels=64, patch=64, delta=2, weight_bound=activation_bound=127`, current topology. Falsified by a topology/config change whose independently counted accumulator exceeds `2^24`, even if the head guard passes. |
| Sheared causal masks | **PASS.** | `patch_group_mask` uses center `(k-1)//2`, offset `col-center+delta*(row-center)`, A `<0`, B `<=0`. Real `group_masks(64,2)` produced 190/190 groups. Coverage over 196,608 pixels/frame was min=max=1; 196,608/196,608 pixels were covered exactly once; group sizes 48..1,536 summed to 196,608. Kernel mask counts were A7=23, B5=14, B3=5. | Canonical 384x512, patch 64, delta 2. Falsified by overlap/gap, changed scan order, or a dimension not divisible by the patch. |
| Probability table | **PASS.** | `codec_hpac_integer.py:27-35` multiplies logits by 8, rounds/clamps to int16, hashes C-order code bytes, then performs float64 softmax and returns float32 probabilities. `IntegerHPAC` returns head codes divided by 8. A supplemental 20,480-value 64x64 mechanism check had maximum distance 0 from the 1/8 lattice. | Source proof is load-bearing; the execution check is explicitly **TOY-BRACKET**, not n600. Falsified by encode/decode differing in code rounding, byte order, softmax construction, or iteration order. |
| Learned self-compression | **PASS.** | It learns 517 scalar bit depths, one per output channel across 9 integer affine modules. Training uses ReLU and an 8-bit maximum; deployment rounds to an integer. Each rounded depth sets a signed per-row radius intersected with `weight_bound`; 4-bit descriptors occupy 259 B. Canonical depth histogram sums to 517: `{0:6,1:2,2:7,3:11,4:38,5:59,6:110,7:275,8:9}`. | Canonical checkpoint/blob. Falsified by a descriptor/row mismatch, non-integer deployed depth, or source/consumer state mismatch. |
| Canonical `hpac.bin.xz` | **PASS, byte exact.** | Blob is 15,164 B, SHA-256 `ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee`. XZ expands to `IHS1`, 20,179 B, SHA-256 `b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58`. The canonical checkpoint reserialized to the exact raw bytes; pinned LZMA filters recompressed to the exact XZ bytes. Packer and shipped-runtime loaders agreed on 28/28 state tensors. Supplemental inference at indices 0 and 599 had max logit difference 0 for both loaders. | Byte equality covers the whole serialized model. The two-index inference is supplemental, not an n600 claim. Falsified by any byte/state inequality or nonzero logit delta. |
| HB2 `-128` fix | **PASS; mechanism-level, not symptom-only.** | Pre-fix, an 8-bit tq1c `conv_a` row could emit `-128` while the consumer clamps at `weight_bound=127`, producing one weight-code error and 0.25 max logit error. The fix intersects the signed bit-depth range with the declared bound in the training/deploy quantizer and packer, and both pack/deploy loaders reject decoded out-of-bound weights. The repaired real tq1c path later matched encode/decode hashes at n600. The canonical PR130 blob also reserialized unchanged under the fix, so the repair does not rewrite a model that never used `-128`. | `IHS1`, `weight_bound=127`, maximum depth 8. Falsified by another source/consumer mismatch on any bit depth or a decoded out-of-bound value accepted by either loader. |
| Packer + extractor | **PASS with one hardening debt.** | `pack_hpac_self_compress.py` bit-packs each active row little-bit-first, stores 4-bit depth metadata, int16 biases, and int8 exponents/fixed fields. `extract_integer_hpac_archive.py` identifies model-section lengths, decompresses the model bundle, and delegates `IHS1` to the same patched runtime loader. However, both `IHS1` loaders accept descriptor nibbles 9..15 if the decoded row happens to remain within `weight_bound`; canonical writers only emit 0..8. | Canonical writers and blob pass. Scoped hardening negative: malformed/noncanonical `IHS1` metadata is not fully fail-closed. Falsified by a test showing both loaders explicitly reject every depth 9..15. |
| Arithmetic codec hash | **PARTIAL.** | `logit_hash_encode=33fd711b...` is SHA-256 over the ordered int16 quantized-logit code bytes that define every probability table. The codec can produce `logit_hash_decode` and can compare decoded tokens with `--cache --require-exact`, but those are separate modes. In the bounded search of 68 intake files plus the 8 pre-RR3 target files, no completed canonical decode-hash receipt matching `33fd711b...` was found. Retained evidence does prove the full CPU decode's raw-token SHA-256 `c5c7671d...`; the shipped inflater itself computes no digest. | Absence is only in the two enumerated roots. Falsified by a locatable canonical n600 decode report with decode hash `33fd711b...`, exact raw-token hash `c5c7671d...`, and exact token equality. |
| `residuals()` | **Formula confirmed; inactive in canonical CPR1.** | It preserves frame 0 and sets later frames to `(tokens[t]-tokens[t-1]) % 5`. This is a class-ring operation for dense semantic labels. Canonical training, encoding, extractor config, and shipped `HPAC_TARGET_MODE` all say `raw`, so CPR1 does not use this residual transform. | Dense 5-class labels only. It is not transferable as-is to #978 latent tokens. Falsified for CPR1 only by a canonical argv/config showing `target_mode=residual`. |
| ZIP assembly | **PASS for canonical/current rebuild.** | One member `p`, explicit one-element order, `ZIP_STORED`, timestamp 1980-01-01, Unix creator, mode 100644, `allowZip64=False`; no filesystem enumeration. Observed archive has one 190,952-B member and 100 B literal ZIP overhead. MAIN's 104-B accounting overhead is 100 B ZIP structure plus the 4-B model-length prefix inside `p`. `compress.sh` pins input/output sizes and hashes and atomically replaces output only after exact verification. | Canonical rebuild. Falsified by a repeated rebuild changing SHA-256. Python `zipfile` header defaults and liblzma implementation are not version-pinned, but the expected output SHA fails closed on drift. |
| Assembly portability | **LOW-SEVERITY DEBT.** | Canonical `repack_carrier.py` and `rebuild_submission_hpac.py` use explicit little-endian `struct`. The earlier-from-checkpoints `build_submission_archive.py` and `extract_integer_hpac_archive.py` use native-endian NumPy uint32 for some lengths. This is stable on the audited little-endian hosts but is not architecture-neutral source. Locale and filesystem order are not involved. | Noncanonical build/extract helpers on little-endian machines. Falsified by converting all wire integers to explicit little-endian and passing canonical byte checks. |

## 3. Ranked findings and falsifiers

1. **HIGH — current shipping dependency gap.** The real decoder needs `constriction`, but no audited
   shipped/current dependency surface supplies it. The bare-venv failure is 1/1. Falsifier: the clean
   current evaluator succeeds without host state, with a locatable resolution/install receipt.
2. **MEDIUM — canonical decode-table agreement is not fail-closed.** The encoder's canonical hash is
   recorded, the shipped decoder emits none, and I did not find a matching completed canonical
   decode-hash receipt in 76 searched files across two roots. Falsifier: n600 decode hash
   `33fd711b...` plus raw-token hash `c5c7671d...`, or an equivalent shipped integrity check.
3. **MEDIUM correction, current configuration closed — the constructor guard is narrower than the
   ledger gloss.** It directly protects the head, not arbitrary future topology. Independent counts
   close all 9/9 current affine sites. Falsifier: a new topology/config without a generated whole-model
   bound.
4. **LOW — malformed `IHS1` depths 9..15 are not rejected by schema.** Weight-bound checks prevent
   the original `-128` mismatch, so this is hardening rather than evidence that HB2 is wrong.
   Falsifier: exhaustive invalid-depth rejection tests in both loaders.
5. **LOW — two helper paths use native-endian uint32 lengths.** Canonical production rebuild is
   explicitly little-endian and exact; this debt affects portable regeneration/extraction helpers.
   Falsifier: explicit little-endian fields plus unchanged canonical hashes.

## RECALL EVIDENCE

Sources and queries consulted before adjudication:

- memory registry: `rr3|pr130|hpac|shipping|constriction` — no relevant registered hit;
- governing corpus: `PROGRAM.md`, byte-identical `AGENTS.md`/`CLAUDE.md`, operating manual, live hot
  state, charter, and common contract;
- intake and local PR130 corpus: `constriction|hpac_self|hpac.bin.xz|encode-tokens|patch_group_mask|codec_hpac_integer|rebuild_submission_hpac`;
- canonical equations and DAG/index surfaces: HPAC, self-compression, PR130, integer, bit-depth,
  round-trip, and shipping terms;
- prior focused evidence: `OFF_THE_SHELF_VS_PORTED.md`, `PR130_REPRODUCED_HERE.md`, HB1/HB2, and
  OP1R receiver evidence.

Found beyond the charter seeds:

- OP1R ran the real source receiver on the first real pair twice with `constriction==0.4.2` and got
  repeat-identical token/raw hashes. This changed the plan: decoder-mechanism compatibility is now
  explicitly separated from dependency-bootstrap closure.
- HB2's real tq1c n600 rerun matched encode/decode hash `e63bd314...` after the `-128` fix. This
  tightened the review from “tests pass” to a source/consumer range proof, and supports accepting the
  fix while still requiring the canonical PR130 decode hash.
- The canonical target is `raw`, so the mod-5 residual function is present but not active. This
  removed it from the canonical shipping-critical path and prevents transferring a label-ring claim
  to latent #978.

Scoped absence: I did not find, in the enumerated current dependency surfaces, a PR130-owned
`constriction` declaration/bootstrap; I did not find, in the 76-file bounded canonical receipt
search, a completed matching canonical decode logit hash.

## 5. Could not check / why

- **Current T4 full-job closure:** not run; this arm has no CUDA device or scorer authority. Historical
  full GPU runs are retained evidence, not a current bootstrap test.
- **Canonical n600 decode logit hash:** not rerun. Retained CPU evidence says the monolithic full token
  decode took 2,197.6 s, and the decoder has no resume/checkpoint boundary. Launching that path here
  would violate the P0 resumability contract.
- **How the historical GitHub job obtained `constriction`:** not recoverable from the retained
  submission tree, its 13 changed paths, the current upstream dependency files, or the captured
  reports. The successful job proves it was available, not how.
- **Cross-version ZIP/LZMA matrix:** not run. Existing current-machine reproduction is byte-identical
  and `compress.sh` fails closed on the canonical SHA; no claim is made for every Python/liblzma
  version.
- **Score/frontier:** no evaluator was run. Pointer delta is zero.

## 6. Git custody boundary

The required serializer was attempted first in the primary checkout and failed at `git add` with
`unable to create temporary file: Operation not permitted` / `failed to insert into database`.
The primary staged index remained empty. To avoid bypassing the serializer or absorbing the dirty
worktree, commit custody moved to the writable isolated shared-object clone
`/Volumes/VertigoDataTier/pact/ddm_rr3_hpac_shipping_commit_20260809`, based exactly on
`3856788c96dd01d38f2d8db9da27146656511935`. Its precommit porcelain contained only these two new
RR3 artifacts and its staged index contained 0 paths. The primary-checkout copies remain untracked
until an operator or Git-writable agent lands the fallback commit into the main repository.

## 7. Follow-on disposition

- **FIRED:** clean empty-venv decode-entrypoint test; canonical HPAC raw/XZ reserialization; dual-loader
  28/28 state comparison; real group-mask coverage; source-complete codec and assembly audit.
- **FOLDED:** HB2 deploy-bound repair is accepted as mechanism-correct for its declared formulation;
  the canonical PR130 blob is unchanged under it.
- **QUEUED-WITH-A-FIRE-ORDER 1:** close `constriction` in an owned runtime copy, then prove the exact
  current evaluator bootstrap before any PR130-derived contest fire.
- **QUEUED-WITH-A-FIRE-ORDER 2:** after dependency closure and a resumable/GPU-capable decoder surface
  exist, produce the canonical n600 decode hash and exact-token receipt.
- **QUEUED-WITH-A-FIRE-ORDER 3:** before accepting new `IHS1` or portable helper changes, add invalid
  depth rejection and explicit little-endian length tests.

No score was measured, no archive was promoted, and the PR130 base remains unchanged.
