# RR1 — PREPARE + VERIFY surface audit

**Date:** 2026-08-09  
**Mode:** fresh-eyes, scorer-free static audit plus the exact local verification command  
**Intake:** `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`  
**SOURCE_REPO_HEAD:** `e34f31bc4969042c0051ac81aa3c56884419a231`  
**Ledger under review:** `OFF_THE_SHELF_VS_PORTED.md` at
`3856788c96dd01d38f2d8db9da27146656511935`; throughput context at
`d28fde10f5c4d9070364009fd614944b485903d6`  
**Authority boundary:** source inspection and deterministic byte/component checks only. No
Metal/CUDA/scorer measurement was performed; no score claim is created.

## Verdict first

`scripts/verify.sh` is a real and useful **frozen-assembly integrity gate**, but it is not a
training, decoder, or score verification gate. In a fresh scorer-free run it passed all 24/24
collected pytest cases and rebuilt the exact 191,052-byte CPR1 archive with SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`. That closes the deterministic
assembly tail from banked artifacts.

The adversarial hypothesis is therefore **confirmed in substance but refuted literally**:

- It is false that the one comparison in `reproduce.sh` is the only load-bearing assertion.
  `audit_repo.py` locks 11 retained artifacts; one carrier golden test independently rebuilds
  and hashes CPR1; four provenance tests connect selected banked components to deployed bytes.
- It is true that none of the 24 pytest cases runs training, decodes the 600 submitted frame
  pairs, invokes SegNet/PoseNet, or runs `upstream/evaluate.py`. Nineteen of 24 collected cases
  never read any retained payload artifact at all.
- The retained-boundary replay has a real target-lineage split: semantic QAT consumes the
  AV-like `gt_cache_600.pt`, while carrier, HPAC, and token encoding consume the DALI
  `gt_cache_600_official_ada.pt`. The strict 49-stage graph does not preserve that history; it
  builds one DALI cache and routes it to 41 downstream stages. Cross-leg metric comparisons in
  the retained replay therefore carry a decoder/target confound.

## Verification receipt

Command, run from the read-only intake with scratch and bytecode redirected to the SSD:

```bash
PYTHON_BIN=/Users/adpena/Projects/pact/.venv/bin/python \
TMPDIR=/Volumes/VertigoDataTier/pact \
bash scripts/verify.sh
```

Observed result:

```text
audit_repo: status=passed, files=69, repository_bytes=3134756, artifact_count=11
CPR1 byte-identical: 191052 0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
24 passed in 28.22s
CPR1 repository verification passed
```

The intake was clean before and after the run. The verification script itself does not check
the Git head or cleanliness; that pin was checked outside it with `git rev-parse HEAD` and
`git status --short`.

## Per-element audit table

| Element | What it actually asserts | What it does **not** assert | Denominator | Verdict + scope |
|---|---|---|---:|---|
| `audit_repo.py`: repository cap | Sum of included regular files is at most 5 MiB. | Nothing under any path component named `.git`, `.pytest_cache`, `__pycache__`, `build`, or `work`; no free-space or runtime-artifact rule. | 1 repository total / 1 evaluated / 0 fired | CLEAN for included-tree size; `INSTANCE` |
| `audit_repo.py`: file cap | Every included file is at most 1 MiB. | Ignored paths; semantic relevance; whether a large file is legitimately required. | 69 files / 69 evaluated / 0 fired | CLEAN for included files; `INSTANCE` |
| `audit_repo.py`: artifact lock | Each of 11 declared rows exists and matches its declared byte count and SHA-256. | Lock completeness; producer command; decoder/device/upstream identity; causal linkage among cache, checkpoint, token stream, and archive. An unlisted artifact can pass. | 11 existence + 11 size/hash checks / 22 evaluated / 0 fired | CLEAN custody for declared bytes; provenance gap is `FORMULATION` |
| `audit_repo.py`: duplicate content | No two non-empty included files have identical `(size, SHA-256)`. | Empty-file duplicates; near-duplicates; semantic duplicates; ignored paths. | 69 non-empty content groups / 69 examined / 0 duplicate groups | CLEAN exact-content census; `INSTANCE` |
| `audit_repo.py`: secrets | Five regexes do not match text-like included files: GitHub classic, GitHub fine-grained, AWS access key, OpenAI-style key, private-key header. | Eleven NUL-containing files are skipped; all other credential classes, private paths, and encoded/binary secrets are outside this detector. | 58 text files × 5 patterns = 290 evaluations / 0 fired; 11 binary files skipped | CLEAN only for five patterns; `FORMULATION` |
| `audit_repo.py`: unreachable code | Every flat `code/*.py` stem is found by a textual `name.py` reference or a static import reachable from scripts/tests. | Runtime callability. Imports in dead branches count; a filename in prose counts; dynamic imports and packages below `code/*/` are not modeled. | 32 flat code modules from 9 entrypoints / 1 aggregate check / 0 unreachable | CLEAN under the implemented approximation; `FORMULATION` |
| `compileall` | All Python sources under `code`, `scripts`, and `tests` compile to bytecode without a syntax error. The new scratch `PYTHONPYCACHEPREFIX` means this is not a reuse-only no-op. | Import success, dependency compatibility, undefined names, CLI/call-site correctness, numerical behavior, shell syntax, or any training/eval result. | 37 Python files / 37 compiled in the fresh scratch prefix / 0 syntax failures | Real syntax gate, not a semantic gate; `FORMULATION` |
| `reproduce.sh` | Banked base archive + banked HPAC blob + banked token stream rebuild CPR1 byte-for-byte equal to `artifacts/final/archive.zip`. | Training, cache production, token re-encoding, frame decode, runtime inflate, or scoring. | 1 final byte equality / 1 evaluated / 0 fired | CLEAN deterministic assembly tail; `INSTANCE` |
| `code/test_carrier_codec.py`: synthetic codec cases | Deterministic exact round-trip; 200 randomized exact round-trips; six truncation cases; bad magic/Rice parameter; over/under-subscribed Huffman tables; padding/surplus bits; Rice overflow/surplus bits. | Any banked training lineage, frame decode, scorer, or final score. | 13/14 collected cases are synthetic | CLEAN codec grammar within generated domains; `FORMULATION` |
| `code/test_carrier_codec.py`: frozen golden | Repacking the reproduced predecessor yields the expected final size/hash and carrier hash; generic repack matches canonical; unpacked semantic state, basis, and coefficients are tensor-equal and hash-pinned. | The selected carrier checkpoint produced those tensors; inflated video is correct; evaluator metrics are preserved. | 1/14 collected cases reads a banked archive | CLEAN lossless repack; checkpoint linkage gap is `INSTANCE` |
| `tests/test_e2e_plan.py` | Exactly 49 stage objects; named first/last stages; declared run-dir inputs appear after declared producers; declared inputs avoid frozen artifacts; five CPU pose boundaries and three selected flags/output names are present. | No stage executes. Commands may be semantically inert, undeclared dependencies can exist, outputs need not be produced, and the target cache can be the wrong historical target. | 3/3 structural cases; 0/49 stages executed | CLEAN plan topology only; `FORMULATION` |
| `tests/test_official_report.py` | A synthetic 600-sample report parses and recomputes a score; 599 samples and an archive-size mismatch raise. | It never reads either retained official report, the final archive, runtime metadata, decoder axis, or an evaluator output. Negative coverage omits the parser's inconsistent-rate and inconsistent-displayed-score branches. | 2/2 synthetic cases; 0 real reports | CLEAN parser unit test; official-evidence verification ABSENT, `INSTANCE` |
| `tests/test_provenance.py`: semantic | Selected semantic checkpoint packs to the semantic bytes in the banked base archive and a fixed hash. | How it was trained or which target cache produced it. | 1/5 cases | CLEAN checkpoint-to-blob identity; `INSTANCE` |
| `tests/test_provenance.py`: base HPAC | Base HPAC bytes deserialize and serialize exactly. | The selected self-compression training path. | 1/5 cases | CLEAN codec identity; `INSTANCE` |
| `tests/test_provenance.py`: selected HPAC | Selected HPAC checkpoint serializes and XZ-compresses to the exact banked deployment blob; raw length is pinned. | The checkpoint encodes the selected cache correctly or generated the banked token stream. | 1/5 cases | CLEAN checkpoint-to-model identity; `INSTANCE` |
| `tests/test_provenance.py`: deploy bound | A synthetic `-128` weight is clipped to the deployable `-127` bound and round-trips. | Any retained payload. | 1/5 cases, synthetic | CLEAN regression test; `FORMULATION` |
| `tests/test_provenance.py`: token stream | Selected banked token stream differs from the old base stream and has the expected length/hash. | It does not run the encoder from the HPAC checkpoint and target cache, so causal token reproduction is not checked. | 1/5 cases | CLEAN token custody; producer linkage ABSENT, `INSTANCE` |

### `audit_repo.py` denominator summary

There are **11 named checks across six conceptual families**: repository cap, file cap,
artifact missing, artifact drift, exact duplicate, five secret regexes, and unreachable code.
All 11/11 have a reachable failure path by static inspection. On the current 69-file tree they
instantiate **452 current predicate evaluations**:

```text
1 repository cap
+ 69 file caps
+ 11 artifact-existence checks
+ 11 present-artifact drift checks
+ 69 non-empty content groups
+ 58 text files × 5 secret patterns
+ 1 aggregate unreachable-code check
= 452 evaluated; 0 fired
```

This was a clean run, not a mutation test of all 11 failure modes.

## The 24 pytest cases: payload-dependence census

The four files define 19 test functions and collect 24 cases because the truncation test is
parameterized six ways. Direct source contains 45 `assert` statements and 11 static
`pytest.raises` blocks; loops execute some clauses many times.

| File | Collected cases | Payload-dependent | Payload-independent |
|---|---:|---:|---:|
| `code/test_carrier_codec.py` | 14 | 1 | 13 |
| `tests/test_e2e_plan.py` | 3 | 0 | 3 |
| `tests/test_official_report.py` | 2 | 0 | 2 |
| `tests/test_provenance.py` | 5 | 4 | 1 |
| **Total** | **24** | **5** | **19** |

Under the precise mutation “replace every retained payload artifact with zeros, then bypass
`audit_repo.py` and `reproduce.sh` so pytest can start,” 19/24 cases still exercise only source,
synthetic values, or a synthetic report and would remain eligible to pass. Five cases read
banked components and would fail or error. Under the narrower mutation “zero only
`artifacts/final/archive.zip`,” 0/24 pytest cases directly opens that file; the full
`verify.sh` still fails earlier because the artifact lock and `reproduce.sh` protect it.

A still sharper gap: the two preserved files under `evidence/source_archive_official_*.txt`
can be emptied without any pytest case noticing. They are not artifact-lock rows, and the
official-report tests synthesize their own strings. **Verdict:** gap, `INSTANCE` scope.

## GT-cache lineage

### What `prepare` actually does

`scripts/train.sh prepare` produces neither cache. It XZ-decompresses two retained files into
`work/caches/` and verifies fixed uncompressed sizes and SHA-256 values
(`scripts/train.sh:79-92`):

| Retained cache | Restore-only identity | Producer/decoder finding |
|---|---|---|
| `gt_cache_600.pt.xz` | 117,981,133 uncompressed bytes; SHA-256 `8248a60d…828` | Exact historical producer command is **ABSENT** from the intake. The cache stores only `pose` and `seg`; no decoder/upstream/weights/device metadata. Full-population content comparison in `OP1R_MEASUREMENTS.json:266-283` found it differs from the local macOS AV label tensor at only 1/117,964,800 sites and from retained official DALI at 20,750 sites. It is therefore **AV-like by measured content**, not proven PyAV by provenance. |
| `gt_cache_600_official_ada.pt.xz` | 117,981,301 uncompressed bytes; SHA-256 `382d7dfe…195` | `build_gt_cache_official.py` is the retained producer implementation; DALI is its default and the strict graph explicitly passes `--dataset dali` (`e2e.py:294-310`). Its raw segmentation tensor matches the deployed token golden (`OP1R_PATH.md:128-139`). The exact historical shell command/log is absent, so “produced by this exact invocation” remains unproven; DALI target identity is supported by code, naming, retained evidence, and content. |

`build_gt_cache_official.py` itself supports two decoder modes while keeping the frozen scorer
on CUDA:

- `--dataset dali`: `DaliVideoDataset` uses DALI/NVDEC and yields device RGB tensors
  (`build_gt_cache_official.py:36-59`; `upstream/frame_utils.py:110-150`).
- `--dataset av`: `AVVideoDataset` uses PyAV to decode YUV420 frames, then the official
  `frame_utils.yuv420_to_rgb` conversion before scorer inference
  (`build_gt_cache_official.py:36-59`; `upstream/frame_utils.py:159-216`).

The output is only `{"pose": ..., "seg": ...}` (`build_gt_cache_official.py:65-67`). The JSON
report records pair count, elapsed time, cache hash/size, and pose ranges, but not dataset mode,
challenge commit, scorer-weight hashes, video hashes, CUDA device, or package versions. A cache
separated from its runner marker is not self-authenticating as DALI or AV.

### Which leg consumes which cache

The retained-boundary replay is mixed:

| Replay action | Cache consumed | Source |
|---|---|---|
| `semantic` | `gt_cache_600.pt` (AV-like content) | `scripts/train.sh:101-118` |
| `carrier` | `gt_cache_600_official_ada.pt` (DALI target) | `scripts/train.sh:121-156`; carrier evidence also names it |
| `hpac` | `gt_cache_600_official_ada.pt` | `scripts/train.sh:169-202` |
| `encode-tokens` | `gt_cache_600_official_ada.pt` | `scripts/train.sh:237-268` |

The strict raw-video graph is internally single-cache: stage `01_targets` builds
`official_targets.pt` with explicit DALI, and **41/49 stages** declare that same path as an
input (all semantic stages except the expansion transform, all pose stages, HPAC training, and
token encode/verify). This removes the internal split but changes the historical selected
semantic tail's target. Consequently, the README statement that a fresh strict run
“reproduces the selected method and all data dependencies from raw video” is too strong unless
“selected method” explicitly permits changing the semantic target lineage.

The #906 result found beyond the charter seeds makes this material, not theoretical. The
same-host T4 receipt at
`/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/result_summary.json` has complete
600/600 coverage and a live positive control: DALI and AV produced different segmentation and
pose targets. That receipt is a decoder-isolation measurement, not a candidate score.

## Ranked findings with falsifiers

### F1 — The historical selected replay mixes target axes; strict E2E silently changes the semantic target

**Severity:** HIGH  
**Verdict:** finding, `INSTANCE` scope (this retained PR130 recipe and graph; not the family)

The selected semantic tail reads AV-like target content. Carrier, HPAC, and tokens read the
official DALI target. The 49-stage graph instead applies one fresh DALI cache across 41 stages.
Therefore:

1. retained semantic-versus-carrier/HPAC metric comparisons have an unstated target-decoder
   confound;
2. the strict graph is a coherent DALI re-training formulation, but not an exact reconstruction
   of the frozen selected target lineage;
3. a fresh DALI semantic result cannot be described as reproducing the historical selected
   semantic training treatment without an explicit formulation-change label.

**Falsifier:** a locatable historical command/marker proving the frozen semantic tail consumed
the retained official DALI tensor, or a byte/content proof that the two target tensors are
identical. Current source and full-population content receipts contradict both.

### F2 — “24 passed” does not verify training, decode, or score

**Severity:** HIGH  
**Verdict:** finding, `FORMULATION` scope (the present `verify.sh` contract)

The suite's 24 is a mixed denominator: 19 synthetic/structural cases and five banked-component
cases. No test runs `train.sh`, `e2e.py run`, `inflate.sh`, SegNet/PoseNet, or the official
evaluator. The e2e tests validate a plan object, not 49 executed stages. The official-report
tests validate a synthetic parser input, not preserved evidence.

The correct claim is: **“24 structural/codec/provenance tests passed, and frozen CPR1 assembly
was byte-identical.”** “Repository verification” is acceptable only with that boundary; it is
not evidence that the raw-video chain reproduced the score.

**Falsifier:** a direct assertion in the four selected files that executes training or official
decode/eval on real inputs and validates its output. Full reading found none.

### F3 — Three causal provenance links are absent

**Severity:** MEDIUM-HIGH  
**Verdict:** finding, `INSTANCE` scope

- The selected carrier checkpoint is artifact-hash-locked but never loaded by the verify tests
  and is not connected to the deployed carrier tensors.
- The selected HPAC checkpoint is connected to the packed HPAC model, and the token stream is
  hash-locked, but verify never re-encodes tokens from checkpoint + target cache.
- The two target caches are hash-locked but never loaded by pytest, and their decoder identity is
  not embedded in the cache object.

The deterministic archive tail remains valid. These gaps limit claims about which training
artifacts produced its sections.

**Falsifier:** checkpoint-to-deployed-carrier tensor equality; a full 600-map token re-encode
matching `94837987…15eb`; and a content-addressed cache manifest recording decoder, upstream,
weights, videos, command, device, and packages.

### F4 — The audit gate does not pin source state or evidence integrity

**Severity:** MEDIUM  
**Verdict:** finding, `FORMULATION` scope

`audit_repo.py` locks 11 artifacts but does not assert `SOURCE_REPO_HEAD`, clean Git status,
source-file hashes, lock completeness, or hashes for retained official reports. A dirty source
tree can pass if behavior still satisfies the tests. The audit's “secrets” and “unreachable”
labels also have the narrower denominators shown above.

**Falsifier:** a source-tree/commit pin and evidence-manifest check in the executed verify path.
They are absent from `audit_repo.py` and `verify.sh`.

### F5 — `compileall` is real but only syntactic

**Severity:** honest non-finding plus boundary  
**Verdict:** CLEAN, `FORMULATION` scope

The claim that `compileall` is a no-op is refuted. `verify.sh` creates a fresh scratch directory
and redirects `PYTHONPYCACHEPREFIX`; 37 Python files therefore compile into an empty destination.
It is a legitimate syntax gate. It makes no behavioral or import claim and checks no shell file.

**Falsifier of the clean finding:** evidence that the prefix already contained valid bytecode or
that compileall did not enumerate the source directories. The fresh `mktemp` path and explicit
three-directory invocation contradict that.

## Honest non-findings

- `reproduce.sh`'s final equality is not a weak size/hash proxy: it compares the complete bytes
  of actual and expected archives before printing the hash.
- The frozen carrier repack test is substantive. It checks archive hash, carrier hash, generic
  versus canonical repack equality, tensor equality, and fixed tensor/state hashes.
- Semantic checkpoint-to-blob and HPAC checkpoint-to-packed-model identities are directly tested.
- `audit_repo.py` examined every one of its 69 included files on this intake, and all 11 declared
  artifacts matched. This is a bounded clean result, not a global completeness claim.
- The intake stayed read-only and clean; no file under it was edited or staged.

## Assumption challenge

**Shared assumption under review:** a repository-level green string plus a count of passing tests
is interchangeable with reproduction of the method that produced the reported score.

Violating that assumption exposes three distinct objects: (1) frozen archive assembly,
(2) historical training lineage, and (3) official scorer reproduction. The present gate strongly
checks (1), partially checks banked component custody around (1), statically describes (2), and
does not execute (2) or (3). This separation changes the safe handoff language and the required
next controls; it does not challenge the existence of the banked CPR1 archive or its published
contest-CUDA row.

## RECALL EVIDENCE

Searched before adjudication:

- `.omx/research/**` for `gt_cache_600`, `build_gt_cache_official`, `DALI`, `AV`, `compileall`,
  `audit_repo.py`, `24 passed`, and `CPR1 repository verification`;
- `.omx/state/{main_hot_state.md,probe_outcomes.jsonl,operator_p0_ledger.jsonl}` for `#906` and
  `#995`;
- `.omx/research/CANONICAL_RESEARCH_INDEX_20260629.md` and all `sub015_DAG_*` files for PR130,
  decoder, and cache terms;
- the canonical equations registry (429 entries) for `pr130`, `dali`, `AVVideoDataset`,
  `gt_cache`, `chroma siting`, and decoder-GT terms.

Beyond the charter seeds, the search found the completed same-host #906 AV/DALI receipt and the
full-population OP1R cache-content receipt. They changed the plan by promoting the cache split
from an unverified source-level concern to F1: a known target-content difference with bounded
decoder-isolation evidence. I did not find a canonical equation that directly governs this
verify/cache-lineage surface in the 429-entry registry searched; broad string hits were
unrelated or downstream representation equations. This is a scoped registry-search negative,
not a claim that no useful formalization can exist.

## What I could not check and why

- **Exact historical producer command for `gt_cache_600.pt`: ABSENT.** The file is AV-like by
  full-population content, but “PyAV produced it” would exceed the retained provenance.
- **Exact historical command/log for the retained official-Ada cache: ABSENT.** Its DALI identity
  is strongly supported by the retained builder, explicit strict-graph command, deployed token
  identity, and evidence naming, but the cache object itself carries no metadata.
- **Training behavior:** not executed. This arm has no Metal device, the charter makes static
  full coverage the reference form, and no GPU measurement is necessary to audit layer 1.
- **Official frame decode and score:** not executed. No scorer slot was owned; MPS is not score
  authority; `verify.sh` is scorer-free.
- **All audit failure branches by mutation:** not executed. The 11/11 fireability conclusion is
  static; the clean run exercised 452 current predicate instances with zero failures.

## Follow-on dispositions

| Disposition | Owner | Consumer store | Fire trigger | Action |
|---|---|---|---|---|
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / task #995 | `scripts/e2e.py`, `recipe/TRAINING.md`, and the PR130 reproduction ledger | Before the next claim that the strict graph reproduces the selected historical method | Declare the DALI-for-all graph as a formulation change or add an explicit historical mixed-cache mode; bind each cache to a decoder/provenance manifest. |
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / verification owner | `scripts/verify.sh` + the four selected test files | Next verification-surface edit, before using “verified” as training/eval evidence | Add a durable verification receipt, source-head/cleanliness pin, official-evidence hashes, selected-carrier linkage, and an optional full token re-encode gate with an explicit cost tier. |
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / #906 harvester | `.omx/state/main_hot_state.md` and task consumers #984/#982 | On RR1 harvest | Replace the stale “#906 OPEN” text with the completed same-host receipt and keep its non-score/axis boundary attached. |
| `FOLDED` | RR1 | `compileall` | Reopen only if Python coverage paths change | No compileall replacement: retain it as the 37-file syntax gate and do not narrate it as behavioral verification. |

## Frontier honesty

This audit moved no score and produced no candidate archive. The PR130 base remains
**S = 0.172141297491896447 at 191,052 bytes `[contest-CUDA, DALI GT, n600]`**; RR1 only narrows
what the local frozen verification proves.
