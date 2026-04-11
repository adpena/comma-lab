# TAC Lossless Design

## Goal

Add a production-grade `tac/lossless` subsystem for the commaVQ lossless compression challenge, with the same rigor as the lossy side:

- exact compliance with the challenge contract
- canonical evaluation and promotion flow
- durable experiment state
- one `tac` CLI that covers both `lossy` and `lossless`

The new work must not trample the existing lossy workflow or leak state between the two domains.

## Non-Negotiable Design Rules

1. **No legacy compatibility**
   - no parallel old/new lossless paths
   - no compatibility shims
   - no “temporary” alternate entrypoints that become permanent
   - if a canonical path is introduced, it replaces the ad hoc one

2. **No signal loss**
   - lossless experiments must have the same durable-state quality as lossy
   - measured outcomes, failures, and promotions live on disk

3. **No lossy trampling**
   - `tac/lossless` must be structurally separate from post-filter training code
   - no silent changes to lossy CLI or lossy experiment semantics

4. **Open-source / production bar**
   - `tac` should be something comma.ai could plausibly use in production
   - explicit interfaces, typed outputs, minimal ambiguity, deterministic workflows

## Why `tac/lossless`

`commavq` already has a clean evaluation contract:

- [compression/evaluate.py](/tmp/commavq-read/compression/evaluate.py)
- [compression/evaluate.sh](/tmp/commavq-read/compression/evaluate.sh)

The correct move is not to scatter lossless utilities across `experiments/`. It is to build a sibling subsystem under `src/tac/` so:

- lossy and lossless share workflow rigor
- state/reporting patterns are consistent
- challenge logic is reusable
- the open-source story stays coherent

## Package Layout

Add:

- `src/tac/lossless/__init__.py`
- `src/tac/lossless/contracts.py`
- `src/tac/lossless/data.py`
- `src/tac/lossless/codecs.py`
- `src/tac/lossless/evaluate.py`
- `src/tac/lossless/submission.py`
- `src/tac/lossless/profiles.py`
- `src/tac/lossless/cli.py`

Responsibilities:

### `contracts.py`

Typed models for:

- token shard metadata
- compression run result
- decompression verification result
- packaged submission result
- failure signature
- promoted lossless result

### `data.py`

Lossless data access:

- load commavq token shards
- iterate exact submission splits
- byte/accounting helpers
- deterministic ordering helpers

### `codecs.py`

Compression engines:

- `lzma` baseline
- optional external tool adapters like `zpaq` when present
- model-based compressors behind a common interface
- no hidden side effects; explicit input/output paths only

### `evaluate.py`

Canonical lossless evaluation:

- exact decompression correctness
- exact compression-rate calculation
- contract parity with `commavq/compression/evaluate.sh`
- single source of truth for measured lossless results

### `submission.py`

Submission packaging:

- deterministic zip assembly
- exact `decompress.py` placement
- challenge-compliant output structure
- preflight verification before packaging succeeds

### `profiles.py`

Named lossless experiment profiles:

- `lzma_baseline`
- `zpaq_baseline`
- `gpt_arithmetic_small`
- `gpt_arithmetic_large`
- `neural_codec_smoke`

Profiles are configuration bundles, not code branches.

### `cli.py`

Canonical `tac` CLI routing for:

- `tac lossy ...`
- `tac lossless ...`

This should standardize and eventually absorb the role of ad hoc scripts like `experiments/train_tac.py`.

## CLI Design

Use a canonical `tac` CLI under `src/tac/cli.py`.

Subcommands:

- `tac lossy train`
- `tac lossy eval`
- `tac lossy promote`
- `tac lossless prepare`
- `tac lossless compress`
- `tac lossless package`
- `tac lossless evaluate`
- `tac lossless promote`

Design choice:

- keep implementation dependency-light and consistent with the current repo
- use `argparse` first
- structure it so migrating to `click` later would be mechanical if we ever really need it

We do **not** add a parallel CLI stack.

## Lossless Experiment Lifecycle

1. **Prepare**
   - resolve dataset paths / split selection
   - verify required assets and tool availability

2. **Compress**
   - run a named profile
   - produce compressed outputs into a deterministic workspace

3. **Package**
   - assemble the exact submission zip
   - include only challenge-compliant files

4. **Evaluate**
   - decompress
   - verify exact token equality
   - compute compression rate

5. **Promote**
   - only after exact evaluation succeeds
   - sync durable state through one canonical path

## Compliance Rules

Lossless promotion must fail unless:

- decompression reconstructs exact original tokens
- submission archive structure matches the challenge contract
- measured compression rate is recorded from the canonical evaluator
- required files are present in the packaged submission

No heuristic proxy gets to promote lossless work.

## State and Reporting

Lossless must have its own promoted pointer and result history.

Do not mix lossless promoted state with lossy promoted state.

Recommended surfaces:

- `.omx/state/lossless_focus.md`
- `.omx/state/lossless_next_experiments.md`
- `.omx/research/lossless_findings.md`
- `reports/lossless_results.jsonl`
- `reports/lossless_timeline.jsonl`
- `reports/lossless_latest.md`

Shared state tooling can be generalized later, but the state itself must stay separate.

## Migration of Existing Lossy Tooling

What should be standardized:

- `experiments/train_tac.py` should become a thin compatibility shell or be deleted after the canonical `tac` CLI fully replaces it
- do not maintain two equally “official” lossy CLIs

Target end state:

- `tac` CLI is the algorithm/experiment CLI
- `comma-lab` remains the repo-ops / state / packaging / publishing CLI

## Strategy for commaVQ

The public `commavq` repo suggests the real frontier was:

- model-based entropy coding (`arithmetic coding with GPT`)
- plus one stronger private winner (`self-compressing neural network`)

The repo did not run submissions through public PR iteration. The winning meta was private.

So `tac/lossless` should optimize for:

- fast private iteration
- exact evaluation
- durable local research state

not public-facing experimentation.

## Success Criteria

- `tac/lossless` exists as a real sibling subsystem
- `tac` CLI exposes both lossy and lossless paths
- lossless evaluation is exact and canonical
- lossless state/reporting is durable and separate from lossy
- no parallel legacy lossless path survives
