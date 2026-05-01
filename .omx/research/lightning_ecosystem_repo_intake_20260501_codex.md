# Lightning Ecosystem Repo Intake - 2026-05-01

Status: in progress. This is a research intake ledger, not contest evidence.
External repositories may motivate implementation patterns, but they do not
promote lanes, validate scores, or relax exact CUDA auth-eval custody.

## 2026-05-01T05:11:56Z Initial Primary-Source Pass

Repositories requested by operator:

- LitModels: https://github.com/Lightning-AI/LitModels
- lightning-thunder: https://github.com/Lightning-AI/lightning-thunder
- utilities: https://github.com/Lightning-AI/utilities

Observed upstream heads by `git ls-remote`:

- LitModels `main`: `2b1f787a3bc9ea99e5361bbe278df630dd7ebb5f`
- lightning-thunder `main`: `99850362f965db57077c3552f81c7f4abd8add7f`
- utilities `main`: `5646a0d6cb263c3750b5d3dc90ed9af3686e166f`

Primary-source observations:

- LitModels describes model checkpoint save/load/hosting and public/private
  registry workflows. Its base `requirements.txt` lists `lightning-sdk` and
  `lightning-utilities`, but `pyproject.toml` extras/examples/tests reference
  `lightning >= 2.0.0` and `pytorch-lightning >=2.0`. Adoption verdict:
  research only. Do not install broad extras or route contest custody through
  cloud model registries. Useful pattern candidates: async checkpoint
  ergonomics, strict named model artifacts, and registry-style manifest
  vocabulary, only if reimplemented locally with archive custody.
- lightning-thunder describes a PyTorch source-to-source compiler for training
  and inference acceleration. Base requirements include `torch >=2.7.1`,
  `lightning-utilities >0.7.0`, `networkx`, `optree`, `opt_einsum`, and
  `dill`; no direct `lightning` package in base requirements observed. Adoption
  verdict: opt-in profiling/training experiment only. It is not allowed in
  canonical exact eval or promotion custody until deterministic CUDA parity,
  compile-cache provenance, numerical drift, and fallback behavior are audited.
  Useful candidate areas: renderer/codec training speed, repeated Fisher or
  sensitivity-map passes, graph/fusion diagnostics.
- utilities is the shared Lightning utility repo. README advertises reusable
  workflows/actions, `lightning_utilities.core`, and
  `python -m lightning_utilities.cli --help`. The source tree contains
  `core/imports.py`, `core/rank_zero.py`, `core/apply_func.py`,
  `cli/dependencies.py`, and a central requirement parser. Adoption verdict:
  pattern-adapt selectively, especially import/version checks, dependency
  doctor UX, rank-zero logging, and recursive object application helpers. Do
  not add it as a dependency to exact-eval paths unless pinned and scanned.

Security and supply-chain rule:

- Continue the repo rule forbidding installation or execution of the PyPI
  package named `lightning` in this project or on promotion runners.
- Any Lightning ecosystem dependency must pass
  `scripts/scan_lightning_supply_chain.py --strict` locally and remotely, and
  must not introduce `lightning` console-script execution.
- No external model registry or remote artifact manager is allowed to become
  canonical custody for contest archives, renderer payloads, sensitivity maps,
  or exact-eval JSON.

Assigned subagent research:

- Ohm: LitModels deep review.
- Banach: lightning-thunder compiler/deep-learning systems review.
- Arendt: utilities CLI/shared-helper review.

Next use decisions:

1. Wait for subagent reports and fold only concrete, audited APIs into a
   candidate adoption matrix.
2. If lightning-thunder remains interesting, create an opt-in benchmark plan
   that compares eager/Torch compile/Thunder on a non-promotable renderer
   training or sensitivity loop with fixed seeds, fixed inputs, CUDA-only
   hardware, output tensor hashes, and max absolute/relative drift gates.
3. If utilities remains useful, adapt local dependency doctor and rank-zero
   patterns without adding `lightning` or changing exact-eval custody.
4. LitModels should remain inspiration for checkpoint manifests only unless a
   later review proves a no-`lightning`, no-cloud-custody, deterministic local
   subset worth vendoring or reimplementing.

## 2026-05-01T05:14Z LitModels Subagent Review - Ohm

Reviewer: Ohm (`019de1f1-c92f-74f2-9e95-4c7f405b2dfc`).
Scope: read-only primary-source review, no MCP, no file edits by subagent.

Upstream snapshot:

- Repository: https://github.com/Lightning-AI/LitModels
- Reviewed commit: `2b1f787a3bc9ea99e5361bbe278df630dd7ebb5f`
- Commit/date noted by reviewer: `2026-04-01`, "Switch to pyproject.toml
  (#146)".
- Latest release noted by reviewer: `v0.1.8`, published `2025-05-14`.

Grand Council intake decision:

- Do not add LitModels to exact-eval, inflate, submission, or promotion
  environments.
- Do not install `litmodels[extra]`, examples, demos, or checkpoint callback
  integrations. The optional/test paths reference `lightning >= 2.0.0` and
  `pytorch-lightning >=2.0`, which conflicts with the current supply-chain
  policy after the PyPI `lightning` compromise.
- Do not use LitModels for deterministic archive construction, score custody,
  CUDA auth eval, SegNet/PoseNet perturbation analysis, KL/aux loss hardening,
  renderer/codec training logic, or Batch orchestration.
- If model storage on Lightning is ever useful, use direct `lightning-sdk`
  with pinned hashes, local source/artifact manifests, and the existing
  supply-chain scanner. Cloud registry state must never become contest
  evidence or artifact custody.

Useful but non-adopted references:

- `src/litmodels/io/cloud.py`: `upload_model_files`,
  `download_model_files`, teamspace/model-version routines.
- `src/litmodels/io/gateway.py`: `save_model`, `load_model`.
- `src/litmodels/integrations/checkpoints.py`: async cloud checkpoint
  callback pattern. This is explicitly not contest-custody-grade because it
  uses daemon-thread/cloud behavior.
- `src/litmodels/integrations/mixins.py`: generic registry mixins.

Security notes:

- Base dependencies observed by reviewer are `lightning-sdk >=0.2.11` and
  `lightning-utilities`.
- Integration code probes/imports `lightning` and `pytorch_lightning` when
  present.
- Generic pickle/joblib loading support is not acceptable for Pact promotion
  custody except in explicitly fenced forensic tooling.
- `src/litmodels/integrations/duplicate.py` was flagged as unsuitable for
  evidence because it downloads Hugging Face `revision="main"` and uploads
  cloud artifacts.

## 2026-05-01T05:15Z lightning-thunder Subagent Review - Banach

Reviewer: Banach (`019de1f1-e094-7b23-9715-7cf30b591945`).
Scope: read-only primary-source review, no MCP, no file edits by subagent.

Upstream snapshot:

- Repository: https://github.com/Lightning-AI/lightning-thunder
- Reviewed commit: `99850362f965db57077c3552f81c7f4abd8add7f`
- Commit/date noted by reviewer: `2026-03-26`.
- Latest stable release noted by reviewer: `0.2.6`, dated `2025-10-22`.

Grand Council intake decision:

- Thunder is allowed only as an `experimental/non_promotable` shadow benchmark
  or profiling tool for training-side loops. It is forbidden in canonical
  `archive.zip -> inflate.sh -> upstream/evaluate.py`, archive inflation,
  exact auth eval, promotion-grade component sensitivity, and score custody.
- Keep Thunder in a separate environment. Its stable dependency floor
  `torch>=2.7.1` is higher than this repo's current broad `torch>=2.0,<3.0`
  allowance and could create scorer-runtime drift if installed into canonical
  eval environments.
- Candidate experiments: compare eager, `torch.compile`, and Thunder on fixed
  CUDA inputs for renderer training, neural weight codec training, and Hessian
  or sensitivity profiling. Any adoption requires tensor-output hash checks,
  gradient parity, max absolute/relative drift gates, deterministic replay, and
  downstream exact archive eval.

Useful candidate APIs/files:

- `thunder.examine.examine` for unsupported-op preflight.
- `thunder.jit` / `thunder.compile`.
- `compile_data`, `compile_stats`, `last_traces`, `last_backward_traces`.
- `thunder.dynamo.ThunderCompiler`.
- `CUDAGraphTransform`.
- `ProfileTransform` and `NvtxProfileTransform`.

Security and maturity notes:

- Runtime base deps observed by reviewer include `lightning-utilities`, not the
  compromised PyPI package named `lightning`.
- Benchmark/test/notebook paths can import `lightning`; do not install or run
  Thunder devel benchmark/test extras inside Pact promotion environments unless
  the supply-chain scanner passes and `lightning` is absent or explicitly
  audited.
- Official docs still identify material alpha limitations: no dynamic shapes,
  in-place limitations, tensor subclass limits, and Python side-effect/trace
  sharp edges. These are incompatible with unreviewed score-path use.

Recommended next experiment:

- Create a non-promotable Thunder shadow-benchmark plan only after the current
  component-sensitivity promotion chain has real CUDA maps and structured
  prediction deltas. Benchmark candidates:
  `experiments/train_segmap.py`,
  `experiments/train_neural_weight_codec.py`, and
  `experiments/profile_hessian_per_weight.py`.

## 2026-05-01T05:18Z utilities Primary-Source Interim Review

Upstream snapshot:

- Repository: https://github.com/Lightning-AI/utilities
- Observed `main`: `5646a0d6cb263c3750b5d3dc90ed9af3686e166f`

Primary-source observations:

- Base package requirements are small: `packaging >=22` and
  `typing_extensions`.
- CLI extra requirements are `jsonargparse[signatures] >=4.38.0` and
  `tomlkit`.
- Useful source areas:
  - `src/lightning_utilities/core/imports.py`: package/module availability,
    version comparison, requirement cache, lazy module pattern.
  - `src/lightning_utilities/core/rank_zero.py`: rank-zero-only decorators,
    rank-prefixed messages, warning de-duplication cache.
  - `src/lightning_utilities/core/apply_func.py`: recursive application over
    nested containers.
  - `src/lightning_utilities/cli/dependencies.py`: dependency-file rewrite and
    requirement-management CLI patterns.

Interim adoption decision:

- Good design-reference source for local preflight and DX hardening patterns.
- Do not add as a dependency to canonical exact-eval/inflate paths yet.
- If borrowed, adapt small patterns locally under Pact's stricter fail-closed
  rules, preserving deterministic JSON output and avoiding hidden imports.
- Do not use the dependency-rewrite helpers directly on Pact lockfiles or
  `pyproject.toml`; our reproducibility policy requires explicit reviewed
  changes and `uv.lock` custody, not broad automated rewrite commands.

Open:

- Await Arendt subagent review for deeper security and CLI-pattern audit.

## 2026-05-01T05:20Z utilities Subagent Review - Arendt

Reviewer: Arendt (`019de1f2-13e9-7311-a06c-966b185dd398`).
Scope: read-only primary-source review, no MCP, no file edits by subagent.

Upstream snapshot:

- Repository: https://github.com/Lightning-AI/utilities
- Reviewed main commit: `5646a0d6cb263c3750b5d3dc90ed9af3686e166f`
- Commit/date noted by reviewer: `2026-04-21`.
- Release/PyPI noted by reviewer: `v0.15.3` / `lightning-utilities==0.15.3`,
  wheel SHA-256
  `6c55f1bee70084a1cbeaa41ada96e4b3a0fea5909e844dd335bd80f5a73c5f91`.

Grand Council intake decision:

- utilities is useful as a pattern library and possible dev-only dependency
  with hash pinning, but not as a contest-runtime dependency without a concrete
  reviewed need.
- Do not add utilities to exact CUDA auth eval, inflate, archive custody, or
  runner preflight authority paths by default.
- It is acceptable to adapt small patterns locally: rank-zero logging, warning
  de-duplication, requirement/import diagnostics, package build checks, schema
  checks, and stricter warning gates.
- For high-risk package names, use metadata-only checks. Do not call
  `module_available("lightning")`, `compare_version("lightning", ...)`,
  `lazy_import("lightning")`, or any helper that imports the compromised
  package namespace.

Adopt candidates:

- `core/imports.py`: `package_available`, `RequirementCache`, `requires`,
  with the import-trigger caveat above.
- `core/rank_zero.py`: `rank_zero_only`, rank-prefixed messages,
  `WarningCache`.
- `test/warning.py`: no-warning test guards.
- `install/requirements.py` and `cli/dependencies.py`: requirement parser and
  CLI UX ideas only.
- CI/package patterns: build wheel/sdist, `twine check --strict`, install in a
  disposable environment, `check-jsonschema`, and pre-commit hooks for private
  keys, large files, and case conflicts.

Avoid:

- Do not use README workflow examples pinned to `@main`; pin action/workflow
  refs to commit SHAs if borrowed.
- Do not run `requirements set-oldest` or dependency-rewrite helpers on Pact.
  They blindly rewrite `>=` to `==` and conflict with upper-bound plus
  `uv.lock` discipline.
- Do not use docs/cloud asset download helpers for evidence custody because
  they do not provide Pact's hash/provenance model.
