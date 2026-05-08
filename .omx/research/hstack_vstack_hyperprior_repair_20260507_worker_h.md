# HStack/VStack Hyperprior Repair Planner — Worker H

Date: 2026-05-07  
Owner: Worker H  
Scope: deterministic planning primitive only; no Ballé/shared-PMF implementation edits.

## Context

The static shared-PMF K=12 path is recorded as a negative control after charged
bytes: `+793 bytes` versus Brotli. This is not a family kill for Ballé/full
learned hyperprior or stacked multipass codecs, but it blocks promotion until a
real learned-model repair proves model-overhead amortization, byte closure, and
runtime packet consumption.

## Implemented Surface

- `src/tac/codec_stack_planner.py` defines scoreless dataclasses for:
  components, serial VStack transforms, parallel HStack groups, compress-time
  passes, byte/evidence semantics, and fail-closed promotion policy.
- `tools/plan_hstack_vstack_multipass.py` emits either a full deterministic
  manifest or compact summary. It performs no dispatch and makes no score claim.
- `src/tac/tests/test_codec_stack_planner.py` covers deterministic manifests,
  Ballé/hyperprior blockers and repair criteria, HStack/VStack/multipass shape,
  manifest roundtrip, and fail-closed score-claim validation.

## Senior Adversarial Review Correction

Codex review found and fixed one semantic inversion in the first worker draft:
HStack and VStack were swapped. The corrected project contract is now locked by
tests and by manifest metadata:

- `hstack_parallel`: horizontal, parallel independent archive-component codecs
  merged into one deterministic packet.
- `vstack_serial`: vertical, serial transforms inside one component stream:
  representation -> prediction -> quantization -> hyperprior -> arithmetic ->
  pack.
- `multipass`: compress-time repeated planning and training passes only.

The schema was advanced to `tac_hstack_vstack_multipass_plan_v2`; `plan_from_manifest`
keeps a read-only compatibility path for the worker's v1 alias fields so
existing intermediate notes are not lost.

## Design Mandates Captured As Non-Score Metadata

The planner now carries the full design contract without turning it into a
score claim:

- six-level nested optimization:
  meta-pass score feedback, bilevel substrate training, multipass refinement,
  HStack parallel components, VStack serial transforms, and per-tensor HStack
  parallel substreams;
- prediction-only nested score band: linear stack `0.130`, full nested
  score-feedback `0.090`, `score_claim=false`, `promotion_allowed=false`;
- DAG-of-CodecOps planning with a single-member monolithic packet target:
  logical stream budgets require parser-proven internal offsets/lengths/hashes,
  not ZIP member-name categories;
- canonical five-pass QAT pipeline: anchor, finetune, joint, QAT, final;
- 11-point quality mandate: beautiful, elegant, human-readable, composable,
  creative, reusable, expressive, canonical, production-hardened, OSS-ready,
  paper-ready.

Materialized planner artifact:
`reports/hstack_vstack_multipass_plan_20260507.json`.

## 2026-05-08 Monolithic Frontier Archive Correction

Follow-up adversarial review verified the public frontier archive shape:

- local PR101 archive: one stored ZIP member `x`, archive `178,258` bytes,
  inner member `178,158` bytes, split by fixed parser offsets into
  `decoder_blob` `162,164` bytes, `latent_blob` `15,387` bytes, and
  `sidecar_blob` `607` bytes;
- local PR106 archive: one stored ZIP member `0.bin`, archive `186,239`
  bytes, inner member `186,131` bytes, split by FF grammar into
  `decoder_packed_brotli` `170,278` bytes and
  `latents_and_sidecar_brotli` `15,849` bytes.

This falsifies ZIP-member-level mask/pose/renderer budgets for these HNeRV
frontier substrates. It does **not** prove that logical streams are absent in a
general architecture; it means any logical budget claim must cite an internal
parser section, not a file member. The planner metadata now encodes this:
`member_level_component_budgets_valid=false` and
`logical_stream_budget_requires_internal_parser_proof=true`.

## Ballé / Full Learned Hyperprior Gate

The learned hyperprior candidate is explicitly `score_claim=false`,
`dispatchable=false`, and `promotion_eligible=false` until all of these are
repaired:

- model overhead is measured and amortized against conditional entropy savings;
- model weights, z stream, entropy tables, headers, and archive delta are all
  counted as charged bytes when score-affecting;
- exact encode/decode reconstruction passes on canonical qint vectors;
- `archive.zip` contains and `inflate.sh` consumes all score-affecting model
  and side-info bytes without sidecars or network/local state;
- old/new archive SHA-256s and charged-byte deltas are recorded;
- full-sample exact CUDA auth eval exists before any score claim.

## Promotion Policy

The planner is fail-closed by construction. Planning manifests retain default
blockers (`planning_artifact_only` and
`score_claim_forbidden_until_exact_cuda_auth_eval`) plus component-level
blockers. Exact-eval dispatch requires charged-bit proof, old/new archive
identity, exact roundtrip, runtime packet closure, and sidecar-free inflate.
Score promotion additionally requires full-sample CUDA auth eval and manifest
identity.

## Verification

Focused verification intended for this patch:

```text
.venv/bin/python -m pytest src/tac/tests/test_codec_stack_planner.py
.venv/bin/python -m ruff check src/tac/codec_stack_planner.py tools/plan_hstack_vstack_multipass.py src/tac/tests/test_codec_stack_planner.py
```
