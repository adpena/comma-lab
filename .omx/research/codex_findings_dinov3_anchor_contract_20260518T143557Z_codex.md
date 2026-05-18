# Codex Findings: DINOv3 Anchor Contract Hardening - 2026-05-18T14:35:57Z

## Finding

Codex adversarial review found a split-brain DINOv3 contract after the library
anchor was hardened: `src/tac/dinov3_cooperative_receiver_anchor.py` uses the
canonical timm/HF DINOv3 base/16 256px transform and excludes register tokens,
but the submitted HF Jobs anchor extraction script still used a stale
224px / 196-patch assumption and emitted score-like
`[contest-CPU frozen-anchor]` labels for a tool-side frozen feature dataset.

This is a downstream-poisoning bug, not a score result. The anchor dataset feeds
MI/probe decisions for ATW V2-1, Z6, and Z7. A stale transform or
register-token-inclusive patch grid would make future DINOv3-vs-scorer probes
compare the wrong feature distribution.

## Patch

- `submitted_jobs/training_dinov3_cooperative_receiver_anchor_20260518T140408Z.py`
  now uses `DINOV3_INPUT_SIZE = 256`.
- The submitted job now resolves the timm model data config and fails closed if
  the resolved input size drifts from the pinned 256px contract.
- The submitted job now strips CLS/register prefix tokens before writing patch
  tokens, using explicit `num_prefix_tokens` when exposed and a square-grid
  fallback otherwise.
- Tool-side extraction labels are no longer contest-score labels:
  `[tool-CUDA frozen-anchor]`, `[tool-CPU frozen-anchor]`, and
  `[macOS-CPU advisory frozen-anchor]`.
- Provenance/README output records `input_size`,
  `register_tokens_excluded_from_patch_grid`, dynamic hidden dimension, and
  dynamic patch count.

## Verification

Focused regression coverage:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_dinov3_cooperative_receiver_anchor.py \
  src/tac/tests/test_build_comma_video_substrate_eval_600pairs_dataset.py \
  src/tac/tests/test_submitted_dinov3_anchor_job_contract.py -q
```

Expected coverage:

- library anchor still resizes to the canonical DINOv3 input size;
- library anchor strips register tokens;
- submitted job uses the 256px contract;
- submitted job does not emit `[contest-CPU]` for local macOS frozen-anchor
  extraction;
- submitted job strips register tokens in both explicit `num_prefix_tokens`
  and square-grid fallback modes.

## Evidence Discipline

No provider dispatch, no score claim, no contest-CUDA claim, and no promotion
claim occurred in this pass. This patch only protects future DINOv3 MI/probe
evidence from stale-transform false authority.

## Strict-Gate Status

A new strict preflight gate for this bug class was not added in this atomic
commit because `.omx/state/subagent_progress.jsonl` showed an active sister
checkpoint with `files_touched=["preflight"]` at
`2026-05-18T14:18:09Z`. Per no-signal-loss/anti-absorption discipline, Codex
did not edit `src/tac/preflight.py` or `CLAUDE.md` while that surface was
owned by another in-flight agent.

Reactivation criterion: after the preflight surface is stable, add a Catalog
gate that scans submitted DINOv3/timm anchor jobs for stale hardcoded transform
sizes, raw `tokens[:, 1:, :]` patch extraction, and contest-score-like labels
on tool-side frozen-anchor datasets.

