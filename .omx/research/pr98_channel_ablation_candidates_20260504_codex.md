# PR98 Channel Postprocess Ablation Candidates - 2026-05-04

## Scope

Local-only candidate preparation for PR98 `hnerv_muon_finetuned_from_pr95`.
No remote GPU dispatch was performed, no dispatch claim was created, and no
score is claimed here. The work is scoped to:

- `experiments/results/pr98_channel_ablation_candidates_20260504_codex/`
- `.omx/research/pr98_channel_ablation_candidates_20260504_codex.md`

The source custody inputs are the sanitized PR98 readiness artifacts:

- Runtime: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/runtime_snapshots/pr98_runtime`
- Archive: `experiments/results/final_packet_readiness_pr98_pr99_20260504_codex/archives/pr98_archive.zip`

## Fixed Postprocess Ops

The three intake-reported fixed postprocess operations are runtime-code
operations in `inflate.py`, not archive payload operations:

```python
up[:, 0, 0].sub_(1.0)  # frame0 red -1
up[:, 0, 2].sub_(1.0)  # frame0 blue -1
up[:, 1, 1].sub_(1.0)  # frame1 green -1
```

Therefore every candidate below keeps the archive bytes unchanged and varies
only the inflate runtime tree. Any exact eval must preserve both archive SHA
and runtime-tree SHA. This is a runtime-custody comparison, not an archive-byte
comparison.

## Archive Custody

All candidates use:

- Archive: `experiments/results/pr98_channel_ablation_candidates_20260504_codex/archives/pr98_archive.zip`
- Bytes: `178392`
- SHA-256: `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- Member: single stored `0.bin`

## Candidates

| Candidate | Disabled ops | Runtime tree SHA-256 | Local preflight |
|---|---:|---|---|
| `baseline_all_ops` | none | `544f97b8834dcdbdbc565ad0dd22253359e7d19aefd7bc7f9bee35fd43c96e0b` | passed |
| `drop_f0_red` | `f0_red_minus1` | `457baab4c3de5f7818b848d9bc27d4554882f5f785bf636b8c6840b3d92ddb99` | passed |
| `drop_f0_blue` | `f0_blue_minus1` | `732095d6c5979f56b218988b20dca8cb2f8a1d29eb61a65f630f2e3ccb23bb79` | passed |
| `drop_f1_green` | `f1_green_minus1` | `5e3aee0d076936c3027b61c24f5fe0f326f0f15b06a2a4c6513cd699cae6fc58` | passed |
| `drop_f0_red_f0_blue` | `f0_red_minus1`, `f0_blue_minus1` | `1b9fa3253a361b8e1ed1fe055ad9a6c1f0970c5606dc467f55796a3e80465d4f` | passed |
| `drop_f0_red_f1_green` | `f0_red_minus1`, `f1_green_minus1` | `fedb84920bd9a7ae683e349c27bd5ec00edda6a96cbc8ab4ad8d8891addc6e30` | passed |
| `drop_f0_blue_f1_green` | `f0_blue_minus1`, `f1_green_minus1` | `d483bd072e62e0770a59e02d5eb9a878ade8516d5a05077cd6912aa586f248b3` | passed |
| `drop_all_three` | all three | `2a1474488568115e557b8f2dfaed5dbae22b84d6ceb4b30700ae037f7c3e0553` | passed |

Primary machine-readable artifact:

- `experiments/results/pr98_channel_ablation_candidates_20260504_codex/candidates_manifest.json`

Each candidate has:

- A runtime tree under `runtime_variants/<candidate>/`
- A static preflight JSON at `<candidate>.preflight.json`
- A dispatch command plan at `eval_command_plans/<candidate>.exact_eval_plan.json`

## Verification

Commands run:

```bash
.venv/bin/python experiments/results/pr98_channel_ablation_candidates_20260504_codex/build_candidates.py
find experiments/results/pr98_channel_ablation_candidates_20260504_codex -name __pycache__ -o -name '._*' -o -name '.DS_Store' -o -name '.git' | sort
git diff --check -- experiments/results/pr98_channel_ablation_candidates_20260504_codex
```

The builder also ran, for every candidate:

- Python syntax compilation via `compile(...)`, without writing pycache.
- `bash -n` on `inflate.sh`.
- `experiments/preflight_public_replay_intake.py --fail-if-not-ready` with
  expected archive SHA, archive bytes, and canonical runtime tree SHA.
- `unzip -t` on the copied archive.

All local checks passed.

## Exact Eval Commands

Do not dispatch from this ledger directly. If one candidate is worth exact eval,
use the corresponding command plan JSON under:

```text
experiments/results/pr98_channel_ablation_candidates_20260504_codex/eval_command_plans/
```

Those plans include the required pre-dispatch `tools/claim_lane_dispatch.py`
claim command and the `scripts/launch_lightning_batch_job.py exact-eval`
command. Before dispatch, replace the timestamp placeholders and stage a
Lightning source manifest that includes the chosen runtime tree and unchanged
archive.

Example candidate plan path:

```text
experiments/results/pr98_channel_ablation_candidates_20260504_codex/eval_command_plans/drop_f0_blue.exact_eval_plan.json
```

The launch command must include:

- `--archive experiments/results/pr98_channel_ablation_candidates_20260504_codex/archives/pr98_archive.zip`
- `--expected-archive-sha256 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- `--expected-archive-size-bytes 178392`
- Candidate-specific `--inflate-sh experiments/results/pr98_channel_ablation_candidates_20260504_codex/runtime_variants/<candidate>/inflate.sh`
- `--adjudicate`
- A valid preexisting dispatch claim for the same lane/job.

## Blockers And Cautions

- No score claim exists until exact CUDA auth eval of a candidate runtime/archive
  pair completes.
- These are not archive variants. The archive SHA and bytes are identical for
  all candidates; the runtime tree SHA is the differentiator.
- PR98 original exact T4 replay should be harvested first. Ablations are only
  worth dispatch if original PR98 is valid and competitive, or if the channel
  postprocess ops look like the cheapest remaining exact-score surface.
- The baseline runtime tree under this candidate directory has a different
  runtime tree SHA than the readiness snapshot because the runtime root name is
  part of the canonical manifest. That is expected and is recorded explicitly.
