# Alpha-Geo-1 Next Patch - 2026-04-30

Scope: Alpha/NeRV geometry and Lane 12 diagnostics only. No score claim.

## Evidence Boundary

This patch is code/test hardening for pre-retraining custody. It is not score
evidence and cannot promote, rank, kill, or retire any method family. Exact
contest score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Inspection Result

Reviewed:

- `experiments/diagnose_nerv_geometry.py`
- `experiments/train_nerv_mask.py`
- `scripts/remote_lane_nerv.sh`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`
- `.omx/research/alpha_geo_1_visual_primitives_design_20260430_agent.md`

The smallest rigorous blocker before more Alpha-Geo-1 retraining spend was not
another CPU geometry metric. The current diagnostic script already records
decoded-mask hashes, ZIP member custody, 2px boundary gates, temporal drift, and
component drift. The remaining blocker was that `experiments/train_nerv_mask.py`
could still only train on fresh SegNet, AMRC, or synthetic targets. That repeats
the known geometry mismatch instead of training to the decoded baseline
`masks.mkv` distribution used by the renderer and optimized poses.

## Patch

Added an explicit `--gt-masks-source decoded-baseline` path to
`experiments/train_nerv_mask.py`.

Behavior:

- Requires `--decoded-baseline-path`.
- Defaults ZIP member selection to `masks.mkv`.
- Rejects unsafe ZIP member paths and duplicate ZIP members.
- Records source archive/file size and SHA-256.
- Records resolved archive member, member bytes, compressed bytes, and member
  SHA-256.
- Records decoded target mask shape, dtype, and deterministic mask tensor
  SHA-256.
- Fails closed if decoded-baseline masks do not match requested scorer geometry.
- Fails closed if target class IDs exceed the profile `nerv_num_classes`.

Also wired `scripts/remote_lane_nerv.sh` so a remote run can set:

```bash
GT_MASKS_SOURCE=decoded-baseline
DECODED_BASELINE_PATH=<baseline archive.zip>
DECODED_BASELINE_MEMBER=masks.mkv
```

The default remote behavior remains `GT_MASKS_SOURCE=segnet`.

## Changed Files

- `experiments/train_nerv_mask.py`
- `scripts/remote_lane_nerv.sh`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`
- `.omx/research/alpha_geo1_next_patch_20260430_worker.md`

Note: the worktree already contained unrelated and pre-existing edits in these
and many other files. This patch did not revert or clean them up.

## Verification

```bash
.venv/bin/python -m py_compile experiments/train_nerv_mask.py src/tac/tests/test_lane12_nerv_dependency_closure.py
bash -n scripts/remote_lane_nerv.sh
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py -q
git diff --check
```

Result:

```text
9 passed in 0.82s
```

`py_compile`, `bash -n`, and `git diff --check` passed with no output.

## Remaining Before Large Retraining Spend

1. Run a small decoded-baseline NeRV training smoke and confirm
   `provenance.json` contains the baseline member SHA and target mask SHA.
2. Run `experiments/diagnose_nerv_geometry.py` candidate-vs-baseline after the
   smoke archive is built; treat failure as empirical rejection only.
3. Add pose regeneration provenance or a pose-rescue gate before exact CUDA eval
   spend on any mask-changing archive.
4. Add renderer embedding drift only after the decoded-baseline target path and
   pose-rescue provenance are exercised, because target custody was the first
   blocker.
