# PR75 All-600 Raw-Output Parity Control - 2026-05-03

Scope: Worker PARITY-CONTROL. Local-only deterministic parity proof for public
PR75 `qpose14_r55_segactions_minp` versus `submissions/robust_current` after
the robust runtime began preserving `optimized_poses.qp1` and decoding QP1
poses directly to float32. No scorer loads and no remote GPU dispatch.

This is empirical local raw-output parity evidence, not score evidence. Any
score claim still requires exact CUDA auth eval through
`archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Artifact

- Tool: `experiments/pr75_raw_output_parity.py`
- Test: `src/tac/tests/test_pr75_raw_output_parity.py`
- Report:
  `experiments/results/pr75_raw_output_parity_20260503_codex/pr75_raw_output_parity.json`
- Public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_deep_codex/downloads/pr75_pr67_qpose14_r55_segactions_minp_archive.zip`
- Public archive SHA-256:
  `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
- Robust runtime SHA-256:
  `57eff4e2d7c4de0ef20c8e0cea66e9464684ed14956872e603bb4a65e8c39d9e`

## Result

The all-600 chunked parity control completed locally on CPU:

- `current_runtime_pr75_comparisons_controlled=true`
- pairs: `600`
- chunk size: `8`
- chunks completed: `75`
- elapsed: `181.2441266250098` seconds

Aggregate parity:

| Surface | Exact | Changed | Max abs / byte delta | Public SHA-256 | Robust SHA-256 |
|---|---:|---:|---:|---|---|
| QP1 pose float32 | true | 0 | 0.0 | `afd13805bc30e79f5a9c0587d08cc8e05fa5508c03f1a2b5e6a425058b530b66` | `afd13805bc30e79f5a9c0587d08cc8e05fa5508c03f1a2b5e6a425058b530b66` |
| pair masks | true | 0 | 0.0 | `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45` | `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45` |
| native renderer before actions | true | 0 | 0.0 | `b7e416623a82af7d25cce25a68c4e42d18139ced5ce5707e0f61e2caa10bddd9` | `b7e416623a82af7d25cce25a68c4e42d18139ced5ce5707e0f61e2caa10bddd9` |
| native renderer after actions | true | 0 | 0.0 | `f24eec1c35b66dbe396cc8b580813c9463a2fbad666008a176570857ce90019d` | `f24eec1c35b66dbe396cc8b580813c9463a2fbad666008a176570857ce90019d` |
| camera-resolution raw after actions | true | 0 | 0 | `260b013f1cff8c40cbc7062c6fc7f3d4655e53a9eeff1144a9d860ddee77959d` | `260b013f1cff8c40cbc7062c6fc7f3d4655e53a9eeff1144a9d860ddee77959d` |

Conclusion: current-runtime PR75 comparisons are controlled for local raw
parity against the public/top-submission reference runtime. Runtime-changed
exact scores are now comparable with respect to this PR75/QP1 raw-output
contract, but they remain score claims only after exact CUDA auth eval.

## Commands

```bash
.venv/bin/python -m py_compile experiments/pr75_raw_output_parity.py src/tac/tests/test_pr75_raw_output_parity.py
.venv/bin/python -m pytest src/tac/tests/test_pr75_raw_output_parity.py -q
.venv/bin/python experiments/pr75_raw_output_parity.py --all-pairs --chunk-size 8 --fast-fail --force
```

## Caveats

- This control does not load PoseNet, SegNet, mini-scorers, or any scorer-side
  model.
- The proof is local CPU raw-output parity, not contest evidence and not a
  score claim.
- The artifact is tied to the public archive SHA and robust runtime SHA listed
  above; any runtime-file change requires a fresh all-600 parity artifact.
