# PR86 HPAC Contract Unblock Worker - 2026-05-04

Worker: HPAC-Parity

Scope: local CPU-only PR86 HPAC replay/parity. No GPU work, no remote dispatch,
no score claim.

## Current Source Context

- Fresh merged intake: `experiments/results/public_pr86_intake_20260504_merged_refresh`
- PR86 head SHA: `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4`
- Merge commit: `14bcede815306415a0005c3cd98804151bce4049`
- Merged at: `2026-05-04T03:36:55Z`
- Archive identity still matches stale cached archive:
  - bytes: `207579`
  - SHA-256: `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- Refreshed `inflate.py`, `training/archive.py`, and `training/hpac.py` are
  byte-identical to the stale cache, so the old archive decode failure remains
  relevant but is now classified under the merged source context.

## Contract Classification

- Training objective in `training/hpac.py`: residual tokens.
- Archive builder in `training/archive.py`: passes raw SegNet token maps into
  `encode_frame(gen, tokens_t[f:f + 1], ...)`.
- Inflate path: decodes HPAC symbols directly as tokens; it does not reconstruct
  residuals.
- Frame 0 cannot distinguish raw from residual semantics because
  `residual[0] == token[0]`.
- Current merged encode/decode source uses clipped and renormalized
  `np.float64` probabilities with `constriction.stream.model.Categorical(...,
  perfect=False)`.
- The README mentions a `1/16384` probability grid, but the merged archive and
  inflate code do not implement that explicit grid.
- Inflate comments claim HPAC is forced to CPU, but the merged code passes the
  main runtime `device` into `decompress_tokens_hpac`; on CUDA hosts this remains
  a comment/code mismatch.

## Local Evidence

Focused replay CLI:

```text
.venv/bin/python experiments/replay_pr86_hpac_tokens.py --max-frames 1 --json-out /tmp/pr86_hpac_contract_unblock_worker_report.json
```

Result: `failed_closed`, `dispatch_unlocked=false`.

- Failure stage: `submitted_tokens_decode`
- Failure reason: `hpac_entropy_decode_contract_mismatch`
- Failed at: frame `0`, group `10`, symbol in group `191`
- Decoded symbols before failure: `5951`
- Tokens SHA-256: `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`

Submitted merged inflate path:

```text
.venv/bin/python experiments/results/public_pr86_intake_20260504_merged_refresh/inflate.py <unzipped_archive_tmpdir> unused /tmp/pr86_inflate_current.raw
```

Result: same constriction assertion in `decompress_tokens_hpac`, before render.

Bounded CPU-only variant probe over frame 0 groups 0-11:

| Variant | Result |
| --- | --- |
| current-source baseline | fails at group 10 symbol 191 |
| residual-state frame 0 | fails at group 10 symbol 191 |
| reversed uint32 words | fails earlier at group 9 symbol 12 |
| big-endian uint32 values | fails earlier at group 8 symbol 483 |
| reversed symbols within group | fails earlier at group 3 symbol 81 |
| explicit 16384 grid | fails at group 10 symbol 274 |
| reversed group order | prefix-only passes 12 groups, but is off-contract and not byte parity |
| `float32` probabilities into `Categorical` | prefix-only passes 12 groups, off-contract |
| `perfect=True` | prefix-only passes 12 groups, off-contract |

Interpretation: raw/residual semantics, uint32 endian/stack direction, symbol
order, and explicit 16384-grid quantization do not explain the current
source-contract failure. The live implementation target is probability
model contract recovery: prove or reject `float32` probability input and
`perfect=True` variants with full decode and byte-exact re-encode before any
PR85 transfer or dispatch.

## Verification

```text
.venv/bin/python -m py_compile src/tac/pr86_hpac_codec.py experiments/replay_pr86_hpac_tokens.py
.venv/bin/python -m pytest src/tac/tests/test_pr86_hpac_codec.py
```

Pytest result: `7 passed, 1 warning`.

## Dispatch State

`dispatch_unlocked=false`. No GPU work or remote dispatch was performed.
