# PR91 HPM1 First-Principles Parity Addendum - 2026-05-04

Scope: local-only PR91/HPM1 entropy and reference-prefix parity. No remote jobs,
no GPU dispatch, no scorer loads, and no score claim.

## Surfaces Touched

- `src/tac/pr91_hpm1_codec.py`
- `src/tac/tests/test_pr91_hpm1_codec.py`
- `.omx/research/pr91_hpm1_parity_first_principles_20260504_codex.md`

## Static Contract

Archive custody remains the downloaded PR91 single-member archive:

- archive: `222404` bytes,
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- member `x`: `222304` bytes,
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- HPM1 mask segment: `145087` bytes,
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- HPM1 token stream: `116796` bytes / `29199` uint32 words,
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPM1 HPAC PPMd model: `28243` bytes,
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- HPM1 config: `N=600`, `H=384`, `W=512`, `P=32`, `delta=2`,
  `ch=64`, `use_spm=1`, `hpac_d_film=8`, `ppmd_order=4`

The PR91 HPAC model blob is byte-identical to the known PR86 HPAC PPMd model;
the token stream is distinct from PR86 (`+2896` bytes, first mismatch at uint32
word `41`).

## Runtime Source Contract

The submitted HPM1 branch delegates to `pr86_hpac.decompress_tokens_hpac`.
The call's device argument is `str(device)`, where the main runtime device is
selected by CUDA availability. A source comment says "Force HPAC decode onto
CPU", but the actual HPM1 call does not force CPU. The local analyzer now records
this contradiction as:

- `decode_device_argument = "str(device)"`
- `explicit_hpac_cpu_force_detected = false`
- `hpac_cpu_force_comment_detected = true`
- `hpac_cpu_force_comment_matches_hpm1_call = false`

No fallback around HPM1 entropy failure was detected. If HPAC decode asserts,
inflate aborts before any old-PR/QMA fallback.

## Entropy Matrix

Local dependencies match the recorded PR86 versions:
`python=3.12.13`, `torch=2.11.0`, `numpy=2.4.4`,
`constriction=0.4.2`, `pyppmd=1.3.1`.

Probability/categorical variants on the submitted raw little-endian uint32 token
stream all fail closed on frame 0:

| variant | decoded before failure | failure coordinate |
|---|---:|---|
| `source_float64_perfect_false` | `5951` | `frame=0 group=10 symbol=191` |
| `source_float32_perfect_false` | `30513` | `frame=0 group=24 symbol=561` |
| `source_float64_perfect_true` | `13822` | `frame=0 group=15 symbol=1534` |
| `source_float32_perfect_true` | `12479` | `frame=0 group=15 symbol=191` |

The off-contract float32/perfect-false path surviving to `30513` symbols is the
strongest local hint that the missing contract may involve device/probability
numeric state, but it still fails and cannot unlock dispatch.

Byte/word stream transforms also fail closed:

| transform | decoded before failure | failure coordinate |
|---|---:|---|
| `raw_le_u32` | `5951` | `frame=0 group=10 symbol=191` |
| `word_byteswap` | `4323` | `frame=0 group=8 symbol=483` |
| `word_reverse` | `5279` | `frame=0 group=9 symbol=479` |
| `byte_reverse` | `12423` | `frame=0 group=15 symbol=135` |

This rules out the simple byte-order, word-order, and whole-stream reversal
contracts.

## PR85 Reference Prefix Probe

Added `run_pr91_hpm1_reference_prefix_probe(...)`, a local-only profiler that
decodes PR91 HPM1 under explicit probability variants and compares the decoded
symbol prefix to the recorded PR85 QMA9 token source in the same patch-group
scan order.

Reference token source:

- path:
  `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`
- bytes: `117964800`
- SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- shape/order: `[600,384,512]`, frame-major header-width/header-height order

Under the submitted source contract (`float64`, clipped/renormalized,
`Categorical(..., perfect=False)`), the decoded PR91 prefix does not match the
PR85 reference before the entropy assertion:

- first PR85 reference mismatch:
  `global_symbol=7`, `frame=0`, `group=0`, `symbol=7`,
  `pixel=(y=0,x=224)`, decoded `2`, PR85 reference `0`
- entropy assertion still occurs at:
  `frame=0`, `group=10`, `symbol=191`, `pixel=(y=37,x=480)`
- reference symbol at the entropy-failure coordinate: `1`
- normalized source-contract probability row at failure:
  `[0.0324105338, 0.0000581093, 0.9667295674, 0.0007811914, 0.0000205981]`

Interpretation: local evidence does not prove that PR91 is a PR85-mask-identity
stream. Because the entropy contract is unresolved, this is not evidence that
PR91 changes the intended masks; it only proves the current local source
contract cannot support the PR85-identity claim.

## Exact Missing Bit

The next missing bit is the real PR91 HPAC token-generation/probability trace.
Specifically:

1. The encoder or build recipe that produced
   `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`.
2. A per-symbol trace for frame `0`, group `0`, symbols `0..8`, showing whether
   symbol `7` is intended to be PR85 reference `0` or decoded value `2`.
3. A per-symbol probability/range state trace at frame `0`, group `10`,
   symbol `191`, ideally on the claimed CUDA evaluation path.

Without one of those, PR91 remains external/forensic intelligence only. Do not
dispatch PR91-derived archives from this lane.

## Verification

- `.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py src/tac/tests/test_pr91_hpm1_codec.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py::test_real_pr91_reference_prefix_probe_shrinks_pr85_identity_claim_if_available -q`
- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q`
