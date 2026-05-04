# PR85-Family Bit/Lagrangian Profile

Local byte/Lagrangian profile only. Exact score truth remains archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA.

- Left: `pr85_stbm_a++_0.25369011029397787`
- Right: `public_pr91_hpm1_self_report_0.24879480490416128`
- Archive delta right-left: `-7352` bytes (`-0.004895395023` score rate)

| segment | left bytes | right bytes | delta bytes | delta score | same sha | left codec | right codec |
|---|---:|---:|---:|---:|---|---|---|
| `mask` | 152439 | 145087 | -7352 | -0.004895395023 | False | `opaque` | `HPM1` |
| `model` | 57074 | 57074 | 0 | 0.000000000000 | True | `brotli_qh_model` | `brotli_qh_model` |
| `pose` | 1487 | 1487 | 0 | 0.000000000000 | True | `brotli_p1d1_pose` | `brotli_p1d1_pose` |
| `post` | 1400 | 1400 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `shift` | 226 | 226 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `frac` | 106 | 106 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `frac2` | 149 | 149 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `frac3` | 154 | 154 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `bias` | 223 | 223 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `region` | 273 | 273 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |
| `randmulti` | 16101 | 16101 | 0 | 0.000000000000 | True | `brotli_qpost_sidechannel` | `brotli_qpost_sidechannel` |

## Lagrangian Target

- Target score buffer: `0.00489530538981659`
- Neutral bytes needed: `7352`

## Compliance Signal

- Source JSON: `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_preflight_frame0_20260504_codex.json`
- Status: `failed_closed`
- Failure: `submitted_tokens_decode` / `hpac_entropy_decode_contract_mismatch`
- Dispatch unlocked: `False`
