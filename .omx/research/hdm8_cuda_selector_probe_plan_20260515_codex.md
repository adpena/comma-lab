# HDM8 CUDA selector probe plan and top001 exact-CUDA result

Date: 2026-05-15
Author: Codex
Axis labels: `[contest-CPU]` and `[contest-CUDA]` are separate evidence spaces.

## Question

The operator asked whether the `0.192` result is legitimate and whether the HDM8
film-grain/selector/water-fill family has been fully engineered and exhausted.

Short answer:

- `0.1920513168811056` is a legitimate `[contest-CPU]` replay for the PR101
  FEC6 fixed-Huffman K16 selector archive, but it is not a `[contest-CUDA]`
  frontier claim. The same archive's exact-CUDA replay scored
  `0.22621002169349796`.
- HDM8/film-grain/selector is not exhausted as a family, but the current
  broad/proxy-ranked selector path is not promotable. The first exact-CUDA
  sparse selector probe regressed.
- The new engineering surface is CUDA-in-loop sparse calibration, not
  MPS/CPU-ranked full selector promotion.

## Code landing

Reusable code landed in the library/tool surface, not in an experiment-only
fork:

- `src/tac/optimization/hdm8_cuda_selector_probe_plan.py`
- `tools/build_hdm8_cuda_selector_probe_plan.py`
- `tools/build_hdm8_film_grain_sidecar_packet.py`
- `src/tac/tests/test_hdm8_cuda_selector_probe_plan.py`
- `src/tac/tests/test_hdm8_film_grain_sidecar.py`

The planner consumes a CUDA-prefix sweep and emits sparse byte-closed selector
configs. It fails closed for non-CUDA/MPS/macOS axes. The packet builder now
accepts explicit selector config JSON with `score_claim=false`, validates
palette/index structure, and can charge selector bytes inside the archive.

An untracked duplicate experiment-level planner/test was removed after the
canonical `tac`/`tools` surface was tightened. This prevents two selector
planner implementations from diverging.

## Local plan artifact

Plan:

- `experiments/results/hdm8_cuda_selector_probe_plan_20260515_codex/probe_plan.json`
- `experiments/results/hdm8_cuda_selector_probe_plan_20260515_codex/probe_plan.md`

Inputs:

- Sweep: `experiments/results/modal_hdm8_postfilter_sweep/hdm8_cuda_full_aggressive_v1_fix1_20260515T023053Z/hdm8_postfilter_sweep.json`
- Axis: `modal-t4-cuda-proxy-prefix`
- Candidate atoms: `64`
- Prefix probes emitted: `1, 2, 4, 8, 16, 32`

Top proxy rows:

| probe | selected pairs | proxy delta vs none | selected modes |
|---|---:|---:|---|
| sparse_cuda_prefix_top001 | 1 | -0.00008726612267542788 | even_checker:6 x1 |
| sparse_cuda_prefix_top002 | 2 | -0.00016257835757399475 | even_checker:6 x1; even_rgb_bias:-1,0.5,0.5+even_grain_chroma:1 x1 |
| sparse_cuda_prefix_top004 | 4 | -0.0002988582778070681 | four one-pair modes |

## Byte-closed top001 packet

Packet path:

- `experiments/results/hdm8_cuda_selector_probe_plan_20260515_codex/sparse_cuda_prefix_top001/archive.zip`

Custody:

- Archive bytes: `186518`
- Archive SHA-256: `98fd0bd779404970f11ca616b5c98dcb3ec41f74fb0a4ffe6d4ce613684d1223`
- Runtime tree SHA-256: `b15b4c9aacd9e94471ca7fbc28fbd26c7eb37a3abdc03ee0ce5e2c084345a5fa`
- Selector codec: brotli format `0x04`
- Selector encoded bytes: `121`
- Archive byte delta vs source: `+123`
- Proxy charged delta vs none: `-0.000005365471441420855`

## Exact-CUDA result

Modal call:

- Call ID: `fc-01KRMSNSHBCHZA83E5FF8K6BZW`
- Lane ID: `lane_hdm8_cuda_selector_sparse_top001_20260515`
- Job ID: `modal_hdm8_cuda_selector_sparse_top001_t4_20260515T030736Z`

Recovered artifact:

- `experiments/results/modal_auth_eval/hdm8_cuda_selector_sparse_top001_t4_20260515T030736Z/contest_auth_eval.json`

Exact CUDA score:

- Score: `0.2064796628814009`
- avg PoseNet distance: `0.00003249`
- avg SegNet distance: `0.0006426`
- Archive bytes: `186518`
- Evidence grade: `[contest-CUDA]`

Comparator:

- HDM8 exact-CUDA baseline used by the review packet:
  `0.20636166502462222`
- Delta: `+0.00011799785677868` regression

Result review:

- `.omx/research/hdm8_cuda_selector_sparse_top001_exact_cuda_result_review_20260515_codex.json`
- `.omx/research/hdm8_cuda_selector_sparse_top001_exact_cuda_evidence_row_20260515_codex.json`
- Classification: `measured_config_retired`
- Family retirement: false

## Prior exact-CUDA selector comparator

The broad 600-pair selector had already been measured on exact CUDA:

- Artifact: `experiments/results/modal_auth_eval/hdm8_selector_cuda_full_aggressive_v1_clean_20260515T023845Z/contest_auth_eval.json`
- Archive SHA-256: `34dc94644f5619ea7e6254079e3e4d3bbf0952f8a0ad287f675f7a249f359071`
- Archive bytes: `187226`
- Score: `0.2095197967107254`
- avg PoseNet distance: `0.00004241`
- avg SegNet distance: `0.0006426`
- Delta vs `0.20636166502462222` HDM8 baseline: `+0.0031581316861031827`

This is a measured-config regression for the broad proxy-ranked selector. It
does not retire film-grain or selector work as a family; it blocks promotion of
the broad selector without exact-CUDA sparse calibration.

## Budget-128 follow-up

A stronger sparse packet existed from a sister planner artifact and was fired as
the next cheap exact-CUDA probe after top001 regressed:

- Packet: `experiments/results/hdm8_cuda_sparse_selector_probe_20260515_codex/sparse_budget_128_packet/archive.zip`
- Archive SHA-256: `b0645ee705cf7fe34300f5b4586efe03d5c3262c66ac72a503b90599a996d004`
- Archive bytes: `186760`
- Selector encoded bytes: `363`
- Archive byte delta vs source: `+365`
- CUDA-prefix charged proxy delta: `-0.0035664157287479403`
- Runtime tree SHA-256: `d854a7b2c2087e23ab2ca7ca3c3ba7ae23a2e35d44f9dba7ba2cf8a9050718f5`
- Lane ID: `lane_hdm8_cuda_selector_sparse_budget128_20260515`
- Modal call ID: `fc-01KRMSYS1SWPD2ZX55YY299GS5`
- Output dir: `experiments/results/modal_auth_eval/hdm8_cuda_selector_sparse_budget128_t4_20260515T031500Z`

Recovered exact-CUDA artifact:

- `experiments/results/modal_auth_eval/hdm8_cuda_selector_sparse_budget128_t4_20260515T031500Z/contest_auth_eval.json`
- Score: `0.20787717836935493`
- avg PoseNet distance: `0.00003710`
- avg SegNet distance: `0.0006426`
- Delta vs `0.20636166502462222` HDM8 baseline: `+0.00151551334473271`

Result review:

- `.omx/research/hdm8_cuda_selector_sparse_budget128_exact_cuda_result_review_20260515_codex.json`
- `.omx/research/hdm8_cuda_selector_sparse_budget128_exact_cuda_evidence_row_20260515_codex.json`
- Classification: `measured_config_retired`
- Family retirement: false

This is a stronger negative than top001 because it had much more proxy mass
(`-0.003566` charged) but still transferred to an exact-CUDA regression. The
current proxy-ranked sparse water-fill selector should not receive broader
spend without a new transfer model or a fixed-mode positive control.

## Selector-family status

What is confirmed:

- The `[contest-CPU]` `0.1920513168811056` PR101/FEC6 K16 selector result is a
  real CPU-axis replay, not an exact-CUDA frontier result.
- The broad selector exact-CUDA replay regressed.
- The top001 exact-CUDA sparse selector replay regressed.
- The budget-128 exact-CUDA sparse selector replay regressed.

What is not exhausted:

- Fixed postfilter modes independent of selector packing.
- CUDA-in-loop atom selection using true exact-CUDA deltas rather than prefix
  proxy pair deltas.
- A learned transfer model that treats CPU/MPS/CUDA prefix deltas as features,
  not authority.
- Low-level PacketIR/sidecar byte shaving, which remains valid when it preserves
  decoded outputs and only moves byte term.

Current recommendation: stop broad selector promotion from the existing proxy
table; use one fixed-mode exact-CUDA positive-control probe, then either build
the transfer model or move the engineering budget to PacketIR/PR106/D4/C6.

## Interpretation

The best one-atom CUDA-prefix proxy selector did not transfer to exact CUDA
after charged bytes. This falsifies the current local linearized water-fill
ranking for at least the top atom and prevents broad selector promotion from
the existing proxy table.

This does not prove film grain is exhausted. It narrows the viable path:

1. Exact-CUDA sparse atom calibration must precede any broad selector.
2. Proxy rows need a transfer model keyed by pair, mode, PoseNet delta, SegNet
   delta, and byte charge.
3. Selector promotion should require a small exact-CUDA positive control before
   spending on larger prefixes.

## Verification

Commands run:

```bash
.venv/bin/ruff check src/tac/optimization/hdm8_cuda_selector_probe_plan.py tools/build_hdm8_cuda_selector_probe_plan.py tools/build_hdm8_film_grain_sidecar_packet.py src/tac/tests/test_hdm8_cuda_selector_probe_plan.py src/tac/tests/test_hdm8_film_grain_sidecar.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_hdm8_cuda_selector_probe_plan.py src/tac/tests/test_hdm8_film_grain_sidecar.py -q
```

Results:

- Ruff: pass
- Pytest: `22 passed in 2.00s`
