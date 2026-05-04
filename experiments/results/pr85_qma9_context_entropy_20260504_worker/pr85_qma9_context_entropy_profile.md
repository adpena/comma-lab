# PR85 QMA9 Context Entropy Planning Profile

- planning_only: true
- score_claim: false
- dispatch_performed: false
- token_source: `/Users/adpena/Projects/pact/experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`
- token_sha256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- tensor_shape: `[600, 512, 384]`
- charged_qma9_mask_bytes: 159011
- charged_qma9_bits_per_token: 0.010783624
- positive_byte_saving_models: 0

## Top Planning Opportunities

| rank | model | est ideal bytes | est bytes saved | rate-score delta | break-even overhead bytes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `left_plus_up` | 329932.045 | -170921.045 | 0.113809308 | -170921.045 |
| 2 | `left_up_time_prev` | 340734.510 | -181723.510 | 0.121002226 | -181723.510 |
| 3 | `left_plus_time_prev` | 370868.011 | -211857.011 | 0.141066888 | -211857.011 |
| 4 | `left_col_prev` | 398870.211 | -239859.211 | 0.159712403 | -239859.211 |
| 5 | `up_plus_time_prev` | 727346.410 | -568335.410 | 0.378431221 | -568335.410 |
| 6 | `up_row_prev` | 1124768.229 | -965757.229 | 0.643058098 | -965757.229 |
| 7 | `up_left_col_row_prev` | 1222481.385 | -1063470.385 | 0.708121277 | -1063470.385 |
| 8 | `time_prev_frame` | 1270545.716 | -1111534.716 | 0.740125342 | -1111534.716 |
| 9 | `global_symbol_model` | 23820963.078 | -23661952.078 | 15.755522640 | -23661952.078 |

## Axis Entropy Highlights

- frames: mean=1.608565 bits/token, max frame_index=449 (1.744491), min frame_index=338 (1.544109)
- cols: mean=1.600298 bits/token, max col_index=322 (1.698317), min col_index=0 (1.485309)
- rows: mean=0.189499 bits/token, max row_index=193 (1.498755), min row_index=0 (0.000000)

## Recommendations

- `do_not_dispatch_simple_symbol_context_entropy_replacement`: all measured global/single-context/multi-context entropy lower bounds exceed the charged PR85 QMA9 mask segment before model overhead
- `if_pursuing_mask_bytes_next_target_qma9_native_run_grammar_not_generic_entropy`: row/time runs are extremely long, but charged QMA9 is already far below the simple conditional entropy bounds; any win needs native grammar/table overhead reduction or a structurally different run representation
- `use_left_plus_up_as_the_first_lossless_parity_control_if_a_coder_is_built`: left_plus_up is the best measured simple context even though it is not byte-positive versus PR85 QMA9

## Run-Length Signals

- storage_order_flat: runs=1584702, avg_run=74.439737, max_run=229
- visual_row_width_axis_runs: same_adjacent_fraction=0.997117, avg_run=207.026614
- time_axis_runs_per_pixel: same_adjacent_fraction=0.987541, avg_run=70.898060

These are entropy/rate planning estimates only. A replacement coder would still need byte-closed archive parity, runtime output parity, and exact CUDA auth eval before any score claim.
