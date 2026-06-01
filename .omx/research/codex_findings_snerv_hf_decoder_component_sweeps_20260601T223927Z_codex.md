# Codex Findings: SNeRV HF Decoder Component Sweeps

UTC: 2026-06-01T22:39:27Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Lane: `lane_snerv_score_weighted_hf_decoder_fit_smoke_20260601`

## What Landed

Added an explicit saliency-component selector for the score-weighted SNeRV
linear HF decoder fit:

- `src/tac/substrates/snerv_inverse_steg_carrier/advisory.py`
  - new `hf_decoder_saliency_component` result field;
  - new archive metadata field;
  - new `_combine_hf_decoder_saliency(...)` helper;
  - supported components: `combined`, `seg`, `pose`.
- `tools/run_snerv_inverse_steg_advisory.py`
  - new `--hf-decoder-saliency-component combined|seg|pose` CLI arg;
  - CLI output now reports the component.
- `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py`
  - focused test that selector output is explicit and rejects bad components.

No receiver grammar change was made. The component value is now visible in the
advisory JSON and archive metadata, so later rows are not command-line folklore.

## Verification

```text
.venv/bin/ruff check src/tac/substrates/snerv_inverse_steg_carrier/advisory.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py tools/run_snerv_inverse_steg_advisory.py
All checks passed!

.venv/bin/python -m pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_carrier.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_step_packet.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_advisory_archive_replay.py src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py src/tac/tests/test_snerv_rate_adjudication.py src/tac/tests/test_snerv_step_map_coder.py -q
42 passed in 1.09s

.venv/bin/python tools/lane_maturity.py validate
OK - 1606 lane(s) validated cleanly.
```

## Component Sweep

Fixed config:

- `n_pairs=1`
- `levels=4`
- `bits_per_coeff=5.0`
- `step_map_coder_mode=waterfill`
- `step_map_waterfill_bits_per_coeff=6.0`
- `hf_decoder_fit_mode=score_weighted`
- `hf_decoder_saliency_gain=0.25`

Combined sweep artifact:

- Path: `.omx/research/snerv_hf_decoder_saliency_component_sweep_20260601T223741Z.json`
- SHA-256: `bf46cab9751148765cc1eabffe479298c92cdaa21c41c99fedecde4afe5b3a2f`
- Best score row: `least_squares_baseline_existing`
- Best pose row: `least_squares_baseline_existing`
- Any component improves score versus baseline: `false`
- Any component improves pose versus baseline: `false`

Adjudication:

- Path: `.omx/research/snerv_hf_decoder_saliency_component_sweep_adjudication_20260601T223741Z.json`
- SHA-256: `a3d251bdbf55eb9bb2d69860b07b1f5e5d832bc2ffe8dd385a142c9e0788d962`
- Classification counts: `{"rate_below_frontier_pose_or_seg_destroyed": 4}`
- Ready for exact eval dispatch: `false`
- Promotion eligible: `false`
- Frontier score claim: `false`

Important rows:

| Row | Archive bytes | d_seg_linf | d_pose_linf | score_linf |
|---|---:|---:|---:|---:|
| least-squares baseline | `33754` | `0.02264404296875` | `2.1390697956085205` | `6.911887587116307` |
| combined, gain 0.25 | `33868` | `0.02226766012609005` | `5.7368927001953125` | `9.823545139190069` |
| seg-only, gain 0.25 | `33863` | `0.02226766012609005` | `5.7368927001953125` | `9.823541809895302` |
| pose-only, gain 0.25 | `33864` | `0.01981608010828495` | `3.579928159713745` | `7.987406744878181` |

Seg-only and combined are effectively identical, so the combined field is being
dominated by SegNet flip-risk. Pose-only is a materially better direction than
SegNet-only, but still loses to the least-squares baseline because PoseNet
damage remains too high.

## Pose-Only Fine Sweep

Pose-only fine sweep artifact:

- Path: `.omx/research/snerv_hf_decoder_pose_saliency_gain_sweep_20260601T223927Z.json`
- SHA-256: `9c0d87416ce3002a290a6f0763b84f779802ccc86a66b8e29611136ee09461fa`
- Best score row: `least_squares_baseline_existing`
- Best pose row: `least_squares_baseline_existing`
- Any pose gain improves score versus baseline: `false`
- Any pose gain improves pose versus baseline: `false`

Adjudication:

- Path: `.omx/research/snerv_hf_decoder_pose_saliency_gain_sweep_adjudication_20260601T223927Z.json`
- SHA-256: `955bc2eed83ef4058e4923e14630e5ce3e27536296a33b7170b0c1b88149b47f`
- Classification counts: `{"rate_below_frontier_pose_or_seg_destroyed": 6}`
- Ready for exact eval dispatch: `false`
- Promotion eligible: `false`
- Frontier score claim: `false`

Pose-only rows:

| Gain | Archive bytes | d_seg_linf | d_pose_linf | score_linf |
|---:|---:|---:|---:|---:|
| `0.01` | `33864` | `0.01947530172765255` | `3.394784927368164` | `7.796557111198232` |
| `0.05` | `33864` | `0.019683837890625` | `3.546154737472534` | `7.945892333638191` |
| `0.1` | `33863` | `0.019775390625` | `3.5670182704925537` | `7.9725390285426805` |
| `0.2` | `33863` | `0.01979573629796505` | `3.577770948410034` | `7.9835687177120365` |
| `0.25` | `33864` | `0.01981608010828495` | `3.579928159713745` | `7.987406744878181` |

The smallest positive gain is best, but it still loses to the least-squares
baseline. The improvement in `d_seg_linf` is not enough to offset PoseNet
damage.

## Verdict

NO-GO for the current closed-form scalar/component-weighted linear HF decoder
fit as a promotion or exact-eval candidate.

The result is not "all score-aware decoder fitting is dead." It is narrower and
useful:

- SegNet-driven HF residual weighting is the wrong direction for PoseNet.
- Pose-only weighting is less wrong and does improve SegNet, but still worsens
  PoseNet and advisory score versus baseline even at `gain=0.01`.
- More scalar/component tuning is low-EV unless a new objective changes the
  measured frame-level detector response.

## Next Code Move

Stop spending turns on closed-form DWT residual weighting. The next useful
implementation should move one level closer to the real objective:

1. reconstructed-frame/scorer-loop decoder-weight training with replay in the
   loop;
2. learned or nonlinear HF decoder QAT, keeping the receiver grammar tiny and
   byte-closed;
3. PoseNet-constrained multi-objective training where SegNet improvement is
   accepted only if `d_pose_linf` stays at or below the least-squares baseline.

Until that lands, keep the least-squares waterfill row as the SNeRV local
control and keep all these rows advisory-only.
