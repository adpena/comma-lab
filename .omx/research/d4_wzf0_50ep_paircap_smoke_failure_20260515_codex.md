# D4 WZF0 50ep Pair-Capped Smoke Failure - 2026-05-15

## Result

- Lane: `lane_d4_wyner_ziv_frame_0_substrate_20260514`
- Dispatch label: `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T054131Z__smoke__50ep`
- Modal call id: `fc-01KRN2FFCZAYKJ85Q9W7GQV0QW`
- Runtime: Modal T4, 50 epochs, `max_pairs=200`
- Elapsed: `1020.943394455` seconds
- Estimated cost: `$0.16732127853568055`
- Terminal claim: `failed_modal_training_rc_1`
- Score claim: `false`
- Promotion eligible: `false`

## Custody

- Harvest directory:
  `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T054131Z__smoke__50ep_modal/`
- Archive zip:
  `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260515T054131Z__smoke__50ep_modal/lane_substrate_d4_wyner_ziv_frame_0_results/output/archive.zip`
- Archive zip SHA-256:
  `e5c9a9fcd0f081f74fd6ff2104cc46dd29704d15c39c0cc8268e575d869550db`
- `0.bin` SHA-256:
  `9429f2d8c9e4644091723ab00231f3ecb848ce33dd6b963adf526c539c086b53`
- `best.pt` SHA-256:
  `4dd0aa58058b81c894aeb120e2430a1fc6484250a5a29a8a7d8c94b20c805938`

## Classification

This is an engineering contract failure, not a D4 score result and not a
model-negative. The smoke recipe capped training/decode at 200 pairs to fit
the timing budget. D4 then attempted full contest auth eval. The runtime
correctly inflated 400 frames, but the evaluator expected the full 1200-frame
raw stream for the contest video:

```text
WRONG-SIZE .raw file(s): 0.raw=1220803200B (expected 3662409600B)
```

No component distances or contest score were produced.

## Hardening Landed

- `experiments/train_substrate_d4_wyner_ziv_frame_0.py` now fails closed when
  `--max-pairs < 600` is used without `--skip-auth-eval`.
- `scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh` now automatically
  adds `--skip-auth-eval` for capped pair smokes.
- The D4 recipe now explicitly records that capped smokes are timing/training
  artifacts, not first-anchor score attempts.
- Focused tests cover the trainer guard and remote-driver skip path.

## Next Valid D4 Actions

1. Timing smoke: keep `max_pairs=200`; expect a training artifact only.
2. First score anchor: run without a pair cap, or with `max_pairs=600`, so
   inflate emits the full 1200-frame raw stream before auth eval.
3. Do not classify the 50ep capped smoke as below-band or above-band; it has no
   score axis.
