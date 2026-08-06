# ddm_hp1 checkpoints

## Inputs

- Charter: `.omx/tmp/codex_runs/hp1_prompt.md`.
- Common contract: `.omx/tmp/codex_runs/_common_contract.md`.
- Source archive: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`.
- Source archive sha256: `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.

## Durable Outputs

- `.omx/research/ddm_hp1_20260806/hp1_results.json`
  - bytes: 10,446
  - sha256: `c4760f8cea0f99753f622736520ae3134e24c0730783d3a8fe9d0afad39f838e`
- `/Volumes/VertigoDataTier/pact/ddm_hp1_20260806/hp1_learned_prior.hp1`
  - bytes: 456,166
  - sha256: `575d7acafd562e75ead009bcb81634a0681859f93e5e1510aebc537c0fc641fc`

## Commands Run

```sh
.venv/bin/python -m py_compile experiments/ddm_hp1_learned_ar_prior_race.py experiments/test_ddm_hp1_learned_ar_prior_race.py
.venv/bin/python experiments/ddm_hp1_learned_ar_prior_race.py --self-test
.venv/bin/python -m pytest experiments/test_ddm_hp1_learned_ar_prior_race.py -q
.venv/bin/python experiments/ddm_hp1_learned_ar_prior_race.py --archive /Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes --expected-sha256 b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06 --receipt-dir .omx/research/ddm_hp1_20260806 --ssd-dir /Volumes/VertigoDataTier/pact/ddm_hp1_20260806
```

## Verification

- Focused tests: `4 passed`.
- HP1 self-test: `{"schema": "ddm_hp1_learned_ar_prior_race.v1", "self_test": "ok"}`.
- Full run verdict: `FAMILY_NEGATIVE_ON_THIS_STREAM`.
- Decode equality: true.
- Canonical re-encode equality: true.
- Scorer forwards: 0.
- `upstream/evaluate.py`: not run.

## Boundary

This checkpoint is a byte-only learned-prior race receipt, not a score row and
not a receiver integration. The own-vehicle frontier remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
