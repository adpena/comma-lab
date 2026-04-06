# runbook: submission

## preferred local eval path

Use the repo-local CLI helper instead of calling the shell wrappers directly.
It:

- ensures the upstream `.venv` is on `PATH`
- syncs the submission into the upstream checkout by default
- can package `robust_current` before scoring
- can copy the raw scorer report into `reports/raw/`
- prints a JSON summary including `current_workflow` accounting and the local `rule_faithful` accounting estimate when available

## exact_current

### safe local command

```bash
source .venv/bin/activate
comma-lab eval-submission exact_current \
  --device cpu \
  --report-copy reports/raw/exact_current-current_workflow-cpu-report.txt \
  | tee reports/raw/exact_current-current_workflow-cpu-summary.json
```

### notes

- This keeps the raw scorer output inside the repo instead of `/tmp`.
- `rule_faithful` remains non-promotable for this track because inflate depends on repo-side public videos.

## robust_current

### safe local command

```bash
source .venv/bin/activate
comma-lab eval-submission robust_current \
  --package \
  --device cpu \
  --report-copy reports/raw/robust_current-current_workflow-cpu-report.txt \
  | tee reports/raw/robust_current-current_workflow-cpu-summary.json
```

### notes

- `--package` refreshes `archive.zip` first.
- The helper re-syncs the packaged submission into upstream before calling `evaluate.sh`, so the scorer does not read stale bytes.
- `--package` intentionally requires sync; packaging without sync would build bytes that are not the bytes under test.
- The JSON summary includes the measured `current_workflow` score and a local `rule_faithful` estimate based on the scorer distortions plus the honest byte burden of the installed runtime payload under test.

### package-only refresh

```bash
python3 -m src.comma_lab.cli package-submission robust_current \
  --upstream-root workspace/upstream/comma_video_compression_challenge
```

This uses the requested challenge root for packaging instead of silently falling back to the default workspace clone.

## required report fields

- track
- packaging view
- archive size
- final score
- segnet distortion
- posenet distortion
- rate
- runtime notes
- upstream snapshot
