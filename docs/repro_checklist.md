# reproducibility checklist

## environment

- Python environment bootstrapped
- upstream checkout available under `workspace/upstream/comma_video_compression_challenge`
- ffmpeg available
- local repo on the expected commit/config

## canonical commands

### bootstrap

```bash
just bootstrap
```

### doctor

```bash
just doctor
```

### reproduce exact_current

```bash
just reproduce-exact-current
```

Expected:
- `reports/raw/exact_current-current_workflow-cpu-summary.json`
- score `0.00`

### reproduce robust_current

```bash
just reproduce-robust-current
```

Expected:
- `reports/raw/robust_current-current_workflow-cpu-summary.json`
- current_workflow `2.12`

### rebuild writeup site

```bash
just rebuild-site
```

Expected:
- `reports/graphs/site/index.html`
- refreshed `reports/graphs/dashboard_data.json`
- refreshed comparison media under `reports/graphs/site/media/`

## canonical evidence files

- `reports/raw/exact_current-current_workflow-cpu-summary.json`
- `reports/raw/robust_current-current_workflow-cpu-summary.json`
- `reports/raw/2026-04-06-hardening/robust_current-hardening-current_workflow-cpu-summary.json`
- `reports/raw/2026-04-06-hardening/robust_current-hardening-smoke.json`
- `reports/raw/2026-04-06-hardening/encoded-ffprobe.json`

## invariants

- Track A remains the only explicitly non-rule-faithful lane
- Track B canonical truth lives in `reports/raw/robust_current-current_workflow-cpu-summary.json`
- rule_faithful charges only:
  - `archive.zip`
  - `inflate.sh`
  - `config.env`
  - `analyze_roi.py`
