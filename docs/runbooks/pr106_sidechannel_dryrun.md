# PR106 Sidechannel Dry-Run

Use the read-only PR106 sidechannel dry-run before turning any PR106
sidechannel builder output into a real dispatch:

```bash
.venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py
```

The default mode validates only local source, argparse, help, score-claim, and
real-mode fail-closed surfaces for:

- `experiments/build_pr106_latent_sidecar.py`
- `experiments/build_pr106_yshift_sidechannel.py`
- `experiments/build_pr106_lrl1_sidechannel.py`
- `experiments/build_pr106_stacked.py`

It does not dispatch, does not require CUDA, does not read provider state, and
reports `score_claim=false`.

For production readiness, pass `--production-readiness` plus the exact PR106
anchor, three sister archives, and four build manifests. Missing manifests or
sister archives fail only in that explicit production mode.
