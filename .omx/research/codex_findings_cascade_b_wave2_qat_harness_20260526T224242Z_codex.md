<!-- SPDX-License-Identifier: MIT -->

# Codex Findings — Cascade B Wave 2 QAT Harness

**UTC**: 2026-05-26T22:42:42Z
**Scope**: `tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py`
**Axis**: `[macOS-MLX research-signal]`
**Authority**: false-authority only; no score claim, promotion, rank/kill, or exact-eval dispatch readiness.

## What Changed

Codex hardened the Cascade B CATALYST sister wave 2 harness so it is repo-canonical and locally executable:

- replaced ad hoc repo path insertion with the standard `tools.tool_bootstrap` import path;
- kept `upstream/` importability for the real SegNet fixture;
- fixed ruff import/format issues;
- verified the CLI help path imports cleanly;
- ran a tiny MLX-local end-to-end smoke over 2 frames, 1 Path-A epoch, and 1 QAT epoch.

The harness remains a local research-signal actuator. It does not create contest score authority.

## Smoke Receipt

Command:

```bash
.venv/bin/python tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py \
  --n-frames 2 \
  --path-a-n-epochs 1 \
  --qat-n-epochs 1 \
  --batch-size 1 \
  --output-dir .omx/research/cascade_b_catalyst_wave2_qat_codex_smoke_20260526
```

Output artifact:

`./.omx/research/cascade_b_catalyst_wave2_qat_codex_smoke_20260526/sweep_results.json`

Key smoke values:

- baseline KL: `6.486544895172119`
- Path A eval KL: `6.375903701782226`
- CATALYST QAT eval KL: `6.3732646942138675`
- CATALYST minus Path A KL: `-0.0026390075683586645`
- sidecar delta vs Path A: `0.0`
- wall clock: `2.419522285461426` seconds
- Catalog #307 smoke verdict: `DEFER_PENDING_QAT_STABILIZATION`

The `DEFER_PENDING_QAT_STABILIZATION` verdict is expected for this tiny 2-frame smoke because the harness's divergence threshold is calibrated for production-scale KL, not a 1-epoch import smoke. The important local receipt is that decode, real SegNet teacher cache construction, Path A training, fake-quant QAT fine-tune, and final JSON emission all execute on the local MLX path.

## Next Operator-Routable Step

Run the full local production-scale harness when the machine can spend roughly 30-35 minutes:

```bash
.venv/bin/python tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py
```

If the production result is paradigm-validating or partially confirming, append a canonical equation anchor for `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1`. If it falsifies, preserve the 6th-order mechanism as implementation-level falsification and route the 7th-order stabilizer lane instead of killing the Cascade B CATALYST paradigm.
