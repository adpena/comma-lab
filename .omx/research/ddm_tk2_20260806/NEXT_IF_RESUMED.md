# NEXT_IF_RESUMED

1. If the scorer slot is free, run C0, C1, then C2 with the D1 fire-order commands in `RECEIPT.md`.
2. Preserve `--chunk-size 120 --resume`; checkpoints are under `stage_checkpoints/<candidate>/`.
3. Do not fire C3 until a real one-hot TR1 adapter/checkpoint shape is declared. A zero/identity placeholder is not a C3 measurement.
4. After any n600 D1 run, update `RECEIPT.md` with the measured per-class/total d_seg and keep `score_claim=false` unless an exact archive is evaluated by `upstream/evaluate.py`.

Current smoke output: `/Users/adpena/Projects/pact/.omx/research/ddm_tk2_20260806/harness_smoke.json`
