# Live best SegNet semantic bridge: b7106 CPU frontier

Schema-bearing artifacts in this directory were generated from the live
`fp11_source_brotli_recode_b7106c9bdbb8` submission closure, not from a proxy
archive. The queue rows are advisory-only and have no score, promotion, budget,
or exact-dispatch authority.

Commands:

```bash
PACT_PYTHON_BIN=.venv/bin/python \
  .omx/research/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/results/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/per_archive/current_contest_cpu_frontier/fp11_source_brotli_recode_v1/fp11_source_brotli_recode/exact_eval_handoff/submission_closure/submission/fp11_source_brotli_recode_b7106c9bdbb8/inflate.sh \
  .omx/research/live_best_bridge_b7106_20260531Tlocal/extracted \
  .omx/research/live_best_bridge_b7106_20260531Tlocal/inflated \
  upstream/public_test_video_names.txt

.venv/bin/python tools/build_segnet_semantic_bridge.py \
  --inflated-dir .omx/research/live_best_bridge_b7106_20260531Tlocal/inflated \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --json-out .omx/research/live_best_bridge_b7106_20260531Tlocal/segnet_semantic_bridge_b7106_live_cpu_full.json \
  --surface-out .omx/research/live_best_bridge_b7106_20260531Tlocal/segnet_semantic_bridge_b7106_live_cpu_full.semantic_surfaces.npz \
  --candidate-id fp11_source_brotli_recode_b7106c9bdbb8_live_cpu_best \
  --generalization-mode mixed \
  --device cpu \
  --batch-size 4 \
  --figure-out .omx/research/live_best_bridge_b7106_20260531Tlocal/segnet_semantic_bridge_b7106_live_cpu_full.png

.venv/bin/python tools/build_repair_cascade_mlx_probe_queue.py \
  --source-payload .omx/research/live_best_bridge_b7106_20260531Tlocal/segnet_semantic_bridge_b7106_live_cpu_full.json \
  --probe-queue-out .omx/research/live_best_bridge_b7106_20260531Tlocal/queue/repair_cascade_mlx_probe_queue_from_live_bridge.json \
  --results-root .omx/research/live_best_bridge_b7106_20260531Tlocal/queue/results \
  --queue-id live_b7106_segnet_bridge_repair_probe_queue

.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/live_best_bridge_b7106_20260531Tlocal/queue/repair_cascade_mlx_probe_queue_from_live_bridge.json \
  init

.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/live_best_bridge_b7106_20260531Tlocal/queue/repair_cascade_mlx_probe_queue_from_live_bridge.json \
  run-loop --execute --max-steps 20 --max-parallel 1 --max-idle-cycles 2
```

Large local custody artifacts are intentionally left outside the committed
slice: `inflated/0.raw` is rebuildable from the live closure, and
`segnet_semantic_bridge_b7106_live_cpu_full.semantic_surfaces.npz` is recorded by
path and SHA-256 inside `segnet_semantic_bridge_b7106_live_cpu_full.json`.
