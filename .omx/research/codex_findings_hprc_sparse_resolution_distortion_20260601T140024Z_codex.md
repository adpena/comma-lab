# Codex Findings: HPRC Sparse Sidecar + Resolution Probe

UTC: 2026-06-01T14:00:24Z
Agent: codex
Axis: `[macOS-MLX research-signal]` advisory only; no CPU/CUDA auth dispatch.

## Verdict

HPRC now has a real sparse protected-residual rate lever, but the current
compact receiver is still not a score candidate. The binding failure moved from
"dense protected sidecar is too large" to "PoseNet geometry is not represented
well enough before or after rate collapse."

The most useful signal is under our noses: the pre-collapse 96x128 full600
trained archive is already terrible on PoseNet. Rate collapse worsens it, but
does not create the failure.

## Evidence

Dense protected sidecar full600:

- Root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_protected_pathway_32_128_600_20260601T130707Z`
- Best archive after collapse: 3,969,285 bytes.
- Rate term: 2.6429839547.
- Gate: blocked before replay.

Sparse 5% protected sidecar full600:

- Root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_sparse_protected_pathway_32_128_600_20260601T132454Z`
- Best archive after collapse: 526,428 bytes.
- Rate term: 0.3505267970.
- Gate: blocked before replay.

Sparse 1% protected sidecar full600:

- Root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_sparse01_protected_pathway_600_20260601T132822Z`
- Best archive after collapse: 234,693 bytes.
- Rate term: 0.1562724353.
- Near miss on rate; still above replay gate.

Sparse 0.8% protected sidecar full600:

- Root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_sparse008_protected_pathway_600_20260601T133014Z`
- Best archive after collapse: 216,046 bytes.
- Rate term: 0.1438561634.
- Receiver proof: succeeded.
- Singleton MLX prefilter score: 23.4645710152.
- Singleton components: PoseNet 28.2353152939, SegNet 0.0651734754.
- Gate: blocked; no local CPU replay or exact auth.

Pre-collapse sparse 0.8% full600 trained archive:

- Archive: 1,244,489 bytes.
- Singleton MLX prefilter score: 17.8559241633.
- Singleton components: PoseNet 15.3454092444, SegNet 0.0463961114.
- This proves the main distortion failure is upstream of the packer.

Importance-aware residual collapse on the same 0.8% trained packet:

| coarsen fraction | archive bytes | rate term | MLX score |
| --- | ---: | ---: | ---: |
| 0.25 | 844,662 | 0.5624257551 | not replayed |
| 0.50 | 638,458 | 0.4251229755 | not replayed |
| 0.75 | 433,025 | 0.2883335732 | not replayed |
| 0.90 | 309,915 | 0.2063596775 | not replayed |
| 0.97 | 242,282 | 0.1613256389 | 23.2207537005 |
| 1.00 | 216,046 | 0.1438561634 | 23.4645710152 |

The q=0.97 point preserves far more residual energy than blind qd10 but still
misses the rate gate and barely improves the MLX score. The present importance
surface is not enough by itself; we need a cheaper geometry/pose representation
instead of keeping raw protected residual cells.

Resolution probe:

| probe | target | pairs | archive bytes | MLX score | PoseNet | SegNet |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| low-res full600 source | 96x128 | 600 | 1,244,489 | 17.8559241633 | 15.3454092444 | 0.0463961114 |
| high-res probe | 192x256 | 32 | 155,566 | 5.8055022395 | 1.3441518167 | 0.0203564962 |
| high-res probe | 192x256 | 128 | 364,924 | 37.4857021928 | 107.4660956459 | 0.0446069243 |

Higher target resolution helps on the tiny slice but fails to scale across
time. That makes the next HPRC design requirement explicit: temporal/predictive
state and protected pose geometry must be native to the representation, not
posthoc residual sidecars over a low-capacity receiver.

## Code Landed In This Slice

The sparse sidecar path keeps the dense P18/P19 protection surface in the
training loss, but emits only a deterministic threshold/top-fraction mask into
the high-res residual sidecar:

- `tools/run_hprc_compact_receiver_training.py`
- `tools/build_hprc_compact_receiver_training_queue.py`
- `src/tac/substrates/hprc/training_adapter.py`
- Tests:
  - `src/tac/substrates/hprc/tests/test_training_adapter.py`
  - `tests/test_build_hprc_compact_receiver_training_queue.py`

The top-fraction mask breaks ties with a deterministic flat-index hash so
binary full-video P18/P19 surfaces do not collapse into the earliest frames.

## Next Action

Do not dispatch these HPRC packets to CPU/CUDA auth. The correct next campaign
is:

1. Add an explicit protected pose/geometry pathway with a compressed grammar,
   not dense RGB residual cells.
2. Add temporal predictive state to HPRC so the 192x256 pose improvement can
   scale beyond a small prefix.
3. Use full-video P18/P19 surfaces to allocate that pathway, then run the same
   archive-rate gate, receiver proof, MLX prefilter, CPU replay, CPU auth, and
   CUDA auth ladder.

