# ddm_rw1 Next If Resumed

1. q3x: rerun the DK1 default without the bounded smoke valve once the scorer slot is free:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_q3x_q3_convergence_measurement.py \
  --selection-mode strided --n 8 --stride 20 --offset 0 \
  --out .omx/research/ddm_rw1_20260806/q3x_dk1_cvp_n8_fullblocks.json \
  --realizer dk1-cvp --solver-form project-after \
  --cap-ladder 25,50,100 --eval-every 5 --threads 6 --max-chunk-pairs 8
```

Do not regrade q3x from the n=1/block64 smoke. It only proves the codepath and a negative bounded delta.

2. q3x solve-within: run a same-selection `--solver-form solve-within-null-basis` A/B only after the full-block DK1 project-after run is not cap-bound or after the scorer slot owner releases the lane.

3. FD: expand the integer near-margin smoke to n=8 and `cvp_tap_radius=1` before any campaign launch. The rw1 n=1 smoke produced 1/6 accepted proposals, enough to reopen the instrument, not enough for a family-level verdict.

4. CA1: keep all five unmeasured Class-B sites labelled CAP-BOUND-at-stop until a fresh converged receipt exists. Do not silently inherit their old verdict text.

5. Registry: merge `.omx/research/ddm_rw1_20260806/INSTRUMENT_REGISTRY.jsonl` into vo2 only after the vo2 registry owner declares the merge path. The standalone registry is complete for rw1.
