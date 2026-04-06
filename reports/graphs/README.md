# research graph artifacts

These artifacts are for tracking the experiment trajectory and hypothesis network over time.

## files

- `score_timeline.json`
  - ordered time series of measured runs
  - includes score, bytes, run id, config, and notes
- `score_timeline.mmd`
  - Mermaid graph of the measured run sequence
- `experiment_graph.json`
  - bipartite-ish graph linking config features to measured runs

## use

- timeline views for writeup / animation
- quick inspection of which features co-occur with wins and regressions
- future candidate-prioritization tooling

## note

This is a research visualization aid, not an authority source for promotion.
The authoritative path remains the local CPU scorer-backed measured results.
