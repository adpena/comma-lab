# program

**Reading order (refreshed 2026-05-16):** new readers (operators or
collaborators) should read `HANDOFF.md` first (30-minute orientation),
then `SYSTEM_MAP.md` (structural diagram), then this file (mission +
architecture), then `CLAUDE.md` (agent-binding contracts).

The "Architecture" / "Experiment profiles" sections below describe the
2026-04 WILDE / SHIRAZ / GREEN cycle. The current substrate canvas
(43+ substrate trainers) is documented in `SYSTEM_MAP.md` §1 and
`.omx/state/lane_registry.json` (758+ lanes). This file is historical
architecture context, not the live frontier ledger. Read `reports/latest.md`
first, then `HANDOFF.md` §2 for the latest durable orientation snapshot.

Nomenclature: `tac` means Task-Aware Compression, the reusable library and
algorithmic engine. A codec is a concrete encoder/decoder or wire format inside
that broader compression stack. `comma_lab` owns lab operations, state
projection, custody, and reporting. See `README.md` and
`docs/terminology_and_boundaries.md` for the canonical public wording.

---

Mission: minimize the official challenge score on a pinned upstream snapshot
using task-aware compression against the frozen scorers.

## Historical 2026-04 Renderer Context

The WILDE / SHIRAZ / GREEN renderer notes below are retained for
reproducibility and paper history. They are not the live substrate canvas; for
current work, read `reports/latest.md`, `SYSTEM_MAP.md`, and the active
`.omx/research/*_directive_*` files.

### Historical primary objectives

1. Train the asymmetric warp renderer (WILDE/SHIRAZ/GREEN profiles) to minimize the combined scoring formula.
2. Compress the trained renderer + masks + poses into a submission archive that achieves the lowest possible score.
3. Maintain contest compliance: no scorers loaded at inflate time, single forward pass, under 30 minutes on T4.
4. Collect clean evidence for the writeup track from day one.

### Historical architecture

The renderer is a CLADE-conditioned U-Net (`AsymmetricPairGenerator` in `src/tac/renderer.py`):
- Frame2 rendered directly from segmentation mask via spatially-adaptive normalization.
- Frame1 derived by warping frame2 with learned optical flow + gated residual correction.
- Trained against frozen SegNet and PoseNet scorers with Fridrich inverse steganalysis losses.
- Quantized to int4+LZMA2 or FP4 for archive compression.

### Historical experiment profiles

| Profile | Philosophy | Status |
|---------|-----------|--------|
| WILDE | Empirical 5-phase freeze/unfreeze | Training complete, proxy 0.407 |
| SHIRAZ | PCGrad + focal STE (principled) | A/B test against WILDE |
| GREEN | WILDE + radial zoom warp | Iteration 2, pending |

## Mutation frontier

The agent may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `mojo/**`
- `jax/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

The agent may **not** edit without explicit human approval:

- the pinned upstream snapshot
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `start.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Evidence rules

- Never claim an improvement without a measured score.
- Prefer the official evaluator over proxies.
- Use proxy evaluation only to rank cheap local follow-up candidates before
  promotion. Proxy/advisory/local-substrate rows are never rank/kill or
  promotion authority by themselves.
- Record config, command, artifact size, and score breakdown for each promoted run.
- Label every score with its exact evidence axis: `[contest-CPU]`,
  `[contest-CUDA]`, `[macOS-CPU advisory]`, `[macOS-MLX research-signal]`,
  diagnostic/proxy, or historical unlimited-compute context. Calibrated MLX
  rows may guide spend triage only with an attached calibration manifest and
  exact-eval follow-through. Never promote a proxy, advisory, or MLX research
  axis into a public leaderboard claim.

## Evidence Axes And Historical Lanes

- **Contest auth eval**: `archive.zip` + `inflate.sh` evaluated by the pinned
  upstream scorer, with `[contest-CPU]` and `[contest-CUDA]` kept separate.
- **Diagnostic/proxy**: local, MPS, macOS CPU advisory, smoke, and component
  probes. These guide work but do not rank or kill submissions.
- **Historical unlimited-compute**: TTO or other compress-time-only studies.
  These are paper/methodology context unless converted into byte-closed
  archives and exact auth-eval artifacts.

Never conflate these axes.

## Pipeline

The canonical pipeline (`experiments/pipeline.py`) runs:

1. Mask extraction (SegNet on GT video)
2. FP4/int4 export
3. Adaptive pose TTO (convergence-driven)
4. QAT fine-tuning (quality-monitored)
5. Fridrich steganalytic refinement
6. Weight compression
7. Archive packaging
8. Auth evaluation

Each step is idempotent. The pipeline iterates until convergence.

## Operating loop

At each cycle:

1. propose at most 3 experiments
2. estimate expected payoff and cost
3. run smoke checks
4. run proxy evals
5. promote only the best candidate(s) to full eval
6. summarize what changed and what the evidence says
7. update the next experiment queue

## Reporting standards

Every promoted run should record:

- upstream snapshot hash
- submission track
- packaging mode
- archive size
- measured score (labeled by lane)
- segnet distortion
- posenet distortion
- rate
- runtime notes
- exact commands or config diff

## Style

Be direct.
Prefer small edits over sprawling rewrites.
Prefer reversible experiments.
Prefer measured evidence over narratives.
