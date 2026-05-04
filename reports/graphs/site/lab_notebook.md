# lab notebook

Last refreshed: `2026-05-04`

## executive summary (current frontier - PR100)

- exact Tesla T4 A++ frontier: **`0.22826947142244708`** (PR100 HNeRV-LC-v2 adapter replay)
- archive: `178981` bytes, SHA-256 `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- runtime tree SHA-256: `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- component distances: SegNet `0.00067623`, PoseNet `0.00017198`, `600` samples
- evidence boundary: public PR bodies, leaderboard displays, static anatomy, GIFs, and roadmap probes are context unless exact CUDA replay validates the same archive/runtime pair

This notebook is layered. Skim the executive summary for state. Drop into the era-specific sections for the detailed arc.

## notebook map

1. Executive summary
2. Era 1: AV1 codec + tiny CNN post-filter (`4.06 → 1.73`) — historical
3. Era 2: Neural renderer that bypasses the codec (`1.73 -> 1.05`) - historical controls
4. Era 3: public semantic/neural sufficient-statistic frontier (`0.3156 -> 0.2283`)
5. Engineering rigor under the hood
6. Runnable reproduction snippets
7. Methodology, evidence, glossary

## era 1: AV1 + tiny CNN post-filter (historical, score range 4.06 → 1.73)

This was the original work that established the lab's measurement discipline and produced the 1.73 score. The full mathematical investigation (Jacobian SVD, trust radius, CNN residual analysis) is preserved in `final_writeup_draft.md`. Headlines:

- branch: `long500_qat_ema_alpha20_h32` then scaled to h64
- key insight: width scaling buys log-linear score improvement
- math: PoseNet trust radius < 0.0001 pixels, Jacobian effective rank ~1, condition number ~399. CNN strategy = 56.6% of pixels nudged with 90.3% energy in mid-frequency DCT band. Closed-form alternatives are mathematically dead on arrival.

This era is no longer the live frontier but its mechanics still inform the renderer-era loss landscape.

## era 2: neural renderer controls (historical, score range 1.73 -> 1.05)

We abandoned the codec entirely. A small renderer (dilated-h64, 287K params) takes per-pair embeddings and produces frames; AV1 only carries low-resolution masks; PoseNet still runs at full resolution.

Key steps:

- **2026-04-25**: first reproducible-from-saved-artifacts contest-CUDA score = `0.90` (pinned dilated h64 + CRF=50 + matched poses). MPS-vs-CUDA drift discovery (PoseNet 23x worse on MPS) invalidates earlier `2.26` MPS readings.
- **2026-04-27**: Lane A = `1.15` [contest-CUDA] (pose TTO warm-started from baseline poses).
- **2026-04-28**: Lane G v3 = `1.05` [contest-CUDA] (KL distill weight=0.002 + pose TTO retry on Lane A anchor). Improves PoseNet 0.247 → 0.0034 (73x) without bloating the rate term.
- **2026-04-29**: Modal T4 reproduces Lane G v3 within 0.01 (1.04 vs 1.05). Modal becomes canonical for >2h training jobs.

Negative results worth noting:

- Lane M-V2 (radial-zoom rank-1 hypothesis): `1.84`. Pose-pad asymmetry between train/inference confirmed and gated by Check 42.
- Lane GP v3 (Gaussian-process pose fit): `89.67` [Modal-T4-CPU]. Runge phenomenon at degree-10 polynomial; off-manifold hypothesis disproved. Lane GP polynomial is dead unless someone wants to try DCT or B-spline basis.
- Lane UNIWARD v8: `1.14` ≈ Lane A noise. Encoder pipeline is no-op on the bitstream without an SLI1 inflate-time decoder. Council 5/5 KILLED standalone.

## era 3: public semantic/neural sufficient-statistic frontier

Late public submissions moved the active family from local renderer polish to
public-source archive anatomy, exact replay, adapter repair, and deterministic
packet hardening.

- C067/PR67 fixed-slice reproduction established a sub-0.4 exact T4 basin at
  `0.31561703078448233`.
- PR85 semantic-bundle replay established `0.25806611029397786`.
- PR95 stem-permutation repack established `0.23089404465634825`.
- PR99 HNeRV/Muon LC adapter established `0.2297226895103603`.
- PR100 HNeRV-LC-v2 adapter replay is the current exact frontier at
  `0.22826947142244708`.

## next-wave roadmap

The Selfcomp 0.38 #2 entry uses paradigms we have not used. Reverse-engineered from PR #56 inflate.py, five concrete shifts:

1. Grayscale-LUT mask encoding (1ch smooth values + Gaussian softmax LUT) vs our 3ch discrete-class
2. Single-mask-per-pair + 6-DOF affine duality (one mask warps to both frames)
3. Analytical pose via affine_delta (no PoseNet predictor)
4. Block-FP weight self-compression at ~1.017 bpw (vs our FP4 4-8 bpw)
5. 94K-param SegMap (vs our 287K-param ASYM)

Hidden-gem next-wave work includes HPM1/HPAC parity, native action atoms,
HNeRV adapter replay hardening, scorer-gradient atoms, byte self-compression,
and field-policy waterfill. These are research and roadmap signals until a
complete charged archive lands exact CUDA auth eval through
`archive.zip -> inflate.sh -> upstream/evaluate.py`.

## engineering rigor under the hood

This is the differentiating story behind the PR100 frontier:

- Strict preflight checks converted repeated harness bugs into fail-closed gates.
- `eval_roundtrip` is a CLAUDE.md non-negotiable; every training path defaults True. Closed the 2-11x proxy-auth gap on PoseNet.
- MPS vs CUDA drift on PoseNet is 23x. ALL auth eval is on CUDA only; MPS scores are tagged `[MPS-PROXY]` and treated as advisory only.
- The strict-scorer rule: no PoseNet/SegNet weights at inflate time, ever. Detection via `check_no_scorer_load_at_inflate`.
- The mask-resolution disaster: 48x64 vs 384x512 catastrophically scored 53.61 instead of 1.15. Fixed by Check 76 STRICT (anchor mask resolution).
- Vast.ai NVDEC roulette: 85% bad-host rate on some nights. Pre-DALI NVDEC probe (Stage 0.5) + Modal pivot for >2h jobs.

## runnable snippets

### 1. authoritative local scorer (Era 1 / Track B)

```bash
source .venv/bin/activate
comma-lab eval-submission robust_current --device cpu
```

### 2. canonical local auth-eval smoke (Era 2 / historical control)

```bash
.venv/bin/python experiments/canonical_local_auth_eval_smoke.py \
  --lane g_v3_corrected_kl_weight --quiet
```

### 3. exact PR100 artifact references

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter \
  --archive experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter/archive.zip \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/contest_auth_eval.adjudicated.json \
  --contest-final
```

### 4. Modal auth eval (historical path)

```bash
.venv/bin/python experiments/modal_auth_eval.py \
  --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
```

### 4. static site rebuild

```bash
python3 reports/graphs/build_dashboard.py
python3 reports/graphs/build_static_site.py
```

## methodology and evidence

- methodology: `../../docs/lab_methodology.md`
- evidence index: `./evidence_index.md`
- glossary: `./glossary.md`
- current live status: `../../reports/latest.md`

## glossary

See the standalone `glossary.md` for definitions used throughout the notebook and packet.

## appendices

### appendix A — Era 1 promoted floor evidence (1.73)

- `reports/raw/robust_current-current_workflow-cpu-summary.json`
- `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-current_workflow-cpu-report.txt`

### appendix B — Era 2 contest-CUDA evidence

- `experiments/results/lane_g_v3_landed/contest_auth_eval.json` (1.05)
- `experiments/results/lane_a_landed/contest_auth_eval.json` (1.15)
- `experiments/results/modal_auth_eval_9b20bdfca246.json` (1.04 Modal reproduction)

### appendix C — recent negative evidence

- `experiments/results/lane_m_v2_landed/contest_auth_eval.json` (1.84, radial-zoom hypothesis dead)
- `experiments/results/lane_h_crf56/auth_eval/contest_auth_eval.json` (3.20)
- Lane GP v3: 89.67 [Modal-T4-CPU], Runge phenomenon
