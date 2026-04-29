# submission packet

## notebook surface

- interactive lab notebook: `reports/graphs/lab_notebook.md`
- methodology: `docs/lab_methodology.md`
- glossary: `reports/graphs/glossary.md`

## current operating point (Era 2 — neural renderer)

- Best contest-CUDA score: **`1.05`** (Lane G v3)
- Modal T4 reproduction: **`1.04`** (within 0.01 noise of Vast.ai)
- Archive: `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` (694,074 bytes)
- Component breakdown: PoseNet 0.0034, SegNet 0.0040, rate 0.0185
- Recipe: dilated-h64 renderer + KL distill weight=0.002 + pose TTO retry on Lane A anchor
- Promoted evidence: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`

## historical operating point (Era 1 — codec + post-filter)

- Best honest Track B **current_workflow** score: **`1.73`**
- Bytes: `864,167`
- Promoted evidence root: `reports/raw/2026-04-09-long1000-h64-authoritative`

## evidence

- contest-CUDA inflate.sh → upstream/evaluate.py against the EXACT submission archive bytes
- Modal T4 independent reproduction within 0.01 of Vast.ai
- canonical local E2E smoke gate passed (10 stages, 0.02s)
- 78 STRICT preflight checks gate every measurement against catastrophic-failure classes
- current_workflow vs rule_faithful separation explicit (Era 1 only)

## path summary (from baseline to current floor)

Era 1 (codec era):

- x265 honest floor reached `3.25`
- repaired AV1 path reached `2.20`
- one-axis AV1 tuning reached `2.18` then `2.12`
- encoder-side `sharpness=1` reached `2.08`
- a tiny learned int8 post-filter reached `2.05`
- longer-horizon QAT+EMA training improved that to `1.99`
- the wider h32 long-500 QAT+EMA branch reached `1.95`
- extending the h16 branch to 1000 epochs established `1.92`
- extending the h32 branch to 1000 epochs established `1.85`
- a bounded ensemble of the `1.85` floor and the best Monte Carlo refinement established `1.84`
- scaling the same long-horizon QAT+EMA recipe to `h64` established the Era 1 floor at `1.73`

Era 2 (renderer era):

- abandoned the codec entirely; trained a small neural renderer (dilated-h64) directly against scorer gradients
- discovered MPS vs CUDA drift on PoseNet at 23x; declared MPS scores `[advisory only]` going forward
- first reproducible-from-saved-artifacts contest-CUDA: `0.90` baseline (2026-04-25)
- Lane A pose TTO from baseline poses: `1.15`
- Lane G v3 KL distill weight=0.002 + pose TTO retry: **`1.05`**

## active follow-on (Selfcomp paradigm)

The Selfcomp 0.38 entry uses paradigms we have not used. Reverse-engineered from PR #56. Eight Modal lanes are live to validate each shift in isolation and stack:

- MM (grayscale-LUT mask)
- SA (94K-param SegMap clone)
- SC++ (SA + KL distill T=2.0)
- SO (SC++ + Hessian-aware block-FP)
- in parallel: q_faithful_v3 (Quantizr 1:1 replica), sz_phase2_v2 (dilated moonshot), mae_v_v2, lane_w_v2

We will not promote any score below `1.05` to this packet until it is contest-CUDA verified on the EXACT submission archive bytes.
