# ddm_dv1 — The device axis is a SCORE lever. Decompose it, and measure the half that is free.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-gated · `[macOS-CPU advisory]` · `score_claim=false`

## THE FINDING THIS ARM ACTS ON

PR130 chose CUDA. `inflate.py:663-669` (READ-ONLY intake) hard-raises:

```python
if not torch.cuda.is_available():
    raise RuntimeError("semantic_pose_landslide requires the official GPU rail")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda")
```

`upstream/README.md:114` selects the runner by whether the inflation script needs a GPU.
`upstream/.github/workflows/eval.yml:20-27` binds `EVAL_DEVICE` to that runner. **So the submitter
picks the scoring axis by construction, and PR130 picked CUDA.** Its 0.172141297491896447 is a
contest-CUDA row. Nobody has bought the CPU axis on those exact bytes.

**Why this is a score lever, not a preference.** The bytes are identical either way, so the rate term
(0.127214) does not move. But d_seg and d_pose do move, for TWO independent reasons that this arm
must not conflate:

1. **GT decode path** — DALI/nvdec on the T4 rail versus PyAV on CPU. Different ground truth.
2. **Scorer numerics** — the SegNet/PoseNet forward in different kernels.

Same-object evidence already in hand, from OUR reproduction of PR130's semantic leg:
DALI-GT d_seg **0.0002857038709852431** versus AV-GT **0.0002764044867621528**. The CPU-path GT is
**3.25% lower**. Small on its own (ΔS ≈ −0.0009) but same-object and the right sign.

**The honest bound.** PR130's pose plus seg terms total **0.0153 S**. That is the ceiling on what any
axis change can win, the direction is UNKNOWN, and it can be negative. **Do NOT quote PR102's 0.033
CUDA−CPU gap** — different vehicle, whole-score, not transferable (m88/m96 discipline).

## WHY NOW

`ddm_pk2` (cfddfc503a) just closed the pose-carrier REPRESENTATION axis at INSTANCE scope: 135 real
candidates, 49 scored at n120, unchanged CPR1 best on every row. The coder axis on that section was
already shut (+4 B worse). So the pose section cannot be re-encoded smaller. The device axis moves
d_pose and d_seg **without touching the carrier at all.** It is now the cheapest untried thing
pointing at that 0.0153 S.

## THE DECOMPOSITION — this is the design, do not skip to the composite

We have no CUDA on this host. A raw local-CPU-versus-published-CUDA comparison is CONFOUNDED (two
causes at once, different hardware). So split it:

**LEG A — GT decode path, $0, measurable TODAY, fully local.** Same host, same weights, same code.
Score the reproduced archive twice: once against DALI-lineage GT, once against AV/PyAV-lineage GT.
Report Δd_seg AND Δd_pose. We have the seg half already (the two numbers above); **the pose half is
the gap** — nobody has measured what the GT decoder does to d_pose, and the PR102 prior says pose is
where axis gaps live. This is the arm's primary deliverable.

**LEG B — CPU-capable inflate, build + prove, no score claim.** Relax the CUDA hard-require (legal:
off-the-shelf grant, this is our archive now — but do NOT edit the intake clone; work on a copy).
Prove the CPU decode reproduces the CUDA decode's tokens/model bit-identically where determinism
allows, and record where it cannot. Note PR130 disables TF32 on both matmul and cudnn — they hit the
numerics question and pinned it; say what that implies for CPU/CUDA agreement.

**LEG C — price the remaining half, do not run it.** Name exactly what a Modal dispatch would buy
(contest-CPU Linux x86_64 on the same bytes), its cost, and its blocking prerequisites. Do NOT
dispatch. Modal is SINGLE-FLIGHT under a ≤$20 envelope and needs operator GO.

## COUPLING — read this before recommending CPU

Choosing CPU puts the HPAC decode's dominant neural forward on 4 cores. Structure MEASURED at source
(`codec_hpac_integer.py:96-124`): the frame loop is SERIAL (frame *f*'s context is frame *f−1*'s
output), the group loop is SERIAL (masked-context refinement), positions within a group are ALREADY
vectorized. So the cost is a per-group neural forward, 600 frames deep, and on CPU that is the whole
wall-clock question `ddm_dt1` owns. If Leg A says CPU wins on score, the recommendation is
CPU **conditional on** dt1's delta — say so, do not recommend in isolation.

Determinism cuts the same way: CPU decode is deterministic by default; CUDA needs pinning. Our
deterministic-reproducibility non-negotiable mildly favours CPU independent of score. Note it; do
not let it decide the score question.

## OPTIMAL FORM

Reference form: a two-arm same-object axis decomposition with per-axis d_seg AND d_pose, on the
identical archive bytes, with the confound named and separated. Declared reductions: SCOPE only —
n≥120 seeded STRATIFIED-RANDOM for the sweep (never a prefix; pose prefix bias is the worst-known,
2.5–4.2× harder, so a prefix pose result is exactly the false-negative shape, m96/`ddm_na2`); the
winner re-measured at full n600 before any composed row. MECHANISM reductions are TOY-BRACKET and
cannot produce a family verdict — in particular, comparing a local macOS-CPU number against a
published contest-CUDA number is a DIFFERENT-OBJECT comparison, not an axis measurement.

Provenance pins (verify each; a pin that does not reproduce is a STOP):
- archive sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B, at
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
- reproduction record commit `12031094d9`; pk2 closure `cfddfc503a`; receipt binding `c21d39b48d`
- the DALI/AV seg numbers above live in `.omx/research/ddm_pr130_reproduce_20260809/` — re-derive
  them, do not inherit them
- `upstream/` is IMMUTABLE. The intake clone at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY: never edit, never
  `git add` inside. Copy out to work.

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_dv1_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Every number carries its axis. A macOS run is `[macOS-CPU advisory]`, never `[contest-CPU]`
  (that requires Linux x86_64). `score_claim=false` throughout. NO Modal dispatch.

## DELIVERABLE

The Leg-A table — Δd_seg and Δd_pose between the two GT decode paths on identical bytes, with the
seg half re-derived and the pose half measured for the first time. Then Leg B's build and its
bit-identity result. Then Leg C's priced, unfired plan. State the axis verdict at the scope the
evidence supports, and say plainly which legs you did not run.

If the GT-path effect is negligible on pose, that is a real finding: it means the axis gap (if any)
is scorer numerics, and only a paired Linux/T4 dispatch can settle it. Either way, say so.
