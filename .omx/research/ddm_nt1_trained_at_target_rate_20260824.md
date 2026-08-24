# ddm_nt1 — rate is already in the objective and mandatory; **pose** is the term nobody differentiates; and the live rate term is an ALLOCATION accounting, not an entropy estimate

`date_utc: 2026-08-24` · `axis: [macOS-CPU scorer-free source + arithmetic]` · `score_claim: false` ·
`frontier_moved: false` · `verdict_scope: INSTANCE:DX2 + tools/train_ddm_cl1_hpac_capacity.py`

## 0. Answer first — including my own retracted headline

**The recall verdict.** The charter's premise — "nobody has trained an object with the rate
constraint in the objective" — is **REFUTED at source**. Rate has been in the live objective the
whole time, and it is *mandatory*:

```python
task_loss = F.cross_entropy(logits, target)                                       # :1320
rate_loss = args.rate_lambda * math.log(2) * variable_weight_bits(model, deployed=False) / pixels
loss = task_loss + rate_loss                                                      # :1322
```
`tools/train_ddm_cl1_hpac_capacity.py:1320-1322`. `--rate-lambda` defaults to 1.0 (`:846`) and
`:964-965` raises `CL1TrainingError` if it is `<= 0.0`. I verified all four legs myself at source.

**`grep -c -i pose` on that trainer returns 0.** Not "few". Zero. Pose — **5.38472% of S, and
6.2647× seg's marginal sensitivity** — is the one term of three that never enters the gradient.

**RETRACTION — my own, prominently, because I published it two hours ago.** I previously reported
that the live trainer was `src/tac/pr130_lift/train_semantic_quantized_resumable.py` and that
`packed_size` (shape-only, never reads a weight) made a rate term structurally impossible on 29.43%
of the archive. **That file is a SISTER trainer, imported by the real one only for EMA helpers. It
did not produce this lineage. WITHDRAWN as a claim about dx2.** MAIN and a sister arm reached the
same wrong file independently, so three readers made one error: we each verified the loss of *a*
trainer whose name matched the object, never *the* trainer. That is the
`measured_object_vs_named_object` genus, and the cure that would have caught all three of us is the
same one that caught it in the end — **trace the lineage from the shipped artifact backwards, never
from a plausible filename forwards.**

**What survives the retraction, and it is the finding worth keeping.** The *lens* was right even
though I pointed it at the wrong file. There are two structurally different kinds of "rate term",
and which one you have decides what a rate gradient can and cannot buy:

| | what it measures | can a gradient move it? | fidelity question |
|---|---|---|---|
| **allocation accounting** (`packed_size`; **and `variable_weight_bits`**) | Σ shape × bit-width | only if bit-width is learnable | is anything *coded* downstream of it? |
| **entropy estimate** (TR1's `token_rate_term`) | predicted coder output | yes | does the surrogate see the axis the coder charges for? |

`variable_weight_bits` (`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/hpac_self_compress.py:84-96`)
is the **first** kind:

```python
bits = F.relu(module.bit_depth)                       # per-OUTPUT-CHANNEL nn.Parameter
if deployed: bits = bits.clamp(max=...).round()       # skipped at deployed=False -> differentiable
layer_bits = (bits * _weights_per_output(module)).sum()
```

It is an **exact accounting of allocated bits**, not a prediction of a coder's output. So MAIN's
question — "is this surrogate anti-correlated like rsf1's ρ = −0.7235?" — **does not apply in the
form asked**, and I should say that rather than answer a different question. rsf1's surrogate was a
marginal histogram *estimating a context coder*, and it could be wrong. This one cannot be "wrong"
about allocation; it can only be **incomplete**, in one specific way: if any entropy coding sits
between allocated bits and shipped bytes, the coder's gain is invisible to the gradient. **That gap
is unmeasured, and measuring it is the cheapest high-value rung available (§3).**

## 1. Recall — scope, and the corrected table

**SEARCH SCOPE.** `grep` over `experiments/`, `src/tac/`, `tools/`, `scripts/` for
`lambda_rate|rate_loss|w_rate|rate_weight|bits_loss|entropy_loss|size_penalty|byte_budget|
target_bytes|rate_term|bpp_loss`; AST scan of 6,609 files declaring `add_argument`; `grep -rlniE`
over `.omx/research/` for rate-in-loss / rate-aware-training / train-at-target-size phrasings; the
four charter-named candidates by name. I did not read every `.omx/research` body and did not run
`graph_memory_recall`. **My scope was adequate to find the mechanisms and inadequate to identify the
live vehicle** — I found rate terms on three trainers and picked the wrong one as live.

| # | rate in the gradient | where | live? |
|---|---|---|---|
| 1 | **`rate_loss`, mandatory positive, per-channel learned bit depths** | `tools/train_ddm_cl1_hpac_capacity.py:1320-1322` | **YES — this is the live vehicle** |
| 2 | `token_rate_term`, 2 modes, on the token field, `w_rate=0.05` | `train_tr1_partition_renderer_mlx.py:5195,5256` | no — TR1 retired |
| 3 | Ballé weight-entropy λ, MLX port | `boundary_math/weight_entropy_penalty_mlx.py` | no — default 0.0, NEVER-FIRED on MLX |
| 4 | `rate_weight` × `target_bytes` hinge — the only *budget-targeting* form in the repo | `train_renderer_fridrich.py:163-164,1466-1468` | no — old renderer lineage |
| 5 | `loss = seg + distill·distill_loss` (+ `rank_penalty`); `packed_size` post-hoc at `:1446` | `train_semantic_quantized_resumable.py:1287` | **no — SISTER trainer, my retracted claim** |

**Prior negatives that do not transfer, restated because they still bind the design.** TR1's rate
surrogate measured ρ = −0.7235 against real bytes (`rsf1`) and was **permutation-blind**: permuting
the pair axis moved it ≤4.8e-07 bits while real bytes moved **+16,062…+18,339 B** (`rg5`). `rg5` also
REFUTED the "backwards gradient" reading — descent lowered real bytes in 4 fields × 4 steps × 3
preconditioners without exception. The transferable lesson is **not** "rate-in-loss fails"; it is
"an aggregate statistic can be blind to the axis the real coder charges for," which is exactly the
allocation-vs-coded gap in §0.

## 2. Arithmetic (recomputed from components per #877; receipt persisted)

Exchange rate **6.658590e-07 S/B** cited from `ddm_tx1_toolbox_crosswalk_20260819.md` §0, not
re-derived.

| term | value | share of S | in the gradient? |
|---|---:|---:|---|
| seg `100·d_seg` | 0.020139000 | 13.58725% | YES (cross-entropy) |
| pose `√(10·d_pose)` | 0.007981228 | **5.38472%** | **NO — absent** |
| rate `25·B/N` | 0.120099648 | 81.02803% | YES (mandatory) |
| **S** | **0.14821987563243377** | matches the pointer exactly | |

`d(pose_term)/d(d_pose) = 5/0.0079812 = 626.47` vs seg's flat `100` ⇒ **6.2647×**. MAIN's shares
reproduce to 5 dp; I confirm rather than restate. **The blind spot is one term, not two — 16×
smaller in share than I wrote this morning, and it is the expensive one.** The measured damage
signature agrees: dg2 93.3%/93.4%, w72 65.3% (`d_pose` ×303,989), ap1 100%.

**The demand.** Fixed distortion: ceiling **137,986 B** (137,986 ⇒ S=0.119999441 PASS; 137,987 ⇒
0.120000107 FAIL), demand **42,382 B**. Zero distortion: ceiling 180,218 B, demand **150 B**.
Residue sums to 180,368 with zero remainder.

## 3. What I would fire, and one lead I must refuse

**REFUSED — the `--lr-bits` / `--bit-eps` lead, and MAIN should not spend a burn on it.** MAIN read
the live run at `--lr-bits 0.01` against an argparse default of `0.1`, and `--bit-eps 1e-6` against
`1e-3`, and called them "10× and 1000× off default." They are not misconfigured. **`0.01` and `1e-6`
are the PREREGISTERED values** — `PREREGISTERED_CONFIG` `:85-86` — and
`_assert_preregistered_config` (`:480-496`) raises `CL1TrainingError` on *any* divergence from them.
The argparse defaults are **inert**: no admissible run can ever use `0.1` or `1e-3`. This is the
`available_field_vs_authoritative_field` genus — the argparse default is the available field, the
preregistration is the authoritative one. An A/B on these knobs is not a cheap race; it requires
amending a receiver-closed preregistration, which is a governance act. `--rate-lambda` is likewise
constrained to `{1.0, 0.5, 0.25}` for profile `cl1` (`:487-492`), so even the λ sweep has exactly
three admissible points.

**The rungs I want, cheapest-decisive first:**

1. **R0 — calibrate the surrogate against shipped bytes. $0, no training, no scorer.** Load the dx2
   checkpoint, compute `estimated_model_bits(model)/8` (`hpac_self_compress.py:133-136` — the
   deployed path, with `round()` and `clamp()` applied), and compare against the shipped
   semantic + HPAC spans (30,856 B and 13,515 B; ar1b's exact archive offsets). **If they agree, the
   rate term is exact and rate is genuinely optimized. If shipped < estimated, the difference is
   coder gain the gradient is blind to, and its size is the ceiling on what more rate pressure can
   ever buy.** Either outcome is a real finding, and it is the precondition for pricing any rate
   move. This is the single measurement I would run first.
2. **R1 — pose into the loss.** The honest unasked term: 5.38% of S, 6.2647× marginal sensitivity,
   0 occurrences in the trainer, and 65–100% of every measured refusal. Nothing about R1 depends on
   R0's outcome, so the two can run in parallel.
3. **R2 — the admissible λ sweep**, `{1.0, 0.5, 0.25}`, only if R0 shows the surrogate is faithful.
   If R0 shows a large coder gap, R2 is measuring pressure on a quantity that does not control the
   shipped bytes, and should not be fired.

**Falsifier for the trained-at-target-rate route.** If a burn reaches ≤137,986 B and distortion rises
so that S ≥ 0.148220, the route is refuted on this vehicle and the campaign needs a different
representation class, not a different weight.

**Prediction, recorded to be scored.** R0 will show shipped **below** estimated — because the shipped
streams are entropy-coded (CABAC/RC64) downstream of a bit-packed store — and therefore that the rate
gradient has been optimizing an **upper bound**, not the shipped quantity. If that is right, more
rate pressure is the wrong lever and pose (R1) is the whole game.

**Resumability: nothing to build.** `CHECKPOINT_SCHEMA = "ddm_cl1_hpac_capacity_checkpoint.v2"`, and
the lineage audit reports optimizer + scheduler restore, step-encoded filenames, EMA shadow as the
deployed weight, atomic writes, and full RNG capture/restore. The charter's P0 build requirement is
**already met by the live trainer**. Any new term must register its state in that existing schema or
resume will silently restart it.

## 4. What I did NOT do — plainly

- **I did not build a harness and I did not smoke the live trainer.** The charter's central build —
  "the first rate-in-objective trainer" — **already exists and is mandatory**, so building it would
  have rebuilt what is there. That is a redirect, not a completed build, and I am not calling it one.
- **I published a wrong headline and it stood for ~2 hours.** §0 retracts it in full.
- **I did not run R0**, the measurement I am recommending most strongly. It needs the dx2 checkpoint
  loaded through the pr130 code root on the SSD; I specified the exact procedure instead of guessing
  its answer.
- Measured cost on the real vehicle is **~48.95 s/epoch on MPS** (7,929 s over epochs 480→642) —
  **received from MAIN, not measured by me.**
- No scorer ran. No Modal. No Metal burn. No candidate archive. Pointer UNMOVED.

## 5. Apparatus repaired in passing (landed `bfcff07016`)

`--help` **crashed** on `experiments/train_tr1_partition_renderer_mlx.py` — the whole 116-flag
surface — for 3 weeks. One unescaped `%` ("~100% of its epochs"): argparse renders help as
`text % params`, so `% o` is read as the `%o` octal specifier and `format_help()` raises. Every agent
obeying CLAUDE.md's "NEVER invent CLI flags — read the target's argparse first" got a `TypeError`.

A repo-wide guard for this exact class already existed and is well built
(`src/tac/tests/test_cli_help_strings_render.py` — positive/negative controls, a denominator, and
this very trainer in its parametrize list). **It was completely unwired.** `06fa0ad37d` landed it and
escaped all 10 then-live sites; `08a472aa29` the same day reintroduced the class on the trainer the
guard names by hand, and nothing unconditional ever ran it: the commit hook's pytest step is
subset-selected from the staged diff, and that selector links tests to sources by module-reference
tokens, which a test naming its targets as *string-literal paths* does not emit. The instrument
existed; its trigger could not reach it.

Repaired: 4 live sites in 4 files (repo-wide live count now 0, denominator 6,609 files declaring
`add_argument`), plus `run_argparse_help_render_scan` in `tools/preflight_hook.py` — a BLOCKING
staged-diff step firing at the moment of introduction, with positive/negative/CWD-independence
controls and a test asserting it is actually called from `main()`. Round-2 self-review of my own cure
found it returned a **false PASS** when CWD was not the repo root, and that its silent success was
the same vacuity signature I had cited to justify it; both fixed, both pinned by tests.

## NOT CLAIMED

No ΔS. No byte credit. No distortion measurement. No frontier movement. **R0 is unmeasured** — I
claim only that `variable_weight_bits` is an allocation accounting by inspection of its source, NOT
that it diverges from shipped bytes; my §3 prediction that it does is a PREDICTION, recorded to be
scored, not a result. The retracted `packed_size` finding is withdrawn as a claim about dx2 and is
retained only as the lens in §0. TR1's ρ, permutation spread, and `w_rate` are MECHANISM on a retired
vehicle and transfer as no number. The 48.95 s/epoch figure is MAIN's, not mine.

## STORES CONSULTED

`ddm_fb1_sub012_feasibility_bound_20260823.md` (`9c137a91ed`) · `ddm_ar1b_archive_residue_purchase_20260822.md`
(`e864cb4ab4`) · `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (CITED) ·
`ddm_sy2_composition_synergy_deep_pass_20260823.md` (`fe2ba12dc2`) · `ddm_tac1_two_axis_composition_20260823.md`
· `ddm_rg5_rate_gradient_sign_20260801.md` + `ddm_rg5_rows_20260801.jsonl` (159 rows) ·
`ddm_rsf1_rate_surrogate_fidelity_20260801.md` · `measured_lever_inventory_for_synergy_pass_20260701T001751Z.md`
· `mallat_balle_deepmath_review_20260707.md` · `ddm_wq1` (`1cc670031c`, via MAIN; not duplicated) ·
source: `tools/train_ddm_cl1_hpac_capacity{,_mps}.py`, `pr130_eureka_intake_20260806/repro_repo/code/hpac_self_compress.py`,
`train_tr1_partition_renderer_mlx.py`, `train_semantic_quantized_resumable.py`,
`lifted/train_semantic_quantized.py`, `boundary_math/weight_entropy_penalty_mlx.py`.

Receipts: `/Volumes/APDataStore/pact/ddm_nt1_trained_at_target_rate/` (`nt1_arithmetic.txt` +
`SHA256SUMS.txt`; APDataStore 201 GiB free — Vertigo at 8.4 GiB was read but NOT written, per charter).

---

`dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]` — gap to 0.12 = 0.028220 ⇒ shed
42,382 B at fixed distortion, or 150 B at zero distortion.
