# ddm_rr2 — recursive adversarial review, LAYER 2: the SEMANTIC leg, element-deep

**Operator directive 2026-08-09:** *"The Recursive adversarial Review must check every step of every
stage of everything, recursive fractal of our port to upstream PR one thirty."*

Fresh eyes. MAIN reviewed his own landings in round 1 and found real defects; your job is the ones his
frame could not see.

## THE OBJECT

`BASE = PR130 CPR1 S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive 191,052 B.
Intake (READ-ONLY): `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`,
`SOURCE_REPO_HEAD = e34f31bc4969042c0051ac81aa3c56884419a231`.
Ledger under audit: `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md` (3856788c96)
+ `THROUGHPUT_ROOT_CAUSE.md` (d28fde10f5).

The semantic leg is 19.15% of archive bytes (36,580 B → marginal 0.0243571 S) and the renderer is
**75.40% of a training step** at 5.5% of the measured 14,472 GFLOP/s device ceiling.

## YOUR SCOPE — every step of the semantic stage

1. **`code/train_semantic_quantized.py` step body (≈lines 260–330)** — read EVERY line. The loss, the
   float-warmup branch, `render_quantized` vs `render_float`, the `exact_path` flag, the CE/softplus
   fraction schedule, grad clipping, EMA if any, the eval branch. Name anything that could silently
   not fire.
2. **`quantized_forward` / `render_quantized` / `fake_quantize`** — the QAT path. `quant_bits=4`,
   per-tensor vs per-channel scales, the `.to(torch.float16).float()` scale rounding, the STE. Is the
   deployed quantization the same object the eval measures?
3. **`code/semantic_renderer_oracle.py`** — `TokenBlock` and `SemanticTokenRenderer.forward`. Verify
   MAIN's architecture read (depthwise 3×3 groups=width → pointwise 1×1 → GroupNorm → FiLM → GELU →
   residual, ×4 blocks, width 96, 384×512 token grid, `[B,H,W]` long conditioning). His FLOP
   derivation (≈15.9 GFLOP fwd/img, ≈47.6 fwd+bwd, 95 GFLOP at B=2) rests on this — re-derive it
   INDEPENDENTLY and report agreement or a corrected figure.
4. **The `--init` checkpoint lineage** — `semantic_renderer_w96_b4_qat4_12k.pt` feeds the 6k tail.
   Where did IT come from? Does its own embedded config also describe an ancestor (the F4 hazard,
   recursively)? How deep does the stale-config chain go?
5. **`evaluate_all` vs `evaluate_rgb`** — what exactly does each measure, on which cache, at which
   batch size, and is `evaluate_all`'s "exact" the same argmax the contest scorer computes?
6. **The F4 blast radius** — every consumer in OUR tree or the intake that reads `ckpt["config"]`
   and could therefore inherit ancestor schedule/precision values.

## ROUND-1 FINDINGS — do NOT re-derive, build past

- F4 (yours to extend): shipped semantic ckpt `config` = ANCESTOR run (steps 3000, lr 1e-3,
  `amp: True`, save path `…w96_b2_12x3000`) vs file `…w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`;
  trainer has NO autocast/GradScaler (the three "amp" grep hits were all `clamp` — substring FP).
  Architecture fields validated by strict `load_state_dict`.
- A4 REFUTED: no `--master-cache` in the semantic invocation → `master_targets is None` → one eval
  per point; 24 in-loop + 1 pre-loop `step: 0` = 25 evals.
- MEASURED on Metal: semantic inference DALI-GT 0.0002857038709852431 = 0.998650× published Ada,
  19 s/n600. Renderer op split: GroupNorm 16.22 ms (~45% of a block, ~zero FLOPs), pointwise 7.40,
  depthwise 7.01, GELU 5.72, QAT re-param+fwd 8.04. fp16/bf16 autocast measured WORSE (0.97×/0.96×);
  `torch.compile` CRASHES on MPS in `aten.convolution_backward`.

## OPTIMAL FORM

- **Reference form:** line-by-line source read of the full step body + the renderer + the quantizer,
  with an INDEPENDENT re-derivation of the FLOP arithmetic (not a re-quote of MAIN's).
- **SCOPE reductions (legal):** static reading instead of execution — arms have no Metal device
  (pp2 lesson). CPU-only spot-checks of pure-python logic are fine.
- **MECHANISM reductions (declare TOY-BRACKET):** re-quoting MAIN's FLOP figure instead of deriving
  it; reading function names instead of bodies; assuming a flag fires because it is defined.
- **Provenance pins:** intake `e34f31bc4969042c0051ac81aa3c56884419a231`; ledger 3856788c96;
  checkpoint `artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`.

## NON-NEGOTIABLES

- Intake READ-ONLY. Never edit in place; never `git add` inside it.
- MPS/MLX never score authority; you have no Metal device — do not claim device measurements.
- No number without a locatable receipt; ABSENT is an honest answer, restating is not.
- verdict_scope on every negative (INSTANCE / FORMULATION / FAMILY).
- Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py`; fine for `.md`/`.json`.

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/RR2_SEMANTIC_LEG_AUDIT.md` — per-element table, ranked
findings with falsifiers, the independent FLOP re-derivation with its arithmetic shown, the
stale-config chain depth, and an explicit "could not check / why" section.
