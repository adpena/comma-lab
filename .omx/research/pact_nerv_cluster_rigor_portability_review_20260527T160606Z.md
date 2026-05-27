# PACT-NeRV `_full_main` cluster — ADVERSARIAL RIGOR + PORTABILITY review — 2026-05-27T16:06:06Z

**Reviewer:** `pact_nerv_rigor_portability_review_20260527` (READ-ONLY; only write is this memo)
**Scope:** the 18 PACT-NeRV `_full_main` variants (commits `259292757`/`b5e6331e4`/`069360c7a`/`f2e1076a8` cluster 1 + `6651189bd`/`142940946`/`9077d19bb`/`9e47a44df` cluster 2) + shared base `src/tac/substrates/_shared/pact_nerv_full_main.py`.
**GPU spend:** $0 (read-only static analysis).

## Headline verdict

- **RIGOR grade: A− (PASS).** All 18 `_full_main` are REAL score-aware training loops. Zero stubs, zero relocated `NotImplementedError`, zero synthetic data in the full path. Catalog #325 gating fully intact (18/18 recipes `dispatch_enabled: false` + `research_only: true`, zero leak).
- **PORTABILITY grade: F (BLOCKER — ALL 18 FAIL the numpy-portable-inflate requirement).** Every one of the 18 `inflate.py` files imports `torch` at module top level AND the decode path is load-bearing torch (`Module(cfg)` reconstruct + `model(idx_tensor)` forward). The landing memos' repeated claim "numpy/PIL-portable inflate (no MLX dep)" is **TRUE about MLX but FALSE about portability** — it conflates "no MLX" with "numpy-portable". The 8th standing directive's portability requirement is numpy-portable inflate (no torch/MLX at decode), which NONE of the 18 satisfy.
- **PyTorch-native-vs-MLX-first recommendation: KEEP PyTorch-native for TRAINING; the inflate framework-dependence is a separate, real shippability gap to resolve before any contest submission — but it is NOT an MLX-migration trigger.** See Q3.

---

## Q1 — Training rigor: VERDICT PASS (A−)

**Evidence (static, all 18 trainers):**
- `raise NotImplementedError` count in `_full_main` = **0** across all 18 (grep-confirmed).
- `pass`-stub in `_full_main` = **0**.
- Routes through canonical `run_pact_nerv_score_aware_training` = **present in all 18** (2–3 references each).
- `load_differentiable_scorers` = present all 18.
- `patch_upstream_yuv6_globally` BEFORE scorer construction (eval_roundtrip non-negotiable / Catalog #6) = present all 18 (verified in ia3 body: `patch_upstream_yuv6_globally()` → `load_differentiable_scorers` → frozen scorers → real-video decode → model build → Lagrangian).
- `apply_eval_roundtrip=True` passed to the loss callback (verified in ia3 `_compute_loss`).
- Synthetic data in the full path (`make_synthetic` / `randn` / `rand`) = **0** in every `_full_main` body (Catalog #114 satisfied; synthetic stays in `--smoke` only).
- Real contest video: `decode_pairs_for_training` → canonical `trainer_skeleton.decode_real_pairs` (Catalog #114).
- EMA: the shared loop instantiates `EMA(model, decay=0.997)`, updates after EVERY `optimizer.step()`, applies EMA-at-eval with snapshot/restore (`orig_state` clone → `ema.apply` → eval → `load_state_dict(orig_state)`), saves the EMA shadow (NOT live weights) as `best.pt` — CLAUDE.md "EMA — NON-NEGOTIABLE" fully honored.
- Best-checkpoint selection by validation Lagrangian + end-of-training EMA fallback if no improving checkpoint.
- NaN watchdog (Council D) with `max_nan_strikes=3` abort-to-preserve-EMA.
- Deterministic archive ZIP (`build_archive_zip`: ZipInfo + fixed timestamp `(2026,1,1,0,0,0)` + DEFLATE per Catalog #19).
- Scorer-loss routing: the `compute_loss` callback routes through the variant's `score_aware_loss.py` → Catalog #164 `score_pair_components_dispatch` (verified ia3: `PactNervIa3ScoreAwareLoss(seg_scorer, pose_scorer)`; scorers loaded with `requires_grad_(False)` + `.eval()` so gradient flows through frozen SegNet/PoseNet, which is correct — the renderer is what trains).
- Auth-eval tail: `_canon_gate_auth_eval_call` (canonical `gate_auth_eval_call`, Catalog #226) producing `contest_auth_eval_cuda.json`, then `posterior_update_locked(ContestResult(...))` (Catalog #128) — fires only when a real `[contest-CUDA]` score returns.

**The only reason this is A− not A:** the rigor signals are STATIC (keyword + body inspection). No paid dispatch has run, so there is no empirical proof the training converges or the loss is well-conditioned per variant. That is correct-by-design (Catalog #325 gates the paid path) — but the rigor grade is "the code is a real loop", not "the loop has been validated to produce score". The landing memos' end-to-end loss-callback backward-pass smoke (6 trickiest variants, finite + backward_ok) is the strongest empirical evidence and is credible.

---

## Q2 — Catalog #325 gating: VERDICT PASS (no leak)

**Evidence:** 18/18 recipes at `.omx/operator_authorize_recipes/substrate_pact_nerv_*_modal_t4_dispatch.yaml` carry `dispatch_enabled: false` AND `research_only: true` (full enumeration confirmed). The full path is CUDA-required: every trainer calls `device_or_die` (5–6 refs each); `_full_main(--device cpu)` raises `SystemExit` (per the landing memos' `test_trainer_full_main_implemented_and_cuda_gated`). No leak: the trigger is gated, the code is complete. This is the correct "implement all without firing council-gated paid paths" posture.

---

## Q3 — THE PORTABILITY QUESTION: VERDICT FAIL (numpy-portability) + KEEP PyTorch-native (training)

This is the load-bearing finding. Three sub-questions:

### (a) Is each variant's INFLATE path genuinely numpy/PIL-portable (no torch/MLX at decode)? — **NO, ALL 18 FAIL.**

Grep of all 18 `inflate.py`:

| inflate.py | LOC | top-level `import torch`/`mlx` |
|---|---|---|
| ALL 18 (asymmetric_boundary, bayesian, cross_codec_a/b, diffusion_distilled/trajectory, distilled_scorer, ia3, ia3_multi, mamba, moe, multi_modal, neural_codec_e2e(/_cross), selector_v2/v3/v4, vq) | 68–101 | **1 (`import torch`)** |

The torch import is **load-bearing**, not stray. Verified in `pact_nerv_ia3/inflate.py`: the decode is

```python
import torch
from .architecture import PactNervIa3Config, PactNervIa3Substrate  # torch.nn.Module
...
model = PactNervIa3Substrate(cfg).to(device).eval()      # builds a torch module
model.load_state_dict(arc.decoder_state_dict, strict=False)
with torch.no_grad():
    rgb_0, rgb_1 = model(idx_tensor)                     # torch forward pass IS the decode
```

`architecture.py` is `torch.nn.Module` (confirmed ia3/vq/mamba: `import torch.nn as nn`, `class ...Substrate(nn.Module)`). The inflate cannot run without torch. The docstring even self-admits it: *"L4 budget: ≤ 2 external deps (torch + brotli)"* — explicitly listing torch as a runtime dep.

**This is a shippability blocker for the numpy-portable-inflate requirement.** Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL STANDING DIRECTIVE": *"INFLATE numpy-portable (no MLX dep; ≤200 LOC + ≤2 ext deps per HNeRV L4) + bridge contract MLX state_dict → npz → ZIP-member → numpy inflate primitives."* The intended portability is **numpy inflate primitives** — NOT a torch forward pass. All 18 use a torch forward pass.

**Failing variants (the blocker list — name + offending import):** every one of the 18 — the exact offending import is `import torch` at the inflate module top level (`pact_nerv_ia3/inflate.py:22`, and the structurally-identical line in the other 17), made load-bearing by the `from .architecture import ...Substrate` (torch.nn.Module) reconstruct + `model(idx)` forward.

> **Nuance for the operator:** the prompt's framing says the rationale was "inflate path is numpy/PIL-portable." That rationale is **factually incorrect** as landed. The inflate path is torch-portable (runs on CPU via torch, no MLX, no CUDA-required), which is a WEAKER portability claim than the directive's numpy-portable bar. A contest reviewer who only has `pip install torch` will run it; a reviewer expecting the HNeRV-L4-class "≤2 deps, numpy + PIL" minimal-runtime will not get it. Whether torch-at-inflate is contest-disqualifying depends on the contest's allowed-runtime contract — but it unambiguously violates the operator's own 8th standing directive.

### (b) Could the inflate use the canonical tinygrad-like numpy primitives in `tac.local_acceleration.pr95_hnerv_mlx`? — **YES, and NONE do.**

Grep: zero of the 18 `inflate.py` reference `bilinear_resize_nhwc`, `_pixel_shuffle_2x_nhwc`, `pr95_hnerv_mlx`, or `local_acceleration`. The PACT-NeRV decoders are HNeRV-class (`_SinAct` + `_DepthSepConv` + pixel-shuffle upsample) — exactly the family the canonical numpy primitives were built to decode framework-free. A numpy-portable inflate IS achievable: re-express the `decoder_state_dict` forward as numpy conv/pixel-shuffle/sin-activation using the canonical primitives (the `parse_archive` already returns a plain state_dict + latents; the only torch dependence is the forward, which is portable to numpy). This is the canonical bridge the directive describes.

### (c) Is the PyTorch-native TRAINING decision HARD-EARNED or CARGO-CULTED? — **HARD-EARNED (Catalog #307: implementation choice, not paradigm shortcut).**

The training-side PyTorch decision is sound and I classify it HARD-EARNED:
- All 18 architectures are pre-existing `torch.nn.Module` families (landed at L0 SCAFFOLD 2026-05-20, before this `_full_main` wave).
- The canonical score-aware loss routes through PyTorch SegNet/PoseNet (`load_differentiable_scorers` + Catalog #164). The scorers ARE torch; gradient-through-scorer training is genuinely a torch operation today.
- The implemented sister `ds_nerv` uses the identical PyTorch loop. Re-using it is the canonical-vs-unique correct call (ADOPT_CANONICAL for the agnostic scaffold).
- Migrating 18 architectures + the differentiable scorer path to MLX is a multi-week architecture migration, NOT a `_full_main` implementation. The 8th directive's MLX-first clause is explicitly scoped to NEW substrates; this is an EXISTING family. Forcing MLX here would be the path-of-MOST-resistance, not least.

**So the training decision is correct. The portability MISCLAIM is the bug** — the memos asserted the inflate satisfies the directive's portability bar when it does not.

---

## Q4 — Canonical-vs-unique soundness (Catalog #290): VERDICT PASS (substantive, not boilerplate)

The `## Canonical-vs-unique decision per layer` section in the shared base + both memos is substantive: it correctly identifies the train/val loop + EMA + NaN watchdog + decode + scorer-load + runtime-emission as ADOPT_CANONICAL (substrate-agnostic) and architecture + archive grammar + inflate + score-aware-loss as FORK (per-package, with concrete rationale per layer — e.g. "ia3 ships ego_poses + pose_dim; vq ships codebook + indices; selector_v2 ships selector_bytes + palette_size"). The `compute_loss` callback as the canonical extension point is the correct design (variant-specific extra terms — moe load_balance_aux positional, bayesian kl_div kwarg, ncec gate_values kwarg — stay in the package). This is a genuine per-layer decision, not a template stamp.

**One soundness caveat:** the section claims the inflate FORK is "numpy/PIL-portable (no MLX dep)" as a settled fact. That is the Q3 misclaim re-stated. The canonical-vs-unique reasoning is sound; the portability ASSERTION embedded in it is wrong.

---

## Q5 — Distinctness: VERDICT PASS-WITH-CAVEAT (genuinely distinct, family is broad)

The 18 are NOT near-duplicate shells. Each has a distinct `class Pact...Substrate` + a real distinguishing mechanism:
- selector_v2/v3/v4 ARE genuinely distinct entropy coders: `selector_v2` arithmetic (Witten 1987), `selector_v3` `RiceGolombSelectorCoder` (Golomb 1966 + Rice 1971), `selector_v4` `RunLengthSelectorCoder` (RLE). Real coder classes, not renamed copies.
- vq (VQ-VAE codebook + commitment), mamba (SSM `ssm.A_log`), moe (MoE routing + load-balance aux), bayesian (variational latents + KL), ia3/ia3_multi (γ-only modulation), neural_codec_e2e(/_cross) (Ballé hyperprior + rate term), cross_codec_a/b (base-bytes + side-info composition), diffusion_distilled/trajectory (student/trajectory), multi_modal (pose+class+odometry fusion), asymmetric_boundary (boundary-FiLM), distilled_scorer (Hinton KL surrogate) — all distinct primitives.

**Caveat (the 12-variant subagent's own honest note, confirmed):** only 3 needed a genuinely distinct loss-term path (moe / bayesian / ncec); the other 15 share the standard callback because their distinguishing primitive is buffer-backed conditioning internal to `model(idx)`. The family is BROAD — many share the same `_SinAct`/`_DepthSepConv` NeRV backbone with the distinguishing layer bolted on. This is defensible per UNIQUE-AND-COMPLETE-PER-METHOD (each IS a complete distinct method) but the EV-per-variant is uneven: the selector-coder trio and the cross-codec composition variants are the most distinct; several (ia3 vs ia3_multi, selector_v3 vs v4) are close cousins. Not a defect — but the operator should expect the per-substrate symposia (Catalog #325) to consolidate near-cousins rather than dispatch all 18.

---

## Operator-routable next steps

1. **PORTABILITY BLOCKER (highest priority before ANY contest submission of this family):** the "numpy-portable inflate" claim in both landing memos + the shared-base docstring is FALSE for all 18. Either (a) re-author each `inflate.py` to a numpy forward path using the canonical `tac.local_acceleration.pr95_hnerv_mlx` primitives (`bilinear_resize_nhwc` / `_pixel_shuffle_2x_nhwc` / sin-activation) so decode needs only numpy + PIL + brotli (the directive's bar), OR (b) explicitly downgrade the memo + docstring claims to "torch-portable inflate (CPU-runnable, no MLX/CUDA; torch IS a runtime dep)" and obtain operator sign-off that torch-at-inflate is acceptable for the target contest-runtime contract. Path (a) is the correct one per the 8th directive. This is a SISTER subagent task — the concurrent inflate-portability-audit subagent owns inflate.py EDITS and should be routed this exact finding (it currently scopes the 5 class-shift PyTorch substrates + PACT-NeRV inflate; this confirms all 18 PACT-NeRV inflates fail).
2. **DOCSTRING/MEMO CORRECTION (append-only per Catalog #110/#113):** the shared-base docstring line "The INFLATE path is already numpy/PIL-portable (no MLX dep)" and the two memos' identical claim are factually wrong. Append a correction note (do NOT mutate the historical memos) classifying the inflate as torch-portable-not-numpy-portable.
3. **KEEP PyTorch-native training (no MLX migration):** the training-side decision is HARD-EARNED and correct. Do NOT migrate the 18 to MLX renderers — that is a separate architecture program, not a portability fix. The MLX-harness sister subagent's 6 NEW substrates are the correct place for MLX-first; the PACT-NeRV family stays PyTorch.
4. **Distinctness consolidation (Catalog #325 symposia):** route the near-cousin variants (ia3/ia3_multi; selector_v3/v4; cross_codec_a/b; diffusion_distilled/trajectory; neural_codec_e2e/_cross) through per-substrate symposia that explicitly justify keeping BOTH vs consolidating, per HNeRV parity L7 + the operator's "is the family over-broad" lens. The EV-per-variant is uneven; the symposium is the right gate.
5. **Catalog #344 FORMALIZATION_PENDING resolution:** the `pact_nerv_family_score_domain_lagrangian_v1` canonical equation is still pending registration (both memos cite the fcntl-lock race with the registry-owning sister). Confirm it lands once the registry sister completes.

## Per-question verdict table

| Q | Verdict | Grade |
|---|---|---|
| Q1 Training rigor | PASS — all 18 real score-aware loops, EMA, eval_roundtrip, best-ckpt, deterministic archive, 0 synthetic in full path | A− |
| Q2 Catalog #325 gating | PASS — 18/18 dispatch_enabled:false + research_only:true + CUDA-required, zero leak | A |
| Q3 Portability | FAIL — all 18 inflate import load-bearing torch; numpy-portability MISCLAIMED; KEEP PyTorch training | F (inflate) / A (training decision) |
| Q4 Canonical-vs-unique | PASS — substantive per-layer decision, not boilerplate (carries the Q3 portability misclaim) | A− |
| Q5 Distinctness | PASS-WITH-CAVEAT — genuinely distinct primitives; family broad; near-cousins should consolidate at symposia | B+ |

**RIGOR grade: A−.** **PORTABILITY grade: F (inflate paths NOT shippable-numpy-portable; all 18 fail; offending import `import torch` at module top + torch.nn.Module forward).** **Recommendation: KEEP PyTorch-native (training is HARD-EARNED correct); FIX the inflate numpy-portability misclaim — either re-author to numpy primitives (preferred, directive-compliant) or downgrade the claim with operator sign-off. NOT an MLX-migration trigger.**

[verified-against: src/tac/substrates/_shared/pact_nerv_full_main.py shared loop + EMA + deterministic zip + runtime emission]
[verified-against: all 18 src/tac/substrates/pact_nerv_*/inflate.py — every one imports load-bearing torch (grep-confirmed)]
[verified-against: src/tac/substrates/pact_nerv_ia3/inflate.py decode path = torch.nn.Module reconstruct + model(idx) forward]
[verified-against: 18/18 .omx/operator_authorize_recipes/substrate_pact_nerv_*.yaml dispatch_enabled:false + research_only:true]
[verified-against: experiments/train_substrate_pact_nerv_ia3.py::_full_main full body — real video + yuv6-patch + frozen scorers + EMA + auth-gate + posterior]
[verified-against: selector_v2/v3/v4 architecture.py — distinct arithmetic/Rice-Golomb/RLE coder classes]
