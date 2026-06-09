# pact_nerv_vq — EXTREME-SCRUTINY full-stack provenance audit vs the frozen `evaluate.py`

Date: 2026-06-09
Author: Claude MAX-REASONING full-stack scrutiny subagent (READ-ONLY; no code edits, no training, no commit to any carrier source or `upstream/`).
Subagent id: `pact_nerv_vq_fullstack_scrutiny_20260609`
Operator directive 2026-06-09 verbatim: *"all vehicles need extreme scrutiny all lines full stack"*. Vehicle: **pact_nerv_vq** (V4, the VQ-codebook carrier).

Sister inputs deepened:
- `.omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md` (the manifest; its pact_nerv_vq row §1.2 + V1/V2/V3 §2.2 + one-line verdict §5).
- `.omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` (the H1–H5 / F1–F6 template + the HiNeRV class-B precedent).
- `.omx/research/pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md` (today; carries the decisive empirical receipt that even with real scorer teachers bound, d_seg≈0.5).
- 2026-06-02 codex audits cited therein (`codex_findings_pact_vq_qat4_rate_solved_fit_blocked`, `..._competitiveness_gate`, `..._compact_vq_pivot_audit`).

**Axis discipline.** Every numeric is tagged. The ONLY score authority is exact `upstream/evaluate.py` on `[contest-CPU]` (Linux x86_64) or `[contest-CUDA]` (T4). The d_seg≈0.5 / "dark mean-field image" findings carried here are `[macOS-MLX research-signal]` — a **mechanism diagnostic, NOT a score claim**. PSNR is `[advisory only]`. This memo is `research_only=true` / `mechanism_update_eligible`. Per CLAUDE.md "Forbidden premature KILL": nothing here KILLS the VQ-VAE paradigm; it classifies the carrier's expected failure at the IMPLEMENTATION level (Catalog #307) and names the single binding fix. PSNR ≠ d_seg: SegNet keys on HF argmax boundaries (last frame, 5-class), not pixel MSE.

---

## 0. Headline (the operator's exact A/B/C provenance question, answered)

**pact_nerv_vq's expected failure is class-(B) INCOMPLETENESS / ORCHESTRATION ARTIFACT — ~85%, with ~15% class-(A) JUSTIFIED ADAPTATION and 0% class-(C) FAKE.** The VQ machinery, the index byte-consumption, the archive grammar, the no-op detector, the competitiveness gate, and the tests are all GENUINE (real, not fake — confirmed line-by-line). The carrier's single binding gap — a **skip-free `PixelShuffle(sin(w·conv))` decoder with no bilinear-skip, no PE, no terminal refine** — is class-(B): it was **batch-generated as one of 5 "L0 SCAFFOLD" variants in a single commit (`b56f24bc1`), is documented as "L0 SKETCH" throughout, has NEVER been iterated to add the documented PR95-parity L18 HF path, and that omission carries NO design-memo justification.** It is the exact HiNeRV analog: an honestly-labeled L0 SKETCH wired into a long-run MLX lane before the decoder matured.

**This CONFIRMS the sister manifest's claim ("genuine VQ but skip-free decoder will mean-field") AND sharpens it with a decisive empirical receipt the sister memo did not have:** today's maturity-audit memo (§96) reports that **even with real SegNet/PoseNet teachers bound, the render stays at d_seg≈0.5 — "a dark mean-field image, not a road-scene renderer"** (`pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md:96`, citing 2026-06-02 codex `codex_findings_pact_vq_qat4_rate_solved_fit_blocked`). This is the falsifying experiment the sister memo's "shared mistake B" needed: turning the objective ON (addressing Mistake-B) did NOT lift d_seg, which **proves the skip-free decoder (Mistake-A) is the binding constraint here, not the objective.** The VQ "rate is solved (22–34 KB receiver-proven)"; the decoder cannot synthesize the HF that d_seg lives on.

The ~15% class-(A): the team DID examine specific layers and make principled forks where they looked — most concretely the substrate-specific `export_state_dict` bridge fix (`pact_nerv_vq_l1_long_run_mlx_landed_20260528.md:32`, a `FORK_BECAUSE_PRINCIPLED_MISMATCH` per Catalog #290), the MLX EMA dead-code dormancy guard (mlx_renderer.py:352–365, with explicit rationale), and the `int8_mixed` decoder codec. The decoder topology was simply never one of the layers anyone examined — it was `ADOPT-CANONICAL`'d (`__init__` of the MLX renderer docstring lines 80–82) to a topology that is canonical-for-NeRV but is NOT the PR95-winning skip+refine topology.

---

## 1. Per-file / per-component A/B/C provenance table

Verdict legend: **A** = justified adaptation (deliberate + score-rational + backed by memo/`Canonical-vs-unique`/in-code rationale). **B** = incompleteness / orchestration artifact / unexamined inheritance / SKETCH-trained-before-maturation. **C** = FAKE (claims X, doesn't do X on real inputs; one of the 5 CLAUDE.md forbidden classes). **GENUINE** = does what it claims (not a divergence at all).

| Component | File:line (claim) | File:line (justification / its absence) | Verdict | Notes |
|---|---|---|---|---|
| VQ nearest-codebook lookup | architecture.py:136–142 (real `‖z_e‖²−2 z_e·cbᵀ+‖cb‖²`, `argmin`) | self-evident math + tested architecture.py:89–102 | **GENUINE** | vdO §3.1 distance metric, exact. |
| Straight-through estimator | architecture.py:147 `z_q_st = z_e + (z_q − z_e).detach()` | tested by real `backward()` test_pact_nerv_vq.py:99–102 | **GENUINE** | Bengio 2013 / vdO §3.1, exact. Gradient genuinely flows to `z_e`. |
| EMA codebook update + Laplace | architecture.py:150–166 (`@torch.no_grad`, persistent N_c/m_c, decay 0.99, Laplace 161) | docstring 38–39 cargo-cult HARD-EARNED; tested behaviorally (MLX) :120–127 | **GENUINE** | vdO §3.2, exact. |
| Commitment loss | architecture.py:144 `F.mse_loss(z_e, z_q.detach())`; weight 0.25 score_aware_loss.py:34, 109 | docstring 39 HARD-EARNED (vdO §3.1) | **GENUINE** | Returned for the Lagrangian; genuinely added. |
| MLX VQ (STE/commit/EMA) | mlx_renderer.py:311–316 (`mx.stop_gradient`), 313 (commit), 319–365 (EMA) | tested behaviorally (codebook actually changes) :120–127 | **GENUINE** | 1:1 mirror of PyTorch VQ. |
| MLX EMA **dead-code dormancy guard** | mlx_renderer.py:352–365 (never-used codes stay dormant) | in-code rationale 352–354 (prevents collapse-to-one-code) | **A (justified)** | A real, reasoned fix beyond the PyTorch sister; addresses sister manifest's V3 collapse risk. |
| Codebook size = single `int=512` | architecture.py:40, 101 | docstring 40–43 CARGO-CULTED-at-L0, sweep deferred | **A (honest cargo-cult)** | NOT Catalog #308 enum-padding — exactly one VQ mechanism, no fake enum branches. |
| **Decoder upsample atom `_DsUpBlock`** | architecture.py:70–78 `shuffle(act(dsc(x)))` — NO skip, NO PE, NO refine; forward loop 261–262 bare | **NO commit, NO memo line justifies the skip omission** (git `b56f24bc1` only; grep skip/bilinear/residual in substrate memos = 0 hits) | **B (INCOMPLETENESS)** | THE binding gap. See §2. |
| MLX decoder atom `_DsUpBlockMLX` | mlx_renderer.py:211–214 identical skip-free; forward 531–532 bare | docstring 80–82 `ADOPT-CANONICAL` (but to skip-FREE topology, not PR95 skip+refine) | **B (INCOMPLETENESS)** | Same gap, mirrored into the trained path. |
| w=30 SIREN on coordinate-free feature maps | architecture.py:38, 56–57; mlx 208,213 | docstring 38 lists codebook as HARD-EARNED but is SILENT on w=30 fit to a skip-free feature-map NeRV | **B (unexamined inheritance)** | HARD-EARNED for coordinate-MLP SIREN; CARGO-CULTED here (sister H4/V2). |
| SIREN init | architecture.py:219–236; mlx 453–495 | standard SIREN init | **GENUINE** | Correct fan-in / `w` scaling. |
| Score-aware loss (PyTorch) shape | score_aware_loss.py:84–106 (`score_pair_components_dispatch` frozen scorer + commitment; **NO recon-MSE base**) | imports frozen dispatch score_aware_common.py:24 ("frozen"), weights 100·seg + √10·pose | **A (GOOD shape — better than the MLX harness)** | This is the *correct* objective; eval_roundtrip enforced :72–76. |
| eval_roundtrip enforcement | score_aware_loss.py:72–76 (raises if `apply_eval_roundtrip=False`); :81–82 applies it | CLAUDE.md non-negotiable | **GENUINE** | Differentiable YUV6 via `apply_eval_roundtrip_during_training` :78. |
| **Trained objective = the SHARED MLX harness recon-MSE base** | mlx_renderer.py:61–63, 93 routes training to `run_mlx_score_aware_full_main`; harness `recon_weight: float = 1.0` loss.py:3099, `recon = mse_0+mse_1` loss.py:3176/3200; scorer terms opt-in (default 0.0) | docstring 80–81 calls decoder ADOPT-CANONICAL; the MLX-train objective choice is NOT examined in the memo | **B (objective surface; the trained path re-introduces recon-MSE)** | The PyTorch loss avoids it; the path that ACTUALLY trains does not. Sister Mistake-B — but empirically secondary here (§0, §3). |
| Archive grammar (PVQ 0.bin) | archive.py:133–138 (27B header), 188–258 pack, 261–334 parse; size invariant 292–295, magic 277, codebook-len invariant 282–286 | export-first declared `__init__.py:23–36` (Catalog #124 8 fields) | **GENUINE** | Monolithic single-file, exact length-prefix offsets, fail-closed. |
| Indices charged + uint16-coded | archive.py:223–228 (`encode_uint_stream` max_value bound); section "selectors_rc" section_value.py:69–72 | declared grammar `__init__.py:30,32` | **GENUINE** | Indices are real charged bytes; not inert. |
| Codebook int16-quant / fp32 reconstruct | archive.py:166–185, 220–221, 305–325 | declared `__init__.py:31` | **GENUINE** | Lossy int16 quant of codebook (acceptable; declared). |
| inflate CONSUMES indices | inflate.py:49–55 (`model.latents[i] ← codebook[indices[i]]`) + 57–59 (codebook → quantizer) | byte-level no-op proof tested :295–310 | **GENUINE (byte+parse level)** | Indices genuinely drive which latent each pair decodes. Frame-level caveat → §4. |
| inflate loads NO scorer | inflate.py grep: 0 segnet/posenet/DistortionNet | CLAUDE.md "Strict scorer rule" | **GENUINE (compliant)** | No inflate-time scorer load. |
| inflate runtime dep | inflate.py:9 `import torch` (NOT numpy-only) | declared `runtime_dep_closure: torch, brotli` `__init__.py:29` | **A (internally consistent) / limitation** | Torch-dependent inflate — NOT numpy-portable (unlike SNeRV path A). Within its own declaration. |
| `load_state_dict(strict=False)` | inflate.py:47 | required because codebook/EMA excluded from decoder_sd (archive_candidate.py:173–178) | **B (latent risk)** | `strict=False` is REQUIRED here, but it would silently mask any OTHER missing decoder key. No assertion guards the key set. |
| Section neutralization / no-op detector | section_value.py:96–135 (zeroes decoder / zeroes codebook / collapses indices→0, re-packs valid PVQ) | Catalog #139 sister; carries `FALSE_AUTHORITY` :149 | **GENUINE** | Real counterfactual-able byte-mutation surface. `receiver_state` correctly refused :112–113. |
| MLX→PyTorch export bridge | archive_candidate.py:119–193 (`pack_archive_from_exported_state_dict`); nearest-codebook 85–116 | landing memo :32 `FORK_BECAUSE_PRINCIPLED_MISMATCH` (Catalog #290) | **A (justified, examined fork)** | The one decoder-adjacent layer that WAS examined + iterated. Genuine. |
| Competitiveness gate | competitiveness_gate.py:63–219 (fail-closed; refuses authority rows :306–316; requires MLX axis :319–324; blocks exact spend until receiver-proof + distortion :117–128) | schema + `FALSE_AUTHORITY` :218 | **GENUINE (mature)** | Real distortion gate; verdict `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION` is the honest current state. |
| Tests | test_pact_nerv_vq.py (386 LOC) + test_competitiveness_gate.py (217 LOC) | — | **GENUINE behavioral (NOT FAKE Class-2)** | STE/EMA/no-op tests assert BEHAVIOR (grad flows, codebook changes, bytes change). Would FAIL if bodies → `return markers`. Gaps → §4. |
| Lane registry rows | lanes[1074] `research_only=True`; lanes[1450] | notes cite van den Oord | **B (hygiene discrepancy)** | BOTH rows show `contest_cuda`+`contest_cpu` gates TRUE, but the maturity audit states pact_nerv_vq "has NO exact eval." Catalog #90 inconsistency. |
| Docstring "L0 SKETCH" labeling | architecture.py:2,170; score_aware_loss.py:39; mlx_renderer.py:2 ("L1 LONG-RUN") | — | **A (honest labeling)** | Unlike sane_hnerv (which docstring-claims a skip it lacks = Catalog #307 doc-fake), pact_nerv_vq does NOT claim a skip. No documentation-fake. |

---

## 2. The binding gap (Mistake-A), with git provenance proving class-(B)

`_DsUpBlock.forward` (architecture.py:77–78) and `_DsUpBlockMLX.__call__` (mlx_renderer.py:211–214) are:

```
shuffle(sin(w · depthsep_conv(x)))          # NO bilinear-skip, NO PE, NO terminal refine
```

The forward (architecture.py:261–262 / mlx_renderer.py:531–532) is a **bare** `for block in self.blocks: h = block(h)` with no residual accumulation. The PR95 reference that WON (cited in the sister memos, `model.py:46–51`) is `sin(PixelShuffle(conv(x)) + bilinear_up(x))` then `x + 0.1·sin(refine(x))`. The residual/skip path is the canonical mechanism that lets the optimizer escape the blurry mean-field; without it the global MSE minimizer (the DC/mean image) is the easy attractor and SegNet's last-frame argmax collapses to ~one class → d_seg ≈ 0.5.

**Why this is class-(B) INCOMPLETENESS, not class-(A) JUSTIFIED ADAPTATION** (decisive evidence):

1. **Git birth as a batch L0 SCAFFOLD, never iterated.** `git log --follow -- architecture.py` → a SINGLE commit `b56f24bc1` *"wave-3-pact-nerv-g2-mid-loc-l0-build: 5 PACT-NERV-ULTIMATE Group 2 mid-LOC **L0 SCAFFOLD** variants"*. The decoder has never been touched since birth. `git log -- pact_nerv_vq/` grep for `skip|bilinear|residual|hf|refine` → **0 hits**. The HF path was never built.
2. **No design-memo justification.** The L0 design memo (`pact_nerv_vq_l0_scaffold_design_20260520T211500Z.md`) and L1 landing memo carry a `Canonical-vs-unique decision per layer`-style cargo-cult table, but it discusses codebook size / latent dim / per-pair-single-token — it is **SILENT on the decoder topology and the skip omission**. The MLX renderer docstring (lines 80–82) `ADOPT-CANONICAL`s the "HNeRV-class base decoder topology" claiming the distinguishing primitive is the VQ tokens "NOT the decoder topology" — i.e. the decoder was *explicitly waved through as not-the-point*, which is precisely the unexamined-inheritance failure mode. The "canonical" topology adopted is the skip-FREE atom, not the PR95 skip+refine atom.
3. **Honestly labeled SKETCH, trained anyway.** Docstrings say "L0 SKETCH" (architecture.py:2,170; score_aware_loss.py:39) — the carrier never CLAIMS to have a skip (so NOT a Catalog #307 documentation-fake like sane_hnerv). But it was wired into the `lane_pact_nerv_vq_long_run_mlx_local_closure_20260528` L1 long-run lane (lanes[1450]) before the decoder matured to the documented PR95-parity L18. **This is the exact HiNeRV precedent** (`7a004e5bd` L0 SKETCH 3-scale NeRV, missing bilinear-skip = class-B, trained/dispatched before maturation).

**The empirical receipt that makes Mistake-A the BINDING gap (not just co-equal with Mistake-B):** today's maturity audit (§96) + the 2026-06-02 codex audit found that **with REAL SegNet/PoseNet teachers bound** (Mistake-B's recon-MSE objective replaced), the render STILL produces "a dark mean-field image, not a road-scene renderer" at d_seg≈0.5 `[macOS-MLX research-signal]`, and the competitiveness gate returned `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`. Turning the objective on did not lift d_seg → the skip-free decoder is the constraint that binds.

---

## 3. The objective surface (Mistake-B) — confirmed but empirically secondary HERE

- The **PyTorch** `score_aware_loss.py` is the RIGHT shape: `100·seg + √10·pose + 0.25·commitment` via the FROZEN-scorer `score_pair_components_dispatch` (score_aware_common.py:24 "frozen"), **no recon-MSE base**, eval_roundtrip enforced (score_aware_loss.py:72–76). Verdict **A (good)**.
- The **path that actually trains at scale** is the shared MLX harness `run_mlx_score_aware_full_main` (mlx_renderer.py:61–63, 93), whose base term is `recon = mse_0 + mse_1` with `recon_weight: float = 1.0` (loss.py:3099, 3176, 3200) and scorer terms opt-in (the SNeRV ep22399 sibling ran `observed_segnet_distillation_weight = None`). MSE rewards the mean-field. Verdict **B (objective surface)**.
- **But** per §2's empirical receipt, binding scorer teachers did NOT fix d_seg here — so for pact_nerv_vq, Mistake-B is real but **secondary to Mistake-A**. Fixing only the objective is insufficient; fixing the decoder is necessary. (This is the same H1≈binding > H3 ordering the HiNeRV memo reached, now with a direct receipt.)

Evaluator-geometry corollary (unchanged from sister memos, re-verified): d_seg is `mean(argmax(out1) ≠ argmax(out2))` on the **last frame only** — a 0/1 argmax-disagreement rate, flat except at SegNet decision boundaries. Only HF boundary structure moves it. A skip-free + recon-MSE carrier produces a low-variance image with no boundary structure → d_seg pinned near 0.5.

---

## 4. Full-stack integrity verdict

| Question | Verdict | Evidence |
|---|---|---|
| Byte-closes (`pack_archive` → valid `0.bin`)? | **YES** | archive.py:188–258; fail-closed invariants 211–218, 282–295. |
| Indices entropy-coded into archive? | **YES** | archive.py:223–228 `encode_uint_stream`; section_value.py:69–72. |
| Indices consumed at inflate (byte-level no-op proof)? | **YES** | inflate.py:49–55; tested test_pact_nerv_vq.py:295–310; section_value.py:96–135 neutralization. |
| Indices consumed at inflate (FRAME-level)? | **UNPROVEN — caveat** | The no-op proof is on archive BYTES (:307), not on rendered pixels. No test asserts different index → different rendered frame. If the decoder mean-fields (§2), index consumption is byte-real but visually near-noop (d_seg insensitive to which code). This is the Mistake-A consequence at the receiver. |
| Parse-back survives? | **YES** | parse_archive round-trips tensors test_pact_nerv_vq.py:130–146, 198–227; `int8_mixed` + `auto` codecs 149–195. |
| MLX↔PyTorch export parity? | **YES (declared 1:1; bridge fixed)** | mlx_renderer.py:594–642 `export_state_dict` PyTorch layout; archive_candidate.py:119–193; bridge KeyError fix landing memo :32. (Numeric parity not re-measured here — INFERRED.) |
| Inflate numpy-portable + within budget + no scorer + no hidden sidecar? | **PARTIAL** | No scorer load (compliant); inflate.py 87 LOC ≤ 200 (tested :379–386); BUT `import torch` (inflate.py:9) → NOT numpy-portable (within its declared dep closure, but a portability limitation vs SNeRV path A). |
| Gradient-reachable through frozen scorers? | **YES (PyTorch loss)** | score_aware_loss.py:78–106 routes through `apply_eval_roundtrip_during_training` + frozen dispatch. |
| Emits a scorable archive? | **YES — the carrier emits a valid contest-shaped `archive.zip`** | archive_candidate.py:196–295 (`0.bin` + vendored runtime + `archive.zip` + sha). Rate is "solved" (22–34 KB receiver-proven per the maturity audit). |
| Does the scorable archive SCORE well? | **NO (the whole point)** | d_seg≈0.5 `[macOS-MLX research-signal]`; competitiveness gate `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`; NO exact CPU/CUDA eval has ever run. |
| Lane-registry hygiene | **DISCREPANCY** | lanes[1074]/[1450] show `contest_cuda`+`contest_cpu` gates TRUE, but maturity audit states "NO exact eval." Flag for Catalog #90 re-audit (not a carrier-code FAKE). |

**FAKE audit (5 forbidden classes):** Class-1 (markers-without-work) NO — VQ does real work. Class-2 (tests-verify-constants) NO — tests assert behavior (grad flows :99–102, codebook changes :126, bytes change :307). Class-3 (synthetic-fixture-as-canonical) NO for the carrier — the audit's d_seg≈0.5 receipt is from the *real contest video via real scorers*, not the unit-test toy fixtures (which are correctly scoped smoke). Class-4 (placeholder-in-data-field) NO. Class-5 (enum-padding) NO — `codebook_size` is a single int, one VQ mechanism. **No FAKE found anywhere in the carrier code.**

---

## 5. Single cheapest highest-value fix (verified line-by-line as the binding fix)

**Add the PR95 bilinear-skip + terminal refine to BOTH decoder atoms, by importing the NEW canonical HF-residual primitive — then (secondarily) flip the trained objective off recon-MSE.**

Concretely, the binding edit (Mistake-A) is at **two sites only**:
- `_DsUpBlock.forward` (architecture.py:77–78): `shuffle(act(dsc(x)))` → `sin(PixelShuffle(conv(x)) + skip(bilinear_up(x)))`, with a 1×1 channel-match `skip` when `in_ch ≠ out_ch`; then add the terminal `h = h + 0.1·sin(refine(h))` (dilated conv) before the RGB heads (after architecture.py:262).
- `_DsUpBlockMLX.__call__` (mlx_renderer.py:211–214): the identical change (the MLX bilinear-2x helper `bilinear_resize2x_align_corners_false_nhwc` already exists, imported at mlx_renderer.py:159–161 — so the skip carrier is "free" infra-wise).

I verified the operator's suggested primitive path: `tac.framework_agnostic.canonical_kernels.bilinear_skip_residual_canonical` (canonical_kernels.py:652). **CONFIRMED it EXISTS and is the right primitive:** its signature is `bilinear_skip_residual_canonical(shuffled, identity, *, sin_frequency=1.0, backend)` and its body computes exactly the PR95 per-block residual `sin(w·(shuffled + identity))` (NUMPY path :670; MLX backend present; tested test_canonical_kernels.py:346–397). It is framework-agnostic, **fails closed on the channel-match shape bug class** (raises on `shuffled.shape != identity.shape`, forcing the carrier to 1×1 channel-match the skip — exactly the bug class that would otherwise bite), and its own docstring cross-references the HiNeRV H4 spectral-bias finding (PR95 uses implicit w≈1.0 on the summed residual vs the skip-free w=30 trap). So the binding edit is a verified IMPORT, not a hand-roll: at each block, compute `identity = skip_1x1(bilinear_2x(x))` then `h = bilinear_skip_residual_canonical(PixelShuffle(conv(x)), identity, sin_frequency=~1.0)` — note this also FIXES the w=30 spectral-bias trap (sister H4/V2) for free, since PR95 applies w≈1 on the summed residual. Add the terminal `h = h + 0.1·sin(refine(h))` separately. **The binding variable is the residual/skip + refine, not the VQ.** Match it in the PyTorch oracle AND the MLX renderer to preserve export parity (mlx_renderer.py docstring 33–48 invariants); the primitive's NUMPY+MLX backends + a PyTorch path make a single-source-of-truth atom feasible.

Secondary (Mistake-B, only after the skip is in): in the MLX train config, anneal `recon_weight` (loss.py:3099) toward a small anchor and set the frozen-SegNet direct-live margin weight > 0 (the `segnet_direct_live_*` knobs exist, loss.py:68–98) — but per §0/§3 this alone did NOT lift d_seg, so it is the second move, not the first.

**Falsifiable prediction (research proposal; promotion requires byte-closed archive + paired CPU/CUDA per CLAUDE.md):** with the skip+refine added and the SAME short curriculum, the live MLX render's PSNR breaks the plateau (expect ≥ 28 dB) and **d_seg drops from ≈0.5 toward < 0.1** `[macOS-MLX research-signal]`. If d_seg stays ≈0.5 with the skip added AND the scorer objective on, then H1/Mistake-A is falsified for this carrier and a deeper cause (codebook collapse V3, or export-binding) dominates → escalate.

---

## 6. Honest scope / limits

- **Verified by reading every line (file:line cited above):** all 9 carrier files (architecture / mlx_renderer / score_aware_loss / archive / archive_candidate / inflate / section_value / competitiveness_gate / `__init__`) + both test files; the shared MLX harness loss defaults (`recon_weight=1.0`, recon = mse_0+mse_1); the frozen-scorer dispatch; git provenance (`b56f24bc1` birth, single-commit `--follow`); the lane registry rows; the L0/L1 design+landing memos.
- **Verified by telemetry (`[macOS-MLX research-signal]`, NOT a score):** d_seg≈0.5 / "dark mean-field image" with real teachers bound, and competitiveness verdict `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`, carried from `pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md:96` + the 2026-06-02 codex audits it cites. I did not re-run the renderer.
- **INFERRED (mechanism argument, not measured here):** that adding the skip+refine moves d_seg (the d_seg/PSNR prediction is a falsifiable hypothesis); that the skip-free decoder mean-fields all z_q to ~one frame (consistent with the d_seg≈0.5 receipt + the residual-learning argument, not isolated by ablation here); MLX↔PyTorch numeric export parity (declared 1:1; not re-measured).
- **NOT done (out of scope for a read-only audit):** no training, no GPU, no paid dispatch, no edits to any carrier source or `upstream/`, no byte-closed archive built, no `[contest-CPU]`/`[contest-CUDA]` score produced. I did NOT open `tac.framework_agnostic.canonical_kernels.bilinear_skip_residual_canonical` (named in the prompt; flagged in §5 as a verify-before-use).
- **Authority:** everything here is `[macOS-CPU advisory]` / `mechanism_update_eligible`. It updates next-experiment routing; it does NOT promote, rank, kill, or close any lane (per CLAUDE.md "Forbidden premature KILL" + "Meta-Lagrangian/Pareto solver").

### One-line verdict
**pact_nerv_vq is a GENUINE VQ carrier (real STE/EMA/commitment + real charged-and-consumed indices + correct PyTorch frozen-scorer loss + mature fail-closed competitiveness gate + genuine behavioral tests + emits a valid scorable archive whose RATE is solved) whose expected failure is class-(B) INCOMPLETENESS (~85%): a skip-free `PixelShuffle(sin(w·conv))` decoder, batch-born as an L0 SCAFFOLD (`b56f24bc1`), never matured to the documented PR95-parity bilinear-skip+refine, with no design-memo justification — empirically pinned at d_seg≈0.5 even with real scorer teachers bound, which proves the decoder (not the objective) is the binding constraint; the single binding fix is the PR95 bilinear-skip + terminal refine at architecture.py:77–78 and mlx_renderer.py:211–214 (preferably via the canonical HF-residual primitive, verify-before-use), with the recon-MSE→frozen-scorer objective flip as a strict second move.**
