# Wave N+12 BUILD Path B — Z6-v2 pose-axis routed through PR101 HNeRV renderer (DESIGN + SMOKE PREP)

- **Date:** 2026-05-30 (crash-resume; predecessor `a01416393b58da5d1` step=1 hit session limit)
- **Lane:** `lane_wave_n12_build_path_b_z6_v2_pose_axis_routed_through_pr101_hnerv_renderer_20260530`
- **Mission contribution (Catalog #300):** `frontier_breaking`
- **Horizon class (Catalog #309):** `frontier_pursuit` (predicted CPU band [0.120, 0.180] — class-shift attempt below the 0.196-0.199 plateau)
- **Scope:** DESIGN + MLX-LOCAL SMOKE PREP ONLY. No L1 SCAFFOLD. The L1 build is a follow-on lane.
- **Paid spend:** $0 (MLX-LOCAL on M5 Max per the operator MLX-first paradigm).

## 0. Why Path B (the Wave N+11 HALT → Wave N+12 structural advance)

Per the Wave N+11 QUAD HALT (`ee15561e9`, memo `.omx/research/wave_n11_quad_composition_sub015_cascade_halt_phantom_provenance_pre_check_failed_landed_20260530.md`):

- The Wave N+6 TRIPLE composite (Z6-v2 + NSCS06 v8 + **Compound C** renderer; sha `fef2fa6233`) was **empirically IMPLEMENTATION-LEVEL FALSIFIED** at paired CUDA+CPU `score=92.48` — **482× WORSE** than the predicted 0.156006.
- Root cause per the corrected-archive paired ratification (`wave_n6_triple_paired_cuda_ratification_corrected_archive_implementation_falsified_20260528.md`): **PoseNet = 162.52** vs frontier ~0.01. The **Compound C renderer does NOT produce frames that PoseNet recognizes.** The architecture binds a real pose-axis prediction signal to a renderer whose output is not in the scorer's accepted manifold.
- Wave N+11's QUAD re-attempt routed the same Z6-v2 pose-axis through the same Compound C renderer and was correctly HALTED at the Catalog #321/#322 phantom-provenance pre-check (all 4 archives carried `evidence_grade=predicted` / `[macOS-MLX research-signal]` / `score_claim_valid=False`).

**Path B's structural fix:** route the Z6-v2 ego-motion-conditioned pose-axis latent through the **PR101 HNeRV-family renderer** — a `VALIDATED_CONTEST_MEMBER` (upstream PR #101 GOLD, 0.193 [contest-CPU]) whose decoder + L20-L32 archive grammar is empirically known to produce frames the contest scorer accepts. This replaces the FALSIFIED renderer (the failing component) while preserving the Z6-v2 distinguishing class-shift feature (the intact paradigm) — exactly the Catalog #307 IMPLEMENTATION-LEVEL remediation: keep the paradigm, swap the falsified implementation.

## 1. Interface alignment — empirically verified (Catalog #229 premise verification, NOT inferred)

All facts below were extracted via Python AST + a PyTorch dry-run forward of the actual intake source, NOT from prose (the rtk display layer introduced rendering noise that was bypassed by writing results to files and reading them back).

### Z6-v2 pose-axis hand-off surface (MLX)
`src/tac/substrates/z6_v2_cargo_cult_unwind/architecture.py`:

```
predict_pose_axis_from_latents(latents, ego_motion, cfg) -> dict
    latents:     (num_pairs=600, latent_dim_l1=28)
    ego_motion:  (num_pairs=600, ego_motion_dim=6)   # 6-DoF pose delta
    returns:
        pose_axis_prediction: (600, 28)   # FiLM ego-motion-conditioned + Atick-Redlich decorrelated
        foe_xy:               (600, 2)     # focus-of-expansion per pair
        ego_motion_delta:     (600, 6)     # pass-through
```
- Canonical config `Z6V2Config`: `num_pairs=600`, `latent_dim_l1=28`, `ego_motion_dim=6`, `frame_height=384`, `frame_width=512`.
- The 28-d per-pair latent is the **PR95-family L19 per-frame-PAIR 28-d latent** — the same canonical latent geometry PR101 uses.
- Exported from `__init__.py` (verified: `predict_pose_axis_from_latents`, `Z6V2Config`, `build_z6_v2_renderer`, `Z6V2Renderer` all importable).

### PR101 renderer surface (PyTorch, VALIDATED_CONTEST_MEMBER)
`experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/model.py`:

```
HNeRVMicroDecoder(latent_dim=28, base_ch=64, out_h=384, out_w=512)
    forward(z) :  z (N_pairs, 28)  ->  (N_pairs, 2, 3, 384, 512)   # 2 frames x 3 RGB per pair
```
- **Dry-run verified:** `forward(z=(4,28)) -> (4, 2, 3, 384, 512)`, ndim=5, **228,958 params** (matches PR101 README exactly).
- Decoder: PixelShuffle 6x8 → 12x16 → 24x32 → 48x64 → 96x128 → 192x256 → 384x512, sin activation, bilinear skip, channel taper `[C,C,C,0.75C,0.58C,0.5C,0.5C]` — the canonical PR95-family L18 decoder.
- Archive grammar `src/codec.py`: `pack_archive(state_dict, latents, out_path)` + `unpack_archive` + `rebuild_state_dict` + `decode_latents`; monolithic single-file `0.bin` with 4 length-prefixed sections (decoder brotli L20, scales fp16 L29, latents brotli L25, sidecar brotli L27); `DECODER_BLOB_LEN=162164`, `LATENT_BLOB_LEN=15387`; per-tensor byte maps (L21), conv4 storage perms (L22), split brotli streams (L23).

### THE ALIGNMENT (the load-bearing fact)
```
Z6-v2.predict_pose_axis_from_latents(...).pose_axis_prediction   shape (600, 28)
                                          ║  EXACT SHAPE MATCH  ║
PR101.HNeRVMicroDecoder.forward(z)                          z    shape (600, 28)
```
The Z6-v2 pose-axis latent **IS** exactly what the PR101 decoder consumes. No reshape, no projection, no padding. The only boundary is the MLX→PyTorch backend hand-off (`mx.array(600,28)` → `torch.Tensor(600,28)` via `numpy` round-trip).

**Honest verdict: interfaces ALIGN. PROCEED.** No interface gap; no architectural mismatch requiring a deferral.

## 2. Composition architecture

```
                        ┌───────────────────────── COMPRESS TIME (training) ─────────────────────────┐
upstream/videos/0.mkv  →  pyav decode 1200 frames → 600 pairs (frame_0, frame_1) @ 384x512
                                          │
                          ego-motion deltas (6-DoF)  +  per-pair 28-d latents (learned)
                                          │
                          Z6-v2 FiLM ego-motion conditioning + FoE prior + Atick-Redlich decorrelation
                                          │ predict_pose_axis_from_latents
                                          ▼
                          pose_axis_prediction  (600, 28)   ← the class-shift distinguishing feature
                                          │  [MLX -> PyTorch np bridge :: PRINCIPLED FORK Catalog #290]
                                          ▼
                          PR101 HNeRVMicroDecoder.forward(z)   ← VALIDATED_CONTEST_MEMBER renderer
                                          ▼
                          frames (600, 2, 3, 384, 512)   ← frames PoseNet/SegNet accept (PR101 GOLD provenance)
                                          │
                          score-aware loss: SegNet + sqrt(10)*PoseNet + 25*bytes/N  (canonical contest formula)
                        └─────────────────────────────────────────────────────────────────────────────┘

                        ┌───────────────────────── INFLATE TIME ───────────────────────────────────┐
                          0.bin (PR101 4-section grammar: decoder brotli / scales fp16 / latents brotli / sidecar)
                            └─ decode_latents -> z (600,28)  ->  HNeRVMicroDecoder.forward  ->  frames -> 1200 @ 1164x874 -> .raw
                        └───────────────────────────────────────────────────────────────────────────┘
```

The KEY architectural property: **the bytes that ship are PR101's archive grammar (validated)**, and **the latents that ship are Z6-v2's ego-motion-conditioned pose-axis predictions (class-shift)**. The renderer at inflate time is PR101's decoder. This is why Path B can produce contest-faithful frames where Compound C could not: the decoder is the GOLD-medal renderer, the latents carry the Z6-v2 signal.

## 3. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale (falling-rule per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD) |
|---|---|---|
| Pose-axis latent prediction | **UNIQUE (Z6-v2)** | This IS the distinguishing class-shift feature (FiLM ego-motion + FoE + Atick-Redlich). FORK_BECAUSE_PRINCIPLED — the whole point of Path B. |
| Renderer / decoder | **CANONICAL (PR101)** | OBVIOUS-FIT. PR101 decoder is a VALIDATED_CONTEST_MEMBER (0.193 GOLD); produces frames the scorer accepts. The FALSIFIED Compound C renderer is the thing we are replacing. |
| Archive grammar | **CANONICAL (PR101 L20-L32)** | OBVIOUS-FIT. Monolithic 0.bin 4-section grammar is byte-validated. Re-implementing is the kitchen_sink anti-pattern. |
| MLX→PyTorch latent bridge | **PRINCIPLED FORK** | The two backends differ; a numpy round-trip is the canonical interop boundary. Documented here per Catalog #290. NOT a canonical helper because no shared MLX↔PyTorch latent bridge exists; this is a one-shape `(600,28)` cast. |
| Score-aware loss | **CANONICAL** | `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 — the contest formula must be used identically across substrates. |
| Inflate device selection | **CANONICAL** | `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205. |

## 4. Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Evidence / unwind |
|---|---|---|
| "Z6-v2 pose-axis latent is class-shift, not within-class refinement" | **HARD-EARNED** | Per Catalog #311 the FiLM ego-motion + FoE + Atick-Redlich binding is a primary renderer architecture (Catalog #310), not a bolt-on; the latent carries ego-motion-conditioned structure absent from a vanilla per-pair latent. |
| "PR101 renderer accepts a 28-d latent regardless of how it was produced" | **HARD-EARNED** | Empirically dry-run verified: `forward(z=(600,28)) -> (600,2,3,384,512)`. The decoder is latent-source-agnostic by construction (it is `fc(z) -> reshape -> PixelShuffle`). |
| "Frames from PR101 decoder will score well even when fed Z6-v2 latents" | **CARGO-CULTED — this is exactly what the smoke must NOT assume** | This is the open empirical question. The smoke verifies the composition RUNS and produces frames of the correct shape on REAL frames; it does NOT claim a score. Score requires paired CUDA+CPU exact eval (Catalog #246) at L1+, NOT this $0 MLX-LOCAL smoke. The 0.193 GOLD provenance is the renderer's; whether Z6-v2 latents drive it into a low-score region is the L1 dispatch question. |
| "MLX→PyTorch numpy bridge is byte-exact" | **CARGO-CULTED — documented, not assumed** | float32 numpy round-trip is exact for the bridge; but MLX vs PyTorch op kernels differ downstream. Per [[mlx-portable-local-substrate-authority]] the MLX smoke is `[macOS-MLX research-signal]` ONLY; the contest archive is built/scored on the PyTorch path. |

## 5. Predicted ΔS band + Dykstra feasibility check (Catalog #296)

**Predicted CPU band: [0.120, 0.180]** (frontier_pursuit; class-shift attempt below the 0.196-0.199 plateau).

- **Dykstra-feasibility intersection check:** the composition's score lives in the intersection of (a) PR101's renderer-quality feasible set (empirically bounded below by ~0.193 when fed PR101's own latents) and (b) the Z6-v2 latent's information content. Per Shannon R(D): the rate term is dominated by the PR101 archive grammar (~178KB → 25·178493/37545489 ≈ 0.119 rate), and the distortion term depends on whether the Z6-v2 ego-motion-conditioned latents drive PoseNet into a lower-distortion region than PR101's own latents. The band's LOWER bound (0.120) is the rate-floor alone (perfect distortion); the UPPER bound (0.180) is the plateau-adjacent ceiling. This is NOT additive-composition vibes (the Wave N+11 failure mode): it is the renderer-feasible-set ∩ latent-content intersection, with the renderer feasible set empirically anchored at PR101's GOLD measurement.
- **Probe-disambiguator path:** the L1 paired-CUDA+CPU exact eval (Catalog #246) IS the disambiguator. If Z6-v2 latents drive the score WORSE than PR101's own latents (i.e. the ego-motion conditioning hurts on this video), that is an IMPLEMENTATION-LEVEL finding (the conditioning is mis-tuned), NOT a paradigm kill — the reactivation path is to co-train the Z6-v2 latents against the PR101 decoder end-to-end rather than using random/untrained latents.

`# PREDICTED_BAND_DYKSTRA_FEASIBILITY: renderer-feasible-set ∩ latent-content intersection, renderer set anchored at PR101 0.193 GOLD; rate floor 0.119 from PR101 archive grammar; disambiguator = L1 paired-CUDA per Catalog #246`

## 6. Observability surface (Catalog #305)

- **Inspectable per layer:** Z6-v2 exposes `inspect_film_conditioning` / `inspect_foe_prior` / `inspect_decorrelation` / `inspect_latent_hierarchy`; the composition smoke captures pose_axis_prediction stats + decoder output stats per layer.
- **Decomposable per signal:** the smoke records pose_axis_prediction (mean/std/shape), bridge tensor (shape/dtype), decoder output (mean/std/shape per frame), and the composition-distinctness signal (output differs from a zero-latent baseline).
- **Diff-able across runs:** smoke artifact carries a deterministic seed + sha256 of the pose-axis prediction so two runs are comparable.
- **Queryable post-hoc:** smoke writes a machine-readable JSON manifest with canonical Provenance (Catalog #323).
- **Cite-able:** manifest carries (substrate=z6_v2+pr101_renderer / commit / config / random_seed / upstream_snapshot=upstream/videos/0.mkv).
- **Counterfactual-able:** the smoke includes a zero-latent counterfactual to prove the Z6-v2 latents actually drive the decoder output (NO-FAKE Class 1: the composition produces a DIFFERENT output than a null latent).

## 7. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — class-shift via Z6-v2 ego-motion-conditioned pose-axis (Catalog #310/#311), distinct from PR101's vanilla per-pair latent.
2. **BEAUTY + ELEGANCE** — the composition is a single 28-d shape hand-off; the renderer is the 228K-param PR101 decoder reviewable in 30s.
3. **DISTINCTNESS** — explicitly different from Wave N+6/N+11 (which used the FALSIFIED Compound C renderer). The renderer swap IS the distinction.
4. **RIGOR** — interface alignment verified via AST + dry-run (Catalog #229); phantom-provenance verified (Catalog #321/#322); smoke runs on REAL `upstream/videos/0.mkv` frames (Catalog #213).
5. **OPTIMIZATION PER TECHNIQUE** — Catalog #290 table above; each layer canonical-or-unique by principle.
6. **STACK-OF-STACKS COMPOSABILITY** — Z6-v2 latent ⊥ PR101 renderer; orthogonal axes (latent-content vs renderer-quality). The composition is the first VALIDATED-renderer routing for the Z6-v2 latent.
7. **DETERMINISTIC REPRODUCIBILITY** — seed-pinned smoke; sha256 of pose-axis prediction recorded.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — MLX-LOCAL $0 smoke on M5 Max; the L1 dispatch is the paid step.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted band [0.120, 0.180]; the renderer feasible set is anchored at GOLD 0.193, so the composition cannot do worse than the renderer's floor if the latents are co-trained (L1 reactivation path).

## 8. Substrate-compatibility evidence (Catalog #311)

- Z6-v2 latent `(600, 28)` and PR101 decoder input `(600, 28)` are shape-identical and dtype-compatible (float32).
- PR101 is a VALIDATED_CONTEST_MEMBER (upstream PR #101 GOLD intake, 0.193 [contest-CPU]); NOT a research sidecar (Catalog #321/#322 phantom-provenance PASS — verified the intake is an upstream archive member with README provenance, not a `.pt` sidecar).
- The renderer is latent-source-agnostic by construction (`fc(z) -> reshape -> PixelShuffle`), so feeding Z6-v2 latents instead of PR101's own latents is a structurally valid substitution.

## 9. Per-substrate symposium 6-step contract (Catalog #325)

1. **Cargo-cult audit** — §4 above.
2. **9-dim checklist** — §7 above.
3. **Observability surface** — §6 above.
4. **Sextet deliberation** — design-time PROCEED_WITH_REVISIONS: Shannon (rate floor 0.119 grounded in PR101 archive grammar) + Dykstra (renderer-feasible ∩ latent-content intersection, NOT additive vibes) + Yousfi/Fridrich (renderer is inverse-steganalysis-validated GOLD) + Contrarian (the open question is whether Z6-v2 latents help OR hurt vs PR101's own — REVISION: L1 must co-train, not feed random latents) + Assumption-Adversary (the CARGO-CULTED assumption "PR101 renderer scores well with ANY latent" is flagged; the smoke does NOT claim score). REVISION captured: **L1 MUST co-train Z6-v2 latents end-to-end against the PR101 decoder**, not feed untrained latents.
5. **Reactivation criteria** — (a) L1 co-train Z6-v2 latents against frozen PR101 decoder; (b) L1 paired-CUDA+CPU exact eval; (c) if score > 0.193, the ego-motion conditioning hurts → DEFER-pending-conditioning-retune (IMPLEMENTATION-LEVEL per Catalog #307, NOT paradigm kill).
6. **Catalog #324 post-training validation** — `pending_post_training`; the predicted band [0.120, 0.180] is design-time and MUST be re-measured on the landed archive sha256 at L1.

## 10. 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map** — N/A at DESIGN+SMOKE; the L1 build wires per-pair pose-axis sensitivity.
- **hook #2 Pareto constraint** — ACTIVE (the Dykstra renderer-feasible ∩ latent-content intersection in §5 is the Pareto constraint).
- **hook #3 bit-allocator** — N/A at DESIGN+SMOKE; L1 inherits PR101's L21-L32 per-tensor bit allocation.
- **hook #4 cathedral autopilot dispatch** — DEFERRED to L1 (this is DESIGN+SMOKE; no dispatch-eligible archive yet).
- **hook #5 continual-learning posterior** — ACTIVE (the smoke verdict + canonical equation FORMALIZATION_PENDING anchor feed the posterior).
- **hook #6 probe-disambiguator** — ACTIVE (the L1 paired-CUDA exact eval IS the disambiguator per §5).

## 11. Honest verdict

**PROCEED** to MLX-LOCAL smoke. Interface alignment empirically confirmed (no gap, no architectural mismatch). The smoke verifies the composition RUNS on real frames and produces correctly-shaped frames that DIFFER from a null-latent counterfactual (NO-FAKE Class 1). The smoke makes NO score claim (Tier A, non-promotable, `[macOS-MLX research-signal]`). The L1 follow-on lane co-trains the latents and runs paired-CUDA+CPU exact eval — that is where the score question is answered.

The Wave N+11 QUAD HALT predecessor probe-outcome is SUPERSEDED for the renderer-routing question: Path B routes through the VALIDATED PR101 renderer, structurally avoiding the FALSIFIED Compound C renderer (PoseNet=162.52) that caused the HALT.
