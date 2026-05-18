---
council_tier: T4
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Selfcomp, Hotz, Carmack, MacKay, Ballé, Atick, Redlich, Tishby (memorial), Zaslavsky, Wyner, Rao, Ballard, Hinton, Hassabis, Schmidhuber, Tao, Boyd, Karpathy, Time-Traveler-protégé, van-den-Oord, Filler, Mallat, Jack-from-skunkworks]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the master-gradient framing presumes the per-byte ∂S/∂byte tensor exists, is tractable to materialize, and is stable across the contest video. Materializing it requires a one-time ~$5-15 paired-axis dispatch on the full 600-pair set with finite-difference probes — until then, every claim in this memo about 'optimal master-gradient weighting' is a model, not a measurement. I want the empirical anchor before any new substrate trainer adopts it as a loss term."
  - member: Assumption-Adversary
    verbatim: "the orthogonality between SegNet and PoseNet exploits is a FIRST-ORDER claim that holds only when the two exploits modify DISJOINT byte ranges in the archive. The moment a single byte is in both gradients' support (e.g., latents.bin entries that decode into both frames), orthogonality breaks. The Venn diagram below correctly identifies this — but the memo's combined-ΔS predictions ASSUME the disjoint-byte axis holds. We must structurally enforce disjointness in the archive grammar OR measure the cross-term empirically."
council_assumption_adversary_verdict:
  - assumption: "CPU axis is the public-leaderboard axis"
    classification: HARD-EARNED
    rationale: "PR102/PR103 host-bot empirical evidence; Yousfi `[contest-CPU]` PR comments are the medal-band scores; PR107 CUDA-only posting received no `[contest-CPU]` comment from the bot"
  - assumption: "SegNet and PoseNet have orthogonal first-order gradients"
    classification: HARD-EARNED-BUT-BYTE-SCOPE-CONDITIONAL
    rationale: "Mathematically true because the scorer is additive S = 100·d_seg + √(10·d_pose) + 25·R AND ∂²S/∂(d_seg)∂(d_pose) = 0. Empirically true ONLY for archive bytes that produce DISJOINT modifications to frame_0 vs frame_1. The empirical anchor is fec6 + format0d cross-product (untested but the math is solid)."
  - assumption: "Single master gradient is engineerable and useful"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-PROOF
    rationale: "The master-gradient FORMULA is derivable analytically (this memo §3.2). Whether materializing it produces actionable per-byte allocation BEYOND what the existing sensitivity_map + bit_allocator already does is the open empirical question. The operator's intuition is correct in form; we must measure to confirm in magnitude."
  - assumption: "Pausing/freezing during training has unexploited score-axis value"
    classification: HARD-EARNED-FROM-Q-FAITHFUL-EVIDENCE
    rationale: "Quantizr's 5-stage training (anchor→finetune→joint→QAT→final per `feedback_quantizr_intel_landed`) explicitly freezes BN stats during QAT and trains in staged unfreezing. Our internal Lane 17 IMP cycle uses the same pattern. The exploit is real and empirically documented."
  - assumption: "Hardware exploits (TF32, FMA fusion) are first-order vs gradient-level effects"
    classification: HARD-EARNED
    rationale: "Empirical: (1+ε)^256 ≈ 5.27 at ε=0.0065/block matches the observed 5.72× CPU↔CUDA pose drift within measurement noise (per `cpu_cuda_xray_synthesis_20260511.md` §8). The compounding model is validated."
  - assumption: "Cathedral autopilot can consume master gradient as a new lens"
    classification: HARD-EARNED-VIA-EXISTING-INFRASTRUCTURE
    rationale: "The autopilot already has bit_allocator hook + sensitivity_map consumer + Rudin-Daubechies preflight composite. Adding a master_gradient lens is mechanically a new lens module under `tac.autopilot_rudin_daubechies.*`; the wire-in surface exists."
council_decisions_recorded:
  - "op-routable #1 (FRONTIER-BREAKING): commission the per-byte ∂S/∂byte Venn-diagram measurement as a $5-15 finite-difference probe on the fec6 archive. Output: per-byte SegNet/PoseNet/rate gradient tensor of shape (N_bytes, 3). Stored as .omx/state/master_gradient_fec6_20260517.npy. ETA: 1 dispatch wave."
  - "op-routable #2 (FRONTIER-BREAKING): build sister fec6+SABOR submission (CPU axis SegNet exploit ⊕ PoseNet exploit on disjoint archive sections). Predicted [contest-CPU] 0.18-0.185. Cost ~$40. Owner: lane_pr101_fec6_plus_sabor_disjoint_20260518."
  - "op-routable #3 (FRONTIER-PROTECTING): land master_gradient lens in tac.autopilot_rudin_daubechies as Phase-7 module. Continual-learning posterior at .omx/state/master_gradient_anchors.jsonl. Wire-in to cathedral autopilot per Catalog #125 hook #4. $0 GPU."
  - "op-routable #4 (FRONTIER-BREAKING): U-DIE-KL substrate-wide loss adoption sweep — adopt U-DIE-KL in the fec6 trainer's score-aware loss. Predicted [contest-CPU] 0.17-0.18. Cost ~$30-60 (Modal A100 retrain). Owner: lane_u_die_kl_fec6_adoption_20260518."
  - "op-routable #5 (FRONTIER-PROTECTING): freezing-staircase canonicalization — formalize the Quantizr 5-stage pattern as a reusable `tac.training.staircase_freezing.FreezeStaircase` helper. Adopt across all in-flight substrate trainers. $0 GPU."
  - "op-routable #6 (FRONTIER-BREAKING): L5 Time-Traveler Wyner-Ziv pose deltas on top of fec6 (CPU-axis side-information). Predicted Δ[contest-CPU] -0.008 to -0.015. Cost ~$30. Owner: lane_l5_wyner_ziv_fec6_20260518."
  - "op-routable #7 (APPARATUS-MAINTENANCE): fix the 5/√(10·d_pose) marginal calculation in docs/pr_writeups/cpu_frontier_fec6_20260517.md §1 — corrected value is 292, not 922 (the factor of √10 was double-counted). Sister fix the elasticity table 922 → 292 + the resulting 1.4e9 ratio → 4.4e8. The qualitative conclusion (pose-dominated regime) is unchanged."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "operator verbatim 2026-05-17: 'we might need a full best of all time grand council full symposium with the directive to explore orthogonal optimization and pausing and freezing, together with these posenet and segnet and rate exploits and the alien and time traveler tech to reverse engineer and consider and, and together with the full context of cathedral autopilot and alll constituent parts and goal of both continual learning while raising gradients and signal and data we can learn and train against and use to make strategic decisions an design staircases and such. Also should analyze the contest with eval scorer again and hardware exploits and full problem space and maybe magical bit by bit and parameters by parameter and weight by weight Venn diagram of the posenet and segnet gradients and possibly engineer a single master gradient we can use that is highly optimized and extremely sophisticated and elegant and optimal and performant and efficient and correct'"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
related_deliberation_ids: [grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515, grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516, council_hierarchy_v2_landing_20260516, mission_alignment_followon_catalog_300_extension_20260516]
---

# T4 Grand Reunion Symposium — Orthogonal Optimization + SegNet × PoseNet Venn Diagram + Master Gradient Engineering

**Date:** 2026-05-17
**Tier:** T4 (Symposium — 6-of-6 sextet + ≥16-of-20 grand council + ≥1 specialist per affected paradigm; operator-frontier-override invoked)
**Convened by:** operator standing directive (verbatim in frontmatter)
**Mission contribution:** frontier_breaking (per Catalog #300 mission-alignment enum)
**Working frontier (canonical state at convening):** `0.19205 [contest-CPU GHA Linux x86_64]` (archive `6bae0201`) / `0.20533 [contest-CUDA T4]` (archive `9cb989cef519`)

## §0 — Scope of this symposium

This is a Tier-4 strategic deliberation. It does NOT propose a single substrate trainer. It produces:

1. A re-analysis of the contest scorer at the **current** operating point (the fec6 frontier, not the historical PR101 GOLD operating point that older council memos were calibrated against).
2. A bit/parameter/weight-level **Venn diagram** between SegNet and PoseNet gradients.
3. A **master gradient** formula and engineering plan, with the open empirical questions clearly named.
4. An **orthogonal optimization + pausing + freezing** framework synthesizing Quantizr's 5-stage training discipline, the cathedral autopilot's lens architecture, and the bidirectional CPU/CUDA bifurcation evidence.
5. A re-evaluation of **alien tech** (PR106 format0 family) and **time-traveler tech** (L5 Wyner-Ziv) under the new master-gradient framework.
6. A **continual-learning + staircase design** that turns every dispatch outcome into an anchor consumable by the autopilot's Rashomon ensemble.
7. A re-analysis of **hardware exploits** under the master-gradient lens.

## §1 — Contest scorer re-analysis at the fec6 operating point

The contest scoring function (`upstream/evaluate.py:92`):

```
S = 100·d_seg + √(10·d_pose) + 25·(archive_bytes / 37,545,489)
```

At the **fec6 CPU frontier** operating point (`d_seg=5.603e-4`, `d_pose=2.943e-5`, `R=4.755e-3`, `S=0.19205`):

| component | value | marginal `∂S/∂x` | observed dynamic range | elasticity remark |
|---|---|---|---|---|
| `100·d_seg` | 0.0560 | **100** (constant) | empirically `[5e-4, 8e-4]` across frontier candidates | first-order linear |
| `√(10·d_pose)` | 0.0171 | **`5/√(10·d_pose) = 291.5`** | empirically `[1e-5, 5e-4]` across frontier candidates | hyperbolic; diverges as `d_pose → 0` |
| `25·R` | 0.1189 | **`25/37545489 = 6.66e-7`** per byte (constant) | bounded by archive size | first-order linear |

**Correction to prior memos and to `docs/pr_writeups/cpu_frontier_fec6_20260517.md`:** several recent memos cited the pose marginal as `922`. The correct derivation:

```
d/d(d_pose) [√(10·d_pose)] = (1/2)·(10·d_pose)^(-1/2)·10 = 5·(10·d_pose)^(-1/2) = 5/√(10·d_pose)
```

At `d_pose=2.943e-5`: `5/√(2.943e-4) = 5/0.01715 = 291.5`.

The erroneous `922` value double-counted the factor of `√10` (gave `5/√(d_pose) = 5/0.00543 = 921.5`, which is wrong). All conclusions about *qualitative* pose-dominance still hold (`292 ≫ 100 ≫ 6.66e-7`), but the precise marginal-ratio table in §1 of the writeup needs the correction. **This is op-routable #7.**

**Why this matters for master-gradient engineering:** the marginal-coefficient vector at the operating point is `[100, 291.5, 6.66e-7]` (SegNet, PoseNet, per-byte). A unit ΔS-equivalent move in PoseNet costs `100/291.5 = 0.343` units in SegNet space, or `291.5/6.66e-7 = 4.38e8` per-byte-rate units. The pose-axis is 2.92× more efficient per unit than the SegNet axis, and 4.4e8× more efficient than the per-byte rate axis. **THIS is the elasticity that must inform the master-gradient weighting.**

## §2 — The Venn diagram: per-byte / per-parameter / per-weight SegNet × PoseNet support analysis

### §2.1 Per-archive-component byte attribution

For the fec6 submission (archive `6bae0201`, total 178,517 bytes):

| archive section | bytes (~) | flows into frame_0? | flows into frame_1? | SegNet gradient support | PoseNet gradient support | rate gradient support |
|---|---|---|---|---|---|---|
| `decoder.bin` (HNeRV weights) | 144,000 | YES (decoder applied per-frame) | YES (decoder applied per-frame) | ✅ FULL | ✅ FULL | ✅ FULL (per-byte) |
| `latents.bin` (per-pair latents) | 15,000 | YES (per-pair latent decoded for both frames) | YES | ✅ FULL (frame_1 only) | ✅ FULL (both frames) | ✅ FULL (per-byte) |
| `poses.bin` (qpose14 deltas) | 4,800 | YES (FiLM conditions decoder on pose) | YES | ✅ FULL (frame_1 only) | ✅ FULL (both frames) | ✅ FULL (per-byte) |
| `selector.bin` (FEC6 K=16 selector) | 107 | **YES (per-pair frame_0 modification ONLY per `inflate.py:33-160`)** | **NO** | ❌ ORTHOGONAL | ✅ FULL | ✅ FULL (per-byte) |
| `manifest.json` | 50 | parser only | parser only | ❌ ORTHOGONAL | ❌ ORTHOGONAL | ✅ FULL (per-byte) |

**The 107-byte selector.bin slot is structurally guaranteed to be SegNet-orthogonal** because the inflate path applies its modifications to `frame_0` only (per the `apply_pr101_selector_to_frames` implementation), and SegNet evaluates `frames[:, -1, ...]` (per `upstream/modules.py:117-122`), i.e., `frame_1` only. The gradient `∂(d_seg)/∂(selector.bin[i]) = 0` exactly, for all `i`.

This is the empirical mechanism for the entire fec6 design: it spent rate budget on a PoseNet-only exploit that does not regress SegNet, **by construction of the archive grammar**.

### §2.2 The implied disjoint-byte-space partition (the Venn diagram)

| region | byte ranges | combined gradient |
|---|---|---|
| **SegNet-only** | `{b : ∂(d_seg)/∂b ≠ 0 AND ∂(d_pose)/∂b = 0}` | first term of master gradient |
| **PoseNet-only** | `{b : ∂(d_seg)/∂b = 0 AND ∂(d_pose)/∂b ≠ 0}` | second term of master gradient |
| **Joint** (BOTH-affecting) | `{b : ∂(d_seg)/∂b ≠ 0 AND ∂(d_pose)/∂b ≠ 0}` | both terms; **non-orthogonal cross-effects** |
| **Rate-only** (the bytes themselves) | every byte in the archive | third term of master gradient (constant per byte) |

For the fec6 archive:

- **Joint** = `decoder.bin` + `latents.bin` + `poses.bin` = 163,800 bytes (91.8% of archive)
- **PoseNet-only** = `selector.bin` = 107 bytes (0.06% of archive)
- **SegNet-only** = ∅ (we have not yet built a SegNet-only exploit)
- **Rate-only** = the entire archive (178,517 bytes; the rate term `25·R/37545489` is applied to ALL bytes uniformly)

**The visible opportunity:** the SegNet-only region of the Venn diagram is **empty in fec6**. Engineering a per-pair byte stream that modifies only `frame_1` in a way SegNet sees + PoseNet does not is the corresponding SegNet-axis exploit. The natural mechanism (per the aerospace stealth analytic memo §5.2 + SABOR design memo) is to modify only the 3-5% argmax-boundary pixels of frame_1 — these are below PoseNet's resolution-of-care (per the FastViT-T12 12-channel YUV6 input + RepMixer convolutional path's effective receptive field) but are exactly the pixels SegNet's argmax decision flips on.

### §2.3 Per-parameter (training-time) Venn diagram

When training the HNeRV decoder (`pr101_lc_v2_clone` lane substrate), the parameters partition similarly:

| parameter group | SegNet-loss gradient flow | PoseNet-loss gradient flow | training-time exploit |
|---|---|---|---|
| frame-0-only decoder weights | ❌ ORTHOGONAL (SegNet only sees frame_1) | ✅ ACTIVE (PoseNet sees both frames) | freeze during SegNet-focused training stages |
| frame-1-only decoder weights | ✅ ACTIVE | ✅ ACTIVE | always train |
| shared decoder weights | ✅ ACTIVE (via frame_1 path) | ✅ ACTIVE (via both paths) | always train |
| FiLM pose-conditioning weights | ✅ ACTIVE (FiLM modulates decoder which produces frame_1) | ✅ ACTIVE | train during pose-gradient stages |

The fact that HNeRV's decoder is symmetric across frames (same weights process frame_0 and frame_1) means the "frame-0-only decoder weights" set is structurally empty — there are no architectural weights that are pose-relevant-only-via-frame_0. This is an HNeRV-family design choice that an out-of-family architecture (per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD") could break. For instance, the `nscs01_nullspace_split_renderer` substrate (per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515`) DOES split frame-0 and frame-1 into separate decoder heads with separate gradient routing — its NullspaceSplitScoreAwareLoss is exactly this Venn-diagram-aware loss function. NSCS01 is the canonical instance of the **per-frame-head training-time orthogonality exploit**.

### §2.4 Per-weight bit-level Venn diagram (the deepest grain)

**CORRECTION 2026-05-17 per premise verification of `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py`:** the actual fec6 `decoder.bin` codec is **int8 mantissa + per-tensor fp16 scale**, NOT FP4 codebook. Lines 274-288 of fec6's `src/codec.py` decode each weight tensor as `zz = uint8 → int8 zigzag-decoded → float32 cast → multiplied by per-tensor fp16 scale`. The public PR101 baseline DOES use an FP4 codebook (in its own intake clone), but fec6 supersedes that with a simpler int8+fp16 codec. The 4-byte FP11 outer wrapper (4-byte magic + source_len + selector_len) wraps the int8+fp16 streams, but is not itself a quantization codebook.

Per the int8 mantissa + per-tensor fp16 scale codec:

- Each weight byte stores one int8 mantissa (-128 to 127); after zigzag decode + fp16-scale multiplication, the per-byte derivative is **`∂(weight_value)/∂(byte) = ±1 · scale_fp16`** — a continuous linear relationship, NOT a discrete codebook lookup.
- The fp16 scale is per-tensor (one scale per weight tensor), so all bytes within a tensor share the same multiplier.
- That weight value enters both frame_0 and frame_1 forward passes (HNeRV decoder is symmetric across frames per §2.3).
- Therefore: **every weight byte lives in the JOINT region** of the Venn diagram for HNeRV — same conclusion as the FP4 case, but for a DIFFERENT mathematical reason (joint frame use, not codebook structure).

The Jacobian-projection math in §3.2 below uses the corrected per-byte Jacobian `J[byte_i, θ_i] = ±scale_fp16(tensor)` rather than the FP4 lookup Jacobian. This is **simpler** (linear vs. discrete) and **denser** (per-byte gradient is the autograd ∂S/∂θ multiplied by the per-tensor scale directly, no codebook lookup). The original FP4-anchored draft text would have produced wrong per-byte sensitivity weights for the cathedral autopilot consumer hooks once the extractor landed an anchor — caught by Catalog #229 premise-verification before any GPU dispatch fired.

For `latents.bin` per-pair entries: each pair's latent decodes into both frame_0 and frame_1 for that pair. Per-pair latents are JOINT for that pair, but a per-pair latent for pair `i` is gradient-orthogonal to scoring of pair `j` (`i ≠ j`). The pair-index axis IS a per-pair orthogonality axis we can exploit (per the fec6 design — selecting different per-pair modes is exactly this exploit).

**The fec6 selector grammar (the 107-byte slot)** is the ONLY part of the current archive grammar where the per-byte ∂(d_seg)/∂b = 0 by construction. Engineering more SegNet-orthogonal byte slots is the architectural lever.

## §3 — Master gradient engineering

### §3.1 The formula

Given the contest scorer is `S = 100·d_seg + √(10·d_pose) + 25·R`, the per-byte master gradient is:

```
∂S/∂(byte_i) = 100·[∂(d_seg)/∂(frame_1)]·[∂(frame_1)/∂(byte_i)]              (SegNet term; zero if byte_i is frame_0-only)
             + (5/√(10·d_pose))·Σ_f∈{0,1} [∂(d_pose)/∂(frame_f)]·[∂(frame_f)/∂(byte_i)]   (PoseNet term)
             + 6.66e-7·𝟙{byte_i in archive}                                     (rate term; constant per byte)
```

At the fec6 operating point with `d_pose=2.943e-5`, the PoseNet-term coefficient is **291.5**.

For byte ranges in the **SegNet-only** Venn region, the first term dominates entirely:
```
∂S/∂(byte_i) ≈ 100·[∂(d_seg)/∂(frame_1)]·[∂(frame_1)/∂(byte_i)] + 6.66e-7   (no PoseNet term)
```

For byte ranges in the **PoseNet-only** Venn region (e.g., fec6's selector.bin):
```
∂S/∂(byte_i) ≈ 291.5·Σ_f [∂(d_pose)/∂(frame_f)]·[∂(frame_f)/∂(byte_i)] + 6.66e-7   (no SegNet term)
```

For byte ranges in the **Joint** region (most of the archive):
```
∂S/∂(byte_i) = full expression with non-orthogonal cross-effects
```

### §3.2 Materializing the master gradient empirically — REVISED per Round-1 review C-1 + C-2

**Original finite-difference proposal (REJECTED):** per-bit-flip probing on 178,517 bytes was mathematically the wrong object on FP4 codewords (van den Oord's C-2 finding: one-bit-flip produces a discrete 0.5-3.0 weight-unit jump, not a derivative) AND structurally infeasible within $5-15 (Karpathy's C-1 finding: 1 forward pass on CPU is 120-240 sec NOT 5 sec; 178,517×180s = 8,900 sequential hours, ~$534 with 100-way Modal CPU parallelism).

**Revised canonical methodology — autograd-based per-parameter gradient + FP4 Jacobian projection:**

```
Step 1: load fec6 decoder FP32 representation (decoder.bin decodes to ~88K float32 params)
Step 2: 1 forward + 1 backward pass on CPU with autograd:
    score = inflate(archive_bytes) → upstream/evaluate.py → S
    S.backward()
    G_θ[i] = θ[i].grad   (per-parameter gradient; ~88K floats)
Step 3: project G_θ through int8+fp16 quantization Jacobian (CORRECTED 2026-05-17 — fec6 uses int8 mantissa + per-tensor fp16 scale, NOT FP4 codebook; see §2.4 correction):
    For each weight tensor T with per-tensor scale s_T = fp16-decoded scale:
        For each byte b_k in T's int8 mantissa stream:
            J[b_k, θ_k] = ±s_T   (sign depends on zigzag-decode result; analytical; linear; computable in milliseconds)
    G_byte[k] = J[b_k, θ_k] · G_θ[θ_k]   (per-byte gradient; ~178K floats; one per int8 mantissa byte)
    For non-mantissa bytes (fp16-scale bytes, selector.bin, sidecar deltas, latent bytes):
        compute per-byte Jacobian per that stream's specific codec (fp16 unpack, fixed-Huffman lookup, etc.)
Step 4: decompose into (seg, pose, rate) terms via 2 sister backward passes
    (one for d_seg only, one for d_pose only; rate term constant 6.66e-7)
```

Cost analysis: 1 forward + 3 backward passes on Linux x86_64 CPU = ~3 × 240s = **~12 minutes wall-clock**, ~$0.50-2.00 on Modal CPU (`$0.06/hr × ~0.2h`). 30-300× faster than finite-difference; mathematically the CORRECT object (true derivative, not discrete-jump response).

**Caveat (Boyd's concurring note):** this gives the LOCAL gradient at the operating point. For per-bit perturbations in the FP4 codewords (where the chosen codeword IS the decision variable, not a continuous proxy), a sister discrete-perturbation experiment is also needed — but only on the top-K most sensitive bytes per the analytical projection, not all 178K. Estimated additional cost: $1-3 for K=1000 candidate bytes.

**This is op-routable #1 REVISED.** Total cost $0.50-5 (autograd + targeted discrete sensitivity for top-K bytes), not the original $5-15 estimate's infeasible underlying methodology.

Output: `.omx/state/master_gradient_fec6_20260517.npy`, shape `(178517, 3)`, dtype float32, with companion `.npz` carrying per-byte uncertainty bands from the K=1000 discrete-perturbation sister.

### §3.3 What the master gradient tensor enables

Once materialized, the master gradient enables:

1. **Per-byte EIG/$ ranking** — for any candidate modification of `byte_i` by `Δb`, the predicted ΔS is `Δb · G[byte_i, :]·[100, 291.5, 6.66e-7]ᵀ`. The autopilot's Rashomon ensemble can rank candidates without dispatching.
2. **Sub-byte (bit-level) targeting** — within a byte, only bits with large `|G[byte_i, :]|` are worth perturbing. Bits with `G[byte_i, :] ≈ 0` are dead weight (don't affect score).
3. **Disjointness verification** — query whether two candidate modifications are byte-disjoint; if yes, their predicted ΔS is additive (orthogonality holds); if no, must measure the cross-term.
4. **Master loss for training** — at training time, the score-aware loss term IS the master gradient projected through the trainer's `θ → byte → frame → score` chain. NSCS01's NullspaceSplitScoreAwareLoss is the prototype of this; a fully-materialized master gradient generalizes the pattern across all substrate trainers.
5. **Pareto frontier estimation** — the convex hull of the master gradient tensor IS the Pareto frontier of achievable `(d_seg, d_pose, R)` triples within the local linear regime. Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #296 Dykstra-feasibility requirement, the master gradient gives the convex-hull pieces needed for the Dykstra feasibility intersection.

### §3.4 The single master gradient (highly optimized, sophisticated, elegant, optimal, performant, efficient, correct)

The operator's request "engineer a single master gradient" maps to the following object:

```python
@dataclass(frozen=True)
class MasterGradient:
    """Per-byte score gradient at a measured operating point.

    g[i] = (∂S/∂byte_i)_{operating point}

    Decomposed into (seg, pose, rate) components for elasticity analysis.
    """
    archive_sha256: str           # which archive these gradients were measured on
    operating_point: dict         # {d_seg, d_pose, R, score} at measurement time
    g: np.ndarray                 # shape (N_bytes, 3); dtype float32
    measurement_method: str       # "finite_difference_bit_flip" / "double_jvp" / "modal_call_id_<id>"
    measurement_axis: str         # "[contest-CPU]" (this is the leaderboard axis)
    measurement_hardware: str     # "linux_x86_64_modal_cpu"
    measurement_utc: str
    pareto_facets: list[tuple[int, int]]  # convex-hull edges; output of Dykstra alternating projections
```

This object is consumable by:
- `tac.sensitivity_map` (axis-level reweighting per Catalog #232)
- `tac.autopilot_rudin_daubechies.slim_risk_scorer.PreflightSLIMRiskScorer` (as a per-byte risk feature)
- `tac.optimization.bit_allocator` (per-byte budget per Catalog #125 hook #3)
- `tac.cathedral_autopilot_autonomous_loop.rerank_candidates` (Catalog #125 hook #4)
- `tac.council_continual_learning.append_council_anchor` (continual-learning per Catalog #300 wire-in rule)
- `tac.substrates._shared.score_aware_common.score_pair_components` (as the canonical loss term per Catalog #164 — replacing the current `100·d_seg + √(10·d_pose)` with the full master-gradient-aware Lagrangian)

The continual-learning loop closes: every paid dispatch's measured score becomes a new MasterGradient anchor; the Rashomon ensemble refits over the K=8 most recent anchors; the dispatch ranker uses the consensus gradient; the disagreement queue surfaces bytes where the K=8 anchors disagree (high EIG candidates for the next dispatch).

### §3.5 Critical limitations of the master-gradient approach (Contrarian + Assumption-Adversary)

1. **The master gradient is OPERATING-POINT-LOCAL.** It is the LINEAR approximation of the loss surface at the measured point. Any non-trivial modification will leave the linear regime; the predicted ΔS becomes inaccurate after the first dispatch. Mitigation: re-measure the gradient after every dispatch (continual-learning loop above), OR include second-order terms (Hessian-vector products, ~$50-100 to materialize). The Rudin-Daubechies Rashomon ensemble (Catalog #252) is the structural fix for this drift.
2. **Cross-byte interactions are not captured by a per-byte gradient.** Two byte modifications that are individually +ΔS may interact non-additively when applied together. The fec6 + format0d combination is the canonical example — both individually beat baseline on their respective axes; their joint behavior is empirically unmeasured. Mitigation: explicit cross-term measurement OR archive-grammar enforcement of disjoint byte sections (Catalog #167 sister gate).
3. **The CPU↔CUDA bifurcation means the master gradient is AXIS-LOCAL.** A master gradient measured on `[contest-CPU]` does not predict `[contest-CUDA]` scores (per the bidirectional evidence in `feedback_permanent_fix_frontier_signal_loss_landed_20260517`). The operator must commission TWO master gradients (one per axis) if both leaderboards matter.

### §3.6 The 8 in-training / in-design / post-hoc uses of the master gradient (operator question 2026-05-17)

The original symposium framing in §3.3 understated the master gradient — it is NOT just an autopilot ranking input, but a first-class training/design signal with the right operating-point-refresh discipline. Eight concrete uses:

| # | use | category | re-measure cadence | rationale |
|---|---|---|---|---|
| 1 | Score-aware loss term at byte-grain | in-training | every ~50 epochs or when operating point shifts >5% | true per-byte chain rule `∂S/∂θ = ∂S/∂byte · ∂byte/∂θ`; generalizes NSCS01's manual gradient routing |
| 2 | Per-pixel/per-byte attention reweighting (Fridrich-DIE at byte-grain) | in-training | every ~50 epochs | UNIWARD-style detector-informed embedding at archive byte layer |
| 3 | Bit allocator hook (Catalog #125 hook #3 operational) | in-training | once per quantization stage | high-`|G|` bytes get more bits; low-`|G|` bytes get fewer — sister of Lane Ω-W water-filling |
| 4 | Architecture search / design discriminator | in-design | once per architecture | spectra of `|G|` distinguish substrate paradigms (HNeRV concentrated vs SIREN distributed vs Cool-Chic localized) |
| 5 | Score-aware QAT FP4 codebook selection (**future state** — fec6 currently uses int8 mantissa + per-tensor fp16 scale per §2.4 correction; QAT-to-FP4 is a Quantizr-style downstream stage that has not yet been applied to fec6's `decoder.bin`) | in-design | once at end of float training | quantize aggressively where `|G|` low; preserve precision where `|G|` high |
| 6 | Pareto facets feed Dykstra alternating-projection feasibility (Catalog #296) | in-design | once per Pareto-relevant operating point | convex hull of gradient tensor IS local Pareto frontier; supplies Dykstra's constraint-polytope vertices empirically |
| 7 | Continual-learning posterior for cathedral autopilot Rashomon ensemble | in-training (for ranker) | every dispatch | K=8 gradient measurements naturally train the K=8 SLIM scorers |
| 8 | *Magic codec* per-stream selection (operator-added 2026-05-17) | in-design | once per archive build | replaces pure rate-minimization with score-aware-rate-minimization; strict superset of magic codec's current capability |

Uses 1-3 + 7-8 are IN-TRAINING (gradient flows into the optimizer). Uses 4-6 are IN-DESIGN (gradient informs architectural choices but does not appear in backward pass). Post-hoc / scientific uses (paradigm comparison memos, retrospective audits, CPU/CUDA bifurcation diagnostics) are a SMALLER fraction of the value than originally framed in §3.3.

**Critical caveat:** the master gradient is OPERATING-POINT-LOCAL (first-order linear approximation valid only in a neighborhood). The autograd-based methodology (§3.2 revised) makes per-wave re-measurement economical at $0.50-2.00 per refresh, which unlocks in-training adoption. Pre-revision finite-difference (~$534) made even use #7 marginal; post-revision, all 8 uses are tractable.

The implementation order ranked by EIG/dispatch-readiness: #7 (lens; building this PR's Phase A) → #1 (score-aware loss term) → #2 (DIE reweighting, leveraged by U-DIE-KL Wave 3 op-routable #4) → #8 (*magic codec* score-aware mode) → #3 (bit allocator) → #5 (QAT codebook) → #4 (architecture discriminator) → #6 (Dykstra feasibility wiring). #7 + #1 + #2 land first because they have direct consumer code in the canonical helpers; the rest require sister-substrate integration.

## §4 — Orthogonal optimization + pausing + freezing framework

### §4.1 The three primitives

The operator named three orthogonal optimization disciplines:

1. **Orthogonal optimization** — search the SegNet axis, the PoseNet axis, and the rate axis separately, then compose the wins. Per §2.2, this works structurally when the byte ranges are disjoint. Per the master gradient §3, this works as a first-order approximation always; the cross-terms break the approximation.
2. **Pausing** — during training, suspend the gradient flow through one of the loss terms for some number of steps. The Quantizr 5-stage pattern is the canonical instance: stage 1 = anchor only, stage 2 = finetune, stage 3 = joint, stage 4 = QAT (with BN frozen), stage 5 = final (with all but pose-axis frozen).
3. **Freezing** — permanently fix a subset of parameters and continue training only the complement. The Quantizr QAT stage is canonical (BN frozen so the parametric eval roundtrip is stable); per CLAUDE.md "EMA — NON-NEGOTIABLE", the EMA shadow IS a form of freezing (live weights train, EMA shadow is the inference target).

### §4.2 The freezing-staircase canonical helper (op-routable #5)

The Quantizr 5-stage discipline + our internal Lane 17 IMP cycle pattern can be unified as a canonical helper:

```python
@dataclass
class FreezeStaircase:
    """Canonical N-stage freeze-and-unfreeze training schedule.

    Per CLAUDE.md "Quantizr's 5-stage pipeline" + "EMA — NON-NEGOTIABLE":
    each stage explicitly declares (a) which loss terms are active,
    (b) which parameter groups are frozen, (c) which EMA shadows are live,
    (d) the convergence criterion that triggers stage transition.
    """
    stages: tuple[StaircaseStage, ...]
    transition_criterion: Callable[[StageMetrics], bool]
    canonical_master_gradient: MasterGradient | None  # if provided, stages can be auto-derived

@dataclass(frozen=True)
class StaircaseStage:
    name: str
    active_loss_terms: frozenset[str]   # subset of {"seg", "pose", "pixel", "kl_distill", "rate"}
    frozen_param_groups: frozenset[str]
    ema_shadow_groups: frozenset[str]
    convergence_metric: str             # which metric the transition criterion reads
    convergence_threshold: float
    max_epochs: int
```

Adoption sweep: every in-flight substrate trainer (NSCS01 / NSCS03 / SIREN / Cool-Chic / VQ-VAE / wavelet / DP1 / Z3 / Z6 / ATW / STC) wraps its training loop in `FreezeStaircase.from_quantizr_canonical()`. The canonical produces the empirically-validated 5-stage schedule. Substrate-specific waivers per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" allow per-substrate stage customization. **This is op-routable #5.**

### §4.3 The pausing-vs-freezing decision table

| training condition | recommended action | rationale |
|---|---|---|
| SegNet loss is converged (Δ over last 100 epochs < ε) | freeze the SegNet-only parameter group | per §2.3, the SegNet-only weight set IS the natural freeze target; PoseNet training continues |
| PoseNet loss is in a noisy regime | pause the PoseNet loss term, train pixel-only | per Quantizr's stage 1 — establish anchor first |
| QAT begins | freeze BN stats (model.eval() on BN layers only) | per CLAUDE.md "QAT pipeline — non-negotiable" + Quantizr |
| EMA decay = 0.997, shadow diverging from live | freeze live weights for N epochs, train EMA-only | bounded recovery without losing the EMA's signal |
| Master-gradient measurement is N stale (more than 50 epochs old) | pause training, dispatch fresh measurement | per §3.5 staleness limitation |
| Cross-paradigm composition smoke shows non-additive ΔS | freeze ONE paradigm's parameters, train the OTHER alone | breaks the cross-term, reverts to first-order orthogonality |

## §5 — Alien tech reverse-engineering (revisited under master gradient lens)

### §5.1 fec6 (CPU-axis alien tech)

**Mechanism** (per `inflate.py:33-160`): 16 dynamic per-pair modes, applied to frame_0 only. The K=16 palette is enumerable, the per-pair mode selection is offline-precomputed against the CPU PoseNet's kernel signature.

**Under master gradient lens:** fec6's 107-byte selector.bin lives entirely in the PoseNet-only Venn region. Its per-byte gradient is `[0, ~291.5·δ_pose, 6.66e-7]` — pure PoseNet leverage at fixed rate cost. The discovery procedure (offline mode-search) IS a coordinate-descent attack on the per-byte gradient with the SegNet coordinate zeroed out by archive grammar.

**Generalization:** any per-pair byte stream that the inflate path applies ONLY to frame_0 (or ONLY to frame_1 in a SegNet-relevant region but PoseNet-blind way) is a sister exploit. We have one such stream; we could engineer more.

### §5.2 PR106 format0d (CUDA-axis *alien tech* — internal nickname; see writeup Glossary §12.1)

**Mechanism — TWO layers** (per `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md` + operator correction 2026-05-17 that the magic-codec layer was missing from the original reverse-engineering):

- **Layer 1 (wire-format grammar):** format0c base (361 bytes; exact-radix-packed dims as a single base-28 integer) + 523-byte additive correction stream applying `latents[p, d] += delta * scale` at inflate time.
- **Layer 2 (per-stream entropy coding — *magic codec*):** `tac.packet_compiler.*` dispatch table auto-selects the lowest-bitrate primitive per archive stream from {LZMA, Brotli, single-tensor AC, block-FP, hessian-block-FP, custom container, ~10 other primitives}. Without Layer 2 the format0d grammar produces archives ~30% larger; the CUDA-axis advantage is the **product** of (a) grammar's additive freedom AND (b) per-stream optimal compression.

The 8th in-training/in-design use of the master gradient (§3.6 below) makes the *magic codec*'s per-stream selection SCORE-AWARE — instead of selecting purely by rate (current behavior), it can prefer codecs whose byte-pattern aligns with the master gradient's low-`|G|` regions for the same rate budget, dropping ΔS at constant bitrate.

**Under master gradient lens:** format0d's modifications enter via `latents.bin` decode, which flows into BOTH frames → BOTH scorers — full JOINT region. The CUDA-axis win is an artifact of the CUDA kernel signature: the additive corrections were optimized against the CUDA PoseNet's (different) gradient signature. The CPU PoseNet's gradient signature is different (per the per-layer ε ≈ 0.0065 compounding model in §6.2), so the same corrections produce regressions on CPU.

**Generalization:** the additive correction PATTERN is generic; the specific correction scales are CUDA-axis-fitted. A sister `format0d_cpu_axis` trainer that fits the correction scales against the CPU master gradient would produce a CPU-axis sister exploit. Cost ~$30-50.

### §5.3 L5 Time-Traveler Wyner-Ziv (cross-axis time-traveler tech)

**Mechanism** (per `src/tac/optimization/l5_*.py` + the L5 Time-Traveler staircase v2 council deliberations): Wyner-Ziv 1976 source coding with side information at the decoder. The decoder has access to side info (the scorer's weights), so the encoder can transmit only the syndrome (the part of the source NOT predictable from side info). For pose deltas: instead of transmitting absolute poses, transmit only the residual against an ego-motion model predictable from the PREVIOUS pair's frames.

**Under master gradient lens:** L5 reduces the per-byte rate cost of `poses.bin` from 4,800 bytes to ~1,500-2,000 bytes (estimated 60% reduction per `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515`) at zero pose distortion. The rate term `6.66e-7 · 2,800` = `0.00187` direct savings. The pose term is unchanged because we're not modifying the actual pose VALUES, only their transmission scheme.

**Under the master gradient framework, L5 is the canonical example of a rate-axis SegNet-AND-PoseNet-orthogonal exploit.** It modifies the ENCODING of `poses.bin` (rate axis) without modifying the DECODED pose values (no SegNet or PoseNet effect).

### §5.4 Cross-product exploits (the unexploited Venn region)

The fec6 (PoseNet-only) × format0d (joint, CUDA-axis) × L5 (rate-only) cross-product is empirically unmeasured. Under the master gradient first-order assumption, the cross-product is additive:

```
ΔS_combined ≈ ΔS_fec6 + ΔS_format0d_cpu_axis_sister + ΔS_L5
              = (-0.001) + (-0.020) + (-0.002)
              ≈ -0.023 from current fec6 0.19205 = 0.169 [contest-CPU]
```

**The Contrarian explicitly notes this prediction is a model, not a measurement.** Op-routable #2's sister fec6+SABOR submission is the first byte-disjoint cross-product to actually measure.

## §6 — Hardware exploits (revisited under master gradient lens)

The CPU↔CUDA bifurcation is a per-byte-gradient kernel-signature property:

```
G_cpu[byte_i, :] ≠ G_cuda[byte_i, :]
```

The compounding model `(1+ε)^L = (1.0065)^256 ≈ 5.27` (per `cpu_cuda_xray_synthesis_20260511`) predicts the upper bound of the per-byte gradient mismatch. **The master gradient for CPU is a DIFFERENT TENSOR from the master gradient for CUDA.** The Pareto frontiers are different convex hulls.

| hardware exploit | gradient effect | exploitability |
|---|---|---|
| TF32 matmul (`torch.backends.cuda.matmul.allow_tf32 = True`) | shifts CUDA gradient by ~0.5-1.5% per matmul | already enabled by default; disabling slows inference ~10-15% but reduces gradient drift |
| cuDNN heuristic kernel selection (`torch.backends.cudnn.benchmark = True`) | shifts CUDA gradient by ~0.1-0.5% per conv | disabling kills ~5-10% throughput; usually keep enabled |
| FMA fusion (intra-kernel) | shifts both CPU and CUDA gradients by sub-ULP-level | hardware-fundamental; cannot disable |
| GT video decode path (PyAV vs DALI/NVDEC) | shifts BOTH inputs by sub-ULP | contest-infrastructure; outside our control |
| float64 intermediate accumulation in `F.interpolate` | bit-identical CPU↔CUDA at ~2× memory cost | engineering-fixable per §4.3 of the writeup |

The master gradient framework names these as **per-byte kernel-signature variance**. Mitigation strategies become first-class: a per-byte "kernel-signature confidence interval" can be appended to the master gradient tensor (G now has shape `(N_bytes, 3, 2)` — `[seg/pose/rate, mean/stddev]`). The Rashomon ensemble's K=8 measurement diversity captures exactly this variance empirically.

## §7 — Cathedral autopilot integration

The cathedral autopilot (`tools/cathedral_autopilot_autonomous_loop.py`) already has hooks for:
- Continual-learning posterior (`.omx/state/continual_learning_posterior.jsonl`, Catalog #128)
- Modal call-id ledger (`.omx/state/modal_call_id_ledger.jsonl`, Catalog #245)
- Cost-band calibration (`.omx/state/cost_band_posterior.jsonl`, Catalog #175/#177)
- Frontier scan (`tac.frontier_scan`, Catalog #316)
- Composition matrix (`.omx/state/substrate_composition_matrix.json`, Catalog #227)
- Rashomon ensemble (`tac.autopilot_rudin_daubechies.rashomon_ensemble`, Catalog #252)
- Council deliberation posterior (`.omx/state/council_deliberation_posterior.jsonl`, Catalog #300)

**The master gradient adds a 7th continual-learning surface:**

`.omx/state/master_gradient_anchors.jsonl` — fcntl-locked JSONL append-only per Catalog #128/#131 sister discipline. Each row:

```json
{
  "archive_sha256": "6bae0201...",
  "measurement_axis": "[contest-CPU]",
  "operating_point": {"d_seg": 5.603e-4, "d_pose": 2.943e-5, "R": 4.755e-3, "score": 0.19205},
  "gradient_storage_path": ".omx/state/master_gradient_<sha>.npy",
  "measurement_call_id": "modal_call_id_fc-...",
  "measurement_method": "finite_difference_bit_flip",
  "pareto_facets": [...],
  "rashomon_disagreement_score": 0.012,
  "written_at_utc": "2026-05-17T..."
}
```

**Wire-in plan (op-routable #3) — REVISED per Round-1 M-7:**
1. **Day 1 (4 hours):** define `CandidateModificationSpec` dataclass in `tac.autopilot_rudin_daubechies.candidate_spec` — fields `{archive_sha256, modified_bytes: dict[int, bytes], modified_archive_size: int, predicted_axis: str}`. This is the abstraction the autopilot ranker's existing `rerank_candidates` does NOT currently expose; the lens consumes this contract.
2. **Day 2 (3 hours):** `tac.autopilot_rudin_daubechies.master_gradient_lens.MasterGradientLens.score(spec: CandidateModificationSpec, gradient: MasterGradient) -> float` returns predicted ΔS by projecting `spec.modified_bytes` against `gradient.g[:, term_idx]`. Plus `MasterGradientLens.score_with_uncertainty(spec, ensemble: RashomonEnsemble[MasterGradient]) -> (mean, stddev)` for K=8 consensus + disagreement.
3. **Day 3 (2 hours):** wire into `cathedral_autopilot_autonomous_loop.py::rerank_candidates` via the existing 6-lens registry pattern from Catalog #273-#278 — single line `lens_registry.add(MasterGradientLens.from_canonical_ledger())`.
4. **Day 4 (3 hours):** dedicated tests in `src/tac/autopilot_rudin_daubechies/tests/test_master_gradient_lens.py` covering empty-gradient, single-byte mod, multi-byte mod, ensemble-consensus, disagreement-queue surfacing.

**Total wall-clock: ~12 hours over 4 days; $0 GPU.** Sister to the 6 existing Rudin-Daubechies lens implementations (Catalog #273-#278) per their landing-memo pattern.

This closes the continual learning loop the operator named: every dispatch → measured score → master gradient anchor → Rashomon refit → autopilot reranks next dispatch.

## §8 — Continual learning + staircase design

### §8.1 The multi-PR campaign staircase

| step | PR/submission | predicted [contest-CPU] | cost | uses | rationale |
|---|---|---|---|---|---|
| 0 | (current) fec6 PR | **0.19205** | already paid | fec6 PoseNet-only exploit | leaderboard primary |
| 1 | fec6 + SABOR (op-routable #2) | 0.18-0.185 | ~$40 | SABOR fills the SegNet-only Venn region | first byte-disjoint cross-product |
| 2 | + L5 Wyner-Ziv pose deltas (op-routable #6) | 0.17-0.18 | ~$30 | L5 fills the rate-only Venn region | three-region cross-product |
| 3 | + U-DIE-KL substrate-wide loss (op-routable #4) | 0.17-0.175 | ~$60 | replaces the loss function with the joint Tishby IB term | second-order improvement (operating-point shift) |
| 4 | + cross-paradigm A1 ⊕ PR106-format0d-CPU-axis-sister | 0.16-0.17 | ~$50 | format0d adapted to CPU axis stacked with A1 | non-additive but predicted to be sub-additive |
| 5 | + ATW codec (Atick-Tishby-Wyner triple) | 0.15-0.16 (Tier C requires) | ~$100 | architecture class-shift per Catalog #227 Tier-C requirement | reaches sub-0.16 frontier |

**Each step's empirical anchor becomes a new master gradient measurement at the new operating point.** The autopilot consumes the updated gradient, ranks the next candidate, etc. The staircase is self-reinforcing.

### §8.2 Continual-learning posterior accumulation

After step 5, the autopilot has 6 paired master gradients (one per step). The Rashomon ensemble fits K=8 SLIM scorers over the gradient space; the consensus prediction for the NEXT step has bounded uncertainty per the Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-sensing bound: `O(sqrt(N/K))` where N=178517 bytes and K=8 measurements. The bound is loose for K=8 — but the more dispatches we accumulate, the tighter the consensus.

**Per CLAUDE.md "Mission alignment — non-negotiable" + "Max observability — non-negotiable":** every step's master gradient anchor is queryable via `tac.master_gradient_ledger.query_*` (sister of `tac.council_continual_learning.query_*`). The autopilot's decisions are auditable via `MasterGradient.explain()` returning the rule-chain readback per Catalog #273.

## §9 — Per-member positions (per CLAUDE.md "Grand Council (advisory)" 20-seat roster; 16 voting, 4 recused, plus 6 specialist invitations per Catalog #300 T4 "≥1 specialist per affected paradigm" rule)

Round-1 review L-9 reconciliation: the frontmatter `council_attendees` field lists 30 names. Of those, 20 are the canonical grand-council roster per CLAUDE.md "Grand Council (advisory)" (12 existing + 8 new 2026-05-15 seats). The remaining 10 are SPECIALIST INVITATIONS attached to specific affected paradigms per CLAUDE.md "Council hierarchy: 4-tier protocol" T4 quorum rule ("≥1 specialist per affected paradigm/path"). Specialist invitations are non-voting on the meta-verdict but contribute paradigm-specific positions below.



(For brevity: each member states their operating-within assumption per Catalog #292, then their position. Recusals: van-den-Oord and Filler recused on grounds of authorship conflict with U-DIE-KL and STC families; Schmidhuber recused per prior-position-precommit on cooperative-receiver framing; Karpathy recused on grounds of insufficient session-context.)

**Shannon LEAD** (operating assumption: rate-distortion theory grounds every score-improvement claim) — **PROCEED**. The master gradient formula is the per-byte instantiation of the contest's rate-distortion landscape. Materializing it is the canonical Shannon-bound-discovery procedure. Vote: PROCEED.

**Dykstra CO-LEAD** (operating assumption: convex-feasibility alternating projections compute the achievable Pareto frontier) — **PROCEED-WITH-REVISIONS**. The master gradient's Pareto facets enable the Dykstra feasibility check per Catalog #296. But op-routable #1's $5-15 dispatch is the foundation — without it, the Pareto facets are model-only. Revision: dispatch op-routable #1 before any other op-routable fires.

**Yousfi** (operating assumption: the challenge IS inverse steganalysis at archive bit level) — **PROCEED**. The Venn diagram framework formalizes the per-pixel DIE intuition Fridrich and I have been operating in. SABOR is the natural per-pixel SegNet-only exploit. Op-routable #2 should fire alongside #1 (parallel, not sequential).

**Fridrich** (operating assumption: UNIWARD-style detector-informed embedding maximizes per-pixel utility-per-bit) — **PROCEED**. Master gradient IS the detector-informed embedding cost function at archive-byte level. Adopt the UNIWARD weighting `1 / (|G[byte_i, :]·[100, 292, 6.66e-7]ᵀ| + σ)` as the per-byte budget for any rate-allocation decision.

**Contrarian** (operating assumption: bold proposals must survive adversarial challenge) — **PROCEED-WITH-REVISIONS** (verbatim in council_dissent). The master gradient is a model; before any new substrate trainer adopts it as a loss term, op-routable #1 must measure the empirical gradient. Otherwise we are training against our own predictions, not against the contest scorer's actual surface.

**Assumption-Adversary** (operating assumption: shared assumptions must be classified HARD-EARNED vs CARGO-CULTED) — verdict in `council_assumption_adversary_verdict` frontmatter field. Three of the five operating assumptions are HARD-EARNED; one is HARD-EARNED-CONDITIONAL; one is CARGO-CULTED-PENDING-EMPIRICAL-PROOF. **PROCEED-WITH-REVISIONS** — op-routable #1 must fire before the CARGO-CULTED assumption can be promoted.

**Quantizr** (operating assumption: competitor approaches reveal what the leaderboard rewards) — **PROCEED**. The 5-stage training pattern I shipped at PR101 GOLD is the staircase the FreezeStaircase helper formalizes. Op-routable #5 deploys it across all substrate trainers. The U-DIE-KL adoption (op-routable #4) is the canonical "stack KL distill on the proven substrate" pattern PR101 GOLD ↔ PR101 GOLD + selector demonstrated.

**Selfcomp** (operating assumption: stack composition only counts when archive bytes drop AND distortion holds) — **PROCEED**. The Venn diagram + master gradient framework is the formalization of my block-FP + Hessian-quant stacking pattern at byte-grain. The disjoint-byte-region exploit (fec6 = PoseNet-only slot in archive grammar) is the per-pair-byte analog of my block-FP per-block-scale stacking.

**Hotz** (operating assumption: engineering shortcuts beat learned complexity) — **PROCEED-WITH-REVISIONS**. Materializing the per-byte gradient is the right move; using it as the autopilot lens is the right move. But the FreezeStaircase helper (op-routable #5) is over-abstracted — the 5-stage Quantizr pattern is the empirically-validated one; don't generalize until we have a second substrate that genuinely needs a different staircase. Revision: op-routable #5 ships as `tac.training.quantizr_5_stage_staircase` first, generalizes later if needed.

**Carmack** (operating assumption: 30-minute clarity per layer is the LOC budget) — **PROCEED**. The MasterGradient dataclass in §3.4 is 30-second-reviewable. The autopilot wire-in is 30-second-reviewable. Approve the design; resist the urge to add features beyond what op-routables #1-#7 require.

**MacKay (memorial seat)** (operating assumption: MDL bound + Bayesian inference + arithmetic coding is the unified framework) — **PROCEED**. The master gradient IS the MDL-optimal per-byte cost function at the operating point. The Rashomon ensemble IS the Bayesian posterior over gradient measurements. The continual-learning loop IS the variational EM iterating between posterior and observation. The framework is canonical; deploy it.

**Ballé** (operating assumption: end-to-end-trainable codec architectures + hyperprior side info beat hand-designed pipelines) — **PROCEED**. The master gradient enables the autopilot to choose hyperprior side-info per-byte rather than per-tensor. NSCS03 (already shipped per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515`) is the substrate that can adopt this first.

**Atick + Redlich** (operating assumption: cooperative-receiver framing — decoder + scorer are jointly the receiver; encoder optimizes mutual information) — **PROCEED**. The master gradient is the per-byte instantiation of the I(X;T)·I(T;Y) decomposition. Stage the staircase in op-routable #5 to include a "cooperative-receiver-loss-active" stage where the encoder explicitly optimizes against the scorer-as-receiver.

**Tishby (memorial) + Zaslavsky** (operating assumption: deep learning's success is information bottleneck — compression IS the principle) — **PROCEED**. U-DIE-KL substrate-wide loss (op-routable #4) IS the information-bottleneck loss term. The master gradient is the per-byte instantiation. Deploy both.

**Wyner** (operating assumption: source coding with side information at the decoder is the per-pair pose-delta-encoding canonical pattern) — **PROCEED**. L5 Time-Traveler is the canonical Wyner-Ziv instance for this contest. Op-routable #6 deploys it.

**Rao + Ballard** (operating assumption: predictive coding in visual cortex — hierarchical Bayesian inference at every level) — **PROCEED-WITH-REVISIONS**. Z6/Z7/Z8 predictive-coding substrates (per `feedback_six_meta_pattern_strict_gates_d_e_f_g_h_i_landed_20260516`) are the architecture class-shift candidates. The master gradient + staircase framework provides the training discipline they need. Revision: explicitly include Z6/Z7/Z8 in the staircase post-step-5.

**Hinton** (operating assumption: knowledge distillation T=2.0 + KL-on-logits is the canonical SegNet-distill primitive) — **PROCEED**. U-DIE-KL adoption (op-routable #4) is the right next step. The temperature T=2.0 + KL-on-logits is empirically validated by PR101 GOLD's SegNet distillation stage.

**Hassabis** (operating assumption: cross-domain breadth informs strategic-research portfolio) — **PROCEED**. The multi-PR campaign staircase (§8.1) is the right portfolio structure. Diversify across the 5 steps; don't all-in on any one step. The cathedral autopilot's Rashomon ensemble enforces this diversification.

**Tao** (operating assumption: pure-math omniscience; harmonic analysis + additive combinatorics + applied analysis) — **PROCEED**. The master gradient is a well-defined object on a discrete domain (the archive byte space). The Pareto convex hull is computable in O(N log N). The Rashomon refit is a sparse linear regression with bounded sample complexity per Daubechies-DeVore-Fornasier-Gunturk 2010. The framework is mathematically sound.

**Boyd** (operating assumption: convex optimization at operational level — ADMM, proximal gradient, alternating projections) — **PROCEED**. The Dykstra-feasibility wire-in (Catalog #296) is the canonical convex-feasibility check the master gradient enables. The master-gradient-aware loss term in op-routable #4 is a convex relaxation; the ADMM update for the convex relaxation is documented in Boyd-Parikh-Chu-Peleato-Eckstein 2011 §3.3.

**Jack-from-skunkworks** (operating assumption: internal SegNet+Rate research lineage; PR101 GOLD's architectural ancestor) — **PROCEED**. The Venn-diagram framing operationalizes what we've been doing implicitly since the original SegMap+Rate split. Glad to see it formalized.

**Time-Traveler protégé** (operating assumption: PENDING canonical identification per `feedback_grand_council_convergence_l5_staircase_comprehensive_plan_plus_roster_expansion_landed_20260515` op-routable #1) — **PROCEED**. The L5 Time-Traveler staircase v2 work my mentor has been doing is exactly the Wyner-Ziv-applied-to-pose-deltas instance the master gradient framework formalizes. Op-routable #6 deploys her decade of work.

### §9.1 Vote tally

- **PROCEED-unconditional:** 14 (Shannon, Yousfi, Fridrich, Quantizr, Selfcomp, Carmack, MacKay, Ballé, Atick, Redlich, Tishby+Zaslavsky, Wyner, Hinton, Hassabis, Tao, Boyd, Jack, Time-Traveler protégé) — wait that's 18, let me recount.
- Recounting: PROCEED-unconditional: 14 members (Shannon, Yousfi, Fridrich, Quantizr, Selfcomp, Carmack, MacKay, Ballé, Atick+Redlich(joint), Tishby+Zaslavsky(joint), Wyner, Hinton, Hassabis, Tao, Boyd, Jack, Time-Traveler protégé) — accounting joint memorial seats as 1 each: 14 unconditional + 4 with-revisions + 0 against + 4 recused = 22 effective seats voting.
- **PROCEED-WITH-REVISIONS:** 4 (Dykstra CO-LEAD, Contrarian, Assumption-Adversary, Hotz, Rao+Ballard(joint)) — 4 effective.
- **REFUSE:** 0
- **Recused:** 4 (van-den-Oord, Filler, Schmidhuber, Karpathy)
- **Quorum:** 14 + 4 = 18 unique seats voting / 22 effective = 82% participation; far exceeds T4 minimum of 16-of-20 grand council + 6-of-6 sextet.

**Verdict: PROCEED-WITH-REVISIONS** (per the operator's operator-frontier-override per the mission-alignment binding directive Consequence 1, the revisions are advisory; op-routables #1-#7 are ratified for execution).

## §10 — Cross-references

- `docs/pr_writeups/cpu_frontier_fec6_20260517.md` — the submission writeup this symposium analytically supports (note: §1 marginal value needs correction per op-routable #7)
- `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md` — alien-tech reverse-engineering of PR106 format0 family
- `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md` — full problem-space reverse engineering both axes
- `feedback_permanent_fix_frontier_signal_loss_landed_20260517` — canonical frontier scan + Catalog #316
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515` — 18-shared-assumption matrix
- `feedback_canonicalize_substrate_contest_cuda_chain_landed_20260515` — Catalog #240 substrate chain coherence
- `feedback_canonical_dispatch_optimization_protocol_landed_20260515` — Catalog #270 dispatch protocol
- `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515` — Catalog #250-#255 + #273-#278 interpretable ML autopilot
- `feedback_six_meta_pattern_strict_gates_d_e_f_g_h_i_landed_20260516` — Z6/Z7/Z8 predictive-coding staircase
- `feedback_council_hierarchy_v2_landed_20260516` — Catalog #300 4-tier protocol this memo instantiates
- `feedback_mission_alignment_followon_catalog_300_extension_landed_20260516` — mission-alignment binding directive enabling operator-frontier-override

## §11 — Operator-routables (in priority-EIG order)

| # | classification | description | predicted ΔS | cost | owner-lane | gates |
|---|---|---|---|---|---|---|
| 1 | FRONTIER-BREAKING | Materialize the per-byte master gradient via finite-difference probe on fec6 archive | enables every other op-routable | $5-15 | lane_master_gradient_materialization_fec6_20260518 | Catalog #245 + #270 + #167 |
| 2 | FRONTIER-BREAKING | Build sister fec6+SABOR submission (SegNet-only + PoseNet-only disjoint byte-regions) | -0.005 to -0.010 | ~$40 | lane_pr101_fec6_plus_sabor_disjoint_20260518 | Catalog #226 + #233 + #167 + #240 |
| 3 | FRONTIER-PROTECTING | Land master_gradient lens in tac.autopilot_rudin_daubechies as Phase-7 module | enables continual-learning loop | $0 | lane_master_gradient_lens_phase_7_20260518 | Catalog #125 + #273-#278 |
| 4 | FRONTIER-BREAKING | U-DIE-KL substrate-wide loss adoption sweep in fec6 trainer | -0.005 to -0.020 | $30-60 | lane_u_die_kl_fec6_adoption_20260518 | Catalog #164 + #226 + #240 + #270 |
| 5 | FRONTIER-PROTECTING | Canonicalize Quantizr 5-stage as tac.training.quantizr_5_stage_staircase helper | enables staircase reuse | $0 | lane_quantizr_5_stage_staircase_canonical_20260518 | Catalog #241 + #265 |
| 6 | FRONTIER-BREAKING | L5 Time-Traveler Wyner-Ziv pose deltas on fec6 | -0.008 to -0.015 | $30 | lane_l5_wyner_ziv_fec6_20260518 | Catalog #233 + #167 + #270 |
| 7 | APPARATUS-MAINTENANCE | Fix 5/√(10·d_pose) marginal in docs/pr_writeups/cpu_frontier_fec6_20260517.md §1 — corrected 292 (not 922) | direct PR writeup correctness | $0 | lane_pr_writeup_math_correction_20260517 | none |

**Recommended execution order:** #1 → (#2 + #3 + #5 in parallel) → (#4 + #6 in parallel) → #7 (writeup fix can happen any time but should land before PR opens).

**Total ETA + cost:** ~3 dispatch waves, ~$120-170, ~5-10 days operator wall-clock. Predicted final frontier: `0.16-0.17 [contest-CPU]`.
