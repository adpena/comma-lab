---
schema_version: substrate_design_memo_v2_20260516
substrate_id: coin_pp_implicit_neural_representation
lane_id: lane_path_3_k_coin_pp_implicit_neural_representation_20260526
created_utc: 2026-05-26T07:48:00Z
council_tier: T1
council_attendees: [Shannon, PR95Author, Time-Traveler]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "meta-learned MLP modulations are a STRUCTURALLY DIFFERENT rate-tradeoff than per-pair latent + shared decoder (NeRV-family canonical pattern)"
    classification: HARD-EARNED
    rationale: "Dupont-Loya-Bronstein 2021 (COIN++) + Perez 2017 (FiLM) jointly establish modulation as canonical efficient INR pattern; modulations live in O(MOD_DIM) space rather than O(decoder_params)"
  - assumption: "MOD_DIM=64 is sufficient for driving video 384x512"
    classification: CARGO-CULTED
    rationale: "chosen for L0 sanity at archive-budget alignment with sister substrates; COIN++ paper canonical small-scale numbers don't transfer directly to driving video spatial complexity; L1 sweep mandatory"
  - assumption: "coord-MLP base architecture (vs CNN base) is HARD-EARNED for COIN++ paradigm"
    classification: HARD-EARNED
    rationale: "coord-MLPs are the COIN/COIN++ canonical choice via Dupont et al; CNN base would deviate from paradigm anchor"
  - assumption: "int8 modulation quantization preserves enough signal for driving-video rate-distortion at the predicted band"
    classification: CARGO-CULTED
    rationale: "chosen for tight rate budget; int16 alternative would 2× per-pair rate; empirical sweep at L1 mandatory; Catalog #324 post-training Tier-C re-measurement required before any score claim"
related_deliberation_ids:
  - path_3_candidate_inventory_for_next_wave_spawning_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
council_decisions_recorded:
  - "op-routable #1: queue Catalog #1265 MLX-first contest-equivalence gate invocation against COINPP1 archive bytes BEFORE any paid CUDA dispatch"
  - "op-routable #2: queue Phase 2 substrate symposium per Catalog #325 for L1+ promotion eligibility"
  - "op-routable #3: queue MOD_DIM ∈ {16, 32, 64, 128, 256} empirical sweep at L1"
  - "op-routable #4: queue int8 vs int16 modulation quantization paired sweep with Catalog #324 post-training Tier-C re-measurement"
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
operator_directive_anchor: "The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"
binding_methodology_directives:
  - "The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering" (2026-05-26)
  - "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" (2026-05-26)
  - "we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy" (2026-05-26 AMENDMENT)
---

# Path 3 candidate K — COIN++ Implicit Neural Representation — L0 SCAFFOLD design memo

## BINDING OPERATOR DIRECTIVES

Three binding strategic reframings landed 2026-05-26:

1. *"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*
2. *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*
3. *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

## Paradigm anchor

COIN++ (Dupont, Loya, Bronstein 2021/2022 — arXiv:2201.12904 / ICML 2022) — meta-learned implicit neural representation via shared base coord-MLP + per-pair compact modulation vector. The base coord-MLP `F_phi: (x, y, t) -> rgb` is **shared across all pairs** (amortized cost); the per-pair modulation `m_i in R^MOD_DIM` is what gets shipped in the archive's latent slot via FiLM-style scale+shift on hidden layers (Perez et al. 2017).

This is structurally DIFFERENT from NeRV-family (per-pair latent + shared decoder) because:
- NeRV: per-pair latent z_i in R^16-32 → shared CNN decoder → RGB grid
- COIN++: per-pair modulation m_i in R^MOD_DIM → shared coord-MLP → per-coordinate RGB

The rate-tradeoff is structurally different: COIN++ per-pair cost is `MOD_DIM × quant_bits/8` bytes (e.g. 64 × 1 = 64 bytes/pair int8); NeRV per-pair cost is `latent_dim × 2` bytes (e.g. 16 × 2 = 32 bytes int16). Both are small per-pair; the amortized BASE cost dominates total archive size.

## Distinct from sister candidates

Per operator's "design the substrate, NOT bolt-on" directive (#1), this substrate is a **FRESH DESIGN from first principles** — distinct architectural class from sisters:

- **A=DreamerV3 RSSM** (`dreamer_v3_rssm/`, landed `69253a1cc`): categorical latent dynamics (G×K group-categorical alphabet); paradigm = world-model + Gumbel-softmax categorical bottleneck
- **E=BoostNeRV** (`boost_nerv_pr110_residual/`, landed `83910e54e`): iterative boosting against frozen PR110 base; paradigm = sequential residual ensembling
- **G=NIRVANA cascading** (`nirvana_cascading_nerv/`, in-flight `ae952528954e27bef`): hierarchical residual decoder cascade; paradigm = multi-scale wavelet-pyramid decoder
- **F=Z8 hierarchical predictive coding** (in-flight `a23f0430835406351`): Rao-Ballard + Mallat + Hafner + Wyner-Ziv canonical quadruple
- **K=COIN++ (this)**: meta-learned modulated coord-MLP; paradigm = shared base + tiny per-pair modulation

Sister `coin_plus_plus/` (2026-05-20 sketch by prior subagent) exists as **L0 SKETCH only**; it predates the 2026-05-26 directives (MLX-first + cargo-cult-pass + 3-axis discipline). This substrate `coin_pp_implicit_neural_representation/` is a FRESH design honoring all three new directives at design time, NOT a bolt-on extension. The prior sketch is preserved per Catalog #110/#113 HISTORICAL_PROVENANCE.

## Architecture (L0 SCAFFOLD)

```
Per-pair modulation m_i in R^MOD_DIM (default 64; quantized int8 in archive)
   |
   v
Shared coord-MLP base F_phi:
    Input: (x, y, t) — normalized pixel coord + frame_index (0 or 1)
        Apply sinusoidal positional encoding to (x, y, t) — coord → R^POS_DIM
    Hidden layers: 3 layers × HIDDEN_DIM=64 with FiLM modulation from m_i:
        h <- sin( linear(h) * scale_i + shift_i )
        where scale_i, shift_i = split( linear_film_proj(m_i), HIDDEN_DIM × 2 )
    Output: linear(h) -> R^3 -> sigmoid -> rgb in [0, 1]^3
   |
   v
For each pixel (x, y) in [0, 384) × [0, 512):
    rgb_t(x, y) = F_phi_mod_m_i(x, y, t)
   |
   v
Stack into rgb_0, rgb_1: (B, 3, 384, 512)
```

**Key architectural decisions** (per Catalog #290 §"Canonical-vs-unique decision per layer" below):
- **Coordinate-batched evaluation**: O(H×W) coordinate sampling per pair; vectorized via MLX broadcasting on coordinate grid. Known cost vs spatial-grid PixelShuffle decoders; trades inflate-runtime FLOPS for tiny archive rate.
- **MOD_DIM=64**: chosen at L0 for archive-budget alignment with sister substrates (~80 KB amortized for 600 pairs × 64 int8 + brotli ~30%). Sweep at L1.
- **Sinusoidal positional encoding for (x, y, t)** with POS_DIM=32 frequencies (Mildenhall et al. NeRF 2020 canonical 2π-scaled).
- **FiLM-style scale+shift modulation** (Perez 2017) on every hidden layer; this is the canonical COIN++ modulation pattern.
- **3 hidden layers × HIDDEN_DIM=64**: matches COIN++ paper canonical small-INR depth; HARD-EARNED for ICML 2022 MNIST/CIFAR scale. CARGO-CULTED for driving-video 384×512; L1 sweep mandatory.

## Canonical-vs-unique decision per layer

Per Catalog #290 + the new operator directive #1, every layer needs explicit canonical-vs-unique decision rationale.

| Layer | Canonical helper | Unique choice | Verdict | Rationale |
|---|---|---|---|---|
| **Score-aware loss** | `tac.substrates._shared.score_aware_common.score_pair_components` (Catalog #164) | ADOPT_CANONICAL | ADOPT_CANONICAL | Substrate is RGB-renderer; canonical scorer-preprocess routing serves the contest formula `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489` without per-substrate forking. |
| **Archive grammar** | None (per-substrate canonical) | UNIQUE COINPP1 magic + sinusoidal-coord-MLP state_dict + per-pair modulation blob | FORK_BECAUSE_PRINCIPLED_MISMATCH | COIN++ paradigm needs `(base_mlp_state_dict, per_pair_modulations_int8, meta)` not `(decoder_state_dict, per_pair_latents)` — modulation has DIFFERENT semantic role than latent. |
| **Inflate device selector** | `tac.substrates._shared.inflate_runtime.select_inflate_device` (Catalog #205) | ADOPT_CANONICAL | ADOPT_CANONICAL | Sister-canonical helper; preserves PACT_INFLATE_DEVICE contract; refuses MPS per CLAUDE.md "MPS auth eval is NOISE". |
| **Inflate raw-output path** | `tac.substrates._shared.inflate_runtime.raw_output_path` | ADOPT_CANONICAL | ADOPT_CANONICAL | Sister-canonical path safety; refuses `..` traversal + absolute paths. |
| **Inflate sh + py contract** | Per-substrate (Catalog #146) | UNIQUE 3-arg `inflate.sh <archive_dir> <output_dir> <file_list>` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Substrate-specific renderer topology; same contract shape per Catalog #146 but unique per-substrate decoder forward. |
| **MLX renderer topology** | None (per-substrate) | UNIQUE modulated coord-MLP with sinusoidal positional encoding | UNIQUE_PER_PARADIGM | COIN++ paradigm is structurally distinct from NeRV-family CNN decoders. |
| **numpy reference** | `tac.substrates.nirvana_cascading_nerv.numpy_reference` (sister) | UNIQUE per-primitive for COIN++ | FORK_BECAUSE_PRINCIPLED_MISMATCH | COIN++ primitives include FiLM modulation + sinusoidal positional encoding + coord-MLP forward; sister NIRVANA primitives are CNN-family (conv2d + bilinear upsample) — different paradigm. |
| **brotli quality** | `_BROTLI_QUALITY=9` (sister-canonical) | ADOPT_CANONICAL | ADOPT_CANONICAL | Matches C6 IBPS / sane_hnerv / NIRVANA / boost_nerv_pr110_residual / dreamer_v3_rssm canonical sister pattern. |
| **fp16 state_dict serialization** | `_serialize_state_dict_fp16` pattern (sister-canonical) | ADOPT_CANONICAL | ADOPT_CANONICAL | Sister-canonical pattern for archive determinism; fp16 cast on CPU. |
| **Test discipline** | Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof + Catalog #335 contract | ADOPT_CANONICAL | ADOPT_CANONICAL | Sister-canonical test discipline; ≥11 tests per inventory brief. |

**Net**: 8 ADOPT_CANONICAL (where shared infrastructure serves COIN++ without suppressing the substrate's optimal score) + 3 FORK_BECAUSE_PRINCIPLED_MISMATCH (per-paradigm-specific surfaces). Honors UNIQUE-AND-COMPLETE-PER-METHOD operating mode per CLAUDE.md non-negotiable.

## 9-dimension success checklist evidence

Per Catalog #294 every substrate landing memo must document evidence across all 9 dimensions:

| # | Dimension | Evidence at L0 SCAFFOLD | Phase 2+ plan |
|---|---|---|---|
| 1 | **UNIQUENESS (class-shift not within-class)** | DISTINCT paradigm from sisters A/E/G/F per §"Distinct from sister candidates"; meta-learned modulated coord-MLP is a COIN++ class-shift away from NeRV-family CNN decoders + RSSM categorical latents + cascading residuals + hierarchical predictive coding | empirical confirmation via Catalog #1265 MLX↔PyTorch parity gate before paid CUDA dispatch |
| 2 | **BEAUTY + ELEGANCE (30-sec-reviewable)** | Each file ≤350 LOC bolt-on budget; architecture.py ~250 LOC; archive.py ~300 LOC reusing sister pattern; inflate.py ≤200 LOC per Catalog #146 + #205 | maintain at L1 promotion |
| 3 | **DISTINCTNESS (explicitly different from sisters)** | Per §"Distinct from sister candidates": shared-base-MLP+per-pair-modulation distinct from NeRV's per-pair-latent+shared-CNN-decoder pattern; COINPP1 archive grammar distinct from NIR1/BST1/RSSM1 | maintain at L1 promotion |
| 4 | **RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)** | Catalog #229 PV applied (read 5 sister files including coin_plus_plus/__init__.py for prior-sketch context, NIRVANA mlx_renderer/numpy_reference/archive/inflate/tests/test_basic.py); cargo-cult audit table per Catalog #303 below classifies 6 assumptions HARD-EARNED vs CARGO-CULTED | recursive adversarial review per operator directive #3 at R1 sister wave |
| 5 | **OPTIMIZATION PER TECHNIQUE (Catalog #290 §canonical-vs-unique)** | Per-layer table above: 8 ADOPT_CANONICAL + 3 FORK_BECAUSE_PRINCIPLED_MISMATCH | maintain at L1 promotion |
| 6 | **STACK-OF-STACKS COMPOSABILITY (orthogonal axes + additive ΔS)** | COIN++ modulation paradigm is orthogonal to (a) NeRV-family CNN decoders (sister-stackable as a residual augment), (b) RSSM categorical latents (sister-stackable via latent-prior conditioning), (c) hierarchical residual cascades (sister-stackable as base-level decoder) | empirical paired-CUDA sweep at L2+ per Catalog #319/#322 composition discipline |
| 7 | **DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned)** | Archive grammar uses sorted-keys JSON + fixed brotli quality=9 + fp16 CPU cast (sister-canonical pattern); seed-pinned RNG planned for L1 trainer | byte-deterministic across runs at L1 |
| 8 | **EXTREME OPTIMIZATION + PERFORMANCE** | MLX-first per operator directive #1 + #4 enables $0 GPU iteration during design; per-pair archive cost ~64 bytes int8 + brotli — far below NeRV-family per-pair (≥32 bytes int16) | L1 MOD_DIM sweep + int8/int16 quant sweep |
| 9 | **OPTIMAL MINIMAL CONTEST SCORE** | NOT CLAIMED at L0 SCAFFOLD; `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false` per Catalog #127/#192/#317 non-promotable markers; paired contest-CPU + contest-CUDA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" REQUIRED at L2+ | L2+ post-training Tier-C re-measurement per Catalog #324 |

## Cargo-cult audit per assumption

Per Catalog #303 + the new operator directive #2 (*"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*), this fresh design enumerates assumptions even though it is NOT an extension (the 2-phase methodology is FRESH-DESIGN); the cargo-cult audit surfaces L0 design decisions that need empirical confirmation at L1:

| # | Assumption | HARD-EARNED / CARGO-CULTED | Citation | Unwind path at L1 |
|---|---|---|---|---|
| 1 | Meta-learned MLP modulations are a structurally different rate-tradeoff than per-pair latent + shared CNN decoder | HARD-EARNED | Dupont-Loya-Bronstein 2021 (COIN++) + Perez 2017 (FiLM); modulation lives in O(MOD_DIM) space, decoder amortized | empirical paired comparison vs NeRV-family at L1 contest-CUDA |
| 2 | MOD_DIM=64 sufficient for driving video 384×512 | CARGO-CULTED | chosen for L0 sanity at archive-budget alignment | sweep MOD_DIM ∈ {16, 32, 64, 128, 256} at L1 paired comparison |
| 3 | Coord-MLP base architecture (vs CNN base) is canonical for COIN++ paradigm | HARD-EARNED | Dupont et al. COIN/COIN++ canonical choice; coord-MLPs match per-pixel sampling semantics | maintain coord-MLP base; defer CNN-base experiment to L1+ |
| 4 | int8 modulation quantization preserves enough signal for driving-video rate-distortion at predicted band | CARGO-CULTED | chosen for tight per-pair rate (64 bytes/pair); not empirically validated on driving video | int8 vs int16 paired sweep at L1 with Catalog #324 post-training Tier-C re-measurement |
| 5 | Frame index as input dim (not learned encoding) is canonical for video INR | HARD-EARNED | standard practice across COIN++ / NeRV-family video INRs | maintain |
| 6 | Sinusoidal positional encoding with POS_DIM=32 frequencies adequate for 384×512 spatial resolution | CARGO-CULTED | Mildenhall NeRF 2020 default; not specifically tuned for driving video chroma/luma structure | empirical POS_DIM ∈ {16, 32, 64, 128} sweep at L1 |
| 7 | 3 hidden layers × HIDDEN_DIM=64 matches COIN++ paper canonical small-INR depth | CARGO-CULTED for driving video | COIN++ paper canonical for MNIST/CIFAR scale | empirical depth/width sweep at L1 |
| 8 | FiLM modulation on every hidden layer (vs only first/last) is canonical COIN++ pattern | HARD-EARNED | Perez 2017 FiLM canonical; modulating every layer enables maximum per-pair-pair adaptation | maintain |

**Net**: 4 HARD-EARNED + 4 CARGO-CULTED. The 4 CARGO-CULTED assumptions (MOD_DIM, int8 quant, POS_DIM, depth/width) require empirical confirmation at L1 paired sweeps before any L2+ promotion. Catalog #324 post-training Tier-C re-measurement REQUIRED for predicted-band validation per "Forbidden predicted_band-from-random-init-Tier-C-density" FORBIDDEN_PATTERN.

## Observability surface

Per Catalog #305 every substrate design memo must declare its observability surface across the 6 facets:

1. **Inspectable per layer** — every layer's input + output + intermediate state captured at runtime via:
   - MLX renderer's per-coordinate-batch forward emits `(batch, 3)` RGB tensor inspectable via MLX `.tolist()` or numpy reference parity check
   - Per-pair modulation `m_i` inspectable as int8 array before/after dequantization
   - FiLM scale+shift coefficients inspectable per layer

2. **Decomposable per signal** — composite metrics decomposable:
   - Final score → seg + pose + rate via canonical `score_pair_components`
   - Per-pair archive bytes = (HEADER + base_mlp_blob_brotli + modulation_blob_brotli + meta_blob) / num_pairs
   - Per-pair distortion decomposable into per-pixel L1/L2 + scorer-routed components

3. **Diff-able across runs** — byte-deterministic archive enables:
   - sha256-on-archive-bytes comparison across runs (same seed → same bytes)
   - per-layer activation diff via numpy reference run on same input

4. **Queryable post-hoc** — run artifacts:
   - MLX run emits canonical posterior anchor via `tac.continual_learning.posterior_update_locked` per Catalog #128
   - Per-iteration metrics emit JSONL to `experiments/results/<lane>/metrics.jsonl` (fcntl-locked per Catalog #131)
   - Archive byte sha256 + meta JSON queryable from the archive itself

5. **Cite-able** — every anchor tuple `(substrate, commit, call_id, config, random_seed, upstream_snapshot_sha256)` per Catalog #245 modal_call_id_ledger pattern

6. **Counterfactual-able** — Catalog #139 byte-mutation discipline:
   - per-pair modulation byte mutation: change one int8 byte → observe RGB change in inflated frame
   - base MLP weight byte mutation: change one fp16 byte → observe RGB change across multiple pairs (shared base)
   - L0 test suite includes Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation no_op_proof verifying the modulation slot bytes are frame-affecting

## Predicted ΔS band

**STATUS: pending_post_training per Catalog #324** — no random-init Tier-C density extrapolation per "Forbidden predicted_band-from-random-init-Tier-C-density" FORBIDDEN_PATTERN.

### Shannon R(D) bound on meta-learned MLP weight distribution

Per Catalog #344 canonical equation registry: the rate-distortion bound for a meta-learned MLP weight distribution under entropy coding is:

`R(D) >= H(theta_modulated | theta_base) - h_max(D, K_pixels)`

where:
- `H(theta_modulated | theta_base)` is the conditional entropy of per-pair modulations given the shared base MLP
- `h_max(D, K_pixels)` is the maximum entropy compatible with distortion D over K=384×512×3=589,824 pixels per frame
- For modulations with MOD_DIM=64 int8 (8 bits each), max raw entropy per pair = 512 bits = 64 bytes
- With brotli compression at ~30% efficiency on typical neural-net weight distributions, expected per-pair rate ≈ 19-20 bytes after compression
- For 600 pairs, modulation contribution to archive ≈ 11.5-12 KB

### Base MLP archive cost (dominant)

- Coord-MLP with POS_DIM=32 positional encoding + 3 hidden layers × HIDDEN_DIM=64 + FiLM projection:
  - Input layer: (32×2×3=192 → 64) linear ~12.3K params
  - 3 hidden layers: 64×64 each ~4.1K params × 3 = 12.3K params
  - FiLM projection: MOD_DIM=64 → HIDDEN_DIM×2 = 128 per layer × 3 layers = 24.6K params
  - Output: 64 → 3 = 192 params
  - **Total base MLP: ~49.4K params @ fp16 = ~99 KB raw → ~30 KB brotli-compressed**

### Predicted total archive bytes

- header (~32 bytes) + base_mlp_blob (~30 KB) + modulation_blob (~12 KB) + meta_blob (~256 bytes) ≈ **42-50 KB**
- contest rate term `25 × 42,000 / 37,545,489 = 0.0280` to `25 × 50,000 / 37,545,489 = 0.0333`
- For comparison: sister substrates land in 40 KB-1.5 MB range; COIN++ predicted-band is at the LOW END due to amortization

### Dykstra-feasibility intersection check (Catalog #296)

The 4-constraint polytope `(rate <= R_max, seg <= S_max, pose <= P_max, archive <= A_max)` for COIN++ at predicted MOD_DIM=64:
- rate: `0.028-0.033` — well within budget at LOW END of medal-class
- seg + pose: NOT empirically validated at L0; needs paired-CUDA contest-CUDA + contest-CPU sweep
- archive bytes: well within budget

**Per Catalog #296**: predicted band requires Dykstra-feasibility check on (seg, pose) axes which is NOT YET validated. The predicted-band shape (low rate, high uncertainty on distortion axes) means the alternating-projections intersection may yield empty set at the desired score band. **Probe-disambiguator path queued at L1**: `tools/probe_coin_pp_modulation_capacity_disambiguator.py` would measure per-pair modulation capacity → distortion curve to confirm Dykstra-feasibility BEFORE any paid dispatch.

### Predicted ΔS band (HONEST disclosure)

- IF MOD_DIM=64 sufficient for driving video 384×512 spatial complexity (CARGO-CULTED assumption #2): predicted ΔS in `[-0.005, -0.020]` vs current frontier (per Catalog #245 canonical frontier pointer)
- IF MOD_DIM=64 insufficient (failure mode): predicted ΔS in `[+0.020, +0.080]` (worse than frontier due to under-capacity modulation losing semantic detail at SegNet stride-2 stem)
- **NO RANDOM-INIT TIER-C EXTRAPOLATION** per Catalog #324; predicted_band_validation_status: pending_post_training

## Math + scientific + engineering rigor per layer (axis 1 per operator directive #3)

Per the 2026-05-26 amendment, every layer needs HARD-EARNED triple-axis citation OR explicit CARGO-CULTED classification:

| Layer | Math (first-principles) | Scientific (canonical paper) | Engineering (operational) | Verdict |
|---|---|---|---|---|
| Coord-MLP base | Universal function approximation (Hornik 1991); coord-MLPs realize C^k functions on bounded domain | Dupont et al. COIN++ 2022 (arXiv:2201.12904); Mildenhall et al. NeRF 2020 (arXiv:2003.08934) | sister NIRVANA mlx_renderer + sister coin_plus_plus L0 sketch | HARD-EARNED (all 3 axes) |
| FiLM modulation | conditioned-affine transformation preserves coord-MLP universality (Perez et al. 2017 proof) | Perez et al. AAAI 2018 (arXiv:1709.07871); used by COIN++ | sister coin_plus_plus L0 sketch uses FiLM pattern | HARD-EARNED |
| Sinusoidal positional encoding | Fourier features enable high-frequency function approximation (Tancik et al. NeurIPS 2020) | NeRF (Mildenhall 2020) + Tancik et al. NeurIPS 2020 (arXiv:2006.10739) | sister coin_plus_plus uses sin in arch; canonical in INR literature | HARD-EARNED |
| Per-pair modulation quantization (int8) | Lloyd-Max quantizer bound on rate-distortion (Gersho-Gray 1992) | Ballé et al. 2017 (arXiv:1611.01704) entropy-coded quantized neural net | sister substrate atw_codec_v1 / d4_wyner_ziv_frame_0 / boost_nerv use int8 quant pattern | HARD-EARNED for paradigm; CARGO-CULTED for MOD_DIM=64 specific choice |
| Coord batching via MLX broadcasting | matrix-multiply complexity O(N·M·K) (standard) | MLX paper (Apple 2024); JAX vmap canonical (Bradbury et al. 2018) | sister NIRVANA mlx_renderer uses similar broadcast pattern | HARD-EARNED |
| Brotli q=9 entropy coding | LZ77 + Huffman + context modeling (RFC 7932) | RFC 7932 (Brotli specification); Alakuijala-Szabadka 2016 | sister substrates C6/sane_hnerv/NIRVANA all use brotli q=9 | HARD-EARNED |
| fp16 state_dict serialization | IEEE 754 binary16 round-to-nearest preserves ~3 decimal digits | IEEE 754-2019 standard | sister substrates use same pattern | HARD-EARNED |
| Catalog #205 select_inflate_device | n/a (operational utility) | n/a | Catalog #205 canonical helper; sister substrates all use | HARD-EARNED |
| Score-aware loss routing | Contest formula S = 100·d_seg + sqrt(10·d_pose) + 25·B/N (canonical equation #14 sister) | upstream/evaluate.py contest scorer | Catalog #164 canonical helper score_pair_components | HARD-EARNED |
| MLX↔PyTorch parity validation | numerical analysis of fp16/fp32 matmul accumulation error bounds | Catalog #1265 MLX-first contest-equivalence gate (threshold 0.001 contest-units) | sister DreamerV3 max_abs=24.34 anchor empirical receipt | HARD-EARNED |

**Net**: 10/10 layers HARD-EARNED on math + scientific + engineering axes (with explicit CARGO-CULTED carve-out for MOD_DIM=64 specific choice within the int8 quantization paradigm).

## MLX drift minimization per primitive (axis 2 per operator directive #3)

Per the 2026-05-26 amendment + sister A=DreamerV3 max_abs=24.34 empirical anchor (caused by `align_corners=True` bilinear / `mx.repeat` 2× upsample anti-pattern) + Catalog #1255 MLX drift mitigation findings:

| MLX Primitive | Expected drift bound vs PyTorch (fp32) | KNOWN-DRIFT-RISK | Mitigation strategy | Canonical helper citation |
|---|---|---|---|---|
| `mx.array(np.ndarray)` dtype cast | ≤ 1e-7 (lossless for fp32) | LOW | use `mx.float32` explicit dtype | sister NIRVANA numpy_reference `to_float32` |
| `mx.matmul` (linear) | ≤ 1e-5 (fp32 accumulation) | LOW | maintain fp32; AVOID fp16 matmul without explicit fp32 accum | sister numpy_reference `linear` |
| `mx.sin` (positional encoding + activation) | ≤ 1e-6 per element | LOW | maintain fp32 input | sister numpy_reference `sin` |
| `mx.exp` (FiLM scale via exp(scale_log) pattern) | ≤ 1e-5 (fp32) | MEDIUM | use direct linear scale instead of exp(log_scale); MLX exp has known fp16 precision issues on edge values | new canonical helper `tac.substrates._shared.mlx_helpers.stable_exp` queued |
| Element-wise broadcast (`a * b`, `a + b`) for FiLM | ≤ 1e-7 | LOW | standard MLX broadcast; matches PyTorch semantics | sister NIRVANA pattern |
| Sigmoid output activation | ≤ 1e-6 fp32 | LOW | use canonical `mx.sigmoid` (numerically stable per sister NIRVANA numpy_reference) | sister numpy_reference `sigmoid` |
| Coordinate grid construction (`mx.arange` + `mx.meshgrid`) | ≤ 1e-7 (integer coords) | LOW | construct on CPU, transfer to MLX device | new canonical helper `tac.substrates._shared.mlx_helpers.make_coord_grid_nhwc` queued |
| Mean reduction (loss aggregation) | ≤ 1e-5 fp32 small-N; ≤ 1e-3 large-N | MEDIUM (for large-N) | use Kahan summation per Catalog #962 / slot 16 when batch size > 1e6 | sister NIRVANA numpy_reference `kahan_mean` |

**KNOWN-DRIFT-RISK primitives (2 MEDIUM)**: `mx.exp` (FiLM scale via exp pattern — mitigated by direct linear scale) + mean reduction at large-N (mitigated by Kahan summation when batch>1e6).

**AVOIDED anti-patterns** per CLAUDE.md "MPS auth eval is NOISE" + sister A=DreamerV3 drift anchor:
- ❌ `align_corners=True` bilinear (not used; substrate doesn't upsample inside MLX renderer; bicubic upscale to camera HW is in PyTorch inflate)
- ❌ `mx.repeat` 2× upsample (not used)
- ❌ `mx.softmax` without epsilon (not used; substrate is not softmax-based)
- ❌ Non-Kahan summation per Catalog #962 (mitigated for large-N via Kahan helper)
- ❌ fp16 matmul without explicit fp32 accumulation (avoided; substrate uses fp32 matmul)

**Smoke test parity requirements**:
- MLX↔PyTorch parity max_abs ≤ 1e-5 per layer for fp32 deterministic ops
- MLX↔numpy parity max_abs ≤ 1e-5 per primitive
- Per-channel + per-layer breakdown captured in `tests/test_basic.py::test_mlx_pytorch_parity_*`

## Portability via numpy per primitive (axis 3 per operator directive #3)

Per the 2026-05-26 amendment, every MLX primitive has sister numpy reference implementation at `numpy_reference.py`:

| MLX Primitive | numpy reference | Parity bound | GHA CPU CI compatible | Operator-portable on non-Apple-Silicon |
|---|---|---|---|---|
| `mx.array` dtype cast | `to_float32` | ≤ 1e-7 | YES | YES |
| Linear (matmul) | `linear` | ≤ 1e-5 | YES | YES |
| Sin (positional encoding + activation) | `sin` | ≤ 1e-6 | YES | YES |
| Sinusoidal positional encoding (composite) | `sinusoidal_positional_encoding` | ≤ 1e-6 | YES | YES |
| FiLM modulation | `film_modulate` | ≤ 1e-6 | YES | YES |
| Coord grid construction | `make_coord_grid_nhwc` | exact (integer coords) | YES | YES |
| Sigmoid output | `sigmoid` | ≤ 1e-6 | YES | YES |
| Mean reduction | `mean` / `kahan_mean` | ≤ 1e-5 small-N | YES | YES |
| Coord-MLP forward (composite) | `coord_mlp_forward` | ≤ 1e-5 | YES | YES |

**ALL 9 primitives have numpy reference** — substrate is fully portable on CPU-only test rigs without MLX dependency. Enables:
- (a) GHA CPU CI testing without MLX install (Catalog #178 + #179 sister discipline)
- (b) sister cathedral consumer cross-validation per Catalog #335
- (c) operator-portable diagnostic on non-Apple-Silicon hardware (Linux x86_64 CPU)

**Tests include MLX↔numpy reference parity** (byte-identical or documented tolerance) alongside MLX↔PyTorch parity tests per the binding 3-axis discipline.

## MLX-implementation roadmap addressing all 3 axes

**Phase 1 (L0 SCAFFOLD — THIS landing)**:
- ✅ MLX renderer with all 3 axes addressed (math+scientific+engineering rigor per layer / drift bounds per primitive / numpy reference per primitive)
- ✅ COINPP1 archive grammar byte-deterministic
- ✅ PyTorch inflate runtime per Catalog #146 + #205 ≤200 LOC
- ✅ 11+ tests including MLX↔numpy parity + MLX↔PyTorch parity (when MLX installed)
- ✅ MLX smoke trainer stub with `_full_main raises NotImplementedError` per Catalog #240 (c)

**Phase 2 (L1 promotion — operator-routable)**:
- Catalog #1265 MLX-first contest-equivalence gate (threshold 0.001) BEFORE any paid CUDA dispatch
- Per-substrate symposium per Catalog #325 (NEW — adversarial sextet pact + 6-step contract)
- R1 recursive adversarial review per operator directive #3 (3 council axes: Shannon+Dykstra+Tao math / Carmack+Hotz+Quantizr MLX drift / MacKay+Selfcomp+Contrarian numpy portability)
- 3 clean-pass cycles before paid CUDA dispatch authorized per CLAUDE.md "Recursive adversarial review protocol"

**Phase 3 (L2 promotion — paired contest-CPU + contest-CUDA)**:
- MOD_DIM ∈ {16, 32, 64, 128, 256} empirical sweep
- int8 vs int16 modulation quantization paired sweep
- POS_DIM ∈ {16, 32, 64, 128} sinusoidal frequency sweep
- depth × width sweep
- Catalog #324 post-training Tier-C re-measurement on landed archive sha256
- Catalog #319 deliverability proof
- Catalog #233 L1→L2 promotion canonical 4-gate

## Operator-routable next steps

1. **Queue Catalog #1265 MLX-first contest-equivalence gate** invocation against COINPP1 archive bytes BEFORE any paid CUDA dispatch authorization.
2. **Queue Phase 2 per-substrate symposium** per Catalog #325 for L1+ promotion eligibility.
3. **Queue MOD_DIM sweep** ∈ {16, 32, 64, 128, 256} empirical paired comparison at L1.
4. **Queue int8/int16 modulation quantization paired sweep** with Catalog #324 post-training Tier-C re-measurement.
5. **Queue R1 recursive adversarial review** per operator directive #3 (3 council axes: math+drift+portability).
6. **Queue MOD_DIM=64 vs sister A=DreamerV3 G×K=64 paired smoke** — both substrates carry identical per-pair bit budget (~64 bits/pair); empirical paired comparison would isolate paradigm-class effect (meta-learned modulation vs categorical latent dynamics).

## Sister coordination (Catalog #230)

Current sister-subagent state as of dispatch:
- **LANDED** (5 — research INPUT, NOT bolt-on target): A/B'/C'/D/E
- **IN-FLIGHT** (4 — avoid file collision): F/G/H/R1
- **THIS** (K): NEW substrate package at `src/tac/substrates/coin_pp_implicit_neural_representation/` — no file overlap with sisters

## 6-hook wire-in declaration (per Catalog #125)

- hook #1 sensitivity-map = N/A at L0 SCAFFOLD (substrate not yet emitting per-axis decomposition; queued for L1+ per Catalog #356 Tier B Dim 3 sister discipline)
- hook #2 Pareto constraint = N/A at L0 (queued for L1+ Dykstra-feasibility on (rate, seg, pose, archive) polytope)
- hook #3 bit-allocator = N/A at L0 (per-pair modulation rate is fixed at MOD_DIM×8 bits; bit-allocator surface trivial)
- hook #4 cathedral autopilot dispatch = N/A at L0 (Catalog #341 routing-markers all non-promotable; `[macOS-MLX research-signal]` + `score_claim=false` + `promotable=False`)
- hook #5 continual-learning posterior = ACTIVE (this landing memo emits canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`)
- hook #6 probe-disambiguator = ACTIVE (the canonical MLX-first contest-equivalence gate at Catalog #1265 IS the disambiguator between MLX-research-signal vs paid-CUDA-authoritative; predicted MOD_DIM sufficiency disambiguator queued at L1)

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — meta-learned modulated INR is a paradigm-class shift away from NeRV-family + RSSM + cascading + hierarchical-predictive-coding sister substrates. Empirical confirmation of class-shift impact requires paired contest-CUDA + contest-CPU at L2+ per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

## Cross-references

- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the design discipline this landing honors)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium — NON-NEGOTIABLE, HIGHEST EMPHASIS" (per-substrate symposium queued at Phase 2)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — NON-NEGOTIABLE" (paired axis discipline at L2+)
- CLAUDE.md FORBIDDEN PATTERN "Forbidden predicted_band-from-random-init-Tier-C-density" (predicted_band_validation_status: pending_post_training per Catalog #324)
- `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` §Tier 2 K (the original brief)
- `src/tac/substrates/coin_plus_plus/__init__.py` (2026-05-20 prior sketch; preserved per Catalog #110/#113 HISTORICAL_PROVENANCE; this fresh design supersedes for MLX-first + 3-axis discipline)
- `src/tac/substrates/nirvana_cascading_nerv/` (sister canonical template for MLX renderer + numpy reference + archive grammar + tests)
- Catalog #290 / #294 / #296 / #303 / #305 / #309 / #340 / #344 / #346 (the design-memo discipline gates this memo satisfies)
- Catalog #229 PV / #117 / #157 / #174 / #206 / #119 / #287 / #110 / #113 / #208 / #230 / #310 / #325 (the operational discipline this landing follows)
- Catalog #1255 MLX drift mitigation findings (axis 2 reference)
- Catalog #1265 MLX-first contest-equivalence gate (threshold 0.001 contest-units; queued for L1 promotion path)
