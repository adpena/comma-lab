<!-- SPDX-License-Identifier: MIT -->
# PACT-NeRV-VQ Maturity Audit — Codebook-Composition Investment Decision

**UTC:** 2026-06-09
**Agent:** read-only audit subagent (no code edits, no commits)
**Axis discipline:** every score below is `[macOS-MLX research-signal]` false-authority per Catalog #192/#127/#323/#341. **NO `[contest-CPU]` or `[contest-CUDA]` exact eval has EVER been run on this substrate.**

**Operator question:** for a COMPOSED/codebook latent representation (K shared codewords + per-pair index + residual, exploiting dashcam cross-pair temporal redundancy), should the investment go into advancing the existing sibling lane `src/tac/substrates/pact_nerv_vq/` (sibling-advance) OR be built as a HiNeRV codebook retrofit (HiNeRV-retrofit)?

**One-line verdict:** **HiNeRV-RETROFIT** (with a precise nuance — see §7). `pact_nerv_vq` has a real, well-built codebook *carrier* but is BEHIND HiNeRV on the only axis that matters here (distortion fit / score-readiness), has NO exact eval, sits at ~90.6 advisory MLX score (≈470× from the ~0.19 frontier), and its own latest codex audit (2026-06-02) explicitly recommends NOT spending more long-run budget on it as a primary carrier. Critically, `pact_nerv_vq` is **NOT a residual-token VQ** — it has no residual term — so it is not even the object the operator's "K codewords + per-pair index + residual" description names.

---

## 1. Modules + codebook structure

Package `src/tac/substrates/pact_nerv_vq/` (8 modules + tests, all real implementation):

| Module | LOC (approx) | Role |
|---|---|---|
| `architecture.py` | 288 | `PactNervVqSubstrate` HNeRV decoder (DepthSep conv + SIREN sin(30) + 7× PixelShuffle → 384×512 RGB) + `VectorQuantizerEMA` + per-pair `latents` Parameter |
| `archive.py` | 335 | PVQ monolithic `0.bin` grammar: 27-byte header + decoder_blob (FP16/brotli) + codebook_blob (int16) + indices_blob (uint16) + meta_blob; `pack_archive`/`parse_archive` |
| `archive_candidate.py` | ~400 | MLX→PVQ exporter; `_quantize_latents_via_codebook` (van den Oord §3.1 Euclidean nearest) |
| `inflate.py` | 88 | ≤150 LOC PVQ consumer; replaces per-pair latents with `codebook[indices]` for byte-stable forward; NO inflate-time scorer load (compliant) |
| `score_aware_loss.py` | 122 | `L = 25·B/N + 100·d_seg + γ·√(10·d_pose) + 0.25·commitment`; eval-roundtrip MANDATORY (raises if `apply_eval_roundtrip=False`); canonical scorer dispatch |
| `competitiveness_gate.py` | ~400 | (NEW 2026-06-02) classifies codec-sweep + full-video replay; verdict taxonomy |
| `section_value.py` | ~150 | per-section value profiling (decoder_qw / codebooks_q / selectors_rc removal deltas) |
| `mlx_renderer.py` | 729 | MLX-LOCAL training mirror |

**Codebook structure (the operator's core question):**
- **Real codebook? YES.** `VectorQuantizerEMA` is a genuine VQ-VAE: nearest-codebook lookup (squared-distance argmin), straight-through estimator (`z_q_st = z_e + (z_q - z_e).detach()`), EMA codebook update with Laplace smoothing (`ema_update`, `@torch.no_grad`), and commitment loss `F.mse_loss(z_e, z_q.detach())`. This is faithful to van den Oord 1711.00937 §3.1–3.2 — NOT a fake/constant stub.
- **K (codewords):** `codebook_size = 512` default (uint16 indices support up to 65535).
- **Per-pair index encoding? YES.** `indices_blob` ships `num_pairs` uint16 codes; `inflate` reconstructs `z_e[i] = codebook[indices[i]]`.
- **Is the codebook the LATENT representation? YES — and this is the key structural finding.** `self.latents = nn.Parameter(shape=(num_pairs, latent_dim))` is quantized to ONE codeword per pair at forward (`z_e = self.latents[pair_indices]` → `quantizer(z_e)`). It is a **primary per-pair single-vector VQ latent carrier**, structurally analogous to HiNeRV's 600 independent per-pair latents but with the continuous latent replaced by a discrete codebook lookup.
- **Residual handling? NONE.** `grep residual` in `architecture.py` returns ZERO matches. There is no residual token, no shallow/inter-frame residual feature path, no codebook-utilization repair. **This is exactly the gap the operator's "+ residual" clause names — and it is absent.**

The 2026-06-02 codex audit (`codex_findings_upstream_eval_compact_vq_pivot_audit`) states this verbatim: *"The current local PACT/VQ implementation is not that RT/VQ object. It is a primary per-pair single-vector VQ latent carrier ... No residual tokenizer, no shallow/inter-frame residual feature path, and no codebook-utilization repair are present."*

---

## 2. Lane level + gates

Multiple registry lanes touch this substrate. The canonical / latest ones:

| Lane id | Level | impl_complete | real_archive_empirical | contest_cuda | contest_cpu | strict_preflight | three_clean_review | memory_entry | deploy_runbook |
|---|---|---|---|---|---|---|---|---|---|
| `lane_pact_nerv_vq_l0_scaffold_20260520` | **L1** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| `lane_pact_nerv_vq_l1_long_run_mlx_local_20260528` | **L1** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| `lane_pact_vq_projection_gap_repair_20260602` (latest) | **L1** | **✗** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**Lane level: L1 across the board.** Gates satisfied are only `impl_complete` + `memory_entry` (+ `deploy_runbook` on the scaffold). The three score-truth gates — `real_archive_empirical` (in the registry's sense of a *contest-axis* anchor), `contest_cuda`, `contest_cpu` — are **all FALSE** on every promotion-relevant lane. The latest 2026-06-02 lane even has `impl_complete=false`. No lane is L2 or L3. (For comparison, the ONLY L3 lane in the whole registry is `lane_g_v3`.)

---

## 3. Tests

| Test file | Count | BEHAVIOR vs constants |
|---|---|---|
| `pact_nerv_vq/tests/test_pact_nerv_vq.py` | 17 | **BEHAVIORAL** |
| `pact_nerv_vq/tests/test_competitiveness_gate.py` | (gate) | behavioral |
| `tests/test_profile_pact_nerv_vq_mlx_section_value.py` | 10 | behavioral (section-value profiling + section-cut materializer) |

The 17 substrate tests test **real behavior, not constants** (passes the CLAUDE.md "tests-verify-behavior-not-constants" bar):
- `test_substrate_forward_produces_unit_interval_rgb` — forward output range.
- `test_vector_quantizer_ema_straight_through_gradient_flows` — STE gradient actually flows through quantization.
- `test_mlx_vq_renderer_post_step_updates_codebook_ema` — EMA update actually mutates the codebook.
- `test_archive_pack_then_parse_roundtrip_recovers_tensors` (+ int8 / scale-bundled variants) — archive round-trips real tensors.
- `test_byte_mutation_changes_archive_no_op_proof` — Catalog #139 byte-mutation/no-op detector.
- `test_archive_grammar_header_size_invariant_is_27_bytes`, `test_inflate_py_loc_under_200`, `test_trainer_routes_through_canonical_scorer_loss_helper`, `test_trainer_patches_differentiable_eval_roundtrip_before_scorer`, `test_recipe_research_only_and_dispatch_disabled`, `test_driver_carries_canonical_nvml_block`.

Test coverage is genuinely good and verifies the codebook *quantizes* and the archive *round-trips*. This is a real, non-fake implementation per the "NO FAKE IMPLEMENTATIONS" bar.

---

## 4. Exact-eval results + axis

**NONE on any contest axis.** A repo-wide grep for `pact_nerv_vq|pact_vq|pvq` intersected with `[contest-CUDA]`/`[contest-CPU]` returns no genuine contest-axis score (every hit is a "deferred / withheld / refused / pending / not executed" statement). Every measured number is `[macOS-MLX research-signal]` false-authority:

| Source (date) | Archive bytes | Advisory MLX `canonical_score` | avg d_seg | avg d_pose | Axis |
|---|---|---|---|---|---|
| L1 long-run MLX (2026-05-28) | 135,960 (archive.zip) | (training-loss only; 0.001825 final loss) | — | — | `[macOS-MLX]` |
| QAT4 int2 full-600 (2026-06-02) | 33,596 | **91.03** | 0.506 | 163.08 | `[macOS-MLX]` |
| int2_mixed combined-cut (2026-06-02) | 22,176 | **90.62** | 0.505 | 160.96 | `[macOS-MLX]` |
| competitiveness-gate replay (2026-06-02) | 37,580 | **90.66** | 0.505 | 161.24 | `[macOS-MLX]` |
| projection-gap repair (2026-06-02) | 31,728 (receiver-proven) | ~90.66 | — | — | `[macOS-MLX]` |

**Interpretation:** The advisory full-video MLX score sits at **~90.6** — roughly **470× above the ~0.19 contest-CPU frontier** (canonical frontier is pointer-only; fec6 PR101 on the contest-CPU axis). The Catalog #1265 contest-equivalence gate also FAILs (max_abs_drift 0.42, SIREN-class drift-vs-depth) — but that's a separate MLX→PyTorch bridge-parity issue, not the score gap. The score gap is real distortion: d_seg ≈ 0.5 (the contest's 100·d_seg term alone → ~50) and d_pose ≈ 161 (√(10·161) → ~40). **The rate axis is SOLVED** (receiver-proven 22–34 KB); the blocker is entirely distortion fit.

---

## 5. Design intent + current blocker

**Design intent (2026-05-20 L0 scaffold → 2026-05-28 L1):** Variant #7 of Group 2 of the PACT-NeRV-ULTIMATE program. The DISTINGUISHING primitive vs the IA3/Selector-V2/V3/V4 sibling cascade is *discrete tokens*: replace the continuous per-pair latent with a VQ codebook lookup, giving a rate-axis lever (per-pair cost = ⌈log2(K)⌉ bits + amortized codebook) the continuous-coder siblings lack. Predicted L0 rate-axis savings ~3.1 KB vs FP16 latents → predicted ΔS ≈ −0.002.

**Current blocker (latest authoritative state, 2026-06-01/02 codex cluster — supersedes the optimistic 2026-05-28 memo):**

1. **Distortion fit, NOT rate** (`codex_findings_pact_vq_qat4_rate_solved_fit_blocked`, `..._competitiveness_gate`): rate is solved (22–34 KB receiver-proven); SegNet boundary fit stays high (d_seg ≈ 0.5) and renders are "a dark mean-field image, not a road-scene renderer" even with real SegNet/PoseNet teachers bound. Competitiveness-gate verdict: `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`.
2. **Wrong VQ object** (`..._compact_vq_pivot_audit`): the substrate is a per-pair single-vector VQ, not the RT-NeRV/VQ-NeRV residual-tokenization object. Audit verdict: `pivot_or_rebuild_vq_before_more_long_run_spend`; spend recommendation: `route_compact_training_budget_to_pr95_hinerv_snerv_stage8_or_rebuild_vq_as_rt_residual_token_bolton`.
3. **Section-value is negative**: removing decoder_qw / codebooks_q / selectors_rc each IMPROVES the advisory objective → the trained bytes are charged + receiver-consumed but locally *harmful* (not carrying score value).

**Codex's explicit next-work ranking (2026-06-02):** (1) do NOT spend more long-run budget on per-pair-latent PACT/VQ as a primary carrier; (2) put PR95 Stage-8 / HiNeRV / SNeRV score-aware decoder-weight fitting AHEAD of PACT/VQ; (3) re-admit VQ only as an RT/VQ-style **residual-token bolt-on** (residual tokenizer + shallow/inter-frame feature path + codebook-utilization repair) with proven value-per-byte.

---

## 6. Score-aware?

**YES — genuinely score-aware, same caliber as HiNeRV.** Not recon-only.
- `score_aware_loss.py`: full score-domain Lagrangian `25·B/N + 100·d_seg + γ·√(10·d_pose) + 0.25·commitment`, routed through canonical `score_pair_components_dispatch` (Catalog #164), pose/seg loader assignment-order per Catalog #222.
- **eval-roundtrip is MANDATORY** — `forward` raises `ValueError` if `apply_eval_roundtrip=False` (Catalog #6), and calls `apply_eval_roundtrip_during_training` on both frames before scorer loss.
- **Real SegNet + PoseNet teachers bind** — the 2026-06-01 real-scorer smoke (`codex_findings_compact_pact_vq_real_scorer_smoke`) confirms "both real SegNet and real PoseNet teachers: bound" via `RendererBundle` with `--segnet-distillation-weight`/`--pose-distillation-weight`; SegNet-only fails closed unless explicitly tagged research. This is NOT a mock teacher (passes the Catalog #322 phantom-provenance bar — pose proxy = 3.996 non-zero).
- EMA shadow used for inference checkpoint.

So on the score-aware-training-infrastructure axis, `pact_nerv_vq` is at parity with HiNeRV. The problem is the *result* of that training (dark mean-field renders), not the absence of the training path.

---

## 7. VERDICT: sibling-advance vs HiNeRV-retrofit

**HiNeRV-RETROFIT**, with one important nuance.

### Decisive evidence

1. **`pact_nerv_vq` is BEHIND HiNeRV on the only axis that matters.** Both have score-aware training (SegNet/PoseNet teacher + eval-roundtrip + EMA). But HiNeRV is a higher-capacity hierarchical INR that the lab is actively fitting toward the frontier, whereas `pact_nerv_vq`'s best advisory MLX score is **~90.6** with renders that are "a dark mean-field image, not a road-scene renderer." A codebook composition built on a carrier that cannot render road semantics inherits that ceiling. The codebook only helps the *rate* term; `pact_nerv_vq`'s problem is *distortion*.

2. **NO exact eval, ever.** `pact_nerv_vq` has zero `[contest-CPU]`/`[contest-CUDA]` anchors. It is an L1 MLX-LOCAL scaffold whose latest lane has `impl_complete=false`. Investing the codebook composition here means building on a substrate that has never touched the contest scorer. HiNeRV is the active frontier-fitting carrier.

3. **The substrate's OWN latest audit says don't.** The 2026-06-02 codex `compact_vq_pivot_audit` verdict is `pivot_or_rebuild_vq_before_more_long_run_spend` and the spend recommendation routes budget to *PR95/HiNeRV/SNeRV*, explicitly de-prioritizing per-pair-latent PACT/VQ as a primary carrier. Choosing sibling-advance directly contradicts the substrate's own ratified system-intelligence demotion (the bounded runner marks matching rows `demoted_by_compact_vq_pivot_audit`).

4. **`pact_nerv_vq` is not even the object the operator described.** The operator wants "K codewords + per-pair index + **residual**." `pact_nerv_vq` has K + per-pair index but **NO residual term** and no codebook-utilization repair. So "sibling-advance" is not "finish a half-built residual VQ" — it is "rebuild the sibling into an RT-VQ object it currently is not," which is comparable effort to a HiNeRV retrofit but on a weaker, never-exact-evaluated carrier.

### The nuance (don't throw the codebook away)

The codebook *machinery* is real, tested, and reusable: `VectorQuantizerEMA` (K=512, EMA + STE + commitment) and the PVQ archive grammar (receiver-proven at 22–34 KB) are production-grade rate primitives. The right move is to **retrofit the composed-codebook representation onto HiNeRV's 600 independent per-pair latents** (replace the continuous leaf with K-codeword + per-pair-index + residual), **reusing `pact_nerv_vq`'s `VectorQuantizerEMA` + PVQ codec as the codebook/archive layer**. That captures the operator's cross-pair temporal-redundancy thesis on a carrier that (a) is actively being fit toward the frontier and (b) renders road semantics — while not discarding the genuinely good codebook engineering the sibling already proved out.

Per the codex audit's third option, if the team instead wants to keep VQ as the carrier, the gating requirement is to first turn it into an RT-VQ residual-token object (residual tokenizer + shallow/inter-frame feature path + codebook-utilization repair) — i.e., a near-rebuild — which is *not* "further along than a HiNeRV retrofit." It is roughly equal build effort on the weaker carrier. Hence: **HiNeRV-retrofit, reusing pact_nerv_vq's codebook/codec primitives.**

### Reactivation criteria for sibling-advance (per "Forbidden premature KILL")
This is NOT a kill of `pact_nerv_vq`. Sibling-advance becomes the better choice if ALL of: (a) `pact_nerv_vq` distortion fit reaches frontier-adjacent advisory MLX score (≪ 90, road semantics preserved) via scorer-faithful retraining or RT-residual rebuild; (b) a paired `[contest-CPU]`+`[contest-CUDA]` anchor lands; (c) the int2_mixed rate primitive (22–34 KB) is shown to compose with a distortion-competitive decoder. Until then the codebook composition belongs on HiNeRV.

---

## Cross-references
- L1 landing (optimistic, superseded by June audits): `.omx/research/pact_nerv_vq_l1_long_run_mlx_landed_20260528.md`
- L0 scaffold design: `.omx/research/pact_nerv_vq_l0_scaffold_design_20260520T211500Z.md`
- **Latest authoritative state (2026-06-01/02 codex cluster):**
  - `codex_findings_pact_vq_qat4_rate_solved_fit_blocked_20260602T005500Z_codex.md` (rate solved, fit blocked)
  - `codex_findings_upstream_eval_compact_vq_pivot_audit_20260602T033500Z_codex.md` (pivot_or_rebuild verdict; route to PR95/HiNeRV/SNeRV)
  - `codex_findings_pact_vq_competitiveness_gate_20260602T112634Z_codex.md` (PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION)
  - `codex_findings_compact_pact_vq_real_scorer_smoke_20260601T152304Z_codex.md` (real teacher bound; dark mean-field renders)
- Lane registry: `lane_pact_nerv_vq_l1_long_run_mlx_local_20260528`, `lane_pact_vq_projection_gap_repair_20260602`
- Canonical frontier (pointer-only): `.omx/state/canonical_frontier_pointer.json`
