# Master-gradient extractor tour

A tool for measuring per-pair / per-byte score-response sensitivities on a fixed substrate operating point. The extractor emits diagnostic tensors that downstream consumers (per-pair bit allocators, candidate rankers, cargo-cult unwind audits, asymptotic-pursuit readiness checks) ingest as side information for the substrate-design loop.

Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) Section E.1, [`canonical_equations_tour.md`](canonical_equations_tour.md) equations 4 and 5. Sister library: [`adpena/tac`](https://github.com/adpena/tac). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## What it is

The extractor runs 1 forward pass + 3 backward passes through the contest's frozen scorer (SegNet + PoseNet weights) against a substrate's decoder output, returning per-parameter and per-byte gradients of the score components (`d_seg`, `d_pose`, rate) with respect to the substrate's weights and the archive's bytes.

The math:

- **Forward**: `z_latents (frozen) -> decoder(weights) -> rgb_pair[0:eval_size]`
- **Eval-roundtrip**: bicubic upscale to 874x1164, then bilinear roundtrip down to 384x512 (per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" — the contest scorer evaluates at the post-roundtrip resolution; gradients must too).
- **Scorers**: SegNet preprocess (last-frame slice + bilinear-to-(384, 512)) → forward → `d_seg`; PoseNet preprocess (differentiable RGB-to-YUV6 + 2-frame stack) → forward → `d_pose`.
- **Backward(d_seg)**: per-parameter `∂(d_seg) / ∂(θ)`. SegNet's last-frame slice means only frame-1 weights receive gradient flow.
- **Backward(d_pose)**: per-parameter `∂(d_pose) / ∂(θ)`. PoseNet uses both frames so both get gradient flow.
- **Rate**: analytical, derived from archive byte counts (the contest rate term is `25 · |archive.zip| / 37545489`, with closed-form derivatives at fixed archive layout).

The fec6 codec Jacobian projects per-parameter gradients onto per-archive-byte gradients via the codec's parametric layer (per-tensor scale × per-element mantissa). For tensors using per-tensor float-16 scales with int8 mantissas, the per-byte gradient is `∂(score)/∂(mantissa_byte[i]) = ∂(score)/∂(w[i]) · scale_fp16` (and `∂(score)/∂(scale) = sum_i mantissa[i] · ∂(score)/∂(w[i])`).

## The 3 surfaces

The extractor emits gradients at three operating points along the encode-decode pipeline:

1. **`M_contest`** — gradient of the contest score w.r.t. substrate weights at the substrate's natural parameterization. The most directly actionable for substrate fine-tuning.
2. **`M_archive`** — gradient projected to the per-archive-byte grain via the codec Jacobian (per-tensor scale × mantissa). Used for per-byte bit-allocation reasoning, with the locality-violation caveat from equation #5 of the canonical equations tour.
3. **`M_inflated`** — gradient at the post-decompress / post-decode grain (per-symbol on arithmetic-coded streams; per-pixel on rendered RGB output). The canonical Lipschitz domain for byte-level master-gradient claims.

The three surfaces are not interchangeable. `M_contest` is for fine-tuning. `M_archive` carries the codec locality caveat. `M_inflated` is the canonical basis for any cross-substrate sensitivity comparison.

## The 10 exploits

Each exploit consumes the extractor's output for a different purpose. None require dispatching another paid run; they all read from the canonical anchor JSONL the extractor writes.

1. **Per-pair difficulty atlas.** Per-pair gradient L1 norms identify which frame pairs the substrate's operating point handles well and which it struggles with. The atlas drives per-pair bit-allocation: pairs with large gradient norm are by Cauchy-Schwarz (equation 4) the largest score-impact targets for marginal rate.

2. **Score-weighted reconstruction error.** Instead of optimizing `||x̂ - x||²` (mean-squared pixel error, which the contest does NOT score on), optimize the score-weighted variant `||M_inflated · (x̂ - x)||²` where the master-gradient acts as a per-pixel saliency mask. Aligns the proxy training objective with the actual score gradient.

3. **Top-K byte ranking.** The K bytes with largest `|M_archive|` are the most leverage-bearing bytes. Useful for sanity-checking proposed byte-level edits before dispatching them: if a proposed edit's byte set has near-zero gradient sum, the edit is unlikely to move score regardless of its byte-count savings.

4. **Bottom-K free-entropy bytes.** Conversely, the K bytes with smallest `|M_archive|` are candidates for compression: they can be coarsened or dropped without measurable score impact. Useful for asymptotic-pursuit substrate scaling.

5. **Per-class chroma allocation.** Applied during the NSCS06 v6 → v7 cargo-cult unwind iteration: the per-pair gradient at the chroma channels identified which color-class structure the contest scorer was sensitive to, motivating the per-class chroma anchor that replaced the v6 cargo-culted global chroma replication. Anchor: 44% score reduction in one iteration (`105.15 → 58.89 [contest-CUDA T4]`).

6. **Substrate-fit diagnostic.** The gradient distribution across substrate weight tensors flags whether the substrate is allocating capacity efficiently. Tensors with nearly-zero gradient norm at the operating point are over-parameterized; tensors with nearly-saturated gradient norm are under-parameterized for the score axis they target.

7. **Information-theoretic floor estimate (Cramér-Rao).** The per-parameter Fisher information (proportional to the squared gradient norm) lower-bounds the variance of any unbiased estimator at that operating point. Aggregated across substrate weights, this gives an information-theoretic estimate of the local-floor of the substrate's reachable score, without dispatching a full sweep.

8. **Bit-level score-critical bits.** A per-byte gradient extended to per-bit identifies which specific bits within the high-leverage bytes carry the score signal. Useful for designing fixed-Huffman / arithmetic-coding tables that align symbol probability with bit-level score sensitivity.

9. **Per-pair gradient clustering (symmetry-breaking).** Per-pair gradient vectors cluster into modes (e.g., highway-driving pairs vs urban-driving pairs cluster differently). The clustering identifies sub-distributions the substrate's single-mode parameterization is underfitting, motivating per-cluster substrate variants or mixture-of-experts decompositions.

10. **Streaming master-gradient during training (Taylor expansion real-time prediction).** Per the canonical Taylor + Cauchy-Schwarz bound (equation 4), the master-gradient at the current operating point upper-bounds the per-pair score impact of any small weight update at the next step. Streaming the gradient during training enables online prediction of which gradient-descent step will move score most, supporting adaptive learning-rate schedules.

## Canonical authoritative anchor (2026-05-20)

The first `[contest-CUDA T4]` authoritative master-gradient anchor for the FEC6 frontier archive (PR #110, sha `6bae0201fb08...`) landed via Modal call `fc-01KS370Z9TF4QZMKQ9ND72KH4N` on 2026-05-20 (n_pairs_used=600 / full set; sidecar `.omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy`, shape `(178417, 3)`, fp32). The anchor is the canonical posterior source any downstream consumer should query via:

```python
from tac.master_gradient import latest_anchor_for_archive
from tac.master_gradient_consumers import load_aggregate_gradient_from_anchor

FEC6_SHA = '6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf'
anchor = latest_anchor_for_archive(FEC6_SHA, axis='[contest-CUDA]')
gradient_array, anchor_dict = load_aggregate_gradient_from_anchor(archive_sha256=FEC6_SHA)
# gradient_array.shape == (178417, 3); axes = (seg, pose, rate)
```

The 600-pair operating point recorded in the anchor is `{d_seg=0.001, d_pose=0.00381654, rate=0.004755, score=0.4175}`; per-byte sensitivity is dominated by the seg axis (78.5%) with secondary pose contribution (12.4%) per the recorded `score_axis_dominance` payload.

### Downstream consumer wire-in (Catalog #125 hook #3 + #4)

Three cathedral consumers already auto-discover this anchor via the canonical `latest_anchor_for_archive` + `load_aggregate_gradient_from_anchor` loaders (per Catalog #335 auto-discovery + Catalog #344 canonical equations registry):

- `tac.cathedral_consumers.canonical_equation_lookup_consumer` annotates fec6 candidates with the 3 registered equations whose `canonical_consumers` overlap the candidate's substrate / archive tokens (`brotli_cascade_bounded_per_stream_v1`, `per_byte_leverage_uniformly_distributed_v1`, `master_gradient_locality_violation_by_codec_v1`).
- `tac.cathedral_consumers.information_theoretic_floor_consumer` (Tier B per Catalog #357) consumes the per-axis Fisher information density from the (178417, 3) tensor to compute the Cramér-Rao floor + emits per-axis decomposition into the ranker.
- `tac.cathedral_consumers.substrate_fit_diagnostic_consumer` (Tier B per Catalog #357) consumes per-axis residuals against the T4 anchor's operating point + emits seg/pose decomposition for substrate-class disambiguation.

The bit-allocator hook (Catalog #125 hook #3) consumes the same anchor directly via the new canonical helper:

```python
from tac.bit_allocator import allocate_per_byte_from_master_gradient_anchor, PerByteAllocationMethod

plan = allocate_per_byte_from_master_gradient_anchor(
    total_budget_bits=256,
    archive_sha256=FEC6_SHA,
    axis_aggregator='score_weighted_sum',  # canonical contest formula coefficients
    method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
    top_k=32,
    per_byte_bit_cap=8,
)
# plan.notes['master_gradient_anchor'] carries the anchor's canonical metadata
# (measurement_axis=[contest-CUDA], measurement_hardware=linux_x86_64_t4_modal,
# measurement_call_id=fc-01KS370Z9TF4QZMKQ9ND72KH4N, n_pairs_used=600).
# Returns observability-only PerByteAllocationPlan per Catalog #341
# (score_claim=False, promotion_eligible=False, axis_tag="[predicted]").
```

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the bit-allocator output is non-promotable by construction; realizing the allocation as actual archive bytes + paired-axis Linux x86_64 auth-eval is required before any contest score claim.

## Example outputs

The canonical state for the FEC6 frontier archive (PR #110, sha `6bae0201fb08...`) lives in the registry's anchor JSONL. A representative excerpt of what the extractor returns (per the [canonical equations tour](canonical_equations_tour.md) equation #3 anchor):

- Per-pair gradient L1 distribution: highly non-uniform; top-3% of pairs (≈18 pairs out of 600) carry ~30% of total gradient norm.
- Top-1% per-byte cumulative leverage: ~6.4% (below the naive `K/N` prediction of 1% but far from the 90% one would naively expect of "important" bytes — equation #3's empirical anchor).
- Per-tensor gradient distribution: dominated by the decoder's stem and the first two encoder blocks; later blocks contribute < 1% of total gradient norm at the operating point.

Anchors for PR101 (`hnerv_ft_microcodec`) and PR106 (`hnerv_lc_ac`) are also registered for cross-substrate comparison via the cross-substrate gradient correlation matrix (exploit #9's clustering plot).

A user who wants to reproduce these numbers can run the extractor against any contest-compliant archive; the canonical helper at `tools/extract_master_gradient.py` consumes a packet path + an output anchor path and emits the per-pair / per-byte / per-tensor breakdowns.

## How to use it for your own substrate design

1. **Train your substrate to a checkpoint.** Build the contest archive; verify its SHA matches what your trainer produced.
2. **Run the extractor on your archive.** Point `tools/extract_master_gradient.py` at the archive path; specify an anchor JSONL output path under your own results directory.
3. **Read the anchor.** The anchor carries per-pair gradients (use exploit #1 to identify high-leverage pairs), per-byte gradients (use exploit #3 + #4 to identify high- and low-leverage bytes), and per-tensor gradients (use exploit #6 for substrate-fit diagnostics).
4. **Choose your next experiment.** If per-pair gradients are highly concentrated, exploit #2's score-weighted reconstruction loss is likely the next win. If per-tensor gradients show under-utilization, consider tensor-level structural changes (more capacity in saturated tensors, less in starved ones). If the per-byte gradient distribution is approximately uniform (equation #3 holds), substrate-class shifts will likely dominate per-byte edits — that is the signal to pivot to a paradigm-class candidate from the asymptotic-floor candidate inventory rather than further within-class bolt-ons.
5. **Re-extract after each change.** The gradient is operating-point dependent; what looked promising at one operating point may not at the next.

## Honest scope

The extractor is a tool, not a contest-score primitive by itself. Running it does not move score; using its output to guide substrate design might. The 10 exploits are observations about what the gradient surfaces enable; each one requires substrate-design work downstream to realize as a score reduction.

The canonical caveats apply: the per-byte master-gradient on entropy-coded archives carries the locality violation from equation #5 (use `M_inflated`, not `M_archive`, for cross-substrate sensitivity comparison); the per-pair Cauchy-Schwarz bound from equation #4 is an upper bound, not a closed-form prediction; the gradient is computed at one operating point and does not extrapolate to substantially different operating points without re-extraction.

Sister library [`adpena/tac`](https://github.com/adpena/tac) carries the canonical master-gradient consumers (`tac.master_gradient_consumers` for the per-pair / aggregate / per-X interfaces; `tac.cathedral_consumers.per_pair_difficulty_atlas_consumer` for the per-pair difficulty atlas exploit at the cathedral-autopilot ranker surface). Reproducing the extractor's full toolchain against a fresh substrate requires both the extractor (in this repo's `tools/`) and the consumer interfaces (in the sister library) — both are MIT-licensed.
