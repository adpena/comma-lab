<!-- SPDX-License-Identifier: MIT -->
<!-- DOCS_LOCAL_PATH_OK:no_local_absolute_paths_used_in_this_memo_per_Catalog_208 -->
<!-- # HISTORICAL_SCORE_LITERAL_OK:design_memo_no_frontier_score_literals_only_predicted_bands_per_catalog_343 -->
---
council_tier: T1
council_attendees: [Wyner, Yousfi]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The existing Z8 M6 wyner_ziv_coder side_info wired from generic top-LL spatial mean IS canonically equivalent to PoseNet 6-dim decoder-side conditioning"
    classification: CARGO-CULTED
    classification_rationale: "Wyner-Ziv 1976 Theorem 1 R(X|Y) <= R(X) holds for ANY decoder-reproducible Y, but the conditional-entropy reduction H(X) - H(X|Y) is maximized when Y is maximally informative about X under the SCORER's distortion measure. The contest distortion is PoseNet 6-dim MSE + SegNet argmax; a generic top-LL spatial mean is a weak, scorer-agnostic Y whose I(X;Y) is far below I(X; PoseNet_6dim) on identical pair frames. Per CLAUDE.md 'HNeRV / leaderboard-implementation parity discipline' L6 (score-domain Lagrangian not weight-domain proxies) + L8 (scorer-preprocess gradient reachability): the canonical Y for a pose-axis latent codec is the SCORER's own pose head output, NOT a hand-picked spatial statistic. The HARD-EARNED canonical claim: Y = PoseNet_6dim(frame_0, frame_1) is decoder-reproducible (PoseNet is part of the contest scorer, not substrate weights) AND maximally aligned with the pose distortion the rate term must trade against."
  empirical_verification_status:
    - assumption: "PoseNet 6-dim output is decoder-reproducible at inflate time on identical pair frames"
      status: INFERRED_FROM_DOMAIN_LITERATURE
      evidence: "PoseNet is part of upstream/modules.py contest scorer (FastViT-T12 12-channel head, first 6 pose dims scored) per CLAUDE.md 'Exact scorer architectures'; it is NOT shipped in archive.zip per CLAUDE.md 'Strict scorer rule'. Encoder and decoder both have the identical contest scorer on identical pair frames, so PoseNet(frame_0, frame_1) is bit-reproducible side-information Y at inflate time. GATE: needs a real WZ-PoseNet-side-info dispatch to land an empirical anchor confirming the predicted conditional-entropy reduction; until then FORMALIZATION_PENDING per Catalog #344."
    - assumption: "Conditioning the per-pair latent codec on Y_posenet yields net rate savings exceeding the side-info-derivation overhead (which is ZERO bytes since Y is decoder-reproducible)"
      status: ASSUMED_AWAITING_VERIFICATION
      evidence: "Wyner-Ziv side-info is FREE in bytes (decoder reproduces it), so the rate hurdle per canonical equation foveation_sidecar_bolt_on_rate_hurdle_v1 is ZERO for the Y channel. Net savings = H(latent) - H(latent | Y_posenet) bits/pair * N_pairs / 8 bytes, minus the encoder-side residual stream that codes the conditional. The ASSUMED part: that the conditional residual stream is smaller than the unconditioned latent stream by more than any model-overhead. This is exactly what a real dispatch must measure."
related_canonical_equation: wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1
related_canonical_equation_registered_utc: 2026-05-30T16:04:07Z
related_canonical_equation_registry_line: 371
horizon_class: frontier_pursuit
---

# Wyner-Ziv decoder-side PoseNet 6-dim side-information: M6 wiring-gap operator-routable + canonical equation completion

## Search-first inventory (Catalog #229 + #340 + #378 premise verification BEFORE edit)

Per the mandatory pre-flight, I grepped the canonical equations registry + searched the
research tree BEFORE writing any code:

- **`tools/list_canonical_equations.py --json | grep -i wyner`** + direct registry
  enumeration (`.omx/state/canonical_equations_registry.jsonl`, 7 distinct wyner-ziv
  equation ids) confirmed the target equation **ALREADY EXISTS** at registry line 371:
  `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1`,
  `event=registered`, `utc=2026-05-30T16:04:07Z` (earlier today; a predecessor partial
  landing in this same session-day).

- The existing equation is **canonically correct**: name = "Wyner-Ziv decoder-side
  PoseNet 6-dim side-information conditional-entropy reduction R(X|Y_posenet) << R(X)";
  latex = `R(X | Y_posenet) <= R(X), Y_posenet = PoseNet_6dim(frame_0, frame_1)
  decoder-reproducible`; `posenet_side_information_dim = 6`;
  `wyner_ziv_theorem_reference = "Wyner-Ziv 1976 Theorem 1: R(X|Y) <= R(X) when Y
  available at decoder"`; producers = `src/tac/codec/wyner_ziv_layer.py` +
  `src/tac/substrates/d4_wyner_ziv_frame_0/score_aware_loss.py`; `empirical_anchors = []`
  (correctly 0); `notes =
  FORMALIZATION_PENDING:wyner_ziv_decoder_side_posenet_6dim_conditional_entropy_no_empirical_anchor_until_real_wz_posenet_side_info_dispatch_lands_per_catalog_344`.

- The existing equation's `out_of_domain_contexts` ALREADY encodes the M6 distinction:
  it lists `generic_spatial_mean_side_info_NOT_canonical_posenet_6dim` as explicitly
  OUT of domain. So the predecessor row already captured the canonical claim that a
  generic spatial-mean Y is NOT the canonical PoseNet-6dim Y.

### Decision per Catalog #287 (HONEST, no fake / no duplicate)

**DO NOT register a new equation or duplicate line 371.** Per CLAUDE.md
"Subagent coherence-by-default" anti-duplication primitive + Slot EEE Class 5
(enum-padding / redundant-sister-without-distinct-implementation): registering a
second equation that captures the same `R(X|Y_posenet) << R(X)` claim would be a
forbidden duplicate, and would corrupt the canonical posterior with two competing
priors for one phenomenon.

**The actual remaining work is to COMPLETE the predecessor's PARTIAL landing.** The
equation row exists, but a search of `.omx/research/`, the memory directory, the lane
registry (`.omx/state/lane_registry.json`), and the probe-outcomes ledger
(`.omx/state/probe_outcomes.jsonl`) found NO landing memo, NO lane registry entry, and
NO probe outcome for this equation. It was an equation-row-only partial landing. This
memo + the sister landing memo + lane registry + probe outcome + sister tests close
that gap WITHOUT mutating the existing APPEND-ONLY equation row (per Catalog #110/#113).

Sister-DISJOINT scope confirmed per Catalog #340: concurrent subagents are working on
UNIWARD wire-in (composition + bit_allocator), DreamerV3 L1 (dreamer_v3_rssm), and a
running Z7 process. This memo touches ONLY `.omx/research/`, lane registry, probe
outcomes, and the canonical equations registry via the canonical helper — no substrate
code, no composition, no bit_allocator, no dreamer_v3_rssm, no z7_mamba2.

## The Wyner-Ziv 1976 Theorem 1 math (rigor anchor)

**Wyner & Ziv (1976), "The rate-distortion function for source coding with side
information at the decoder," IEEE Trans. Inf. Theory IT-22(1):1-10, Theorem 1.**

For a source X and side-information Y jointly distributed as p(x, y), where Y is
available at the DECODER but not the ENCODER, the rate-distortion function is

```
    R_{WZ}(D) = min_{ p(u|x), f } [ I(X;U) - I(Y;U) ]
```

subject to a distortion constraint `E[d(X, f(U, Y))] <= D`, where U is an auxiliary
random variable. The key consequence relevant here: **when Y is correlated with X, the
conditional rate R(X|Y) is strictly below the unconditional rate R(X)** —
`R(X|Y) <= R(X)`, with equality iff `I(X;Y) = 0`. The savings is governed by the
mutual information `I(X;Y)`, equivalently the conditional-entropy reduction
`H(X) - H(X|Y)`.

### The decoder-reproducibility property (why this is FREE for our codec)

In the canonical Wyner-Ziv setting, Y must be transmitted or otherwise known at the
decoder. In our contest setting, **Y_posenet is decoder-reproducible at ZERO byte
cost**:

1. `Y_posenet = PoseNet_6dim(frame_0, frame_1)` — the first 6 pose dimensions of the
   contest scorer's FastViT-T12 hydra head, run on the pair frames.
2. PoseNet is part of the contest SCORER (`upstream/modules.py`), NOT substrate weights
   shipped in `archive.zip` (CLAUDE.md "Strict scorer rule" — loading scorers at inflate
   destroys the rate term, so scorers are NEVER in the archive; they live in the eval
   harness on both encoder and decoder sides).
3. Therefore at inflate time, the decoder has the identical contest scorer and the
   identical reconstructed pair frames, so it can recompute `PoseNet_6dim(frame_0,
   frame_1)` byte-for-byte. No bytes are added to the archive for the Y channel.

This makes the conditional-entropy reduction `H(latent) - H(latent | Y_posenet)` a
**pure rate-axis savings** with ZERO offsetting rate hurdle (the rate hurdle per
canonical equation `foveation_sidecar_bolt_on_rate_hurdle_v1` is `+25*N/37545489`, and
N=0 bytes for a decoder-reproducible Y).

NOTE the subtle dependency: at inflate time the decoder needs Y_posenet to DECODE the
latent, but Y_posenet requires the reconstructed frames, which require the decoded
latent — a chicken-and-egg. The canonical resolution (per Wyner-Ziv binning / coset
coding) is that the encoder codes X in cosets indexed so the decoder resolves the coset
using Y_posenet computed from a BASE reconstruction (e.g. frame_0 alone, or a coarse
prior frame), then refines. The exact binning structure is an IMPLEMENTATION detail the
real dispatch must validate; the equation captures the upper bound `R(X|Y) <= R(X)`.

## The M6 side-info wiring-gap (the canonical operator-routable)

### What today's Yousfi-voice review (Axis 3) found

Per the Yousfi-voice canonical inverse-steganalysis review of Z8 M12a
(`.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md`),
Axis 3 found that the EXISTING Z8 M6 `wyner_ziv_coder` wires the decoder-side-information
Y from a **generic top-LL spatial mean** (a scorer-agnostic spatial statistic), NOT the
canonical PoseNet 6-dim pose output. The canonical equation #371 intent is correct
(`R(X|Y_posenet) << R(X)`), but the WIRING is lower-canonical: the substrate computes a
weak Y rather than the scorer-aligned Y the equation names.

### Why the gap matters (information-theoretic argument)

The conditional-entropy reduction `H(X) - H(X|Y)` is monotone in `I(X;Y)`. For a
per-pair pose-axis latent X (the residual the codec must transmit to reconstruct the
pose-relevant content of the pair):

- **Generic top-LL spatial mean Y_spatial**: a coarse luminance statistic. `I(X;
  Y_spatial)` is small because X is the pose-distortion-relevant residual and a spatial
  mean carries little pose information. The conditional rate `R(X | Y_spatial)` is only
  marginally below `R(X)`.
- **Canonical PoseNet 6-dim Y_posenet**: by construction the 6-dim pose embedding the
  contest distortion measure is computed from. `I(X; Y_posenet)` is large because
  Y_posenet IS the sufficient statistic for the pose distortion the rate must trade
  against. The conditional rate `R(X | Y_posenet)` approaches the Wyner-Ziv floor.

In short: **the equation already names the right Y; the substrate wiring uses the wrong
Y.** The fix is to rewire the M6 coder's `side_info` input from
`top_ll_spatial_mean(frames)` to `posenet_6dim(frame_0, frame_1)` via the canonical
scorer-preprocess path (the same path PR #95/#106 use, with differentiable YUV6 +
gradient-reachable PoseNet per CLAUDE.md "eval_roundtrip" + L8).

### Operator-routable (for the Z8/Z6-v2 hierarchical-PC composition wave)

This is a canonical operator-routable per CLAUDE.md "Results must become system
intelligence" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch":

> **OP-ROUTABLE WZ-M6-1**: Before the next Z8 M6 / Z6-v2 hierarchical-PC paid Modal
> dispatch that claims Wyner-Ziv decoder-side conditioning, rewire the M6
> `wyner_ziv_coder` `side_info` input from the generic top-LL spatial mean to the
> canonical PoseNet 6-dim decoder-reproducible Y per equation #371's
> `in_domain_contexts` (`z8_hierarchical_pc_posenet_side_info`,
> `z6_v2_rao_ballard_posenet_side_info`). The rewire MUST route through the canonical
> scorer-preprocess path (`preprocess_input` per Catalog #164; gradient-reachable
> differentiable YUV6 per CLAUDE.md eval_roundtrip non-negotiable) so the
> training-time conditional residual is computed against the SAME Y the inflate-time
> decoder reproduces. The current generic-spatial-mean wiring falls in equation #371's
> `out_of_domain_contexts` (`generic_spatial_mean_side_info_NOT_canonical_posenet_6dim`)
> — it is NOT the canonical claim and must not be cited as a Wyner-Ziv-PoseNet anchor.

> **OP-ROUTABLE WZ-M6-2** (the empirical anchor that lifts FORMALIZATION_PENDING):
> A real WZ-PoseNet-side-info paired dispatch (encode against the contest video with
> the rewired M6 coder, measure `H(latent) - H(latent | Y_posenet)` empirically as
> wire-byte savings, run paired CUDA+CPU auth eval per Catalog #246) will land the
> FIRST empirical anchor for equation #371 via
> `tac.canonical_equations.update_equation_with_empirical_anchor`. Until then the
> equation correctly stays FORMALIZATION_PENDING per Catalog #344 (no fake anchor per
> Slot EEE Class 3 — synthetic-fixture-instead-of-real-input is forbidden; the anchor
> must measure the actual `upstream/videos/0.mkv` pair frames).

## Canonical-vs-unique decision per layer (Catalog #290)

This memo is a documentation + canonical-equation-completion landing, not a new
substrate. The single layer with a canonical-vs-unique decision is:

- **Canonical equation registry** — ADOPT_CANONICAL_BECAUSE_SERVES. The predecessor
  equation row at line 371 IS the canonical surface; this landing completes its
  partial wire-in (memo + lane + probe + tests) rather than forking a sister. Forking
  would be a forbidden duplicate per anti-duplication.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — Wyner-Ziv decoder-side PoseNet-6dim conditioning is a class-shift
   from within-class entropy-coder refinement: it exploits the scorer-as-shared-prior
   structure (decoder reproduces the scorer's pose head) for FREE conditional-entropy
   reduction. Distinct from generic spatial-mean side-info.
2. **BEAUTY + ELEGANCE** — the canonical claim is one line (`R(X|Y_posenet) <= R(X)`,
   Y decoder-reproducible) and the equation row is reviewable in 30 seconds.
3. **DISTINCTNESS** — explicitly distinct from sister equations #129 (general WZ
   decoder-side), #210 (per-pair pose-axis savings), #173 (pipeline-stage codec):
   #371 is the SPECIFIC PoseNet-6dim conditioning claim with the M6 wiring distinction
   encoded in its out-of-domain contexts.
4. **RIGOR** — Wyner-Ziv 1976 Theorem 1 citation + the decoder-reproducibility property
   + the I(X;Y) monotonicity argument for the M6 gap + HONEST FORMALIZATION_PENDING
   (no empirical anchor fabricated).
5. **OPTIMIZATION PER TECHNIQUE** — the equation routes the M6 coder toward the
   scorer-aligned Y that maximizes I(X;Y), not the path-of-least-resistance spatial mean.
6. **STACK-OF-STACKS COMPOSABILITY** — `in_domain_contexts` lists the Z8 hierarchical-PC
   + Z6-v2 Rao-Ballard composition surfaces; the WZ-PoseNet conditioning is an
   orthogonal rate-axis primitive that composes with the predictive-coding substrates.
7. **DETERMINISTIC REPRODUCIBILITY** — Y_posenet is bit-reproducible at the decoder
   (identical scorer + identical pair frames); the equation registry is APPEND-ONLY +
   fcntl-locked.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — the Y channel is ZERO bytes (decoder
   reproduces it), so the conditional-entropy reduction is pure rate-axis savings with
   no offsetting hurdle.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted rate-axis savings `H(latent) -
   H(latent | Y_posenet)` bits/pair * N_pairs / 8 bytes; HONEST: predicted band
   FORMALIZATION_PENDING until a real dispatch measures it (no phantom predicted-band
   per Catalog #296 / #324).

## Observability surface (Catalog #305)

- **inspectable per layer** — the WZ codec's per-pair conditional residual + Y_posenet
  per pair are inspectable via the codec's debug hooks; the equation row is queryable
  via `tools/list_canonical_equations.py --json`.
- **decomposable per signal** — rate savings decomposes per-pair into `H(latent_p) -
  H(latent_p | Y_posenet_p)`.
- **diff-able across runs** — equation registry is APPEND-ONLY; recalibration events are
  diff-able rows.
- **queryable post-hoc** — registry JSONL + this memo + the lane registry row.
- **cite-able** — equation_id `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1`
  + registry line 371 + this memo + Wyner-Ziv 1976 Theorem 1.
- **counterfactual-able** — the M6 gap IS a counterfactual: "what if the M6 coder used
  Y_posenet instead of Y_spatial?" The answer is the predicted conditional-entropy
  reduction the OP-ROUTABLE WZ-M6-2 dispatch will measure.

## Predicted band + Dykstra-feasibility (Catalog #296)

Predicted rate-axis savings: `Delta_rate_bytes = (H(latent) - H(latent | Y_posenet)) *
N_pairs / 8`, a Wyner-Ziv lower-bound (Shannon R(D) + Wyner-Ziv 1976 Theorem 1
first-principles bound, NOT a vibes band). The conditional-entropy reduction is bounded
above by `I(X; Y_posenet) <= H(Y_posenet) <= 6 * 32 bits` (6 fp32 pose dims) per pair,
but in practice far smaller. EXACT band is FORMALIZATION_PENDING per Catalog #324 (no
post-training Tier-C measurement exists yet); the OP-ROUTABLE WZ-M6-2 dispatch lands it.
The feasibility is a single rate-axis constraint (WZ conditional coding lowers the rate
term while holding pose distortion); per Dykstra this is the intersection of {rate <=
R_WZ(D)} with the existing pose/seg/archive feasible sets — a one-constraint tightening,
trivially feasible.

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map** — N/A at this documentation-landing surface; the WZ codec
  itself surfaces per-pair conditional residual sensitivity when wired (OP-ROUTABLE
  WZ-M6-1).
- **hook #2 Pareto constraint** — ACTIVE. Wyner-Ziv conditional coding is a rate-axis
  Pareto constraint: `R(X|Y_posenet) <= R(X)` tightens the rate term while holding the
  pose/seg distortion constraints. The equation's `measurement_axes` =
  `[contest-CUDA]`, `[contest-CPU]` route the rate-axis savings into the Pareto polytope.
- **hook #3 bit-allocator** — N/A at this surface; the rewired M6 coder (OP-ROUTABLE
  WZ-M6-1) would feed per-pair conditional bit allocation.
- **hook #4 cathedral autopilot dispatch** — ACTIVE via the existing equation's
  consumer `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered
  per Catalog #335); the cathedral ranker can surface the WZ-PoseNet conditioning
  prediction as an observability-only annotation.
- **hook #5 continual-learning posterior** — ACTIVE. The equation lives in the canonical
  posterior (`.omx/state/canonical_equations_registry.jsonl`); the OP-ROUTABLE WZ-M6-2
  empirical anchor flows back via `update_equation_with_empirical_anchor` (which triggers
  Catalog #371 auto-recalibration once 3+ in-domain anchors land).
- **hook #6 probe-disambiguator** — ACTIVE. The M6 gap IS the probe-disambiguator
  between two defensible Y choices (generic spatial mean vs canonical PoseNet 6-dim);
  the equation's `in_domain` / `out_of_domain` contexts ARE the canonical
  disambiguation, and OP-ROUTABLE WZ-M6-2 resolves it empirically.

## Cross-references

- Equation #371 `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1`
  (registry line 371; the predecessor partial landing this completes).
- Sister equations #129 (general WZ decoder-side conditional entropy), #210 (per-pair
  pose-axis WZ savings), #173 (pipeline-stage codec decoder-side Y).
- Yousfi-voice review Axis 3:
  `.omx/research/council_yousfi_voice_canonical_inverse_steganalysis_review_z8_m12a_modal_t4_l2_long_training_pre_dispatch_20260530.md`.
- `src/tac/codec/wyner_ziv_layer.py` (canonical Wyner-Ziv 1976 Theorem 1 codec; the M6
  coder's rewire target).
- CLAUDE.md "Canonical equations + models registry" (Catalog #344), "HNeRV /
  leaderboard-implementation parity discipline" L6 + L8, "Strict scorer rule",
  "eval_roundtrip", "Meta-Lagrangian/Pareto solver".
- Wyner, A. D. & Ziv, J. (1976). The rate-distortion function for source coding with
  side information at the decoder. IEEE Trans. Inf. Theory, 22(1), 1-10.
