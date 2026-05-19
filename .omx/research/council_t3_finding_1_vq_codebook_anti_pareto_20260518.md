---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, MacKay, Selfcomp, vanDenOord, Tao, Boyd]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The R-D theoretic optimum K=2 at lambda=1.0 is correct ONLY under the canonical IID-symbol assumption with squared error distortion. Substrate VQ codebooks operate on score-aware loss in BTCHW residual space where the distortion measure is NOT squared error — it's the contest scorer's PoseNet+SegNet gradient response. Without a paired empirical anchor that says 'K=2 wins at smoke on D4 substrate', the analytical Pareto claim is principled but unverified for the actual substrate geometry."
  - member: vanDenOord
    verbatim: "VQ-VAE codebook usage in practice (WaveNet+image generation) shows hugely-utilized large codebooks (K=512+); the R-D bound is achievable only when training signal is rich enough to populate the codebook. K=2 may collapse to mode-averaged outputs at the substrate level."
council_assumption_adversary_verdict:
  - assumption: "Wave 2A R-D optimum K=2 at lambda=1.0 generalizes from IID-symbol R-D analysis to substrate-VQ score-aware-loss regime"
    classification: CARGO-CULTED
    rationale: "The analytical R-D bound assumes squared-error distortion and IID symbol entropy; substrate VQ operates on score-aware loss where distortion is the SegNet/PoseNet response. The pole-in-the-Pareto-frontier finding is REAL (K=64, K=256 are dominated), but the K=2 OPTIMAL claim requires per-substrate empirical validation."
  - assumption: "Per-substrate VQ codebook wire-in across 14 substrates yields composite [-0.070, -0.014]"
    classification: HARD-EARNED
    rationale: "Per-substrate yield bound is derived from canonical 5% × R-D-gap × 14 substrates × λ-band; matches Catalog #233 promotion-gate evidence model. The composite bound is theoretically defensible."
council_decisions_recorded:
  - "op-routable #1: dispatch K-sweep paired-comparison smoke ($1-3 Modal T4) on D4 or sane_hnerv: K in {2, 4, 8, 16, 32, 64, 256} at lambda=1.0 to empirically validate the Pareto pole"
  - "op-routable #2: BEFORE substrate wire-in: per-substrate symposium per Catalog #325 to declare canonical-vs-unique decision for VQ codebook (some substrates may principled-fork)"
  - "op-routable #3: if K=2 empirically confirmed: 14-substrate wire-in wave with per-substrate cargo-cult-audit + 9-dim checklist per Catalog #294"
  - "op-routable #4: DEFER VQ-codebook autopilot-ranker pickup until empirical confirmation (do not consume Wave 2A analytical row as score-claim per Catalog #323)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_breaking
finding_action_class: research
finding_followup_dispatch_envelope_usd: 3.00
finding_canonical_path: analytical_solve
---

# Finding 1: VQ codebook K=64 + K=256 ANTI-PARETO at λ=1.0

## What happened

Wave 2A `8b987215a` (rows #2 + #3) reported the analytical Pareto-frontier solution for substrate VQ codebooks at λ=1.0 with d=4/64 dimensional embedding: **K=2 is optimal**, K=64 + K=256 (current hand-tuned defaults across 14+ substrates) are ANTI-PARETO (dominated by K=2 in both rate AND distortion axes under canonical IID R-D analysis).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323 canonical provenance: this is an **analytical** prediction (`evidence_grade=predicted`, `score_claim=false`, `promotion_eligible=false`). It cannot ship as a substrate-trainer wire-in without empirical paired-comparison validation per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".

## Council deliberation

### Shannon LEAD (operating-within: R-D theoretical floor IS achievable given sufficient training signal)
The R-D bound for VQ at λ=1.0 with d=4/64 is well-established: K=2 saturates the rate axis at log2(2)=1 bit/symbol while preserving the bulk of mutual information. Anything K>2 PAYS rate (log2(K) bits) for marginal distortion reduction in the canonical squared-error regime. The Wave 2A row #2+#3 finding IS theoretically correct under canonical assumptions. The question is whether substrate VQ codebooks operate UNDER canonical assumptions or in a regime where the bound doesn't apply.

### Dykstra CO-LEAD (operating-within: Pareto-feasibility region is convex; alternating projection finds the achievable frontier)
A point being inside the convex hull of the Pareto frontier (DOMINATED) is a structural property. Wave 2A's analytical solve shows (rate(K=64), distortion(K=64)) is DOMINATED by (rate(K=2), distortion(K=2)) under canonical R-D. The dominance is alternating-projections-confirmable. **However** if substrate VQ operates on score-aware-distortion (NOT squared-error), the Pareto frontier itself shifts — the dominance MAY survive but is not guaranteed. Cannot adopt without empirical confirmation.

### Yousfi (operating-within: contest scorer's PoseNet+SegNet response is the actual distortion measure)
Per the steganalysis-as-codec framing: the contest scorer IS the distortion oracle. K=2 codebook applied to substrate residuals may collapse the spatial-temporal signal that PoseNet/SegNet rely on for high-quality outputs. The R-D analysis assumes the distortion measure is the actual loss function; here the loss IS the contest scorer's gradient response, not squared error. K-sweep MUST be empirically validated at smoke before any 14-substrate wire-in.

### Fridrich (operating-within: inverse-steganalysis — codebook design embeds errors in scorer-blind-spots)
The Fridrich UNIWARD principle says: pay bits where the detector cannot see them, save bits where it can. A K=2 codebook may concentrate errors in regions PoseNet IS sensitive to (because it can only choose 2 quantization points). K=64+ codebooks can spread errors via per-residual lookup — paying log2(64)=6 bits per symbol but landing in steganalysis-blind regions. This is the OPPOSITE of the IID-R-D prediction.

### Contrarian (operating-within: the analytical result MUST survive substrate-empirical regime-change before adoption)
The R-D theoretic optimum K=2 at λ=1.0 is correct ONLY under the canonical IID-symbol assumption with squared error distortion. Substrate VQ codebooks operate on score-aware loss in BTCHW residual space where the distortion measure is NOT squared error — it's the contest scorer's PoseNet+SegNet gradient response. Without a paired empirical anchor that says 'K=2 wins at smoke on D4 substrate', the analytical Pareto claim is principled but unverified for the actual substrate geometry. **Veto on direct wire-in; allow research-only K-sweep dispatch.**

### Assumption-Adversary (operating-within: classify every shared assumption as HARD-EARNED or CARGO-CULTED)
The shared assumption operating under this deliberation is "R-D analytical results from canonical IID-symbol analysis transfer 1:1 to substrate VQ score-aware regimes." This is **CARGO-CULTED** per the hard-earned-vs-cargo-culted addendum. The yield estimate `[-0.070, -0.014]` composite is HARD-EARNED (matches per-substrate canonical 5% × λ-band × 14 = canonical). The K=2 OPTIMAL claim itself is CARGO-CULTED until empirical paired-comparison lands.

### MacKay (memorial; operating-within: MDL → encoder's bit-cost MUST account for codebook size)
Two-part code: codebook description bits (K × d × bits/coeff) PLUS per-symbol encoding bits (log2(K) per symbol). For substrate residuals at BTCHW scale, the codebook description is non-trivial; K=2 vs K=256 codebook description differs by ~2KB+ but per-symbol encoding differs by log2(256)−log2(2)=7 bits × N symbols. For large N the per-symbol term dominates; for small N (sparse residuals) the codebook term may dominate. **MDL says: empirical sweep, do not analytical-extrapolate.**

### Selfcomp (operating-within: per-pixel-residual entropy is non-IID; ANS/range coder squeezes spatial correlation)
PR101 selfcomp paradigm: residuals have HUGE spatial correlation; ANS coder squeezes log2(K) bits to closer to H(symbol). For K=256, ANS may achieve 2-3 bits/symbol (vs analytical 8). For K=2, ANS achieves ~1 bit/symbol (close to analytical). **The R-D analytical solve does NOT account for ANS post-coding** — actual rate at K=256 is much closer to K=2 than the analytical comparison suggests. The "anti-Pareto" claim may be **partially false** in the ANS-post-coded regime.

### vanDenOord (operating-within: VQ-VAE empirical codebook usage saturates around K=512)
WaveNet+image generation VQ-VAEs (van den Oord et al. 2017+) saturate codebook usage around K=512 with d=64; smaller K collapses to mode-averaged outputs. K=2 substrate-VQ may produce visually-degraded reconstructions that PoseNet/SegNet penalize. R-D analytical bound is achievable IF training signal is rich; substrate residuals MAY not be rich enough.

### Tao (operating-within: cross-domain analytical bound rigor)
The R-D bound K* = 2^H(symbol) is mathematically rigorous IF symbol distribution and distortion measure are fixed. The Wave 2A analytical solve assumes canonical IID symbols + squared-error distortion; substrate residuals may violate both. The mathematical content of the finding is "analytical bound K=2 IF (canonical assumptions); empirical K∈{2,4,...,256} sweep otherwise." The HARD-EARNED finding is the existence of the analytical solve apparatus; the CARGO-CULTED extrapolation is the unverified jump to substrate adoption.

### Boyd (operating-within: convex-optimization feasibility per Dykstra co-lead)
The Pareto-frontier claim is a convex-feasibility statement. ADMM/Dykstra alternating-projections CAN confirm K=2 dominance IF the score-aware-loss regime is convex. Substrate score-aware loss is NOT convex (PoseNet+SegNet are non-convex). The analytical solve is therefore **necessary but not sufficient**. Empirical sweep IS the canonical fallback per CLAUDE.md "Meta-Lagrangian/Pareto solver".

## Verdict + rationale

**PROCEED_WITH_REVISIONS**: pursue the finding via cheap empirical K-sweep BEFORE any substrate wire-in. Council is unanimous that the analytical Pareto pole IS a real-and-interesting finding but CANNOT be adopted as ground truth for substrate VQ without empirical confirmation. The Contrarian + vanDenOord + Selfcomp dissent specifically warns against direct adoption.

**Revisions binding before promotion**:
1. Dispatch K-sweep paired-comparison smoke ($1-3 Modal T4) on D4 substrate (or sister substrate that already wires VQ codebook); K in {2, 4, 8, 16, 32, 64, 256} at λ=1.0
2. Per-substrate symposium per Catalog #325 BEFORE 14-substrate wire-in
3. Cargo-cult-audit per Catalog #303 for the canonicalization decision (each substrate's VQ codebook is potentially fork-or-share)
4. DEFER autopilot-ranker pickup of Wave 2A row as score-claim until empirical confirmation per Catalog #323

## Action class + next-step dispatch

**research** (NOT pursue directly). Editor + $1-3 Modal T4 smoke dispatch. Critical-path: must complete BEFORE any of the 14 substrates' next wave-3 dispatch consumes VQ codebook defaults.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t3_finding_1_vq_codebook_anti_pareto_20260518", deliberation_id="finding_1_vq_codebook_anti_pareto", council_tier="T3", council_verdict="PROCEED_WITH_REVISIONS", predicted_impact_lower=-0.070, predicted_impact_upper=-0.014, cost_envelope_usd=3.00, memory_path=".omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md")`
- Posterior anchor: `tac.council_continual_learning.append_council_anchor(CouncilDeliberationRecord(deliberation_id="finding_1_vq_codebook_anti_pareto", topic="VQ codebook K=64/K=256 ANTI-PARETO at lambda=1.0", council_tier="T3", ...))`
- Probe outcome: `register_probe_outcome(probe_id="vq_codebook_k_anti_pareto_analytical_20260518", substrate="multi-substrate-vq-codebook", verdict="PARTIAL", metric_name="r_d_pareto_dominance_analytical", metric_value=1.0, evidence_path=".omx/research/arbitrariness_extinction_audit_20260518.jsonl", next_action="dispatch K-sweep empirical smoke before wire-in")`
- MEMORY.md index entry: yes (paired with deliberation wave landing entry)
- Cross-references: `feedback_findings_review_grand_council_deliberation_standing_directive_20260518.md` finding #1; `.omx/research/arbitrariness_extinction_audit_20260518.jsonl` rows #2+#3; Catalog #233 promotion gate; Catalog #303 cargo-cult audit; Catalog #325 per-substrate symposium

## Reactivation criteria (if K-sweep smoke confirms or refutes)

- **CONFIRMS K=2 optimal**: trigger 14-substrate wire-in wave per Catalog #325 per-substrate symposium pattern
- **REFUTES K=2 optimal**: register canonical "analytical R-D doesn't transfer to substrate-VQ score-aware regime" finding to `tac.atom` posterior; update Wave 2A row schema to mark `regime_transfer_invalid: true`
- **MIXED RESULTS** (K=8 wins on some, K=64 on others): per-substrate canonical-vs-unique decision per Catalog #290; codebook K becomes a substrate-design parameter, not a global default
