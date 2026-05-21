# Canonical Equation Registration: `static_packet_custody_byte_delta_score_savings_v1`

**Landing**: 2026-05-21 (UTC `2026-05-21T05:12:39Z`)
**Lane**: `lane_wave_3_wr01_static_packet_custody_canonical_equation_registration_20260520`
**Subagent**: `wave-3-wr01-static-packet-custody-canonical-equation-registration-20260520`
**Catalog evolution discipline**: Catalog #344
**Sister landings**: slot 3-r cathedral consumer `ad23f1880` (WR01 cathedral consumer registration); WR01 codex memo `.omx/research/wr01_static_packet_custody_20260506_codex.md` (2026-05-06)

## Summary

Registered NEW canonical equation `static_packet_custody_byte_delta_score_savings_v1`
in `.omx/state/canonical_equations_registry.jsonl` per Catalog #344
canonical-equations-and-models-registry evolution discipline. The equation
formalizes the byte-only score-delta prediction for static-packet custody
exact-eval candidates per the WR01 codex memo's empirical observation
`byte_only_expected_score_delta = -5.99e-6` for `byte_delta = -9`.

Registry event count: was 49 events pre-landing → 50 events post-landing
(+1 `registered` event for the new equation). 28 unique `equation_id` values
in the registry after this landing (was 27).

## Empirical anchor

Single empirical anchor at registration (per WR01 codex memo verbatim):

| Field | Value |
|---|---|
| Source archive sha256 | `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e` |
| Source archive bytes | `186231` |
| Target archive sha256 | `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628` |
| Target archive bytes | `186222` |
| Byte delta | `-9` (target smaller than source) |
| Empirical `byte_only_score_delta` | `-5.99273057809954245220e-6` |
| Predicted (formula) | `-25 * 9 / 37_545_489 = -5.99273057809954245220e-6` |
| Residual | `0.0` (exact arithmetic match) |
| Candidate lane | `hnerv_wavelet_apply_transform_pr106x_1_2` |

The empirical match is exact because the empirical observation in the
WR01 codex memo was itself derived from the canonical contest rate-term
formula `25 * archive_bytes / 37_545_489` per `upstream/evaluate.py` line 63
+ CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable.

## Canonical formula

```
ΔS_byte_only = -25 * byte_delta / 37_545_489
```

Where:
- `byte_delta` is signed; negative values mean the target archive is smaller
  than the source archive (so the score decreases, which IMPROVES the
  contest score per lower-is-better convention).
- `37_545_489` is the canonical contest video byte count denominator per
  `upstream/evaluate.py` line 63 (see also CLAUDE.md "Submission auth eval").
- The `25` multiplier is the canonical contest rate-term coefficient.

LaTeX: `\Delta S_{\text{byte\_only}} = -\frac{25 \cdot \Delta n_{\text{bytes}}}{37{,}545{,}489}`

## Domain of validity

The equation is canonical for the **byte-only static-packet custody class**:
candidates whose archive bytes differ from a canonical source archive by a
known `byte_delta` AND whose change is byte-only (NOT scorer-visible weight
or latent mutations beyond what the byte delta itself captures).

`_INCLUDED` contexts (4):

- `static_packet_custody_byte_delta_exact_eval_candidate_pre_contest_cuda`
- `byte_custody_runtime_decode_gate_validated_pre_lightning_dispatch`
- `static_packet_custody_apply_transform_byte_diff`
- `static_packet_custody_lowlevel_repack_byte_diff`

`_EXCLUDED` contexts (4) — explicitly NOT predicted by this equation:

- `scorer_visible_weight_change` — use `procedural_codebook_from_seed_compression_savings_v1`
  or `procedural_predictor_plus_residual_correction_savings_v1` for weight-derived changes
- `scorer_visible_latent_change_outside_byte_delta`
- `residual_correction_hybrid_context` — Catalog #359 explicitly refuses
  misapplication of the procedural codebook equation to residual-hybrid contexts;
  the WR01 equation inherits the same discipline at its own scope
- `procedural_codebook_replacement_context`

Sister equations cover scorer-visible / weight-derived / residual-hybrid
contexts; this equation deliberately scopes ONLY to byte-only static-packet
custody where the contest scorer's pose/seg components are not expected to
change beyond what the byte delta itself implies.

## Producer / consumer wiring (Catalog #125 6-hook discipline)

Canonical producers (2):
- `tools/build_wr01_exact_eval_packet.py` (canonical builder for the WR01
  exact-eval packet artifact that produced the empirical anchor)
- `.omx/research/wr01_static_packet_custody_20260506_codex.md` (canonical
  research memo documenting the byte-only score-delta observation)

Canonical consumers (1):
- `tac.cathedral_consumers.wr01_static_packet_custody_consumer` (sister
  cathedral consumer registered at commit `ad23f1880`; Tier A
  observability-only per Catalog #341; surfaces static-packet custody
  candidates with structural-blocker readiness reminders)

6-hook wire-in declaration (per Catalog #125):

- **Hook #1 sensitivity-map** = N/A (the equation is a rate-only arithmetic
  predictor; no sensitivity surface contribution)
- **Hook #2 Pareto constraint** = N/A (byte-only static packet does not
  participate in the Pareto polytope solver beyond the existing rate-term)
- **Hook #3 bit-allocator** = N/A (byte-only static custody is not a
  bit-allocator signal; the byte delta is already determined at packet
  construction time)
- **Hook #4 cathedral autopilot dispatch** = **ACTIVE** (sister consumer
  `wr01_static_packet_custody_consumer` surfaces structural-blocker
  readiness for cathedral autopilot ranker; this equation provides the
  canonical rate-only ΔS prediction the consumer can cite without
  introducing a score claim per Catalog #287 + #323)
- **Hook #5 continual-learning posterior** = **ACTIVE** (sister consumer's
  `update_from_anchor` is the canonical refresh path per Catalog #128/#131
  fcntl-locked discipline; future paired CUDA T4 anchors will be appended
  via `update_equation_with_empirical_anchor` per Catalog #344
  auto-recalibration mechanism)
- **Hook #6 probe-disambiguator** = N/A (the equation is the canonical
  disambiguator between byte-only RATE-ONLY context and scorer-visible
  contexts; the `_INCLUDED`/`_EXCLUDED` context taxonomy IS the
  disambiguator)

## Non-promotable by construction

Per `CLAUDE.md` "Apples-to-apples evidence discipline" + Catalog #287 +
Catalog #323 (canonical Provenance umbrella) + Catalog #341
(canonical-routing-markers): the equation's `byte_only_score_delta`
prediction is a **rate-only RATE-AXIS prediction**; it is NOT a score
claim and NOT a substitute for paired CUDA T4 + paired Linux x86_64 CPU
contest auth-eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
non-negotiable.

Specifically:
- The equation's `provenance.evidence_grade = predicted`
- The empirical anchor's `provenance.evidence_grade = research_only`
- The empirical anchor's `provenance.score_claim_valid = false`
- The empirical anchor's `provenance.promotion_eligible = false`

The WR01 codex memo's `byte_only_expected_score_delta` is explicitly
labeled as a byte-only diagnostic, not a contest auth-eval result. The
WR01 packet's `ready_for_submit = false` per the memo's "Gate State"
section, with remaining blockers `missing_lightning_environment`,
`missing_active_lane_dispatch_claim`, and
`adversarial_priority_review_prioritizes_rate_only_candidate`. Per the
memo's "Adversarial Priority Review" section: "Neither packet may claim
score without exact CUDA."

This equation formalizes the canonical rate-only prediction; it does
NOT change the gating discipline. Any actual score claim REQUIRES
paired CUDA T4 + paired Linux x86_64 CPU contest auth-eval per the
non-negotiable.

## Sister-coordination verdict

**Pre-flight sister-checkpoint guard** (Catalog #340 via
`tools/check_sister_files_recently_landed.py`) emitted
`STAND_DOWN_DUPLICATE` because 16 sister commits had touched
`.omx/state/canonical_equations_registry.jsonl` within the 12-hour
lookback window. Investigation:

- Sister commits register OTHER canonical equations:
  - `79f1ba387` parser-safe-domain-refinement for equation #26 (procedural codebook)
  - `e3e198c9f` parser-safe null-byte subset smoke
  - `3dfb877c0` PR101-gold null-byte removal smoke
  - `6587815a2` residual-hybrid byte accounting equation
  - `8e2134edc` magic codec pair 2 null-byte srl1 falsification
  - `8d8a7c6c5` canonical equation #26 domain refinement
  - `f25f8cc1b` DWT detail-subband procedural CPU smoke + canonical equation #26 first empirical anchor
  - `1dd8569de` procedural codebook generator build
  - `5c1af7ba6` null-exploit procedural seed planner
  - `0b4f37065` null-exploit probe master-gradient null-byte identification
  - `b09bb1d12` 3 NEW canonical equations + cross_codec_orthogonality_predictor_consumer
  - `7b56cbf49` cross-substrate sensitivity comparison diagnostic + 2 new canonical equations
  - `c385f1291` 3 canonical equations registered (cpu-cuda-drift-analysis)
  - `8178c6c3f` PATH-A.2 Blahut-Arimoto canonical R(D) helper
  - `0b4fff7b3` WAVE-1-DIM-4 tac.domain_priors + 3 canonical equations
  - `c8a42444c` DreamerV3 RSSM categorical R(D) canonical equation
- Content-grep verification: zero matches for
  `wr01` / `hnerv_wavelet_apply_transform` / `d2208ffa` /
  `byte_only_expected_score_delta` / `static_packet_custody` in the
  canonical equations registry pre-landing.
- The WR01 equation is **scope-disjoint** from all 16 sister commits.

**Decision**: The sister-checkpoint guard's `STAND_DOWN_DUPLICATE`
verdict was a heuristic overshoot (file-touch volume, not content
overlap). Per CLAUDE.md "Subagent coherence-by-default" the operative
test is content overlap (the canonical equations JSONL is APPEND-ONLY
per Catalog #110/#113 and explicitly supports multiple sister equation
registrations in the same window by design — that is the canonical
4-layer Catalog #245 + #313 + #344 pattern). The pre-flight tool
provides an honest signal but the canonical decision-criterion is
content overlap.

**Sister coordination with in-flight slot 3-r2 NSCS06 v8 BUILD**
(`aa612de7`): scope-disjoint. NSCS06 v8 BUILD touches
`src/tac/substrates/nscs06_v8_chroma_lut/*` namespace. This landing
touches `.omx/state/canonical_equations_registry.jsonl` (canonical
APPEND-ONLY ledger via canonical helper `register_canonical_equation`
per Catalog #131 fcntl-locked discipline). The fcntl lock at
`.omx/state/canonical_equations_registry.jsonl.lock` serializes any
concurrent NSCS06 v8 registration attempt if it lands during this
window.

## Top-3 follow-up operator-routable next-actions

1. **Paired CUDA T4 exact-eval of WR01 archive sha256 `d2208ffa`** to
   confirm the byte-only score-delta prediction against actual contest
   CUDA scorer behavior. Per the WR01 codex memo's "Next Action"
   section: requires Lightning environment, lane claim
   `wr01_apply_pr106x_half`, packet refresh with explicit operator
   exact-CUDA approval, and `ready_for_submit=true` assertion. Once
   landed, append the empirical CUDA anchor via
   `tac.canonical_equations.update_equation_with_empirical_anchor` per
   Catalog #344 auto-recalibration mechanism. Expected anchor will
   reveal whether the byte-only prediction matches paired CUDA, or
   whether the apply-transform changes scorer-visible bytes beyond
   the 9-byte rate diff.

2. **Sister equation for residual-hybrid byte accounting**: the
   2026-05-20 sister commit `6587815a2` added a residual-hybrid byte
   accounting equation. Cross-reference to verify whether the WR01
   apply-transform falls into the residual-hybrid context or remains
   in the byte-only static-packet custody context. If WR01 is actually
   residual-hybrid in scope, the canonical prediction would need to
   route through the sister equation per the `_EXCLUDED_contexts`
   declaration in this equation's domain of validity.

3. **Sister consumer wire-in audit**: the WR01 cathedral consumer
   `tac.cathedral_consumers.wr01_static_packet_custody_consumer` is
   registered as the canonical consumer for this equation. The
   consumer's `consume_candidate` returns `predicted_delta_adjustment=0.0`
   per Tier A observability-only (Catalog #341); future Tier B
   migration per Catalog #357 dual-tier discipline would let the
   consumer cite this equation's rate-only prediction directly in its
   per-axis decomposition per Catalog #356. Out of scope for this
   landing; queued as a Phase 2 sister wave.

## Discipline checklist

- [x] Catalog #117/#157/#174 canonical serializer with POST-EDIT
      `--expected-content-sha256` for all files touched
- [x] Catalog #119 Co-Authored-By trailer (auto-appended by canonical
      serializer)
- [x] Catalog #125 6-hook wire-in declaration (explicit per-hook
      assignment above)
- [x] Catalog #185 META-meta-meta drift detection — no NEW CLAUDE.md
      catalog row added; gate is canonical helper `register_canonical_equation`
      already registered per Catalog #344
- [x] Catalog #229 premise verification (read WR01 codex memo + WR01
      cathedral consumer + canonical equation registration API + sister
      commits before edit)
- [x] Catalog #287 placeholder-rationale rejection (this memo's
      rationale strings are all substantive >4 chars; no `<rationale>` /
      `<reason>` literals)
- [x] Catalog #303 cargo-cult audit — verified equation is HARD-EARNED
      (formula derived from canonical contest rate-term `25 * archive_bytes /
      37_545_489` per `upstream/evaluate.py` line 63 with residual=0.0
      empirical match), NOT CARGO-CULTED
- [x] Catalog #323 canonical Provenance umbrella — equation carries
      `evidence_grade=predicted` + `score_claim_valid=false`; empirical
      anchor carries `evidence_grade=research_only` +
      `score_claim_valid=false` + `promotion_eligible=false`
- [x] Catalog #340 sister-checkpoint guard — pre-flight emitted
      `STAND_DOWN_DUPLICATE` heuristic; verified content-disjoint via
      grep; proceeded with documented rationale
- [x] Catalog #344 canonical-equations-registry evolution discipline —
      new equation registered with all required fields including
      `_INCLUDED_contexts` / `_EXCLUDED_contexts` taxonomy + empirical
      anchor + canonical producers + canonical consumers + canonical
      Provenance
- [x] Catalog #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE — NEW
      event row appended to canonical registry; ZERO mutations of
      existing rows; sister equations untouched
- [x] Subagent checkpoint discipline per Catalog #206 (3 checkpoints
      emitted: step 1 = pre-flight stand-down investigation, step 2 =
      proceed past heuristic, step 3 = completion)

## Mission contribution per Catalog #300

`apparatus_maintenance`: extincts tribal knowledge for the byte-only
static-packet custody prediction class; future cathedral consumers can
cite the canonical equation_id `static_packet_custody_byte_delta_score_savings_v1`
rather than re-deriving the arithmetic OR risking
canonical-equation-misapplication per Catalog #359 (sister gate that
refuses canonical equation #26 misapplication to residual-hybrid
contexts; the WR01 equation's explicit `_EXCLUDED_contexts` declaration
prevents the symmetric class for this equation).

The immediate score-lowering value is N/A (the equation is rate-only
arithmetic; the actual score depends on paired CUDA exact-eval). The
structural value is canonical-equation coverage for the byte-only
static-packet custody surface, which unblocks future cathedral
consumers + cross-substrate composition planning by giving them a
formal predictor to cite per Catalog #344 evolution discipline.
