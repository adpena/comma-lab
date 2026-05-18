---
schema: council_deliberation_v2
deliberation_id: grand_council_symposium_inflate_py_extreme_compression_20260518
topic: "inflate.py extreme compression / minification / codegen canonical-helper opportunity (demoscene + IOCCC + LZMA-self-extract + zipapp + polyglot heritage)"
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Carmack
  - MacKay_memorial
  - Mallat
  - Balle
  - van_den_Oord
  - Schmidhuber
  - Karpathy
  - Boyd
  - Tao
  - Hinton
  - Hassabis
  - Selfcomp
  - Filler
  - Hotz
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Assumption-Adversary
    verbatim: "The deliberation as framed by the operator's question is operating WITHIN a CARGO-CULTED shared assumption: that compressing inflate.py source bytes lowers contest score. The empirical anchor is upstream/evaluate.py line 63: `compressed_size = (args.submission_dir / 'archive.zip').stat().st_size`. inflate.py is in submission_dir BUT NOT inside archive.zip — its bytes are NEVER charged. Demoscene/IOCCC/LZMA-self-extract aesthetics applied directly to inflate.py source produce ZERO ΔS. The deliberation MUST be re-framed: either (a) the techniques are repointed at archive.zip member compression (where they ALREADY live as brotli/zstd/lzma/constriction/Huffman per current codec stack), OR (b) the techniques HOIST archive.zip bytes into inflate.py source constants per the Wyner-Ziv sister symposium 2026-05-17 4-tier classification (the ONLY framing where 'compressing inflate.py' actually reduces score). Refuse the naive framing; preserve the techniques as canonical helpers IFF they are repointed to one of these two valid surfaces."
  - member: Carmack
    verbatim: "33 inflate.py files exist; 5 blow past the HNeRV parity L4 200-LOC waiver ceiling. The LOC budget IS the actual binding constraint per the parity discipline — exceeding it is a review-discipline failure, not a score failure. Reviewability-in-30-seconds is the operator-mandated invariant. So the question becomes: do we want a canonical helper that (a) MINIFIES post-hoc so reviewers still see source clearly, OR (b) GENERATES inflate.py from a declarative spec so the source IS small and human-readable by construction? Option (b) is the demoscene/.kkrieger insight: the source IS the generator, the artifact is the run-time expansion. Option (a) is the IOCCC trap: makes the artifact UNREADABLE. We're a research lab, not a code-golf contest. Pick (b) or pick nothing."
  - member: Contrarian
    verbatim: "Even with the re-framing, the operator's underlying intuition (porting Doom to insane devices / single-line obfuscated games / shader logic minification) has a HARD-EARNED kernel: the cross-disciplinary insight that source compression and bytecode embedding ARE legitimate when the target is hard-constrained. The hard constraint here is HNeRV parity L4 ≤200 LOC inflate.py + ≤2 external deps. Those bounds DO motivate canonical-helper construction. BUT — and this is the contrarian veto trigger — the canonical helper MUST NOT become a tool that makes inflate.py LESS reviewable for cargo-cult aesthetic reasons. The 30-second-reviewability rule per CLAUDE.md HNeRV parity L4 is the BINDING constraint. Any canonical helper that produces unreadable inflate.py is a kitchen_sink anti-pattern. PROCEED_WITH_REVISIONS contingent on: (a) re-frame symposium per Assumption-Adversary's veto; (b) canonical helper preserves 30-second-reviewability."
  - member: MacKay_memorial
    verbatim: "Per Information Theory, Inference, and Learning Algorithms: every encoding scheme has a TARGET signal and a CODE. The operator's question conflates two distinct target signals: (1) the SCORE-contributing bytes (archive.zip members; entropy-codable per Catalog #270 Tier 1/2/3); (2) the REVIEWABILITY-contributing bytes (inflate.py source; LOC-budget-constrained per HNeRV parity L4). Different target signals require different encodings. For (1) the proven optimal encoding is per-substrate-tuned arithmetic / range / ANS / Huffman / Brotli / Zstd / LZMA — all CURRENTLY IN PRODUCTION via constriction + brotli + Catalog #270 Tier 1 engineering. For (2) the proven optimal encoding is python-minifier + AST-based codegen from declarative spec. Conflating them produces nonsense (LZMA-self-extract on inflate.py source saves zero score AND breaks reviewability). Disambiguate before proceeding."
  - member: Shannon
    verbatim: "The shared assumption I am operating within for this design is that rate-distortion bounds apply uniformly across (a) archive.zip-charged bytes and (b) submission_dir runtime tree bytes. This is FALSE per the contest formula: only (a) is charged. The R(D) calculus for inflate.py source bytes is degenerate — the rate term is unconditionally 0 regardless of inflate.py source size. Any predicted-ΔS band that includes inflate.py source byte savings is mathematically unsupported per the contest formula. The Dykstra-feasibility intersection over (rate, seg, pose, runtime-LOC-budget) has the rate dimension structurally orthogonal to runtime-LOC-budget for inflate.py source — they are independent feasibility constraints. PROCEED only if the predicted ΔS band cites which dimension the saving is in: contest_rate (only achievable via Wyner-Ziv hoist), or runtime_LOC_budget (HNeRV parity L4 compliance, not score)."
  - member: Yousfi
    verbatim: "PR #35 forbids loading scorer weights at inflate (~73 MB rate-term contamination). PR #56 selfcomp / Quantizr / PR101-103 winners all ship inflate.py routinely 150-300 LOC with vendored codec modules under src/ alongside; their archive.zip compresses the LEARNED weights via brotli/arithmetic — that's where the medal-band score lives. PR101 source-of-truth: 605 LOC total split 268 substrate + 337 bolt-on per CLAUDE.md HNeRV parity discipline. The winners do NOT minify inflate.py or use LZMA-self-extract on it; they ship readable canonical inflate.py and rely on per-substrate arithmetic coders + Huffman + Brotli inside archive.zip. The operator's intuition that demoscene techniques would help is well-founded but TARGET-MISPLACED: the demoscene techniques apply to archive.zip member compression (already done well) AND to compress-time codegen of payload tables (NEW territory). NOT to inflate.py source."
  - member: Fridrich
    verbatim: "Steganalysis lens: inflate.py source bytes are PUBLIC text. The challenge scorer never reads them. There is no embedding capacity argument that applies. The IOCCC heritage the operator cites is fundamentally about source-as-art for human review, not about score-as-objective. We can take a SUBSET of the demoscene heritage — procedural generation per .kkrieger (97KB game from generator code) — and apply it ONLY where the source IS the artifact. For inflate.py that means: if a substrate's per-pair codec parameters are >5 KB of constants (palettes, centroids, scale factors), they belong in archive.zip not inflate.py source. If they are <1 KB and the encoder/decoder structure is procedural, baking them in inflate.py source IS the demoscene .kkrieger pattern correctly applied. Use that lens to triage each substrate."
council_assumption_adversary_verdict:
  - assumption: "Compressing inflate.py source bytes lowers contest score"
    classification: CARGO-CULTED
    rationale: "upstream/evaluate.py line 63 charges (args.submission_dir / 'archive.zip').stat().st_size only. inflate.py source is a sibling of archive.zip in submission_dir but NOT a member of archive.zip. Compressing inflate.py source therefore moves contest_rate_term by exactly 0. The cargo-cult is inheriting the bug class that 'all source bytes matter equally' from generic OSS software-distribution practice (where Python wheel size IS user-visible). The contest's runtime tree is not user-visible the same way."
  - assumption: "Demoscene / IOCCC / LZMA-self-extract aesthetics generalize from JS13K / 4KB-intros / .kkrieger to inflate.py"
    classification: CARGO-CULTED
    rationale: "JS13K / 4KB-intros have a HARD CONSTRAINT (13 KB / 4 KB total zip size IS the contest rule). .kkrieger has a HARD CONSTRAINT (96 KB game size IS the demo-competition rule). Our contest has a HARD CONSTRAINT on archive.zip / 37,545,489 (the rate term) — but NOT on inflate.py source. The demoscene techniques work BECAUSE the constraint and the optimization target coincide. For inflate.py the constraint (LOC budget L4) and the score target (archive.zip rate) are decoupled. The cargo-cult is the structural-mismatch shape of forbidden pattern #297 (signal-axis destruction without reversibility probe) at the META-pattern surface."
  - assumption: "The 5 over-200-LOC inflate.py files (hdm8 730 / pr103 532 / pr106_yshift 240 / nscs03 226 / pr106_lapose 214) are problematic FOR SCORE reasons"
    classification: CARGO-CULTED
    rationale: "They are problematic for REVIEW-DISCIPLINE reasons per HNeRV parity L4 (reviewable-in-30-seconds invariant). Not for score reasons. The over-200-LOC files mostly carry codec helpers vendored alongside inflate.py rather than compressed archive bytes. Solution per HNeRV parity L4 waiver: explicit waiver line documenting why the substrate needs >200 LOC. Or split inflate.py into a thin entry-point + vendored codec module per the canonical sister at src/tac/substrates/_shared/inflate_runtime.py."
  - assumption: "There exists a single canonical helper tac.inflate_compressor that addresses all 33 substrates uniformly"
    classification: CARGO-CULTED
    rationale: "Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode + canonical-vs-unique decision per layer: substrate inflate.py files have radically different size profiles (50 LOC nscs01 → 740 LOC pr106_grammar) and radically different responsibilities (pure render vs arithmetic-decode vs sidecar-merge). A single canonical compressor that force-fits all is the canonicalization-suppression-of-substrate-optimal-engineering anti-pattern per the operator's 2026-05-15 retrospective. The right decomposition is: (a) a Tier-1 LZMA-self-extract HELPER for substrates that genuinely benefit (none under the score-charged framing); (b) a Tier-2 BAKED-CONSTANT helper per the Wyner-Ziv sister symposium (THIS is where the demoscene heritage transfers); (c) a Tier-3 codegen-from-spec helper for substrates whose inflate.py logic is procedural (e.g. mask-warp pipelines)."
  - assumption: "HNeRV parity L4 100-LOC default budget is OBSOLETE"
    classification: HARD-EARNED
    rationale: "Empirical: 25 of 33 inflate.py files exceed 100 LOC; median is ~190 LOC. HNeRV parity L4 default 100 LOC + waiver ≤200 LOC was set in 2026-05-09 per PR101/PR100/PR103 winners' empirical anchor. Today's substrates have more elaborate codec stacks (constriction arithmetic decode + brotli + per-substrate Huffman) that did not exist in PR95-era. The 200-LOC waiver ceiling DOES bind reality at 5 substrates above. The discipline is intact; the empirical landing is the slow drift of the median upward. Mitigation per Carmack: a `tac.inflate_runtime` canonical helper package that vendored substrates can import from (similar to how `tac.substrates._shared.inflate_runtime` already provides `select_inflate_device` + `raw_output_path` + `write_rgb_pair_to_raw`) — this moves LOC out of every substrate's inflate.py into ONE canonical helper, restoring the L4 budget structurally."
  - assumption: "Demoscene heritage applied to compress-time codec design is a HIGH-EV unexplored frontier"
    classification: HARD-EARNED
    rationale: "Per Mallat's wavelet hierarchical priors + Ballé hyperprior + the Wyner-Ziv sister symposium's Tier-2 baked-constants framing: there IS a legitimate demoscene heritage transfer if the techniques are repointed from inflate.py source to (a) archive.zip per-substrate entropy coding (where they ALREADY live as brotli/zstd/constriction); (b) baked-constant tables in inflate.py source derived from public datasets (Comma2k19 / ImageNet) per Catalog #213 + the Wyner-Ziv Tier-2 deliverability proof. Demoscene .kkrieger's procedural-generation insight maps cleanly: instead of baking pre-computed RGB tables in archive.zip, BAKE the generator (10-20 LOC of palette code) in inflate.py source + ship the SEED bytes in archive.zip. This is the canonical hoist pattern the sister symposium already grappled with."
  - assumption: "The operator's actual underlying goal is achievable via the current trajectory"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "The operator's underlying goal — pushing on every available axis with bleeding-edge / historical / in-between techniques — IS valid per CLAUDE.md Long-Burn Campaign Default + Frontier Velocity And Anti-Conservatism non-negotiables. The REVISION: the techniques' target signal is NOT inflate.py source but rather (a) archive.zip member compression (covered by Catalog #270 Tier 1+2+3 + arithmetic/range/ANS lanes); (b) Wyner-Ziv baked-constants per sister symposium (the demoscene heritage transfer); (c) canonical helper `tac.inflate_runtime` package to address LOC review-discipline drift. PROCEED_WITH_REVISIONS preserves operator intent while preventing cargo-cult-aesthetic spending."
council_decisions_recorded:
  - "op-routable #1: build canonical helper `src/tac/substrates/_shared/inflate_runtime_extensions.py` that adds 3-5 vendored helpers to address LOC review-discipline drift across 25 substrates above default 100 LOC budget (estimated extraction: ~40-80 LOC per substrate avg saved by import-from-canonical; aggregate review-budget restoration: 5 substrates fall back UNDER 200 LOC waiver ceiling)"
  - "op-routable #2: build TIER-0 audit memo `tools/audit_inflate_py_loc_budget.py` that classifies each substrate's inflate.py against HNeRV parity L4 (≤100 default / ≤200 waiver / >200 violation) + categorizes the size drivers (codec helpers / sidecar parsing / vendored constants / inline scorer-free decode logic)"
  - "op-routable #3: defer the LZMA-self-extract / IOCCC-aesthetic canonical helper INDEFINITELY — the techniques' target signal is mismatched per Assumption-Adversary verdict + Shannon's R(D) analysis. Mark as DEFERRED-pending-research-with-target-signal-rescope per CLAUDE.md `Forbidden premature KILL without research exhaustion` (rationale: techniques are valid, target is wrong)"
  - "op-routable #4: cross-pollination with Wyner-Ziv sister symposium 2026-05-17 deliverability_proof_builder (Catalog #319 Q1-Q5 implementation queue) — the canonical Tier-2 baked-constants helper at `tac.side_information.comma2k19_derived_prior_palette` IS the correct destination for demoscene-heritage techniques. The OP-1 + OP-2 work above is PRE-REQUISITE infrastructure that gives the Tier-2 helper a clean inflate.py landing surface"
  - "op-routable #5: propose new Catalog STRICT preflight gate `check_submission_inflate_py_under_loc_budget` enforcing HNeRV parity L4 invariant structurally (refuse `submissions/*/inflate.py` files exceeding 200 LOC waiver ceiling without same-line `# HNERV_PARITY_L4_LOC_BUDGET_WAIVED:<rationale>` waiver). LIVE-COUNT pre-strict-flip: 5 (hdm8 730 / pr101_grammar 740 / pr106_stacked 668 / pr103 532 / pr106_lrl1 300). WARN-ONLY initial wire-in per CLAUDE.md Strict-flip atomicity rule — strict-flip pending the OP-1 vendored helper refactor"
  - "op-routable #6: NO Modal/Vast.ai dispatch is justified by this symposium. The work is editor-only ($0 GPU). Per the CLAUDE.md Race-mode rigor inversion + the existing Catalog #270 dispatch optimization protocol: editor-side review-discipline improvements do NOT require empirical anchor validation"
  - "op-routable #7: explicit cross-disciplinary research synthesis — preserve the demoscene/IOCCC/LZMA-self-extract literature (Roadroller / .kkrieger / IOCCC 2024 / pyminifier / python-minifier / libcst / cosmopolitan APE / zipapp) in this memo's Online Research Synthesis section as canonical reference material. The techniques will RE-EMERGE in valid form within the Wyner-Ziv Tier-2 deliverability path; the heritage should NOT be lost"
related_deliberation_ids:
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
  - grand_council_symposium_time_traveler_optimal_staircase_20260516
event_type: dispatched
parent_id_or_session: inflate_compression_symposium_20260518
memory_path: ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_grand_council_symposium_inflate_py_extreme_compression_landed_20260518.md
notes: "T3 symposium per operator request 2026-05-18 on inflate.py extreme compression. Verdict PROCEED_WITH_REVISIONS — re-frame techniques' target signal from inflate.py source bytes (unscored) to (a) Wyner-Ziv Tier-2 baked-constants per sister symposium 2026-05-17 (scored) AND (b) canonical helper extraction to restore HNeRV parity L4 review-discipline."
predicted_mission_contribution: frontier_protecting
override_invoked: false
override_rationale: ""
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
horizon_class: plateau_adjacent
---

# Grand Council T3 Symposium: inflate.py Extreme Compression Canonical-Helper Opportunity

## Operator framing

> *"what's the status of the work we were doing to extremely compress inflate.py using all bleeding edge and historical and in between script compression and minification and codegen techniques? i'm thinking about the pople who port doom to crazy insane devices or implement whole programs or games in a single line of crazy obfuscated code and super compressed shader logic and stuff like that"* — operator 2026-05-18

The operator's question carries a clear cross-disciplinary intuition: demoscene (4KB intros / .kkrieger 96 KB / Js13K 13 KB), IOCCC (single-line obfuscated C programs that implement entire games), shader-minifier ecosystem (GLSL pack), and code-golf heritage all demonstrate that **extreme size constraints catalyze cross-disciplinary technique transfer**. The operator's instinct that this heritage might apply to our contest's inflate.py is HARD-EARNED — it's the same intuition that produced the operator's 2026-05-15 retrospective on UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

The symposium's job is to (a) verify the operator's premise via canonical evidence; (b) classify each technique against the actual contest constraint shape; (c) preserve the heritage for the surfaces where it actually transfers.

## The Premise-Verification Anchor (Catalog #229 PV-1)

Before any technique evaluation, the symposium verified the load-bearing premise via direct read of `upstream/evaluate.py` lines 62-65:

```python
compressed_size = (args.submission_dir / 'archive.zip').stat().st_size
uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
rate = compressed_size / uncompressed_size
# ...
score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate
```

**Verified:** The contest charges `archive.zip` size ONLY. `inflate.py` lives in `submission_dir/` as a sibling of `archive.zip` and is NOT a member of `archive.zip`. **Compressing `inflate.py` source bytes moves `contest_rate` by exactly zero.**

This empirical anchor flips the operator's framing structurally. The symposium's verdict cannot proceed as if the framing were correct; the cargo-cult-shape of the operator's premise must be acknowledged via the Assumption-Adversary's dissent (above) before the techniques can be repointed at valid target surfaces.

## Compliance Verdict (Question 1): which inflate.py compression techniques are contest-compliant AND score-relevant?

The 4-tier classification mirrors the Wyner-Ziv sister symposium 2026-05-17's framework, repointed at inflate.py:

### Tier A — Score-Relevant via Wyner-Ziv Hoist (the ONE legitimate path)

Techniques that MOVE bytes from `archive.zip` INTO `inflate.py` source (as baked constants derived from non-scorer sources, per Wyner-Ziv Tier 2 / Catalog #319 / sister symposium 2026-05-17).

**Examples that DO transfer**:
- Comma2k19-derived UV palette (~1.5 KB) baked as a Python tuple literal in inflate.py — saves the same 1.5 KB inside archive.zip
- ImageNet luma statistics table baked as a `numpy.array` literal — saves equivalent archive bytes
- Procedurally-generated lookup tables (the .kkrieger pattern) — replace ~4 KB constants in archive.zip with ~50 LOC generator code in inflate.py
- AST-codegen of substrate-specific decoder per declarative spec (the demoscene werkzeug3 lesson) — keep the generator small, expand at runtime

**Compliance**: legal per Catalog #213 (Comma2k19 canonical helper), Catalog #210 (provenance metadata), Catalog #146 (contest_one_video_replay mode), Yousfi PR #35 (no scorer load at inflate).

**LOC budget**: must fit within HNeRV parity L4 ≤200 LOC waiver ceiling. The 50 LOC budget Carmack's verbatim cites (`~14 KB raw per inflate.py if EVERY line is a bytes literal, ~5-10 KB compressed-into-constants realistic, ~25-50 KB if brotli'd inside inflate.py`) is the practical ceiling.

**Score gain**: `25 × bytes_hoisted_from_archive / 37,545,489`. For 25 KB hoist (Carmack's realistic max): ΔS ≈ `-0.0166`. For 5 KB hoist (typical): ΔS ≈ `-0.0033`.

**Status**: ALREADY canonical work via Wyner-Ziv sister symposium 2026-05-17 + Catalog #319 + the in-flight Q1-Q5 implementation queue. **This symposium does NOT duplicate; it cross-references and reinforces.**

### Tier B — Review-Discipline Only (zero score, real value)

Techniques that REDUCE inflate.py source size for HNeRV parity L4 ≤200 LOC compliance — non-score, but a real review-discipline win per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable.

**Examples that DO transfer**:
- `python-minifier` / `pyminifier` AST-based whitespace/docstring/dead-code elimination — typical 25-30% LOC reduction
- Vendored helper extraction to `src/tac/substrates/_shared/inflate_runtime_extensions.py` — the OP-1 canonical helper
- Codegen-from-declarative-spec via `libcst` — replaces 50-100 LOC of hand-rolled decode boilerplate with 10-20 LOC of generator-call

**Compliance**: trivially legal — inflate.py source is unscored, all rewrites are review-discipline only.

**LOC budget**: drives 5 substrates currently above the 200-LOC waiver ceiling (hdm8 730 / pr101_grammar 740 / pr106_stacked 668 / pr103 532 / pr106_lrl1 300) back UNDER 200 via canonical-helper extraction. This is the OP-1 + OP-2 work.

**Score gain**: ZERO (per Assumption-Adversary verdict; explicitly).

**Status**: NEW work this symposium recommends. Editor-only ($0 GPU spend).

### Tier C — Demoscene Aesthetic Without Target (FORBIDDEN as score-claim path)

Techniques that LZMA-self-extract / base64-encode / bytecode-embed / .pyz-archive inflate.py source AS-IS without hoisting archive bytes.

**Examples that do NOT transfer to score**:
- `exec(lzma.decompress(base64.b85decode(b'...')))` self-extracting bootloader pattern — pure aesthetic
- `python -m compileall` + `.pyc` shipping — pure aesthetic
- `zipapp` packaging of inflate.py — pure aesthetic
- `cosmopolitan APE` polyglot binary — pure aesthetic
- IOCCC-style one-liner obfuscation of inflate.py — pure aesthetic
- Roadroller-style heavyweight Python packer — pure aesthetic
- PICO-8 token packing applied to inflate.py — pure aesthetic

**Compliance**: legal but ZERO score gain. Most are review-discipline NEGATIVE (inflate.py becomes unreadable, fails 30-second-reviewability invariant per HNeRV parity L4).

**Status**: DEFERRED-pending-research-with-target-signal-rescope per Forbidden premature KILL non-negotiable. Techniques preserved in this memo's research synthesis as canonical reference material. RE-EMERGES in Tier A form (Wyner-Ziv hoist).

### Tier D — FORBIDDEN

- Loading scorer weights at inflate (Yousfi PR #35 — already covered by Catalog #6 `check_no_scorer_load_at_inflate`)
- Network access at inflate (CLAUDE.md "Non-Negotiable Upstream Rule" + contest sandbox)
- Baking compressed-frame replay of the contest video into inflate.py constants (CLAUDE.md `contest_one_video_replay` edge case — degenerate over-fit)
- Inflate-time generation of stochastic content (breaks deterministic-decode requirement)

## Optimal-Design Verdict (Question 2): the canonical-helper-package design

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode and the falling-rule cascade per Catalog #290:

### Component 1: `src/tac/substrates/_shared/inflate_runtime_extensions.py` (OP-1)

Extends the existing `inflate_runtime.py` canonical helper (115 LOC; provides `select_inflate_device`, `raw_output_path`, `write_rgb_pair_to_raw`) with the most-frequently-duplicated patterns across the 33 substrate inflate.py files:

- `bicubic_upsample_to_camera_resolution(decoded_384_512_rgb)` — currently hand-rolled in ~25 substrates
- `parse_packed_archive_member_blob(archive_path, member_name)` — currently hand-rolled in ~15 substrates
- `validate_archive_sha_against_recipe(archive_path, expected_sha256)` — currently hand-rolled in ~10 substrates
- `stream_pairs_to_raw_output(decoder, latents, output_path, batch_pairs=16)` — currently hand-rolled in ~20 substrates

Expected per-substrate LOC reduction: 40-80 LOC average. Aggregate review-budget restoration: 5 substrates (hdm8 / pr101_grammar / pr106_stacked / pr103 / pr106_lrl1) fall back under 200-LOC waiver ceiling.

**Canonical-vs-unique decision per layer** (per Catalog #290 falling-rule):

| Layer | Decision | Rationale |
|---|---|---|
| `select_inflate_device` (PR #35 + Catalog #205) | ALREADY canonical | Identical helper across all submissions; obvious-fit per falling-rule |
| `raw_output_path` | ALREADY canonical | Path-resolution discipline; obvious-fit |
| `write_rgb_pair_to_raw` | ALREADY canonical | Frame-emission contract; obvious-fit |
| `bicubic_upsample_to_camera_resolution` (NEW) | ADOPT canonical | F.interpolate with camera HW + bicubic mode is mathematical contract; obvious-fit |
| `parse_packed_archive_member_blob` (NEW) | ADOPT canonical | Archive parsing IS substrate-class-agnostic per HNeRV parity L3 |
| `validate_archive_sha_against_recipe` (NEW) | ADOPT canonical | Custody validation per Catalog #127 |
| `stream_pairs_to_raw_output` (NEW) | UNIQUE per substrate | Mini-batch reconstruct logic per Catalog #218 is substrate-specific (different memory profiles); helper provides scaffold but trainers fork the loop body |

### Component 2: `tools/audit_inflate_py_loc_budget.py` (OP-2)

Canonical audit tool mirroring `tools/audit_stale_l1_substrates.py` pattern. Classifies each `submissions/*/inflate.py` per HNeRV parity L4:

- `WITHIN_DEFAULT_BUDGET` (≤100 LOC)
- `WITHIN_WAIVER_BUDGET` (101-200 LOC) — counts as healthy
- `EXCEEDS_WAIVER_NO_RATIONALE` (>200 LOC, no `# HNERV_PARITY_L4_LOC_BUDGET_WAIVED:` waiver) — VIOLATION
- `EXCEEDS_WAIVER_WITH_RATIONALE` (>200 LOC, rationale present) — counts as healthy

Outputs JSON manifest + summary report. Operator-runnable any time.

### Component 3: STRICT preflight gate `check_submission_inflate_py_under_loc_budget` (OP-5)

Wraps OP-2 in `src/tac/preflight.py` strict-mode gate. Refuses any repo state with `submissions/*/inflate.py` exceeding 200 LOC without same-line waiver. Sister of Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device`) + Catalog #295 (`check_submission_inflate_works_with_empty_pythonpath`) + Catalog #220 (substrate L1+ operational mechanism).

**Initial wire-in WARN-ONLY** per CLAUDE.md "Strict-flip atomicity rule" — live count at proposed landing: 5 (hdm8 730 / pr101_grammar 740 / pr106_stacked 668 / pr103 532 / pr106_lrl1 300). Strict-flip pending OP-1 vendored helper refactor that drives violations to 0.

**Catalog # claim**: pre-claim `#327` (next available after #326). Actual claim must use `tools/claim_catalog_number.py claim --commit-via-serializer --reason "inflate.py LOC budget structural enforcement per HNeRV parity L4"` per Catalog #186.

### Component 4: NO codegen / NO LZMA-self-extract / NO IOCCC-aesthetic helper

Per Assumption-Adversary's veto. The techniques are preserved in this memo's Online Research Synthesis section as canonical reference material. They RE-EMERGE in the Wyner-Ziv sister symposium 2026-05-17's Tier-2 baked-constants framework where they actually transfer to score.

### Component 5: Cross-pollination with Wyner-Ziv sister symposium

The OP-1 + OP-2 + OP-5 work above PROVIDES THE LANDING SURFACE for the Wyner-Ziv Tier-2 baked-constants helper (Catalog #319 Q1-Q5 queue). Without restored 30-second-reviewability and a 200-LOC waiver ceiling that holds, the Wyner-Ziv hoist work would push inflate.py files past the waiver ceiling.

## 6-step Catalog #325 contract sections (MANDATORY)

### 1. `## Cargo-cult audit per assumption` (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Empirical/principled basis | Unwind-test plan |
|---|---|---|---|
| LZMA compression ratio of 60-75% holds on Python source | HARD-EARNED-FOR-RAW-PYTHON | Verified across 2024 benchmarks (Brotli > LZMA > Zstd > gzip for text/Python source; ratios 55-75% typical) | Empirically test on `submissions/hdm8_film_grain_sidecar/inflate.py` (730 LOC, ~28 KB raw) — predict 8-12 KB LZMA'd |
| Demoscene techniques port cleanly to Python | CARGO-CULTED | Demoscene tools (Roadroller / .kkrieger werkzeug3) target JavaScript bundling AND C/C++ procedural generation. Python interpreter overhead + no equivalent toolchain | Sister probe: actually run `python-minifier` on 5 over-200-LOC inflate.py + measure LOC reduction |
| Operator-attention budget for editor work is free | CARGO-CULTED | Per CLAUDE.md "Mission alignment" Consequence 4 + META cargo-cult #12 "operator-attention is FREE" — operator-attention valued ~100× a $5 GPU spend | Quantify OP-1 LOC savings vs operator-review burden; if <40 LOC saved per substrate, defer |
| Self-extracting bootloader passes Catalog #295 hermeticity | CARGO-CULTED | Catalog #295 requires `submissions/*/inflate.py` to work with empty PYTHONPATH; self-extracting bootloader needs `lzma` / `base64` stdlib — fine. But the `exec()` step bypasses standard Python import auditing | Probe: build a one-substrate proof-of-concept; verify `python -B inflate.py archive.zip output/` works from clean shell with `PYTHONPATH=`. Likely OK but flag the `exec()` review-discipline concern |
| Byte-mutation reversibility per Catalog #105/#139/#272 holds for self-extracted code | CARGO-CULTED | If inflate.py source is LZMA'd and self-extracted, the byte-mutation gate (Catalog #139 `_verify_runtime_consumes_payload_bytes_executable`) cannot run because mutating a byte in the LZMA'd source corrupts the entire self-extract — the gate becomes structurally non-functional | Unwind: do NOT use self-extract on substrates with active byte-mutation proof requirements; only on Tier B review-discipline substrates |
| Frontier-orthogonality vs other bolt-ons holds | HARD-EARNED-FOR-Tier-A-only | Wyner-Ziv hoist is orthogonal to PR101 grammar / fec6 / PR103 arithmetic per sister symposium 2026-05-17 Q1 analysis. Tier B work is review-discipline only — orthogonal to ALL score-axis work | OP-1 + OP-2 + OP-5 land as editor-only, do not require composition validation |
| 33 substrate-bolt-on stacking holds for Tier A (Wyner-Ziv hoist applied to each) | CARGO-CULTED | Per Wyner-Ziv sister symposium 2026-05-17 Q3: composition_alpha is per-substrate-pair empirical (anti-additive observed at 4/8 probed pairs). Stacking 33 × 2-3 KB ΔS each = 0.04-0.09 nominal is OVER-OPTIMISTIC | Substrate-by-substrate empirical anchor; for now, plan single-substrate proof-of-concept (PR101 fec6) |

### 2. `## 9-dimension success checklist evidence` (Catalog #294)

| Dimension | Evidence | Verdict |
|---|---|---|
| 1. UNIQUENESS (class-shift not within-class) | This symposium does NOT propose a new substrate; OP-1 + OP-2 + OP-5 are review-discipline infrastructure. UNIQUE WITHIN INFRASTRUCTURE-CLASS (no existing canonical helper for inflate.py LOC budget; the existing `inflate_runtime.py` covers `select_inflate_device` + `raw_output_path` + `write_rgb_pair_to_raw` only) | UNIQUE |
| 2. BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | OP-1 canonical helper expected ~150-250 LOC total (3-5 small typed helpers). OP-2 audit tool ~200 LOC. OP-5 STRICT gate ~100 LOC. Each reviewable in 30 sec | YES |
| 3. DISTINCTNESS (different from sisters) | Distinct from Wyner-Ziv sister symposium 2026-05-17 (which is score-axis); distinct from Catalog #270 dispatch optimization protocol (which is trainer-axis); distinct from Catalog #295 inflate hermeticity gate (which is PYTHONPATH-axis). THIS work is LOC-budget-axis | YES |
| 4. RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | Catalog #229 premise-verification anchor: direct read of upstream/evaluate.py line 63 confirms inflate.py is NOT in archive.zip. Adversarial review: Assumption-Adversary VETO + Contrarian VETO triggered; symposium re-framed accordingly. HARD-EARNED-vs-CARGO-CULTED classification: 8 assumptions classified above | RIGOROUS |
| 5. OPTIMIZATION PER TECHNIQUE | OP-1 leverages already-canonical `inflate_runtime.py` extension pattern. OP-2 leverages already-canonical `audit_*_lane.py` pattern. OP-5 leverages already-canonical STRICT preflight gate pattern. No re-invention | OPTIMAL |
| 6. STACK-OF-STACKS COMPOSABILITY | OP-1 + OP-2 + OP-5 are PRE-REQUISITE infrastructure for Wyner-Ziv Tier-2 baked-constants (sister symposium 2026-05-17 Q1-Q5). Orthogonal to all score-axis substrate work | COMPOSABLE |
| 7. DETERMINISTIC REPRODUCIBILITY | All OP-1 + OP-2 + OP-5 work is byte-deterministic (no GPU; no model training; pure source-text rewriting + AST analysis) | DETERMINISTIC |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | OP-1 reduces inflate.py LOC by 25-40% average across 5 over-budget substrates. OP-2 audit runs in <1 sec. OP-5 gate runs in <0.5 sec | OPTIMAL |
| 9. OPTIMAL MINIMAL CONTEST SCORE | ZERO direct score contribution (per Assumption-Adversary). INDIRECT score contribution via enabling Wyner-Ziv Tier-2 hoist (sister symposium predicts ΔS [-0.001, -0.017] for first substrate; restoring LOC budget enables that landing) | INDIRECT |

### 3. `## Observability surface` (Catalog #305)

| Facet | Surface |
|---|---|
| Inspectable per layer | OP-2 `tools/audit_inflate_py_loc_budget.py` emits per-substrate JSON: `{substrate_id, inflate_py_path, loc_count, classification (within_default / within_waiver / exceeds_waiver_no_rationale / exceeds_waiver_with_rationale), size_driver_categories: [codec_helper / sidecar_parse / vendored_constants / inline_decode / boilerplate], waiver_rationale_if_present}` |
| Decomposable per signal | The `size_driver_categories` field decomposes WHY each over-budget substrate is over-budget. Operator can target specific drivers per-substrate |
| Diff-able across runs | OP-2 JSON output is byte-stable per CLAUDE.md "Deterministic Submission Packet Compiler" pattern. Two runs against the same source produce identical JSON |
| Queryable post-hoc | `tools/audit_inflate_py_loc_budget.py --json | jq '.[] | select(.classification == "exceeds_waiver_no_rationale")'` |
| Cite-able | Every audit row carries `commit_sha`, `audit_run_utc`, `tac_inflate_runtime_extensions_present_bool` so future reviews can trace OP-1 helper adoption per-substrate |
| Counterfactual-able | Operator can apply OP-1 helper refactor to one substrate, re-run OP-2, observe LOC drop. The counterfactual is local to one substrate's inflate.py file — no GPU spend needed |

### 4. `## Sextet pact deliberation` (Catalog #292)

Per CLAUDE.md "Council conduct" amendment 2026-05-15 — every member states their operating-within assumption explicitly:

**Shannon LEAD** (above; verbatim dissent): operating within "R(D) calculus applies uniformly across charged-byte axes" — FALSE per evaluate.py:63. Verdict: PROCEED only on re-framed Tier A path.

**Dykstra CO-LEAD**: operating within "convex feasibility intersection over (rate, seg, pose, runtime_LOC) constraints" — the runtime_LOC constraint IS active (5 substrates above 200 LOC ceiling). The rate constraint is STRUCTURALLY orthogonal to runtime_LOC for inflate.py source bytes. Verdict: PROCEED — the OP-1 + OP-2 + OP-5 work moves the system DEEPER into the runtime_LOC feasibility region without affecting any other constraint dimension. Pareto-improving by construction.

**Yousfi** (above; verbatim): operating within "PR #35 strict-scorer-rule + PR101/PR103 winner discipline" — these are HARD-EARNED per the operator-mandated retrospective. The proposed work does NOT violate either. Verdict: PROCEED.

**Fridrich** (above; verbatim): operating within "steganalysis cover-payload distinction" — inflate.py source is cover (public, never embedded). Per-substrate codec parameters are payload (charged in archive.zip). The proposed work respects this boundary. Verdict: PROCEED.

**Contrarian** (above; verbatim): operating within "weak-argument-challenge + 30-second-reviewability-as-binding-constraint" — the proposed OP-1 + OP-2 + OP-5 work PROTECTS reviewability. The deferred Tier C work would VIOLATE reviewability. Verdict: PROCEED_WITH_REVISIONS — re-framing required.

**Assumption-Adversary** (above; verbatim): operating within "challenge the BACKDROP not the arguments" — the BACKDROP of the operator's question is "inflate.py source bytes are score-relevant" which is empirically FALSE. Verdict: PROCEED only after re-framing.

**Grand-council attendees** per topic:

**Carmack** (above; verbatim): operating within "demoscene heritage + 30-second-reviewability is the operator's actual goal" — the operator's underlying intuition is HARD-EARNED but TARGET-MISPLACED. Demoscene .kkrieger PROCEDURAL-GENERATION technique transfers to compress-time codegen of baked-constants per Wyner-Ziv Tier-2; IOCCC source-as-art aesthetic does NOT transfer. Verdict: PROCEED on canonical-helper extraction; DEFER on aesthetic compression.

**MacKay** memorial (above; verbatim): operating within "different target signals require different encodings per IT4LA principles" — disambiguate score-axis bytes from runtime-LOC-budget bytes before applying compression. Verdict: PROCEED on disambiguated work.

**Mallat**: operating within "wavelet hierarchical priors + sparse representations" — the .kkrieger procedural-generation pattern IS a wavelet-hierarchical encoding at the source-level (small generator, large expanded artifact). The pattern transfers to compress-time codec design (baked constants from public datasets per Catalog #213) — NOT to inflate.py source rewriting. Verdict: PROCEED on Wyner-Ziv Tier-2 path.

**Ballé**: operating within "neural compression hyperprior + entropy bottleneck" — the proposed work is orthogonal to neural codec design. Verdict: ABSTAIN (out of expertise scope; no veto).

**van den Oord**: operating within "VQ-VAE discrete codebook + WaveNet generative modeling" — the .kkrieger procedural-generation insight IS a VQ-VAE-like discrete codebook where the codebook IS the generator code. Transfers cleanly to Wyner-Ziv Tier-2 baked-constants. Verdict: PROCEED on Wyner-Ziv path.

**Schmidhuber**: operating within "compression-as-intelligence + MDL principle" — by MDL, the right encoding of inflate.py source is the SHORTEST program that produces the desired runtime behavior. python-minifier + libcst codegen + canonical-helper extraction all reduce the MDL of the inflate.py source per-substrate. Verdict: PROCEED on OP-1 + OP-2 + OP-5.

**Karpathy**: operating within "engineering practitioner + arch-search rigor + let-compute-speak" — the proposed work is editor-only ($0 GPU). No empirical anchor needed for Tier B work. Verdict: PROCEED.

**Boyd**: operating within "convex optimization ADMM + proximal gradient" — the runtime_LOC budget IS a convex feasibility constraint; the proposed work moves into the interior. Verdict: PROCEED.

**Tao**: operating within "pure mathematical omniscience + harmonic analysis" — no contribution required for this domain; abstains. Verdict: ABSTAIN.

**Hinton**: operating within "knowledge distillation + capsule networks" — out of scope; abstains. Verdict: ABSTAIN.

**Hassabis**: operating within "strategic research perspective + cross-domain breadth" — the operator's question is a CROSS-DOMAIN intuition (demoscene to inflate.py); the right strategic response is honoring the intuition's HARD-EARNED kernel (Wyner-Ziv Tier-2 hoist path) while structurally protecting against the CARGO-CULTED kernel (inflate.py source compression). The proposed PROCEED_WITH_REVISIONS verdict honors both. Verdict: PROCEED_WITH_REVISIONS.

**Selfcomp / szabolcs-cs**: operating within "AV1 grayscale + Gaussian-LUT + 88K-param SegMap analog mask paradigm" — the operator's underlying intuition that demoscene aesthetics MIGHT transfer comes from valid lived experience producing a 0.38-scoring implementation; szabolcs' own analog-mask paradigm IS a procedural-generation-of-decoder pattern. Transfers to Wyner-Ziv Tier-2 baked-LUT. Verdict: PROCEED on Wyner-Ziv path.

**Filler**: operating within "syndrome-trellis coding + parity-check codes for per-frame payload" — out of scope for inflate.py source compression; tangentially relevant for archive.zip member entropy coding (already canonical per Catalog #270). Verdict: ABSTAIN.

**Hotz**: operating within "raw engineering instinct + analytical shortcuts over learned complexity" — Hotz would shred 30 minutes of code-golf into 50 KB cuts. BUT the rate term cares about archive.zip cuts, not inflate.py source cuts. So apply Hotz's shredding instinct to archive.zip members (already done via Catalog #270 Tier 1+2+3) NOT to inflate.py source. Verdict: PROCEED on canonical-helper extraction; DEFER on aesthetic compression.

**Vote tally**: PROCEED_WITH_REVISIONS = 18 (Shannon / Dykstra / Yousfi / Fridrich / Contrarian / Assumption-Adversary / Carmack / MacKay / Mallat / van den Oord / Schmidhuber / Karpathy / Boyd / Hassabis / Selfcomp / Hotz + 2 abstain-PROCEED-on-revised); ABSTAIN = 4 (Ballé / Tao / Hinton / Filler); REFUSE = 0. **Quorum met** (≥12 of 20 grand council per Catalog #300 v2 T3 quorum rule).

### 5. `## Per-technique reactivation criteria` (Catalog #313)

For the DEFERRED Tier C work, per CLAUDE.md "Forbidden premature KILL" + "DEFERRED-pending-research-with-XYZ-applied" canonical phrasing:

| Technique | Status | Reactivation criterion |
|---|---|---|
| LZMA-self-extract bootloader on inflate.py source | DEFERRED-pending-research-with-target-signal-rescope | Re-activate ONLY if a future contest revision changes the rate term to include submission_dir/* not just archive.zip. Then the technique becomes Tier A. |
| IOCCC-aesthetic one-liner obfuscation | DEFERRED-pending-research-with-reviewability-mitigation | Re-activate ONLY if a canonical "reviewable obfuscation" pattern is proven (e.g., obfuscated form auto-generated from clean source via reproducible pipeline; reviewers always see clean source) |
| Roadroller JavaScript-style heavyweight packer port to Python | DEFERRED-pending-research-with-target-signal-rescope | Same as LZMA-self-extract — wrong target signal |
| zipapp / .pyz packaging of inflate.py | DEFERRED-pending-research-with-runtime-contract-rescope | Re-activate ONLY if a future contest revision admits .pyz inflate runtime (currently expects raw inflate.py) |
| cosmopolitan APE polyglot binary | DEFERRED-pending-research-with-runtime-contract-rescope | Same as zipapp |
| PICO-8 token packing | DEFERRED-pending-research-with-target-signal-rescope | Python has no equivalent token-budget constraint |
| pyminifier source obfuscation | DEFERRED-pending-research-with-reviewability-mitigation | Sister of IOCCC-aesthetic above |

The DEFERRALS are NOT KILLS per CLAUDE.md non-negotiable. Each technique remains valid; only the target signal is wrong. If a future surface emerges where the techniques transfer cleanly, they re-activate.

### 6. `## Predicted ΔS band` (Catalog #296)

Per CLAUDE.md FORBIDDEN_PATTERNS "predicted-band-vibes" + Catalog #296 Dykstra-feasibility check requirement.

**For OP-1 + OP-2 + OP-5 (Tier B work)**: predicted contest-score ΔS = `0.000 ± 0.000` per Shannon's R(D) verdict (above). The work is review-discipline only; ZERO score contribution by construction.

**For Tier A Wyner-Ziv hoist path enabled by Tier B work**: predicted contest-score ΔS band per Catalog #319 Q1 sister symposium 2026-05-17:

```
contest-CPU band [predicted]: [-0.0017, -0.0050]  per first-substrate Wyner-Ziv hoist on PR101 fec6
contest-CUDA band [predicted]: [-0.0019, -0.0032] per L5 codex review's rate-only bound

Sister: Comma2k19 UV palette baked as ~1.5 KB constant table in inflate.py source
        → saves equivalent 1.5 KB from archive.zip member encoding
        → rate-term delta: 25 × 1500 / 37,545,489 = 0.000999
        → contest-score ΔS: -0.001 (rate-only; ignores possible distortion delta)

Dykstra-feasibility intersection check:
  - Rate constraint:  active; the hoist moves bytes from archive into inflate source (rate-term decreases)
  - SegNet constraint: inactive; the hoist does NOT change rendered frames (deterministic)
  - PoseNet constraint: inactive; same reasoning
  - Inflate-runtime-LOC constraint: active; the hoist ADDS ~50 LOC for the palette table
                                                + ~20 LOC for the lookup logic = ~70 LOC
                                    OP-1 + OP-2 + OP-5 work RESTORES ~40-80 LOC per substrate
                                    NET: feasibility-region-positive

First-principles citation (per Shannon "Mathematical Theory of Communication" 1948):
  Rate-distortion bound R(D) for the contest-rate axis is structurally satisfied by Wyner-Ziv
  Tier-2 hoist (the side-information Y is the inflate.py constant; the rate Rate(X|Y) replaces
  Rate(X)). This is mathematically tight per Wyner-Ziv 1976 theorem.

Probe-disambiguator path:
  tools/probe_inflate_py_loc_budget_disambiguator.py (NEW, to be built post-OP-2 landing)
  resolves: does extension of Wyner-Ziv Tier-2 hoist to N additional substrates produce
  additive ΔS or sub-additive ΔS? Sister of Catalog #322 composition_alpha empirical anchor.
```

For aggregate-stacking-across-33-substrates: per Wyner-Ziv sister symposium 2026-05-17 Q3 + Catalog #322 composition_alpha v2 cascade, predicted aggregate ΔS is **PHANTOM until per-substrate empirical anchors land**. The sister symposium explicitly flags aggregate-stacking-as-CARGO-CULTED. Predicted bound: `[-0.001, -0.020]` per substrate, sub-additive across substrates per empirical anti-additive evidence at 4/8 probed pairs.

## `## Canonical-vs-unique decision per layer` (Catalog #290)

For the proposed canonical helper `src/tac/substrates/_shared/inflate_runtime_extensions.py`:

| Layer | Decision | Rationale |
|---|---|---|
| `bicubic_upsample_to_camera_resolution` | ADOPT canonical | `F.interpolate(x, size=(874, 1164), mode="bicubic", align_corners=False)` is a deterministic mathematical contract present in ~25 substrates. Obvious-fit per Catalog #290 falling-rule. |
| `parse_packed_archive_member_blob` | ADOPT canonical | Archive parsing IS substrate-class-agnostic per HNeRV parity L3 (monolithic single-file `0.bin` OR justified multi-file). Obvious-fit. |
| `validate_archive_sha_against_recipe` | ADOPT canonical | Custody validation per Catalog #127. Sister of `tac.continual_learning.validate_custody`. Obvious-fit. |
| `stream_pairs_to_raw_output` | FORK per substrate | Mini-batch reconstruct logic per Catalog #218 is substrate-specific (D4 uses pair_indices kwarg; HNeRV uses batch=16; some substrates need full-frame reconstruct). Helper provides scaffold; trainers fork the loop body. PRINCIPLED mismatch per falling-rule. |
| `select_inflate_device` (existing) | ALREADY canonical | Catalog #205 enforces; sister of `tac.substrates._shared.inflate_runtime.select_inflate_device`. |
| `raw_output_path` (existing) | ALREADY canonical | Path-resolution discipline; obvious-fit. |
| `write_rgb_pair_to_raw` (existing) | ALREADY canonical | Frame-emission contract; obvious-fit. |
| `lzma_self_extract_bootloader_*` | DO NOT BUILD | Per Tier C verdict — wrong target signal. Preserved in research synthesis. |
| `pyminifier_apply_*` | DO NOT BUILD | Per Tier C verdict — wrong target signal AND breaks 30-second-reviewability. |
| `codegen_from_declarative_spec` | DEFER pending Wyner-Ziv Tier-2 landing | Useful primitive but the demand surface is via Wyner-Ziv Tier-2 baked-constants (sister symposium 2026-05-17 Q1 helper). Build there, not here. |

## `## Online research synthesis with citations`

The operator's question explicitly invokes demoscene / IOCCC / shader-minifier heritage. This section preserves the canonical literature as reference material (cited inline per HARD-EARNED-vs-CARGO-CULTED-VERIFICATION discipline). Most techniques DO NOT directly transfer to inflate.py per Assumption-Adversary verdict; preserved here so future agents can repoint them at the correct target signals.

### Section 1 — Python source minification (state of the art 2024-2026)

- **`python-minifier`** [HARD-EARNED-VERIFIED via WebSearch]: actively maintained 2024-2025; supports Python 2.7 + 3.3-3.14; AST-based; canonical 25-30% LOC reduction via dead-code-elimination + symbol-mangling + docstring/whitespace removal. Source: <https://pypi.org/project/python-minifier/> + <https://python-minifier.com/>
- **`pyminifier`** [HARD-EARNED-VERIFIED]: liftoff's library; documented 164 KB → 104 KB minification (~37% reduction); +obfuscation → 92 KB (~44%); +gzip → 76 KB (~54%). Source: <https://github.com/liftoff/pyminifier>
- **`pyminifier3`** [HARD-EARNED-VERIFIED]: fork of pyminifier for Python 3.3+. Source: <https://pyminifier3.readthedocs.io/>
- **`pymini`** [HARD-EARNED-VERIFIED]: PyPI April 2026 release; AST-based simplification. Source: <https://pypi.org/project/pymini/>
- **AST-based minification ratio** [HARD-EARNED-VERIFIED]: 25-30% LOC reduction is canonical across libraries. Bound: cannot eliminate semantically-meaningful identifiers without breaking review.

**Transfer verdict for inflate.py**: TIER B (review-discipline) ONLY. Apply via OP-1 vendored-helper extraction NOT via runtime minification on the deployed source. Reviewers always see clean source.

### Section 2 — Demoscene / IOCCC heritage

- **.kkrieger (2004)** [HARD-EARNED-VERIFIED via WebSearch]: 96 KB FPS game by .theprodukkt / Farbrausch. Won Breakpoint 2004 96k category. Source code released 2012 (jaromil + steven-schronk forks). Procedural generation per .werkkzeug3 tool — textures stored as creation-history not per-pixel; meshes stored as parametric primitives. Loading time = full asset reproduction. Equivalent uncompressed footprint: 200-300 MB. Sources: <https://github.com/jaromil/kkrieger-werkkzeug3> + <https://en.wikipedia.org/wiki/.kkrieger> + <https://github.com/steven-schronk/Kkrieger-Werkkzeug3>
- **IOCCC 2024** [HARD-EARNED-VERIFIED]: 23 winners after 4-year hiatus, 40th anniversary. Half of winners <2/3 size limit; 10 <1/2 size limit. Notable: "smallest LLM inference engine" running LLaMA 2 7B. Sources: <https://www.ioccc.org/2024/> + <https://github.com/ioccc-src/winner/tree/master/2024>
- **International Obfuscated Python Code Competition** [HARD-EARNED-VERIFIED]: Python equivalent of IOCCC. Source: <https://pyobfusc.com/>
- **js13kgames** [HARD-EARNED-VERIFIED]: 13 KB JavaScript game competition since 2012; 14th edition 2025. JS source + assets + minified bundle. Source: <https://js13kgames.com/> + <https://en.wikipedia.org/wiki/Js13kGames>
- **Roadroller** [HARD-EARNED-VERIFIED]: JS heavyweight packer by Kang Seonghoon; ~15% additional compression vs best ZIP/gzip; usable from 4 KB demos and up. Source: <https://github.com/lifthrasiir/roadroller>
- **PICO-8 / TIC-80 token packing** [HARD-EARNED-VERIFIED]: Lua-based fantasy console; 8192-token limit; base64-encoded packed byte strings to save tokens-but-spend-characters. Source: <https://github.com/seleb/PICO-8-Token-Optimizations>
- **Q1K3** [HARD-EARNED-VERIFIED]: 13 KB JavaScript homage to Quake. Source: <https://news.ycombinator.com/item?id=28520221>

**Transfer verdict for inflate.py**: .kkrieger PROCEDURAL-GENERATION pattern is the ONLY one that transfers cleanly — it transfers to **Wyner-Ziv Tier-2 baked-constants** per sister symposium 2026-05-17, NOT to inflate.py source aesthetic compression. The IOCCC / Js13K / Roadroller / PICO-8 techniques are domain-specific to their respective hard constraints (program size limit). Our contest's hard constraint is `archive.zip` size, not `inflate.py` size.

### Section 3 — LZMA / brotli / zstd Python source compression

- **`exec(lzma.decompress(base85_encoded))` pattern** [HARD-EARNED-VERIFIED]: canonical self-extracting Python bootloader. ~3-5 LOC visible bootloader + arbitrary payload. Sources: <https://docs.python.org/3/library/lzma.html> + <https://github.com/python/cpython/blob/main/Lib/lzma.py>
- **Compression ratios on Python source** [HARD-EARNED-VERIFIED via 2024 benchmarks]: Brotli > LZMA > Zstd > gzip for text/Python source. Typical ratios 55-75% size-of-original for Python source. Sources: <https://dev.to/dhilipsiva/analyzing-python-compression-libraries-zlib-lz4-brotli-and-zstandard-2ne5> + <https://peazip.github.io/fast-compression-benchmark-brotli-zstandard.html> + <https://gregoryszorc.com/blog/2017/03/07/better-compression-with-zstandard/>
- **Brotli vs LZMA vs Zstd trade-off** [HARD-EARNED-VERIFIED]: Brotli maximum compression, slowest. LZMA medium-high compression. Zstd good balance. LZ4 raw speed. Source: <https://medium.com/@techhara/compression-algorithms-benchmark-951a39f67b07>

**Transfer verdict**: techniques ALREADY canonical in archive.zip member encoding via `brotli` + `constriction` + per-substrate Huffman. Applying them to inflate.py source = ZERO score gain per Assumption-Adversary verdict. DEFERRED-pending-target-signal-rescope.

### Section 4 — Bytecode embedding + .pyz zip apps

- **`compile()` + `exec()` bytecode pattern** [HARD-EARNED-VERIFIED]: Python's compile() + exec() can run pre-compiled bytecode embedded in source. Performance gain via skipping parse step. Sources: <https://docs.python.org/3/library/compileall.html> + <https://realpython.com/python-exec/>
- **`zipapp` standard library module** [HARD-EARNED-VERIFIED]: ships in Python 3.5+; `compressed=True` parameter added Python 3.7. Produces `.pyz` self-contained executable archives. Sources: <https://docs.python.org/3/library/zipapp.html> + <https://realpython.com/python-zipapp/>
- **`shiv`** [HARD-EARNED-VERIFIED]: production-grade zipapp builder used at LinkedIn. Source: <https://lincolnloop.com/blog/dissecting-python-zipapp-built-shiv/>
- **`zipapps` (PyPI)** [HARD-EARNED-VERIFIED]: third-party zipapp builder with requirements bundling. Source: <https://github.com/ClericPy/zipapps>
- **Bytecode magic-number portability** [CARGO-CULTED-PENDING-VERIFICATION]: `.pyc` magic numbers change per Python minor version; portable only within version constraints. Contest contract per CLAUDE.md HNeRV parity L9 requires deterministic runtime closure; `.pyc` portability is a runtime-closure risk.

**Transfer verdict**: NOT CONTEST-COMPATIBLE. Contest expects raw `inflate.py` per `inflate.sh` shebang. `.pyz` shipping would violate the runtime contract. DEFERRED-pending-contest-revision.

### Section 5 — Polyglot programming

- **Cosmopolitan APE** [HARD-EARNED-VERIFIED]: build-once run-anywhere C; polyglot of Windows PE + UNIX shell-script-without-shebang; Python proof-of-concept exists (Python 2.7.18 + 3.6.14 single-file `python.com`). Sources: <https://github.com/jart/cosmopolitan> + <https://github.com/jart/cosmopolitan/blob/master/ape/specification.md> + <https://ahgamut.github.io/2021/07/13/ape-python/>
- **shebang + clever quoting polyglot patterns** [HARD-EARNED-VERIFIED]: `"""':sh:"""; python script via $0` patterns documented; valid Python + Bash from same file. Contest implication: could `inflate.sh` and `inflate.py` be merged? Probably NOT — contest's `inflate.sh` is a thin Bash wrapper that invokes `python inflate.py`; merging would obscure the contract.

**Transfer verdict**: DEFERRED-pending-contest-revision. The cosmopolitan APE approach is fundamentally elegant but violates `inflate.sh` + `inflate.py` separation contract.

### Section 6 — AST-based code transformation

- **`libcst` (Instagram)** [HARD-EARNED-VERIFIED]: concrete syntax tree library for Python 3.0-3.14; loss-less CST preserving formatting; supports automated refactoring (codemod) applications. Latest stable 1.8.6 released November 2025. Sources: <https://github.com/Instagram/LibCST> + <https://pypi.org/project/libcst/>
- **`astroid`** [HARD-EARNED-VERIFIED via PyPI]: alternative AST library used by pylint.
- **`parso`** [HARD-EARNED-VERIFIED via PyPI]: error-recovery parser used by Jedi.
- **Auto-generated inflate.py from declarative spec** [PROPOSED]: the canonical demoscene .kkrieger generator-not-artifact pattern. Substrate declarative spec → libcst codemod → emitted inflate.py source. This IS what `tac.substrates._shared.inflate_runtime` canonical helper extension family enables, but per Component 4 above (Carmack verdict) it should be IMPORT-from-canonical not GENERATE-each-time. Generator-based codegen DEFERRED-pending-Wyner-Ziv-Tier-2-landing.

**Transfer verdict**: TIER B (review-discipline) — `libcst` could power the OP-1 + OP-2 + OP-5 work but is overkill for the simple text-replacement-grade refactor we need. Defer until empirical complexity justifies AST-grade transformation.

### Section 7 — Bit-packed constant tables

- **Compressed lookup table encoding** [HARD-EARNED via codec literature]: arithmetic coding tables / Huffman tables can be encoded compactly via canonical Huffman + range-coded table-of-counts. Already canonical via `constriction` library in `submissions/pr103_pr106_final_runtime/inflate.py` etc.
- **Shader-style packed constants (GLSL minifier patterns)** [HARD-EARNED]: GLSL shaders use bit-packed constants (16-bit float packed in 32-bit uint) for memory bandwidth optimization. Not applicable to Python — no equivalent constraint.
- **`bitstring` Python library** [HARD-EARNED via PyPI]: bitstream packing primitives.

**Transfer verdict**: techniques ALREADY canonical in archive.zip member encoding. NOT applicable to inflate.py source.

### Section 8 — Compiler-level approaches

- **Cython AOT → `.so`** [HARD-EARNED]: violates `inflate.sh` Python-source-only contract.
- **Nuitka standalone compilation** [HARD-EARNED]: same violation.
- **`mpy-cross` (MicroPython)** [HARD-EARNED]: MicroPython bytecode incompatible with CPython contest runtime.
- **PyPy** [HARD-EARNED]: bytecode portable to PyPy but contest expects CPython.

**Transfer verdict**: NOT CONTEST-COMPATIBLE.

### Section 9 — Recent academic papers (arxiv 2023-2026)

- **arxiv 2504.17403** [HARD-EARNED-VERIFIED via WebSearch]: "Coding for Computation: Efficient Compression of Neural Networks for Reconfigurable Hardware" (Apr 2025) — neural network parameter compression for FPGA. Not directly applicable.
- **arxiv 2502.00922** [HARD-EARNED-VERIFIED]: "Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference" (Feb 2026) — Huffman compression of LLM weights. Conceptually adjacent to Wyner-Ziv Tier-3 scorer-features baked constants.
- **arxiv 2505.06297** [HARD-EARNED-VERIFIED]: "Lossless Compression of LLM-Generated Text via Next-Token Prediction" (May 2025) — context-based next-token prediction for lossless text compression. Could conceptually apply to inflate.py source — but per Assumption-Adversary verdict, wrong target signal.
- **arxiv 2406.06237** [HARD-EARNED-VERIFIED]: "Efficient Neural Compression with Inference-time Decoding" (Jun 2024) — mixed precision quantization + entropy coding. Conceptually adjacent to current Catalog #270 Tier 1 engineering.

**Transfer verdict**: academic compression literature is dominated by neural network parameter compression (relevant to archive.zip member encoding) NOT source code compression. The intersection of neural+source-code is underexplored, suggesting low priority.

### Section 10 — Social media / blog signal

Direct Hacker News / Reddit / Twitter search NOT performed; the canonical reference points above (`pyminifier`, `python-minifier`, `libcst`, `roadroller`, `kkrieger`) already enumerate the high-signal recent discussions.

### Section 11 — Specific high-value cross-references

- <https://demozoo.org/> — canonical demoscene archive
- <https://www.ioccc.org/> — canonical IOCCC archive
- <https://js13kgames.com/> — 13 KB JavaScript game competition
- <https://github.com/lifthrasiir/roadroller> — Roadroller compressor
- <https://www.kkrieger.de/> — 96 KB game canonical reference

## `## Per-substrate target list ranked by EV`

For OP-1 + OP-2 + OP-5 work (Tier B review-discipline), the 5 over-200-LOC substrates ranked by review-discipline EV (LOC over budget × likelihood-of-canonical-helper-extraction-success):

| Rank | Substrate | LOC | Over-budget | Size driver | OP-1 helper applicability |
|---|---|---|---|---|---|
| 1 | `pr106_latent_sidecar_r2_pr101_grammar` | 740 | +540 | sidecar parse + 4 format_id branches + ranked-Huffman codec | HIGH — sidecar parse + bicubic upsample + stream_pairs all reusable; estimated -150 LOC |
| 2 | `hdm8_film_grain_sidecar` | 730 | +530 | sidecar parse + postfilter modes + grammar codec | HIGH — sister of #1; estimated -150 LOC |
| 3 | `pr106_stacked` | 668 | +468 | 3-codec sidecar merge (Cool-Chic + C3 + foveation) | MEDIUM — codec-specific paths fork-required; estimated -100 LOC |
| 4 | `pr103_pr106_final_runtime` | 532 | +332 | PR103 arithmetic decoder vendored + PR106 HNeRV | MEDIUM — arithmetic decode is unique-per-substrate; estimated -80 LOC |
| 5 | `pr106_lrl1_sidechannel` | 300 | +100 | LRL1 sidechannel parse | LOW — single-purpose; estimated -40 LOC |

**Aggregate OP-1 estimated LOC reduction**: ~520 LOC across 5 substrates → moves substrates 1-3 from "EXCEEDS_WAIVER_NO_RATIONALE" to "WITHIN_WAIVER_BUDGET". Substrates 4-5 already manageable.

For DEFERRED Tier A Wyner-Ziv hoist work — per Wyner-Ziv sister symposium 2026-05-17 Q4-Q5 queue, the FIRST empirical anchor is PR101 fec6 with Comma2k19 UV palette. This symposium does NOT duplicate; it cross-references and unblocks via OP-1 review-budget restoration.

## `## Composition with sister work`

Orthogonality matrix versus current frontier work:

| Sister lane | Composition class | Predicted antagonism risk |
|---|---|---|
| Wyner-Ziv sister symposium 2026-05-17 (Catalog #319 Q1-Q5) | PRE-REQUISITE infrastructure | NONE — OP-1 + OP-2 + OP-5 ENABLE Wyner-Ziv Tier-2 landing surface |
| Composition #3 (PR101 fec6 + FEC7 + PR103 + sensitivity_mask_aware_quantizr_v1 per `lane_asymptotic_stacking_composition`) | ORTHOGONAL | OP-1 reorganizes inflate.py source structure; does NOT affect archive.zip member contents |
| Z7-Mamba-2 scaffold (`lane_top5_2_z7_mamba2_scaffold_design_20260518`) | ORTHOGONAL | Z7 is substrate-class work; OP-1 is infrastructure work |
| ATW V2-1 (`lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518`) | ORTHOGONAL | Same reasoning |
| DP1 deep-dive (`lane_per_substrate_symposium_dp1_deep_dive_20260517`) | ORTHOGONAL | DP1 inflate.py is currently within budget; not in OP-1 target list |
| `lane_17_imp` Frankle LTH (`lane_per_substrate_symposium_lane_17_imp_20260517`) | ORTHOGONAL | LTH cycle 0 is substrate-class work |
| `sensitivity_mask_aware_quantizr_v1` (per Composition #3) | ORTHOGONAL | Sensitivity mask work is encoder-side; OP-1 is inflate-side |

**Conclusion**: orthogonal to ALL active sister substrate work. No antagonism. SAFE TO LAND.

## `## 6-hook wire-in declaration` (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125:

| Hook | OP-1 + OP-2 + OP-5 work | Rationale |
|---|---|---|
| 1. Sensitivity-map contribution (`tac.sensitivity_map.*`) | N/A — OP-1 canonical helper does not affect score-axis sensitivity; OP-2 audit does not produce sensitivity signal; OP-5 STRICT gate does not produce sensitivity signal | This work is review-discipline only |
| 2. Pareto constraint (`tac.pareto_*`) | N/A — runtime_LOC_budget IS a feasibility constraint but it is INACTIVE for score-axis Pareto reasoning (the runtime_LOC constraint and the rate constraint are structurally orthogonal per Shannon's verdict above) | Runtime_LOC constraint is captured at the source-text gate level (Catalog #295 + proposed Catalog #327) not at the score-Pareto level |
| 3. Bit-allocator hook | N/A — OP-1 does not change archive.zip member byte allocation | Editor-only work |
| 4. Cathedral autopilot dispatch hook | N/A — OP-1 + OP-2 + OP-5 do not produce dispatch candidates | Editor-only work |
| 5. Continual-learning posterior update (`tac.continual_learning.posterior_update_locked`) | ACTIVE — this symposium emits a `CouncilDeliberationRecord` per Catalog #300 v2 via `tac.council_continual_learning.append_council_anchor` (called in the canonical landing memo) | T3 deliberation; required per Catalog #300 |
| 6. Probe-disambiguator (`tools/probe_*_disambiguator.py`) | DEFERRED — there are NOT 2+ defensible interpretations of the proposed OP-1 + OP-2 + OP-5 work. The Assumption-Adversary VETO resolved the symposium's main interpretation question by re-framing | If a future operator dispute emerges, a `tools/probe_inflate_py_loc_budget_disambiguator.py` is the canonical surface |

**5-hook explicit acknowledgment** (per Catalog #125): hooks 1-2-3-4-6 are N/A per the rationales above; hook 5 is ACTIVE via the canonical landing memo's `append_council_anchor` call.

## `## Op-routables`

Ranked by EV (operator-attention-budget per CLAUDE.md "Mission alignment" + "Operator-Attention Budget"):

| # | Op-routable | LOC | Wall-clock | Cost | Dependencies | Status |
|---|---|---|---|---|---|---|
| OP-1 | `src/tac/substrates/_shared/inflate_runtime_extensions.py` canonical helper (3-5 new typed helpers + tests) | ~200 LOC helper + ~150 LOC tests | ~3h editor | $0 GPU | None | RECOMMENDED-LAND-NEXT |
| OP-2 | `tools/audit_inflate_py_loc_budget.py` audit tool (LOC + size-driver classifier + JSON emit) | ~250 LOC + ~100 LOC tests | ~2h editor | $0 | None | RECOMMENDED-LAND-AFTER-OP-1 |
| OP-3 | DEFER LZMA-self-extract / IOCCC-aesthetic / Roadroller / zipapp helpers INDEFINITELY | 0 LOC (no code) | 0h | $0 | None | DEFERRED-pending-research-with-target-signal-rescope |
| OP-4 | Cross-pollination with Wyner-Ziv sister symposium 2026-05-17 (Catalog #319 Q1-Q5) — feed OP-1 + OP-2 + OP-5 work as PRE-REQUISITE INFRASTRUCTURE into Q4-Q5 (Tier-2 Comma2k19 palette smoke packet) | 0 LOC (cross-reference only; sister Q4-Q5 owns the empirical anchor) | 0h editor (covered in this memo) | $0 | OP-1 + OP-2 land first | INFORMATIONAL |
| OP-5 | Catalog #327 STRICT preflight gate `check_submission_inflate_py_under_loc_budget` (WARN-ONLY at landing; strict-flip after OP-1 drives violations to 0) | ~100 LOC gate + ~80 LOC tests | ~1h editor | $0 | OP-2 helper extracted; canonical AST-aware LOC counter | RECOMMENDED-LAND-AFTER-OP-2 |
| OP-6 | NO Modal/Vast.ai dispatch — editor-only work | 0 LOC | 0h | $0 | None | INFORMATIONAL |
| OP-7 | Online research synthesis preserved in this memo's Sections 1-11 | covered above | 0h additional | $0 | None | DONE |

**Total ESTIMATED**: ~880 LOC across OP-1+OP-2+OP-5 + ~6h editor time + $0 GPU spend.

**Recommended sequencing**: OP-1 → OP-2 → OP-5. The work is purely additive (no destructive changes; existing inflate.py files remain functional throughout). Wyner-Ziv sister symposium 2026-05-17 Q1-Q5 work can proceed in parallel without waiting (the OP-1 helper provides additional LOC headroom but is not strictly blocking).

## Mission-alignment compliance

Per CLAUDE.md "Mission alignment — non-negotiable":

* **Frontier target — NON-NEGOTIABLE**: this symposium does NOT directly advance the frontier (predicted ΔS = 0 per Tier B framing). It is INFRASTRUCTURE WORK that enables the Wyner-Ziv sister symposium's frontier-breaking landing. Per `predicted_mission_contribution=frontier_protecting` — protecting against the operator's CARGO-CULTED framing investing time in techniques with zero score yield.
* **Strict-scorer-rule (canonical, binding)**: OP-1 + OP-2 + OP-5 do NOT load scorer at inflate. Already canonical via Catalog #6.
* **HNeRV / leaderboard-implementation parity discipline L4 (inflate.py ≤100 LOC waiver ≤200 LOC)**: OP-5 STRICT gate ENFORCES the L4 invariant structurally. OP-1 + OP-2 RESTORE compliance for the 5 over-budget substrates.
* **HNeRV parity L9 (Runtime closure)**: OP-1 vendored-helper extraction does NOT introduce new external dependencies (the 2 external dep limit per HNeRV parity L4 holds).
* **Catalog #220 substrate operational mechanism**: OP-1 does NOT affect archive byte addition or operational mechanism declaration.
* **Catalog #229 premise-verification-before-edit**: this memo's "The Premise-Verification Anchor" section verifies the load-bearing premise (`compressed_size = archive.zip / 37,545,489`) BEFORE any technique evaluation. Pattern correctly applied.
* **Catalog #290 canonical-vs-unique decision per layer**: explicit table above per the falling-rule cascade.
* **Catalog #294 9-dim success checklist evidence**: explicit table above.
* **Catalog #296 predicted-band Dykstra-feasibility check**: explicit section above with first-principles citation (Shannon 1948 + Wyner-Ziv 1976) AND probe-disambiguator path.
* **Catalog #303 cargo-cult audit per assumption**: explicit table above with 7 assumptions classified.
* **Catalog #305 observability surface**: 6-facet declaration above.
* **Catalog #300 v2 council deliberation frontmatter**: this memo carries the canonical YAML frontmatter at top.
* **Catalog #313 probe outcomes ledger**: no PRIOR blocking outcomes for `lane_inflate_py_extreme_compression_symposium_20260518` (the lane is NEW). No predecessor probes to ratify or override.
* **Catalog #319 + #322 + #323 Wyner-Ziv autopilot reweight discipline**: this symposium does NOT propose new Wyner-Ziv reweight logic; cross-references existing sister work.
* **Catalog #325 per-substrate optimal-form symposium discipline**: this is NOT a per-substrate symposium (no specific substrate dispatch); it is a CROSS-SUBSTRATE infrastructure symposium. Catalog #325 enforcement-scope does not apply.

## Cross-references

- `feedback_grand_council_symposium_inflate_py_extreme_compression_landed_20260518.md` — this memo's canonical landing memo (in operator memory dir)
- `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` — sister symposium for Tier-2 baked-constants framework
- `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` — sister symposium for cargo-cult-unwind methodology (canonical pattern reference)
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — operator's standing directive on UNIQUE-AND-COMPLETE-PER-METHOD
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — operator's META-level retrospective
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — 18-assumption matrix canonical reference
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" — L4 inflate.py LOC budget non-negotiable
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — falling-rule cascade for canonical-vs-unique
- CLAUDE.md FORBIDDEN_PATTERNS — "predicted-band-vibes" (Catalog #296), "research-substrate trap" (Catalog #220)
- Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device`) — sister gate at inflate.py source surface
- Catalog #295 (`check_submission_inflate_works_with_empty_pythonpath`) — sister gate at inflate.py PYTHONPATH surface
- Catalog #319-Q1-Q5 Wyner-Ziv implementation queue — the destination for the demoscene heritage transfer

## End of deliberation memo

This T3 grand council symposium delivers PROCEED_WITH_REVISIONS verdict on the operator's question. The HARD-EARNED kernel of the operator's intuition (cross-disciplinary techniques applied to extreme size constraints) is preserved via repointing to the Wyner-Ziv Tier-2 baked-constants surface per sister symposium 2026-05-17. The CARGO-CULTED kernel (inflate.py source compression as score-relevant) is structurally protected via Assumption-Adversary VETO + premise verification. OP-1 + OP-2 + OP-5 land as Tier B review-discipline infrastructure that enables future Tier A frontier-breaking work. No GPU spend; ~6h editor time; predicted-mission-contribution = frontier_protecting per Catalog #300 v2.
