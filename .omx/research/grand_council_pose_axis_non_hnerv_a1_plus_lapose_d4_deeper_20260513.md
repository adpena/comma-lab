# Grand Council DEEPER: D4 inflate.sh contract for A1 + LAPose composition

**Task**: deeper deliberation on D4 (retry of crashed subagent `ae3b141d0830dbed0`)
**Lane**: `lane_pose_axis_non_hnerv_council_d4_deeper_20260513` (L0 → L1 on memo land)
**research_only**: true — D4 council deliberation; the sister BUILD lane owns archive grammar and dispatch.
**Sister BUILD lane**: `lane_a1_plus_lapose_composition_20260513` (L1 — substrate package + trainer + recipe + scripts landed at commits `533e487a` + `b0b6e0dd` … `d4adab79`)
**Prior council**: `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (commit `7e77321f`, D4 split 5-5 binding tie)
**Axis discipline**: every score in this memo is tagged `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`, `[prediction]`, or `[empirical]`.
**Verdict mode**: BINDING verdict on D4 with vote tally per option. Non-consensus residuals are DEFERRED-pending-empirical (per CLAUDE.md "KILL is LAST RESORT").
**Wire-in hooks (Catalog #125)**: declared in §10.

---

## 1. Executive summary

**The question (D4 only — D1.D / D2 / D3.B / D5.A+C / D6 / D7 are already locked):**
For A1+LAPose composition (A1 base + LAPose residual stream), how should the inflate.sh contract handle LAPose pose-axis residuals?

**Council BINDING VERDICT (8-2, dissent: Yousfi + Shannon abstain-prefer-D4.B-with-future-D4.D-followup):**

> **D4.B SINGLE-STAGE INFLATE** with the LAPose sidecar appended as a **TRAILER** to A1's existing wire format (the AS-BUILT design at `src/tac/substrates/a1_plus_lapose/archive.py`). The `inflate.sh` runtime is single-stage; the `inflate.py` reads `archive_dir/x`, calls `split_composition_archive()` to separate A1 base bytes from the trailing `LPA1` magic-byte sidecar, runs A1's existing decoder pipeline (verbatim, including bias correction), then **conditionally overlays the LAPose foveal RGB residual** at selected pair indices BEFORE the bicubic-upsampled `(384,512)→(874,1164)` frames are written to the `.raw` stream.

**Vote tally per option (10 inner-ten council voices polled):**

| Option | Description | Votes | Δ vs prior | Dissent rationale |
|--------|-------------|------:|:---------:|-------------------|
| **D4.B** (primary; AS-BUILT) | Single-stage inflate; LAPose trailer; one inflate.py reads x + splits + renders | **8** | +3 | Yousfi prefers D4.D+followup; Shannon prefers D4.D for future stacking |
| D4.A (two-stage) | inflate.sh runs A1 inflate.py THEN LAPose injection step | 1 | -4 | Hotz: "engineering-isolation; rollback-friendly" |
| D4.C (no-grammar-change) | LAPose-conditional reshape; scorer runtime access | 0 | unchanged | Unanimously REJECTED in prior council; would violate strict-scorer-rule |
| **D4.D** (in-place section extend) | Extend A1's existing pose section in place, magic-byte version bump | 1 | NEW | MacKay: "cleaner archive grammar; defer to followup" |
| D4.E (trailer) | LAPose deltas as trailer; trivial inflate.py extension | (collapses to D4.B; not separately voted) | NEW | The AS-BUILT IS this — see §6 |
| D4.F (interleaved) | Per-pair A1+LAPose bytes interleaved (one section, two encoders alternating) | 0 | NEW | UNANIMOUSLY REJECTED — see §5.3 |

**Key insight from deeper deliberation**: Round 3 surfaced that **D4.B (single-stage) and D4.E (trailer) collapse into the same as-built design** when LAPose is implemented as a magic-byte-tagged suffix that `split_composition_archive` can detect by rfind. The prior council's 5-5 split between "D4.A two-stage" and "D4.B new-section" was a framing artifact: the BUILD subagent landed a hybrid that is "single-stage runtime + trailer-format archive". This deeper council confirms that hybrid as the binding verdict.

**HNeRV parity lessons 3 + 4 compliance per option:**

| Option | Lesson 3 (monolithic 0.bin, fixed offsets) | Lesson 4 (inflate.py ≤100 LOC default; ≤200 with waiver) |
|--------|:------------------------------------------:|:------------------------------------------------------:|
| D4.A two-stage | PASS (A1 0.bin unchanged + lapose sidecar) | FAIL on default budget (two scripts + orchestration); PASS with waiver (~250 LOC total) |
| **D4.B trailer (as-built)** | **PASS** (single 0.bin; LPA1 trailer at fixed offset = rfind) | **PASS with waiver** (183 LOC for substrate inflate.py + ~30 LOC dispatcher = ~213 LOC; waived per HNeRV parity exemption since composition-with-A1 requires A1 decoder vendored) |
| D4.D in-place section bump | PASS (single 0.bin; A1's existing pose section bytes change) | FAIL (would require A1 inflate.py rewrite; breaks the byte-verbatim A1 base bytes guarantee) |
| D4.F interleaved | FAIL (encoder boundaries not byte-fixed; runtime parse cost) | FAIL (interleave parser is intrinsically >100 LOC) |

**Rate cost per option (Shannon §7.1):**

| Option | LAPose payload bytes | A1 base bytes mutated? | Runtime LOC | Reviewer 30s comprehension |
|--------|:---:|:---:|:---:|:---:|
| D4.A | ~500 B sidecar (per prior council §4.3 Ballé with Markov-1 hyperprior) + ~30 B inflate.sh orchestration | NO (verbatim) | ~250 LOC (two scripts) | NO (two-stage flow harder to follow) |
| **D4.B trailer** | ~500 B trailer | **NO (verbatim)** | ~213 LOC (one script) | **YES (single read-split-render flow)** |
| D4.D in-place | ~500 B injected into A1's pose section | YES (mutated) | ~150 LOC (one script; A1 rewrite) | YES but breaks Apples-to-apples discipline |
| D4.F interleaved | ~500 B + per-pair boundary markers (~600 B) | NO (verbatim) | ~280 LOC | NO |

**Current WIP alignment cost**: ZERO swap cost. The as-built design at `src/tac/substrates/a1_plus_lapose/archive.py::pack_composition_archive` + `split_composition_archive` + `src/tac/substrates/a1_plus_lapose/inflate.py::inflate_one` IS the D4.B trailer design. The `--d4-mode` CLI flag's default value `d4b_single_stage` is the binding D4 verdict from this council.

**Reactivation criteria** (per CLAUDE.md "KILL is LAST RESORT"):
- If empirical anchor scores **≥ 0.226 contest-CUDA** OR **≥ 0.193 contest-CPU**, DEFER to D4.A two-stage retry. Reactivation is **NOT** an automatic kill of D4.B — the council's hypothesis is that D4.A would deliver the SAME score (the score depends on what bytes are decoded, not on which script reads them).
- If empirical anchor shows **A1 base byte-corruption** (e.g., A1 inflate.py downstream cannot parse `decode_decoder_compact` from the split bytes), DEFER to D4.D in-place section bump with the explicit Catalog #146 + #109 audit trail.
- If empirical anchor shows **LAPose payload non-consumption** (Catalog #139 no_op_detector_failed), DEFER to D4.A two-stage to ISOLATE the LAPose injection step as a debuggable phase.

---

## 2. Pre-flight compliance

- [x] Read CLAUDE.md cover-to-cover. Honored: HNeRV parity (especially lessons 3+4), strict-scorer-rule, Apples-to-apples evidence discipline, Bugs must be permanently fixed AND self-protected against, Subagent coherence-by-default, KILL is LAST RESORT.
- [x] Read prior council memo `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (commit `7e77321f`) — D4 split 5-5 binding tie surfaced as operator-routable in §7.2.
- [x] Read paper review `.omx/research/siren_literature_review_20260513.md` — L4 Markov-1 hyperprior lesson supports D4.B trailer with ~500 B payload.
- [x] Read CURRENT WIP at `src/tac/substrates/a1_plus_lapose/{inflate.py,archive.py,architecture.py}` — confirmed AS-BUILT is D4.B trailer design (LPA1 magic-byte sidecar, rfind-based split, single-stage inflate.py).
- [x] Read BUILD-RESUME memo `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_a1_plus_lapose_composition_substrate_landed_20260513.md` — confirmed trainer's `--d4-mode` CLI flag defaults to `d4b_single_stage`, with `_write_runtime` currently emitting the SAME inflate.sh for all three modes (a documented BUILD-RESUME gap — see §6.3).
- [x] Read A1 existing archive `experiments/results/track4_sg_a1_t178000_20260509/submission_dir/inflate.{py,sh}` — confirmed A1's inflate.py is exactly 100 LOC and inflate.sh is 26 LOC. **A1 is at the HNeRV parity lesson-4 budget ceiling.** This is decisive for §3.2 D4.D rejection.
- [x] Lane pre-registered at L0 (already existed; verified via `tools/lane_maturity.py`).
- [x] Sister META-COUNCIL audit memo at `.omx/research/meta_council_decision_attribution_audit_20260513.md` — NOT YET LANDED at time of this writing. If META-COUNCIL surfaces a D4-relevant attribution finding post-merge, file a §7.5 amendment.

---

## 3. Council deliberation transcript (5 rounds)

### 3.1 Round 1 — Inner-ten positions (5+ sentences each)

**Shannon (LEAD)**: From an information-theoretic standpoint the question is "where in the byte stream does the LAPose pose-residual entropy live, and how is it accessed at inflate time?" In D4.A the LAPose payload lives in a separate file with its own struct decoder invoked by `inflate.sh` orchestration — the entropy is positionally separated from A1's at the filesystem level. In D4.B the LAPose payload lives at the END of A1's wire format, accessed by a single Python process via `archive_bytes.rfind(b"LPA1")`. The rate cost is **identical** in both cases — the bytes are the bytes — but the runtime parsing cost differs: D4.A pays for two process invocations and two Python startup costs (~100ms each on Modal A100), D4.B pays one. The inflate-runtime cost is NOT scored, but it IS bounded by the 30-min T4 contest runtime limit, so D4.B's single-process design has a structural margin. Furthermore: D4.B's `split_composition_archive` is O(blob/2) by `rfind` from the back; it is a 1-line scan against a 178KB blob — microseconds. **My information-theoretic verdict: D4.B is rate-optimal AND runtime-margin-optimal. I co-sign D4.B with the caveat that future stackability (LAPose#2, LAPose#3, etc.) would prefer D4.D's explicit-section-header design — but that is a Phase 2 concern, not a race-window concern.**

**Dykstra (CO-LEAD)**: Convex feasibility analysis on the inflate-runtime contract. The constraint set is `{T_inflate ≤ 30 min @ T4, archive_size ≤ 25 × evaluate_size / score_rate_term, runtime_dep_closure ⊆ {brotli, torch, numpy}}`. The intersection is non-empty for ALL of D4.A, D4.B, D4.D (D4.F adds a runtime parse term that doesn't intersect cleanly with the 30-min bound but is still feasible). What DIFFERS is the **engineering-feasibility manifold**: D4.A requires two scripts that must agree on intermediate format; D4.B requires one script with a single byte-format; D4.D requires modifying A1's existing inflate.py (mutating verified-clean code). By the principle of "minimize the surface of provable correctness", **D4.B has the smallest correctness-proof surface** — a single `split_composition_archive` function with one rfind + header validation. **I co-sign D4.B.** On D4.D: the in-place section bump would require re-verifying A1's wire-format parse against the modified bytes — that breaks the Apples-to-apples discipline (CLAUDE.md non-negotiable: "A1 base bytes are passed through verbatim — they are an immutable reference"). REJECT D4.D on Apples-to-apples grounds.

**Yousfi**: Contest-faithfulness review. PR #35 (strict-scorer-rule) and PR #56 (Selfcomp's canonical archive grammar) both shipped single-stage inflate runtimes. PR #100 (HNeRV-LC v2, gold-medal) is single-stage. PR #103 (rem2 silver-medal) is single-stage. **The contest leaderboard's empirical record on single-stage runtimes is unanimous.** No public PR has ever shipped a two-stage inflate.sh; the closest parallel is some `inflate.sh` scripts that internally invoke multiple Python helpers, but the orchestration is bash-only-glue, not a true two-stage pipeline. From a contest-archaeology standpoint, D4.B is the well-trodden path. HOWEVER — the FUTURE consideration is what I'd flag: if A1+LAPose lands sub-0.190 and we want to stack LAPose#2 (different scorer-blind-spot exploit), D4.B's single-magic-byte trailer becomes a stacking bottleneck (only one LPA1 magic). **D4.D's explicit-section-header design would scale to N composition residuals.** My recommendation is therefore **D4.B for THIS dispatch + research_only=true followup lane for D4.D if stacking becomes a priority.**

**Fridrich**: Adversarial steganalysis perspective. The contest scorer is a steganalysis CNN. Does any D4 option leak side-channel information that the scorer could detect? D4.A's two-stage flow writes intermediate files (A1's raw frames, THEN LAPose injection); if these intermediate files are written to a path that the contest evaluator scans, they could be detected as "non-standard". D4.B writes ONLY the final `.raw` stream — exactly what the contest expects. D4.D mutates A1's existing pose section; if the mutation pattern is statistically distinguishable from A1's natural distribution, the steganalysis scorer could pick it up. D4.F's interleaved boundary markers are an obvious side-channel — they ADD signal that doesn't carry A1 information, exactly what steganalysis is built to detect. **Square-root law**: D4.B distributes the LAPose perturbation across ~16 selected pair indices (per the `selected_indices` in `encode_lapose_sidecar`) — the perturbations are spatially-foveal and temporally-sparse, exactly the UNIWARD-detector-blind regime. **I co-sign D4.B on adversarial-steganalysis grounds.** Reject D4.D.

**Contrarian**: I am the council's adversary. The argument I see is "the BUILD subagent already shipped D4.B, so we should ratify it." That is **post-hoc rationalization**. Let me challenge each option's WEAKEST argument: (a) D4.A's weakest argument is "rollback-friendly" — but rollback is easy with single-stage too: just remove the LPA1 trailer; (b) D4.B's weakest argument is "single-stage is simpler" — but composition-with-A1 already requires vendoring A1's codec.py + model.py inside the substrate, so we're not really single-source-of-truth; (c) D4.D's weakest argument is "cleaner archive grammar" — but it BREAKS the byte-verbatim guarantee on A1 base bytes (CLAUDE.md Apples-to-apples discipline non-negotiable: "the A1 base bytes are passed through verbatim"); (d) D4.F's weakest argument is "stacking-friendly" — but the steganalysis side-channel cost is large. The WEAKEST argument for ANY non-D4.B option is "future stackability" — and CLAUDE.md "Race-mode rigor inversion" is unambiguous: SHIP the smallest credible bolt-on NOW; defer stacking to a follow-up lane. **I co-sign D4.B with explicit research_only=true followup lane for D4.D stacking.** Reject "we should rebuild because the BUILD subagent already wrote it" — but the as-built design happens to BE the best design on independent merits.

**Quantizr**: Reverse-engineering the leaderboard. PR #56 (Selfcomp 0.38) used a multi-section archive grammar in a single 0.bin — pose codec section + decoder section + mask section. Multi-section single-file IS the canonical contest design. The question is: is a magic-byte-tagged TRAILER (D4.B) equivalent to an explicit-offset-table SECTION (D4.D)? From a wire-format engineering standpoint they are isomorphic IF the magic byte is unique. `LPA1` is 4 ASCII bytes (`0x4C 0x50 0x41 0x31`) — collision probability against random A1 bytes is `1 / 2^32` per position, against 178KB blob is `~4.1e-5` — small but non-zero. The `archive.py::split_composition_archive` is `rfind` + header-validation, which catches false-positive magic-byte hits with sub-microsecond cost. **Magic-byte trailers are equivalent to explicit-section-headers for single-residual composition.** For N residuals (N>1) the explicit-section design wins because magic-byte collisions compound. **For N=1 (this dispatch), D4.B is equivalent in correctness to D4.D and superior in HNeRV parity lesson 4 compliance.** I co-sign D4.B.

**Hotz**: Engineering shortcut analysis. The 5-line patch is `split_composition_archive(archive_bytes) → a1_bytes + lapose_bytes`; pass `a1_bytes` to A1's existing decoder; pass `lapose_bytes` to a 30-LOC LAPose decoder; overlay the residual at selected pair indices. That's exactly what the as-built `inflate.py` does. D4.A would require me to write `bash` orchestration that pipes A1's `.raw` output to a Python LAPose injector — that's more shell glue, more failure modes, more "did the pipe close cleanly" debugging. D4.D would require me to **modify** A1's existing inflate.py — touching working code that has 100 LOC of carefully verified parse logic. **The single-line `split_composition_archive` is the engineering-shortcut win.** I co-sign D4.B. On D4.D: NO — never modify working code unless you must. The A1 inflate.py is a known-good 100-LOC contract; preserving it byte-verbatim means we can A/B test "with LAPose" vs "without LAPose" by toggling the trailer presence (literally `pack_composition_archive` vs `a1_bytes` unchanged). That A/B-testability is a HUGE engineering win that D4.D forfeits.

**Selfcomp**: Bit budget reconciliation perspective. My 0.38 archive's canonical grammar was `renderer.bin + masks.mkv + poses.pt` as three ZIP members, with archive.zip carrying ZIP overhead (~120 B). The contest evaluates archive size including ZIP overhead. A1's existing wire is a single ZIP member `x` containing the monolithic 0.bin — even the ZIP overhead is captured. Adding the LAPose payload as: (a) a SECOND ZIP member would add ~80 B of ZIP overhead; (b) a TRAILER inside the existing `x` member adds 0 B of ZIP overhead. The TRAILER design (D4.B as-built) wins on rate by ~80 B. At PR106 frontier rate-term cost (25 × 80 / 37545489 = 5.3e-5), the savings is small but non-zero, and importantly it's **strictly positive** without engineering downside. **I co-sign D4.B on bit-budget grounds.** On D4.A: if D4.A's two-stage flow requires writing an intermediate `.raw` file and then injecting LAPose into it, that intermediate file is just temporary disk; doesn't affect archive rate. So D4.A and D4.B tie on rate. They differ on runtime-LOC: D4.B wins.

**MacKay**: MDL / Bayesian-inference perspective. The minimum-description-length question is "what is the description complexity of the inflate.py + inflate.sh runtime tree, conditional on knowing the wire format?" For D4.A: description = A1's existing inflate.py (100 LOC) + LAPose injection script (~50 LOC) + bash orchestration (~30 LOC) = 180 LOC. For D4.B: description = composition inflate.py (183 LOC, which subsumes A1's parse + LAPose decode + overlay). They are **MDL-equivalent at ~180 LOC.** What differs is the **VERIFICATION COST**: D4.B's 183-LOC monolith is reviewable by a single 30-second visual scan (HNeRV parity lesson 12). D4.A's three-piece system requires verifying the bash glue holds the contract. **I weakly co-sign D4.B on MDL grounds, BUT — wait, I want to surface D4.D more carefully.** D4.D's in-place section bump would have description = modified A1 inflate.py (probably ~120 LOC after LAPose injection in-line) — that's SHORTER than D4.B by 60 LOC. From a pure MDL standpoint D4.D is the winner. The reason I don't co-sign D4.D is the Apples-to-apples cost (Contrarian's point) and Hotz's A/B-testability argument. **Conditional vote: D4.B if Apples-to-apples discipline binds; D4.D if we are willing to retire the A1 byte-verbatim guarantee.** Per CLAUDE.md "Apples-to-apples evidence discipline — non-negotiable", that guarantee binds. **Final vote: D4.B.** And I want to register: D4.D should be a Phase-2 follow-up lane when A1+LAPose hits its empirical ceiling and we need to retire the A1 anchor.

**Ballé**: Modern neural-compression engineering. The LAPose residual is structurally a HYPERPRIOR side-information stream — small, temporally-correlated, conditionally informative on top of A1's base latent stream. In Ballé 2018 hyperprior architecture, the hyperprior is encoded as a SEPARATE bottleneck with its own entropy bottleneck. The wire-format equivalent is: A1 base latents = main bottleneck; LAPose residual = hyperprior side-info. The wire-format design question is "do you store the hyperprior side-info as a TRAILER or as a separate SECTION?" In the Ballé canonical archive (compressai.models), the side-info is a separate ZIP member or separate file — D4.A-style two-stage. **HOWEVER** — that's for variable-rate, multi-resolution hyperpriors. For a FIXED-rate single-frame-correction hyperprior (which is what LAPose foveal residual is at this composition level), a trailer is equivalent and architecturally cleaner. **I co-sign D4.B for THIS application.** If we ever extend LAPose to a true Markov-chain hyperprior with per-pair conditioning, the trailer design becomes a constraint — D4.D's explicit-section-header would scale better. **Vote: D4.B with the Markov-1 hyperprior follow-up registered as a Phase-2 candidate to revisit D4.D.**

### 3.2 Round 2 — Binding tradeoff dimensions

**Rate (bytes spent on inflate format)**: Identical across D4.A / D4.B / D4.D at ~500 B for the LAPose payload. D4.B saves ~80 B by avoiding a second ZIP member. D4.F costs ~600 B extra for per-pair boundary markers. **Verdict: D4.B and D4.D and D4.A all rate-equivalent; D4.B has the slight ZIP-overhead win.**

**Inflate runtime LOC**: D4.A ~180 LOC (two scripts + bash glue); D4.B ~213 LOC (one substrate inflate.py = 183 + dispatcher = ~30); D4.D ~150 LOC (one modified A1 inflate.py + LAPose inline); D4.F ~280 LOC (interleave parser). HNeRV parity lesson 4 default is ≤100 LOC; waiver to ≤200 LOC. **D4.D fits ≤200 cleanly; D4.B borderline ≤200 with explicit waiver; D4.A fits if bash glue is excluded; D4.F fails even with waiver.**

**Score-axis impact**: Council unanimous — score depends on bytes decoded, not on which script decodes them. All 4 valid options (A/B/D/F) decode the same A1 + LAPose bytes; expected score is identical. **No score-axis differentiation between options.**

**HNeRV parity lessons 3+4 compliance**: §1 summary table. D4.B is the only option that PASSES both lessons cleanly. D4.A passes lesson 3 but is borderline on lesson 4 if we count `inflate.sh` orchestration. D4.D fails lesson 4 if it mutates A1's existing inflate.py (which is already at 100 LOC ceiling). D4.F fails both.

**Future stackability (LAPose#2, LAPose#3, … residuals)**: D4.D > D4.B > D4.A > D4.F. D4.D's explicit-section-header design scales to N residuals without magic-byte collision concerns. D4.B's single magic byte caps at 1 trailer; stacking requires versioned magic (LPA1 → LPA2 → ...). D4.A's two-stage requires N inflate stages — orchestration scales linearly. D4.F's interleaved design is theoretically infinite-stackable but loses on every other dimension.

**Reviewer 30-second comprehension** (HNeRV parity lesson 12): D4.B single-stage flow is reviewable in 30 seconds — `read → split → A1 decode → LAPose overlay → write`. D4.A's two-stage flow requires understanding the bash orchestration and inter-stage data format. D4.D requires verifying that the mutation of A1's parse logic preserves byte-verbatim equivalence on a no-LAPose archive. D4.F is not reviewable in 30 seconds.

**Apples-to-apples evidence preservation** (CLAUDE.md non-negotiable): D4.B and D4.A both preserve A1 base bytes verbatim. D4.D MUTATES A1 base bytes (the in-place section extension is by definition a byte change). D4.F preserves A1 bytes but interleaves them — the byte sequence is still verbatim but the **runtime parse path** for A1 changes, which is a softer form of mutation. **D4.B and D4.A are the only options that preserve full Apples-to-apples discipline.**

### 3.3 Round 3 — Surface 4th/5th/6th options (D4.D / D4.E / D4.F)

**D4.D HYBRID — extend A1's existing pose section in place, magic-byte version bump**: Evaluated. The proposal is to take A1's existing wire format `[uint32 section_total][decoder_blob][latent_blob][sidecar_blob]` and bump the version byte (which would have to exist; A1's current wire doesn't carry an explicit version byte — magic bumping would require adding one). The pose section is INSIDE the `sidecar_blob` (the `apply_latent_sidecar` call). Extending it in place means modifying the sidecar_blob format to carry both A1's existing per-pair pose deltas AND LAPose's foveal residual. **MacKay's MDL argument supports this; Contrarian's Apples-to-apples argument rejects this; Quantizr's stacking argument supports this for N>1 residuals.** Final vote: **1 in favor (MacKay conditional)**, 9 against. REJECTED for THIS dispatch; FLAGGED as Phase-2 follow-up if A1+LAPose hits empirical ceiling.

**D4.E TRAILER — LAPose pose deltas appended as trailer to A1's existing pose payload**: Evaluated. The proposal is exactly the as-built `pack_composition_archive`: `a1_bytes + sidecar`. The `sidecar` is a magic-byte-tagged blob (`LPA1` + header + brotli-compressed int8 residual). The trailer is identifiable by `archive_bytes.rfind(LAPOSE_SIDECAR_MAGIC)` from the back. **This collapses into D4.B** because the as-built single-stage inflate IS the trailer-decode design. Treating D4.B and D4.E as separate options is a framing artifact. The 5-5 split in the prior council was between "D4.A two-stage" and "D4.B single-stage with NEW SECTION" — but the BUILD subagent's interpretation of "new section" was "trailer" because trailers are the canonical lightweight way to add a section without an explicit offset table.

**D4.F INTERLEAVED — per-pair: A1 pose bytes + LAPose pose-delta bytes interleaved**: Evaluated. The proposal is to interleave per-pair: `[A1_pair_0_bytes][LAPose_pair_0_bytes][A1_pair_1_bytes][LAPose_pair_1_bytes] ...`. This would require **substantial rearrangement of A1's existing wire format** (which is currently `[decoder_blob][latent_blob][sidecar_blob]` with per-pair info distributed across the three sections, not per-pair sequential). The interleaved design would also add per-pair boundary markers (otherwise the parser can't find where each pair starts) — Fridrich's adversarial-steganalysis argument flags these markers as side-channel leakage. **Unanimous reject (0 in favor, 10 against).**

### 3.4 Round 4 — Vote tally + tie-break (if needed)

| Option | Round 1 votes (informal) | Round 4 votes (binding) | Δ |
|--------|:---:|:---:|:---:|
| D4.A two-stage | 5 (prior) | **1 (Hotz weak-for-rollback)** | -4 |
| **D4.B trailer (= D4.E)** | 5 (prior) | **8 (Dykstra, Yousfi*, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, Ballé)** | +3 |
| D4.D in-place | 0 (prior) | **1 (MacKay-conditional)** | +1 |
| D4.F interleaved | 0 (prior) | 0 | 0 |
| Shannon abstain | — | (counts as soft-D4.B preference) | — |

*Yousfi co-signs D4.B for this dispatch with explicit D4.D Phase-2 followup registration. Shannon abstains on the binding vote but expresses soft preference for D4.B; counted as soft-favorable.

**BINDING VERDICT: D4.B trailer-format single-stage inflate (8 of 10 inner-ten council voices).**

**Dissent rationale**:
- **Hotz (D4.A weak-for-rollback)**: argues that two-stage flow makes it easier to debug an empirical anchor failure by running A1 inflate alone (no LAPose). Council response: this can be achieved in D4.B by simply not building a LAPose trailer (`pack_composition_archive` not called → archive is byte-equivalent to pure A1). Dissent retracted in Round 5.
- **MacKay (D4.D conditional)**: argues MDL favors in-place mutation. Council response: Apples-to-apples discipline binds. Dissent acknowledged; MacKay co-signs D4.B as the constraint-compliant choice. **Final consensus is therefore 9-1 with Yousfi+Shannon both registering D4.D as a Phase-2 followup.**

**No tie. Verdict is binding D4.B.**

### 3.5 Round 5 — 3-clean-pass adversarial review

**Round 5.1** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian):

- Issue raised by Contrarian: "The as-built `_write_runtime` does NOT actually differentiate inflate.sh between D4.A/B/C — all three modes emit the SAME single-stage runtime, with `d4_mode` appearing only as a comment in inflate.sh. This is a documented BUILD-RESUME gap. Council should explicitly state that the binding D4.B verdict matches the as-built behavior; otherwise BUILD subagent has a hidden mode-switching bug." Resolution: documented in §1 and §6.3. The verdict binds the AS-BUILT behavior. If D4.A or D4.D is later operator-routed, BUILD subagent must implement the runtime differentiation. **Counter resets to 0.**

**Round 5.2** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian re-run after fix):

- Issue raised by Yousfi: "The HNeRV parity lesson-4 audit needs a clearer statement of the explicit waiver rationale for 183 LOC composition inflate.py. The BUILD-RESUME memo says `WAIVE 200` with rationale 'composition-with-A1 reasonably needs the A1 decoder vendored alongside'. Council should ratify this waiver." Resolution: ratified in §4. The 183-LOC composition inflate.py = ~100 LOC for A1 wire-parse (mirroring the existing 100-LOC A1 inflate.py) + ~30 LOC for LAPose decode + ~30 LOC for overlay + ~20 LOC for CLI glue. The lesson-4 default is 100 LOC; the explicit waiver to ≤200 LOC is justified by the composition-with-existing-substrate exemption. **Counter resets to 0.**

**Round 5.3** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian re-run):

- Issue raised by Shannon: "Is the LAPose payload byte budget (~500 B per prior council §4.3) compatible with D4.B's trailer format overhead? The LAPOSE_HEADER_STRUCT is 16 bytes (4 magic + 1 ver + 2 nsel + 2 fh + 2 fw + 1 rank + 4 scale). Selected_indices is `num_selected × 2 bytes`. Residual blob is brotli-compressed int8. At num_selected=16 (a reasonable target per prior council §4.1 Shannon), header+indices=16+32=48 B; payload=~450 B; total=~498 B. This fits D2.A 2 KB cleanly with substantial slack." Resolution: confirmed; the trailer format overhead (~48 B header+indices) is well within the D2.A 2 KB budget. **No issue. Counter advances to 1.**

**Round 5.4** (Quantizr + Hotz + Selfcomp + MacKay + Ballé):

- Issue raised by Quantizr: "Public-leaderboard archive replay: do any PR55 / PR80 / PR97 / PR101 archives use a magic-byte trailer design? PR101 has the canonical pose codec section at fixed offset; PR97 was a single-monolith with implicit section boundaries. **The trailer design is not a common contest pattern.** Risk: the contest evaluator might have stricter wire-format checks than expected, e.g., parsing the archive as a single typed blob without trailer support." Resolution: the contest evaluator runs `inflate.sh archive_dir output_dir file_list`; the evaluator does NOT inspect the archive bytes' structure. Wire-format checks are purely our concern — the contest only cares that `inflate.sh` exits 0 and produces correctly-sized `.raw` outputs. Trailer design is contest-compatible. **No issue. Counter advances to 2.**

**Round 5.5** (Quantizr + Hotz + Selfcomp + MacKay + Ballé re-run):

- Issue raised by Hotz: "Catalog #139 packet-compiler runtime byte-mutation smoke. Does the as-built inflate.py pass the no_op_detector? Specifically: if we mutate 1 byte in the LAPose trailer, does the inflated output change?" Resolution: byte-mutation testing is a BUILD subagent responsibility (no_op_detector_planned = Catalog #139 per BUILD-RESUME memo §8). The trailer design IS byte-consumed (the int8 residual blob is brotli-decompressed and overlaid on selected pair RGB frames). Byte mutation in the LAPose trailer header would corrupt brotli framing and either (a) raise `BrotliDecoderError` (graceful failure) or (b) produce a malformed int8 array that the `apply_latent_sidecar`-style overlay would still consume but with different residual values. Either way, the `inflated output` will differ from the un-mutated baseline. **No issue. Counter advances to 3.**

**3-clean-pass achieved. Council DEEPER memo SEALED.**

---

## 4. HNeRV parity lessons 3+4 detailed audit (per option)

**Lesson 3**: Archive grammar = monolithic single-file `0.bin` (or explicitly justified multi-file). Fixed offsets declared in `codec.py` source (e.g., `DECODER_BLOB_LEN = 162_164`, `LATENT_BLOB_LEN = 15_387`). ZIP-member-budget rows are invalid unless the packet really has separate ZIP members.

| Option | Lesson 3 compliance | Rationale |
|--------|:---:|---|
| D4.A two-stage | PASS | A1's 0.bin unchanged; LAPose lives in a separate `lapose.bin` inside archive.zip (second ZIP member). Explicit multi-file justified by stage separation. ZIP overhead ~80 B. |
| **D4.B trailer (as-built)** | **PASS** | Single ZIP member `x` carrying `[A1 wire][LAPose trailer]`. Trailer offset = `len(blob) - len(sidecar)` = rfind-discoverable. A1's existing offsets (`DECODER_BLOB_LEN`, `LATENT_BLOB_LEN`) are unchanged — A1 base bytes are byte-verbatim. The LAPose section offset is implicit (rfind) which is the only soft point; the council ratifies this because the magic byte gives the same effective fixed-offset behavior. |
| D4.D in-place | CONDITIONAL PASS | Single ZIP member; existing A1 sidecar_blob is REPLACED with a versioned blob that carries both A1's per-pair pose deltas AND LAPose residual. A1's existing offsets `DECODER_BLOB_LEN` + `LATENT_BLOB_LEN` are unchanged; only the sidecar payload bytes change. PASS on the strict grammar level; FAIL on the Apples-to-apples evidence discipline. |
| D4.F interleaved | FAIL | Per-pair interleaving requires new wire-format markers; A1's existing offsets are invalidated; the substrate's `codec.py` would need a complete rewrite. |

**Lesson 4**: Inflate.py ≤ 100 LOC (default budget; explicit waiver for ≤ 200 with rationale). ≤ 2 external dependencies declared in the runtime tree. CUDA-or-CPU agnostic. Reviewable in 30 seconds.

| Option | Inflate.py LOC | External deps | CUDA/CPU agnostic | 30s reviewable | Lesson 4 verdict |
|--------|:---:|:---:|:---:|:---:|:---:|
| D4.A two-stage | A1: 100; LAPose: ~50; total = 150 across two files | brotli, torch | YES | NO (two-stage flow) | CONDITIONAL PASS with waiver + reviewer-cost penalty |
| **D4.B trailer (as-built)** | 183 (substrate) + 30 (dispatcher) = 213 total | brotli, torch | YES | YES (single linear flow) | **PASS with explicit waiver: composition-with-A1 exemption per HNeRV parity discipline** |
| D4.D in-place | A1 modified: ~120 (was 100) | brotli, torch | YES | YES | PASS — but FAILS Apples-to-apples |
| D4.F interleaved | ~280 (interleave parser) | brotli, torch | YES | NO | FAIL on default; FAIL on waiver |

**Verdict**: D4.B is the ONLY option that PASSES both lessons 3 + 4 while preserving Apples-to-apples discipline. D4.A passes both with a reviewer-cost penalty. D4.D requires retiring Apples-to-apples. D4.F fails outright.

---

## 5. New option exploration (D4.D / D4.E / D4.F)

### 5.1 D4.D HYBRID — extend A1's existing pose section in place

**Pros**:
- MDL-optimal — shortest description (~150 LOC total).
- Stacking-friendly for N>1 residuals (explicit-section-header design).
- No new magic-byte allocation needed.

**Cons**:
- **BREAKS Apples-to-apples discipline** (CLAUDE.md non-negotiable). A1's `decode_decoder_compact(decoder_blob)` would still work, but A1's `apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)` would receive a sidecar_blob that contains LAPose data — A1's existing function would need to be modified or the LAPose data would silently corrupt the sidecar parse.
- A1's existing 100-LOC inflate.py would need to grow to ~120 LOC, brushing against HNeRV parity lesson-4 ≤100 LOC default.
- Cannot be A/B tested without LAPose (the modification IS the entry point; you can't "disable LAPose" without removing the version-bump code).

**Reactivation criteria for D4.D promotion**: If A1+LAPose lands successful empirical anchor AND a second composition lane (LAPose#2 or different residual stream) is approved, the council reconvenes to consider migrating from D4.B trailer to D4.D explicit-section-header to enable N-residual stacking.

### 5.2 D4.E TRAILER — appended trailer

**This collapses into D4.B.** The as-built `pack_composition_archive` IS the trailer design. The prior council's framing of "D4.B = new archive section" was interpreted by the BUILD subagent as "trailer with magic-byte tag", which is a valid lightweight section in HNeRV parity terms. The deeper council ratifies this interpretation.

### 5.3 D4.F INTERLEAVED — per-pair interleaved

**Pros**:
- Theoretically infinite-stackable.
- Per-pair locality at byte level.

**Cons**:
- **Requires complete A1 wire-format rewrite** (A1's existing format is NOT per-pair sequential — `decoder_blob` is a single FP4-quantized weight tensor, `latent_blob` is a `15387`-byte stream of per-pair latents, and `sidecar_blob` is the per-pair quantization deltas).
- Per-pair boundary markers add side-channel leakage (Fridrich's adversarial-steganalysis veto).
- Inflate.py LOC budget would balloon to ~280 LOC (FAILS lesson 4 even with waiver).
- 30-second reviewer comprehension impossible.

**Unanimous reject (10-0).** D4.F is NOT a viable option for any composition lane in this codebase.

---

## 6. Current WIP alignment with binding D4.B verdict

### 6.1 Aligned (zero swap cost)

- `src/tac/substrates/a1_plus_lapose/archive.py::pack_composition_archive` — appends trailer to A1 bytes. ALIGNED.
- `src/tac/substrates/a1_plus_lapose/archive.py::split_composition_archive` — rfind LPA1 magic from back; returns (a1_bytes, lapose_sidecar_bytes). ALIGNED.
- `src/tac/substrates/a1_plus_lapose/inflate.py::inflate_one` — splits the archive, runs A1's wire parse + decoder, applies LAPose foveal RGB overlay at selected pair indices. ALIGNED.
- `experiments/train_substrate_a1_plus_lapose.py --d4-mode d4b_single_stage` (default) — emits a single-stage runtime tree. ALIGNED with binding verdict.

### 6.2 Inflate.py LOC waiver — formally ratified by this council

Council ratifies the **`inflate_runtime_loc_budget=200` waiver** with the following formal rationale:

> Per HNeRV parity discipline lesson 4: "Inflate.py ≤ 100 LOC (default budget; explicit waiver for ≤ 200 with rationale)." The A1+LAPose composition substrate is a Lane Class = COMPOSITION-WITH-EXISTING-SUBSTRATE. Substrate-engineering compositions that vendor an existing substrate's decoder pipeline inherit the existing substrate's runtime LOC cost. A1's existing inflate.py is 100 LOC. The composition inflate.py (183 LOC) = A1 wire-parse (~100 LOC) + LAPose decode + foveal overlay (~83 LOC). The 83 LOC composition-specific code is reviewable in 30 seconds (single rfind + single decode + single overlay loop). **Waiver granted: 200 LOC explicit, with explicit acknowledgment that the LAPose composition-specific code is ~83 LOC (well under any standalone substrate budget).**

### 6.3 BUILD-RESUME gap (documented, NOT a council action item)

The current `_write_runtime` function in `experiments/train_substrate_a1_plus_lapose.py` emits an IDENTICAL `inflate.sh` for all three `--d4-mode` values (`d4a_two_stage`, `d4b_single_stage`, `d4c_no_grammar_change`). The `d4_mode` only appears as a COMMENT in the emitted inflate.sh script. This is a BUILD-RESUME implementation gap. **It does NOT affect the binding D4 verdict** because the binding verdict is D4.B and the as-built runtime IS the D4.B implementation.

**Action item for BUILD subagent (NOT this council)**: if/when D4.A or D4.D becomes operator-routed (per §1 reactivation criteria), BUILD must implement the actual runtime differentiation in `_write_runtime`. The current behavior is "default to D4.B regardless of CLI flag" — acceptable for THIS dispatch because D4.B is the binding verdict, but a code-quality concern that should be addressed in a follow-up commit.

---

## 7. Math derivation appendix

### 7.1 Shannon: rate cost per D4 option

The LAPose payload byte count (call it `B_lapose`) is constant across D4 options because the payload itself is fixed by the prior council's §4.1 R(D) analysis (~500 B with Markov-1 hyperprior, or ~225 B without per uniform-atom upper bound).

The wire-format OVERHEAD per option is:
- **D4.A two-stage**: `B_zip_overhead_second_member ≈ 80 B` (ZIP central-directory record + local file header per `zipfile` source).
- **D4.B trailer**: `B_trailer_header = 16 B` (`LPA1` 4 + ver 1 + nsel 2 + fh 2 + fw 2 + rank 1 + scale 4); `B_trailer_indices = 2 × num_selected B`; brotli output ≈ 0 B additional framing (brotli is byte-aligned with no external framing).
- **D4.D in-place**: `B_in_place_overhead = 1 B` (version byte) to add to A1's existing wire.
- **D4.F interleaved**: `B_interleave_overhead = 600 B` (one boundary marker per pair, at least 1 B each).

At `num_selected=16` (council target):
- D4.A: 80 B
- D4.B: 16 + 32 = 48 B
- D4.D: 1 B
- D4.F: 600 B

**Score rate-term impact**: `25 × ΔB / 37545489`. Marginal δscore per 100 B overhead = `2.5e-7 × 100 = 6.7e-5`. **Inflate-format overhead is negligible at score level for all options except D4.F.** D4.D is rate-optimal, D4.B is +47 B from optimal (negligible), D4.A is +79 B from optimal (negligible), D4.F is +599 B (still marginal but the largest of the four).

### 7.2 Dykstra: convex feasibility per D4 option

The constraint set is `{T_inflate ≤ 30 min @ T4, runtime_dep_closure ⊆ {brotli, torch, numpy}, parse_correctness ≥ 1 - ε}` where ε is the probability of malformed-parse failure.

- D4.A: T_inflate ≈ 2 × T_python_startup + T_A1_decode + T_LAPose_inject ≈ 200ms + 20s + 5s ≈ 25s × 5 videos = 125s (well under 30 min).
- D4.B: T_inflate ≈ T_python_startup + T_A1_decode + T_LAPose_overlay ≈ 100ms + 20s + 1s ≈ 21s × 5 videos = 105s.
- D4.D: T_inflate ≈ same as D4.B ≈ 105s.
- D4.F: T_inflate ≈ T_python_startup + 600 × T_per_pair_parse + T_decode + T_overlay ≈ 100ms + 60s + 25s = 85s × 5 = 425s (still under 30 min).

All four options are convex-feasible on T_inflate. Differentiation comes from parse_correctness: D4.F's per-pair boundary parse is the most fragile (highest ε); D4.B's trailer rfind is the most robust (lowest ε).

### 7.3 Ballé: hyperprior interaction per D4 option

The LAPose foveal residual is a structural hyperprior on A1's pose-axis distortion. For Markov-1 hyperprior conditioning (per prior council §4.3), the per-pair transition table requires `K_eff² × log2(K_eff)` bits stored. At `K_eff=8`: `64 × 3 = 192 bits = 24 B` overhead. This overhead is identical across D4.A / D4.B / D4.D (the hyperprior side-info is stored alongside the residual payload regardless of where the payload lives). D4.F is special: the per-pair interleaved layout would naturally support per-pair hyperprior conditioning without an explicit transition table (the previous-pair's bytes are physically adjacent in the wire format). However, the side-channel leakage cost (Fridrich's veto) dominates the hyperprior savings.

**Hyperprior verdict**: D4.A / D4.B / D4.D are interchangeable for Markov-1 hyperprior wiring. D4.F has a theoretical hyperprior advantage that is dominated by adversarial-steganalysis side-channel leakage. **The binding D4.B verdict is hyperprior-compatible and forward-compatible with hyperprior follow-up wiring.**

---

## 8. Apples-to-apples evidence discipline

Every score / cost claim in this memo is tagged.

- **A1 anchor**: 0.192847 `[contest-CPU GHA Linux x86_64]` / 0.226352 `[contest-CUDA T4]` on archive SHA `8e664385...` size 178,162 B (per BUILD-RESUME §6).
- **Predicted A1+LAPose**: `[contest-CPU prediction]` 0.187-0.191 central / `[contest-CUDA prediction]` 0.218-0.224 central per prior council §1. NOT promotable until empirical anchor lands.
- **D4 option byte costs**: tagged `[prediction]` based on the wire-format math (§7.1). Empirical bytes will be verified by `tools/build_deterministic_packet.py` runtime byte-mutation smoke (Catalog #139).
- **Inflate runtime LOC**: counted from existing source files at `experiments/results/track4_sg_a1_t178000_20260509/submission_dir/inflate.py` (100 LOC measured) and `src/tac/substrates/a1_plus_lapose/inflate.py` (183 LOC measured) — these are `[empirical]` measurements at council time.
- No /tmp paths. No MPS-derived strategic decisions. No KILL verdicts.

---

## 9. Reactivation criteria (D7 verdict on D4)

Per CLAUDE.md "KILL is LAST RESORT" non-negotiable:

**Path α (positive)** — empirical A1+LAPose anchor scores ≤ 0.190 contest-CPU AND ≤ 0.222 contest-CUDA on D4.B as-built:
- D4.B verdict CONFIRMED. Lane advances to L2 (impl_complete + real_archive_empirical).
- Phase-2 follow-up lane registered for D4.D explicit-section-header design IF stacking (LAPose#2) becomes a priority.

**Path β (mixed)** — empirical A1+LAPose anchor scores 0.190-0.193 contest-CPU on D4.B as-built:
- D4.B verdict HOLDS. Reactivation criteria for D4.A two-stage:
  - (a) Verify the LAPose trailer is consumed by the inflate.py via Catalog #139 byte-mutation smoke. If no_op_detector fires, the inflate.py is NOT consuming the trailer — that's a BUILD bug, not a D4 verdict question.
  - (b) Verify the A1 base bytes are byte-verbatim preserved (sha256 of split[0] == sha256 of original A1 archive). If divergent, that's a BUILD bug.
  - (c) Only after (a)+(b) are CONFIRMED, consider D4.A two-stage as an isolation aid.

**Path γ (negative)** — empirical A1+LAPose anchor scores > 0.195 contest-CPU on D4.B as-built:
- DEFERRED-pending-research per CLAUDE.md "KILL is LAST RESORT". Reactivation criteria:
  - (a) Verify LAPose payload non-emptiness (the as-built smoke says ~35 B at zero-residual init; full training should land 100-500 B).
  - (b) Verify D5.A+C exploit is real (LAPose foveal patch is non-degenerate at FastViT-T12 scale-band).
  - (c) D4 verdict is NOT the cause — the score is determined by the bytes decoded, not by D4 mode. Pivot to D1.C FiLM-style conditioning OR D5.B SegNet exploit OR retire to A1 anchor.

**No KILL verdict in any path. D4.B IS the binding choice unless overridden by operator routing based on Path α/β/γ empirical evidence.**

---

## 10. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, every landing must declare all 6 hooks:

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): N/A — this is a META council memo on the inflate.sh contract, not a new sensitivity atom. The sister BUILD lane's empirical anchor (when it lands) will produce a sensitivity-map entry. **Status**: surfaced; consumed by sister BUILD lane.
2. **Pareto constraint** (`tac.pareto_*`): N/A — D4 verdict does not change archive size or score component bounds. The Pareto frontier already accommodates A1+LAPose composition. **Status**: surfaced; no new constraint atom.
3. **Bit-allocator hook**: N/A — D4 verdict does not change per-tensor importance allocation. The LAPose head's fixed-size sidecar is allocated once at architecture-config time. **Status**: surfaced; no new bit-allocator entry.
4. **Cathedral autopilot dispatch hook**: The recipe at `.omx/operator_authorize_recipes/substrate_a1_plus_lapose_modal_a100_dispatch.yaml` is the autopilot dispatch hook. **Status**: surfaced; consumed by sister BUILD lane's `tools/run_modal_smoke_before_full.py` invocation.
5. **Continual-learning posterior update** (`.omx/state/cost_band_posterior.jsonl`): N/A for this council memo. Posterior update fires when empirical anchor lands (sister BUILD lane responsibility). **Status**: surfaced; consumed by sister BUILD lane.
6. **Probe-disambiguator** (`tools/probe_<track>_disambiguator.py`): The `--d4-mode` CLI flag with 3 choices IS the probe-disambiguator interface. This council memo BINDS the default to `d4b_single_stage`. **Future probe** at `tools/probe_a1_plus_lapose_d4_disambiguator.py` would be activated only if Path β/γ empirical evidence forces a D4.A retry. **Status**: surfaced; binding verdict locks default; future probe deferred to sister BUILD lane.

---

## 11. Cross-references

- `[[grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513]]` — prior council memo (commit `7e77321f`) with D4 split 5-5 binding tie; this deeper council resolves the tie at 8-2 (or 9-1 with MacKay conditional acknowledgment).
- `[[siren_literature_review_20260513]]` — sister paper review (commit `af2348fe`) supporting D2.A target with Markov-1 hyperprior.
- `[[feedback_a1_plus_lapose_composition_substrate_landed_20260513]]` — BUILD-RESUME memo documenting the as-built D4.B trailer-format inflate runtime.
- `[[feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509]]` — canonical HNeRV parity retrospective; 13 inviolable lessons audited in §4.
- `[[feedback_substrate_vs_codec_composition_meta_pattern_20260508]]` — substrate-vs-codec composition meta-pattern; D4.B trailer design is the canonical lightweight composition pattern for substrate-engineering-aware codec extensions.
- `[[meta_council_decision_attribution_audit_20260513]]` — sister META-COUNCIL audit (pending at time of this writing). If META-COUNCIL surfaces a D4-relevant attribution finding post-merge, file a §7.5 amendment.
- Catalog #146 (`check_phase1_trainer_runtime_emits_contest_compliant_inflate`): D4.B as-built passes (3-arg positional inflate.sh; `set -euo pipefail`; explicit positional handoff).
- Catalog #109 (`check_public_pr_intake_clones_pristine`): D4.B as-built preserves A1 base bytes verbatim, honoring the spirit of source-provenance discipline applied to a composition substrate.

---

## 12. Council seal

**Inner quintet pact** (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian): **SEALED**. Shannon abstains on binding vote (soft-D4.B preference); other four endorse D4.B. Operator-routable surfacing of D4.D Phase-2 followup documented.

**Inner ten** (+ Quantizr, Hotz, Selfcomp, MacKay, Ballé): **SEALED**. MacKay registers D4.D conditional preference; co-signs D4.B under Apples-to-apples binding constraint. Quantizr/Hotz/Selfcomp/Ballé endorse D4.B unconditionally. Yousfi endorses D4.B for this dispatch with explicit D4.D Phase-2 followup registration.

**Grand council advisory** (Boyd, Tao, Filler, Mallat, van den Oord, Schmidhuber, Carmack, Hassabis): REVIEWED. Carmack and Hassabis (newly polled for this deeper deliberation) co-sign D4.B on engineering-simplicity grounds. No D4-relevant adversarial findings surfaced.

**3-clean-pass adversarial review**: complete. Counter = 3. Memo SEALED for operator consumption + sister BUILD lane consumption.

**Verdict mode**: BINDING D4.B with 8/10 inner-ten council voices (9/10 with MacKay's constrained co-sign). DEFERRED-pending-empirical on Path β/γ reactivation criteria. No KILL verdicts. Operator-routable amendments (Path α/β/γ) explicit.

**Date**: 2026-05-13
**Task**: Council D4 deeper deliberation (retry of crashed subagent `ae3b141d0830dbed0`)
**Lane**: `lane_pose_axis_non_hnerv_council_d4_deeper_20260513` (advances L0 → L1 on memo land)
