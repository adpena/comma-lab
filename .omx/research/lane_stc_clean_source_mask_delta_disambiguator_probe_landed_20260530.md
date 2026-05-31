<!-- SPDX-License-Identifier: MIT -->
<!-- DOCS_LOCAL_PATH_OK:no_local_absolute_paths_in_this_memo_per_Catalog_208 -->
---
council_tier: T1
council_attendees: [Yousfi, Fridrich, Filler, Shannon, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "lane_stc_clean_source needs a NEW STC codec built; the symposium/lane do not exist"
    classification: CARGO-CULTED
    rationale: "FALSIFIED by search-first inventory — lane_stc_clean_source EXISTS at L1 in the registry; council_per_substrate_symposium_stc_clean_source_20260517.md EXISTS (PROCEED-WITH-REVISIONS); the canonical Filler-Fridrich STC codec tac.codec.syndrome_trellis_codec + pose_filler_stc_codec EXIST with 73 tests. My FIRST design-memo draft (deleted) wrongly claimed these did not exist and imported a non-existent module tac.codec.stc_pose_encoder + non-existent selfcomp_archive_bytes/GrayscaleLUT. STAND_DOWN on rebuild; iterate the EXISTING codec via the $0 probe."
  - assumption: "the Selfcomp grayscale-LUT is the canonical STC clean-source cover"
    classification: CARGO-CULTED
    rationale: "The prompt framed STC as a DELTA over Selfcomp grayscale-LUT. The PROCEEDed symposium (2026-05-17) does NOT bless that cover — its CC#1 unwind is STC on the ternary mask-DELTA stream (the canonical Filler sparse-ternary target). The original 2026-04-29 FALSIFICATION was STC on DENSE mask-argmax. The Selfcomp-LUT-cover idea is an UNTESTED alternative I do not pursue; the symposium-canonical target is mask-delta, which is what the $0 probe measures."
  - assumption: "STC near-optimality (sparse additive distortion) makes it beat brotli on the mask-delta stream"
    classification: CARGO-CULTED
    rationale: "EMPIRICALLY FALSIFIED by THIS probe: uniform-cost STC self-syndrome bytes are 2.4-2.6x LARGER than brotli(mask-delta) at every contest-realistic sparsity (rho<=0.10). The symposium §6 reactivation bar (STC < brotli by >=5%) is NOT met. This is a research-DEFERRAL not a kill (the detector-informed cost map CC#2 is the untested next revision)."
  - assumption: "the contest scorer rewards minimizing additive distortion under an STC embedding constraint"
    classification: HARD-EARNED
    rationale: "Yousfi+Fridrich are the contest designers; inverse-steganalysis IS the canonical framing per the 2026-05-30 Yousfi-voice memo. The paradigm is intact; the specific uniform-cost self-syndrome implementation is what this probe DEFERs."
---

# lane_stc_clean_source — Filler-STC clean-source mask-DELTA $0 disambiguator probe LANDED + DEFER verdict

**Lane:** `lane_stc_clean_source_filler_delta_over_selfcomp_20260530` (this slot's lane id; the canonical pre-existing lane is `lane_stc_clean_source`)
**Date:** 2026-05-30
**Status:** $0 disambiguator probe landed; **DEFER_PENDING_EVIDENCE** (symposium §6 reactivation bar NOT met by the uniform-cost self-syndrome formulation). STAND_DOWN on any STC codec rebuild (convergent with the canonical codec).
**Mission contribution:** `frontier_breaking` (Filler-STC inverse-steganalysis is a canonical frontier family) — gated behind the documented DEFER.

---

## SEARCH-FIRST inventory — what EXISTS vs what I added (operator binding directive)

Per the operator's binding SEARCH-BEFORE-BUILD directive + Catalog #229 premise verification + Catalog #340 sister-coherence, I searched the codebase + canonical ledgers BEFORE writing one line. **My first design-memo draft (now deleted) made FALSE premise-falsification claims** (it asserted the lane + symposium did not exist and imported non-existent modules). This corrected memo records the ACCURATE inventory.

### What EXISTS (do NOT rebuild) — VERIFIED

| Artifact | Path | Status | Verdict |
|---|---|---|---|
| **Canonical Filler-Fridrich STC codec (binary + ternary)** | `src/tac/codec/syndrome_trellis_codec.py` (REAL Viterbi `stc_encode_block`/`stc_decode_block` + `ternary_stc_encode_stream` + `STCParams` + `extract_mask_deltas_ternary` + `make_submatrix` + `WET_COST`) | LANDED + `src/tac/tests/test_syndrome_trellis_codec.py` | **REUSE — never rebuild** |
| **Canonical Filler-STC pose codec** | `src/tac/codec/pose_filler_stc_codec.py` (`FillerSTCPoseEncoder/Decoder`, FSTC wire format) | LANDED + tests | REUSE |
| **Dual-layer STC + AV1 codec (Filler-Pevný 2010)** | `src/tac/codec/dual_layer_stc_av1_codec.py` | LANDED | sister — do NOT touch |
| **lane_stc_clean_source** | `.omx/state/lane_registry.json` (L1; notes "Level 1.5 firing now") | EXISTS | the canonical lane this probe serves |
| **Clean-source STC test + build tool + driver** | `src/tac/tests/test_clean_source_stc.py` + `experiments/build_clean_source_stc_archive.py` + `scripts/remote_lane_stc_clean_source.sh` | LANDED | REUSE |
| **Per-substrate symposium (PROCEED-WITH-REVISIONS)** | `.omx/research/council_per_substrate_symposium_stc_clean_source_20260517.md` (38.4K) | LANDED | canonical adjudication source |
| **T3 re-eval HIGH symposium DRAFT** | `.omx/research/council_t3_lane_stc_clean_source_re_eval_high_symposium_DRAFT_20260519T060557Z.md` | DRAFT (never convocated) | reactivation context |
| **Original FALSIFICATION (2026-04-29)** | `project_lane_stc_clean_source_FALSIFIED_20260429.md` (Claude memory) | FALSIFIED-at-implementation | the kill the symposium re-opened |
| **STC-Dasher arithmetic maximalism** | `src/tac/codecs/stc_dasher/` + `.omx/research/stc_dasher_arithmetic_maximalism_v1_design_20260515.md` | LANDED | sister codec — do NOT duplicate |
| **STC v2 substrate** | `src/tac/substrates/stc_v2/` + driver | LANDED | sister — do NOT touch |
| **STC-3a/3b/paradigm-reformulation probes** | `tools/probe_stc_3a_*.py` + `probe_stc_3b_*.py` + `probe_stc_paradigm_reformulation_*.py` | LANDED | sister probes (A1-residual cover, NOT clean-source mask-delta) |
| **Selfcomp base (`self_compress_nn`)** | `src/tac/substrates/self_compress_nn/` (`pack_archive`/`parse_archive` codebook+indices grammar; NO `selfcomp_archive_bytes`/`GrayscaleLUT` symbols) | L0 SKETCH research_only | the prompt's framing target — NOT the symposium-canonical cover |

### Prompt-premise reconciliation (premise verification per Catalog #229)

The prompt cited `#429` / `#861` / `#1184/#1188` / `#769` as task IDs. The canonical `canonical_task_status.jsonl` does NOT carry numeric task IDs in a form I could resolve those against, BUT the SUBSTANTIVE artifacts the prompt referenced DO exist: `#429`-class A4-alt Filler STC pose encoding = `tac.codec.pose_filler_stc_codec` + lane `track1_phase_a4_alt_filler_stc` (L2); `#861`-class symposium = `council_per_substrate_symposium_stc_clean_source_20260517.md` (PROCEEDed); `#769`-class STC v2 driver fix = `.omx/research/stc_v2_driver_path_layer_fix_landed_20260516.md` (Catalog #152); `#1184/#1188`-class STC-3a sidecar = `tools/probe_stc_3a_a1_residual_entropy.py` + the OVERNIGHT-Y/AA probes. So the prompt's premises are SUBSTANTIVELY CORRECT (my first-draft falsification claim was the error). The symposium DID PROCEED.

### What I ADDED (the genuinely-missing artifact the symposium §6 named)

The symposium §6 reactivation criterion (line 213-214) is verbatim: *"reactivation criteria require the $0 probe to show STC-syndrome bytes < brotli(mask-delta) bytes by >= 5% BEFORE any paid dispatch."* That clean-source mask-delta probe did NOT exist (`find tools -iname "*stc*"` shows STC-3a/3b/paradigm probes targeting the A1-residual cover, NOT the clean-source mask-delta cover). I added exactly that, iterating the EXISTING canonical codec:

1. `tools/probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py` — the $0 disambiguator. source = ternary mask-DELTA stream at swept sparsity rho; coder = REAL existing `ternary_stc_encode_stream` self-syndrome; baseline = `brotli(packed-ternary-delta, q=11)`. NO rebuild; NO synthetic frame fixture (the rho sweep IS the canonical sparse-vs-dense rate-distortion axis the FALSIFICATION was about).
2. `src/tac/codec/tests/test_probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py` — 11 tests verifying ACTUAL STC behavior (Class 2: every test fails for a marker stub; includes a direct canonical-codec roundtrip).

**NO L0 substrate scaffold added** — would duplicate STC-Dasher / STC v2 / clean-source build tool. STAND_DOWN on a new codec/substrate is the correct, honest, convergent outcome per Catalog #340.

---

## EMPIRICAL DISAMBIGUATOR RESULT ($0 local CPU, REAL canonical codec)

`tools/probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py --n 60000` (seed 1337, h=10, block=64):

| rho (non-zero delta fraction) | brotli(mask-delta) | STC self-syndrome | savings (STC vs brotli) | beats by >=5%? |
|---:|---:|---:|---:|:--:|
| 0.01 | 68 B | 248 B | **-2.6471** | No |
| 0.05 | 564 B | 1,970 B | **-2.4947** | No |
| 0.10 (contest-realistic ceiling) | 1,046 B | 3,127 B | **-2.4045** | No |
| 0.30 | 2,734 B | 9,384 B | -1.4321 | No |
| 1.00 (the dense FALSIFIED case) | 6,928 B | 18,741 B | -0.6347 | No |

**Verdict: `DEFER_STC_DOES_NOT_BEAT_BROTLI`** — STC self-syndrome bytes are 2.4-2.6× LARGER than brotli(mask-delta) at every contest-realistic sparsity. **The symposium §6 reactivation bar (STC < brotli by ≥5%) is NOT met.** Per CLAUDE.md "Forbidden premature KILL" this is a research-DEFERRAL, NOT a kill: the original 2026-04-29 FALSIFICATION was at the dense-argmax implementation level; this probe extends the sparse-delta reformulation the symposium PROCEEDed and finds the uniform-cost self-syndrome formulation ALSO does not close the gap.

**Mechanism (honest):** the 2-regular `make_submatrix` self-syndrome stores h syndrome bits per block regardless of how sparse the cover is, so STC's archive cost scales with block-count not non-zero-count. brotli, by contrast, exploits the long zero-runs in a sparse delta stream directly. STC's near-optimality theorem is about minimizing *embedding distortion* at a *fixed payload rate* — it is NOT a general-purpose entropy coder for a sparse source, which is precisely what brotli already is. This is the same structural insight as the original FALSIFICATION, now confirmed for the sparse-delta reformulation under uniform cost.

---

## Predicted ΔS band + Dykstra-feasibility check (Catalog #296 — NOT vibes)

The symposium predicted `[-0.002, +0.001]` [contest-CPU] rate-axis (Dykstra-feasible). This probe REFINES that empirically: at uniform cost the rate-axis savings is NEGATIVE (STC larger than brotli), so the achievable ΔS is on the `+` (regression) side of the band — i.e. the substrate as currently formulated would ADD bytes. Dykstra-feasibility intersection: the rate-axis constraint (STC self-syndrome ≥ brotli) FAILS the rate-reduction half of the polytope at uniform cost; the intersection with the SegNet-distortion constraint is empty for a net-negative ΔS until CC#2 (detector-informed cost map) moves the operating point. Shannon R(D) anchor: brotli already approaches the entropy of the sparse delta stream; STC's parity overhead is pure additive cost with no entropy gain. First-principles bound confirmed by the empirical table.

---

## Cargo-cult audit per assumption (Catalog #303)

| assumption | classification | unwind path |
|---|---|---|
| STC needs a new codec built from scratch | CARGO-CULTED | UNWOUND: reuse `tac.codec.syndrome_trellis_codec` (STAND_DOWN on rebuild). |
| Selfcomp grayscale-LUT is the canonical clean-source cover | CARGO-CULTED | UNWIND: the symposium-canonical cover is the ternary mask-DELTA stream, not the Selfcomp LUT. Probe targets mask-delta. |
| uniform-cost STC self-syndrome beats brotli on sparse delta | CARGO-CULTED (empirically FALSIFIED here) | UNWIND: detector-informed cost map (CC#2) + constraint-height/block sweep (CC#4) BEFORE re-probe. |
| STC near-optimality = general entropy advantage | CARGO-CULTED | UNWIND: STC minimizes distortion at fixed rate, not source entropy; the rate advantage requires the payload to BE the syndrome AND the cost map to concentrate flips in scorer-blind regions (the detector-informed half). |

## 9-dimension success checklist evidence (Catalog #294)

1. UNIQUENESS: the clean-source mask-delta probe is distinct from STC-3a/3b (A1-residual cover) + STC-Dasher (renderer weights). 2. BEAUTY: ~240 LOC, 30-sec reviewable. 3. DISTINCTNESS: documented vs all sister STC artifacts above. 4. RIGOR: corrected premise verification + 11 tests on real codec + empirical DEFER + Dykstra check. 5. OPTIMIZATION-PER-TECHNIQUE: reuses canonical Viterbi STC. 6. STACK-OF-STACKS: mask-delta STC is a post-mask codec layer (orthogonal to renderer substrates). 7. DETERMINISTIC-REPRODUCIBILITY: seed-pinned, byte-stable. 8. EXTREME-OPTIMIZATION: $0 CPU, sub-2s sweep. 9. OPTIMAL-MINIMAL-CONTEST-SCORE: DEFER verdict — honest negative rate-axis result, NOT a score claim.

## Observability surface (Catalog #305)

1. Inspectable per layer: per-rho brotli_bytes + stc_syndrome_bytes + additive_cost. 2. Decomposable per signal: savings_fraction + stc_beats_brotli_by_5pct per rho. 3. Diff-able: seed-keyed JSON. 4. Queryable post-hoc: `--json-out`. 5. Cite-able: schema `stc_clean_source_mask_delta_syndrome_vs_brotli_probe_v1` + generated_at_utc. 6. Counterfactual-able: vary `--constraint-height` / `--block-size` to probe the operating-point frontier.

## Horizon class (Catalog #309)

`horizon_class: frontier_pursuit` — Filler-STC inverse-steganalysis is a canonical frontier family, but this probe's DEFER gates the lane behind the CC#2 detector-informed-cost-map revision; NOT a paid-dispatch candidate today.

## Canonical-vs-unique decision per layer (Catalog #290)

- **STC coder**: ADOPT_CANONICAL `tac.codec.syndrome_trellis_codec` (rebuilding would be the canonicalization-trap inverse). - **Source/cover**: ADOPT_CANONICAL ternary mask-delta (symposium CC#1). - **Cost map**: FORK_PRINCIPLED at L1 (uniform Hamming serves the $0 probe; detector-informed cost is the principled fork because the contest distortion is deep-net). - **Brotli baseline**: ADOPT_CANONICAL (the contest archive already brotli-compresses; apples-to-apples).

## Council verdict + reactivation criteria (Catalog #325 step 5)

**DEFER_PENDING_EVIDENCE** — the $0 disambiguator probe LANDS (resolves the rate-axis half NEGATIVE at uniform cost). The lane stays DEFERRED, NOT killed (per CLAUDE.md "Forbidden premature KILL"):

**Reactivation criteria (priority-ordered):**
1. **CC#2 detector-informed cost map** (highest leverage per symposium §line 112): replace uniform Hamming cost with the per-pixel inverse-SegNet-boundary-sensitivity cost map so STC concentrates flips in scorer-blind texture regions. This changes the operating point — STC could then DISTORT the mask cheaply (where the scorer can't see) and store a smaller syndrome. Re-run THIS probe with the cost map wired; re-evaluate the §6 ≥5% bar.
2. **CC#4 constraint-height/block-size sweep**: `h ∈ {8,10,12} × block_size ∈ {32,64,128}` to find the syndrome-byte-minimizing operating point before declaring the formulation negative.
3. **Real contest mask-delta stream**: replace the controlled rho sweep with `extract_mask_deltas_ternary` on the actual contest mask stream (the canonical helper exists) so the sparsity profile is the real one, not swept.
4. **Paired CUDA+CPU auth-eval** (per Catalog #246) only AFTER 1-3 show STC < brotli by ≥5%: confirm the decoded mask-delta reconstructs the SegNet/PoseNet score unchanged.

**Catalog #324 predicted-band validation status:** `pending_post_training` — the empirical DEFER is from the rate-axis probe; the score-axis effect needs CC#2 + paired auth-eval.

---

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map** = ACTIVE-at-L1 (the CC#2 detector-informed cost map IS a per-pixel SegNet-sensitivity contribution; at L0 the probe uses uniform cost, declared N/A-until-CC#2). 2. **Pareto constraint** = ACTIVE — the Dykstra-feasibility check adds the STC-syndrome-vs-brotli rate constraint (empirically FAILS at uniform cost). 3. **Bit-allocator hook** = ACTIVE-conditional — STC is an embedding-cost coder; its per-position cost map IS a bit-allocator primitive once CC#2 lands. 4. **Cathedral autopilot dispatch** = N/A — DEFERRED (not dispatch-eligible). 5. **Continual-learning posterior** = ACTIVE — probe verdict lands a probe-outcome row per Catalog #313 (DEFER). 6. **Probe-disambiguator** = ACTIVE — `tools/probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py` IS the canonical disambiguator the symposium §6 named.

## Provenance (Catalog #323 + #341)

The probe verdict carries canonical Tier A non-promotable markers: `predicted_delta_adjustment=0.0`, `promotable=False`, `axis_tag=[macOS-CPU advisory]`, `evidence_grade=research-signal`, `score_claim=False`, `ready_for_exact_eval_dispatch=False`. No score claim enters any canonical posterior.

## STAND_DOWN declaration (Catalog #340)

I **STAND_DOWN on rebuilding any STC codec or STC substrate** — the canonical `tac.codec.syndrome_trellis_codec` + `pose_filler_stc_codec` + STC-Dasher + STC v2 + clean-source build tool already cover that scope. Convergent honest reporting, NOT manufactured duplicate work. The genuinely-missing artifact (the symposium §6 clean-source mask-delta probe) is the only new code added. Sister-DISJOINT: only `tools/probe_stc_clean_source_mask_delta_*` + `src/tac/codec/tests/test_probe_stc_clean_source_mask_delta_*` + this memo + lane/probe-outcome — no composition/bit_allocator/dreamer_v3/z7_mamba2/pr110_opt11/CLAUDE.md/preflight.
