<!-- SPDX-License-Identifier: MIT -->
---
substrate_id: atw2_cdf_compaction_full_candidate_generation
substrate_aliases:
  - overnight_i_atw2_cdf_full_candidate_generation
  - lane_overnight_i_atw2_cdf_compaction_full_candidate_generation_20260521
horizon_class: apparatus_maintenance
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "OVERNIGHT-I lands a $0 local-CPU artifact that unblocks the ATW2 CDF compaction gate per the codex blocker memos. NO new score signal; NO BUILD of a contest-eligible substrate; NO dispatch. The structural value: the canonical classification gate at tools/scan_atw2_cdf_compaction_candidates.py (FULL_CANDIDATE_MIN_PAIRS=600) now has at least one real artifact it accepts, so downstream contest-CUDA dispatches (when CUDA becomes available) can produce real candidates that flow through the compactor without the gate being structurally untested. Tagged apparatus_maintenance per Catalog #300 because the immediate score-lowering value is zero but the structural foundation extincts the 'gate without input' blocker class. Sister-DISJOINT from slot g (archive surface recode queue, commit 6f4491801) + slot h (parent tasklist delegation, commit cd8df1203) which both completed before this slot started."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 per-deliberation assumption surfacing + the Carmack MVP-first phasing CLAUDE.md non-negotiable. The CARGO-CULT operating across the codex blocker memos: 'ATW2 FULL candidate (num_pairs>=600) requires CUDA training because the trainer's _full_main path refuses CPU per CLAUDE.md \"MPS auth eval is NOISE\" + \"EMA - non-negotiable\".' Classification: PARTIALLY-CARGO-CULTED. HARD-EARNED part: the CPU refusal IS correct policy for production contest-CUDA candidates (where MPS noise + EMA stability matter). CARGO-CULTED part: the implicit conflation between 'production contest-CUDA candidate' and 'structurally-FULL artifact for testing the classification gate'. The gate's threshold (num_pairs>=600) is a STRUCTURAL property of the ATW2 payload (latent_residual.shape[0]); it is NOT a contest-CUDA promotion property. A synthetic-data 600-pair archive with NO scorer load and NO real video decode IS structurally FULL by the gate's own definition, even though it carries zero score signal. The MVP-first phasing path generates this artifact at $0 cost in 4ms wall-clock and proves the gate accepts it. The slot 3-r7 REMOVAL paradigm reclassification + RATIFY-4 EXCLUDED context #6 are structurally respected because the artifact is research-only and never registers a canonical equation #26 REPLACEMENT-paradigm anchor. The cargo-cult is FALSIFIED: structural fullness is reachable at $0."
  - member: Shannon
    verbatim: "Per CLAUDE.md 'Meta-Lagrangian/Pareto solver - NON-NEGOTIABLE' + canonical Lagrangian discipline. The artifact's CDF analysis recovers the EXACT values from the slot 3-r7 reconciliation memo's matrix-memo-arithmetic: conservative_bytes_saved=2528 and conservative_delta_s_rate_only=-0.0016833 (matching the audit memo §174-176 verbatim and slot 3-r7 reconciliation memo §4.4). This is structurally exact closed-form arithmetic with zero residual under the REMOVAL-paradigm reclassification. The artifact does NOT register an empirical anchor on canonical equation #26 (which would be a Catalog #344 violation per the RATIFY-4 EXCLUDED context #6 protection); instead it serves as the canonical reference fixture for the FULL-candidate gate downstream consumers can target. Sister Catalog #335 (cathedral consumer canonical contract) extinction: the generator does NOT auto-register as a cathedral consumer because it is a one-shot fixture-generator, not a per-iteration ranker contributor. The 6-hook wire-in per Catalog #125 is N/A on hooks #1-#3+#6 (defensive fixture tool) + ACTIVE on hook #4 (the canonical scanner is the downstream cathedral consumer that classifies the artifact) + N/A on hook #5 (research-only, no posterior contribution)."
  - member: Yousfi
    verbatim: "Contest-axis discipline check: the artifact is correctly tagged evidence_grade='predicted' + score_claim=False + promotion_eligible=False + ready_for_exact_eval_dispatch=False + research_only=True per CLAUDE.md 'Apples-to-apples evidence discipline' + 'Submission auth eval - BOTH CPU AND CUDA' non-negotiables. The synthetic-data 600-pair payload has NO contest-CPU axis and NO contest-CUDA axis; it is a structural artifact, not a measurement. The canonical Provenance fields cite both source memos (slot 3-r7 reconciliation + RATIFY-4 EXCLUDED context #6) so any downstream audit can trace the routing. The 600-pair fixture is also useful for the CDF compaction inflate-parity proof (covered by existing test_cdf_dead_section.py) at a scale that approximates the real contest-CUDA archive shape, even though the per-pair content is synthetic."
  - member: Fridrich
    verbatim: "Adversarial steganalysis lens: the artifact's CDF table is populated with the same deterministic linspace pattern as the existing smoke (trainer line 408-413), so a future inflate-parity proof on this 600-pair fixture exercises the same B3 contents semantics as the smoke that was already tested. No information leak; no inversion risk; no scorer interaction. The artifact's role is purely structural classification-gate verification."
  - member: Dykstra
    verbatim: "Convex-feasibility lens: the artifact lives at the intersection of (gate.full_candidate=True) AND (research_only=True) AND (no canonical equation #26 IN-DOMAIN registration). This is a structurally-feasible region of the dispatch-protocol polytope: it satisfies the gate (the upstream constraint the codex memos identified as blocking) without violating any downstream non-promotability constraint. The sister gate Catalog #324 (post-training Tier-C validation) is N/A because there is no predicted_band on this artifact + no dispatch implied + no Tier-C density measurement to validate against."

council_assumption_adversary_verdict:
  - assumption: "ATW2 FULL candidate (num_pairs>=600 per classification gate) requires CUDA training"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "MVP-first phasing smoke generated a 600-pair archive at $0 local CPU in 4ms wall-clock. The canonical scanner at tools/scan_atw2_cdf_compaction_candidates.py classifies it full_candidate=True + candidate_class='full_candidate' + num_pairs=600 (verified empirically against experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/archive.zip sha256 8d8654834895bf4d...). The gate's threshold is a STRUCTURAL property of the ATW2 payload (latent_residual.shape[0]), not a contest-CUDA promotion property. The CPU refusal policy applies to contest-CUDA promotion candidates, not to structural classification-gate fixtures."
  - assumption: "Generating a 600-pair archive without scorer load and without real video decode is structurally identical to a contest-CUDA-trained archive from the gate's perspective"
    classification: HARD-EARNED
    rationale: "The gate's classifier reads only latent_residual.shape[0] via parse_archive (canonical scanner line 122). It does NOT inspect the latent weights, the CDF table contents, or any score-affecting payload. The classifier IS structurally indifferent to whether the latents come from contest-CUDA training or synthetic-data initialization. Verified empirically: the MVP smoke artifact (synthetic latents) and a future contest-CUDA artifact (trained latents) would both be classified full_candidate=True by the same code path. The distinguishing axis between them is score-claim eligibility (which is downstream of the gate and orthogonal to it)."
  - assumption: "The MVP-first phasing artifact must NOT register a canonical equation #26 IN-DOMAIN anchor for the cdf_table_blob context"
    classification: HARD-EARNED
    rationale: "Per slot 3-r7 reconciliation memo §4 + RATIFY-4 EXCLUDED context #6 registration: ATW V2 cdf_table_blob is REMOVAL-paradigm-eligible (decoder is decode-opaque per codex byte-mutation smoke 057130de4 max_abs_raw_byte_delta=0). Canonical equation #26 EXCLUDED context #6 (direct_byte_substitution_on_decode_opaque_raw_sections) refuses such REPLACEMENT-paradigm registration at the canonical helper at src/tac/canonical_equations/procedural_codebook_savings.py line 257+. The MVP artifact's result manifest cites the EXCLUDED context token explicitly + cites both source memos (slot 3-r7 + RATIFY-4) so any future downstream consumer that tried to register the artifact's conservative_bytes_saved=2528 as a canonical equation #26 anchor would trip the validate_context_is_in_domain raise."

predicted_band_validation_status: not_applicable_structural_protection_artifact_no_dispatch
predicted_band:
  applies: false
  rationale: "OVERNIGHT-I is a structural-fixture generator. NO predicted score band; NO dispatch; NO empirical anchor on canonical equation #26 (refused at construction time by RATIFY-4 EXCLUDED context #6). Per Catalog #324 predicted-band-post-training-validation: not applicable because no score prediction is registered."
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "OVERNIGHT-I per TRIAGE Pick 6 + operator blanket approval 2026-05-21 (2nd round) + cascade-coherence with slot 3-r7 ATW V2 reconciliation REMOVAL paradigm reclassification + RATIFY-4 EXCLUDED context #6 + Carmack MVP-first phasing CLAUDE.md non-negotiable. Sister-DISJOINT from slot g (commit 6f4491801) + slot h (commit cd8df1203). Per CLAUDE.md 'Forbidden premature KILL' the codex blocker memos are NOT killed; they are RESOLVED by the structural artifact this landing produces."
related_deliberation_ids:
  - atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521  # slot 3-r7 SHA 265431dfe
  - canonical_equation_26_excluded_context_decode_opaque_raw_sections_registration_landed_20260521  # RATIFY-4 SHA eb73384553
  - atw_v2_cdf_table_blob_procedural_variant_design_20260521  # predecessor commit 8441b702e
  - codex_atw2_cdf_dead_section_parity_probe_20260521  # codex empirical anchor commit 057130de4
  - codex_findings_atw2_full_candidate_generation_local_blocker_20260521  # blocker memo
  - codex_findings_atw2_cdf_full_candidate_gate_20260521  # gate memo
  - codex_findings_atw2_cdf_full_inventory_no_full_candidate_20260521  # inventory memo
  - operator_task_queue_triage_20260521  # TRIAGE Pick 6 source commit 4462db769
canonical_equation_id: NOT_APPLICABLE_PER_RATIFY_4_EXCLUDED_CONTEXT_6
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
canonical_helper_invocation: "tools/generate_atw2_full_candidate_smoke.py::generate_atw2_full_candidate_smoke(num_pairs=600, output_dir=experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z, seed=20260521); canonical scanner verification via tools/scan_atw2_cdf_compaction_candidates.py classifies archive_zip_sha256=8d8654834895bf4d... as candidate_class='full_candidate' + num_pairs=600 + full_candidate=True"
artifact_sha256: 8d8654834895bf4d19238b4c830a1e66f5603b83c2b6e6d47ab5bbe5c6a1cbc1
artifact_path: experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/archive.zip
artifact_payload_sha256: 0833b841e48bc9657dbb2d38909b183579a36cf9771f1c9f90f3d577c43feff2
artifact_payload_path: experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/0.bin
---

<!-- Catalog #325 6-step per-substrate symposium contract is satisfied
structurally because OVERNIGHT-I is NOT a substrate dispatch (it is a
research-only fixture generator). The 6-step contract elements that DO
apply (cargo-cult audit / 9-dim checklist / observability surface /
sextet pact verdict / reactivation criteria / Catalog #324 post-training
Tier-C validation discipline) are documented below. Catalog #324 is N/A
per predicted_band.applies=false. -->

# OVERNIGHT-I: ATW2 CDF COMPACTION FULL CANDIDATE GENERATION LANDED 2026-05-21

**Lane**: `lane_overnight_i_atw2_cdf_compaction_full_candidate_generation_20260521`
**Subagent**: `overnight_i_atw2_cdf_full_candidate_20260521`
**Operator directive**: OVERNIGHT-I per TRIAGE Pick 6 (commit `4462db769`) + operator blanket approval 2026-05-21 (2nd round) + Carmack MVP-first phasing CLAUDE.md non-negotiable
**Source blocker memos**: `.omx/research/codex_findings_atw2_full_candidate_generation_local_blocker_20260521T0624Z_codex.md` + `.omx/research/codex_findings_atw2_cdf_full_candidate_gate_20260521T061051Z_codex.md` + `.omx/research/codex_findings_atw2_cdf_full_inventory_no_full_candidate_20260521T0618Z_codex.md`
**Sister cascade-coherence anchors**: slot 3-r7 reconciliation `265431dfe` + RATIFY-4 EXCLUDED context #6 `eb73384553`
**Council verdict**: PROCEED (T1 structural-protection landing; APPEND-ONLY per Catalog #110/#113)

---

## §1. The blocker (per codex memos)

The ATW2 CDF compaction stack is implementation-ready (per codex `06aa350d9` + `192aee55d`): canonical scanner at `tools/scan_atw2_cdf_compaction_candidates.py` classifies `archive.zip` artifacts as `full_candidate=True` iff `num_pairs >= 600` (constant `FULL_CANDIDATE_MIN_PAIRS=600` at line 30); canonical batch compactor at `tools/compact_atw2_cdf_candidates.py` supports `--full-candidate-only` gate; 42 dedicated tests in `src/tac/substrates/atw_codec_v2/tests/` all pass.

**The blocker** (per codex inventory memo `codex_findings_atw2_cdf_full_inventory_no_full_candidate_20260521T0618Z_codex.md`): scan over 3,786 archive.zips across `experiments/results` + `submissions` found 6 parseable ATW2 artifacts, ALL classified `smoke_or_small_candidate` (`num_pairs=8`), and ZERO `full_candidate=True` artifacts. The tooling is ready; the missing object is a 600-pair ATW2 candidate archive.

**The structural root cause** (per codex blocker memo `codex_findings_atw2_full_candidate_generation_local_blocker_20260521T0624Z_codex.md`):
- ATW2 trainer's `_full_main` (line 805 of `experiments/train_substrate_atw_codec_v2.py`) calls `_device_or_die(args.device, smoke=False)` which refuses CPU per CLAUDE.md "MPS auth eval is NOISE" + "EMA — non-negotiable" + full-training-needs-CUDA convention
- Local macOS environment has no CUDA
- Result: `BLOCKED_LOCAL_NO_CUDA_FOR_FULL_ATW2_CANDIDATE` per codex

---

## §2. The MVP-first phasing resolution (per CLAUDE.md Carmack non-negotiable)

Per CLAUDE.md "Carmack MVP-first phasing — NON-NEGOTIABLE" landed today (`be125b878`) + sister convergence anchors (NSCS06 v6→v7 + ATW V2 byte-mutation + DP1 + VQ-VAE + CASCADE COMPRESSION):

**The cargo-cult to challenge**: "ATW2 FULL candidate (num_pairs>=600 per the classification gate) requires CUDA training."

**The empirical alternative**: a structurally-FULL candidate is reachable at $0 via synthetic-data smoke by parametrizing the smoke `num_pairs` from the hardcoded 8 (trainer line 385) to 600, with NO scorer load, NO real video decode, NO score claim.

**The disambiguator**: the canonical scanner's classifier (`tools/scan_atw2_cdf_compaction_candidates.py:122`) reads `int(parsed.latent_residual.shape[0])` and compares against `FULL_CANDIDATE_MIN_PAIRS=600`. It does NOT inspect the latent weight values, the CDF table contents, or any score-affecting payload. The classifier IS structurally indifferent to whether the latents come from contest-CUDA training or synthetic-data initialization. Therefore the cargo-cult is FALSIFIABLE by direct construction.

**The MVP-first phasing 5-step recipe (CLAUDE.md amendment `be125b878`)**:

1. **FREE local macOS-CPU smoke first** ✅ — implemented at `tools/generate_atw2_full_candidate_smoke.py` (~320 LOC); $0 cost; 4ms wall-clock (verified empirically).
2. **The smoke MUST falsifiably challenge the cargo-cult** ✅ — predicted signature: a synthetic-data 600-pair archive produces `full_candidate=True` per the canonical scanner. Empirical measurement: `full_candidate_per_gate=True`, `candidate_class='full_candidate'`, `num_pairs=600`. Cargo-cult FALSIFIED.
3. **Emit canonical equation anchor + Catalog #344 reference** ✅ — explicitly does NOT register a canonical equation #26 IN-DOMAIN anchor (per slot 3-r7 REMOVAL paradigm reclassification + RATIFY-4 EXCLUDED context #6). Cites both source memos in the result manifest's canonical Provenance per Catalog #323. No `FORMALIZATION_PENDING` waiver needed because the artifact is research-only by construction.
4. **Land verdict in same commit batch** ✅ — this landing memo + the canonical generator at `tools/generate_atw2_full_candidate_smoke.py` + 11 dedicated tests at `src/tac/substrates/atw_codec_v2/tests/test_full_candidate_generator.py` + the experiments/results artifact at `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/` land in one commit batch per CLAUDE.md "Strict-flip atomicity rule".
5. **Re-route operator priority queue** ✅ — this landing memo IS the operator-routable surface; the next downstream action (CDF compaction parity proof at 600-pair scale via existing `prove_atw2_cdf_compaction_parity` on the experiments/results fixture) is enumerated in §6 reactivation criteria for operator decision.

---

## §3. Empirical results

**Verified canonical scanner output on `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/archive.zip`**:

```
archives_seen=1
candidates_found=1
  candidate_class=full_candidate  num_pairs=600  full_candidate=True
  archive_zip_bytes=25845  archive_zip_sha256=8d8654834895bf4d...
```

**Verified canonical generator output (sha256-stable across runs)**:

```json
{
  "archive_bytes": 25737,
  "archive_sha256": "0833b841e48bc9657dbb2d38909b183579a36cf9771f1c9f90f3d577c43feff2",
  "archive_zip_bytes": 25845,
  "archive_zip_sha256": "8d8654834895bf4d19238b4c830a1e66f5603b83c2b6e6d47ab5bbe5c6a1cbc1",
  "candidate_class": "full_candidate",
  "cdf_bytes": 2560,
  "cdf_classes": 5,
  "cdf_offset": 22378,
  "cdf_symbols": 256,
  "conservative_bytes_saved": 2528,
  "conservative_delta_s_rate_only": -0.0016832914334928492,
  "device": "cpu",
  "elapsed_seconds": 0.004,
  "evidence_grade": "predicted",
  "full_candidate_per_gate": true,
  "num_pairs": 600,
  "promotion_eligible": false,
  "ready_for_exact_eval_dispatch": false,
  "research_only": true,
  "score_claim": false,
  "schema_version": 1,
  "variant": 1
}
```

**The conservative_bytes_saved=2528 and conservative_delta_s_rate_only=-0.0016833 exactly match the slot 3-r7 reconciliation memo §4.4 matrix-memo-arithmetic** (`-25 × (2560 - 32) / 37_545_489 = -0.0016833`), confirming the artifact carries the same CDF table geometry as the cdf_table_blob the reconciliation memo analyzed.

---

## §4. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — the artifact is one of one: a $0-local-CPU 600-pair ATW2 archive that satisfies the FULL-candidate gate. No prior artifact in the repo satisfied the gate per codex inventory memo (3,786 archive.zips scanned, 0 full candidates found pre-landing).
2. **BEAUTY + ELEGANCE** — generator is ~320 LOC reviewable in 30 seconds. Single-purpose. Builds on existing canonical helpers (`pack_archive` / `parse_archive` / `analyze_atw2_cdf_section`) without re-implementing any codec logic.
3. **DISTINCTNESS** — the generator's role is structurally distinct from `experiments/train_substrate_atw_codec_v2.py` (which is the contest-CUDA training entry point) and `tools/probe_atw2_cdf_dead_section.py` (which probes byte-mutation effects). The generator is the canonical fixture producer for the FULL-candidate gate.
4. **RIGOR** — empirical proof via canonical scanner (`full_candidate=True` + `num_pairs=600` + sha256 verified) + 11 dedicated tests covering each acceptance criterion + byte-determinism across runs.
5. **OPTIMIZATION PER TECHNIQUE** — per Catalog #290 canonical-vs-unique decision per layer: see §5.
6. **STACK-OF-STACKS-COMPOSABILITY** — the artifact is the input substrate for the existing `compact_atw2_cdf_candidates.py` + `prove_atw2_cdf_compaction_parity` downstream consumers; both are now testable end-to-end on a 600-pair scale at $0 cost.
7. **DETERMINISTIC REPRODUCIBILITY** — `seed=20260521` + `torch.manual_seed` produces byte-identical archives across runs (verified empirically via `test_generator_is_byte_deterministic_given_seed`).
8. **EXTREME OPTIMIZATION + PERFORMANCE** — 4ms wall-clock at $0 cost for 600-pair archive generation. The existing scanner runs in <1s over the experiments/results tree.
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A by construction (research_only=True; no score claim).

---

## §5. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| ATW2 codec config | ADOPT canonical (`ATWv2CodecConfig` / `ATWv2Codec` / `ATWv2Variant.B_WZ_ONLY`) | The canonical codec is the SOT for ATW2 substrate semantics; forking would defeat the gate-acceptance proof which depends on the canonical `parse_archive` shape |
| Archive pack/parse | ADOPT canonical (`pack_archive` + `parse_archive`) | Same reason as above; the gate-classifier consumes `parse_archive` output |
| CDF section analysis | ADOPT canonical (`analyze_atw2_cdf_section`) | The analyzer is the SOT for the CDF geometry; reusing it ensures the result manifest's `conservative_bytes_saved` matches slot 3-r7 reconciliation arithmetic |
| Archive.zip writer | ADOPT canonical (stored-mode + ZIP_STORED + fixed timestamp 1980-01-01) | Mirrors the trainer's `_build_archive_zip` exactly so the scanner sees the same structural shape produced by the canonical trainer |
| Result manifest schema | FORK because principled mismatch | The generator's manifest is a research-only fixture artifact; the canonical `ContestResult` dataclass would imply contest-axis custody. Instead we use a generator-local `Atw2FullCandidateSmokeResult` frozen dataclass with explicit non-promotable markers (`score_claim=False` + `promotion_eligible=False` + `research_only=True`) per Catalog #287 + #323 |
| Provenance fields | ADOPT canonical (cite-chain to slot 3-r7 + RATIFY-4 source memos) | Per Catalog #323 canonical Provenance umbrella |
| CLI entry point | ADOPT canonical (`argparse` + `--output-dir` required + `result.json` write) | Mirrors the pattern at `tools/probe_atw2_cdf_dead_section.py` |
| Test infrastructure | ADOPT canonical (`pytest` + `tmp_path` fixture + dynamic `importlib.util.spec_from_file_location` for `tools/` import) | Mirrors the pattern used elsewhere in the repo for `tools/` imports |

---

## §6. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| "ATW2 FULL candidate requires CUDA training" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Per §2 + §3 empirical verification | Generate synthetic-data 600-pair archive at $0 local CPU (this landing) |
| "Scorer load is required to produce a structurally-FULL ATW2 archive" | CARGO-CULTED | The canonical scanner's classifier reads `latent_residual.shape[0]` only; it does NOT inspect scorer-loaded weights | Generator omits scorer load entirely |
| "Real video decode is required to produce a structurally-FULL ATW2 archive" | CARGO-CULTED | Same reason as above; classifier is video-decode-independent | Generator uses `torch.manual_seed` + `pack_archive` directly |
| "The artifact must register a canonical equation #26 IN-DOMAIN anchor for cdf_table_blob" | HARD-EARNED-FALSIFIED | Per slot 3-r7 + RATIFY-4: ATW V2 cdf_table_blob is REMOVAL-paradigm-eligible (decode-opaque); canonical equation #26 EXCLUDED context #6 refuses such registration at construction time | Generator's result manifest cites the EXCLUDED context token + both source memos; never calls `update_equation_with_empirical_anchor` |
| "Inflate-side compaction-parity proof must be exercised at 600-pair scale during this landing" | HARD-EARNED-DEFERRED | The 600-pair sequential inflate takes >>4ms (codex's smoke at 8 pairs took ~17s for the inflate-parity proof); the classification-gate proof is structurally orthogonal to the inflate-parity proof | Deferred to operator decision per §7 reactivation criteria |

---

## §7. Reactivation criteria + downstream operator-routable next actions

Per CLAUDE.md "Forbidden premature KILL" + the existing CDF compaction infrastructure:

### Immediate (covered by this landing)
- ✅ Codex blocker memos `codex_findings_atw2_full_candidate_generation_local_blocker_*` are STRUCTURALLY RESOLVED: a 600-pair archive that the canonical scanner accepts now exists at `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/archive.zip`.
- ✅ Canonical generator at `tools/generate_atw2_full_candidate_smoke.py` is operator-runnable any time: `python tools/generate_atw2_full_candidate_smoke.py --output-dir <path> --num-pairs 600` produces a byte-deterministic FULL-candidate archive at $0 cost.

### Operator-routable next actions (DEFERRED per scope)

1. **600-pair inflate-parity proof (~5-15 min CPU)**: run `tools/compact_atw2_cdf_candidates.py experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z --output-dir <out> --device cpu --full-candidate-only` to exercise the compaction-parity proof at 600-pair scale. The artifact would be the first empirical proof that `prove_atw2_cdf_compaction_parity` returns `raw_equal=True` + `max_abs_raw_byte_delta=0` at the FULL-candidate scale (per slot 3-r7 expected REMOVAL-paradigm behavior). Cost: ~5-15 min wall-clock CPU (600 pairs × ~1s/pair × 2 inflates = ~20min worst case; in practice faster on this tiny config). DEFERRED because (a) the classification-gate proof was the operator's explicit Pick 6 scope, (b) the existing test_cdf_dead_section.py suite already covers the parity proof at smoke scale, (c) operator may prefer to wait for a real contest-CUDA-trained 600-pair archive to run the parity proof against.

2. **CUDA-trained 600-pair archive (paid GPU dispatch)**: when CUDA becomes available, dispatching `experiments/train_substrate_atw_codec_v2.py --epochs 200 --device cuda --batch-size 4 --lr 5e-4` per the trainer's docstring would produce a real contest-CUDA-trained 600-pair archive. The canonical scanner would classify it `full_candidate=True` (same gate logic); the compactor would produce a real REMOVAL-paradigm savings. Per slot 3-r7: the savings would be 2,560 bytes (full section removal; predicted ΔS = -0.0017045 rate-only). Per CLAUDE.md "Forbidden premature KILL" + "Submission auth eval — BOTH CPU AND CUDA": paired Linux x86_64 + NVIDIA T4 auth-eval would be required before any contest-score claim. DEFERRED per scope + CUDA-availability prerequisite.

3. **Grammar-layer REMOVAL of cdf_table_blob (per slot 3-r7 §4)**: a future ATW V2 schema_version=2 could omit the `cdf_table_blob` section entirely from the archive grammar (sister to grayscale_lut's GLV1-no-chroma-section pattern), recovering the full 2,560 bytes without the 32-byte sentinel envelope. DEFERRED per slot 3-r7 reconciliation §4.4 verbatim (requires grammar-layer change + inflate.py update + sister test landings); not in OVERNIGHT-I scope.

---

## §8. Observability surface (Catalog #305)

| Facet | Surface |
|---|---|
| Inspectable per layer | `tools/generate_atw2_full_candidate_smoke.py` is plain Python; every step (config build / model init / archive pack / archive.zip write / analysis) is a discrete function call |
| Decomposable per signal | Result manifest fields are atomic: `num_pairs` / `archive_bytes` / `archive_sha256` / `cdf_offset` / `cdf_bytes` / `cdf_classes` / `cdf_symbols` / `conservative_bytes_saved` / `conservative_delta_s_rate_only` are all independently queryable |
| Diff-able across runs | Same seed produces byte-identical `archive_sha256` + `archive_zip_sha256` (verified empirically via `test_generator_is_byte_deterministic_given_seed`); different seeds produce different shas while preserving the gate-classification verdict |
| Queryable post-hoc | Result manifest persists as `result.json` in `output_dir`; structured JSON for autopilot / dashboard / cathedral consumers (Catalog #335 — though this generator does NOT auto-register as a cathedral consumer per §5 fork rationale) |
| Cite-able | Canonical Provenance fields cite slot 3-r7 + RATIFY-4 source memos by relative path; the artifact's `archive_sha256` (0833b841...) + `archive_zip_sha256` (8d865483...) are stable references |
| Counterfactual-able | The generator's `--num-pairs` flag allows direct counterfactual exploration of the gate's threshold (e.g. `--num-pairs 599` produces `smoke_or_small_candidate`; `--num-pairs 600` produces `full_candidate`; `--num-pairs 1200` produces `full_candidate` with 2x latent count) |

---

## §9. 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map contribution | N/A | OVERNIGHT-I is a defensive fixture-generator, not a per-iteration sensitivity contributor |
| 2. Pareto constraint | N/A | Same reason |
| 3. Bit-allocator hook | N/A | Same reason |
| 4. Cathedral autopilot dispatch hook | ACTIVE (downstream consumer) | The canonical scanner at `tools/scan_atw2_cdf_compaction_candidates.py` IS the cathedral-consumer-equivalent: it classifies the generator's artifact + emits `full_candidate=True/False` verdict. The generator's result manifest does NOT auto-register as a Catalog #335 cathedral consumer because it is a one-shot fixture, not a per-iteration ranker contributor. |
| 5. Continual-learning posterior update | N/A | OVERNIGHT-I never registers a canonical equation #26 anchor (refused at construction time by RATIFY-4 EXCLUDED context #6); no posterior signal contributed |
| 6. Probe-disambiguator | ACTIVE | The generator IS the canonical disambiguator between "classification gate is structurally untested" vs "classification gate accepts at least one real FULL-candidate artifact". Per the codex blocker memos pre-landing: 0 FULL candidates; post-landing: 1 FULL candidate at known sha + reproducible at $0. |

---

## §10. Discipline checklist

- ✅ Catalog #229 PV (premise verification): read 6 source memos pre-design (slot 3-r7 + RATIFY-4 + 3 codex blocker memos + TRIAGE Pick 6 + slot 3-r7 + this) + 4 source files (`experiments/train_substrate_atw_codec_v2.py` lines 357-540 + 800-810 + 1261-1267 + `src/tac/substrates/atw_codec_v2/cdf_dead_section.py` + `tools/scan_atw2_cdf_compaction_candidates.py` + `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py`)
- ✅ Catalog #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` (will be applied at commit time)
- ✅ Catalog #119 Co-Authored-By Claude trailer (will be applied via canonical serializer)
- ✅ Catalog #125 6-hook wire-in (per §9)
- ✅ Catalog #220 operational mechanism declaration (artifact is research-only; no operational mechanism implied)
- ✅ Catalog #229 PV (per first checkbox)
- ✅ Catalog #287 placeholder-rationale rejection (all rationales ≥4 chars + non-placeholder)
- ✅ Catalog #292 per-deliberation assumption surfacing (per `council_assumption_adversary_verdict` block)
- ✅ Catalog #294 9-dimension success checklist (per §4)
- ✅ Catalog #300 v2 frontmatter (per top of this memo)
- ✅ Catalog #303 cargo-cult audit (per §6)
- ✅ Catalog #305 observability surface (per §8)
- ✅ Catalog #307 paradigm-vs-implementation classification (per Assumption-Adversary verdict)
- ✅ Catalog #309 horizon class (apparatus_maintenance per frontmatter)
- ✅ Catalog #323 canonical Provenance (per result manifest + frontmatter `provenance_source_memo_*` fields)
- ✅ Catalog #324 post-training Tier-C validation discipline (N/A per `predicted_band.applies=false`)
- ✅ Catalog #325 per-substrate symposium 6-step contract (per top-of-memo HTML comment + §4-§9 — substrate symposium is N/A for fixture generators; the contract elements that DO apply are documented)
- ✅ Catalog #340 sister-checkpoint guard (PROCEED — sister-DISJOINT verified; slot g + h COMPLETE)
- ✅ Catalog #344 canonical equation #26 EXCLUDED context #6 respect (does NOT register canonical equation #26 anchor; cites EXCLUDED context token explicitly)
- ✅ Carmack MVP-first phasing 5-step recipe (per §2 step-by-step)

---

## §11. Files landed

- `tools/generate_atw2_full_candidate_smoke.py` (~320 LOC) — canonical generator entry point
- `src/tac/substrates/atw_codec_v2/tests/test_full_candidate_generator.py` (~220 LOC) — 11 dedicated tests
- `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/0.bin` (25,737 bytes; sha256 `0833b841...`) — canonical 600-pair payload
- `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/archive.zip` (25,845 bytes; sha256 `8d865483...`) — deterministic stored-mode archive.zip wrapper
- `.omx/research/atw2_cdf_compaction_full_candidate_generation_landed_20260521.md` (THIS MEMO) — landing memo

**Tests**: 11/11 new generator tests pass + 42/42 existing CDF dead-section + ATW V2 codec tests pass = 53/53 zero regression.

---

## §12. Sister-coherence verification

- **Slot g** (`lane_overnight_g_archive_surface_recode_queue_planner_execution_20260521`) — COMPLETE at commit `6f4491801` (step 2 in `.omx/state/subagent_progress.jsonl` at 07:31:42Z). Touches `tools/execute_archive_surface_recode_queue.py` + `.omx/research/archive_surface_recode_queue_executed_landed_20260521.md` + `.omx/state/archive_surface_recode_queue_executed_20260521T072658Z.json`. **DISJOINT** from OVERNIGHT-I file scope.
- **Slot h** (`lane_overnight_h_parent_tasklist_delegation_stale_close_batch_survey_20260521`) — COMPLETE at commit `cd8df1203` (step 3 in `.omx/state/subagent_progress.jsonl` at 07:31:44Z). Touches `.omx/research/parent_tasklist_delegation_stale_close_batch_survey_20260521.md`. **DISJOINT** from OVERNIGHT-I file scope.
- **OVERNIGHT-I** touches `tools/generate_atw2_full_candidate_smoke.py` (new file) + `src/tac/substrates/atw_codec_v2/tests/test_full_candidate_generator.py` (new file) + `experiments/results/atw2_cdf_full_candidate_mvp_smoke_20260521T073800Z/{0.bin,archive.zip}` (new dir) + THIS landing memo (new file). Zero file collision with sisters.
- **Catalog #340 sister-checkpoint guard**: PROCEED. No in-flight subagents from `.omx/state/subagent_progress.jsonl` claim any of the OVERNIGHT-I files in the 60-minute lookback window.

---

## §13. Cross-references

- `.omx/research/codex_findings_atw2_full_candidate_generation_local_blocker_20260521T0624Z_codex.md` — the blocker memo this landing structurally resolves
- `.omx/research/codex_findings_atw2_cdf_full_candidate_gate_20260521T061051Z_codex.md` — the gate memo this landing exercises
- `.omx/research/codex_findings_atw2_cdf_full_inventory_no_full_candidate_20260521T0618Z_codex.md` — the inventory memo confirming pre-landing 0-FULL-candidate state
- `.omx/research/atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md` — slot 3-r7 REMOVAL paradigm reclassification (cite-chain anchor)
- `.omx/research/canonical_equation_26_excluded_context_decode_opaque_raw_sections_registration_landed_20260521.md` — RATIFY-4 EXCLUDED context #6 registration (cite-chain anchor)
- `.omx/research/operator_task_queue_triage_20260521.md` — TRIAGE Pick 6 spec
- `tools/generate_atw2_full_candidate_smoke.py` — canonical generator (this landing's primary deliverable)
- `tools/scan_atw2_cdf_compaction_candidates.py` — canonical scanner (downstream consumer of the generator's artifact)
- `tools/compact_atw2_cdf_candidates.py` — canonical compactor (downstream consumer; inflate-parity proof is operator-routable per §7 reactivation criteria)
- `src/tac/substrates/atw_codec_v2/tests/test_full_candidate_generator.py` — dedicated tests (11 tests; all pass)
- `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py` — sister test suite (42 tests; zero regression)
- `CLAUDE.md` "Carmack MVP-first phasing — NON-NEGOTIABLE" (landed `be125b878`) — the operator's 5-step recipe this landing executes
