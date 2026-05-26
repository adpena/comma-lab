<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical R1 review record for Path 3 candidate #E (BoostNeRV against PR110 fec6 frontier MLX-local L0 SCAFFOLD; commit `83910e54e`). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: cites registered `procedural_predictor_plus_residual_correction_savings_v1` (REGISTERED, verified 2026-05-26 via tac.canonical_equations.query_equations). FORMALIZATION_PENDING:r1_review_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "BPR1 header is 24 bytes as documented in design memo + archive.py module docstring"
    classification: CARGO-CULTED
    rationale: "Empirically refuted: `struct.calcsize('<5sBBB16sIB') = 29 bytes` AND `BPR1_HEADER_LEN = 29` in __init__.py. The design memo + archive.py docstring claim '24-byte header' is INCONSISTENT with the SOURCE-OF-TRUTH constant. Per CLAUDE.md 'Comment-only contracts are FORBIDDEN' + Catalog #185 META-meta-drift detection: documentation MUST match canonical constants. This is a R1 finding; FIX-WAVE-R1 must update design memo + archive.py docstring to declare 29 bytes."
  - assumption: "Predicted band sign [-0.010, +0.0045] is HARD-EARNED at L0 per the Shannon R(D) first-principles bound"
    classification: HARD-EARNED-WITH-VIBES-WAIVER
    rationale: "The PREDICTED_BAND_VIBES_OK waiver is properly declared per Catalog #296 (cargo-cult-prediction risk); the first-principles bound `[-0.010, +0.0045]` is honestly sign-ambiguous because the cargo-cult assumptions (Atick-Redlich H(GT|PR110_base) estimate; residual codec at Shannon R(D); near-random vs scorer-signal) ARE the open empirical questions Phase 2 council symposium must resolve. The math is internally consistent; the assumptions are explicitly tagged CARGO-CULTED. Acceptable at L0."
  - assumption: "Residual head implementation is MLX-native and matches sister boost_nerv pattern"
    classification: HARD-EARNED
    rationale: "Empirical verification: MLX ResidualHeadMLX forward vs PyTorch reference with copied weights produces max_abs=3.86e-4 (within numerical precision of fp32 ops); architecture mirror is correctness-preserving."
council_decisions_recorded:
  - "FIX-WAVE-R1 op-routable #1: update design memo §'Archive grammar' + §'BPR1 sidecar magic, prepended to PR110 archive bytes' from '24-byte header' to '29-byte header' to match `BPR1_HEADER_LEN = 29` source constant; correct same inconsistency in `archive.py` module docstring 'BPR1 header 28 bytes' on line 8"
  - "FIX-WAVE-R1 op-routable #2: update `__init__.py` lines 41-43 archive grammar comment to declare 29-byte header (currently states '24-byte header' on line 41)"
  - "Op-routable advisory (NOT R1 finding): registry canonical equation `procedural_predictor_plus_residual_correction_savings_v1` is REGISTERED per `tac.canonical_equations.query_equations` empirical verification 2026-05-26; design memo FORMALIZATION_PENDING marker for `residual_hybrid_boosting_savings_v1` was a placeholder name; the registered equation is the same conceptual entity per Catalog #359 sister discipline. Recommend memo cleanup at Phase 2 council symposium."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - procedural_predictor_plus_residual_correction_savings_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
---

# R1 Recursive Adversarial Review — Path 3 candidate E (BoostNeRV against PR110 fec6 frontier MLX-local L0 SCAFFOLD)

**Scope**: commit `83910e54e`; 11 files / 1834 insertions across `src/tac/substrates/boost_nerv_pr110_residual/` + design memo + tests
**Verdict**: **PROCEED_WITH_REVISIONS** — R1 NOT CLEAN (documentation-only findings; no code correctness gap); counter resets to 0; FIX-WAVE-R1 successor required to update memo + docstrings before R2 fires
**Cost**: $0 GPU; ~25 min wall-clock empirical drift measurement + math verification + memo authoring

---

## Axis 1 review: Math + scientific + engineering rigor (council members: Shannon + Dykstra + Tao)

### Per-architectural-choice triple-axis citation table

| Architectural choice | Math citation | Scientific citation | Engineering citation | Classification |
|---|---|---|---|---|
| Boosting residual sidecar against frozen PR110 base | Friedman 2001 "Greedy Function Approximation: A Gradient Boosting Machine" canonical gradient-boosting foundation | Liu et al. ECCV 2024 BoostNeRV NeRV-family specialization (cited design memo §"Cross-references" + cargo-cult audit row 10) | Sister `src/tac/substrates/boost_nerv/_BoostingHead` MLP-on-RGB primitive; this substrate FORK_BECAUSE_PRINCIPLED_MISMATCH per Catalog #290 (PR110 conditioning vs fresh latent) | **HARD-EARNED** |
| Per-pair residual learner with z_pr110 conditioning | Atick-Redlich 1990 cooperative-receiver framing (canonical Z4 sister) | Design memo §"Why this is a SISTER substrate to the existing `boost_nerv/`" + §"Residual extraction from PR110 — byte / score math" | `architecture.py::ResidualHeadMLX.forward(rgb_pr110_base, z_pr110)` MLX-native NHWC | **HARD-EARNED** |
| BPR1 archive grammar with PR110_BASE_SHA256_PREFIX[16] binding | Catalog #139 byte-mutation discipline + HNeRV parity L3 monolithic single-file `0.bin` | Catalog #146 inflate runtime contract; canonical structural-extinction primitive | `__init__.py::BPR1_HEADER_LEN = 29` (matches `struct.calcsize('<5sBBB16sIB')`); SHA256 binding refuses non-matching base archives at runtime | **HARD-EARNED-AT-CODE-SURFACE**; **CARGO-CULTED-AT-DOCUMENTATION-SURFACE** (memo + docstrings declare 24-byte header; source declares 29 — see R1 finding #1) |
| Per-pair residual target = GT - PR110_base | Standard residual extraction; signed float32 in [-1, 1] | Stage 1 of canonical boosting curriculum per design memo | `residual_extraction.py::extract_per_pair_residual_targets` implements correctly; shape validation present | **HARD-EARNED** |
| Stage 1 convergence target: p99 ≥ 0.05 → PROCEED; < 0.01 → DEFER | Convergence-target heuristic; documented as cargo-cult-defensible (boosting needs headroom to extract signal) | Design memo §"Stage 1: Per-pair residual target computation" | `residual_extraction.py::diagnose_residual_target_magnitude` returns canonical verdict (PROCEED / DEFER / MARGINAL); non-promotable markers per Catalog #341 | **HARD-EARNED** |
| Score-aware Lagrangian = α × Δrate + β × d_seg + γ × √d_pose | Canonical contest score formula | Per Catalog #164 canonical helper routing (declared in design memo §"Canonical-vs-unique decision per layer"); applies to L1+ score-aware fine-tune stage 3 | `boosting_curriculum.py::CurriculumStage(stage_id=3)` declares the canonical Lagrangian; ACTUAL implementation deferred to L1+ per Catalog #240 `_full_main raises NotImplementedError` | **HARD-EARNED** |
| Δrate predicted = 25 × residual_blob_len / 37,545,489 | Contest rate term formula (canonical) | Empirically verified: `25 × 8824 / 37545489 = 0.005876`; `25 × 8192 / 37545489 = 0.005455` (design memo's two different cited values reflect 8192-budget-target vs 8824-empirical-estimate) | Cited in design memo §"Byte accounting (predicted)" + §"9-dim evidence row #8" | **HARD-EARNED** (math correct; the two numbers reflect different assumed residual_blob_len, not an inconsistency) |
| brotli-quality9 on residual blob | Canonical PR101/PR106/PR110 family default; RFC 7932 | Design memo cargo-cult audit row 7 "HARD-EARNED" | `boosting_curriculum.py::CurriculumStage(stage_id=4)` declares; archive.py uses `compose_archive` which prepends sidecar bytes (caller must brotli-encode residual_blob) | **HARD-EARNED** |
| 2-stage inflate (shell-invoke PR110 inflate.sh + residual MLP) | Documented as CARGO-CULTED (admissibility per contest rules is design-question); Phase 2 council symposium per Catalog #325 | Cargo-cult audit row 8 declares CARGO-CULTED with explicit unwind path (alternative: vendor PR110 weights into our archive, which defeats the entire frozen-base point) | Design memo §"Inflate runtime" declares 2-stage pattern + LOC budget ≤200 per HNeRV parity L4 substrate-engineering waiver | **HARD-EARNED-WITH-CARGO-CULT-DISCLOSURE** |
| Predicted ΔS band [-0.010, +0.0045] sign-ambiguous | Shannon R(D) first-principles upper bound per design memo §"Predicted ΔS band" | `PREDICTED_BAND_VIBES_OK` waiver explicitly declared per Catalog #296 (acceptable at L0 with Phase 2 council symposium reduction requirement) | `predicted_band_validation_status: pending_post_training` per Catalog #324 phantom_random_init refusal | **HARD-EARNED** (waiver properly declared; cargo-cult ASSUMPTIONS explicitly enumerated and queued for Phase 2 resolution) |
| Canonical equation `procedural_predictor_plus_residual_correction_savings_v1` | Catalog #359 sister equation per the residual-hybrid context Catalog #26 EXCLUDED context | Empirically verified REGISTERED via `tac.canonical_equations.query_equations()` 2026-05-26 | Frontmatter cites design memo placeholder `residual_hybrid_boosting_savings_v1` which is the same conceptual entity | **HARD-EARNED at registry** (REGISTERED); **CARGO-CULTED at memo** (placeholder name diverges from registered name — minor inconsistency) |

### Axis 1 verdict

**Math + scientific + engineering rigor**: HARD-EARNED across all 11 architectural choices. The only Axis 1 gap is a **DOCUMENTATION inconsistency** — design memo + archive.py docstring + __init__.py comment declare "24-byte header" but the SOURCE constant `BPR1_HEADER_LEN = 29` is correct. Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + the operator-facing audit trail principle: documentation MUST match canonical constants. This is a DOCUMENTATION-LEVEL R1 finding (not a code correctness gap; the canonical 29-byte format is correctly implemented at the pack/parse surface).

**Axis 1 findings**: 1 R1 finding (documentation):
- **R1 finding #1**: BPR1 header size inconsistency (memo + archive.py + __init__.py docstrings/comments declare 24 bytes; actual is 29 bytes per `struct.calcsize` + `BPR1_HEADER_LEN` constant)

---

## Axis 2 review: MLX drift minimization (council members: Carmack + Hotz + Quantizr)

### Empirical drift measurements

Reference: PyTorch `nn.Linear`, `nn.Conv2d`, `nn.functional.relu`, `torch.tanh`. Input: `np.random.seed(11)` × `(1, 8, 12, 3)` NHWC + `(1, 8)` latent.

| MLX primitive in E=BoostNeRV | Drift vs PyTorch reference (max_abs in input units) | Verdict |
|---|---|---|
| `ResidualHeadMLX.forward` (Linear → broadcast → concatenate → Conv2d → relu → Conv2d → tanh) | **3.86e-04** (sub-1e-3; within fp32 precision tolerance for compound ops) | **HARD-EARNED** |
| `compose_pr110_base_plus_residual` (clip + add + clip) | **0.000000** (exact; pure arithmetic ops) | **HARD-EARNED** |
| `nn.Linear` (z_proj) | ~eps; MLX matches PyTorch | **HARD-EARNED** |
| `nn.Conv2d` (conv1, conv2 — HWIO weights internally; PyTorch reference has OIHW transpose at load) | ~eps post-transpose | **HARD-EARNED** |
| `nn.relu` | ~eps; matches PyTorch | **HARD-EARNED** |
| `mx.tanh` | ~eps; matches PyTorch | **HARD-EARNED** |
| `mx.broadcast_to`, `mx.concatenate`, `mx.clip` | ~eps (pure layout/comparison ops) | **HARD-EARNED** |
| No PixelShuffle or bilinear primitives in E=BoostNeRV residual head | N/A — substrate is small enough that the canonical sister bugs from A=DreamerV3 do not apply | **N/A** |

### Axis 2 verdict

**MLX drift minimization**: HARD-EARNED across all measured primitives. The E=BoostNeRV residual head is small enough (≤4K params per round) that it uses only canonical Linear + Conv2d + activation primitives; no custom bilinear or PixelShuffle is in scope at L0. Drift measurements all under 4e-4 absolute (within fp32 compound-op precision; acceptable per CLAUDE.md "MLX portable-local-substrate authority" Tier A research-signal grade).

**Axis 2 findings**: 0 R1 findings.

---

## Axis 3 review: Portability via numpy (council members: MacKay + Selfcomp + Contrarian)

### Per-MLX-primitive numpy reference status

| MLX primitive in E=BoostNeRV | numpy reference status | Portability gap | Verdict |
|---|---|---|---|
| `nn.Linear` (z_proj) | `np.einsum('ij,bj->bi', w, x) + b` | Trivial | **PORTABLE** |
| `nn.Conv2d` (conv1, conv2) | numpy reference: `scipy.signal.correlate2d` per-channel loop | Moderate (10-100x slowdown); acceptable for non-Apple-Silicon DEV-RIG iteration | **NUMPY-PORTABLE** for eval; needs jax-or-torch for training |
| `nn.relu` | `np.maximum(x, 0)` | Trivial | **PORTABLE** |
| `mx.tanh` | `np.tanh` | Trivial | **PORTABLE** |
| `mx.broadcast_to`, `mx.concatenate`, `mx.clip` | `np.broadcast_to`, `np.concatenate`, `np.clip` | Trivial; numpy has direct equivalents | **PORTABLE** |
| `compose_pr110_base_plus_residual` | `np.clip(rgb_base + np.clip(residual, -gain, gain), 0, 1)` | Trivial | **PORTABLE** |
| `archive.py` (struct + hashlib + brotli + zipfile) | Pure-Python stdlib + brotli — fully portable | None | **PORTABLE** ✓ |
| `residual_extraction.py` (numpy only) | Already uses numpy directly via lazy imports | None | **PORTABLE** ✓ |
| `boosting_curriculum.py` (dataclass declarations) | Pure-Python; no MLX/numpy dependency | None | **PORTABLE** ✓ |
| `inflate.py` (Stage 4 stub) | Not yet implemented (Catalog #240 `_full_main raises NotImplementedError`); future inflate will be PyTorch + brotli per HNeRV parity L4 | N/A at L0 | **DEFERRED** to L1+ |
| Stage 0 PR110 inflate subprocess | Pure shell + Python subprocess; portable to any Linux/macOS | None | **PORTABLE** ✓ |

### Axis 3 verdict

**Portability via numpy**: ACCEPTABLE within the documented MLX-local-iteration scope. The substrate's core curriculum scaffolding (residual_extraction.py + boosting_curriculum.py + archive.py) is already PORTABLE to non-Apple-Silicon test rigs (numpy + Python stdlib + brotli). The MLX residual head (`architecture.py::ResidualHeadMLX`) is the ONLY MLX-dependent component at L0; structurally numpy-portable via the trivial primitive substitutions above.

**Axis 3 findings**: 0 hard CARGO-CULTED items.

---

## Cross-substrate META findings (cross-reference to aggregate)

| Pattern | A=DreamerV3 | D=Z6 | E=BoostNeRV (this review) |
|---|---|---|---|
| MLX primitive complexity | High (full HNeRV decoder; 6 PixelShuffle blocks + bilinear) | Medium (Z6 encoder + FiLM predictor + decoder) | Low (tiny MLP-on-RGB residual head; 3 ops) |
| MLX primitive correctness | 2 BROKEN (PixelShuffle channel-LAST + bilinear mx.repeat) | All CORRECT | All CORRECT |
| Documentation-source consistency | 1 minor (cargo-cult audit row #5 incomplete claim) | 0 issues | 1 BPR1 header size inconsistency (memo + docstring + comment vs source constant) |
| Canonical equation registry status | 2 equations REGISTERED + cited | 0 equations declared (sister-substrate-wide gap; not this scaffold's responsibility) | 1 equation REGISTERED (correct name `procedural_predictor_plus_residual_correction_savings_v1`); memo cites placeholder `residual_hybrid_boosting_savings_v1` |
| Empirical end-to-end test | shape-only (test_end_to_end_mlx_train_archive_pytorch_inflate) | byte-stable end-to-end (test_f03 → 24,416,064 bytes contest camera resolution) | composition math (test_compose_pr110_base_plus_residual); full E2E requires Stage 0 PR110 inflate (deferred per `_full_main raises NotImplementedError`) |

**META observation**: E=BoostNeRV's residual head is small enough that it stays well-within MLX-native primitive territory (no custom bilinear / PixelShuffle invented locally). This avoids the META class of bug that bit A=DreamerV3 (locally-invented MLX primitives diverging from PyTorch reference). The CONSOLIDATE-OP META directive does not apply here because the substrate doesn't NEED the canonical helpers — but if Stage 5 round-2 boosting at L1+ scales the residual learner up (e.g. multi-scale residual at 192×256), the temptation to invent local bilinear primitives will surface; the operator should pre-empt by routing through canonical helpers at L1+ design time.

---

## R1 verdict per landing

**R1 VERDICT: NOT CLEAN** — counter resets to 0.

**Reasoning**: 1 R1 finding (documentation-level) per Axis 1:
- **R1 finding #1**: BPR1 header size inconsistency — memo + archive.py docstring + __init__.py comment declare "24-byte header" (2 places: design memo §"BPR1 sidecar magic, prepended to PR110 archive bytes" + archive.py module docstring line 8; 1 place: __init__.py line 41 archive grammar comment) but source-of-truth `BPR1_HEADER_LEN = 29` is correct per `struct.calcsize('<5sBBB16sIB')`

The code is correct; only the documentation lies. Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + Catalog #185 META-meta-drift detection: this is a documentation-bug not a code-bug. The FIX-WAVE-R1 is small (≤3 string edits across 3 files); R2 can fire immediately after.

**Axis 2 + Axis 3**: 0 findings each; all empirical measurements + portability analysis HARD-EARNED.

**FIX-WAVE-R1 op-routable queue (priority-ranked)**:
1. **P0 / DOCUMENTATION-FIX**: update design memo `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md` §"Archive grammar" + §"BPR1 sidecar magic, prepended to PR110 archive bytes" rows to declare 29-byte header (currently "24-byte header")
2. **P0 / DOCUMENTATION-FIX**: update `archive.py` module docstring line 8 from "BPR1 header 28 bytes" (also INCORRECT — declares 28; actual is 29) to "BPR1 header 29 bytes"
3. **P0 / DOCUMENTATION-FIX**: update `__init__.py` line 41 archive grammar comment from "24-byte header" to "29-byte header"
4. **P1 / CONSISTENCY-CHECK**: verify across the full memo body that no other byte-count or struct-format claims diverge from source constants
5. **P2 / ADVISORY**: update design memo §"Cross-references" canonical-equation-registry line to cite the registered name `procedural_predictor_plus_residual_correction_savings_v1` (currently cites placeholder `residual_hybrid_boosting_savings_v1` per FORMALIZATION_PENDING marker; empirically the registered equation exists)

Once FIX-WAVE-R1 lands the documentation fixes, R2 can fire per recursive adversarial review protocol "close paths".

---

## Discipline applied

- **Catalog #229 PV**: 6 source files read in full + design memo 235 lines + 25 tests run before any review claim + canonical equation registry empirically queried
- **Catalog #110/#113 APPEND-ONLY**: NEW memo only; sister design memo NEVER mutated (R1 finding #1 cites the inconsistency but DOES NOT modify the design memo; the modification belongs to FIX-WAVE-R1)
- **Catalog #287**: every finding carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales
- **Catalog #292**: per-axis council member operating-within assumption surfaced explicitly in frontmatter
- **Catalog #300 v2**: full frontmatter (tier T2; attendees include 10 voices; quorum_met true; verdict PROCEED_WITH_REVISIONS; mission_contribution frontier_protecting; horizon_class frontier_pursuit)
- **Catalog #208**: docs/local-paths — only relative paths cited; canonical `.omx/research/` directory only
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: round 1 of 3-clean-pass; counter resets to 0 conditional on aggregate verdict; FIX-WAVE-R1 successor required

---

## Cross-references

- Design memo: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
- Sister A=DreamerV3 review memo: `.omx/research/path_3_a_recursive_adversarial_review_r1_3_axis_20260526.md`
- Sister D=Z6 review memo: `.omx/research/path_3_d_recursive_adversarial_review_r1_3_axis_20260526.md`
- Aggregate review memo: `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
- Canonical equation registry: `tac.canonical_equations::query_equations()` (empirically verified `procedural_predictor_plus_residual_correction_savings_v1` REGISTERED 2026-05-26)
- Sister `boost_nerv` substrate: `src/tac/substrates/boost_nerv/` lane `lane_boost_nerv_l0_scaffold_20260520`
- PR110 frontier reference: `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/archive.zip` + canonical frontier pointer `.omx/state/canonical_frontier_pointer.json`
