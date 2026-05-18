# Mechanism investigation: lane_g_v3_renderer + siren_renderer α=4.74 SUPER_ADDITIVE

**Lane:** `lane_super_additive_lane_g_v3_siren_topology_integration_20260517` (Task #823)

**Empirical anchor:** Q6 OP-3 extended sweep at `.omx/state/wyner_ziv_deliverability/pairwise_alpha_extended_20260517T215739Z.json` reported α=4.74 SUPER_ADDITIVE band for the lane_g_v3_renderer + siren_renderer pair under brotli.

**Investigation outcome:** primary hypothesis = **H2** (H2: byte-identity artifact).

## 9-dimension success checklist evidence

(Per Catalog #294 + CLAUDE.md '9-dim checklist' standing directive.)

- **Dimension 1 (UNIQUENESS):** this investigation surfaces a NEW META-pattern — candidate-loader-points-at-placeholder false-signal artifact — distinct from any existing apples-to-apples discipline anchor.
- **Dimension 2 (BEAUTY + ELEGANCE):** the mechanism is byte-identity, the cleanest possible explanation. SHA-256 equality is operationally definitive.
- **Dimension 3 (DISTINCTNESS):** distinct from MPS-vs-CUDA drift (Catalog #1 / #192) + phantom-baseline (CLAUDE.md FORBIDDEN_PATTERNS) — this is the loader-side variant.
- **Dimension 4 (RIGOR):** empirical SHA-256 + byte-histogram KL divergence + overlap fraction computed independently; cross-referenced with SIREN smoke timeout call_id.
- **Dimension 5 (OPTIMIZATION PER TECHNIQUE):** the v2 cascade extension is structurally needed for FUTURE real SUPER_ADDITIVE topologies; bounded at 2.0× to prevent runaway.
- **Dimension 6 (STACK-OF-STACKS-COMPOSABILITY):** non-applicable to this finding (false signal does not compose); v2 cascade composes orthogonally with sister predicted-delta adjusters.
- **Dimension 7 (DETERMINISTIC REPRODUCIBILITY):** SHA-256 + byte counts are deterministic + the empirical artifact carries written_at_utc.
- **Dimension 8 (EXTREME OPTIMIZATION + PERFORMANCE):** O(N) byte-histogram + O(256) KL computation; runs in ~1s on M5 Max.
- **Dimension 9 (OPTIMAL MINIMAL CONTEST SCORE):** no direct contest-score impact; the investigation PREVENTS misallocation of dispatch budget toward a false-signal topology.

## Cargo-cult audit per assumption

(Per Catalog #303 + CLAUDE.md HARD-EARNED-vs-CARGO-CULTED addendum.)

| Assumption | HARD-EARNED vs CARGO-CULTED | Rationale | Unwind path |
|---|---|---|---|
| 'high α in Q6 sweep means real composition discovery' | CARGO-CULTED | Inherited from compression-codec literature assuming distinct files | Add sha256-equality check at probe time |
| 'submission-builder produces real trained weights' | CARGO-CULTED | Inherited from successful-smoke assumption | Check returncode + elapsed_seconds vs configured timeout |
| 'CANONICAL_CANDIDATE_SUBSTRATES paths are always trained weights' | CARGO-CULTED | Inherited from initial probe-design assumption | Add per-candidate liveness verification (sha256-against-known-placeholders) |
| 'brotli SUPER_ADDITIVE α > 1.0 always indicates cross-substrate redundancy' | CARGO-CULTED | Inherited from cross-source compression literature where source distinctness was assumed | Add explicit byte-identity guard at sweep + autopilot consumer surfaces |

## Observability surface

(Per Catalog #305 + CLAUDE.md 'Max observability — non-negotiable' standing directive.)

- **Inspectable per layer:** candidate-A SHA, candidate-B SHA, byte-histogram per candidate, KL divergence, overlap fraction, empirical-anchor α, mechanism verdict.
- **Decomposable per signal:** byte-identity (sha256-level) decomposable from byte-histogram-similarity (structural-level) decomposable from KL divergence (distribution-level).
- **Diff-able across runs:** SHA-256 + byte counts are deterministic; two runs of this tool produce byte-identical reports.
- **Queryable post-hoc:** verdict dataclass is JSON-serializable + the markdown report has explicit fields per hypothesis.
- **Cite-able:** report cross-refs empirical artifact path + SIREN smoke call_id + canonical pre_entropy_substrate_pivot_prober.py lines.
- **Counterfactual-able:** if either candidate's bytes change (e.g. SIREN smoke succeeds + produces real weights), re-running this tool produces a different verdict; the byte-identity branch is the canonical counterfactual probe.

## Findings

### Candidate A: `lane_g_v3_renderer`

- **Path:** `experiments/results/lane_g_v3_landed/iter_0/renderer.bin`
- **Exists:** True
- **Size bytes:** 296,776
- **SHA-256:** `08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529`
- **Byte entropy (bits/byte):** 7.2934
- **Distinct byte values:** 256 / 256
- **Nonzero byte count:** 296,101

### Candidate B: `siren_renderer`

- **Path:** `experiments/results/lane_substrate_siren_modal_a100_dispatch_20260513T140410Z__smoke__100ep_modal/submissions/robust_current/renderer.bin`
- **Exists:** True
- **Size bytes:** 296,776
- **SHA-256:** `08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529`
- **Byte entropy (bits/byte):** 7.2934
- **Distinct byte values:** 256 / 256
- **Nonzero byte count:** 296,101

### Cross-candidate analysis

- **SHA-256 identity:** True
- **Byte identity:** True
- **Byte histogram KL divergence (A || B):** 0.000000 bits
- **Byte histogram overlap fraction:** 1.000000

### Empirical anchor cross-ref (Q6 OP-3 extended sweep)

- **Source:** `.omx/state/wyner_ziv_deliverability/pairwise_alpha_extended_20260517T215739Z.json`
- **α (savings_ratio_form):** 4.743780
- **Compressed alone A:** 261,806 bytes
- **Compressed alone B:** 261,806 bytes
- **Compressed concat:** 261,772 bytes

## Hypothesis verdict

**Primary hypothesis:** H2

**H1 (BYTE_LEVEL_STRUCTURE_SHARING) supported:** False
**H2 (BYTE_IDENTITY_ARTIFACT) supported:** True

### Mechanism explanation

BYTE_IDENTITY_ARTIFACT: lane_g_v3_renderer and siren_renderer point to byte-identical files (sha256 == 08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529). SIREN smoke (call_id fc-01KRGTEM56EXCV94Q7DC1HF0PB) TIMED OUT at 3601s (rc=124) and produced no trained weights; the submission-builder placed a placeholder renderer.bin into the SIREN dispatch dir which turned out to be the lane_g_v3 canonical reference checkpoint. The α=4.74 SUPER_ADDITIVE finding is brotli deduplication of identical bytes, NOT a real cross-substrate redundancy discovery.

## Operator-routable fix

Fix tools/pre_entropy_substrate_pivot_prober.py:189-192 to either (a) point to an actually-trained siren_renderer if a successful smoke exists, (b) remove siren_renderer from CANONICAL_CANDIDATE_SUBSTRATES until a successful run produces real weights, or (c) add a sha256-against-known-placeholders guard in the sweep that flags byte-identity as a false-signal artifact at probe time. Sister-anchor reinforces Catalog #215 (modal_smoke_recipe_min_gpu_class_consistent) — the SIREN T4 timeout is exactly the bug class that gate prevents going forward.

## Fail-closed canonical fields

- `score_claim`: False
- `promotion_eligible`: False
- `ready_for_exact_eval_dispatch`: False
- `evidence_grade`: `predicted_byte_level_analysis`
- `measurement_axis`: `[diagnostic; byte-level mechanism investigation]`

## Cross-references

- CLAUDE.md 'Apples-to-apples evidence discipline' (axis labels + custody mandatory)
- CLAUDE.md 'Forbidden component-aliasing for baselines' (the phantom-baseline pattern)
- CLAUDE.md 'Subagent coherence-by-default' + Catalog #125 (6-hook wire-in)
- CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' (v2 cascade extends canonical)
- Catalog #127 (per-call-site custody routing)
- Catalog #131 (fcntl-locked JSONL discipline for matrix posterior)
- Catalog #215 (modal_smoke_recipe_min_gpu_class_consistent — sister anchor for SIREN T4 timeout)
- Catalog #220 (substrate L1+ scaffold operational mechanism)
- Catalog #227 (substrate composition matrix — autopilot consumer)
- Catalog #229 (premise verification before edit)
- Catalog #230 (sister subagent ownership map)
- Catalog #272 (distinguishing-feature integration contract)
- `feedback_batched_815_816_q6_op3_extended_landed_20260517.md` (Q6 OP-3 extended sweep source)
- `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_20260517.md` (this landing memo)
- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` (v2 cascade)
- `.omx/state/substrate_composition_matrix.json` (canonical posterior surface; SUPER_ADDITIVE row appended with FALSE_SIGNAL blockers)
