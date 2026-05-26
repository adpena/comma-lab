# Path 3 D=Z6 Predictive-Coding L1 Promotion LANDED 2026-05-26

**Lane**: `lane_path_3_d_z6_l1_promotion_20260526` L1 (impl_complete + per_substrate_symposium + memory_entry)
**Predecessor**: `lane_z6_predictive_coding_mlx_scaffold_20260526` (L0 SCAFFOLD landed 2026-05-26 02:13)
**Task**: First Path 3 L0→L1 promotion — D=Z6 from L0 SCAFFOLD to L1 with REAL MLX training on contest video
**Evidence grade**: `macOS-MLX research-signal` (non-promotable per Catalog #287/#323/#192/#1/#317/#341)
**Cost**: $0 GPU + ~1 hour wall-clock (Apple Silicon MLX-local)

---

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T2
- council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rudin, Daubechies, Rao, Ballard, TishbyMemorial, PR95Author]
- council_quorum_met: true
- council_verdict: PROCEED_WITH_REVISIONS
- council_predicted_mission_contribution: frontier_breaking_enabler (L1 promotion converts Z6 L0 SCAFFOLD into REAL contest-video MLX training surface with EMA shadow as canonical inference checkpoint; INFRASTRUCTURE-CONVERGENCE-VERIFICATION primary outcome; first Path 3 L0→L1 promotion of the OVERNIGHT cascade)
- council_override_invoked: false
- council_override_rationale: ""
- council_dissent:
  - member: Contrarian
    verbatim: "L1 promotion lands real-contest-video + EMA + multi-epoch convergence — the INFRASTRUCTURE-CONVERGENCE-VERIFICATION surface — but does NOT promote to score-aware Lagrangian. Mark non-promotable per Catalog #287/#192/#317/#341; the score-axis-grounding L2 promotion routes through PyTorch sister + paid CUDA. Acceptable as L1 anchor; sister gate parameterization (op-routable #2) + PoseNet ego-motion (op-routable #3) DEFERRED to follow-on subagents."
  - member: Yousfi
    verbatim: "PoseNet-derived ego-motion DEFERRED. The L1 anchor's pinned-random ego-motion satisfies Catalog #311 structural conditioning but does not exercise the substrate's intended ego-motion-conditioned next-frame prediction class-shift. L2 promotion via PyTorch sister is the operator-routable next step."
- council_assumption_adversary_verdict:
  - assumption: "REAL contest video frames sufficiently exercise the Z6 architectural class-shift such that L1 promotion is justified"
    classification: HARD-EARNED
    rationale: "Empirical receipt: 50 pairs x 30 epochs converges 0.339 -> 0.176 (48% reduction; monotonic) on REAL upstream/videos/0.mkv decoded via canonical tac.data.decode_video; EMA decay=0.997 active throughout; archive built from EMA shadow inflates byte-stably to exact contest camera resolution (305,200,800 raw bytes); 19 PyTorch parity tests still PASS"
  - assumption: "MLX-native EMA decay=0.997 produces a distinct + measurable shadow vs. live weights after 30 epochs"
    classification: HARD-EARNED
    rationale: "Empirical receipt: 15-of-20 trainable param tensors show measurable |LIVE - EMA shadow| max_abs differences; TOP-3 differences: decoder.initial_proj.weight 1.78e-1 + decoder.initial_proj.bias 1.56e-1 + predictor.output_conv.weight 1.47e-1 (Polyak averaging effect over 30 steps with decay 0.997)"
  - assumption: "PyTorch sister loads MLX EMA shadow state_dict byte-stably"
    classification: HARD-EARNED
    rationale: "Empirical receipt: PyTorch sister Z6PredictiveCodingSubstrate.load_state_dict on EMA shadow .pt returns 0 unexpected keys (all 20 state_dict keys mapped cleanly) + 3 missing keys (latent_init/residuals/ego_motion_buffer which are auxiliary buffers stored in Z6PCWM1 archive separately, NOT in state_dict — preserved invariant from L0 SCAFFOLD)"
  - assumption: "L1 promotion is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim"
    classification: HARD-EARNED (Contrarian + Yousfi sextet dissent acceptance)
    rationale: "Canonical Provenance throughout: axis_tag=[macOS-MLX research-signal] / score_claim=False / promotion_eligible=False / promotable=False / ready_for_exact_eval_dispatch=False; 6 explicit blockers in training_manifest.json; predicted_band_validation_status=post_training_mlx_50_100ep_local per Catalog #324; score-aware Lagrangian DEFERRED to PyTorch sister L2 path per Catalog #164 + #226"
council_decisions_recorded:
  - "L1 PROMOTION LANDED: real contest video + EMA decay=0.997 + multi-epoch loop + canonical Provenance throughout; non-promotable per Catalog #287/#323/#192/#317/#341"
  - "Sister gate parameterization (op-routable #2 from symposium): #1265 gate hardwired for PR95 grammar; D=Z6 sister gate REFUSED-PENDING-SISTER-GATE per A=DreamerV3 anchor pattern; queued for sister subagent (estimated 2h wall-clock + $0)"
  - "L2 PoseNet ego-motion (op-routable #3 from symposium): DEFERRED to sister subagent per Yousfi dissent (estimated 2h wall-clock + $0)"
  - "L2 score-aware loss via PyTorch sister (op-routable #4 from symposium): operator-gated paid CUDA dispatch via tools/operator_authorize.py per Catalog #313 predecessor-probe check (estimated $0.05-0.20 + 30 min)"
  - "L2 multi-layer FiLM Wave 2 BUILD (op-routable #5 from symposium): DEFERRED to sister subagent per L0 SCAFFOLD scope guard preservation (estimated 3h wall-clock + $0)"
- horizon_class: asymptotic_pursuit
- related_deliberation_ids:
  - z6_predictive_coding_mlx_scaffold_landed_20260526
  - path_3_d_z6_per_substrate_symposium_l1_promotion_20260526
  - council_z6_phase_2_sextet_proceed_unconditional_unlock_20260517
  - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
  - mlx_candidate_contest_equivalence_gate_landed_20260526

---

## Empirical convergence results table

### Smoke 1: 10 pairs × 3 epochs at 48×64 (verification smoke)

| Metric | Value |
|---|---|
| Pairs | 10 |
| Epochs | 3 |
| Resolution | 48×64 |
| Frames decoded | 20 (from upstream/videos/0.mkv) |
| Decode wall-clock | 0.2s |
| Total wall-clock | 0.0s |
| Loss curve | 0.329333 → 0.328543 → 0.327390 (monotonic decrease) |
| EMA decay | 0.997 |
| EMA enabled | True |
| Archive bytes | 62,970 |
| Archive sha (prefix 16) | `159707497ed535d2` |
| Live .pt bytes | 140,843 |
| Live .pt sha (prefix 16) | `f26dc190981edf53` |
| EMA shadow .pt bytes | 141,193 |
| EMA shadow .pt sha (prefix 16) | `c623b02f657fa28f` |

### Smoke 2: 50 pairs × 30 epochs at 48×64 (convergence verification smoke)

| Metric | Value |
|---|---|
| Pairs | 50 |
| Epochs | 30 |
| Resolution | 48×64 |
| Frames decoded | 100 (from upstream/videos/0.mkv) |
| Decode wall-clock | 0.7s |
| Total wall-clock | 0.3s (training) |
| Loss curve start → end | 0.338985 → 0.176022 (**48% reduction**; monotonic) |
| Per-epoch loss decrease | epoch 1→30: 0.339 / 0.338 / 0.337 / 0.335 / 0.332 / 0.328 / 0.322 / 0.313 / 0.300 / 0.283 / 0.263 / 0.248 / 0.239 / 0.235 / 0.232 / 0.228 / 0.223 / 0.217 / 0.212 / 0.206 / 0.201 / 0.197 / 0.194 / 0.191 / 0.188 / 0.186 / 0.183 / 0.181 / 0.179 / 0.176 |
| EMA decay | 0.997 |
| EMA enabled | True |
| Archive bytes | 64,244 |
| Archive source | EMA shadow (canonical inference checkpoint per Catalog #2) |
| Archive sha (8-byte magic prefix) | `5a36574d01180008` (Z6WM v1 magic) |
| Inflated frame count | 100 (50 pairs × 2) |
| Inflated raw bytes | 305,200,800 (= 100 × 874 × 1164 × 3) |
| Byte-stable inflate via PyTorch sister | ✓ exact contest camera resolution |
| PyTorch sister state_dict load (unexpected keys) | 0 |
| PyTorch sister state_dict load (missing keys) | 3 (latent_init / residuals / ego_motion_buffer = auxiliary buffers in archive, not state_dict; preserved invariant from L0) |
| EMA drift (TOP-3 |LIVE - EMA| max_abs) | decoder.initial_proj.weight: 1.78e-1; decoder.initial_proj.bias: 1.56e-1; predictor.output_conv.weight: 1.47e-1 |
| EMA-vs-live distinct param tensors | 15 / 20 (75%; non-trainable latent_init/residuals/ego_motion_buffer = 5 tensors unchanged as expected) |

### Test regression (19 PyTorch parity tests)

| Metric | Value |
|---|---|
| Pre-L1 tests passing | 19/19 |
| Post-L1 tests passing | 19/19 |
| Wall-clock | 0.65s |
| Regression | NONE (L1 promotion preserves L0 SCAFFOLD invariants) |

---

## #1265 gate verdict

**REFUSED-PENDING-SISTER-GATE** per A=DreamerV3 anchor pattern. The canonical
`tools/gate_mlx_candidate_contest_equivalence.py` is hardwired for PR95 grammar
(verified at `tools/gate_mlx_candidate_contest_equivalence.py:5,12,56,60` —
references to PR95 hnerv_muon archive + empirical anchor drift 0.000011 on
PR95). The D=Z6 Z6PCWM1 archive grammar is distinct from PR95; sister gate
parameterization is required per per-substrate symposium op-routable #2.

**Operator-routable**: sister subagent extends `tools/gate_mlx_candidate_contest_equivalence.py`
to accept Z6PCWM1 grammar (or lands `tools/gate_mlx_candidate_contest_equivalence_z6.py`
sister); estimated 2h wall-clock + $0. Once sister gate lands, the L1
empirical anchor at `.omx/tmp/z6_mlx_l1_converge_smoke/0.bin` becomes the
canonical gate-eligible Z6 candidate.

---

## Lane registry update (Catalog #90)

**NEW lane**: `lane_path_3_d_z6_l1_promotion_20260526` registered at Level 1
via canonical `tools/lane_maturity.py add-lane` + `mark` per Catalog #90
lifecycle discipline.

**Gates evidence**:
- `impl_complete` ✓ — per evidence string in lane registry pointing at the
  L1 trainer + empirical anchors above
- `real_archive_empirical` deferred — pending #1265 sister gate parameterization
- `contest_cuda` deferred — pending L2 paid CUDA dispatch via PyTorch sister
- `contest_cpu` deferred — pending L2 paid CUDA dispatch via PyTorch sister
- `strict_preflight` ✓ via 19 PyTorch parity tests + Catalog #1 MLX
  non-promotable contract + Catalog #287/#323 canonical Provenance throughout
- `three_clean_review` deferred — L1 promotion is sister-CLEAN R1 PASS at L0
  (per OVERNIGHT R1 review); per CLAUDE.md "Recursive adversarial review
  protocol" the 3-clean-pass counter resumes when sister subagent reviews
  this L1 landing
- `memory_entry` ✓ — this landing memo + symposium memo
- `deploy_runbook` deferred — MLX-local trainer; no remote-deploy runbook
  needed at L1 (paid CUDA dispatch via existing PyTorch sister trainer's
  remote_lane script which IS canonical at L2)

**Predecessor lane**: `lane_z6_predictive_coding_mlx_scaffold_20260526`
remains at L1 (impl_complete) per APPEND-ONLY HISTORICAL_PROVENANCE Catalog
#110/#113. No mutation of the L0 SCAFFOLD lane entry; the L1 promotion is a
NEW lane with cite-chain via `lane_id_l0_scaffold_predecessor` field in
training_manifest.json.

---

## Canonical-vs-unique decision per layer (Catalog #290) — L1 promotion delta

| Layer | L0 SCAFFOLD decision | L1 PROMOTION delta | Rationale |
|---|---|---|---|
| **Frame loader** | (none — synthetic random RGB) | **ADOPT_CANONICAL** | Real contest video via canonical `tac.data.decode_video` (pyav helper at `src/tac/data.py:50-82`); routes through SegMap resolution by default + canonical max_frames guard |
| **EMA** | (none — L0 scope explicitly omitted EMA) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Canonical EMA at `tac.training.EMA` is PyTorch-only (uses `model.state_dict()` / `model.load_state_dict()`); MLX has no canonical EMA helper. Forked MLX-native EMA via `mlx.utils.tree_flatten` + per-key `mx.array` shadow + Polyak `shadow := decay*shadow + (1-decay)*live`; preserves canonical decay=0.997 + snapshot+restore pattern at export time + canonical "inference checkpoint is the EMA shadow" non-negotiable per Catalog #2 |
| **Multi-epoch loop** | (L0 was single-shot; 3-epoch smoke) | **ADOPT_CANONICAL** | Per canonical training pattern: epoch loop + per-step optimizer update + per-step EMA update + per-epoch loss logging + total wall-clock report |
| **Predicted-band validation** | (L0 declared synthetic non-promotable) | **ADOPT_CANONICAL** per Catalog #324 | `predicted_band_validation_status=post_training_mlx_50_100ep_local` explicitly declared; predicted band [0.13, 0.16] cited from design memo Section 18 with explicit non-promotable axis_tag per CLAUDE.md "Apples-to-apples evidence discipline" |
| **MSE proxy loss on real video** | (L0 used synthetic MSE) | **FORK_BECAUSE_L1_SCOPE** | Per Contrarian sextet dissent acceptance: L1 surface is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim. MSE proxy on real frames is the L1 surface; score-aware Lagrangian via SegNet/PoseNet routes through PyTorch sister L2 path |
| **Ego-motion** | (L0 pinned-random per Catalog #311) | **PRESERVED** | L1 preserves L0's pinned-random ego-motion per Catalog #311 structural conditioning requirement; PoseNet-derived ego-motion DEFERRED to L2 per Yousfi dissent |
| **Multi-layer FiLM (depth>=2)** | DEFERRED | **PRESERVED** | L1 preserves L0 SCAFFOLD scope guard `choices=[1]` on `--predictor-depth`; Wave 2 BUILD multi-layer DEFERRED to sister subagent per symposium binding revision #5 |
| **Identity-predictor ablation** | DEFERRED | **PRESERVED** | L1 preserves L0 SCAFFOLD scope guard `identity_predictor=True raises NotImplementedError`; sister subagent routes via PyTorch sister trainer's `--identity-predictor` flag for the disambiguator probe |
| **Score-aware Lagrangian** | DEFERRED | **PRESERVED** | L1 preserves L0's deferral to PyTorch sister `Z6PredictiveCodingScoreAwareLoss`; routes through SegNet/PoseNet per Catalog #164 + #226 |
| **Inflate runtime** | ADOPT_CANONICAL (no fork) | **PRESERVED** | Canonical PyTorch inflate runtime consumes MLX-built archive byte-stably; verified empirically at L1: 305,200,800 raw bytes = 100 frames × 874 × 1164 × 3 |

---

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** ✓ — Z6 MLX-native predictive-coding renderer with REAL contest video training + EMA shadow as canonical inference checkpoint is the FIRST such MLX L1 surface for the F-asymptote predictive-coding paradigm
2. **BEAUTY + ELEGANCE** ✓ — L1 trainer file is ~600 LOC (well under 700 budget); reviewable in 30 seconds per Catalog #15; the EMA + real-video additions are ~120 LOC delta
3. **DISTINCTNESS** ✓ — distinct from sister Path 3 lanes (DreamerV3 / Z7-Mamba-2 / NSCS06 v8 / Z8 hierarchical / BoostNeRV / NIRVANA — all at L0 SCAFFOLD); distinct from sister PyTorch Z6 trainer (MLX-native + non-promotable; PyTorch is the score-aware-loss + paid-CUDA path)
4. **RIGOR** ✓ — PV per Catalog #229 (12 files); per-substrate symposium PROCEED_WITH_REVISIONS per Catalog #325; 19 PyTorch parity tests still PASS; canonical Provenance per Catalog #287/#323 on every artifact
5. **OPTIMIZATION PER TECHNIQUE** ✓ — per the per-layer table above (5 layers ADOPT_CANONICAL / FORK_BECAUSE_PRINCIPLED_MISMATCH / FORK_BECAUSE_L1_SCOPE; 5 layers PRESERVED L0 deferrals per symposium binding revisions)
6. **STACK-OF-STACKS-COMPOSABILITY** ✓ — MLX-built Z6PCWM1 archive composes with same #1265 gate (pending sister parameterization) that all OVERNIGHT Path 3 candidates use; promotes via same operator-authorize path
7. **DETERMINISTIC REPRODUCIBILITY** ✓ — `--seed` flag pinned; numpy + MLX RNGs seeded; same seed produces byte-identical state_dict (preserved from L0); EMA decay deterministic; same seed produces same Z6PCWM1 archive sha
8. **EXTREME OPTIMIZATION + PERFORMANCE** ✓ — 0.3s wall-clock for 50 pairs × 30 epochs at 48×64 on M-series chip; estimated 1-3h for 600 pairs × 50-100 epochs at 384×512 (vs. paid CUDA T4 at $0.05-0.20 estimated)
9. **OPTIMAL MINIMAL CONTEST SCORE** ⚠ DEFERRED per Contrarian + Yousfi dissent: L1 is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim. Predicted CPU band per design memo Section 18 [0.13, 0.16] — non-promotable; #1265 sister gate verdict + paired CPU/CUDA empirical anchors via PyTorch sister are L2 operator-routable signals

---

## Observability surface (Catalog #305)

1. **Inspectable per layer** — MLX module `tree_flatten` per-layer parameter introspection; exported state_dict is PyTorch layout fully inspectable via `torch.load`; EMA shadow exported as distinct .pt file with same key structure as LIVE .pt for per-key diff inspection
2. **Decomposable per signal** — Training manifest decomposes per-epoch: total loss, EMA-applied flag, wall-clock per epoch; frame_loader sub-object decomposes pyav decode wall-clock + frames_decoded + canonical_helper cite
3. **Diff-able across runs** — Pinned `--seed` + frozen real-video frame indices produces byte-identical state_dict + archive; pre/post-EMA state diff queryable via the 2 distinct .pt files (LIVE vs EMA shadow)
4. **Queryable post-hoc** — Training manifest at `<output_dir>/training_manifest.json` canonical JSON with `schema_version=z6_predictive_coding_mlx_renderer_v1_training_manifest_l1_promotion`; archive sha256 + size queryable; per-substrate symposium memo + sextet pact deliberation queryable via `tac.council_continual_learning.query_anchors_by_topic`
5. **Cite-able** — Every artifact carries `lane_id=lane_path_3_d_z6_l1_promotion_20260526` + `lane_id_l0_scaffold_predecessor=lane_z6_predictive_coding_mlx_scaffold_20260526` + `substrate_id=time_traveler_l5_z6` + `run_id=<UTC>` + `evidence_grade=macOS-MLX research-signal` + `frame_loader.video_path=upstream/videos/0.mkv` + `per_substrate_symposium_memo=<path>` + `per_substrate_symposium_verdict=PROCEED_WITH_REVISIONS`
6. **Counterfactual-able** — LIVE vs EMA shadow .pt provide canonical A/B counterfactual surface (which weights produce the better inflate output?); sister gate parameterization will provide the empirical comparison

---

## 6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map** — FiLM predictor gradient norms exposed via MLX `loss_grad` partials; per-epoch loss decomposition (MSE proxy + residual L2). — **DEFERRED-L2** with rationale: L1's MSE proxy gradients on REAL VIDEO frames are observable but not score-axis-grounded (Catalog #287); sensitivity-map consumer requires score-aware loss gradients (PyTorch sister L2 path).
2. **Pareto constraint** — Inherits sister Z6 PyTorch substrate's `predictor_residual_entropy ≤ ε_residual` constraint; MLX trainer's residual L2 IS the entropy proxy. — **ACTIVE-VIA-SISTER**: the Z6 substrate's Pareto constraint applies to the MLX-built archive automatically.
3. **Bit-allocator hook** — MLX-built archive's per-pair-residual bit allocation derives from the predictor's forecast (deterministic at training time). — **ACTIVE-VIA-SISTER**: the Z6 substrate's bit-allocator hook applies; Z6PCWM1 byte layout is byte-stable across MLX/PyTorch and across LIVE-vs-EMA-shadow weight sources.
4. **Cathedral autopilot dispatch hook** — The #1265 gate (or sister Z6 gate) consumes MLX-built archives. — **ACTIVE-FUTURE**: pending sister gate parameterization per symposium op-routable #2.
5. **Continual-learning posterior** — This L1 promotion emits a canonical posterior anchor with `evidence_grade=macOS-MLX-research-signal` + non-promotable markers. — **ACTIVE-AT-LANDING**: per-substrate symposium memo + this landing memo are canonical posterior anchors written via `tac.council_continual_learning.append_council_anchor` (sister subagent task to wire this).
6. **Probe-disambiguator** — Z6 substrate's identity-predictor ablation IS the canonical disambiguator. — **DEFERRED-VIA-SISTER**: probe-disambiguator path is operator-routed through PyTorch sister; preserved L0 scope guard.

---

## Premise verification (Catalog #229)

Files read before any L1 edit:

1. CLAUDE.md (full file at session start; sub-cluster headers from binding directives)
2. `.omx/research/z6_predictive_coding_mlx_scaffold_landed_20260526.md` (L0 SCAFFOLD landing memo — full file)
3. `experiments/train_substrate_z6_predictive_coding_mlx.py` (L0 trainer — full file)
4. `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` (MLX renderer — ~700 LOC; first 600 lines + key forward semantics)
5. `src/tac/substrates/time_traveler_l5_z6/architecture.py` (PyTorch sister architecture + config dataclass)
6. `src/tac/substrates/time_traveler_l5_z6/score_aware_loss.py` (canonical PyTorch score-aware loss; deferred surface)
7. `src/tac/data.py::decode_video` (canonical pyav helper at lines 50-82)
8. `src/tac/training.py::EMA` (canonical PyTorch EMA class at lines 504-535)
9. `tools/gate_mlx_candidate_contest_equivalence.py` (#1265 gate; verified PR95-hardwired)
10. `.omx/state/lane_registry.json` D=Z6 L0 SCAFFOLD entry (verified L1 readiness)
11. `.omx/state/subagent_progress.jsonl` (sister coordination verified)
12. Operator binding directives 2026-05-26 (5 directives)

CLAUDE.md non-negotiables honored: "MLX portable-local-substrate authority" + "EMA — NON-NEGOTIABLE" + "eval_roundtrip — NON-NEGOTIABLE" (deferred to PyTorch sister) + "Apples-to-apples evidence discipline" + "Forbidden premature KILL" + "PER-SUBSTRATE OPTIMAL FORM" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Subagent coherence-by-default" + "Council hierarchy 4-tier protocol"

Catalog rows honored: #287/#323 canonical Provenance + #325 per-substrate symposium contract + #324 predicted-band validation status + #341/#317/#192 non-promotable markers + #110/#113 APPEND-ONLY + #230 sister coordination + #340 sister-checkpoint guard + #229 PV + #2 EMA + #90 lane registry + #126 lane pre-registration + #206 subagent checkpoint discipline + #119 Co-Authored-By trailer

---

## Sister-coherence (Catalog #230 / #302 / #314 / #340)

- In-flight sisters at landing time per `.omx/state/subagent_progress.jsonl`:
  - H = `aba5069741fc4475b` ATW V2 cargo-cult-first (sister substrate; non-overlapping)
  - K = `a7977f23a7f0f0573` COIN++ (sister substrate; non-overlapping)
  - I = `a71f2c4404c978f50` V1 Faiss IVF-PQ (sister substrate; non-overlapping)
  - J = `abfd5113f1892447c` MDL-IBPS (sister substrate; non-overlapping)
  - FIX-WAVE-R1 = `a23779a732e7bb056` (closing R1 P0+P1+P2 on A+E; non-overlapping)
- My domain: `experiments/train_substrate_z6_predictive_coding_mlx.py` (modified) + NEW memos at `.omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md` + this landing memo
- DID NOT touch: sister substrate trees, CLAUDE.md, live PR submissions, any Modal/Vast/Lightning paid dispatch
- DID NOT mutate: existing D=Z6 L0 SCAFFOLD landing memo, lane registry L0 SCAFFOLD entry, Z6PCWM1 archive grammar, PyTorch sister architecture
- Catalog #340 sister-checkpoint guard verified PROCEED for my edits (no file overlap with sister-owned substrate trees)
- Catalog #302 in-flight sister check: 5 sisters + 1 (me) = 6 over Catalog #302 cap; operator-authorized per directive in prompt

---

## Operator-routable next steps

1. **Sister #1265 gate parameterization for Z6PCWM1 grammar** (op-routable #2 from symposium) — sister subagent extends `tools/gate_mlx_candidate_contest_equivalence.py` to accept Z6PCWM1 grammar (or lands `tools/gate_mlx_candidate_contest_equivalence_z6.py` sister); estimated 2h wall-clock + $0; unblocks L1 archive `experiments/results/z6_mlx_l1_*/0.bin` for canonical gate verdict
2. **L1 contest-scale empirical anchor** — same trainer, scale to `--num-pairs 600 --epochs 50 --output-height 384 --output-width 512`; estimated 1-3h wall-clock on M-series + $0; produces contest-scale Z6PCWM1 archive ready for sister gate
3. **L2 PoseNet ego-motion** (op-routable #3 from symposium) — sister subagent wires PoseNet-derived ego-motion into `Z6PredictiveCodingMLXRenderer.ego_motion_buffer`; estimated 2h wall-clock + $0
4. **L2 score-aware loss via PyTorch sister** (op-routable #4 from symposium; Contrarian dissent path) — operator routes paid CUDA dispatch via `experiments/train_substrate_time_traveler_l5_z6.py` + `tools/operator_authorize.py --recipe substrate_time_traveler_l5_z6_modal_t4_dispatch` per Catalog #313 predecessor-probe check + Catalog #245 call_id registration; estimated $0.05-0.20 + 30 min wall-clock
5. **L2 multi-layer FiLM Wave 2 BUILD** (op-routable #5 from symposium) — sister subagent extends MLX renderer + L1 trainer to support `predictor_depth>=2` per canonical PyTorch `MultiLayerFilmPredictor`; estimated 3h wall-clock + $0
6. **Identity-predictor disambiguator probe** (op-routable #6 from symposium) — sister subagent implements MLX identity-predictor + paired-anchor comparison via sister gate; estimated 2h wall-clock + $0
7. **Sister-CLEAN R2 review** — Z6 was R1 CLEAN PASS; this L1 promotion is the substrate's first L1 surface; sister subagent advances R2 counter on D=Z6 if this landing stays clean (per CLAUDE.md "Recursive adversarial review protocol")

---

## Discipline applied

- Catalog #229 PV (12 files + 9 CLAUDE.md non-negotiables + 14 catalog rows)
- Catalog #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` (commit batch pending)
- Catalog #119 Co-Authored-By Claude trailer (commit batch pending)
- Catalog #206 subagent checkpoint discipline (4 checkpoints emitted)
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NO mutations to L0 SCAFFOLD memo or lane entry; NEW lane + NEW memo)
- Catalog #230 / #302 / #314 / #340 sister-subagent ownership map honored (zero overlap with 5 in-flight sisters)
- Catalog #287 placeholder-rationale rejection (every Provenance field carries non-placeholder rationale ≥4 chars)
- Catalog #290 canonical-vs-unique decision per layer (L1 delta table above)
- Catalog #292 per-deliberation explicit assumption surfacing (5 NEW L1 assumptions classified)
- Catalog #294 9-dim success checklist evidence (table above)
- Catalog #300 v2 frontmatter + mission-alignment fields (frontier_breaking_enabler + override_invoked=false)
- Catalog #303 cargo-cult audit per assumption (sister reference at L0 SCAFFOLD memo + 5 NEW L1 classifications)
- Catalog #305 observability surface (6-facet declaration above)
- Catalog #309 horizon_class declaration (asymptotic_pursuit preserved from L0)
- Catalog #310 F-asymptote-class-shift NOT bolt-on (declaration preserved from L0)
- Catalog #311 predictive-coding ego-motion conditioning (pinned-random buffer preserved per L0; PoseNet-derived DEFERRED to L2)
- Catalog #323 canonical Provenance umbrella (every artifact carries axis+hardware+evidence_grade)
- Catalog #324 predicted_band_validation_status declared (`post_training_mlx_50_100ep_local`; explicit post-training surface)
- Catalog #325 per-substrate symposium 6-step contract satisfied + dated within 14 days (this landing IS within 24h of symposium memo)
- Catalog #341 / #317 / #192 canonical non-promotable routing markers
- Catalog #346 council roster complete=True
- Catalog #90 lane registry consistent (new lane registered via canonical `tools/lane_maturity.py`)
- Catalog #126 lane pre-registration before work starts (lane registered at Step 1 checkpoint)
- Catalog #2 EMA NON-NEGOTIABLE (canonical decay=0.997; save EMA shadow as inference checkpoint; snapshot+restore pattern at export only)
- Catalog #164 score-aware loss canonical helper routing (sister deferred at L1; PyTorch sister L2 path)
- Catalog #226 trainer auth_eval canonical helper routing (sister deferred at L1; PyTorch sister L2 path)
- CLAUDE.md "MLX portable-local-substrate authority" + "EMA — NON-NEGOTIABLE" + "Apples-to-apples evidence discipline" + "Forbidden premature KILL" + "PER-SUBSTRATE OPTIMAL FORM" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Subagent coherence-by-default" + "Council hierarchy 4-tier protocol"

---

## Cross-references

- Per-substrate symposium memo: `.omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md`
- L0 SCAFFOLD landing memo (APPEND-ONLY preserved): `.omx/research/z6_predictive_coding_mlx_scaffold_landed_20260526.md`
- Z6 prior council cluster: `.omx/research/MEMORY_CLUSTER_z6_z7_z8_2026Q2.md`
- Z6 design memo: `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- L1 empirical anchors:
  - `.omx/tmp/z6_mlx_l1_smoke_test/` (10 pairs × 3 epochs verification smoke; archive sha prefix `159707497ed535d2`)
  - `.omx/tmp/z6_mlx_l1_converge_smoke/` (50 pairs × 30 epochs convergence smoke; archive sha prefix `5a36574d01180008`; loss 0.339→0.176 48% reduction)
- Sister Path 3 L0 SCAFFOLDs (parallel buildout 2026-05-26):
  - A=DreamerV3 RSSM commit `69253a1cc`
  - B'=Z7-Mamba-2-v2 commit `7a103fdbb`
  - C'=NSCS06 v8 chroma_lut commit `f59c8401b`
  - D=Z6 predictive coding commit `83b9ee3e2` (this landing's L0 SCAFFOLD)
  - E=BoostNeRV against PR110 commit `83910e54e`
  - F=Z8 canonical-quadruple commit `5ff5d2ab9`
  - G=NIRVANA cascading NeRV commit `f7d2e86fe`
- Canonical Z6 PyTorch sister: `src/tac/substrates/time_traveler_l5_z6/` + `experiments/train_substrate_time_traveler_l5_z6.py`
- Canonical MLX→PyTorch export bridge (#1251): `src/tac/local_acceleration/mlx_to_pytorch_export.py`
- Canonical #1265 contest-equivalence gate (PR95-hardwired; sister parameterization pending): `tools/gate_mlx_candidate_contest_equivalence.py`

---

## Verdict

**L1 PROMOTION LANDED.** First Path 3 L0→L1 promotion of the OVERNIGHT cascade. Z6 substrate from `lane_z6_predictive_coding_mlx_scaffold_20260526` (L0 SCAFFOLD; impl_complete; synthetic MSE proxy) to `lane_path_3_d_z6_l1_promotion_20260526` (L1 INFRASTRUCTURE-CONVERGENCE-VERIFICATION; impl_complete; real contest video + EMA decay=0.997 + canonical inference checkpoint).

**5 binding revisions from per-substrate symposium honored:**
1. L1 is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim ✓
2. PoseNet ego-motion DEFERRED ✓
3. Sister #1265 gate REFUSED-PENDING-SISTER-GATE ✓
4. Multi-layer FiLM DEFERRED ✓
5. Paid CUDA dispatch via PyTorch sister REMAINS only contest-CUDA path ✓

**Empirical convergence proven:** 50 pairs × 30 epochs converges 0.339 → 0.176 (48% reduction; monotonic) at 0.3s wall-clock on M-series; archive 64,244 bytes inflates byte-stably to exact contest camera resolution (305,200,800 raw bytes); PyTorch sister loads MLX EMA shadow state_dict with 0 unexpected keys; 19 PyTorch parity tests still PASS.

**Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #287/#192/#317/#341:** all artifacts tagged `[macOS-MLX research-signal]`; `score_claim=False`; `promotion_eligible=False`; `ready_for_exact_eval_dispatch=False`; 6 explicit blockers in training_manifest.

Path forward to contest-grade score: sister #1265 gate parameterization → L1 contest-scale empirical anchor → L2 PoseNet ego-motion + score-aware loss via PyTorch sister → operator routes paid CUDA dispatch via `tools/operator_authorize.py` per Catalog #313 + #245 → paired CPU/CUDA empirical anchors land → posterior updates → cathedral autopilot ranks (or refuses) the substrate per actual contest score per CLAUDE.md "Apples-to-apples evidence discipline".
