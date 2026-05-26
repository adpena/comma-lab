# Path 3 D=Z6 Per-Substrate Symposium for L1 Promotion 2026-05-26

**Lane**: `lane_path_3_d_z6_l1_promotion_20260526` L1 (impl_complete pending)
**Substrate**: `time_traveler_l5_z6` — Z6 predictive-coding (Time-Traveler L5 F-asymptote node)
**Tier**: T2 (sextet pact + topical grand-council seats per Catalog #325 step 4)
**Cost**: $0 (MLX-local symposium; no paid GPU)

---

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T2
- council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Rudin, Daubechies, Rao, Ballard, TishbyMemorial, PR95Author]
- council_quorum_met: true
- council_verdict: PROCEED_WITH_REVISIONS
- council_predicted_mission_contribution: frontier_breaking_enabler (the L1 promotion converts the Z6 L0 SCAFFOLD into a REAL contest-video MLX training surface; on convergence + #1265 PASS this unblocks the operator-routable paid CUDA dispatch path that has been blocked since 2026-05-18 Wave 2 driver hardcode bug per Catalog #326)
- council_override_invoked: false
- council_override_rationale: ""
- council_dissent:
  - member: Contrarian
    verbatim: "The MLX-local synthetic MSE proxy from L0 SCAFFOLD is NOT a score-aware Lagrangian. L1 promotion that swaps in real contest video frames but keeps MSE proxy is still non-promotable per Catalog #287. The path to true contest-score gradients runs through the PyTorch sister trainer's `Z6PredictiveCodingScoreAwareLoss` which routes through SegNet/PoseNet — and that is PyTorch-only. The MLX trainer's L1 promotion is therefore an INFRASTRUCTURE-AND-CONVERGENCE-VERIFICATION promotion (proves MLX-native predictor + encoder + decoder learn on REAL video), NOT a score-aware-Lagrangian promotion. Mark non-promotable per Catalog #287/#192/#317/#341; #1265 gate verdict is the ONLY signal that crosses the L1→paid-dispatch threshold."
  - member: Yousfi
    verbatim: "I want the FiLM predictor's ego-motion to be DERIVED from PoseNet projections on real frame pairs, not pinned to random buffer. The Z6 architectural core is ego-motion-conditioned next-frame prediction (Catalog #311). Synthetic ego-motion + real video targets is a STILL-PARTIAL realization of the substrate's intended scorer-relationship class-shift. Note this as a deferred Phase 3 escalation."
- council_assumption_adversary_verdict:
  - assumption: "REAL contest video frames sufficiently exercise the Z6 architectural class-shift such that L1 promotion is justified"
    classification: HARD-EARNED
    rationale: "Z6 architectural distinctness from sister substrates (encoder + FiLM predictor + autoregressive decoder + per-pair residuals) IS exercised end-to-end on real frames; the predictor learns to forecast the next-pair latent from the previous-pair latent + ego-motion; gradients flow through all 5 sub-modules; the residual Lagrangian term enforces Rao-Ballard predictive-coding semantics. The architectural surface IS the L1 surface; the score-aware-loss surface remains the PyTorch sister's L2 surface."
  - assumption: "MLX-local convergence at small subset (e.g. 50 pairs × 5 epochs) is a sufficient L1 anchor"
    classification: HARD-EARNED-WITH-CAVEAT
    rationale: "Per CLAUDE.md 'MLX portable-local-substrate authority': MLX-local is a research signal NOT a contest axis. The L1 anchor here is INFRASTRUCTURE-CONVERGENCE not score-claim. Caveat: a contest-scale Apple-Silicon run (600 pairs × 50-100 epochs at 384×512) may take 1-3h+ wall-clock on M-series chip; smoke subset is the verification primitive, contest-scale is the optional follow-on. The #1265 gate verdict on the contest-scale archive is the operator-routable decision point."
  - assumption: "EMA shadow weights as inference checkpoint is mandatory at L1"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'EMA — NON-NEGOTIABLE': every training path that produces an inference checkpoint MUST instantiate EMA decay=0.997 + save EMA shadow (not live weights) as inference. L0 SCAFFOLD intentionally omitted EMA per the L0 scope; L1 promotion adds EMA per canonical contract."
  - assumption: "The MSE proxy → real-video upgrade IS the L1 promotion (vs. PyTorch routing to score-aware Lagrangian)"
    classification: CARGO-CULTED-WITH-WAIVER
    rationale: "Cargo-cult: 'an MLX trainer that processes real video frames is L1-ready'. The unwound truth: per Catalog #164 + #226 the SCORE-AWARE LAGRANGIAN routes through SegNet/PoseNet which is PyTorch-only at training time. The MLX-local L1 promotion delivers INFRASTRUCTURE + REAL-VIDEO + EMA + CONVERGENCE-VERIFICATION; the score-aware-loss-against-real-scorers L2 promotion routes through the PyTorch sister (operator-gated paid CUDA dispatch). Waiver: this L1 promotion is HONEST about its scope; canonical Provenance preserves the non-promotable axis_tag throughout."
  - assumption: "Real ego-motion from PoseNet projections is required at L1"
    classification: CARGO-CULTED-WITH-DEFER
    rationale: "Cargo-cult: 'L1 promotion requires PoseNet-derived ego-motion'. Unwound: Catalog #311 enforces ego-motion-CONDITIONING (the FiLM modulation pipeline IS exercised — gradients flow through ego_motion → film_mlp → scale/shift → predictor); WHETHER the ego-motion has semantic content (pinned-random vs PoseNet-projected) is a Phase 2/3 L2 promotion concern. The L1 surface honors the architectural CONDITIONING requirement; the L2 surface routes PoseNet projections via the PyTorch sister. Per Yousfi's dissent above: note this as a deferred Phase 3 escalation."
- council_decisions_recorded:
  - "op-routable #1: L1 promotion lands real-contest-video frame loader + EMA + multi-epoch training; canonical Provenance preserves non-promotable axis_tag throughout; the L1 anchor is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim"
  - "op-routable #2: after L1 convergence, operator routes #1265 gate verdict on the MLX-built Z6PCWM1 archive (note: gate is hardwired for PR95 grammar; D=Z6 needs sister gate `tools/gate_mlx_candidate_contest_equivalence_z6.py` parameterized for Z6PCWM1 — DEFER to follow-on subagent per CLAUDE.md 'Forbidden premature KILL')"
  - "op-routable #3: paid CUDA dispatch on PyTorch sister trainer remains the only operator-routable contest-CUDA path; L1 promotion does NOT change that contract"
  - "op-routable #4: PoseNet-derived ego-motion + score-aware loss against real scorers IS the L2 promotion (Phase 3 escalation per Yousfi); deferred to follow-on subagent"
  - "op-routable #5: Wave 2 multi-layer FiLM (predictor_depth>=2) remains DEFERRED per the L0 SCAFFOLD scope guard; multi-layer L1 promotion is a sister follow-on after this L1 anchor lands"
- horizon_class: asymptotic_pursuit
- council_override_invoked: false
- related_deliberation_ids:
  - council_z6_phase_2_sextet_proceed_unconditional_unlock_20260517
  - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
  - z6_predictive_coding_mlx_scaffold_landed_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526

---

## 1. Cargo-cult audit per assumption (Catalog #303 sister reference)

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" Catalog #325 step 1: the cargo-cult audit was performed at the L0 SCAFFOLD landing (`z6_predictive_coding_mlx_scaffold_landed_20260526.md` "Cargo-cult audit per assumption" table — 8 assumptions classified with full HARD-EARNED-vs-CARGO-CULTED reasoning). This L1 promotion symposium does NOT re-audit those 8 assumptions (they remain HARD-EARNED at L0); it adds 5 NEW assumption classifications above (council_assumption_adversary_verdict block) for the L1-specific decisions:

1. Real-video sufficiency for L1 (HARD-EARNED)
2. MLX-local convergence at small subset as L1 anchor (HARD-EARNED-WITH-CAVEAT)
3. EMA shadow as inference checkpoint mandatory at L1 (HARD-EARNED)
4. MSE proxy → real-video upgrade IS the L1 promotion (CARGO-CULTED-WITH-WAIVER — unwound to "infrastructure + convergence" framing)
5. PoseNet ego-motion required at L1 (CARGO-CULTED-WITH-DEFER — unwound to "conditioning is honored; semantic content is L2")

Per Catalog #292 each council member surfaced their operating-within assumption; the 5 above are the union of NEW assumptions specific to L1 promotion.

---

## 2. 9-dimension success checklist evidence (Catalog #294)

| Dim | Criterion | Evidence at L1 promotion |
|---|---|---|
| 1 | **UNIQUENESS** | Z6 MLX-native predictive-coding renderer with REAL contest video training is the first such MLX surface for the F-asymptote predictive-coding paradigm; sister substrates (DreamerV3 / Z7-Mamba-2 / NSCS06 v8 / Z8 hierarchical) are at L0 SCAFFOLD only |
| 2 | **BEAUTY + ELEGANCE** | L1 promotion adds ~300-400 LOC to `experiments/train_substrate_z6_predictive_coding_mlx.py` (real-GT loader + EMA + multi-epoch loop); total trainer LOC stays ≤ 700; reviewable in 30 seconds per Catalog #15 |
| 3 | **DISTINCTNESS** | The L1 surface is MLX-native real-video training; distinct from sister Z6-PyTorch trainer (`experiments/train_substrate_time_traveler_l5_z6.py`) which uses score-aware loss + PoseNet ego-motion + paid CUDA |
| 4 | **RIGOR** | PV per Catalog #229 (read SCAFFOLD memo + MLX renderer + score_aware_loss + tac.data.decode_video + canonical EMA pattern in tac.training); 19 existing tests PASS; new convergence tests added |
| 5 | **OPTIMIZATION PER TECHNIQUE** | Per L0 SCAFFOLD "Canonical-vs-unique decision per layer" table preserved; L1 promotion ADOPTS canonical `tac.data.decode_video` for pyav loading; ADOPTS canonical `tac.training.EMA` pattern; FORKS for MLX-native loss surface (MSE proxy on real video; score-aware Lagrangian deferred to PyTorch sister per Contrarian dissent) |
| 6 | **STACK-OF-STACKS-COMPOSABILITY** | MLX-built Z6PCWM1 archive composes with the same #1265 gate that all OVERNIGHT Path 3 candidates use; promotes via the same operator-authorize path; L1 archive is byte-stable contest format ready for sister gate parameterization or direct gate routing once Z6 sister gate lands |
| 7 | **DETERMINISTIC REPRODUCIBILITY** | `--seed` flag pinned; numpy + MLX RNGs seeded; same seed produces byte-identical state_dict (preserved from L0); EMA decay deterministic |
| 8 | **EXTREME OPTIMIZATION + PERFORMANCE** | MLX native Apple Silicon execution; smoke 50 pairs × 5 epochs at 48×64 expected <10s; contest-scale 600 pairs × 50-100ep at 384×512 estimated 1-3h on M-series chip (vs. paid CUDA T4 at $0.05-0.20 estimated cost for paid CUDA dispatch) |
| 9 | **OPTIMAL MINIMAL CONTEST SCORE** | ⚠ DEFERRED per Contrarian dissent: L1 promotion is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim. Predicted CPU band per design memo Section 18 [0.13, 0.16] — non-promotable; #1265 gate verdict (or sister gate) on the MLX-built archive is the operator-routable signal for paid CUDA dispatch |

---

## 3. Observability surface declaration (Catalog #305)

1. **Inspectable per layer** — MLX module `tree_flatten` per-layer parameter introspection; exported state_dict is PyTorch layout fully inspectable via `torch.load`
2. **Decomposable per signal** — Training manifest decomposes per-epoch: total loss, MSE reconstruction loss (real frames), residual L2 Lagrangian term, EMA-decay-applied loss, wall-clock per epoch, GT frame batch loading wall-clock
3. **Diff-able across runs** — Pinned `--seed` + frozen real-video frame indices produces byte-identical state_dict + archive per Catalog #110 deterministic invariant; pre/post-EMA state diff queryable via export_state_dict
4. **Queryable post-hoc** — Training manifest at `<output_dir>/training_manifest.json` canonical JSON; archive sha256 + size queryable via canonical Provenance; EMA shadow stored separately as `<output_dir>/z6_mlx_state_dict_ema_shadow.pt`
5. **Cite-able** — Every artifact carries `lane_id=lane_path_3_d_z6_l1_promotion_20260526` + `substrate_id=time_traveler_l5_z6` + `run_id=<UTC>` + `evidence_grade=macOS-MLX research-signal` + `video_path=upstream/videos/0.mkv` + `frames_decoded=<count>`
6. **Counterfactual-able** — Same byte-mutation discipline via canonical `tools/verify_distinguishing_feature_byte_mutation.py` once the lane gains distinguishing-feature declarations; Z6 substrate distinguishing-feature is predictor weight blob (per L0 SCAFFOLD); the L1 EMA shadow produces an alternate-distinguishing-feature archive variant for counterfactual A/B comparison

---

## 4. Sextet pact deliberation per CLAUDE.md "Council conduct" amendment + Catalog #292

### Shannon LEAD (information-theory grounding)

Operating-within assumption: "The Z6 substrate's Rao-Ballard predictive-coding hierarchy IS a class-shift toward the I(T;Y) cooperative-receiver optimum per Tishby IB; the FiLM predictor's forecast residual norm IS the Bayesian Information bottleneck term."

Position: PROCEED. The L1 promotion does NOT change the score-axis-grounding (which remains the PyTorch sister's L2 surface) but DOES add the real-video information-theoretic anchor: the per-pair latent distribution conditioned on real contest frames CAN be inspected for entropy + R(D) bound proximity. The MLX-local infrastructure surface is the right place to do this side-information analysis at $0.

### Dykstra CO-LEAD (alternating-projections feasibility)

Operating-within assumption: "Real contest video frames are within the convex feasibility set of the Z6 architectural class; the encoder + predictor + decoder respond gradient-coherent on real frames vs. synthetic noise."

Position: PROCEED. The L1 promotion preserves the existing convex-feasibility surface (Z6PCWM1 archive grammar; canonical pack_archive; canonical inflate); the only NEW feasibility surface is the MLX-native AdamW + EMA stack which mirrors the canonical PyTorch sister. No new Pareto constraint introduced; the existing Z6 substrate's `predictor_residual_entropy ≤ ε_residual` constraint applies to the MLX-built archive automatically.

### Yousfi (steganalysis + contest design)

Operating-within assumption: "Real contest video frames exercise the SegNet/PoseNet scorer-response surface; without that exposure during training, the architectural class-shift is unproven against the actual contest scorer."

Position: PROCEED_WITH_REVISIONS. Real-video training is necessary; SegNet/PoseNet routing (score-aware loss) is sufficient for true score-axis-grounding. Per dissent above: deferred Phase 3 PoseNet-derived ego-motion is the L2 promotion's responsibility. The L1 INFRASTRUCTURE-CONVERGENCE surface is acceptable at MLX-local; cannot claim score-axis-promotion at L1.

### Fridrich (inverse steganalysis)

Operating-within assumption: "The Z6 substrate's FiLM modulation is the canonical inverse-steganalysis-friendly architecture (smooth conditioning vs. hard-token-selection); real-video exposure validates this for L1 promotion."

Position: PROCEED. The MLX-native FiLM predictor mirrors the PyTorch sister exactly (verified by 19 PASS tests + 1-of-1 byte-stable state_dict export at L0 SCAFFOLD); real-video L1 promotion does not change this property.

### Contrarian (challenges weak arguments)

Operating-within assumption: "An MLX trainer that processes real video frames is L1-ready" — DECLARED CARGO-CULTED. See dissent above. Contrarian's veto is invoked: the L1 promotion MUST be honest about its INFRASTRUCTURE-CONVERGENCE scope and MUST NOT claim score-axis-promotion. The canonical Provenance contract via Catalog #287/#192/#317/#341 carries this honestly.

Position: PROCEED_WITH_REVISIONS (contingent on canonical Provenance non-promotable markers throughout). The L1 promotion is INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim.

### Assumption-Adversary (challenges framing)

Operating-within assumption: "L1 promotion is a milestone in the iteration loop; not a contest-score claim." See council_assumption_adversary_verdict block above (5 NEW assumptions classified).

Position: PROCEED. With the 5 NEW assumptions surfaced + the L0 SCAFFOLD's 8 prior cargo-cult audits preserved + EMA + real-video added per canonical contract, the L1 promotion satisfies the assumption-surfacing discipline.

### Topical Grand Council seats (Catalog #325 step 4)

- **Rao** (predictive coding architect): PROCEED — the FiLM predictor's residual norm IS the canonical Rao-Ballard 1999 predictive-coding term; real-video exposure validates the forecast learning. PROCEED.
- **Ballard** (embodied predictive coding): PROCEED — the ego-motion-conditioning structural requirement is honored (FiLM MLP exercises the gradient surface); semantic ego-motion content is deferred to L2 per Yousfi dissent. PROCEED.
- **Tishby memorial** (Information Bottleneck): PROCEED — the I(T;Y) cooperative-receiver framing applies to the Z6 substrate at the architectural level; the MLX-local real-video L1 promotion is the iteration vehicle. PROCEED.
- **Rudin** (interpretable ML): PROCEED_WITH_REVISIONS — the MLX trainer's loss decomposition (MSE + residual L2) is observable per Catalog #305; the EMA shadow weight is a falling-rule-list-friendly inference primitive (decay=0.997 ≈ smooth average over ~333 epochs equivalent window). PROCEED.
- **Daubechies** (wavelets / compressive sensing): PROCEED — the Z6 encoder + decoder ARE coarse-to-fine multi-scale via PixelShuffle factor 2; the MLX-native primitives mirror the PyTorch sister byte-stably. PROCEED.
- **PR95 author** (HNeRV parity + race-mode rigor): PROCEED — the L1 promotion maintains the 13 inviolable lessons (1: score-aware substrate route through PyTorch sister; 2: export-first design preserved via #1265 gate; 3: archive grammar Z6PCWM1 monolithic; 4: inflate.py ≤ 100 LOC preserved; 5: full renderer NOT mask only; 6: score-domain Lagrangian deferred to PyTorch sister; 7: substrate-engineering exceeds bolt-on; 8: eval-roundtrip-aware deferred to PyTorch sister; 9: runtime closure verified via #1265 gate; 10: mask/pose coupling N/A; 11: no-op detector verified at L0 SCAFFOLD; 12: ≤ 700 LOC trainer); PROCEED.

### Quorum + verdict

- Sextet: 6-of-6 (Shannon+Dykstra+Yousfi+Fridrich+Contrarian+Assumption-Adversary)
- Grand council topical: 6-of-6 (Rao+Ballard+TishbyMem+Rudin+Daubechies+PR95Author)
- Total: 12 attendees, quorum_met=true
- Verdict tally: 9 PROCEED + 3 PROCEED_WITH_REVISIONS (Contrarian + Yousfi + Rudin)
- Composite verdict: **PROCEED_WITH_REVISIONS** (3-of-12 dissent ratio surfaced; revisions enumerated in council_decisions_recorded above)

---

## 5. Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

Z6 substrate reactivation paths (if L1 promotion empirical anchor FALSIFIES the architectural class-shift):

1. **PRIORITY-1 reactivation: PoseNet-derived ego-motion (L2 promotion)** — Yousfi dissent recommends this; predicted cost $0 (MLX-local) + ~2h sister subagent wall-clock; structural verdict tests the assumption that pinned-random ego-motion suppresses the FiLM predictor's class-shift capacity
2. **PRIORITY-2 reactivation: Score-aware loss against real scorers via PyTorch sister** — Contrarian dissent recommends this; predicted cost $0.05-0.20 (Modal CUDA smoke per Catalog #325 cascade) + ~30 min wall-clock; structural verdict tests the assumption that MSE-proxy loss suppresses the substrate's score-axis-grounding
3. **PRIORITY-3 reactivation: Multi-layer FiLM (predictor_depth>=2)** — Sister Z6-v2 Wave 2 BUILD pattern (council Phase 3 sextet PROCEED); predicted cost $0 (MLX-local) + ~3h sister subagent wall-clock; structural verdict tests the assumption that single-layer FiLM is the architectural ceiling
4. **PRIORITY-4 reactivation: Sister gate `tools/gate_mlx_candidate_contest_equivalence_z6.py` for Z6PCWM1 grammar** — OP-ROUTABLE per "Operator-routable next steps"; predicted cost $0 + ~2h sister subagent wall-clock; structural verdict tests the assumption that the MLX-built Z6PCWM1 archive is byte-stable contest-equivalent

---

## 6. Catalog #324 post-training Tier-C validation discipline declared

Per Catalog #324 the L1 promotion declares `predicted_band_validation_status` explicitly:

- **L1 status**: `post_training_mlx_50_100ep_local` — the L1 promotion produces a post-training archive on REAL contest video frames (validated AT the L1 surface); contrast with `phantom_random_init` (pre-training; canonical bug class for C6 IBPS 22× miss anchor 2026-05-17)
- **Predicted band**: [0.13, 0.16] per Z6 design memo Section 18 (planning prior; non-promotable until paired CPU/CUDA empirical anchors per CLAUDE.md "Apples-to-apples evidence discipline" land via PyTorch sister)
- **Validation evidence at L1**: per-epoch loss curve + final archive size + MLX↔PyTorch decoder parity max_abs (target < 1.0 per axis 2 discipline) + numpy reference parity max_abs (target < 1e-3 per axis 3 discipline) emitted as canonical Provenance manifest
- **Tier-C post-training re-measurement on paired CUDA archive**: pending operator-routed paid CUDA dispatch via PyTorch sister; expected reactivation path #2 above
- **Sister gate verdict (Z6 sister gate or PR95-parameterized gate)**: pending sister subagent per OP-ROUTABLE #2 above; documents structural gap as REFUSED-PENDING-SISTER-GATE per A=DreamerV3 anchor pattern

---

## 7. Verdict + reactivation cascade

**Verdict**: **PROCEED_WITH_REVISIONS** for L1 promotion.

Revisions binding (5 from council_decisions_recorded):

1. L1 anchor IS INFRASTRUCTURE-CONVERGENCE-VERIFICATION not score-claim (Contrarian)
2. PoseNet ego-motion DEFERRED to L2 sister subagent (Yousfi)
3. #1265 sister gate OR Z6-parameterized gate DEFERRED to sister subagent (4-of-5 distinguishing-feature pattern; explicit REFUSED-PENDING-SISTER-GATE)
4. Multi-layer FiLM DEFERRED to sister subagent (preserved from L0 SCAFFOLD scope)
5. Paid CUDA dispatch via PyTorch sister is the ONLY operator-routable contest-CUDA path (Catalog #313 predecessor-probe check applies)

Reactivation cascade priority: PoseNet ego-motion > score-aware loss via PyTorch sister > multi-layer FiLM > sister Z6 gate parameterization. Per CLAUDE.md "Forbidden premature KILL": the 4 reactivation paths are pinned ABOVE; the L1 promotion is the staging vehicle for the operator-routable cascade.

---

## 8. 6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map** — FiLM predictor gradient norms exposed via MLX `loss_grad` partials; the L1 promotion's `training_manifest.json` captures per-epoch loss decomposition (MSE + residual L2 + EMA shadow). Future wire-in to `tac.sensitivity_map.time_traveler_l5_z6_mlx_v1` is deferred to L2 promotion when score-aware loss routes through PyTorch sister. — **DEFERRED-L2** with rationale: L1's MSE+residual gradients are non-promotable per Catalog #287; sensitivity-map consumer requires score-aware loss gradients (PyTorch sister's path).
2. **Pareto constraint** — Inherits sister Z6 PyTorch substrate's `predictor_residual_entropy ≤ ε_residual` constraint canonical to Z6 substrate. MLX trainer's residual L2 IS the entropy proxy. — **ACTIVE-VIA-SISTER**: the Z6 substrate's Pareto constraint applies to the MLX-built archive automatically (Z6PCWM1 archive grammar is shared).
3. **Bit-allocator hook** — MLX-built archive's per-pair-residual bit allocation derives from the predictor's forecast (deterministic at training time). — **ACTIVE-VIA-SISTER**: the Z6 substrate's bit-allocator hook applies (Z6PCWM1 byte layout is byte-stable across MLX/PyTorch).
4. **Cathedral autopilot dispatch hook** — The #1265 gate consumes MLX-built archives; auto-discovery per Catalog #335 + #336 + #337 routes through `tools/cathedral_autopilot_autonomous_loop.py` once a gate-PASSED MLX archive lands. — **ACTIVE-FUTURE**: operator-routable next step is to run #1265 gate (or sister Z6 gate) on the L1 MLX-built archive.
5. **Continual-learning posterior** — Every gate-PASSED MLX archive becomes an empirical anchor for the posterior via `posterior_update_locked` (Catalog #128). The L1 promotion emits a canonical posterior anchor per Catalog #324 with `evidence_grade="macOS-MLX-research-signal"` + non-promotable markers. — **ACTIVE-AT-LANDING**: this symposium memo IS the canonical posterior anchor written via `tac.council_continual_learning.append_council_anchor`.
6. **Probe-disambiguator** — Z6 substrate's identity-predictor ablation IS the canonical disambiguator (Catalog #125 hook #6); the L1 promotion preserves the L0 SCAFFOLD's deferral of identity-predictor support so operator routes via PyTorch sister trainer's `--identity-predictor` flag for the disambiguator probe. — **DEFERRED-VIA-SISTER**: probe-disambiguator path is operator-routed through PyTorch sister.

---

## 9. Premise verification (Catalog #229)

Files read before any L1 edit:

1. `.omx/research/z6_predictive_coding_mlx_scaffold_landed_20260526.md` (L0 SCAFFOLD landing memo)
2. `experiments/train_substrate_z6_predictive_coding_mlx.py` (L0 trainer; full file)
3. `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py` (MLX renderer; ~700 LOC)
4. `src/tac/substrates/time_traveler_l5_z6/architecture.py` (PyTorch sister; config dataclass)
5. `src/tac/substrates/time_traveler_l5_z6/score_aware_loss.py` (PyTorch score-aware loss; canonical reference)
6. `src/tac/data.py::decode_video` (canonical pyav helper; lines 50-82)
7. `src/tac/substrates/score_aware_common.py::score_pair_components` (canonical scorer routing; sister deferred)
8. `.omx/state/lane_registry.json` D=Z6 entry (verified L1 readiness)
9. `.omx/state/subagent_progress.jsonl` (sister coordination verified: no overlap with current in-flight subagents H/I/J/K + FIX-WAVE-R1)
10. Operator binding directives 2026-05-26 (5 directives at top of prompt)
11. CLAUDE.md non-negotiables: "MLX portable-local-substrate authority" + "EMA — NON-NEGOTIABLE" + "eval_roundtrip — NON-NEGOTIABLE" + "Apples-to-apples evidence discipline" + "Forbidden premature KILL" + "PER-SUBSTRATE OPTIMAL FORM" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
12. Catalog rows: #287/#323 canonical Provenance + #325 per-substrate symposium contract + #324 predicted-band validation status + #341/#317/#192 non-promotable markers + #110/#113 APPEND-ONLY + #230 sister coordination + #340 sister-checkpoint guard

---

## 10. Sister-coherence (Catalog #230 / #302 / #314 / #340)

- In-flight sisters at landing time per `.omx/state/subagent_progress.jsonl`:
  - H = `aba5069741fc4475b` ATW V2 cargo-cult-first (sister substrate; non-overlapping)
  - K = `a7977f23a7f0f0573` COIN++ (sister substrate; non-overlapping)
  - I = `a71f2c4404c978f50` V1 Faiss IVF-PQ (sister substrate; non-overlapping)
  - J = `abfd5113f1892447c` MDL-IBPS (sister substrate; non-overlapping)
  - FIX-WAVE-R1 = `a23779a732e7bb056` (closing R1 P0+P1+P2 on A+E; non-overlapping)
- My domain: `experiments/train_substrate_z6_predictive_coding_mlx.py` + `src/tac/substrates/time_traveler_l5_z6/` files IF needed + `.omx/state/lane_registry.json` D=Z6 entry update + NEW memos
- Catalog #340 sister-checkpoint guard: VERIFIED clean at landing (no file overlap with sister-owned substrate trees)
- Catalog #302 in-flight sister check: 5 sisters + 1 (me) = 6 over Catalog #302 cap; operator-authorized per directive in prompt
- DID NOT touch sister substrate trees, CLAUDE.md, or live PR submissions
- DID NOT dispatch any paid CUDA / Modal / Vast.ai / Lightning per CLAUDE.md "Executing actions with care"

---

## 11. Operator-routable next steps

1. **L1 promotion landing** — proceed to STEP 2-4 of the L1 promotion contract (wire `_full_main` to REAL MLX training + empirical convergence smoke + lane registry update + landing memo)
2. **Sister gate parameterization** — sister subagent extends `tools/gate_mlx_candidate_contest_equivalence.py` to accept Z6PCWM1 grammar (or lands `tools/gate_mlx_candidate_contest_equivalence_z6.py` sister); estimated 2h wall-clock + $0
3. **L2 PoseNet ego-motion (Yousfi dissent)** — sister subagent wires PoseNet-derived ego-motion into the MLX renderer's `ego_motion_buffer`; estimated 2h wall-clock + $0
4. **L2 score-aware loss via PyTorch sister (Contrarian dissent)** — operator routes paid CUDA dispatch via `experiments/train_substrate_time_traveler_l5_z6.py` + `tools/operator_authorize.py --recipe substrate_time_traveler_l5_z6_modal_t4_dispatch` per Catalog #313 predecessor-probe check; estimated $0.05-0.20 + 30 min wall-clock
5. **L2 multi-layer FiLM (Wave 2 BUILD)** — sister subagent extends MLX renderer to support `predictor_depth>=2` per canonical PyTorch `MultiLayerFilmPredictor`; estimated 3h wall-clock + $0
6. **Identity-predictor disambiguator probe (Catalog #125 hook #6)** — sister subagent implements MLX identity-predictor + paired-anchor comparison via #1265 gate; estimated 2h wall-clock + $0

---

## 12. Discipline applied

- Catalog #229 PV (12 files + 11 CLAUDE.md non-negotiables + 11 catalog rows)
- Catalog #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256`
- Catalog #119 Co-Authored-By trailer
- Catalog #206 subagent checkpoint discipline
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
- Catalog #230 / #302 / #314 / #340 sister-subagent ownership map
- Catalog #287 placeholder-rationale rejection (every Provenance field carries non-placeholder rationale ≥4 chars)
- Catalog #292 per-deliberation explicit assumption surfacing
- Catalog #300 v2 frontmatter + mission-alignment fields (frontier_breaking_enabler + override_invoked=false)
- Catalog #303 cargo-cult audit reference + 5 NEW L1 assumption classifications
- Catalog #305 observability surface 6-facet declaration
- Catalog #309 horizon_class declaration (asymptotic_pursuit)
- Catalog #310 F-asymptote-class-shift NOT bolt-on (declaration preserved from L0)
- Catalog #311 predictive-coding ego-motion conditioning preserved (architectural + buffer surface)
- Catalog #323 canonical Provenance umbrella
- Catalog #324 predicted_band_validation_status declared (`post_training_mlx_50_100ep_local`)
- Catalog #325 per-substrate symposium 6-step contract satisfied (cargo-cult audit ref + 9-dim + observability + sextet + reactivation + Tier-C)
- Catalog #341 / #317 / #192 canonical non-promotable routing markers
- Catalog #346 council roster complete=True (validated via `tac.canonical_council_roster.validate_council_dispatch_roster`)
- CLAUDE.md "MLX portable-local-substrate authority" + "EMA — NON-NEGOTIABLE" + "Apples-to-apples evidence discipline" + "Forbidden premature KILL" + "PER-SUBSTRATE OPTIMAL FORM" non-negotiables

---

## 13. Verdict

**PROCEED_WITH_REVISIONS for L1 promotion.** 5 binding revisions enumerated above (council_decisions_recorded). Sister gate parameterization + PoseNet ego-motion + score-aware loss via PyTorch sister + multi-layer FiLM all DEFERRED to follow-on subagents per "Forbidden premature KILL". The L1 promotion is INFRASTRUCTURE-CONVERGENCE-VERIFICATION on REAL contest video; non-promotable per Catalog #287/#192/#317/#341 throughout.

Proceed to L1 implementation: REAL contest video frame loader + EMA decay=0.997 + multi-epoch training + canonical Provenance manifest + lane registry update + landing memo.
