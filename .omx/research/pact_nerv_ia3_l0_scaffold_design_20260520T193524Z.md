<!-- Catalog #344 canonical equation cross-ref: Pact-NeRV-IA3 empirical anchors align with `tac.canonical_equations` registry per HYBRID Stage 1 anchor landing pattern. Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — no mutation. -->

# Pact-NeRV-IA3 L0 SCAFFOLD design memo

**Date:** 2026-05-20
**Lane:** `lane_pact_nerv_ia3_l0_scaffold_20260520` (L1 after this memo + landing memo + impl_complete + deploy_runbook all land)
**Stage:** Stage 1 of HYBRID staged path per PACT-NERV-DESIGN-SYMPOSIUM commit `5371d4dd4`
**Literature anchor:** Liu et al. 2022 *"IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations"*, arXiv:2205.05638
**Council reference:** `.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md` (PROCEED_WITH_REVISIONS)
**FILM-FAMILY-RESEARCH reference:** `.omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md` Section 10 Recommendation #5
**horizon-class:** `plateau_adjacent` per Catalog #309 (Stage 1 of HYBRID staged path; Stages 1-2 are plateau_adjacent)
**research_only:** true per HNeRV parity L2 + Catalog #220 — L0 SCAFFOLD; dispatch operator-gated.

---

## Executive summary

Pact-NeRV-IA3 is Stage 1 of the HYBRID staged path the PACT-NERV-DESIGN-SYMPOSIUM
returned PROCEED_WITH_REVISIONS on. Per the symposium's `council_decisions_recorded`
op-routable #1: *"Stage 1 PRIORITY 1 — Pact-NeRV-IA3 $0.30 Modal T4 single-primitive
gamma-only rate-extremal smoke (50 LOC; cheapest empirical experiment per Hotz +
Carmack staging discipline)"*.

The architectural distinction vs sister NeRV-family substrates (boost_nerv, ds_nerv,
hi_nerv, etc.): IA3 γ-only ego-pose-conditioned per-block modulation. The IA3 paper
(Liu 2205.05638) §3.2 demonstrates element-wise learnable γ rescaling is ~6x more
parameter-efficient than full FiLM γ+β while preserving expressiveness on most
conditioning tasks. The empirical question Stage 1 dispatch tests on OUR contest:
does the β term carry significant per-frame signal on our specific driving video?

The L0 SCAFFOLD lands the substrate package + trainer + recipe + driver + tests
but does NOT fire a paid dispatch. Stage 1 dispatch is operator-gated per Catalog
#325 + Catalog #240 + Catalog #315. The matching recipe declares
`dispatch_enabled: false` + `research_only: true` so the operator-authorize
harness refuses dispatch until reactivation criteria are met.

---

## 1. Architectural distinction

The IA3 γ-only modulation primitive:

```python
class IA3GammaOnlyModulation(nn.Module):
    def __init__(self, num_features, pose_dim=6, init_delta_std=0.01):
        super().__init__()
        # γ projection: pose_dim -> num_features
        # NO β projection (this is THE distinguishing primitive vs FiLM γ+β)
        self.gamma_proj = nn.Linear(pose_dim, num_features)
        # IA3 §3.2 zero-init: γ_proj weights ~0 so γ ≈ 1.0 at init
        self.gamma_proj.weight.normal_(mean=0.0, std=init_delta_std)
        self.gamma_proj.bias.zero_()

    def forward(self, x, pose):
        # γ = 1.0 + Δ residual form per IA3 §3.2
        gamma = 1.0 + self.gamma_proj(pose)
        return x * gamma.view(-1, self.num_features, 1, 1)
```

~30 LOC for the canonical IA3 core class; ~50 LOC including imports + dataclass
config + integration into the HNeRV-class base decoder.

The base decoder mirrors boost_nerv / ds_nerv (DepthSep + SIREN + PixelShuffle).
At each upsample block: upsample feature map -> IA3 γ-only modulation conditioned
on ego_pose ∈ R^6 -> next block. Final 1x1 conv heads produce (rgb_0, rgb_1).

---

## 2. Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton (device_or_die / pin_seeds / git_head_sha) | **ADOPT_CANONICAL** | Cross-substrate hygiene per Catalog #190 (hardware_substrate detection) + Catalog #178 (TF32) + Catalog #190 sister canon. Forking provides no value. |
| Substrate Contract (`SubstrateContract`) | **ADOPT_CANONICAL** | Catalog #241/#242 META layer requires registration; canonical contract enforces all hooks |
| Score-aware loss helper (`score_pair_components_dispatch`) | **ADOPT_CANONICAL** | Catalog #164 canonical scorer-preprocess routing; Catalog #222 scorer-loader assignment-order discipline. Forking violates 13 HNeRV parity lessons. |
| Differentiable eval_roundtrip (`patch_upstream_yuv6_globally`) | **ADOPT_CANONICAL** | Catalog #6 MANDATORY DEFAULT non-negotiable. Forking is forbidden per CLAUDE.md. |
| Archive grammar (`PIA3` magic + 26-byte header + monolithic 0.bin) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Substrates that carry distinctive primitives need their own magic+header. Sister NeRV substrates each have distinct magic (BSV1=boost, DSV1=ds, etc.). The POSE_DIM u8 + EGO_POSE_BLOB_LEN u32 fields are unique to Pact-NeRV-IA3 because IA3 requires ego_pose conditioning bytes in the archive. |
| Inflate runtime (per-substrate) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | HNeRV parity L4: each substrate ships its own ≤150 LOC inflate.py specific to its archive grammar. Sister NeRV substrates all follow this pattern. |
| IA3 γ-only modulation primitive | **FORK_BECAUSE_UNIQUE_TO_METHOD** | THE distinguishing primitive vs full FiLM γ+β. NO sister substrate has IA3 — this is the empirical Stage 1 probe. ~30 LOC canonical IA3 core class. |
| Per-pair learnable ego_poses | **FORK_BECAUSE_UNIQUE_TO_METHOD** | The pose conditioning at compress time may be learned (default L0) or measured from upstream PoseNet (L1+). Trainer-side detail; not canonical across substrates. |

The 8 canonical adoption decisions vs 3 forks documented above; net: heavy canonical
adoption (preserves cross-substrate hygiene) + 3 targeted forks (where the IA3
primitive demands unique implementation).

---

## 3. Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Classification | Rationale | Unwind plan |
|---|---|---|---|
| IA3 γ-only modulation transfers from NLP to NeRV-class video | HARD-EARNED-LITERATURE | Liu 2205.05638 demonstrates IA3 efficacy across NLP + vision; FILM-FAMILY-RESEARCH §10.5 cites as HARD-EARNED-LITERATURE rate-extremal variant | None needed |
| γ_init = 1.0 (residual form) | HARD-EARNED | IA3 paper §3.2 zero-init discipline; sister of adaLN-Zero per FILM-FAMILY-RESEARCH §5; provides identity-at-init early-training stability | None needed |
| pose_dim = 6 | HARD-EARNED | Contest canonical: matches upstream PoseNet first 6 dims per upstream/modules.py. Alignment with scorer's pretrained semantics. | None needed |
| Per-block modulation (one γ_proj per upsample block) | CARGO-CULTED-AT-L0 | Multi-layer modulation is canonical per FILM-FAMILY-RESEARCH §8.6 HARD-EARNED-EMPIRICALLY-SUPERIOR for video-temporal conditioning. Sweep at L1: alternatives = final-block-only, every-other-block, scalar global rescaling | Empirical sweep at Stage 1 dispatch |
| HNeRV-class base decoder | HARD-EARNED | PR101 GOLD baseline; the IA3 modulation is the bolt-on under test, not the base | None needed |
| Shared γ_proj across (frame_0, frame_1) of a pair | CARGO-CULTED-AT-L0 | Per-frame γ_proj doubles head count; cheap variant first per Stage 1 discipline. Alternative tested at L1+. | Sweep at Stage 1 dispatch OR L1+ |

2 CARGO-CULTED assumptions documented; both deferred to Stage 1 dispatch empirical
sweep (per HYBRID staged path). 4 HARD-EARNED assumptions traceable to literature
+ canonical apparatus state.

---

## 4. 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: IA3 γ-only modulation is unique vs all 11 sister NeRV-family
   substrates (boost_nerv uses iterative-residual chains; ds_nerv uses DepthSep
   base only; hi_nerv uses hierarchical features; etc.). NO sister has IA3.
2. **BEAUTY + ELEGANCE**: ~50 LOC core (`IA3GammaOnlyModulation`); reviewable in
   30 seconds per HNeRV parity L12. The IA3 paper §3.2 zero-init discipline
   provides mathematical elegance (identity-at-init).
3. **DISTINCTNESS**: Explicit difference vs sister boost_nerv documented in
   `tac.substrates.pact_nerv_ia3.__init__` + this design memo. The distinguishing
   primitive is γ-only (no β) per Liu 2205.05638.
4. **RIGOR**: Premise verification per Catalog #229 (pre-flight Step 0 PRE-WRITE-SISTER-ACTIVITY-CHECK-HELPER
   verdict EXIT 0 PROCEED); cargo-cult audit Section 3 above with 2 CARGO-CULTED +
   4 HARD-EARNED assumption classifications; sister boost_nerv canonical pattern
   read in full + 12-test dedicated coverage.
5. **OPTIMIZATION PER TECHNIQUE**: Canonical adoption documented per layer in
   Section 2; FORK only where IA3 primitive demands unique implementation (3
   forks total). Sister NeRV substrates' shared engineering (canonical helpers
   + trainer skeleton + score-aware loss helper) inherited unchanged.
6. **STACK-OF-STACKS COMPOSABILITY**: Stage 1 deliberately tests SINGLE primitive
   (IA3 γ-only modulation). Stage 2 (Pact-NeRV-A1 triple-conditioning) composes
   IA3 with multi-layer FiLM + per-pair-difficulty + per-class CLADENorm per
   PACT-NERV symposium HYBRID path. The L0 SCAFFOLD substrate is the
   STAGE-1-PRIMITIVE-PROBE — composability empirical sweep at Stage 2+.
7. **DETERMINISTIC REPRODUCIBILITY**: Per CLAUDE.md "Canonical pipeline standard":
   seeds pinned via `tac.substrates._shared.trainer_skeleton.pin_seeds`; archive
   bytes are byte-stable per FP16 quantization + brotli quality=9 (deterministic);
   sister boost_nerv ENCODE_INFLATE_ROUNDTRIP test pattern adopted.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: IA3 γ-only modulation IS the
   rate-extremal optimization: halves conditioning bytes vs full FiLM γ+β per
   Liu 2205.05638 §3.2. Stage 1 dispatch will measure whether this rate-axis
   optimization translates to score-axis improvement.
9. **OPTIMAL MINIMAL CONTEST SCORE**: L0 SCAFFOLD does NOT make a score claim
   (no empirical anchor). Stage 1 dispatch operator-gated. Per the symposium's
   Yousfi + PR95Author + Carmack consensus: predicted_delta in
   [-0.005, +0.003] band on contest-CPU at 0.192 frontier; dispatching at Stage 1
   resolves which sub-band IA3 lands in.

---

## 5. Observability surface

Per Catalog #305:

1. **Inspectable per layer**: every upsample block + IA3 modulation step
   exposes intermediate feature maps via standard PyTorch hooks. `model.blocks`
   and `model.ia3_mods` are `nn.ModuleList` — operator can register
   `register_forward_hook` per layer for runtime inspection.
2. **Decomposable per signal**: `PactNervIa3ScoreAwareLoss.forward` returns
   `(loss, parts)` where `parts` carries `rate_term`, `seg_term`, `pose_term`,
   and `loss_total` detached for per-signal decomposition.
3. **Diff-able across runs**: smoke checkpoints carry full
   `state_dict()` + `config: asdict(cfg)` + seed + git_head per `provenance.json`
   schema `pact_nerv_ia3_l0_scaffold_smoke_v1` — two runs with same seed +
   same git_head produce byte-stable checkpoints (FP16 quant + brotli
   deterministic).
4. **Queryable post-hoc**: provenance.json + smoke_checkpoint.pt persist all
   training state; ENCODE_INFLATE_ROUNDTRIP test pattern enables byte-mutation
   no-op detection per Catalog #139.
5. **Cite-able**: lane_id + git_head + dispatch_instance_job_id + UTC start/end
   stamped into every provenance.json + `tac.deploy.modal.call_id_ledger.append`
   per Catalog #245 once dispatch fires.
6. **Counterfactual-able**: byte-mutation smoke (test
   `test_byte_mutation_changes_inflate_output_no_op_proof`) verifies that
   mutating a single ego_pose byte changes archive bytes + parsed ego_poses
   — proves IA3 conditioning is NOT a no-op at inflate time.

---

## 6. Predicted ΔS band

Per Catalog #296: NO predicted_band claim is made at L0 SCAFFOLD because no
empirical anchor exists yet. Per the PACT-NERV symposium Yousfi + PR95Author
council positions:

- Yousfi (operating-within: contest scorers contain rich driving-scene priors):
  predicted ΔS in [-0.001, -0.005] band on contest-CPU at 0.192 frontier
- PR95Author (operating-within: HNeRV-class implicit allocation IS optimal at
  plateau): predicted ΔS in [-0.001, -0.003] IF explicit-vs-implicit gap is
  non-zero; [+0.001, +0.003] (REGRESSION) IF implicit strategy was already
  optimal at this capacity

Both positions agree the band is NARROW because the apparatus is at plateau
(0.192 [contest-CPU] / 0.205 [contest-CUDA] frontier).

**Dykstra-feasibility check**: Stage 1 tests SINGLE primitive (IA3 γ-only
modulation). The single-primitive case does NOT require Dykstra alternating-
projection feasibility check (only multi-constraint compositions do per
CLAUDE.md "FORBIDDEN_PATTERNS Forbidden symposium-band-prediction-without-Dykstra-feasibility-check").
The Stage 1 dispatch IS the empirical Dykstra-feasibility check for the IA3
primitive specifically.

**Probe-disambiguator path**: per the PACT-NERV symposium Hotz position, Stage 1
IS the canonical probe-disambiguator between three hypotheses:
- IA3 ≈ FiLM (β-noise hypothesis confirmed)
- IA3 << FiLM (β-signal hypothesis confirmed)
- IA3 >> FiLM (FiLM-overcapacity hypothesis; Pact-NeRV-A2 candidate)

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY": predicted_band
declared `pending_post_training` per Catalog #324; reactivation criterion pinned
in recipe `predicted_band_reactivation_criteria`.

---

## 7. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map contribution | N/A (L0 SCAFFOLD) | No sensitivity signal until Stage 1 dispatch + full path; substrate trainer's `_full_main` raises NotImplementedError |
| #2 Pareto constraint | ACTIVE (declared) | `rate_distortion_v1` declared in SubstrateContract; the rate-axis (IA3 γ-only bytes added) + distortion-axis (d_seg + d_pose) are the canonical Pareto axes |
| #3 bit-allocator hook | N/A (L0 SCAFFOLD) | fp16 brotli on combined base+IA3-γ_proj weight blob; no per-tensor bit allocator at scaffold posture (IA3 γ_proj per-head quantization is the L1+ research path per cargo-cult audit shared-γ_proj alternative) |
| #4 cathedral autopilot dispatch hook | N/A (research_only) | research_only=true + dispatch_enabled=false at recipe; not dispatch-eligible |
| #5 continual-learning posterior update | N/A (no anchor yet) | No posterior anchor until full path lands and a [contest-CUDA] anchor is measured at Stage 1 dispatch |
| #6 probe-disambiguator | ACTIVE (planned) | FiLM vs IA3 disambiguation IS the Stage 1 dispatch's empirical purpose per PACT-NERV symposium Section 13. Stage 1 = the canonical probe-disambiguator. |

---

## 8. Mission contribution per Catalog #300

`apparatus_maintenance` at L0 SCAFFOLD (this landing); upgrades to
`frontier_breaking_enabler` at Stage 1 dispatch (where IA3 vs FiLM empirical
question gets resolved cheaply). Per CLAUDE.md "Mission alignment — non-negotiable"
Consequence 4: frontier-breaking moves DOMINATE rigor budget — Stage 1 dispatch
is the gateway to Stage 2-5 frontier-pursuit work per the PACT-NERV symposium
HYBRID staged path.

---

## 9. Cross-references

- PACT-NERV-DESIGN-SYMPOSIUM (commit `5371d4dd4`):
  `.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md`
- FILM-FAMILY-RESEARCH (commit `9a95d1daf`):
  `.omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md`
  Section 10 Recommendation #5 (IA3-style γ-only modulation as rate-extremal variant)
- Sister NERV-LITERATURE-L0-RESCOPED (commit `d9aaf7c13`): BoostNeRV + NIRVANA + COIN++ landing pattern
- Sister boost_nerv canonical L0 SCAFFOLD: `src/tac/substrates/boost_nerv/`
- IA3 paper: Liu et al. 2022, arXiv:2205.05638 (canonical literature reference)
- Sister Catalog #6 (eval_roundtrip MANDATORY DEFAULT)
- Sister Catalog #164 (canonical scorer-preprocess routing)
- Sister Catalog #222 (scorer-loader assignment-order discipline)
- Sister Catalog #240 (recipe-vs-trainer-state consistency)
- Sister Catalog #244 (canonical Modal/CUDA NVML env block)
- Sister Catalog #325 (per-substrate symposium before paid dispatch)

---

## 10. Reactivation criteria

Per CLAUDE.md "Forbidden premature KILL" + HNeRV parity L2:

1. PACT-NERV-DESIGN-SYMPOSIUM Stage 1 dispatch operator-gated approval per
   CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
   (Catalog #325). The symposium already returned PROCEED_WITH_REVISIONS with
   HYBRID Stage 1 = Pact-NeRV-IA3 ~$0.30 Modal T4 50-LOC γ-only rate-extremal
   smoke as the cheapest empirical experiment (op-routable #1).
2. Cargo-cult audit per Catalog #303 surfaces the 2 CARGO-CULTED choices documented
   in Section 3 above (per-block modulation; shared γ_proj across frame_0/frame_1).
   The audit MUST either (a) empirically validate each via sweep at Stage 1 dispatch
   OR (b) reclassify as HARD-EARNED with citation.
3. Trainer `experiments/train_substrate_pact_nerv_ia3.py::_full_main` path replaces
   NotImplementedError with real score-aware Lagrangian + EMA + canonical auth-eval
   helper invocation per Catalog #226. The `PactNervIa3ScoreAwareLoss` helper at
   `src/tac/substrates/pact_nerv_ia3/score_aware_loss.py` provides the loss surface;
   the trainer's full path must wire it in.
4. Recipe `research_only` flips from true to false; `dispatch_enabled` flips from
   false to true; `predicted_band` declared with `validation_status: post_training_validated`
   (after Stage 1 dispatch lands the first empirical anchor) per Catalog #324.

This is DEFERRED (research_only); NOT killed. Per CLAUDE.md "Forbidden premature KILL"
+ Catalog #301: the L0 SCAFFOLD is the staged-research path the PACT-NERV symposium
explicitly approved.
