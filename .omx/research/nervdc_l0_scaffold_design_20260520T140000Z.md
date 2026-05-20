# nervdc L0 SCAFFOLD design memo

horizon_class: plateau_adjacent

substrate_id: nervdc
lane_id: lane_nervdc_l0_scaffold_20260520
trainer: experiments/train_substrate_nervdc.py
substrate_library: src/tac/nervdc_as_renderer.py
recipe: .omx/operator_authorize_recipes/substrate_nervdc_modal_a10g_diagnostic_dispatch.yaml
driver: scripts/remote_lane_substrate_nervdc_l0_scaffold.sh
posture: L0 SCAFFOLD (research_only=true; dispatch_enabled=false)
generated_at_utc: 2026-05-20T14:00:00Z
predecessor_anchor: feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md

## Mission

nervdc is the NeRV Decoder Conditioning substrate in the BUILD-1 2026-05-20
NeRV-trio queue fill. It mirrors the canonical tc_nerv trainer skeleton but
substitutes a decoder-internal `_CondSummary` module for the temporal-
consistency regularizer. The decoder receives a small conditioning summary
of the previously decoded frame at inflate time (the prev_frame is always
available because the per-pair decoder runs sequentially). Hypothesis:
prev-frame conditioning acts as a cheap autoregressive prior that improves
temporal consistency without any extra archive bytes (the _CondSummary
module weights are part of the decoder weight blob) [predicted; canonical-
NeRV-family derivative].

L0 SCAFFOLD posture: smoke runs on CPU; full path raises NotImplementedError
per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + Catalog #220 + #240 + #315 + #325. Phase 2 council symposium
per Catalog #325 is the unlock for the full-mode + paid dispatch path.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
+ Catalog #290 self-protection:

| Layer | Decision | Rationale (falling-rule) |
| --- | --- | --- |
| Trainer skeleton (`tac.substrates._shared.trainer_skeleton`) | ADOPT canonical | OBVIOUS-FIT. |
| Substrate library (`tac.nervdc_as_renderer`) | FORK from canonical `tac.substrates/<id>/` pattern | PRINCIPLED: predates canonical convention. |
| EMA (`tac.training.EMA(decay=0.997)`) | ADOPT canonical (Phase 2) | OBVIOUS-FIT. |
| eval_roundtrip | ADOPT canonical | PRINCIPLED. |
| Scorer load | ADOPT canonical (Phase 2) | OBVIOUS-FIT. |
| Score-aware Lagrangian | ADOPT canonical `score_pair_components` (Phase 2) | PRINCIPLED: Catalog #164. |
| Auth-eval invocation | ADOPT canonical `gate_auth_eval_call` (Phase 2) | PRINCIPLED: Catalog #226. |
| Inflate device selection | ADOPT canonical `select_inflate_device` (Phase 2) | PRINCIPLED: Catalog #205. |
| `_CondSummary` decoder-internal module | UNIQUE per substrate | PRINCIPLED: decoder-internal prev-frame conditioning is nervdc's distinguishing feature; sister substrates do not use sequential prev-frame conditioning. |
| Conditioning signal source (prev decoded frame) | UNIQUE per substrate | PRINCIPLED: the prev_frame is always available at inflate time because per-pair decoder runs sequentially. NO extra archive bytes; conditioning module weights are part of the decoder weight blob. |
| Sequential inflate runtime | UNIQUE per substrate | PRINCIPLED: nervdc requires the per-pair decoder run order to be sequential (not random-access). HNeRV parity L9 runtime closure MUST verify this fits the contest runtime wallclock budget; this is a substantive Phase 2 council question. |
| Archive grammar (format ID 0x66) | UNIQUE per substrate | PRINCIPLED: 4 sections (no extra section for conditioning); the _CondSummary module weights are folded into the decoder_blob. |
| Inflate runtime (≤200 LOC budget) | ADOPT canonical (Phase 2) | OBVIOUS-FIT. |

## 9-dimension success checklist evidence

Per Catalog #294 self-protection:

1. **UNIQUENESS**: nervdc distinguishes itself from sister NeRV variants
   via decoder-internal sequential prev-frame conditioning. No sister
   substrate uses sequential inflate-time conditioning.

2. **BEAUTY + ELEGANCE**: scaffold trainer is ~570 LOC. 30-second
   reviewable per PR101 standard.

3. **DISTINCTNESS**: distinguishing-feature integration contract
   (Catalog #272 4-fields) declared in SubstrateContract:
   - `distinguishing_feature_name`: decoder_internal_prev_frame_conditioning
   - `distinguishing_bytes_path`: `decoder_blob` (the `_CondSummary` module
     weights are folded into the decoder_blob; no extra archive section)
   - `inflate_consumer_function`: `NeRVdcRenderer.forward(z, prev_frame)`
     consumed sequentially per-pair at inflate time
   - `byte_mutation_smoke_passes`: PENDING Phase 2 + sequential-inflate
     runtime closure verification

4. **RIGOR**: scaffold honors CLAUDE.md non-negotiables; smoke uses
   uniform-gray prev_frame for shape verification (real sequential
   inflate is the Phase 2 reactivation gate). Premise verification: smoke
   end-to-end runs on CPU; forward shape verified (verified 2026-05-20).

5. **OPTIMIZATION-PER-TECHNIQUE** (Dim 5; Catalog #290): see table above.

6. **STACK-OF-STACKS-COMPOSABILITY**: nervdc has a SEQUENTIAL inflate
   dependency that limits parallel composition with sister substrates;
   Phase 2 council MUST evaluate whether composition with A1 is feasible
   given the sequential constraint.

7. **DETERMINISTIC-REPRODUCIBILITY**: seed pinning via canonical
   `pin_seeds`; archive bytes deterministic from EMA shadow. Sequential
   inflate is deterministic given a fixed initial prev_frame (typically
   uniform gray).

8. **EXTREME-OPTIMIZATION-PERFORMANCE**: Tier-1 engineering primitives
   declared via argparse per Catalog #151 manifest; activation deferred
   to Phase 2. Sequential inflate constraint may bottleneck total
   wallclock (HNeRV parity L9 runtime closure question).

9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: NO empirical anchor at L0 SCAFFOLD.
   Per Catalog #324 predicted_band omitted entirely until Phase 2 council.

## Cargo-cult audit per assumption

Per Catalog #303 self-protection:

| # | Assumption | Classification | Unwind path |
| --- | --- | --- | --- |
| 1 | Prev-frame conditioning improves temporal consistency without extra bytes | CARGO-CULTED | Empirical test: train nervdc with cond_dim=0 (collapse to canonical NeRV) vs cond_dim=8 vs cond_dim=16; measure score delta. Pending Phase 2. |
| 2 | Sequential inflate fits within the contest runtime wallclock budget | CARGO-CULTED | The contest scorer runs inflate per-video; sequential decoding of 600 pairs adds wallclock overhead. HNeRV parity L9 runtime closure verification required. Unwind path: time the sequential inflate locally before Phase 2 GPU dispatch. |
| 3 | `_CondSummary` (3→8 dim via 3-stride-4 conv + global pool) is right inductive bias | CARGO-CULTED | Architecture inherited from sister NeRV variants without per-method validation. Pending Phase 2 ablation. |
| 4 | Folding conditioning module weights into decoder_blob is the right archive format | HARD-EARNED | This preserves the no-extra-archive-bytes property which is the substrate's core hypothesis. Alternative (separate section) would defeat the purpose. |
| 5 | Decoder channel ladder (base_channels=36) is optimal | CARGO-CULTED | Inherited; sweep at Phase 2. |
| 6 | EMA decay 0.997 + shadow at inference | HARD-EARNED | Universal across substrates per CLAUDE.md. Preserve. |
| 7 | The "no extra bytes for conditioning" claim survives the rate-distortion analysis | CARGO-CULTED | The _CondSummary module weights ADD bytes to the decoder_blob even if no new section is added. Real question: do those extra weight bytes pay for themselves in score improvement? Pending Phase 2 paired comparison. |

## Observability surface

Per Catalog #305 self-protection:

1. **Inspectable per layer**: `NeRVdcRenderer` exposes the
   `_CondSummary` intermediate output per forward; `NeRVdcLatentTable`
   exposes per-pair latents.

2. **Decomposable per signal**: Phase 2 full path produces seg/pose/rate
   decomposition. Scaffold only surrogate `decoded.abs().mean()`.

3. **Diff-able across runs**: seed-pinned (--seed 20260520).

4. **Queryable post-hoc**: provenance.json schema
   `nervdc_l0_scaffold_smoke_v1` includes all argparse args + git_head +
   torch_version + n_params.

5. **Cite-able**: every run anchored to (commit, substrate, call_id, config).

6. **Counterfactual-able**: PENDING Phase 2 byte-mutation smoke + the
   special case of conditioning-disabled forward (cond_dim=0 ablation).

## Predicted ΔS band

L0 SCAFFOLD predicted_band EXPLICITLY UNSET per Catalog #324.

# PREDICTED_BAND_VIBES_OK:no_predicted_band_at_L0_scaffold_per_catalog_324_pending_phase_2_council

## Reactivation criteria for full path

1. Per-substrate adversarial grand council symposium per Catalog #325
   returns PROCEED or PROCEED_WITH_REVISIONS for nervdc. Symposium MUST
   specifically evaluate whether sequential-inflate-time conditioning is
   contest-permissible within the contest runtime budget per HNeRV parity
   discipline L9 runtime closure.
2. Canonical `src/tac/substrates/nervdc/` package migration lands.
3. `submissions/nervdc_substrate/inflate.{py,sh}` smoke proves contest-
   compliant 3-positional-arg runtime per Catalog #146 AND proves the
   sequential-inflate-time prev-frame conditioning fits within the contest
   runtime wallclock budget.
4. This memo's 5 binding sections all preserved/refreshed.
5. Operator-frontier-override or Phase 2 approval flips `research_only`
   to false.

## Cross-references

- `src/tac/nervdc_as_renderer.py` (substrate library)
- `experiments/train_substrate_tc_nerv.py` (canonical sister)
- `experiments/train_nervdc_as_renderer.py` (pre-canonical scaffold)
- `.omx/operator_authorize_recipes/substrate_nervdc_modal_a10g_diagnostic_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_nervdc_modal_a100_dispatch.yaml` (legacy)
- `scripts/remote_lane_substrate_nervdc_l0_scaffold.sh`
- `scripts/remote_lane_substrate_nervdc.sh` (legacy)
- `feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md`
- `feedback_slot_build_1_nerv_trio_scaffolds_landed_20260520.md`
