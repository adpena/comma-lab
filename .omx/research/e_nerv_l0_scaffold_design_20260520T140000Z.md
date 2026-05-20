# e_nerv L0 SCAFFOLD design memo

horizon_class: plateau_adjacent

substrate_id: e_nerv
lane_id: lane_e_nerv_l0_scaffold_20260520
trainer: experiments/train_substrate_e_nerv.py
substrate_library: src/tac/e_nerv_as_renderer.py
recipe: .omx/operator_authorize_recipes/substrate_e_nerv_modal_a10g_diagnostic_dispatch.yaml
driver: scripts/remote_lane_substrate_e_nerv_l0_scaffold.sh
posture: L0 SCAFFOLD (research_only=true; dispatch_enabled=false)
generated_at_utc: 2026-05-20T14:00:00Z
predecessor_anchor: feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md

## Mission

e_nerv is the E-NeRV (Efficient NeRV; Li et al. NeurIPS 2022 arXiv:2207.08132)
substrate in the BUILD-1 2026-05-20 NeRV-trio queue fill. It mirrors the
canonical tc_nerv trainer skeleton but substitutes a compress-time encoder
(`ENeRVEncoder`) for the temporal-consistency regularizer. The encoder maps
`(B, 2, 3, H, W)` frame pairs to a latent space; only the decoder + per-pair
latents ship in the archive (encoder is COMPRESS-TIME ONLY per strict-scorer-
rule discipline). Hypothesis: a structured encoder learns a more compact
latent space than the pure-table approach, reducing latent_blob bytes at
fixed reconstruction quality [predicted; literature derivative].

L0 SCAFFOLD posture: smoke runs on CPU; full path raises NotImplementedError
per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + Catalog #220 + #240 + #315 + #325. Phase 2 council symposium
per Catalog #325 is the unlock for the full-mode + paid dispatch path.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
+ Catalog #290 self-protection:

| Layer | Decision | Rationale (falling-rule) |
| --- | --- | --- |
| Trainer skeleton (`tac.substrates._shared.trainer_skeleton`) | ADOPT canonical | OBVIOUS-FIT: substrate-agnostic helpers. |
| Substrate library (`tac.e_nerv_as_renderer`) | FORK from canonical `tac.substrates/<id>/` pattern | PRINCIPLED: predates canonical package convention (2026-05-11 legacy monolithic format). Phase 2 migration is the canonical-package-conversion reactivation gate. |
| EMA (`tac.training.EMA(decay=0.997)`) | ADOPT canonical (Phase 2) | OBVIOUS-FIT: CLAUDE.md non-negotiable applies universally. |
| eval_roundtrip (`patch_upstream_yuv6_globally`) | ADOPT canonical | PRINCIPLED: CLAUDE.md non-negotiable; FORK would re-introduce proxy-auth gap. |
| Scorer load (`load_differentiable_scorers`) | ADOPT canonical (Phase 2) | OBVIOUS-FIT: PR #95/#106 contract. |
| Score-aware Lagrangian | ADOPT canonical `score_pair_components` (Phase 2) | PRINCIPLED: Catalog #164. |
| Auth-eval invocation | ADOPT canonical `gate_auth_eval_call` (Phase 2) | PRINCIPLED: Catalog #226. |
| Inflate device selection | ADOPT canonical `select_inflate_device` (Phase 2) | PRINCIPLED: Catalog #205 + F1/F11 CPU-CUDA gap anchor. |
| ENeRVEncoder architecture (compress-time only) | UNIQUE per substrate | PRINCIPLED: encoder is e_nerv's distinguishing feature; sister NeRV variants do not ship an encoder. Stride-2 conv stack + GroupNorm + global-avg-pool + linear head per the canonical NeRV encoder pattern. |
| Latent quantization (uint8 asym-delta in archive) | ADOPT canonical | PRINCIPLED: per the substrate library's existing `_quantize_latent_table_uint8_delta_split`. Sister substrates share this. |
| Archive grammar (format ID 0x65) | UNIQUE per substrate | PRINCIPLED: 4 sections (no pose_table); sister substrates have different section layouts. Per HNeRV parity L3 monolithic single-file 0.bin convention. |
| Inflate runtime (≤200 LOC budget) | ADOPT canonical (Phase 2) | OBVIOUS-FIT: HNeRV parity L4. |

## 9-dimension success checklist evidence

Per Catalog #294 self-protection:

1. **UNIQUENESS**: e_nerv distinguishes itself from sister NeRV variants
   via the explicit compress-time encoder (`ENeRVEncoder`). No sister
   substrate has an encoder-decoder split shipped at compress-time. Per
   Li et al. NeurIPS 2022 arXiv:2207.08132 the encoder enables more
   efficient latent allocation.

2. **BEAUTY + ELEGANCE**: scaffold trainer is ~590 LOC (smaller than
   tc_nerv 1287 LOC). 30-second reviewable per PR101 standard.

3. **DISTINCTNESS**: distinguishing-feature integration contract (Catalog
   #272 4-fields) declared in SubstrateContract:
   - `distinguishing_feature_name`: compress_time_enerv_encoder
   - `distinguishing_bytes_path`: archive section `latent_blob` (uint8
     asym-delta, n_pairs × latent_dim bytes); the encoder weights are
     NOT in the archive (compress-time only per strict-scorer-rule).
   - `inflate_consumer_function`: `ENeRVRenderer.forward(z)` consumed
     from latent_blob at inflate time; encoder is dormant at inflate.
   - `byte_mutation_smoke_passes`: PENDING Phase 2 + canonical inflate
     runtime landing.

4. **RIGOR**: scaffold honors CLAUDE.md non-negotiables (eval_roundtrip
   patch, no MPS, --smoke gates synthetic batches, encoder explicitly
   tagged compress-time-only-not-in-archive). All 8 declared catalog
   compliance tags. Premise verification: smoke end-to-end runs on CPU
   and produces a non-trivial loss; encoder forward verified (verified
   2026-05-20).

5. **OPTIMIZATION-PER-TECHNIQUE** (Dim 5; Catalog #290): see table above.

6. **STACK-OF-STACKS-COMPOSABILITY**: e_nerv is composable with A1 as
   sidecar. composition_alpha unmeasured.

7. **DETERMINISTIC-REPRODUCIBILITY**: seed pinning via canonical
   `pin_seeds`; archive bytes deterministic from EMA shadow.

8. **EXTREME-OPTIMIZATION-PERFORMANCE**: Tier-1 engineering primitives
   declared via argparse per Catalog #151 manifest; activation deferred
   to Phase 2.

9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: NO empirical anchor at L0 SCAFFOLD.
   Per Catalog #324 the predicted_band is omitted entirely until Phase 2
   council symposium + post-training Tier-C validation.

## Cargo-cult audit per assumption

Per Catalog #303 self-protection:

| # | Assumption | Classification | Unwind path |
| --- | --- | --- | --- |
| 1 | An explicit encoder learns a more compact latent than per-pair-table | CARGO-CULTED | Empirical test: train e_nerv vs sane_hnerv at matched param/byte budget; measure latent_blob bytes + reconstruction quality. Pending Phase 2. |
| 2 | Encoder stride-2 conv stack is the right inductive bias | HARD-EARNED | Canonical NeRV encoder pattern (Li et al.); validated in sister NeRV-family literature. Preserve. |
| 3 | Encoder is COMPRESS-TIME-ONLY (not in archive) | HARD-EARNED | strict-scorer-rule discipline + HNeRV parity L5: the renderer at inflate time must produce RGB from the latent table alone. Forbidden to ship the encoder. |
| 4 | Latent uint8 asym-delta split is optimal quantization | CARGO-CULTED | Inherited from sister NeRV variants. Unwind path: measure rate-distortion at Phase 2; alternatives include uint16 or arithmetic-coded. |
| 5 | Decoder channel ladder (base_channels=36) is optimal | CARGO-CULTED | Inherited from sister NeRV variants; sweep at Phase 2. |
| 6 | E-NeRV's efficiency gains generalize from synthetic video to driving | CARGO-CULTED | Original E-NeRV paper used UVG dataset; driving video has different spatial statistics. Pending Phase 2 empirical anchor. |
| 7 | EMA decay 0.997 + shadow at inference | HARD-EARNED | Universal across substrates per CLAUDE.md. Preserve. |

## Observability surface

Per Catalog #305 self-protection:

1. **Inspectable per layer**: `ENeRVRenderer.schema` exposes state_dict
   keys + shapes; `ENeRVEncoder` exposes intermediate feature maps via
   the canonical NeRV encoder pattern.

2. **Decomposable per signal**: Phase 2 full path produces seg/pose/rate
   decomposition. Scaffold only surrogate `decoded.abs().mean()`.

3. **Diff-able across runs**: seed-pinned (--seed 20260520).

4. **Queryable post-hoc**: provenance.json schema
   `e_nerv_l0_scaffold_smoke_v1` includes `n_encoder_params_compress_time_only`
   (machine-readable) so any future audit can verify the encoder is NOT
   counted toward archive bytes.

5. **Cite-able**: every run anchored to (commit, substrate, call_id, config).

6. **Counterfactual-able**: PENDING Phase 2 byte-mutation smoke.

## Predicted ΔS band

L0 SCAFFOLD predicted_band EXPLICITLY UNSET per Catalog #324. No Dykstra-
feasibility check needed at scaffold posture per Catalog #296.

# PREDICTED_BAND_VIBES_OK:no_predicted_band_at_L0_scaffold_per_catalog_324_pending_phase_2_council

## Reactivation criteria for full path

1. Per-substrate adversarial grand council symposium per Catalog #325
   returns PROCEED or PROCEED_WITH_REVISIONS for e_nerv.
2. Canonical `src/tac/substrates/e_nerv/` package migration lands.
3. `submissions/e_nerv_substrate/inflate.{py,sh}` smoke proves contest-
   compliant 3-positional-arg runtime per Catalog #146.
4. This memo's 5 binding sections all preserved/refreshed.
5. Operator-frontier-override or Phase 2 approval flips `research_only`
   to false.

## Cross-references

- `src/tac/e_nerv_as_renderer.py` (substrate library)
- `experiments/train_substrate_tc_nerv.py` (canonical sister)
- `experiments/train_e_nerv_as_renderer.py` (pre-canonical scaffold)
- `.omx/operator_authorize_recipes/substrate_e_nerv_modal_a10g_diagnostic_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_e_nerv_modal_a100_dispatch.yaml` (legacy)
- `scripts/remote_lane_substrate_e_nerv_l0_scaffold.sh`
- `scripts/remote_lane_substrate_e_nerv.sh` (legacy)
- `feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md`
- `feedback_slot_build_1_nerv_trio_scaffolds_landed_20260520.md`
