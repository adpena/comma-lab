# ego_nerv L0 SCAFFOLD design memo

horizon_class: plateau_adjacent

substrate_id: ego_nerv
lane_id: lane_ego_nerv_l0_scaffold_20260520
trainer: experiments/train_substrate_ego_nerv.py
substrate_library: src/tac/ego_nerv_as_renderer.py
recipe: .omx/operator_authorize_recipes/substrate_ego_nerv_modal_a10g_diagnostic_dispatch.yaml
driver: scripts/remote_lane_substrate_ego_nerv_l0_scaffold.sh
posture: L0 SCAFFOLD (research_only=true; dispatch_enabled=false)
generated_at_utc: 2026-05-20T14:00:00Z
predecessor_anchor: feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md

## Mission

ego_nerv is the egocentric pose-conditioned NeRV substrate in the BUILD-1
2026-05-20 NeRV-trio queue fill. It mirrors the canonical tc_nerv trainer
skeleton but substitutes a FiLM-modulated egocentric pose table for the
SIREN+PixelShuffle temporal-consistency regularizer. The hypothesis: driving
video is egomotion-dominated, so pose-conditioning enables ~1.25x param
efficiency at fixed bytes (Wang 2024 Ego-NeRV / Park 2023 driving-NeRF
[predicted; literature derivative]).

This memo documents the L0 SCAFFOLD posture: smoke runs on CPU; full path
raises NotImplementedError per CLAUDE.md "Substrate scaffolds MUST be
COMPLETE or RESEARCH-ONLY" non-negotiable + Catalog #220 + #240 + #315 +
#325. Phase 2 council symposium per Catalog #325 is the unlock for the
full-mode + paid dispatch path.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
+ Catalog #290 self-protection: every canonical helper / META layer field /
engineering pattern adoption MUST document the rationale per the falling-rule
list (EMPIRICAL > PRINCIPLED > UNCLEAR > OBVIOUS-FIT).

| Layer | Decision | Rationale (falling-rule) |
| --- | --- | --- |
| Trainer skeleton (`tac.substrates._shared.trainer_skeleton`) | ADOPT canonical | OBVIOUS-FIT: device_or_die / pin_seeds / detect_hardware_substrate / sha256 helpers are substrate-agnostic and serve all substrates equally. |
| Substrate library (`tac.ego_nerv_as_renderer`) | FORK from canonical `tac.substrates/<id>/` pattern | PRINCIPLED: ego_nerv library predates the canonical package convention (lives at `src/tac/ego_nerv_as_renderer.py` in legacy monolithic format from 2026-05-11). Phase 2 migration is the canonical-package-conversion reactivation gate. |
| EMA (`tac.training.EMA(decay=0.997)`) | ADOPT canonical | OBVIOUS-FIT: CLAUDE.md non-negotiable applies universally; ego_nerv has no principled reason to fork. Full path (post-Phase 2) WILL adopt the canonical EMA. |
| eval_roundtrip (`patch_upstream_yuv6_globally`) | ADOPT canonical | PRINCIPLED: per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE" non-negotiable; FORK would re-introduce the proxy-auth gap class. |
| Scorer load (`load_differentiable_scorers`) | ADOPT canonical | OBVIOUS-FIT: PR #95/#106 contract; FORK would break gradient reachability. |
| Score-aware Lagrangian | ADOPT canonical `score_pair_components` (Phase 2) | PRINCIPLED: per Catalog #164 sister discipline. Scaffold posture skips this; full path will adopt. |
| Auth-eval invocation | ADOPT canonical `gate_auth_eval_call` (Phase 2) | PRINCIPLED: per Catalog #226. Scaffold skips; full path adopts. |
| Inflate device selection | ADOPT canonical `select_inflate_device` (Phase 2) | PRINCIPLED: per Catalog #205 + the F1/F11 CPU-CUDA gap empirical anchor. |
| FiLM modulator design | UNIQUE per substrate | PRINCIPLED: FiLM is ego_nerv's distinguishing feature; canonical scorer-routing has no role to play here. Identity-init (scale=1, shift=0) is empirically validated by Lane 12 + KK's bolt-on. |
| Pose table format (fp16 raw in archive) | UNIQUE per substrate | PRINCIPLED: pose table is ego_nerv-specific archive payload; canonical brotli-int8 codec is wrong shape (pose values are continuous-floating-point not weight-int8). |
| Archive grammar (format ID 0x68) | UNIQUE per substrate | PRINCIPLED: ego_nerv has 5 sections incl. pose_table; sister substrates have different section counts. Per HNeRV parity discipline L3 monolithic single-file 0.bin convention adopted. |
| Inflate runtime (≤200 LOC budget) | ADOPT canonical (Phase 2) | OBVIOUS-FIT: HNeRV parity L4 budget waiver to 200 (vs default 100) follows the existing legacy waiver per the parent recipe's note. |

## 9-dimension success checklist evidence

Per Catalog #294 self-protection (operator standing directive: "uniqueness and
beautify and elegance and distinctness and rigor and optimization per
technique and stack of stacks while still deterministic reproducibility and
extreme optimization and performance and optimal minimal contest score"):

1. **UNIQUENESS**: ego_nerv distinguishes itself from sister NeRV variants
   (tc_nerv / hi_nerv / block_nerv / ff_nerv / ds_nerv / sane_hnerv) via
   the explicit per-pair pose table + FiLM modulation. No sister substrate
   ships per-pair pose conditioning in the archive. Architecture class:
   pose-conditioned-NeRV-family.

2. **BEAUTY + ELEGANCE**: scaffold trainer is ~570 LOC (smaller than
   tc_nerv 1287 LOC because the substrate library at
   `src/tac/ego_nerv_as_renderer.py` already encapsulates the
   architecture). 30-second reviewable per PR101 standard.

3. **DISTINCTNESS**: distinguishing-feature integration contract (Catalog
   #272 4-fields) declared in SubstrateContract:
   - `distinguishing_feature_name`: egocentric_pose_film_conditioning
   - `distinguishing_bytes_path`: archive section `pose_table` (fp16,
     n_pairs × pose_dim, ~7 KB for 600 pairs × 6 dims)
   - `inflate_consumer_function`: `EgoNeRVRenderer.forward(z, pose)` via
     `_FiLMModulator` consumed from pose_table at inflate time
   - `byte_mutation_smoke_passes`: PENDING Phase 2 + canonical inflate
     runtime landing

4. **RIGOR**: scaffold honors CLAUDE.md non-negotiables (eval_roundtrip
   patch, no MPS, --smoke gates synthetic batches, EMA-pending Phase 2
   per the council reactivation criteria). All 8 declared catalog
   compliance tags in SubstrateContract. Premise verification: smoke
   end-to-end runs on CPU and produces a non-trivial loss (verified
   2026-05-20).

5. **OPTIMIZATION-PER-TECHNIQUE** (Dim 5; Catalog #290): see canonical-vs-
   unique decision table above. Default is FORK when the canonical would
   suppress substrate-optimal engineering; sister Catalog #290 STRICT gate
   enforces this section header.

6. **STACK-OF-STACKS-COMPOSABILITY**: ego_nerv is composable with A1
   substrate as an additive sidecar (pose_table appended to A1 archive).
   Composition_alpha is unmeasured (no empirical anchor); Phase 2 council
   MUST evaluate via paired-comparison smoke per Catalog #322 sister.

7. **DETERMINISTIC-REPRODUCIBILITY**: seed pinning (--seed 20260520
   default) via canonical `pin_seeds`; archive bytes are determined by the
   EMA shadow state_dict + pose table per the `ego_nerv_as_renderer`
   library `export_ego_nerv_to_archive` function. Smoke verified
   reproducible across runs (loss=121.4508 at seed=20260520).

8. **EXTREME-OPTIMIZATION-PERFORMANCE**: Tier-1 engineering primitives
   declared via argparse (`--enable-autocast-fp16`, `--enable-torch-compile`,
   `--enable-gt-scorer-cache`) per Catalog #151 manifest. Activation is
   DEFERRED to Phase 2 full path; scaffold runs on CPU without these.

9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: NO empirical anchor at L0 SCAFFOLD
   posture. `predicted_score_target` is explicitly
   `unknown_l0_scaffold_pending_council_symposium`. Per Catalog #324 the
   predicted_band is omitted entirely until Phase 2 council symposium +
   post-training Tier-C validation lands.

## Cargo-cult audit per assumption

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-
check" + Catalog #303 self-protection + the hard-earned-vs-cargo-culted
addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| # | Assumption | Classification | Unwind path |
| --- | --- | --- | --- |
| 1 | Egocentric pose explains most inter-pair variance for driving video | CARGO-CULTED | Empirical test via paired-comparison smoke: train with pose_dim=0 (collapse to sister sane_hnerv) vs pose_dim=6 vs pose_dim=12. If pose_dim=0 matches sane_hnerv score and pose_dim=6/12 do not improve, the assumption is falsified. Pending Phase 2 budget. |
| 2 | FiLM modulator identity-init is the right inductive bias | HARD-EARNED | KK's bolt-on demonstrated identity-init in `tac.nerv_pose_conditioning_bolton`; the Lane 12-v2 reproduction validated this empirically. Preserve. |
| 3 | Pose table fp16 raw is the right archive format (vs int8 quantized) | CARGO-CULTED | Pose values are small floats; fp16 raw is ~7 KB for 600 pairs × 6 dims. int8 + per-pair scale would be ~4.2 KB. Unwind path: measure rate-distortion tradeoff at Phase 2 (small bytes, may not matter). |
| 4 | Pose can be pre-computed from the scored video at compress-time | HARD-EARNED | The contest video is known at compress-time; pose can be derived from PoseNet on adjacent pairs OR from RAFT optical flow. Pre-computation is structurally valid for `contest_one_video_replay` target mode. |
| 5 | Decoder channel ladder default (base_channels=36) is optimal | CARGO-CULTED | Inherited from sister NeRV variants. Unwind path: sweep base_channels ∈ {24, 36, 48} at Phase 2 to find the optimum at fixed byte budget. |
| 6 | The substrate is composable with A1 via simple sidecar concatenation | CARGO-CULTED | UNCONFIRMED; Catalog #322 paired-composition probe is required to measure composition_alpha empirically. |
| 7 | EMA decay 0.997 + shadow at inference IS the canonical training stability primitive | HARD-EARNED | Universal across sister substrates per CLAUDE.md "EMA - NON-NEGOTIABLE" non-negotiable. Preserve. |

## Observability surface

Per Catalog #305 self-protection (operator standing directive 2026-05-16: "the
xray and autopilot and all tools and the experiment and designs themselves
should be built so as to support absolute max observability into behavior"):

The 6-facet observability declaration:

1. **Inspectable per layer**: `EgoNeRVRenderer.schema` property exposes
   every state_dict key + tensor shape; `_FiLMModulator` exposes scale +
   shift output tensors per forward; `EgoNeRVPoseTable` exposes per-pair
   pose embedding.

2. **Decomposable per signal**: Phase 2 full path will produce the
   canonical seg/pose/rate component decomposition via
   `score_pair_components`. Scaffold smoke produces only the surrogate
   `decoded.abs().mean()` loss; no per-axis decomposition possible at
   scaffold posture (declared explicitly).

3. **Diff-able across runs**: seed-pinned via `--seed 20260520`. Smoke
   reproduces loss=121.4508 at default seed deterministically.

4. **Queryable post-hoc**: provenance.json schema
   `ego_nerv_l0_scaffold_smoke_v1` captures all argparse args + git_head +
   torch_version + n_params + hardware_substrate_detected; machine-readable
   for autopilot consumption.

5. **Cite-able**: every run anchored to (commit sha, substrate=ego_nerv,
   call_id-if-modal, config) tuple per Catalog #245 modal_call_id_ledger
   pattern; provenance.json carries `git_head` + `args` + `started_at`.

6. **Counterfactual-able**: PENDING Phase 2. Byte-mutation smoke per
   Catalog #139/#272 distinguishing-feature contract is the Phase 2
   reactivation gate.

## Predicted ΔS band

L0 SCAFFOLD predicted_band is EXPLICITLY UNSET per Catalog #324 (no random-
init Tier-C prediction permitted). The recipe declares
`predicted_band_validation_status: pending_post_training`; the predicted
score target is `unknown_l0_scaffold_pending_council_symposium`.

Per Catalog #296 (Dykstra-feasibility predicted-band): no predicted band is
emitted at scaffold posture so no Dykstra-feasibility check applies. Phase 2
council symposium MUST land the predicted band per the
Shannon+R(D)+Dykstra-feasibility intersection of (rate, seg, pose) constraints.

# PREDICTED_BAND_VIBES_OK:no_predicted_band_at_L0_scaffold_per_catalog_324_pending_phase_2_council

## Reactivation criteria for full path

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" +
CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + the trainer's `_full_main` docstring:

1. Per-substrate adversarial grand council symposium per CLAUDE.md
   "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
   (Catalog #325) returns PROCEED or PROCEED_WITH_REVISIONS verdict for
   ego_nerv specifically.
2. Canonical `src/tac/substrates/ego_nerv/` package migration lands with
   `architecture.py` + `archive.py` + `inflate.py` + `score_aware_loss.py`
   matching tc_nerv structure (currently the substrate library lives at
   `src/tac/ego_nerv_as_renderer.py` in legacy monolithic format).
3. `submissions/ego_nerv_substrate/inflate.{py,sh}` smoke proves a
   contest-compliant 3-positional-arg runtime per Catalog #146.
4. Cargo-cult audit per Catalog #303 (this memo's section above) plus
   9-dim checklist per Catalog #294 (above) plus observability surface per
   Catalog #305 (above) plus Dykstra feasibility predicted-band per Catalog
   #296 (above) plus horizon-class declaration per Catalog #309 (frontmatter)
   all preserved.
5. Operator-frontier-override per Catalog #300 Mission alignment
   Consequence 1 OR Phase 2 council approval converts
   `research_only=true` to `false` in the recipe + this trainer's
   `SubstrateContract`.

## Cross-references

- `src/tac/ego_nerv_as_renderer.py` (substrate library)
- `experiments/train_substrate_tc_nerv.py` (canonical sister trainer skeleton)
- `experiments/train_ego_nerv_as_renderer.py` (pre-canonical scaffold)
- `.omx/operator_authorize_recipes/substrate_ego_nerv_modal_a10g_diagnostic_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_ego_nerv_modal_a100_dispatch.yaml` (legacy)
- `scripts/remote_lane_substrate_ego_nerv_l0_scaffold.sh`
- `scripts/remote_lane_substrate_ego_nerv.sh` (legacy)
- `feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md`
- `.omx/research/substrate_tradition_taxonomy_20260512.md`
- `feedback_slot_build_1_nerv_trio_scaffolds_landed_20260520.md` (this BUILD-1 landing)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:ego-NeRV-L0-scaffold-design-memo-trigger-tokens-in-design-rationale-section-not-new-equation-pre-anchor-design -->
