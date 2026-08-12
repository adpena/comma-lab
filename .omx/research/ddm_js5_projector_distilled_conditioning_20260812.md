# ddm_js5 projector-distilled conditioning receipt — 2026-08-12

## RESULT

**INSTANCE verdict:** the bounded stratified-random n32 build-to-admission did not admit a useful projector-distilled module. The α curve had no point that jointly passed the realized pose endpoint and moved robust flips, and the hidden 4/8/12 ladder accepted 0/15 proposals. This is not a family kill: F1 requires at least 200 realized proposals, F2 never became eligible because no nonzero projected stage passed, and F3 had no finite B/robust-flip observations.

Axis: `[macOS-CPU advisory, instrument floor 0.0131 S]`. The `projected_n600_*` fields below are weighted projections from the sealed stratified n32 relative gauge, not an n600 scorer run. No archive was built, no exact contest score was measured, and the canonical own-vehicle frontier did not move.

## ALPHA LEAKAGE CURVE

The curve ran first on the inherited JS4 selected step-25 module and retained every camera, correction, pose-error, logits, and argmax payload. The preregistered continuous second-order model was directionally supported (measured log-slope 2.511 versus 2.0), but the post-uint8 log-slope was only 0.957. At small amplitudes, the receiver lattice dominated while robust motion disappeared.

| α | continuous correction leakage | uint8 correction leakage | total realized pose Δ | projected n600 robust Δflips | endpoint pass |
|---:|---:|---:|---:|---:|:---:|
| 1 | 8.8360566e-4 | 8.5741907e-4 | +8.4787149e-4 | -305 | no |
| 1/2 | 1.0038738e-4 | 1.3687468e-4 | +1.2732710e-4 | -112 | no |
| 1/4 | 1.6397902e-5 | 6.6803035e-5 | +5.7255457e-5 | 0 | no |
| 1/8 | 3.4996673e-6 | 4.2786291e-5 | +3.3238713e-5 | 0 | no |
| 1/16 | 7.8618451e-7 | 5.5702489e-5 | +4.6154911e-5 | 0 | no |

The zero-rerender reference had pose Δ `-9.5475783e-6`. No α qualified. The controller therefore used the smallest measured α, `1/16`, only as a bounded fallback, with a preregistered relinearization cadence of 16 accepted steps. That fallback is not a useful-overlap claim.

## REALIZED ACCEPTANCE AND CAPACITY

Each proposal was projected during training and then run through the full held-out n32 receiver/uint8/custody PoseNet path before acceptance. The step-1 budget was `-8.5852801e-6`, allocating only 1/12 of the distance from the zero-rerender reference to the 2e-6 endpoint. The 15 deterministically seeded realized proposal deltas ranged from `-7.7983052e-6` to `-1.3753277e-6`; none preserved enough of the zero-rerender advantage. Every proposal was therefore rejected, and model, optimizer, Torch RNG, and NumPy RNG state were restored before the next shrink.

| hidden | proposed | accepted | acceptance | bare Brotli q11 bytes | bare pose Δ | robust Δflips | B/robust-flip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 0 | 0% | 801 | -9.5475783e-6 | 0 | undefined |
| 8 | 5 | 0 | 0% | 1,286 | -9.5475783e-6 | 0 | undefined |
| 12 | 5 | 0 | 0% | 1,821 | -9.5475783e-6 | 0 | undefined |

The selected 801-byte int8 module is the restored zero-output model. It passes the absolute pose gate (`-9.5475783e-6 < 2e-6`) but moves zero robust flips, so it is explicitly **not admitted**. The bare packet contains no projector basis and invokes no scorer at decode. The economics target is ≤1.28 B/robust-flip; all three denominators are zero, so there is no economics trend and no T4 admission row.

Periodic relinearization and the distillation-loss activation path are implemented and tested, but neither became empirically active because the controller accepted no step and never produced a nonzero projected-passing stage. Claiming either as measured would be false.

## FALSIFIERS AND ROUTING

- **F1 not eligible:** the α curve had no useful point and acceptance was 0%, but only 15 proposals were measured versus the required 200. The family remains open.
- **F2 not eligible:** there was no nonzero projected pose-gated robust-improving stage on which to test whether the bare module failed distillation.
- **F3 not eligible:** there were no finite B/robust-flip rows. The bounded instance is rate-useless, but the charter's across-rung trend falsifier was not observed.
- **Long burn blocked now:** firing the sealed command from zero would repeat the rejected first-step ladder. The recipe is sealed but has `fire_allowed_now=false`; it requires a content-distinct producer to seed a nonzero admitted checkpoint first.

## RECALL EVIDENCE

Stores consulted before deciding and building:

- `.omx/research/codex_findings_ddm_j5_366_realized_acceptance_warmstart_20260723_codex.md` and `.omx/research/ddm_j5_366_realized_acceptance_canonical_equations_20260723.md`: accept only realized receiver state; reject, restore, and shrink on a budget miss.
- `.omx/research/codex_findings_ddm_v17_iterative_realized_trust_region_20260723T034200Z_codex.md`: the validity ratio controls the relinearization radius/cadence and does not replace realized acceptance.
- `.omx/research/ddm_la1_20260805/RECEIPT.md`: raw Q3 projection is not a load-bearing window-training cure; curvature-corrected conditioning and export-time placement remained live.
- `.omx/research/ddm_rvs2_geometry_survival_crosswalk_20260811.md`: no blanket post-hoc projection claim; a pose-null sibling needs a current receiver-coordinate projector.
- `.omx/research/ddm_js2b_edge_conditioning_relative_gauge_20260812.md`, `.omx/research/ddm_js3_learned_implicit_conditioning_20260812.md`, and `.omx/research/ddm_js4_pose_null_projected_conditioning_20260812.md`: sealed δ/gauge, inherited trainer, current projector, and the F2-realized/F3 walls.

## IMPLEMENTATION AND CUSTODY

- Source: `experiments/ddm_js5_projector_distilled_conditioning.py`; immutable measured-source custody SHA-256 `981240bef78e195595978241b383ea4b5ad4ac23ab321ffd4609f6c645dc5d80` at `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/source_custody/by_sha256/981240bef78e195595978241b383ea4b5ad4ac23ab321ffd4609f6c645dc5d80.py`.
- Tests: `experiments/tests/test_ddm_js5_projector_distilled_conditioning.py`.
- Final receipt: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/FINAL_RESULT.json`, SHA-256 `315860c13d7b06b9c0d9e73fad87856d8e7bf8194570dcedfea3707dd247338a`.
- α receipt: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/alpha_curve/RESULT.json`, SHA-256 `9fa8411f71a3e0a81a0273fa362bc9b89b387ea3f6d036c6298b1a8f7a8dc7fe`.
- Selected null packet: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/capacity/hidden_4/stages/stage_01_target_000004/ema/retained/conditioner.int8.br`, 801 B, SHA-256 `09e4e877b44c692fbe9836f74257863c9b3e3723608b4f33ff8ba9eda836df6c`.
- Payload store: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/`, 7.5 GiB at completion. All α candidates, all 15 proposals, live/EMA stage modules, and checkpoints are retained with byte counts and SHA-256 records.

Three implementation defects were caught before landing: an inference-mode projector cache could not participate in autograd after proposal evaluation; the zero-output module could be mislabeled as a useful admission; and rung initialization depended on whether the α curve was computed or loaded because it shared the global Torch RNG stream. The trainer now clones the active basis into ordinary tensors before every retry, reports pose-gate and robust-movement gates separately, and uses a recorded hidden-specific sub-seed. A resume repeat left the α receipt and all three rung finals byte-identical; only the aggregate receipt's live storage `free_bytes` field changed.

## SEALED MAIN RECIPE AND QUEUE ANNEX

Sealed recipe: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/SEALED_MAIN_RECIPE.json` (SHA-256 `6d983334538d51b7e02f8240c8bb1fecb751ac18729efa093dd4718f002dfbcc`). Queue annex: `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/QUEUE_ANNEX.md` (SHA-256 `551c16cc033ff23ac135aff377a304fca0a2f1465766f6864a114a77542b446f`). No skeleton/shared-ledger file was edited.

Verbatim queued rows:

- **Action:** content-distinct realized-acceptance proposal extension. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`. **Fire trigger:** MAIN provides a representation-level conditioning proposal source that is not another replay of this five-shrink first-step ladder; retain every realized payload and stop at the first nonzero useful bare admission or after 200 unique proposals for the F1 decision.
- **Action:** projector-distilled conditioning MAIN burn. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/main_burn`. **Fire trigger:** MAIN first obtains a nonzero bare pose-gated robust-improving checkpoint from a content-distinct proposal extension, seeds it into the consumer store, verifies that the >=200-proposal F1 gate has not fired, owns the training leg, observes the sole n600 scorer slot free, and passes storage/memory preflights.

## LIVE-HYPOTHESES

- A content-distinct proposal source may find the small pose-null/seg-reachable overlap that five scalar LR shrinks could not: the present 15 proposals varied width and step size but not the correction representation.
- Receiver-lattice-aware or curvature-corrected conditioning may preserve the observed continuous `α^2.51` leakage scaling through uint8: the continuous law behaved as expected while the uint8 law flattened to `α^0.96`.
- Distillation may close the projector/bare gap after a nonzero projected stage exists: the current run never reached the prerequisite and therefore did not test this mechanism.

## DEAD-ENDS

- Amplitude-only shrinking of the inherited JS4 step-25 module is closed at INSTANCE scope: useful robust movement disappeared before the realized pose gate passed.
- Replaying the current five-shrink first-step ladder as a long burn is closed: all widths restored to zero after 0/5 acceptance, so the sealed burn is not fireable from the current state.
- The 801-byte zero module is not an admission candidate: its favorable pose delta is the zero-rerender reference and it moves no robust flips.
