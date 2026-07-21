# M1 quantization-aware realization machinery — TIER-0 handoff

**Written:** 2026-07-21T03:40:00Z

**Lane:** `m1_quant_aware_retrain_20260721T025441Z`

**Code commit:** `c89b1e020aa5e6ce143f2d32a8167593f5dc1c6a`

**Verdict:** `BUILT_TESTED_PORTABLE`

**verdict_scope:** realization machinery and bounded fixture behavior only; not a full-n600,
SegNet, PoseNet, archive-byte, contest-score, launch, promotion, family, or frontier verdict.

## Outcome first

The from-scratch constructive pipeline now has portable S3 realization machinery: exact uint8
forward with saturation-aware STE, a typed fixed-magnitude/dead-zone parameterization, derived
receiver-reachable amplitude floors, exact changed-pixel/target-band-entry telemetry, an
event-driven band-fit stopping rule, and a fail-closed zero-entry halt before rate polish or
carrier export. The implementation is committed and sealed by `97 passed, 5 skipped`, Ruff,
`py_compile`, `git diff --check`, and two clean review passes for each edited Python file.

There was **no full-n600 relaunch and no old-archive candidate**. The operator's
`2026-07-21T03:27:51Z` re-scope makes landed, tested, portable realization machinery the complete
deliverable; a source-lineage n600 run would pull the work back toward the inherited archive and
away from the constructive crux.

## CONTENT LINEAGE

- `fixture=inherited`: the 83,838-byte base archive, base decoder, source-plane band manifest,
  and r1b4 carrier binding were used only as a fast deterministic test fixture.
- `machinery=ours`: the typed quantization modes and target modes, floor derivation, exact
  receiver telemetry, event continuation, inertness refusal, checkpoint/resume custody, and
  regression tests are new in commit `c89b1e020a`.
- The fixture produced no candidate packet. `carrier_packet=null`; pointer mutation, promotion,
  paid dispatch, score claims, and launch claims are all false.

## CRUX ALIGNMENT

The consumer is `joint_planes_direct_strike:S3_realization`. The CLI now accepts
`--band-target {source_planes,joint_solved_planes}`. A joint target is accepted only when the
supplied plane bytes match `--joint-solved-planes-sha256` and that SHA also equals the rebuilt
band manifest's `source_sha256`. This makes the mechanism portable to a freshly constructed
joint Seg/Pose target while refusing to attach inherited source radii to different plane bytes.

The earlier #549 receipt is an existence anchor, not the active S3 target. A new joint-plane band
manifest with exact source custody remains an input owed by the constructive pipeline; this arm
does not invent those bytes or infer n600 efficacy from the n24 receipt.

## Root-cause re-derivation

The delegated premise that the old trainer never simulated uint8 in-loop is **FALSIFIED**.
`torch_uint8` already used exact clipped/rounded uint8 values in the forward pass with a
saturation-aware straight-through backward. The remaining structural failure was subtler:
bilinear `pair_plane_codes × shared_rgb_head` could keep both factors below a receiver quantum,
and the run had no exact realized-entry gate to distinguish continuous descent from useful
discrete actuation.

The cure therefore preserves exact STE and adds a reachable nonzero action:

1. `plain_ste` remains the legacy A/B control.
2. In `fixed_magnitude_deadzone`, exact zero remains OFF. After an optimizer step chooses a
   nonzero site, the code is projected to sign times at least a floor derived from integer
   target-band entry debt, the largest active curvelet feature response, and the shared RGB head
   norm lower bound.
3. Fresh codes, EMA, and moments start at exact zero so the optimizer still chooses **where** to
   activate. Head rows are projected to norm at least one; inactive moments are cleared.
4. Exact NumPy clip-rint uint8 receiver telemetry is emitted at every checkpoint and persisted
   with the floor SHA, target custody, flip history, and halt reason.
5. Band fit continues until a realized-entry plateau. Patience is derived as
   `ceil(log2(residual_width))`, which is 2 for width 4; configured band-fit epochs are only a
   safety cap. A plateau at zero refuses with rc 7 before rate polish and emits no carrier.

## Bounded real fixture evidence

Receipt:
`/Volumes/VertigoDataTier/pact/evidence/m1_quant_aware_smoke_full_n2_v2_20260721/training_receipt.json`

- Receipt SHA-256: `76a6bad74ada559a59f8e757898a6260729c9a52413214838514b29a2f4aebe5`
- Config SHA-256: `e4ea519ab0fb6f8ad9d3f95843717d1411088e66b97162507d9df2c5fbb63252`
- Axis/scope: `[macOS-CPU advisory]`, capped prefix n2, non-n600, non-score
- Quantization/target: `fixed_magnitude_deadzone` / `source_planes`
- Global step: 4; flip history: `[0, 0, 0]`; derived patience: 2
- `realized_changed_pixel_count=256450`
- `realized_changed_channel_count=503198`
- Changed inside/outside selected band: `1218 / 255232`
- Selected band pixels: `2605`
- `realized_target_band_entry_count=0`; `realized_flip_count=0`
- Result: expected rc 7, `band_fit_zero_realized_flip_count_at_plateau`
- `carrier_packet=null`; rate polish not entered; peak RSS `3,356,688,384` bytes

This is a useful two-sided result. **MEASURED:** fixed-mode parameters produce abundant exact
uint8 actuation, so the generation path is no longer quantization-inert. **MEASURED:** those
writes have large outside-band spill and enter none of the inherited target bands. Therefore the
halt gate works and the old four-code source-plane curvelet fixture is rejected. This is a
fixture/formulation negative only, not an S3 machinery or representation-family negative.
`realized_flip_count` is target-band residency telemetry, not SegNet argmax authority.

## Equation refinement

This extends anchor
`realization_is_quantization_gated_minimal_writes_die_at_uint8_20260720`
with the generation-side refinement `M1-QA-S3-20260721`:

Let

```text
R_x(theta) = clip_255(round(x + A(c, h)))
q_x(theta) = R_x(theta) - x
```

and for an optimizer-selected nonzero code let

```text
c'_ijk = sign(c_ijk) max(|c_ijk|, a_min_ijk)
a_min_ijk = delta_entry(i,j) sqrt(3) / max_{p in active(i,j)} |phi_k(p)|
```

where `delta_entry` is the smallest positive integer step needed to enter a selected target band,
the RGB head row is constrained to norm at least one, and exact zero remains OFF. This lower bound
makes a single active factor receiver-reachable before cancellation; it does **not** guarantee a
useful discrete write, so the exact receiver remains authority.

The refined admission law is

```text
exact_uint8_forward
AND changed_pixels(R_x(theta), x) > 0
AND target_band_entries(R_x(theta)) > 0
AND downstream_joint_hard_oracle_debt decreases
AND marginal_score_gain_per_byte >= 25 / 37_545_489.
```

Thus exact STE is necessary but insufficient; nonzero changed pixels are also necessary but
insufficient. A generation actuator is admissible only when it has a reachable discrete action,
survives the exact receiver, enters the intended target set, improves the joint Seg/Pose authority
surface, and pays its byte rate. The fixture reached the second predicate and failed the third, so
the machinery correctly refused before the later predicates were claimed.

The anchor's older shorthand that “fixed-magnitude” was positive must not be transferred across
formulations. The later R1b7 autopsy found its local fixed-magnitude and closest-sign writes
byte-identical and nonpositive on that bounded control. Here `fixed_magnitude_deadzone` names a
typed reachability mechanism only; neither that prior result nor this n2 fixture proves efficacy.

## Triality and system wire-in

- **DSL:** typed `QuantizationMode` and `BandTarget` CLI surfaces; no invented launch flag.
- **DAG:** `.omx/research/m1_quant_aware_retrain_DAG_FEED_20260721T034000Z.md` routes the machinery
  into `joint_planes_direct_strike:S3_realization` and records the zero-entry refusal edge.
- **Equations:** `M1-QA-S3-20260721` refines the quantization-gated generation anchor by ID above.
- **Sensitivity/Pareto:** receiver telemetry exposes changed inside/outside the selected band; the
  downstream consumer must add joint hard-oracle and actual-byte terms before admission.
- **Bit allocator:** zero-entry or non-paying carriers are rejected; later S3 allocation must use
  joint marginal score gain per actual counted byte.
- **Autopilot:** rc 7 and `carrier_packet=null` make inert formulations non-dispatchable.
- **Continual learning:** the premise correction and fixture-scoped negative are durable here and
  in the DAG FEED, rather than promoted into a family verdict.
- **Probe disambiguator:** `plain_ste` and `fixed_magnitude_deadzone` remain callable A/B modes;
  `source_planes` and `joint_solved_planes` remain typed target interpretations.

## STORES CONSULTED

Authority prompt and live per-arm/broadcast inboxes through `2026-07-21T03:27:51Z` and
`2026-07-19T19:48:01Z`; `CLAUDE.md`; `AGENTS.md`; v7.5/v8 operating specs; current pointer and
lane/subagent ledgers; the governed M1 config and inherited fixture manifests; #549 receipt; live
trainer, emitter, receiver-roundtrip, observer, and tests; final SSD smoke receipt; the existing
`realization_is_quantization_gated_minimal_writes_die_at_uint8_20260720` memory anchor; latest
sister findings/session/design/council surfaces.

## No-fake / handoff

The contest-CPU pointer remains `0.1910828242 [contest-CPU Linux x86_64]`. No score, archive,
governor ADMIT, PID, dispatch, promotion, or joint-target efficacy is claimed. Durable config:
`.omx/research/m1_quant_aware_retrain_config_20260721T034000Z.json`, SHA-256
`fbd9931d5673106992de53fb6010b862406cb3b5cc637eabdd3f2d83ecacf788`.

**MAIN landing review is required.** Review the branch diff from `da726f4129` through the two M1
commits, with particular attention to fixed-floor sufficiency, plateau semantics, exact receiver
telemetry definitions, joint-target SHA closure, and the intentionally fixture-scoped verdict.
