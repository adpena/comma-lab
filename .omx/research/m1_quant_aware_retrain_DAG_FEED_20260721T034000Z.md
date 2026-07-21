# DAG FEED — M1 quantization-aware S3 realization machinery

```yaml
schema: pact_dag_feed.v1
written_at_utc: 2026-07-21T03:40:00Z
lane_id: m1_quant_aware_retrain_20260721T025441Z
research_only: true
verdict: BUILT_TESTED_PORTABLE
verdict_scope: >-
  Realization machinery and bounded inherited-fixture behavior only; not full-n600,
  SegNet, PoseNet, archive-byte, contest-score, launch, promotion, family, or frontier authority.
content_lineage: fixture=inherited, machinery=ours
crux_alignment: joint_planes_direct_strike:S3_realization
code_commit: c89b1e020aa5e6ce143f2d32a8167593f5dc1c6a
main_landing_review_required: true
```

## Nodes

| Node | Type | State | Custody |
|---|---|---|---|
| `M1_OLD_CONTINUOUS_DESCENT` | measured formulation negative | settled | final EMA had subquantum factors; observer rows had zero changed pixels |
| `M1_EXISTING_EXACT_STE` | premise correction | confirmed in code | the pre-edit trainer already used exact clip-rint uint8 forward plus saturation-aware STE |
| `M1_FIXED_MAGNITUDE_ACTUATOR` | machinery | built/tested | typed projection, derived floor, head norm floor, resume custody |
| `M1_EXACT_RECEIVER_GATE` | machinery | built/tested | exact NumPy uint8 changed/entry telemetry per checkpoint |
| `M1_EVENT_CONTINUATION` | machinery | built/tested | plateau patience `ceil(log2(residual_width))`; epochs are safety cap |
| `M1_SOURCE_FIXTURE` | inherited fixture | test-only | exact base/band/carrier hashes in the config receipt |
| `M1_N2_FIXTURE_RECEIPT` | bounded empirical anchor | measured advisory | receipt SHA `76a6bad74ada559a59f8e757898a6260729c9a52413214838514b29a2f4aebe5` |
| `M1_ZERO_ENTRY_REFUSAL` | fail-closed gate | fired | rc 7; no carrier; no rate polish |
| `JOINT_TARGET_SHA_BINDING` | portability/custody | built/tested | target file SHA must equal rebuilt band manifest source SHA |
| `JOINT_PLANES_DIRECT_STRIKE_S3` | downstream consumer | awaiting constructive inputs | rebuilt joint-plane target plus joint band manifest, then hard-oracle/byte admission |
| `CONTEST_POINTER` | frontier authority | unchanged | `0.1910828242 [contest-CPU Linux x86_64]` |

## Edges

```text
M1_OLD_CONTINUOUS_DESCENT
  --premise_rederive--> M1_EXISTING_EXACT_STE

M1_EXISTING_EXACT_STE
  --necessary_not_sufficient--> M1_FIXED_MAGNITUDE_ACTUATOR

M1_FIXED_MAGNITUDE_ACTUATOR
  --exact_uint8_emit--> M1_EXACT_RECEIVER_GATE

M1_EXACT_RECEIVER_GATE
  --checkpoint_observation--> M1_EVENT_CONTINUATION

M1_SOURCE_FIXTURE
  --fixture_only--> M1_N2_FIXTURE_RECEIPT

M1_N2_FIXTURE_RECEIPT
  --changed_pixels=256450--> M1_EXACT_RECEIVER_GATE

M1_N2_FIXTURE_RECEIPT
  --target_band_entries=0,outside_spill=255232--> M1_ZERO_ENTRY_REFUSAL

M1_ZERO_ENTRY_REFUSAL
  --blocks--> carrier_export
M1_ZERO_ENTRY_REFUSAL
  --blocks--> rate_polish
M1_ZERO_ENTRY_REFUSAL
  --blocks--> inherited_fixture_n600_launch

M1_FIXED_MAGNITUDE_ACTUATOR
  --portable_with--> JOINT_TARGET_SHA_BINDING
JOINT_TARGET_SHA_BINDING
  --future_input--> JOINT_PLANES_DIRECT_STRIKE_S3

JOINT_PLANES_DIRECT_STRIKE_S3
  --requires_before_admission--> joint_seg_pose_hard_oracle
JOINT_PLANES_DIRECT_STRIKE_S3
  --requires_before_admission--> actual_counted_archive_bytes

M1_N2_FIXTURE_RECEIPT
  --no_edge--> CONTEST_POINTER
```

## Equation refinement by anchor ID

**Extended anchor ID:**
`realization_is_quantization_gated_minimal_writes_die_at_uint8_20260720`

**Refinement ID:** `M1-QA-S3-20260721`  
**Operation:** `domain_refinement` / `generation_side_anchor_refinement`  
**Registry action:** none; this FEED does not invent a canonical-registry row.

The old M1-specific clause “the training loss never simulated uint8 in-loop” is falsified by
source re-derivation. The trainer already computed

```text
R_x(theta) = clip_255(round(x + A(theta)))
```

in the forward pass with an STE backward. The corrected law is:

```text
Q0(theta) := R_x(theta) - x

c'_ijk := 0                                      if c_ijk = 0
          sign(c_ijk) max(|c_ijk|, a_min_ijk)   otherwise

a_min_ijk := delta_entry(i,j) sqrt(3)
             / max_{p in active(i,j)} |phi_k(p)|

patience(width) := max(1, ceil(log2(width)))

admit(theta) :=
  exact_uint8_forward(theta)
  AND count_nonzero(Q0(theta)) > 0
  AND target_band_entries(R_x(theta)) > 0
  AND Delta_joint_hard_oracle(theta) < 0
  AND (-Delta_S(theta) / Delta_bytes(theta)) >= 25 / 37_545_489.
```

`a_min` is a reachability lower bound under a shared RGB-head row norm of at least one; basis
cancellation and saturation mean the exact receiver remains authoritative. Consequently:

```text
exact STE alone                         = necessary, insufficient
changed uint8 pixels                    = necessary, insufficient
selected target-band entry              = necessary, insufficient
joint Seg/Pose improvement per byte      = final downstream admission authority
```

The bounded fixture establishes `count_nonzero(Q0)>0` but falsifies target-band entry. The result
therefore validates the actuator and its gate while rejecting only this inherited fixture
formulation. It does not falsify S3 or any basis/representation family.

The `fixed_magnitude_deadzone` label is mechanism identity, not an efficacy posterior. The later
R1b7 local-write autopsy superseded the anchor's older positive shorthand for that domain: fixed
magnitude and closest-sign were byte-identical and nonpositive there. This S3 refinement carries
no positive score inference from either formulation.

## Triality legs

- **DSL:** `QuantizationMode={plain_ste,fixed_magnitude_deadzone}` and
  `BandTarget={source_planes,joint_solved_planes}`; joint mode fails closed on SHA mismatch.
- **DAG:** typed nodes/edges above, including explicit no-edge to the contest pointer.
- **Equations:** refinement `M1-QA-S3-20260721` above, extending the named anchor without creating
  false registry authority.

## Unified-solver wire-in

1. **Sensitivity map:** consume exact changed-inside/outside-selected-band counts; S3 must add
   joint Seg/Pose hard-oracle deltas before ranking writes.
2. **Pareto constraint:** no realization is admissible without target entry, joint debt reduction,
   and actual byte break-even.
3. **Bit allocator:** allocate only among receiver-surviving joint writes; zero-entry carrier rate
   has infinite effective cost and is rejected.
4. **Cathedral/autopilot:** rc 7 plus `carrier_packet=null` is a strict non-dispatchable outcome;
   only hash-closed constructive joint inputs may reach S3.
5. **Continual-learning posterior:** record that exact STE existed, fixed-magnitude caused exact
   actuation, and the source fixture had zero useful entries with large spill. Scope all three.
6. **Probe disambiguator:** retain both quantization modes and both target modes as callable A/B
   interpretations; the exact receiver and downstream hard oracle arbitrate.

## Evidence pointers

- Config/evidence index:
  `.omx/research/m1_quant_aware_retrain_config_20260721T034000Z.json`, SHA-256
  `fbd9931d5673106992de53fb6010b862406cb3b5cc637eabdd3f2d83ecacf788`
- Findings/TIER-0 handoff: `.omx/research/m1_quant_aware_retrain_20260721T034000Z.md`
- Code commit: `c89b1e020aa5e6ce143f2d32a8167593f5dc1c6a`
- Test seal: `97 passed, 5 skipped`; Ruff and `py_compile` pass; two clean Python reviews.

No governor ADMIT, PID, full-n600 run, archive candidate, score row, promotion, or pointer mutation
exists. MAIN must review and merge this branch before the machinery is treated as landed.
