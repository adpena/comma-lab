# Blocker spec — hi_nerv_sidecar_scorer_effect.v1 (the OTHER half of the lowering race)

UTC: 2026-06-08T210000Z · claude (solo) · planning, no score claim. Consumes the
LANDED decoder-section guilt sweep (`06598d8a5`). Pre-registered blocker per the
turn contract: "emit sidecar-scorer row in the same tranche OR explicitly block
it. Payload survival is not enough" (Rule #8).

## Why this is BLOCKED this tranche (not deferred-by-laziness)

The sidecar-scorer row must be priced over the SAME L-set the backend sweep
prices: `L = W_fake \ W_parseback`. That set is produced by
`measure_birth_decoder_section_guilt_sweep` (the `lset_certificate` on its
`section_rows[0]`). The lowering race is `argmin ΔS_total` over {guilty-section
QAT, joint QAT, byte-priced sidecar, discard}; all arms must be compared at the
same L or the comparison is apples-to-oranges (the apples-to-apples discipline).
Therefore the sidecar row is sequenced AFTER the sweep names L, not before.

It is a distinct measurement (new scorer pass on the sidecar-applied receiver
frame), not a field add — so it is its own bounded build, gated on a birth whose
sweep produced a non-empty L (`l_set_size > 0`).

## Exact patch target

`src/tac/substrates/hi_nerv/archive_candidate.py` ::
`build_hi_nerv_target_region_action_parseback_survival` (def at line 206).

It already: parses the archive, decodes the target-region action sidecar, renders
the receiver WITH and WITHOUT the sidecar, and verifies every encoded support
pixel is overwritten with the exact uint8 RGB action value (payload/program
survival). What it does NOT do: run SegNet/PoseNet on the with-sidecar frame and
price the scorer effect over L. That is the missing half.

### Required new signature inputs (additive, keyword-only)

```
scorer_teacher: Any,          # build SegNet logits on the with-sidecar frame
pose_teacher: Any | None,     # pose delta with/without sidecar
target_labels: Any,           # birth region reconstruction
live_birth_payload: Mapping,  # action_id / worst_region / support continuity
l_set_mask_bhw: np.ndarray,   # the EXACT L from the sweep (W_fake \ W_parseback)
fakequant_logits_bhwc: np.ndarray,   # the won-surface reference (sweep emits)
parseback_logits_bhwc: np.ndarray,   # the collapsed-surface reference (sweep)
```

### Required emitted fields (`hi_nerv_sidecar_scorer_effect.v1`)

```
sidecar_wrong_to_target          # argmax-won-to-target count over region, with sidecar
sidecar_margin_on_L_p10          # evaluated (with-sidecar) target margin over L
sidecar_margin_on_L_p50
sidecar_retention_vs_fakequant   # |sidecar wins ∩ L| / |W_fake|
sidecar_pose_delta               # pose(with) - pose(without), first-6 MSE
sidecar_delta_bytes              # archive bytes added by the sidecar program
sidecar_exact_delta_score_total  # 100*Δd_seg + (√(10·d_pose') − √(10·d_pose)) + 25*Δbytes/37_545_489
parseback_scorer_effect_survived # currently null/unmeasured -> bool over L
scorer_effect_survival_measured  # True (this row is the measurement)
authority = planning_control_false_authority   # no score claim
```

Margins via `tac.substrates.hi_nerv.target_region_birth.lset_subset_conditioned_margin_certificate`
with `evaluated_logits_bhwc = sidecar_applied_logits`, the same v2 cert the sweep
uses, so backend and sidecar arms share ONE margin definition.

## The lowering race (decision the two rows jointly decide)

```
best_lowering = argmin over arms of ΔS_total, each gated on:
  (a) scorer-effect survival over L (margin p50 > 0 after the arm), AND
  (b) Pose trust preserved (sidecar_pose_delta within trust band).

arms:
  backend_section_qat[s]   from the sweep guilty section   (ΔS from the section cert)
  joint_qat[s+t]           from the sweep guilty commutator
  byte_priced_sidecar      from THIS row                   (ΔS includes sidecar bytes)
  discard                  ΔS = 0 (keep the parse-back-collapsed birth)
```

If `byte_priced_sidecar` has the lowest ΔS_total with positive L survival, the
canonical result is `best_lowering = byte_priced_sidecar;
backend_realization_complete = false` — a legitimate frontier-compiler outcome
(the action survives as a sidechannel cheaper than fixing the backend).

## DO NOT

- Do not call the sidecar a win on payload/program survival alone (Rule #8): the
  pixels are overwritten, but SegNet argmax over L after uint8/resize is the only
  authority.
- Do not price the sidecar over a DIFFERENT L than the backend sweep (apples-to-
  apples).
- Do not build this before a birth's sweep yields `l_set_size > 0` (no L → nothing
  to price; the smoke-scale birth may not collapse — replay a real birth first).
