# Codex Findings: IAS1 Descriptor-Only Rate Penalty

UTC: 2026-05-23T21:26:39Z

## Verdict

IAS1 should be modeled as `descriptor_parsed_not_consumed` /
`descriptor_only_rate_penalty`, not as a negative verdict on inverse-scorer atom
quality.

The IAS1 archive added a descriptor tail, but the IAS1-aware runtime parsed the
descriptor without using it to change decoded frames. The candidate/source
full-frame parity probe reported byte-identical output, so exact eval measured
almost entirely the extra archive rate.

## Empirical Signal

- IAS1 archive: `2d085078...e549`, `181232` bytes.
- Source DQS1 archive: `3c4e15bf...5d59`, `178592` bytes.
- Serialized delta: `+2640` bytes.
- Realized component gain: `0`.
- Realized net delta versus same-source DQS1 on `[contest-CPU]`:
  `+0.001758867636`, decomposed as `+0.001757867636` rate,
  `+0.000001` SegNet, `+0` PoseNet.
- Realized net delta versus same-source DQS1 on `[contest-CUDA T4]`:
  `+0.001757867636`, entirely rate.

The MLX-selected atoms should not be down-ranked as idea-quality negatives from
this result. The planner should down-rank or block descriptor-only materializers
whose expected gain is not tied to a consumed runtime action.

## Guardrail Work

1. Exact-ready promotion should fail closed when source/candidate full-frame
   parity is true and archive bytes increase, unless the row is explicitly
   labeled `rate_only_control`.
2. IAS1 materialization should map selected cells to actual DQS1/q/selector
   runtime mutations, then prove selected-frame or full-frame output changes
   before exact eval.
3. Byte allocator dispatch economics must use realized serialized archive delta;
   `water_fill_cost_bytes` can remain a model feature but not dispatch authority.
4. Modal exact-eval dispatch should require a bound expected runtime tree/content
   SHA, not a blank expected runtime identity with provenance nearby.

## Next Consumer

Wire this into the exact-ready queue/materializer gates and byte-shaving campaign
planner as a false-positive class: expected score gain with parsed-only metadata
and unchanged decoded outputs is not exact-eval-ready.
