# PACT-NeRV Spine Emitters Landed

UTC: 2026-06-01T04:51:27Z

## What Changed

PACT-NeRV family archive exporters now emit the same HPRC representation spine
used by PR95/HNeRV and PACT-NeRV-VQ.  The new generic projection handles the
26-byte header grammar shared by PACT-NeRV IA3 and selector variants:

- `PIA3` -> `pact_nerv_ia3_pia3`, side channel `ego_pose_conditioning`
- `PSV2` -> `pact_nerv_selector_v2_psv2`, side channel `arithmetic_selector_k16`
- `PSV3` -> `pact_nerv_selector_v3_psv3`, side channel `rice_golomb_selector`
- `PSV4` -> `pact_nerv_selector_v4_psv4`, side channel `rle_selector`

All project to charged HPRC sections:

- decoder weights -> `decoder_qw`
- per-pair latents -> `latents_rc`
- selector/conditioning stream -> `selectors_rc`
- charged header/meta constants -> `receiver_state`
- allocation hint/no-score-authority manifest -> `rdo_plan`

## Guardrail

The spine projection is acquisition input only.  It carries false-authority
fields and cannot promote a score without archive/runtime custody, receiver
proof, and exact CPU/CUDA eval.  This preserves contest compliance while making
all PACT-NeRV variants comparable by section bytes and value-per-byte.

## Tests

Focused checks passed:

- `ruff` on the touched HPRC/PACT-NeRV files.
- `pytest src/tac/substrates/hprc/tests/test_representation_spine.py src/tac/substrates/pact_nerv_selector_v2/tests/test_pact_nerv_selector_v2.py src/tac/substrates/pact_nerv_selector_v3/tests/test_pact_nerv_selector_v3.py src/tac/substrates/pact_nerv_selector_v4/tests/test_pact_nerv_selector_v4.py -q`
  -> `61 passed`.

## Next

The acquisition queue can now compare PR95/HNeRV, packed HNeRV, PACT-NeRV-VQ,
PACT-NeRV IA3, and selector-family packets under one byte-value contract.  The
next score-lowering actuator is a full-coverage compact-base sweep under the
`178k/216k/285k` byte ceilings, with residual sidecars admitted only after
full-video scorer replay proves value-per-byte.
