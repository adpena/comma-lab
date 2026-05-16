# L5 v2 TT5L Probe Template Emitted

Date: 2026-05-16
Author: codex
Lane: time_traveler_l5_autonomy / L5 v2 staircase
Status: landed
Score claim: false
Promotion eligible: false
Ready for exact eval dispatch: false

## Context

After the TT5L Dykstra feasibility artifact and contest full-frame side-info
consumption proof both validate, the L5 v2 planner advances to:

`emit_c1_z5_tt5l_probe_template`

The template is deliberately a control-plane artifact, not a score artifact. It
requires observations for all three candidates before architecture lock:

- `c1_world_model_foveation`
- `z5_predictive_coding_world_model`
- `time_traveler_l5_autonomy`

Each observation requires paired `contest_cpu` and `contest_cuda` axis evidence,
artifact/log custody, archive/runtime hashes, component score fields, and
byte-closed side-info/archive predicates.

## Command

```bash
.venv/bin/python tools/probe_l5_v2_staircase_disambiguator.py \
  --emit-template \
  --output-json .omx/research/l5_v2_probe_template_20260516_codex.json
```

## Artifact

- Template:
  `.omx/research/l5_v2_probe_template_20260516_codex.json`
- Template SHA-256:
  `525c4c7483b554ce8a3103b22e9b0f0d156007a7738aa11f4367663c40a7ef97`
- Schema: `tac_l5_v2_probe_disambiguator_v1`

## Next Gate

Populate the template from measured paired CPU/CUDA probe artifacts. The
disambiguator must remain fail-closed until all required candidates are
represented by eligible paired evidence; a single good TT5L row is not enough
to lock the architecture if C1/Z5 are missing or ineligible.
