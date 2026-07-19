# 02_horizon_weighted_margin — revision_claude_20260719

Verdict: `READY-FOR-GAUNTLET`

Measured at HEAD `679e78ab0352`. Containment: compile/static evidence only — NO launch occurred; pointer 0.1910828242 [contest-CPU] UNMOVED (MEANS).

| arm | program | full_dsl_compile_hash | typed_config_hash | schedule gate |
|---|---|---|---|---|
| off | `v9c3_duty_hwm_off` | `c49087ce7077c9abe3f8c06dafe85a909b6867da3319ac5c33a4d2ad24fbcb1d` | `9169f422bb6c901811b168fa299b8c4954e4d7ac0cd39489ddd120d58a00f29e` | rc0, 6 verdicts, 0 violations |
| on | `v9c3_duty_hwm_on` | `8ac7b6cd816f4d21d42c477fa9abd0c21904425cbf14dccd5ac627c150565137` | `77cabaa0d796765fe0519b3c6510ce15bcb2d0d0daa1bd115a2f2936357d8bc1` | rc0, 7 verdicts, 0 violations |

Signed OFF->ON argv delta:
```json
{
  "--seg-horizon-margin-derived-live": [
    null,
    true
  ],
  "--seg-horizon-margin-hi": [
    null,
    "0.5"
  ],
  "--seg-horizon-margin-lo": [
    null,
    "0.3"
  ],
  "--seg-horizon-margin-start-epoch": [
    null,
    "753"
  ],
  "--seg-horizon-margin-target": [
    null,
    "0.5"
  ],
  "--seg-horizon-margin-weight": [
    null,
    "0.15"
  ],
  "--seg-horizon-row-hi": [
    null,
    "288"
  ],
  "--seg-horizon-row-lo": [
    null,
    "96"
  ]
}
```

Pre-registered thresholds (derived from the donor run's measured verdict noise, sigma=1.876308e-05/point):
- primary K=4 @ ep 775-850: h95 = 2.600382e-05 d_seg
- secondary K=10 @ ep 775-1000: h95 = 1.644626e-05 d_seg
- PAYS/HURTS/NEUTRAL/INDETERMINATE rules + admissibility preconditions + positive-control sentinels: see compiled_pair.json

