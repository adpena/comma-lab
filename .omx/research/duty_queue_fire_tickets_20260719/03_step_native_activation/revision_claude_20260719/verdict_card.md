# 03_step_native_activation — revision_claude_20260719

Verdict: `READY-FOR-GAUNTLET`

Measured at HEAD `679e78ab0352`. Containment: compile/static evidence only — NO launch occurred; pointer 0.1910828242 [contest-CPU] UNMOVED (MEANS).

| arm | program | full_dsl_compile_hash | typed_config_hash | schedule gate |
|---|---|---|---|---|
| off | `v9c3_duty_step_off` | `8c0e962064e4587741f6b24851b0d9f7ebcfd0c38553a73604260ca05dfdaad8` | `551e23903e719e408177d5a2524e8a00f4be206c195fdff26615a10f86fde121` | rc0, 6 verdicts, 0 violations |
| on | `v9c3_duty_step_on` | `028bd07d738ab4f21a68d8c38abcbcec9f611228507e2e13a2a4b78a6aef4b56` | `12548534eac4942e7b20c6583ac36d45336862d6091fe566bc8691e8e298a317` | rc0, 6 verdicts, 0 violations |

Signed OFF->ON argv delta:
```json
{
  "--hosc-beta-end": [
    "4.0",
    "8.0"
  ]
}
```

Pre-registered thresholds (derived from the donor run's measured verdict noise, sigma=1.876308e-05/point):
- primary K=4 @ ep 775-850: h95 = 2.600382e-05 d_seg
- secondary K=10 @ ep 775-1000: h95 = 1.644626e-05 d_seg
- PAYS/HURTS/NEUTRAL/INDETERMINATE rules + admissibility preconditions + positive-control sentinels: see compiled_pair.json

