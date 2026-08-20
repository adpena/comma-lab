# DDM SN1 v19c residual error-source budget

Axis: `[macOS-CPU frozen-SegNet advisory]`. `score_claim=false`; pointer unmoved.

| source | stratum | errors | global d_seg | conditional error rate |
|---|---:|---:|---:|---:|
| DESCRIBED_BUT_REALIZATION_LOST | Road | 885,232 | 0.007504204644 | 0.032299431321 |
| DESCRIBED_BUT_REALIZATION_LOST | Undrivable | 5,029 | 0.000042631361 | 0.000086093435 |
| DESCRIBED_BUT_REALIZATION_LOST | MyCar | 2,449 | 0.000020760430 | 0.000081651000 |
| NEVER_DESCRIBED | Road | 521,250 | 0.004418690999 | 0.019018831873 |
| NEVER_DESCRIBED | Undrivable | 216,840 | 0.001838175456 | 0.003712169498 |
| NEVER_DESCRIBED | MyCar | 0 | 0.000000000000 | 0.000000000000 |
| STRUCTURALLY_HARD_IRREDUCIBLE | Road | 528,658 | 0.004481489393 | 0.019289127329 |
| STRUCTURALLY_HARD_IRREDUCIBLE | Undrivable | 71,565 | 0.000606664022 | 0.001225149466 |
| STRUCTURALLY_HARD_IRREDUCIBLE | MyCar | 34,788 | 0.000294901530 | 0.001159850953 |

Total: **2,265,811 errors**, global d_seg contribution **0.019207517836**.

The three sources are exclusive and exhaustive at the SHA-pinned v19c endpoint. `STRUCTURALLY_HARD_IRREDUCIBLE` is scoped only to the current semantic program plus the tested DV1 `spline_plus_events` extension.
