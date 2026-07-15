# DAG FEED — V9 CGauge top-3 one-delta ISO configs — 2026-07-15

**FEED:** `FEED-V9-CGAUGE-432-ISO-CONFIGS-20260715`  
**Lane:** `lane_v9_cgauge_iso_configs_20260715`  
**Status:** `BUILD_COMPLETE / THREE_DRY_RUNS_PASS / PREPARED_NOT_FIRED`  
**Pointer:** submittable `0.19108`, borrowed-bank `0.18804`; both unchanged.

## State transition

The Phase-0 ARM-ISO build blocker in `P0_campaign_queue_20260715.md` is closed for the three
highest duty-to-measure levers. The result is config authority only: it does not authorize a GPU
launch, score claim, or pointer movement.

| Config ID | Duty | Typed scientific edge | Governed dry-run |
|---|---:|---|---|
| `v9_cgauge_432_taper_off` | 78.9% | remove the complete four-flag/four-LawRef `DsegAwareTaper` Lever | `220/220` flags, DSL PASS, schedule PASS, rc 0 |
| `v9_cgauge_432_horizon_iso` | 47.3% | add HWM with seven LawRefs plus derived-live boundary weight consumer | `232/232` flags, DSL PASS, schedule PASS, rc 0 |
| `v9_cgauge_432_step_iso` | 34.2% | change only the HOSC endpoint from sealed beta `3.177` to separately owned beta `8.0` | `224/224` flags, DSL PASS, schedule PASS, rc 0 |

Raw launch artifacts, constant manifests, hashes, and exact argv deltas are in
`.omx/research/v9_iso_configs_20260715_dryrun/`.

## Validation and recursive review

- Focused config/provenance/launcher suite: `187 passed, 1 skipped` in `52.39s` on local CPU.
- Fatal Ruff rules (`E9,F63,F7,F82`), `git diff --check`, JSON parsing, and dry-run artifact
  SHA-256 re-derivation: clean.
- Review tracker clean passes: `v9_iso_round1_semantic`, `v9_iso_round2_structural`, and
  `v9_iso_round3_blind_rederive`; all 12 changed Python surfaces were clean in every pass.
- A broader related-suite run produced `205 passed, 2 skipped` plus one pre-existing failure in
  unchanged `test_polyak_sizing_degenerate_is_genuinely_inert_over_the_real_loop`: the current
  epoch-budget guard rejects that legacy 3-epoch v7 schedule. Verdict scope is legacy-v7 test
  baseline only; it is not an ARM-ISO config/consumer failure, and no guard was weakened.

## Edges

```text
V9_C0 = v9_cgauge_ideal_mod19
  |
  +-- remove whole DsegAwareTaper Lever ----------------> v9_cgauge_432_taper_off
  |       compiler checks exact four-flag removal
  |
  +-- add HWM seven-scalar scientific declaration -----> v9_cgauge_432_horizon_iso
  |       parser + AST consumer + boundary receipt
  |       q=0.15 --all n600 frozen scan at ep726--> persisted resolved w_h
  |
  +-- replace beta_end 3.177 with distinct beta=8 law --> v9_cgauge_432_step_iso
          compiler checks the sole scientific argv change

each typed config --launcher ID--> governed dry-run --PASS--> PREPARED_NOT_FIRED
                                                    |
                         SSD/cache/safe-compile/GO + lane claim
                                                    |
                                                    v
                                      future governed n600 run
                                                    |
                         receiver close + exact CPU/CUDA evaluation
                                                    |
                                                    v
                                           score authority only
```

## Exact argv deltas versus `v9_cgauge_ideal_mod19`

`v9_cgauge_432_taper_off` removes, rather than zeros:

```text
--dseg-aware-taper                         PRESENT -> ABSENT
--dseg-aware-taper-strength 1.0            PRESENT -> ABSENT
--dseg-aware-taper-scale 0.0               PRESENT -> ABSENT
--dseg-aware-taper-floor 0.05               PRESENT -> ABSENT
```

`v9_cgauge_432_horizon_iso` adds:

```text
--seg-horizon-margin-weight 0.15
--seg-horizon-margin-target 0.5
--seg-horizon-margin-lo 0.3
--seg-horizon-margin-hi 0.5
--seg-horizon-row-lo 96
--seg-horizon-row-hi 288
--seg-horizon-margin-start-epoch 726
--seg-horizon-margin-derived-live
```

The first seven flags have scientific LawRefs. The Boolean selects the real boundary consumer. At
the frozen epoch-726 boundary, it scans all 600 pairs with unit HWM weight, measures raw `L_h` and
`L_o`, resolves `w_h=(0.15/0.85)*L_o/max(L_h,eps)`, writes
`hwm_v9_stage_share_boundary.v1`, and persists the resolved weight in every later checkpoint.

`v9_cgauge_432_step_iso` changes exactly:

```text
--hosc-beta-end 3.177 -> 8.0
```

The treatment endpoint is owned by `step_native_activation_edge_optimality_v1`; it does not mutate
the sealed control declaration.

## Refusal graph

- An ISO Lever with an absent real argparse flag, absent executable `args.<dest>` consumer, absent
  LawRef/compiler match, or absent receipt schema raises `ISO provenance/consumer REFUSE`.
- TAPER-off refuses unless the ideal program has exactly one fully custodied `DsegAwareTaper` owner.
- Each compiler diffs its emitted scientific argv against the freshly compiled control and refuses
  any added, dropped, or changed flag outside the registered one-Lever contrast.
- HORIZON resume past the boundary refuses if the checkpoint lacks the frozen resolved weight.

## Triality

- **DSL:** the three named typed configs, complete Lever ownership, launcher dispatch, real parser
  validation, expected-active-lever manifests, and lever-registry MAPPED coverage.
- **DAG:** this FEED plus `dry_run_receipt.json` and the six hashed dry-run artifacts.
- **Equations:** `dseg_aware_fourier_taper_reweight_v1`,
  `horizon_weighted_margin_hinge_v1`, and
  `step_native_activation_edge_optimality_v1` are executable LawRef evaluators.

## Verdict scope and remaining blockers

`BUILD_COMPLETE` applies only to the three ARM-ISO typed config/consumer paths. It is not a family
efficacy verdict and not a fire-now authorization. In this isolated tree, `gt_n600.npz` was absent
and the safe-compile certificate reported `fingerprint_ok=false`; both are real-launch prerequisites,
not config-build failures. The campaign's SSD-root/memory-waterfall closure remains owned by
ARM-INFRA/main and was not reverified here. No GPU, provider, evaluator, archive, or score path ran.
