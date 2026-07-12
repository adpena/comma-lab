# Exact JRD coefficient-prefix probe — fixture GO; V9/v8 NEEDS-MORE

**OUTCOME.** `[MEASURED] [macOS-CPU advisory] [recovery-written-UNREVIEWED]` The single
frozen v7.5.2 pair-0 planning fixture contains removable coefficient precision. The exact
combined replay reduced the archive from **83,905 B to 81,154 B**, saving **2,751 B**, while
`d_seg` changed from `0.023157755533854168` to `0.0218505859375` and `d_pose` changed from
`116.59830629690003` to `92.42743674059255`. This is fixture-level **GO**.

**ROUTE VERDICT.** `[DERIVED] [recovery-written-UNREVIEWED]` **NEEDS-MORE**, not V9/v8 GO.
The content-addressed inventory found zero typed, hash-bound, sealed, non-live V9/v8 LVLS1
payloads. The fixture is v7.5.2 staged for a V9 apply-pass, only pair 0 was scored, and
early/boundary/late saved regimes were not available. The canonical probe outcome is `DEFER`;
the canonical task blocker is `eligible_nonlive_v9_v8_payload_missing_or_unresolved`.

**STORES CONSULTED:** loaded the retrieval-first corpus query over research, equations, memory,
DAG, council, tasks, and docs; the sealed fixture; canonical receiver/R/scorer contracts; local
and connected-SSD payload inventories; `CLAUDE.md`; `AGENTS.md`; the operating manual; the two
named route memos; the canonical task and probe ledgers. Deliberately not loaded: the protected
live V9 run and the full 5 GB GT cache.

## Measured receipt

- `[MEASURED]` Source video: `upstream/videos/0.mkv`; 37,545,489 B; SHA-256
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.
- `[MEASURED]` Sealed checkpoint SHA-256:
  `bbd0567439298c7c6eac236aa7215e39d5190703540eeaf4a71fc4839af931b0`.
- `[MEASURED]` Baseline archive SHA-256:
  `6bde3b1749c0a790f747e24254d308c45a063bcad7ace1376d4463c29f5ec10f`.
- `[MEASURED]` Selected archive SHA-256:
  `6b86c273c7c114768b60c8d834d9da59d8390128513dc8cdab2f3f1097efb48b`.
- `[MEASURED]` Complete response surface: 18 sections; 71,223 coefficients; two prefix
  families; eight nonzero planes; 288 exact receiver/R/scorer rows.
- `[MEASURED]` Removed raw precision: 40,416 bit positions, or 5,052 raw-byte equivalents.
  The counted ZIP reduction is 2,751 B; raw bit count is not substituted for archive bytes.
- `[MEASURED]` Positive repeat noise floor: `delta_d_seg=0`; `delta_d_pose=0`.
- `[MEASURED]` Negative all-zero control: 926 B; `d_seg=0.5069224039713541`;
  `d_pose=193.05575882587024`. The meter separates a broken payload.
- `[MEASURED]` Baseline and selected receiver canaries both report exact uint8 frame equality
  under the canonical NumPy-fp32 receiver plus canonical R.

## Per-section last-safe planes

`[MEASURED]` Every numeric table cell below comes from an exact receiver/R/scorer response row.
The table reports the coarsest component-safe plane for each family. `none` means no nonzero
plane satisfied both zero-tolerance component guards. A negative byte saving means that the
coarsest safe plane enlarged the exact ZIP and was therefore not rate-admissible.

| Section | Uniform: plane / ZIP bytes saved / delta d_seg / delta d_pose | Dead-zone: plane / ZIP bytes saved / delta d_seg / delta d_pose |
|---|---:|---:|
| `in_proj.weight` | none | none |
| `in_proj.bias` | 4 / 31 / -0.00017293294270833565 / -1.687846779320239 | none |
| `film.weight` | none | none |
| `film.bias` | 1 / 55 / -0.00013224283854166782 / -4.973376567803427 | none |
| `hidden.0.weight` | none | none |
| `hidden.0.bias` | 5 / 75 / -0.0016428629557291678 / -9.551886263540936 | none |
| `hidden.1.weight` | 2 / 1,426 / -0.00034077962239583565 / -12.505517277362344 | none |
| `hidden.1.bias` | none | none |
| `hidden.2.weight` | none | none |
| `hidden.2.bias` | none | none |
| `hidden.3.weight` | 2 / 1,073 / -0.000035603841145835646 / -4.13093222526021 | none |
| `hidden.3.bias` | none | none |
| `out_sdf.weight` | 5 / 217 / -0.0017547607421875 / -3.6804504833241936 | none |
| `out_sdf.bias` | 8 / -13 / -0.00013224283854166782 / -3.327567823329261 | 8 / -13 / -0.00013224283854166782 / -3.327567823329261 |
| `out_tex.weight` | none | none |
| `out_tex.bias` | none | none |
| `palette` | none | none |
| `code_scored_pair_prefix` | none | none |

`[MEASURED]` The combined allocator accepted `hidden.1.weight`, `hidden.3.weight`,
`out_sdf.weight`, `film.bias`, and `in_proj.bias`. It rejected `hidden.0.bias` because the
combined exact replay breached the Seg guard and did not shrink the current ZIP. It rejected
`out_sdf.bias` because the combined exact ZIP grew. Individual byte deltas are not additive
under compression; the selected combined replay is the authority for the 2,751 B total.

## Control law and derivation

`[FROM-LITERATURE]` Just-recognizable distortion is imported only as the idea that a fixed
perception model defines a maximum tolerable coding distortion: Wuyuan Xie, Zhenming Li, Ye Liu,
Jian Jin, Yun Song, and Miaohui Wang (2026), *The Last Byte: Learning Just Enough for
Machine-Oriented Image Compression*, DOI `10.1609/aaai.v40i19.38635`. The DOI and named paper
resolve on the official AAAI abstract page. MVR-Net, its frame/VVC QP object, and paper code were
not copied. `[DERIVED]` The mission's license retrieval found no official JRD/MVR-Net software
license, so this landing is a clean-room implementation from the published method description
and Pact's existing receiver/scorer contracts only.

`[FROM-LITERATURE]` Nested analytic dead-zone quantization for Laplacian latent sources is imported
from Shaohui Li, Han Li, Wenrui Dai, Chenglin Li, Junni Zou, and Hongkai Xiong (2023), *Learned
Progressive Image Compression With Dead-Zone Quantizers*, DOI
`10.1109/TCSVT.2022.3229701`. The DOI/title and six-author record resolve on the authors' Shanghai
Jiao Tong University publication surface. No paper code was copied.

`[DERIVED]` The Pact control law is an event-conditioned tested predicate, not a borrowed paper
threshold. For section `s`, family `f`, and plane `k`, a candidate is component-safe exactly when
`d_seg(s,f,k) <= d_seg_0 + epsilon_seg` and
`d_pose(s,f,k) <= d_pose_0 + epsilon_pose`. Both epsilons are constants set to the measured
positive-repeat floor, zero in this run. The last-safe plane is the largest safe `k`. A combined
step is admitted only if a fresh exact replay is component-safe against the sealed baseline and
its exact ZIP is strictly smaller than the current combined ZIP. Enumeration completes after
plane 8, so the search has a completion guarantee.

`[DERIVED]` Falsifier before build: fixture `NO-GO` if no combined replay both saves at least one
exact ZIP byte and passes both component guards. That falsifier did not fire. The V9/v8 family
falsifier is not adjudicated because the required payload/regime evidence is missing.

## Boundaries, risk, and triality

- `[UNKNOWN]` Transfer to an actual typed V9/v8 coefficient payload.
- `[UNKNOWN]` Early, boundary, and late saved-regime survival.
- `[UNKNOWN]` Across-seed variance; this is a single frozen-payload spine.
- `[UNKNOWN]` Full-600-pair result.
- `[UNKNOWN]` Contest-CPU and contest-CUDA behavior.
- `[MEASURED]` `upstream/evaluate.py` was not run; `score_claim=false`;
  `promotion_eligible=false`; the contest pointer is unmoved.
- `[MEASURED]` The run used local macOS CPU only. No paid GPU, Modal, cloud dispatch, training,
  live-trainer edit, or protected-live-run access occurred.
- `[DERIVED]` Verdict scope is INSTANCE for the fixture. It is not a formulation-level V9/v8
  verdict and does not close the JRD/prefix family.
- `[DERIVED]` Review status at write time is `recovery-written-UNREVIEWED`. Any later clean review
  receipt must name its independent pass count; this memo does not pre-claim it.
- `[DERIVED]` Triality: equations = `jrd_exact_coefficient_prefix_selection_v1`; DAG = the
  `FEED-jrd-coeff-prefix-pair0` append; DSL = N/A-with-reason because this is an offline frozen-
  payload receiver/byte-allocator oracle, not a trainer, curriculum, launch, or actuator change.

## Durable artifacts

- Measurement receipt: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/measurement_receipt.json`.
- Complete response surface: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/section_precision_response_curves.json`.
- Per-section summary: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/stage_03_per_section_summary.json`.
- Allocator execution: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/allocator_planning_input.json`.
- System integration receipt: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/system_integration_receipt.json`.
- Preserved resume surface: `experiments/results/jrd_coeff_prefix_probe_20260712T221747Z/resume/`.
- Implementation commits before measurement: `144d724fc5`; `be4775ed3c`.
