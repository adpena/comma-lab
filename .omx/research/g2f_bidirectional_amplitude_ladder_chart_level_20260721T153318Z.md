# Task #578 G2f chart-level amplitude ladder addendum

`lane_id=lane_g2f_chart_bidirectional_amplitude_ladder_578_20260721` ·
`verdict=MEASURED_G2F_CHART_TRUST_RESCUES_PIXEL_EMPTY_N64_FAMILY_OPEN` ·
`research_only=true` · `[macOS-CPU advisory]` · `score_claim=false` ·
`promotion_eligible=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Outcome first

The operator's level-disambiguation hint survived an exact contiguous n64
measurement, but only locally. A coherent openpilot centerline-intercept chart
move selected `6/64` pairs versus `5/64` for the preserved pixel-level rank-4
chart. Four pairs were chart-only rescues (`0, 34, 37, 46`), three were
pixel-only (`24, 31, 63`), two selected at both levels (`22, 30`), and `55`
selected at neither level.

The correct alphabet verdict is therefore pair-conditional:

- prefer a chart symbol for the four measured chart-only rescues, pending a
  counted receiver packet;
- retain pixel structure for the three measured pixel-only pairs;
- compare receiver-closed bytes and hard-oracle value on the two overlapping
  pairs;
- infer no useful symbol for the 55 pairs selected at neither level.

This is not a receiver admission. The chart run was measurement-only: no
chart-symbol packet was built, correction bytes stayed zero, and the hard
Seg/Pose/rate gate did not run. Therefore it does not supersede the preserved
pixel-level D3 result (`5/5` QPs infeasible), does not authorize n600, and
does not move the pointer.

## D1 — coherent chart-coefficient response

The numeric ladder remained `[0.5, 1, 2, 4, 8, 16]`, but its chart unit is
native scorer centerline pixels, not a physically equivalent pixel-RGB unit.
For each pair the runner selected the support-maximizing openpilot lane line,
perturbed `LaneLine.centerline_coeffs[-1]`, normalized coefficient magnitude
so the maximum active-row centerline displacement equaled the rung amplitude,
rasterized the full coherent AA-SDF coverage change, applied the counted frozen
Lane-minus-Road palette action, then measured through exact factor-2 R and the
native CPU-Torch SegNet.

MEASURED n64 chart custody:

- `384` paired chart/rung observations and `768` signed branches;
- `696` class/bucket trust regions, `28` usable;
- `6/64` pairs selected at least one fully usable chart rung;
- no zero-applied branch at any rung;
- target-class 0: `206/906` sign-consistent responses (`0.2273730684`);
- target-class 1: `268/1116` sign-consistent responses (`0.2401433692`);
- nonexclusive refusals: `625` sign/zero and `644` relative-residual.

| centerline amplitude | selected pairs at rung | usable trust regions | mean changed pixels | negative / positive uint8 saturations |
|---:|---:|---:|---:|---:|
| 0.5 | 4 / 64 | 17 / 116 | 162.359375 | 682 / 685 |
| 1.0 | 1 / 64 | 7 / 116 | 201.171875 | 2298 / 2624 |
| 2.0 | 1 / 64 | 4 / 116 | 278.25 | 5236 / 5964 |
| 4.0 | 0 / 64 | 0 / 116 | 431.234375 | 10809 / 12402 |
| 8.0 | 0 / 64 | 0 / 116 | 714.6484375 | 20853 / 23473 |
| 16.0 | 0 / 64 | 0 / 116 | 887.078125 | 25773 / 26068 |

The chart knee is the smallest rung, `0.5` native centerline pixels. Usability
decays as coherent support and saturation grow. This differs from the
pixel-level knee at amplitude `1.0`; the numeric amplitudes are shared only to
make the level comparison explicit, not to assert equal physical actions.

## D2 — level attribution

MEASURED exact pair partition:

| level outcome | count | pair indices |
|---|---:|---|
| chart only | 4 | `0, 34, 37, 46` |
| pixel only | 3 | `24, 31, 63` |
| both | 2 | `22, 30` |
| neither | 55 | receipt-custodied complement |

The comparison predicate is identical at both levels: a pair selects only when
every effective direction at that level has a usable rung. The chart has one
coherent coefficient direction per pair; the pixel level has up to four local
rank directions. That structural difference is part of the result and forbids
treating `6/64` versus `5/64` as an equal-dimensional global tournament.

## D3 / D4 — receiver gate and stop rule

`D3=NOT_RUN_MEASUREMENT_ONLY` and `D4=NOT_RUN_D3_NO_ADMISSION`.

The next discriminating build is an n16 counted chart-symbol packet for the
rescued pair 0, with deterministic parse/re-encode identity, reconstructed
coefficient custody, full raster/R closure, hard Seg/Pose oracle, and measured
marginal score-units per byte against `25/37,545,489`. Only a positive
receiver-closed n16 admission can authorize the same packet path at n64. Only
a surviving n64 receiver admission can justify n600. Empty or negative receiver
rows grant no spend authority.

## REUSE MANIFEST delta

| Surface | Disposition | Evidence |
|---|---|---|
| `src/tac/optimization/realized_secant_custody.py` | EXTENDED IN PLACE | typed chart branch/rung custody, shared odd/even rederivation, strict receipt validation |
| `tools/measure_realization_g2_lattice.py` | EXTENDED IN PLACE | resumable coherent chart measurement, level comparison, immutable pair/chunk checkpoints |
| G2f pixel n64 receipt | CONSUMED, NOT REMEASURED | exact level comparator; file SHA `0a09b7b5022ff64eebc54d086f00c89378d7eb7091c5963cf1056120469bc38e` |
| openpilot static/lane charts and frozen palette | REUSED | counted base packets are copied and parse-back checked under existing custody |
| receiver packet / admission helper | NOT CLAIMED | measurement intentionally stops before a counted chart-symbol codec exists |
| new measurement CLI | NOT CREATED | existing G2 runner received one mutually exclusive typed mode |
| measurement-time source bytes | PRESERVED AS PATCH | `g2f_chart_measurement_runtime_sources_20260721T153318Z.patch` applies to base `c9abc61b2e` and reproduces every as-run source hash |

## Custody and verification

Full external receipt:
`/Volumes/VertigoDataTier/pact/evidence/g2f_chart_amplitude_20260721/receipt.json`

- file SHA-256:
  `47d3ca538f1b876f7639223a1a9a7714b7db2083eaa0971936b9a43a1e6d0d04`;
- embedded canonical receipt SHA-256:
  `2e0ccc5a6822b584ad66d7dc522551684ef20c358628b20ebdbb9bdd02cfe120`;
- config SHA-256:
  `8b357c1d9c7c7ac5257cc67e996851597d3989f3c2921e9c60e28b319550e055`;
- receipt bytes: `21,067,321`;
- external tree: `87` files, `63,328,634` bytes, sorted
  `sha256 + relative-path` manifest hash
  `5abc13b8303f95c02b6c3774ca63a405b2f24b9d71bc454c7f6c53438ff62c33`;
- immutable stages: `64`; prefix checkpoint: `prefix_n64.json`;
- top-level and nested receipts are byte-identical;
- strict receipt rederivation returned the embedded canonical SHA;
- measurement-time source patch SHA-256:
  `e2dca7957ee3d96b9d5a116077ee3c2fcf7ede52e28c5c640b62cfaf35a16d5d`
  (`80,646` bytes), applied and hash-checked against base `c9abc61b2e`;
- reconstructed as-run implementation SHA-256 values match the receipt:
  `df17d05ce2cff439587972a7f5ca2e7938a1a614d81f98c5418563166e0c93e0`
  for `realized_secant_custody.py` and
  `887147e369c05c2663a48f005d48b9f4332508d291f69ef22be251fc355855b1`
  for the measurement runner;
- final formatted-tree verification: `37 passed`, Ruff check/format check,
  Python compile, strict receipt validation, and `git diff --check` passed.

## STORES CONSULTED

Delegated checkpoint and authority file; preserved G2f pixel n64 receipt and
stages; openpilot static/lane charts; frozen scorer palette and weights; n600 GT
cache; G1 motion custody; exact R kernel; per-arm and broadcast inboxes; lane,
checkpoint, and canonical-equation state; the original G2f memo and DAG feed.

## Triality and landing boundary

- DSL/code: typed coherent-chart branch/rung custody and an explicit
  `--chart-amplitude-ladder` mode expose level as a first-class measurement
  choice while retaining the existing resumable runner.
- DAG: the new feed routes chart-only, pixel-only, overlap, and neither sets to
  different packet/admission actions.
- Equations: the hard admissibility equation is unchanged and receives no false
  empirical anchor because the chart packet and receiver gate were not run; the
  strict typed receipt is the measurement record pending that gate.

MAIN must review the branch diff, receipt/tree custody, pair-conditional
alphabet verdict, and the explicit no-n600 boundary before merging. Until that
review, this addendum is not repository truth.
