# ddm_mi1 measurement integrity receipt - 2026-08-04

Axis: `[scorer-free measurement-integrity audit]`. No scorer job was run. No score
claim is made. Pointer moved: false.

## #931 pose subset-bias ratios

Instrument: reparsed
`/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl`
at sha256 `d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5`,
field `d_pose_shipped_f16`, 600 rows.

What it reports:

| n | prefix/population ratio | seeded-random p01..p99 band | verdict |
|---:|---:|---:|---|
| 8 | 1.2789847387x | 0.0114704407..4.1964921756 | MATCHED |
| 24 | 2.5354755796x | 0.1524377507..2.7778327062 | MATCHED |
| 48 | 2.6401816892x | 0.3322813323..2.0900912479 | DIFFERENT_POPULATION |
| 64 | 2.6477688500x | 0.3970243767..1.8689177314 | DIFFERENT_POPULATION |
| 96 | 4.2067709320x | 0.4959428276..1.6674383927 | DIFFERENT_POPULATION |

The quoted 2.54/2.64/2.65/4.21 pose ratios are re-derived and not withdrawn.
The correction is the smaller-sample interpretation: n=8 does not bank a
population verdict, and n=24's 2.535x ratio is still inside the current p01/p99
random band. The 60-pair block means reproduce the skew mechanism:
`0.4064574677, 0.8162419729, 0.0828458373, 0.0413443050, 0.0254981230,
0.0433979904, 0.0102753873, 0.0134305197, 0.1091590879, 0.0464385006`;
max/min is `79.4366139854x`.

What consumers believed: the old selector test used a synthetic high-bias pose
fixture and claimed the pose bias was caught even at n=8. That was not the live
receipt. It is now replaced by a receipt-derived fixture and tests pinning the
n=8/n=24 nuance.

Structural cure: `src/tac/tests/fixtures/subset_selection/pfs1_d2_pose_population.json`
plus `src/tac/tests/test_subset_selection.py` re-derive the live ratios. The
subset-selection docs and staged-diff guard now cite the receipt hash.

## #885 git log line-count undercount

Instrument: current shell and `rtk` comparison at `HEAD`
`5f54e74c1ee90546d03b7914166c07cff5fca820`.

What it reports:

| command | result |
|---|---:|
| `git log --oneline \| wc -l` | 14065 |
| `git rev-list --count HEAD` | 14065 |
| `rtk git log --oneline \| wc -l` | 50 |
| `rtk git rev-list --count HEAD` | 14065 |

The active undercount is runner-specific: plain zsh does not reproduce it, but
`rtk` does, with a `281.3x` undercount here.

What consumers believed: historical docs and prior receipts used the `git log |
wc -l` idiom as if formatted log rows were a count authority. In this repository
scope, the visible consumers are existing research notes and worktree mirrors,
not live Python code paths.

Structural cure: `src/tac/measurement_integrity.py` now exposes
`find_git_log_wc_count_consumers` and `assert_no_git_log_wc_count_consumers`.
The accepted instrument is `git rev-list --count <range>`.

## #877 rounded Final score consumers

Instrument: static audit of `Final score` parsing consumers.

What it reports and what consumers believed:

| path | prior belief | cure |
|---|---|---|
| `src/comma_lab/evaluate.py` | `current_workflow_score` used parsed upstream display score | recompute from pose, seg, rate |
| `experiments/contest_eval.py` | `parse_report` stored display score as `score` | store `reported_final_score_display_rounded`; set `score` from `score_recomputed_from_components` |
| `src/tac/eval/auth_eval.py` | `ReportMetrics.best_score` preferred parsed display score | prefer `computed_score` from components |

Structural cure: focused tests in `src/tac/tests/test_measurement_integrity.py`
pin recomputation and display-score separation.

## #875 scope censoring / subset defaults

Instrument: `src/tac/subset_selection_gate.py:scan_repo` on tracked in-scope
Python files.

What it reports: `207` violations across `136` files, denominator `10754`
tracked in-scope `.py` files. This is the live migration debt.

What consumers believed: a silent subset was sometimes read as "unknown" or a
legitimate small verdict. The actual default is video-order prefix, and prefix
bias changes sign by axis: easier on seg, harder on pose.

Structural cure: the existing staged-diff hook remains strict for newly-added
silent subset slices, while the whole-repo sweep stays a debt report until the
207-site migration is explicitly worked.

## Boundary

No exact eval, no scorer, no archive row, no promotion. The own-vehicle frontier
therefore remains `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`; the
contest pointer remains `0.1910828242` borrowed/harvest-only.
