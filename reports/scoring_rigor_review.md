# scoring rigor review

## purpose

Senior-engineer review of score/evidence discipline.

## authoritative promotion evidence

- local CPU scorer path is the promotion authority
- canonical live summary/report paths:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`

These now reflect the promoted **`2.05`** learned-postfilter floor.

## packaging views

- `current_workflow` = scorer-backed promotion view
- `rule_faithful` = local estimate only

## new rigor rule

Every future promoted floor must include a written critical review against:
- official contest scoring formula
- official evaluation path
- byte accounting
- rule-faithful interpretation
- known ffmpeg/evaluator bug classes
- pre-scorer smoke gate for raw output count, geometry, and sampled RGB semantic sanity

## current promoted floor

- Track B floor: **`2.05`**
- Config: `522x392 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / learned int8 post-filter`
- Current-workflow bytes: `861,986`
- Rule-faithful estimate: `2.0778631822069484` at `896,432` bytes
- Installed runtime payload:
  - `archive.zip`
  - `inflate.sh`
  - `inflate.py`
  - `inflate_postfilter.py`
  - `inflate_grain_mask.py`
  - `postfilter_int8.pt`
  - `config.env`
  - `analyze_roi.py`

## current-cycle lesson

- The new learned post-filter was the first decode-side lane to beat the 2.08 floor in a scorer-backed run.
- The main rigor corrections in this pass are now:
  - package uses the requested upstream root
  - eval clears stale inflated raws
  - rule-faithful charges the installed payload actually under test
  - smoke and eval share the same clean-run assumption
  - smoke now checks semantic sample MAE on first/middle/last frames instead of stopping at geometry-only validation
  - flat-path colorspace/range handling is explicit instead of implicit
  - submission locking now prevents overlapping package/smoke/eval operations on the same track/upstream root
  - ffmpeg capability guards now fail fast when a host lacks the required AV1 encoder or explicit color-contract scale options


## 2026-04-07 learned post-filter promotion

- New promoted floor: **2.05** at `861,986` bytes
- The learned post-filter ships honestly in the installed runtime payload and was verified by local smoke, BAT00 smoke, and a local scorer-backed run.
- Grain-mask was fully verified at `2.30` and remains a reject.
