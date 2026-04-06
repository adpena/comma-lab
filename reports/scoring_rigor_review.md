# scoring rigor review

## purpose

Senior-engineer review of score/evidence discipline.

## authoritative promotion evidence

- local CPU scorer path is the promotion authority
- canonical live summary/report paths:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`

These now reflect the promoted **`2.12`** AV1 floor.

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
- pre-scorer smoke gate for raw output count/geometry

## current promoted floor

- Track B floor: **`2.12`**
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp / explicit bt709-tv encode / explicit rgb24(pc) decode`
- Current-workflow bytes: `864486`
- Rule-faithful estimate: `2.1418040615200598` at `897745` bytes
- Installed runtime payload:
  - `archive.zip`
  - `inflate.sh`
  - `config.env`
  - `analyze_roi.py`

## current-cycle lesson

- The explicit color-contract hardening was not just defensive cleanup; it materially improved score from `2.18` to `2.12`.
- The main rigor corrections in this pass are now:
  - package uses the requested upstream root
  - eval clears stale inflated raws
  - rule-faithful charges the installed payload actually under test
  - smoke and eval share the same clean-run assumption
  - flat-path colorspace/range handling is explicit instead of implicit
