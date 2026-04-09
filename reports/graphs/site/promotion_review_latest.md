# latest promotion review

## candidate

- `robust_current-long1000-h64-promoted-cpu-2026-04-09`
- Prior floor: `1.84` (`ensemble h32 + MC refine1`)
- Candidate: `1.73` (`524x394 + long1000 QAT+EMA h64 learned post-filter`)

## evidence

- local smoke:
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-smoke.json`
- authoritative scorer summary:
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-current_workflow-cpu-summary.json`
- authoritative scorer report:
  - `reports/raw/2026-04-09-long1000-h64-authoritative/robust_current-long1000-h64-current_workflow-cpu-report.txt`

## measured result

- current_workflow score: **`1.73`**
- current_workflow bytes: `864,167`
- rule_faithful estimate: `1.7947470454539947` at `966,071` bytes
- bytes changed by `-1` vs the `1.84` floor
- PoseNet improved to `0.03317023`
- SegNet moved to `0.00575544`

## review verdict

- smoke gate: passed locally
- scorer-backed local eval: passed
- honest payload still ships the learned post-filter assets in the installed runtime payload
- decision: **PROMOTE**
