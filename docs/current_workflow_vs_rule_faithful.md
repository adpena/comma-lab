# current workflow vs rule-faithful accounting

This repo keeps two packaging views.

## current_workflow

Use this when describing what the published GitHub Action appears to measure today.

Properties:

- `archive.zip` is the scored artifact
- repo-side code next to `inflate.sh` may effectively be free
- repo-side public videos may be visible during evaluation

## rule_faithful

Use this when describing what a stricter interpretation would likely count.

Properties:

- any helper assets should be treated as part of the true submission burden
- a decoder should not depend on public repo-side originals being present
- model and binary bytes should be charged honestly

## Reporting rule

Every serious result should say which view it uses.
If a result only exists under `current_workflow`, say so plainly.

### Local rule-faithful estimate

When the official scorer only charges `archive.zip`, this repo may still record a local `rule_faithful` estimate by:

1. reusing the scorer's measured distortions
2. replacing the rate term with the honest byte burden for the installed runtime payload actually under test
3. recomputing the final score with the published formula

This estimate is useful for internal decisions, but it is not the official published score.

For `robust_current`, that payload is intentionally the minimal installed set:

- `archive.zip`
- `inflate.sh`
- `config.env`
- `analyze_roi.py`
