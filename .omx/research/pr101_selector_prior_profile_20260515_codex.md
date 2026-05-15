# PR101 Selector Prior Profile

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence axis: pair-row proxy/advisory for component risk; exact CPU only for FEC6 anchor/rate math
- FEC6 exact CPU score: `0.1920513168811056`
- same-component bytes needed for `<0.192`: `78`

## Byte Accounting

- FEC6 archive bytes: `178517`
- source archive bytes if selector is removed and computed from source: `178258`
- archive bytes saved vs FEC6: `259`
- removed selector payload bytes: `249`
- removed wrapper overhead bytes: `10`
- score if FEC6 components were unchanged: `0.19187885941224697`
- allowable component delta for `<0.192`: `0.00012114058775303249`

## Rule Search

- candidates tested: `25541`
- plausible `<0.192` rows under proxy + compliance filter: `0`

| rank | family | rule | estimated CPU | component delta | mismatches | source bytes | risk | verdict |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | threshold_stump | `threshold_t127` | 0.193029229046 | 0.001150369633 | 455 | 74 | low | blocked_by_component_risk |
| 2 | threshold_stump | `threshold_t131` | 0.193029564841 | 0.001150705429 | 456 | 74 | low | blocked_by_component_risk |
| 3 | threshold_stump | `threshold_t130` | 0.193029827070 | 0.001150967658 | 456 | 74 | low | blocked_by_component_risk |
| 4 | threshold_stump | `threshold_t139` | 0.193030501491 | 0.001151642079 | 456 | 74 | low | blocked_by_component_risk |
| 5 | threshold_stump | `threshold_t140` | 0.193030506272 | 0.001151646860 | 456 | 74 | low | blocked_by_component_risk |
| 6 | threshold_stump | `threshold_t133` | 0.193030552808 | 0.001151693396 | 456 | 74 | low | blocked_by_component_risk |
| 7 | threshold_stump | `threshold_t138` | 0.193030817769 | 0.001151958357 | 457 | 74 | low | blocked_by_component_risk |
| 8 | threshold_stump | `threshold_t134` | 0.193030902983 | 0.001152043571 | 457 | 74 | low | blocked_by_component_risk |
| 9 | threshold_stump | `threshold_t141` | 0.193030992943 | 0.001152133531 | 457 | 74 | low | blocked_by_component_risk |
| 10 | threshold_stump | `threshold_t142` | 0.193031087299 | 0.001152227887 | 458 | 74 | low | blocked_by_component_risk |

## Conclusion

- any rule plausibly beats `<0.192`: `false`
- best rule: `threshold_t127` (`threshold_stump`)
- best estimated CPU score: `0.19302922904569675`
- best component delta vs FEC6 proxy: `0.001150369633449791`
- best selector mismatches: `455`
- blocker: pair-index-only tiny rules do not preserve enough of the FEC6 selector's component gain; the rate saving is large enough, but component proxy loss is roughly an order of magnitude above the CPU allowance

## Compliance Risks

- A true per-pair selector table in runtime source is forbidden and not profiled as a valid candidate.
- Periodic/bucket source tables are flagged medium/high once they become selector-like rather than general rules.
- Even a low-risk formula rule needs exact CPU/CUDA replay before any score or promotion claim.

