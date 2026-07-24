# DDM SN1 five-type derivation addendum

Date: 2026-07-24
Lane: `ddm_sn1_segnet_telemetry_asymmetry`
Axis: `[macOS-CPU frozen-SegNet+PoseNet advisory]`
Authority: `research_only=true`, `score_claim=false`, pointer unmoved
Landing: isolated Codex worktree; MAIN landing review required

## Why this addendum exists

The settled SN1 n600 receipt predates the five-type amendment. Its measurement
bytes remain historical and immutable. This addendum binds those bytes by SHA
to representation type, L1-L5 home, and the recursion level in
`upstream/evaluate.py` that generates the type. It does not rerun a scorer,
change a result, or manufacture a new score.

At fire time, `TypedStreamTag` was absent from
`tac.optimization.ddm_min_description_contract`. The addendum therefore does
not fork an enum. It validates against the already-landed canonical
`REPRESENTATION_TYPES` tuple and records a compatibility status. If TS1 lands
its shared schema before merge, MAIN must adapt these rows mechanically during
landing review.

## Derivation from the evaluator

At recursion level 0, the evaluator composes three unlike terms:

\[
S=100D_{\rm seg}+\sqrt{10D_{\rm pose}}
  +\frac{25}{37{,}545{,}489}B.
\]

This produces L5 verdict rows and makes the identity telemetry guard a
`GAUGE`: hooks change the trace while exact logits, argmax, and therefore the
verdict remain unchanged. This is a telemetry no-op only, not a claim about
receiver `ker(R)` bytes.

At recursion level 1, \(D_{\rm seg}\) recurses into discrete argmax-cell
membership and continuous within-cell distance. Ordered cell adjacency and
grammar tokens are `SKELETON`; the exact rank-4 head distance

\[
d_{c\to c'}(p)=
\frac{z_c(p)-z_{c'}(p)}
     {\lVert w_c-w_{c'}\rVert_2}
\]

is a margin-Fisher `FIBER`; target mismatches and receiver corrections are
`RESIDUAL`. No Euclidean metric is admitted. Pose rows use the exact
at-most-six-dimensional output quadratic.

At recursion level 2, pairs compose into one clip trajectory. Cross-frame,
cross-pair, and \(\xi\)-advected feature comparisons are `CONNECTION`, while
continuous PoseNet pair state remains a pair-native `FIBER`.

## L1-L5 homes

| Home | Meaning | SN1 examples |
|---|---|---|
| L1 | program and grammar | SDWL1 boundary token; solve-menu program move |
| L2 | receiver, uint8, and \(R\) | bounded inverse residual |
| L3 | scorer feature state | SegNet fibers/connections; PoseNet pair fiber |
| L4 | scorer decision | sided margin fiber; v19c decision residual |
| L5 | evaluator verdict | no-op gauge; exact three-way residual budget |

The machine-readable addendum is
`.omx/research/ddm_sn1_five_type_derivation_addendum_20260724.json`.
Every row contains a content SHA, selector, type, home, recursion level,
metric, first rung, and verdict scope.

## Exact limits

This is a typing and freshness closure over existing evidence. It does not
alter the prior measured counts, create receiver-closed bytes, run official
contest CPU/CUDA evaluation, or authorize the next receiver probe.
