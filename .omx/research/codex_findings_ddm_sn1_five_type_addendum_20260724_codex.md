# Codex findings — DDM SN1 five-type amendment closure

Date: 2026-07-24
Lane: `ddm_sn1_segnet_telemetry_asymmetry`
Axis: `[macOS-CPU frozen-SegNet+PoseNet advisory]`
Verdict: `DERIVED`, `research_only=true`, `score_claim=false`, pointer unmoved
Landing: isolated Codex worktree; MAIN landing review required

## Disposition

The prior SN1 implementation and n600 receipts already close the delegated
telemetry, official-video analysis, sided asymmetry, three-segment inverse
demonstration, and error-source tensor. Fresh review found one binding gap:
those historical streams predate the five-type amendment and carry no
representation type, L1-L5 home, or `evaluate.py` recursion citation.

The new SHA-bound addendum closes that gap without rerunning or rewriting the
measurement. Sixteen typed rows cover all five representation types and all five
layer homes. Seg rows use margin-Fisher/rank-4 geometry, Pose rows use the
exact at-most-six-dimensional output quadratic, and no Euclidean row is
verdict-bearing.

## Source custody and result

The builder consumes the existing SN1 telemetry/asymmetry receipt and the
error-source tensor receipt by live SHA, then revalidates every referenced
artifact at consumption. Any receipt or artifact drift is a hard refusal.
The checked-in addendum is deterministically rebuildable from those bytes.
Accepted addendum:
`.omx/research/ddm_sn1_five_type_derivation_addendum_20260724.json`,
SHA-256
`949935045f113da7b814149bd24b8112c86f3a3e8b8caac558b3909eb1da8b17`.

`TypedStreamTag` was not present in
`tac.optimization.ddm_min_description_contract` at fire time. No parallel enum
was created: the adapter reuses
`tac.optimization.ddm_dimension_conditioned_two_type.REPRESENTATION_TYPES`.
MAIN must check whether the independently owned TS1 schema lands before this
branch and mechanically adapt the compatibility rows if so.

## First rung

The typing result does not authorize execution. The exact next measurement
remains a separately claimed receiver-closed candidate for the highest-mass
solve-menu cluster, with same-candidate frozen SegNet, exact Pose output
quadratic, parse-back, and exact counted-byte custody.

No upstream file, pointer, archive, scorer result, or historical receipt was
changed. No training, remote/GPU dispatch, or official evaluation was run.
