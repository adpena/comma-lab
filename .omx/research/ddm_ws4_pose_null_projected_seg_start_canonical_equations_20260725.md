# DDM WS4 pose-null projected Seg-start equations

Status: research-only; `MAIN_REVIEW_REQUIRED`  
Authority: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`

## Lawful component join

For a W_seg correction `c` and a DM4 row `r`, projection is admissible only
when an explicit custody join exists:

```text
J(c,r) =
  [c.source_decision_path = r.source_decision_path]
  and [c.source_decision_sha256 = r.source_decision_sha256]
  and [c.correction_id = r.wseg_correction_id].
```

Pair, bucket, stratum, or stream-type coincidence is not a foreign key.
The projected set is

```text
C_plus = { c : exists r, J(c,r) and Delta_pose(c) > 0 }.
```

For the sealed inputs, `C_plus = empty`: the DM2/DM4 rows have no W_seg join
keys, the base correction rows lack per-move PoseNet coupling, and the only
separable suffix has `Delta_pose < 0`. Hence

```text
Pi_pose_null(W_seg) = W_seg
```

for this instance. This equality is a custody conclusion, not a claim that a
nontrivial pose-null projector was executed.

## Realized objective and fail-fast gate

The realized evaluator action is

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489.
```

WS4 preserves the WS3 component gate:

```text
accept(q) iff Delta S(q) < 0 and Delta d_seg(q) <= 0.
```

The exact first W_seg proposal satisfies `Delta S<0` but
`100*Delta d_seg=+0.002464294433594194`; therefore it is rejected as
`SEG_REGRESSION` and the accepted-step window stops at the start state.

The registered slope threshold remains
`R*=4.1215446777965665`. The measured proposal ratio is
`1.1735894458608507 < R*`, independently selecting W_joint.

## Domain and falsifier

This derivation is valid only for the SHA-bound WS2 W_seg archive and the
specific DM2/DM4 25-row G3 instance. It must be reopened if and only if a
producer lands:

1. per-correction W_seg PoseNet coupling;
2. an explicit correction-to-DM4 foreign key;
3. receiver/uint8 parse-back custody for the projected result.

STORES CONSULTED: FEED-603-ws4 and every receipt named there.
