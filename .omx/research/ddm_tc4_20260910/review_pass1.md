# TC4 independent review pass 1

Disposition: FINDINGS; this is not a clean pass. Source inspection only; no
experiments, scorers, or source edits were performed by this reviewer.

Reviewed `experiments/ddm_tc4_maps.py` SHA256
`178f74b15b59d43921f1587743872d2dcf125f1dd494b70de31ff9f4df915411`
and `experiments/ddm_tc4_price.py` SHA256
`ee53c5cc42ae1ad1e49a79eae8075d37c4c51c606256099e543b6b987a4adb12`.
Owner of fixes and consumer: parent ddm_tc4 implementation/pricing chain.

## Findings

1. HIGH — inherited stage and dependency bindings are incomplete. In price.py
   `pin` (98–136), the source list omits `ddm_tc3_geometry.py`, which the nested
   receiver actually imports. `prepare` checks only the control's success flag;
   `fit` checks only preparation's frame count; `encode` checks only fit
   convergence. None verifies the inherited stage binding against the current
   producer/dependency/input binding. The TC3 source NPZ hashes are checked
   against sidecars, but their trace/preparation manifests and trace producer
   binding are not pinned. A map/producer edit between completed stages can
   therefore combine stale features with new runtime code without an early
   refusal. Add explicit inherited binding checks and source-manifest pins;
   verify source frame field and producer linkage. Encode must also verify the
   selected columns equal the requested mask and the retained weights.bin is
   exactly the rounded matrix being shipped.

2. MEDIUM — resume can silently move backwards in the requested reporting
   boundary. `control` (185–227) and `prepare` (230–266) accept a restored frame
   greater than `stop`. Their empty loops then write a result labelled with
   the earlier `stop` while retaining state/stream from the later boundary.
   Require `0 <= start <= stop <= 600`; validate the LATEST frame against the
   loaded checkpoint frame as well. A completed stage can be reused at its own
   boundary, never relabelled as an earlier stage.

3. MEDIUM — native build custody is checked locally but is absent from encoder
   checkpoint binding. `native` validates BUILD.json against current files;
   `restore` compares a different binding that contains neither BUILD.json nor
   library identity. A rebuilt library plus rewritten build receipt can resume
   old arithmetic state. Put the validated native build identity into control
   and encode resume bindings before restore/checkpoint, and likewise bind the
   objective library to fit STATE.json. Preserve any changed-build run under a
   distinct explicit stage instead of accepting its older checkpoint.

4. MEDIUM — a zero feature contribution is not a literal no-op. Both
   `ContextMixer.coding` (139–147) and `encode` (438–440) always pass through
   frequencies and normalization for a nonzero coefficient bank, even when
   all selected features on that row are zero. That includes the initial
   frame's empty KT tables. The old TC3 path explicitly restores original rows
   for zero features. Preserve original float32 rows for the TC4 zero-feature
   case in one shared helper used by both pricing and the receiver. This keeps
   map attribution separate from an incidental probability requantization and
   supplies a meaningful zero-feature control.

## Causality and accounting inspected

The row-run map resets every 64 columns; within a row block each earlier x has
a lower group index. The complete-field vertical map excludes transition
endpoints at remainder zero and requires the transition's lower endpoint
remainder to be strictly below the query remainder. This matches the online
sentinel plane for same-column history. The temporal map uses only the complete
previous decoded pair. The saved previous-row lane bin was computed at that
above cell's own earlier event, and remainder-zero queries are unavailable.
The Movable map uses the same causal row-run state. I did not find future-token
access in these five formulas by source inspection; real-frame complete versus
prefix equivalence remains a necessary implementation check owned by parent.

The new rider accounts for magic4, selection-mask1, old weights40, every
selected five-byte weight column, and actual RC64 bytes. Tail accounting adds
the existing fixed 96-byte prefix: 119,675 B is the move41 comparator, not the
119,579 B rider alone. The archive builder preserves the complete old prefix.
The source control reconstructs old TC3 A rows and checks full raw stream
identity before preparation. Full n600 encode computes continuous and integer
codelength and retains both raw streams, envelopes, riders, and archives.
The fitting gap is correctly scoped to KEEP and is not a full-population
optimality certificate. Actual composite pricing and receiver seal remain
subsequent charter gates, not achievements of these files alone.

## Assumption challenge

The work freezes old40 weights and adds frame-frozen KT maps using fixed
generic bins. Those restrictions define this formulation; failure cannot close
all causal context models or prove a universal geometry bound. Different bins
or joint refitting could change the gain, but they are untested leads, not a
reason to replace the charter's immediate five-map real-encode slate. Prior
temporal MODEL saturation does not logically close the temporal mixer INPUT.

## Boundary

No real TC4 gain, execution timing, storage total, public identity, or score was
measured by this review. Parent owns Ruff, real-frame controls, two final clean
source reviews, serializer landing, and all retained experimental artifacts.
