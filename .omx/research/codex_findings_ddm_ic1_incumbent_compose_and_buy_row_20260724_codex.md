---
schema: codex_findings_ddm_ic1_incumbent_compose_and_buy_row.v1
date_utc: 2026-07-24
lane_id: lane_ddm_ic1_incumbent_compose_and_buy_row_20260724
delegation_checkpoint_key: codex_delegate:ddm_ic1_incumbent_compose_and_buy_row:20260724T143642Z
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
execution_allowed: false
score_claim: false
pointer_before: "0.1910828242 [contest-CPU]"
pointer_after: "0.1910828242 [contest-CPU]"
main_review_required: true
---

# DDM IC1 incumbent compose-and-buy row

## Outcome first

**Verdict: `INCUMBENT_V0_RECEIVER_CLOSED_ADVISORY_MEASURED__CONTEST_CPU_BUY_PREPARED_NOT_DISPATCHED`.**

The best admitted composition available in this lane is the exact W_joint E5
parent followed by the zero-payload PA1 scorer-only frame-0 transform, packaged
through an explicit strict IC1 route and Brotli-Q11. The archive is `131,582 B`,
SHA-256 `aba831de…9d9`.

Fresh full-n600 batch32 authority on the decoded bytes measured:

- `[MEASURED] d_seg=0.07051923116048177`
  (`8,318,787 / 117,964,800` errors);
- `[MEASURED] d_pose=27.298487616378203`;
- `[DERIVED] S_advisory=23.66179213623354`
  = `7.051923116048177` Seg
  + `16.522253967415644` Pose
  + `0.08761505276972155` rate.

The separate locked upstream `evaluate.sh` harness passed in `1554.008926 s`
and printed rounded `d_seg=0.07051922`, `d_pose=27.29848671`, `score=23.66`.
Both rows are macOS CPU advisory. No contest score exists and the pointer stays
**`0.1910828242 [contest-CPU] UNMOVED`**.

## What composition actually bought

Against the exact E5 W_joint packet at `131,294 B`, PA1:

- preserves d_seg and every per-class error count exactly;
- preserves all frame-1 camera bytes exactly
  (`0b574f6f…9a4b` on both parent and child; zero changed channel values);
- reduces d_pose by `9.319697135033131`;
- adds `288` packed runtime/manifest bytes despite zero serialized PA1 payload;
- improves advisory objective by `2.6134328056364815`.

The per-class Seg endpoint is:

| Class | Errors / sites | d_seg |
|---|---:|---:|
| Lane | 369,103 / 690,639 | 0.5344369489704462 |
| Movable | 960,428 / 1,460,325 | 0.6576809956687724 |
| MyCar | 4,072,489 / 29,993,509 | 0.13577901138543008 |
| Road | 2,689,055 / 27,407,046 | 0.09811546271714215 |
| Undrivable | 227,712 / 58,413,281 | 0.0038982915546209433 |

This is a real composition result, not a sum of W_joint and PA1 historical
deltas. The companion frame-position control measured `1,830,376,393` changed
frame-0 channel values and zero changed frame-1 values.

## Forest decision and conflicts

V19C is included transitively in W_joint and supersedes V19B. E4/E5 and #636
supply the packer, typed grammar admission, and runtime compiler surfaces.
E3 is a receiver-composition precedent, not pooled payload.

The other named pieces were excluded with exact scope:

- MC1's best measured static-stored arm is `+4.850055382139988` joint score
  units worse on its parent.
- E2 is a separate `343,466 B` receiver-closed endpoint; it has no admitted
  additive or typed W_joint compose route.
- DM2's freshly remeasured aggregate is `+2.350835831188035`; one favorable
  independent row cannot be promoted into an additive pool.
- W_seg has lower Seg debt but much higher Pose debt than W_joint, so its exact
  packed endpoint has worse joint objective.

No excluded family is closed; these are finite measured conflicts or missing
composition routes.

## Recursive-scorer provenance correction

The exact W_joint point stands only as a **`[naive-menu upper bound]`**. Its
generic local-statistics/hard-placement amplitude menu does not establish a
paint ceiling, winner ordering over omitted scorer-recursive candidates, or a
transferable stream price.

PA1 is scorer-derived: it targets frozen PoseNet first-stem/BN moments and uses
the exact evaluator factorization to place the transform only on Seg-free frame
0. The scorer-derived replacement for the W_joint construction is a corrected
inner-Jacobian bank with exact resize-footprint and stride-2 stem-lattice write
support, ERF-aware support, Fisher-margin ranking, shearlet residuals, and an
explicit Pose-null or Pose-priced field.

## A2 stage-curve disposition

The companion stage-curve receipt lands actual parse-back stream bytes and the
same-parent W_joint→PA1 sequential exact-R measurement. Its local two-point
secant is measured. It does **not** finish the A2 five-item pull list:
scorer-recursive W_joint paint/support/exception construction, pairwise controls
over those omitted streams, and their joint KKT envelope remain absent.

Therefore:

- A2-06: partially measured, still suspended as a price claim;
- A2-15/A2-17: still suspended;
- every per-stream/KKT price: `NULL`;
- the c1 split: reservation identity, not “waterfilled.”

## Receiver and custody

- source W_joint: `138,801 B`,
  `5aa45850…433e`;
- contiguous streams: V19C nested archive `137,827 B`,
  `dc767b59…e4c9`, then warm payload `974 B`, `47056b15…4f4ee`;
- compiled archive: `131,582 B`, `aba831de…9d9`;
- compiler determinism: PASS ×2;
- source parse-back identity and receiver byte-home bijection: PASS;
- raw output: `3,662,409,600 B`, `e69be18f…4938`;
- all 38 base and 38 composed checkpoints: preserved on the SSD tier;
- no source or proof bytes deleted.

The amplitude-transform proof carries composition binding
`c260e7cd…d27a` in a legacy internal helper field named `manifest_sha256`.
It is a stable transform binding, not the outer ZIP manifest SHA; the typed
stage-curve receipt makes that distinction explicit.

## Contest-CPU buy row

`ddm_ic1_incumbent_modal_contest_cpu_bundle_20260724.json` seals the exact
archive, inflate runtime, projected Modal CPU runtime tree
`0e7a7d1a…e9e7d`, atomic claim plan, one-command wrapper invocation, and harvest
requirements. Status is `PREPARED_NOT_DISPATCHED`: no live lane claim, Modal
call, paid action, or exact-axis score was created here.
The tracked custody object is `packet/archive.zip.receipt-bytes`; it is
byte-identical to the locally generated `archive.zip` and retains SHA-256
`aba831de…9d9`.

## Triality

- **DSL:** `DDMIC1RuntimeExporterConfigV1` and
  `ddm_ic1_runtime_archive.v1` are explicit closed schemas; legacy E1/E5
  literals reject IC1-only fields.
- **DAG:** the companion feed records included, conflicting, suspended-price,
  measurement, and not-dispatched buy edges.
- **Equations:** the companion note records exact objective decomposition,
  compose-then-measure, local-secant/KKT separation, and scorer-recursive typing.

## MAIN landing review

`main_review_required=true`. MAIN must:

1. re-hash archive, runtime, three primary receipts, and all piece-manifest
   sources;
2. verify the IC1 route does not widen legacy E1/E5 schemas and that packaged
   receiver replay matches local source behavior;
3. verify W_joint remains `[naive-menu upper bound]`, PA1 alone is
   scorer-derived, and every stream/KKT price remains `NULL`;
4. independently recompute both stage objectives and the `288 B` secant;
5. rerun the clean-pass receipt after any merge conflict or substantive edit;
6. only then use the stored Modal command, allowing its wrapper to atomically
   claim the lane and record Modal call-id custody.

## HISTORICAL_PROVENANCE

Append-only delegated landing from isolated worktree
`ddm_ic1_incumbent_compose_and_buy_row_20260724T143642Z`, based on commit
`23a801c4c6d4533fa7f3098a4977ed1926eccde4`. It consumed the later
2026-07-24 recursive-scorer directive and MAIN-reviewed A2 audit. It supersedes
no historical result and moves no frontier pointer.
