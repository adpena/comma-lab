Executed and landed the charter in commit `d5b2a7de7a`.

- Joint candidate: **180,083 B**, SHA `5b856e66…28c9`
- Exact marginal: **−132 B vs gb1**, clearing the 30 B bar by 102 B
- Same-distortion projection: **0.14803010583079396**
- Native full decode: 600 pairs, 1,028.6 s, raw SHA `7246a4ff…f2de7`—byte-identical to gb1
- Contest-CUDA and contest-CPU seals both validated and passed dispatcher dry-runs
- No scorer, lane claim, or Modal dispatch occurred; frontier remains **0.14811799921260607**

Artifacts: [verdict memo](/Users/adpena/Projects/pact/.omx/research/ddm_lb1_banked_lossless_joint_collect_20260829.md), [implementation](/Users/adpena/Projects/pact/experiments/ddm_lb1_banked_lossless_joint_collect.py), [dual-axis fire order](/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/measurement_v1/DUAL_AXIS_FIRE_ORDER.json).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER, ordinal 1 contest-CUDA** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/fire_cuda/`; fire trigger: pointer still names `ba1f3830…`, CUDA seal revalidates, no scorer/duplicate lane is active, and MAIN records the named CUDA lane claim.
- **QUEUED-WITH-A-FIRE-ORDER, ordinal 2 contest-CPU** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/fire_cpu/`; fire trigger: CUDA is harvested, single-flight clears, CPU seal revalidates, and MAIN records the paired CPU lane claim.

## LIVE-HYPOTHESES

- The T4 row should preserve gb1’s distortion legs and realize the full −132 B rate delta because both archives decode to identical tokens and share the renderer; CUDA remains untested until MAIN fires.
- Another decode-derived position chart may pay if it partitions structure orthogonal to patch192; patch192 retained 109 B beyond jt21 despite 43.62% measured overlap.

## DEAD-ENDS

- Additive ledger arithmetic is closed: the physical pool retained 132 of the projected 234.13 B.
- Coder and ZIP-framing reraces are closed at 0 B on this body.
- The paid patch table is closed because it misses break-even by 47.4×.
- `tile48*4+subtile4` is not the correct flat index; the verified receiver expression is `(y//32)*16+(x//32)`.
- The unmodified generation-20 C corrector is incompatible; the retained generation-22 port is config-matched and full-payload identical.