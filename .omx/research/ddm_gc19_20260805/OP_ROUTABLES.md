---
id: ddm_gc19_20260805_op_routables
kind: op-routables
date: 2026-08-05
axis: "[macOS-CPU advisory] research/text only"
---

# GC19 Op-Routables

## What The JD1 Endpoint Should Decide

JD1 should decide whether the first live joint seg+pose descent on the real vehicle is a usable controller physics event. The endpoint should be consumed as a predicate table:

| Predicate | Minimum evidence | Disposition | Fire order |
|---|---|---|---|
| Joint descent alive | d_pose improves materially while d_seg stays at or below the calibrated hold neighborhood from the ep1334 start | QUEUED-WITH-FIRE-ORDER | byte-close the endpoint; then run the smallest terminal pose/rate compose gate |
| Pose alive, seg harmed | d_pose improves but d_seg regresses enough to erase score | QUEUED-WITH-FIRE-ORDER | run seg-recovery terminality from the endpoint; if it fails, retune seg-hold controller |
| Seg alive, pose flat | d_seg holds but d_pose is not meaningfully lower | QUEUED-WITH-FIRE-ORDER | fire JB1 derived-Jacobian pose basis race against the same boundary |
| Both axes harmed | d_seg and d_pose both worse than start | FOLDED | do not promote, byte-close, or compose; classify JD1-v2 as controller/formulation failure |

The endpoint decision must record:

| Required endpoint field | Why |
|---|---|
| start checkpoint sha/path and epoch | proves the ep1334 never-weaker-state chain |
| calibrated seg-hold floor source and value | prevents the TP1 loss-scale bug from recurring |
| d_seg trajectory | primary axis for score gap |
| d_pose trajectory and term contribution | verifies joint pose physics |
| any checkpoint chosen for byte-close | prevents cherry-picked chat-only endpoint claims |
| whether JB1 boundary race is triggered | connects endpoint to next owned action |

## Ranked Routables

| Rank | Routable | Projected value | Disposition | Fire order |
|---:|---|---:|---|---|
| 1 | Consume JD1 endpoint as the physics discriminator | Unknown until endpoint | QUEUED-WITH-FIRE-ORDER | When JD1 exits, apply the predicate table above before any other successor |
| 2 | OD3/SQ2-style terminality after a live endpoint | OD3 n32 projected delta 0.069298934880481 = 11.9103% of live gap = 104,074.5 rate-equivalent bytes | QUEUED-WITH-FIRE-ORDER | Only after endpoint state is chosen; require receiver/scorer survival before n600 |
| 3 | OD2/open-Stage-1 pose-aware compose | GC17 projected delta 0.062236702464336054 = 10.6965% of live gap = 93,468.3 rate-equivalent bytes | QUEUED-WITH-FIRE-ORDER | Close terminality, weak receiver packet, and pose-aware rate waterfill gates first |
| 4 | Task-description archive corridor | Target corridor 90-155 KB, still conjectural | QUEUED-WITH-FIRE-ORDER | Use OD5 generator/worldsheet coordinates and PE3 geometry only after n>=32 receiver survival |
| 5 | JB1 derived-Jacobian pose basis | No score finding yet; n=8 smoke only | QUEUED-WITH-FIRE-ORDER | Fire if JD1 pose is flat or antagonistic; compare against generic rank-6 basis at same boundary |
| 6 | Multi-objective gradient conflict controller | No Pact score yet; #956 high disagreement makes it plausible | QUEUED-WITH-FIRE-ORDER | If endpoint shows seg/pose conflict, add default-off PCGrad/CAGrad/MGDA-style controller with direct gradient-cosine logging |
| 7 | Pure rate crumbs on qo1 | 0.0001651 score = 0.0284% of gap = 247.95 B equivalent | FOLDED | Do not spend a unit here unless it unlocks a larger packet |
| 8 | Flat sparse solved-pixel ship-the-solve | OD9 projection rate-dead | FOLDED | Do not relaunch this formulation |
| 9 | OD4 sparse per-flip weak packet | GC18 killed formulation only | FOLDED | Keep family distinction; do not resurrect this packet shape |
| 10 | PE3 current hybrid packet as-is | PK1 scorer gate negative, S=1.852721897902562 @ 432,428 B | FOLDED | Reuse only as geometry prior for a different receiver-closed task-description packet |

## Backcast Guardrails

| Guardrail | Value | Consequence |
|---|---:|---|
| Current own-vehicle bytes | 357,836 B | Rate term alone is about 0.2382683043494253 |
| PR130 byte ceiling at terminal pose and d_seg 0.0003 | 190,541 B | Need about 167 KB byte collapse from current size |
| sub-0.15 byte ceiling at terminal pose and d_seg 0.0003 | 157,289 B | Need about 200 KB byte collapse from current size |
| PR130 byte ceiling at terminal pose and d_seg 0.0005 | 160,505 B | Distortion target loosening tightens rate pressure |
| sub-0.15 byte ceiling at terminal pose and d_seg 0.0005 | 127,253 B | Task-description archive must become real, not decorative |
| Zero-seg at current pose/rate still missing | 0.15066043219922431 score = 25.89% gap | Distortion-only endpoint cannot promote to target without rate/pose action |

## Follow-On Ledger

| Follow-on | Status | Consumer | Exact blocker |
|---|---|---|---|
| JD1 endpoint predicate application | QUEUED-WITH-FIRE-ORDER | next boundary/operator tick | JD1 live run must finish or yield endpoint telemetry |
| Byte-close selected JD1 checkpoint | QUEUED-WITH-FIRE-ORDER | byte-close/eval owner | Requires endpoint predicate "joint descent alive" |
| Terminality from JD1 endpoint | QUEUED-WITH-FIRE-ORDER | OD3/SQ2 successor owner | Requires chosen endpoint checkpoint and receiver gate |
| JB1 boundary race | QUEUED-WITH-FIRE-ORDER | JB1 owner | Requires pose-flat or pose-antagonistic JD1 endpoint |
| Multi-objective controller variant | QUEUED-WITH-FIRE-ORDER | trainer/tooling owner | Requires measured seg/pose gradient conflict at endpoint or short diagnostic |
| Flat sparse ship-the-solve | FOLDED | none | OD9 rate-dead scoped formulation |
| Current PE3 hybrid packet promotion | FOLDED | none | PK1 scorer-negative as-is |
| Pure byte crumb hunt below 8,738 B | FOLDED | none | Below 1% live-gap bar absent unlock |

## Proof Requirements Before Any Promotion

| Claim | Required proof |
|---|---|
| "JD1 helped" | endpoint d_seg and d_pose through the actual scorer path, start/end paths, and no weaker-start substitution |
| "Terminality composes" | receiver-closed packet plus n>=32 scorer survival on the selected endpoint object |
| "Task-description archive saves bytes" | actual `archive.zip` bytes for the packet shape, not component projection |
| "Controller removes antagonism" | before/after gradient conflict metric plus score-axis movement, not only smoother loss |
| "This can move pointer" | exact archive under custody through `upstream/evaluate.py`; advisory rows stay advisory |

## Decision Priority

1. If JD1 has a usable endpoint, byte-close it before inventing a new model.
2. If JD1 exposes conflict, run the smallest controller/basis race that answers the conflict.
3. If JD1 fails both axes, fold the formulation and advance the task-description archive route.
4. Do not route another broad PR95-shaped or RGB-reconstruction run as "capstone" without a predicate it uniquely answers.
