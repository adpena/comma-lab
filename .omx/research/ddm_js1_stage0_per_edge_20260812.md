# ddm_js1 Stage 0 — retained n600 per-edge diagnostic and promoted-axis blocker

## Outcome

The charter's first scorer-slot action is **not admitted**. Both bound archives were fully
materialized and scored at n600 with every payload retained, but the only locally available
receiver surface is `[macOS-CPU frozen-SegNet advisory, n600] NON-PROMOTABLE`. It does not
reproduce the CUDA-locked CP135 terminal row:

| object | retained local flips / 117,964,800 | local `d_seg` | required reference | disposition |
|---|---:|---:|---:|---|
| CP135 base, archive `6eb1a3b7…edb6` | 50,395 | 0.0004272037082248264 | 34,964 / 0.00029639352578669786 | **BLOCKED_AXIS_MISMATCH** |
| T1R1 C1-composed, archive `12a5b181…ce80` | 47,950 | 0.00040647718641493056 | no promoted argmax field retained | diagnostic only |
| retained C1 token target | 17,927 | 0.0001519690619574653 | 17,926 on the batch-16 reference | one-pixel reference-surface mismatch |

The matched-local diagnostic improves by 2,445 flips and has a diagnostic rho of
`0.07530491560921523`, but that number is **withheld from the charter gate**. The admitted
`rho_measured` is `null`; `rho_gate_passed` is false. Comparing the diagnostic rho to
`rho_required = 0.827795` would violate the charter's same-axis refusal rule.

The machine-readable verdict is
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/STAGE0_RESULT.json`,
31,009 B, SHA-256 `4247320a2e824e03ed94ac9fbe6a96a97d2b125dadb4fbdfe5e59fd813d7d76f`.

## Measured local edge diagnostic — not a Stage-0 map

All 600 non-overlapping pairs, all 117,964,800 scorer pixels, all 20 directed confusion cells,
and all 10 undirected interfaces were counted. Directed asymmetry is preserved in the receipt.
The largest local changes from CP135 base to the composed object were:

| edge | base flips | composed flips | change | local edge rho |
|---|---:|---:|---:|---:|
| Road↔MyCar | 7,086 | 5,940 | +1,146 | 0.4790969899665552 |
| Road↔Undrivable | 11,095 | 10,369 | +726 | 0.11551312649164677 |
| Road↔Lane | 19,410 | 18,995 | +415 | 0.029434711681679552 |
| Road↔Movable | 5,076 | 4,893 | +183 | 0.04688700999231361 |
| Undrivable↔Movable | 7,233 | 7,220 | +13 | 0.002423564504101417 |

Four small interfaces regress by 7–11 flips locally. These are INSTANCE-scope, local-axis
observations only. They do not aim the joint solver, kill an edge family, or establish CUDA
survival. Road-incident flips are 84.665% of the local CP135 field and 83.831% of the local
composed field.

## Custody and reproducibility

The storage preflight observed 260,314,488,832 free bytes and required 56,465,276,416 bytes,
including a 34,359,738,368-byte reserve. The completed retained tree occupies 19 GiB at:

`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/`

It contains, per candidate:

- exact `archive.zip` and extracted `p`;
- adopted exact shipped-receiver decoded token field plus its source binding;
- decoded semantic weights, pose basis, and pose coefficients;
- complete preselector and selected `0.raw` streams;
- complete float32 SegNet logits and uint8 argmax fields;
- render/selector checkpoints every 24 pairs and scorer checkpoints every 120 pairs;
- per-pair directed/undirected JSONL and content-hash receipts.

Key retained hashes:

| payload | bytes | SHA-256 |
|---|---:|---|
| CP135 selected raw | 3,662,409,600 | `a641d1ef149f8da8f06af3da9234d6d2f6be9702c3f606b7acf838b4b298ed47` |
| CP135 logits | 2,359,296,128 | `3e14fd9e9fd72f3d139af9cb6f52193462e88652cfcd52a7e0aeea15a77baebe` |
| CP135 argmax | 117,964,928 | `b8f063eb53891ca89d02c80adc7fca7d8c5f638f80cecf364a7f7a89bc68647a` |
| T1R1 selected raw | 3,662,409,600 | `ee4a013d5e8dcefc096b915a1b3371746c4a4919dd7b6c38ea5d92477346071f` |
| T1R1 logits | 2,359,296,128 | `8782db728842d31a9da40356660e5c9ecfad36163aad175991e2f73f33f5216e` |
| T1R1 argmax | 117,964,928 | `e5e8c1a464f29624d1eaed0ca3d739b98205cbf39f8bdaeaf008f599e4e3b64f` |

Reproduction commands:

```bash
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py preflight
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py inflate --candidate cp135_base
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py score --candidate cp135_base --chunk 120 --batch 1
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py inflate --candidate t1r1_c1_composed
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py score --candidate t1r1_c1_composed --chunk 120 --batch 1
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py summarize
```

The lane was claimed as `lane_ddm_js1_stage0_per_edge_20260812` before the scorer run. A promoted
CUDA dispatch was then checked. The local single-flight ledgers were clear, but
`.venv/bin/modal app list --json` failed with `Could not connect to the Modal server`; this
sandbox therefore cannot produce the missing promoted argmax custody. No paid job was launched.

## Scope and boundaries

- **MEASURED:** exact bound archive and token identities; receiver-closed Mac CPU raw fields;
  frozen-SegNet logits and argmax; full-population per-pair/per-edge counts; payload hashes.
- **NOT MEASURED:** promoted CUDA argmax fields; admitted Stage-0 rho; PoseNet; a complete score;
  public `upstream/evaluate.py`; V0–V5; any realization arm; any pointer-moving row.
- **INSTANCE verdict:** this Mac CPU renderer/scorer surface cannot adjudicate CP135/T1R1 Stage 0.
- **No family verdict:** the composed C1 carriage and distortion-side edge conditioning remain open
  until the promoted same-axis decomposition exists.

The effective composed pointer remains **S = 0.16195513827824176 @ 186,252 B
`[contest-CUDA T4, n600]`**. The own-vehicle frontier remains **S =
0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## RECALL EVIDENCE

The full corpus was searched across research memos/receipts, equations, memory, DAG, council,
tasks, and docs. Queries were:

- `current shipping base per edge decomposition m91 Road Lane hub`
- `cp135 C1 composed argmax seg decomposition t1r1`
- `global joint solve implicit edge conditioning decoder derived state capacity allocation`
- `seg per edge confusion matrix GT rendered denominator n600`
- `terminal composed candidate scorer retained argmax fields`
- `m94 instrument capacity object capacity claim units`
- `same parent hash enforcement m37`
- `Road Lane asymmetry differential depth dust n600`

Beyond the charter seeds, recall found five changes that bound this run: preserve directed
asymmetry rather than only undirected totals; include the C1 target plane so edge rho has a real
denominator; adopt the already-retained shipped RC64 token fields rather than re-decode them;
carry m94 claim-unit scope and m37 content-parent bindings; and treat sr1's rate-side edge
conditioning as closed while leaving only distortion-side conditioning alive. The primary
sources were the m91/pc2 per-edge work, fd135, sr1, hy1, hr1/hr2, rvs1/rvs2, ip1/lv2, m94, m37,
the canonical equation registry, the research index/DAG, and the task ledger.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN/js1 scorer-lane owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`. Fire trigger: a 1:1 T4 CUDA lane with working artifact return becomes available; materialize both exact bound archives through the shipped CUDA receiver, retain raw/logits/argmax payloads, recompute the full n600 edge map and rho on one batch-16 axis, then fire V0–V5 only if that receipt is admitted.**

## LIVE-HYPOTHESES

- Exact CUDA survival can differ materially from the local 7.53% diagnostic because changing only
  the renderer device moves the CP135 field by 15,431 flips relative to the promoted row.
- Road-incident edges may remain the dominant allocation hub on CUDA because both the earlier m91
  evidence and this independent local full-population diagnostic put most flips on Road edges.
- The largest recoverable non-Lane pocket may be Road↔MyCar; it owns nearly half of the available
  local edge-specific recovery, but this must be re-ranked on promoted CUDA custody.

## DEAD-ENDS

- Using the Mac CPU decomposition as Stage-0 rho is closed at INSTANCE scope: it fails the terminal
  CP135 positive control by 15,431 flips and misses the C1 batch-16 reference by one pixel.
- Repeated Modal dispatch from this unchanged sandbox is closed until connectivity changes: the
  cloud inventory call cannot reach the Modal server, and no paid call ID exists to harvest.
- Rate-side probability calibration for implicit edge conditioning remains closed at FORMULATION
  scope by sr1; only distortion-side decoder-derived state may continue.
- More CP135 lossless-coder hunting remains closed by LP135/lv2; it cannot substitute for the
  missing promoted scorer surface.
