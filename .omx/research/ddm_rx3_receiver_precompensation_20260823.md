# DDM RX3 receiver precompensation — the repair discriminator is real, four counted-parameter receivers are ready, and the n600 verdict is queued behind AP1

`status`: **PARTIAL — SCORER-FREE WORK COMPLETE; FULL n600 DISTORTION TABLE NOT MEASURED**  
`axis`: discriminator `[scorer-free read of retained MST1 macOS-CPU fields joined to contest-CUDA DALI-GT membership]`; candidate bytes `[exact byte-closed]`; future score rows `[macOS-CPU advisory; pinned contest-CUDA DALI-GT tables; n600]`  
`score_claim=false` · `promotable=false` · `shipping_candidate_built=false`  
`verdict_scope`: **INSTANCE** for the pinned DX2 repair discriminator and four executable candidates; no transform-family verdict is licensed before the queued n600 Seg/Pose rows.

## Outcome

The retained fields reproduce the required membership exactly: **28,602 gross native-render breaks =
11,685 later repaired + 16,917 terminal-persistent** over **117,964,800** Seg pixels and all **600**
pairs. The most important distinction is not residual magnitude. It is how close the native render already
is to the correct scorer cell:

- native true-class margin separates repaired from persistent positions at **AUC 0.826963** over
  **28,602** positions;
- preuint8 and uint8 margins rise to **0.995029** and **0.9999998**, respectively, but those are downstream
  outcome reads, not legal receiver inputs;
- total R+uint8 residual magnitude is nearly non-separating at **AUC 0.517405**;
- distance to the DALI-GT class boundary is **AUC 0.500594** because virtually every gross transition is
  already on the boundary;
- the strongest measured receiver-observable coordinate is native luma gradient at **AUC 0.590403**.

That evidence supports testing a generic image-local treatment, but it does not predict a distortion win.
Four real receiver variants now exist. Their fitted values live in the counted ZIP member, not in source;
their token stream remains byte-identical to DX2. Each candidate parses through the copied real receiver and
has a deterministic repeat archive. The exact archive deltas are **+12, +12, +12, and +13 B**.

I did **not** run the scorer. At closure, `.omx/state/main_hot_state.md` still grants the fleet's sole
full-n600 scorer lane to AP1, and AP1's latest queue receipt is nonterminal (its control is complete and
`semantic_l1` is pending). RX3 therefore did the permitted scorer-free work and emitted a complete canonical
fire order. Launching RX3 concurrently would violate the common single-flight contract.

## Pins and retained custody

| object | bytes | SHA-256 | result |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | MATCH |
| RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | independently parsed from the shipped archive; MATCH |
| WJ1 target list | 798,964 | `bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d` | independently hashed; MATCH |
| DALI Seg argmax | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | MATCH |
| DALI Pose6 | 14,528 | `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff` | MATCH |
| 28,602-row discriminator payload | 3,032,772 | `2e3f05a3dadf547616dc88d3960c8868977d0e45bb6c5a8d017aaa5faab555db` | retained locally |

The selected tier is
`.omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation/`. At preflight the local tier had
**501,399,457,792 B** free; Vertigo had **8,986,886,144 B** and APDataStore **11,861,753,856 B**.
Per the charter's explicit opt-in, RX3 wrote nothing under `/Volumes/*`. No materialized payload is marked
rebuildable or eligible for cleanup.

## The measured repaired-versus-persistent discriminator

Positive class is “later repaired.” Every continuous row has denominator **28,602 = 11,685 repaired +
16,917 persistent**. `AUC separation` is `max(AUC, 1-AUC)`, so 0.5 means no rank separation. KS is reported
as a distributional effect, not a distortion predictor.

| coordinate | repaired mean | persistent mean | AUC separation | KS | interpretation |
|---|---:|---:|---:|---:|---|
| native true-class margin | -0.056116 | -0.232030 | **0.826963** | 0.521520 | decisive state-closeness discriminator; scorer-only, not a receiver input |
| preuint8 margin | +0.063110 | -0.215741 | 0.995029 | 0.926706 | confirms the preuint8 leg performs the repair; downstream outcome |
| uint8 margin | +0.069362 | -0.214430 | 0.9999998 | 0.999882 | membership outcome, not a proposal feature |
| total residual L2 | 32.0424 | 31.4941 | **0.517405** | 0.039809 | magnitude alone does not distinguish repair |
| uint8-only residual L2 | 0.31160 | 0.30832 | **0.507318** | 0.014134 | uint8 repair is not selected by perturbation size |
| DALI-GT boundary distance | 0.01103 px | 0.00904 px | **0.500594** | 0.001189 | both groups are already boundary events |
| native luma gradient | 116.550 | 101.116 | **0.590403** | 0.166353 | strongest legal image-local proxy |
| same-class fraction, 3x3 | 0.55609 | 0.51962 | 0.578885 | 0.170773 | repaired positions have slightly more local class support |
| same-class fraction, 5x5 | 0.51550 | 0.47298 | 0.578550 | 0.194557 | same signal at a wider neighborhood |
| total residual G, signed | -9.18275 | -5.72496 | 0.553296 | 0.095546 | weak directional bias; repaired is more negative |

The GT→native wrong-class pair is materially more informative than residual sign: categorical total
variation is **0.214855** across **18** class pairs, versus **0.109886** across the **8** RGB residual-sign
patterns. Examples with useful denominators are `2→0`: **3,382 / 5,310 repaired (63.69%)**; `0→2`:
**167 / 1,414 (11.81%)**; `4→0`: **629 / 1,130 (55.66%)**. A generic receiver cannot read the DALI class
pair, so this supports the existence of scorer-cell structure but cannot itself key a legal transform.

The measured receiver-observable gradient threshold with the largest Youden separation is **113.704**:
it selects **6,639 / 11,685 repaired (56.82%)** and **6,804 / 16,917 persistent (40.22%)**, `J=0.165965`.
That is the basis of the gradient-band rung.

### Class membership of the 28,602 gross transitions

This is the discriminator population, not a candidate d_seg delta.

| DALI-GT class | gross denominator | repaired | persistent |
|---|---:|---:|---:|
| Road | 9,665 | 3,469 | 6,196 |
| **Lane** | **6,367** | **2,423** | **3,944** |
| Undrivable | 8,088 | 4,302 | 3,786 |
| Movable | 3,160 | 824 | 2,336 |
| MyCar | 1,322 | 667 | 655 |

The repaired membership is not Lane-scoped: **9,262 / 11,685 repaired positions (79.26%)** are outside Lane.

## The real receiver insertion point

The copied shipped code establishes the exact order:

`native 384x512 float RGB → fixed bilinear/bicubic lift to camera 874x1164 float RGB → RX3 function → clamp/round/uint8 → evaluator bilinear resize → frozen scorers`.

This corrects the charter's ambiguous “between render and R” wording. RX3 matches the actual L28 mechanism
surface: camera-resolution float RGB immediately before quantization. It does not edit the semantic render,
token field, HPAC model, coder, or shipped receiver. The generic runtime reads a small trailer from member
`p`, strips it before the original F26 parser, and applies one of three public algorithms. Parameter values
are absent from receiver source.

Runtime verification parsed all four child archives through the copied `runtime.residual_archive`, recovered
the exact **113,777 B** RC64 stream and its pinned hash for each, parsed each parameter trailer, and passed
the top-level archive/extraction verifier. Canonical `tools/fire_local_advisory.py --dry-run` also passed for
all four with `passthrough_env={}`: `F26_TOKEN_DECODER` remains its permitted Python default.

A pre-fire review caught one operation-order defect in generated runtime generation 1: it inherited the
camera `.round()` before calling RX3. No scorer or transformed raw had run. All four generation-1 runtimes
were moved intact to each candidate's `retained/superseded_runtime_postround_v1/`; nothing was deleted. The
live generation was rebuilt and reverified with exactly one final round after `apply_precomp`. The retained
`RUNTIME_REBUILD.json` names both old and new tree hashes, and `RUNTIME_VERIFICATION.json` v3 certifies the
camera-float-before-round order independently for both frame roles on every live tree. The review tracker
records two final-tree passes: operation-order/custody, then resume/scorer/no-fake. The second pass added
chunk-statistic recomputation, contiguous n600 closure, DALI geometry checks, and a fixed class-census guard
to the queued scorer consumer before it was marked reviewed.

## Candidate table — byte legs measured, distortion legs queued

`parameter count` counts fitted scalar values. `parameter bytes` is their actual stored section. The archive
also carries a 6-byte generic trailer (`magic`, algorithm id, section length); the **real archive delta** is
the measured size of the complete child ZIP minus 180,368 B. The rate price uses the TX1 §0 exact exchange
rate `25 / 37,545,489 = 6.658590e-7 S/B`.

| candidate | fitted parameters | stored values | parameter bytes | real archive delta | ΔS_rate | Δd_seg per class, Lane row | Δd_pose | ΔS_distortion | net ΔS |
|---|---:|---|---:|---:|---:|---|---|---|---|
| `l28_exact_counted` | 6 | frame0 `(-1,0,-1)`; frame1 `(0,-1,0)` | 6 | **+12 B** | **+0.0000079903** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** |
| `global_repair_mean` | 6 | frame0 `(0,0,0)`; frame1 `(-2,-2,-2)` | 6 | **+12 B** | **+0.0000079903** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** |
| `local_highpass_regression` | 6 | frame0 gains `(0,0,0)`; frame1 gains `(-17,-18,-18)/64` | 6 | **+12 B** | **+0.0000079903** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** |
| `gradient_band_repair_mean` | 7 | frame1 `(-2,-2,-2)` where image gradient ≥114; frame0 unchanged | 7 | **+13 B** | **+0.0000086562** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** | **NOT MEASURED** |

Archive hashes, respectively: `e0c3ea7e…e8978aa`, `3d3d053c…729a67b`, `4e433770…073ac7`, and
`8b7ed79f…109dd9`. Each archive has a byte-identical deterministic repeat. These are research candidates,
not shipping candidates.

### Scorer-free fire order

The ordering proxy is the mean reduction in squared distance to the already-observed total R+uint8 residual.
It is measured over the same **11,685 / 16,917** membership split and is explicitly **not** interpolated or
converted into d_seg/d_pose.

1. `local_highpass_regression`: repaired proxy gain 1,077.55, repair selectivity 5.56.
2. `gradient_band_repair_mean`: repaired proxy gain 72.44, selectivity 32.47.
3. `global_repair_mean`: repaired proxy gain 67.79, selectivity 30.17.
4. `l28_exact_counted`: repaired proxy gain 17.37, selectivity 6.92.

The exact canonical advisory and pinned-DALI post-score argv for every ordinal are retained in
`.omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation/FIRE_ORDER.json`. Each row uses batch 16,
preserves the 3,662,409,600-byte transformed raw, persists every candidate argmax/Pose6 chunk and concatenated
n600 field, and only advances after the preceding advisory plus DALI post-score are terminal.

## What is concluded, and what is not

- **Concluded at INSTANCE scope:** the 28,602/11,685 split is real; repair is margin-selected rather than
  perturbation-magnitude-selected; a legal image-gradient proxy has weak but nonzero separation; four real
  counted-parameter receiver variants exist and preserve the pinned token stream.
- **Not concluded:** no candidate has a d_seg, per-class/Lane delta, d_pose, distortion delta, or net score.
  The charter's falsifiable prediction is neither confirmed nor falsified. The L28 mechanism transfer and
  the whole receiver-transform family remain open.
- **Not touched:** WJ1's join/target list, MST1's stage split, BL1's cost field, AR1B's census, the sacred JO1
  r9 directory, `upstream/`, the shipped DX2 runtime, and the JF1/MP3/AP1 rate-axis trees.
- **Concurrency boundary:** AP1 owns the sole scorer lane. JF1 and MP3 own their named rate axes. RX3 has
  appended no lane claim and fired no advisory, scorer, Modal, or Metal job.

## RECALL EVIDENCE

The charter seeds were a floor. I searched:

- `.omx/research/`, local arm receipts, the canonical research index, the full `sub015_DAG_*` graph, design
  specs, and the canonical task ledger by content with `ddm_rx3|receiver precompensation|decode-side channel
  postprocess|L28|R uint8 repair|current-terminal receiver-treatment`;
- the canonical equation registry with
  `.venv/bin/python tools/list_canonical_equations.py --json`, filtering for score, receiver, roundtrip,
  uint8, and boundary surfaces;
- current lane authority in `.omx/state/main_hot_state.md`, `.omx/state/active_lane_dispatch_claims.md`, and
  AP1's live `QUEUE_STATE.json`;
- directive filenames modified in the preceding 24 hours; the bounded search found none in `.omx`.

Beyond the seeds, recall found:

1. canonical task `ddm_l28_current_terminal_receiver_ab_20260813` is still pending, owned by the
   “current-terminal receiver-treatment successor.” RX3 is the concrete successor, but its old Vertigo
   consumer path is replaced by the charter-mandated local store because both SSD tiers are nearly full;
2. the DAG's “L28 not adaptable” sentence is scoped to the old **non-RGB witness**. DX2 emits RGB, so it does
   not prohibit this current-vehicle mechanism test; it does prohibit transferring the ancestor number;
3. the canonical receiver-admission law from PZ4R says receiver repeat identity does not imply semantic Pose
   preservation. That reinforced the mandatory DALI Pose6 post-score on every rung;
4. the canonical score-marginal law agrees with TX1 §0's exact rate coefficient; no second exchange-rate
   derivation was used;
5. AP1 had already begun a matched DX2 control on the same DALI tables. RX3 will consume that read-only
   control after AP1 completes rather than duplicate a baseline scorer row.

No prior current-DX2 n600 L28/precompensation result was found in those bounded scopes. This is a scoped
absence statement, not a global nonexistence claim.

## Follow-on disposition

**QUEUED-WITH-A-FIRE-ORDER.** Owner: `ddm_rx3_receiver_precompensation`. Consumer store:
`.omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation/advisory_and_dali_scorer/`. Fire trigger:
`main_hot_state` no longer grants the sole full-n600 scorer lane to AP1, AP1's latest queue state is terminal,
and RX3 has appended a non-conflicting local-scorer claim. Then fire one ordinal at a time from
`FIRE_ORDER.json`; aggregate only after all four DALI post-scores are complete.

**Own-vehicle frontier: UNMOVED at DX2 S=0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`. RX3 moved it by 0.**
