# DDM SR3 — AP closed-lane retention certified, compressed, and reclaimed

Date: 2026-08-26  
Owner: `ddm_sr3`  
Disposition: **STOP CONDITION MET / W96B STORAGE TRIGGER GREEN**  
Authority: scorer-free storage custody; no score claim

## Result first

Two terminal retention trees were converted to deterministic zstd archives only after every
source file had a path/size/SHA-256 manifest, a complete local-scratch reconstruction had matched
every file SHA-256, a pre-reclaim certificate was durable, the live fleet was idle, and a second
full source hash had matched immediately before exact-target removal.

| quantity | measured result |
|---|---:|
| AP free, charter denominator | 12,942,966,784 B (12.054 GiB) |
| AP free, live final `df` | **48,109,191,168 B (44.805 GiB)** |
| net gain versus charter denominator | **35,166,224,384 B (32.751 GiB)** |
| sum of per-tree certified `df` deltas | **35,197,550,592 B (32.780 GiB)** |
| W96B fire trigger | 33,569,378,304 B — **GREEN by 14,539,812,864 B** |
| SR3 stop condition | 36 GiB = 38,654,705,664 B — **MET by 9,454,485,504 B** |

The 31,326,208 B difference between the charter-denominator net gain and the certificate-delta
sum is bounded to the interval before SA1's recorded `df_before`: the B2E first-attempt manifest,
failure receipt, AppleDouble metadata, and other filesystem accounting. The custody conclusion
uses the smaller live-final number.

The exact contest pointer did **not** move. No scorer, Modal job, training run, or evaluator ran;
`upstream/` was not modified.

## AP survey and adjudication

The direct-child census contained **223 directories** under `/Volumes/APDataStore/pact`.
Classification was exhaustive at this boundary:

- 7 charter-prioritized large trees received source-level adjudication below;
- 2 exact stores were protected without mutation: `ddm_bs3_born_small_resolved` and
  `ddm_w96a_aligned_window`;
- 5 `cold_store*` / `vertigo_coldstore*` namespaces were deferred without mutation because their
  existing reconstruction contracts were not needed to meet the target;
- the remaining 209 direct-child trees stayed `HOLD`: this pass did not establish a terminal,
  superseded owner for them in the searched authority surfaces, and the group includes recent or
  still-consumed trees. This is a bounded absence, not a claim that all 209 are live.

The seven prioritized rows were ranked by closure confidence first, then allocated bytes and the
presence of raw-array/checkpoint content. Allocated/logical denominators are the pre-action survey,
except that the B2E certificate's own allocated denominator includes 1,048,576 B of already-durable
SR3 manifest/failure-receipt allocation.

| rank | tree | allocated / logical bytes | closure evidence and ownership | measured compression | disposition |
|---:|---|---:|---|---:|---|
| 1 | `ddm_sa1` | 26,447,839,232 / 24,595,085,369 | `ddm_pc2_pose_carrier_live_remainder_20260826.md`: SA1 is `CLOSED-FAMILY`; canonical `ddm_pc2_sa1_film` completed; GB1 supersedes this retained lane | **2.557689x**, 9,616,134,370 B | `RECLAIMED_VERIFIED` |
| 2 | `ddm_b2e_f2_alone_run` | 29,980,622,848 / 29,814,524,961 | `ddm_b2e_edit_replay_admission_verdict_20260816.md`: specific burn-2 window is `REGIME_THESIS_INSTANCE_REFUTED`, receiver-closed, and says not to rerun the same window | **2.568771x**, 11,606,530,604 B | `RECLAIMED_VERIFIED` after SA1 created headroom |
| 3 | `ddm_ai1_20260809` | 34,312,290,304 / 34,266,565,221 | `ddm_ai1_ans_receiver_integration_20260809.md`: pure and temporal receiver instances are `FIRED-AND-FOLDED`; an external contest-row follow-on remains queued | not materialized | `HOLD`, target already met |
| 4 | `ddm_wd2_width_distillation` | 33,809,235,968 / 33,527,581,424 | `ddm_wd2_ep60_advisory_refusal_verdict_20260815.md` and `ddm_hv2r_arm_disposition_20260817.jsonl`: ep60 instance refused/folded, but the wider family is not globally closed | not materialized | `HOLD`, lower closure confidence |
| 5 | `ddm_tv1_tolerance_curve` | 51,398,049,792 / 51,271,931,930 | TV2 proves consumption of TV1 products, but the searched source did not provide an explicit TV1 tree-retirement receipt | not materialized | `HOLD`, no terminal tree citation found |
| 6 | `ddm_wc1_advisory_decode_wallclock_20260815` | 31,051,087,872 / 30,773,970,364 | `ddm_wc1_advisory_decode_wallclock_20260815.md` retains a `QUEUED-WITH-A-FIRE-ORDER` full-n600 consumer | not materialized | `HOLD`, live follow-on |
| 7 | `ddm_rx2_current_mc36_label_hpac` | 58,603,995,136 / 47,458,937,798 | `ddm_rx2_mc36_label_hpac_20260814.md` retains terminal-identity and exact-T4 consumers | not materialized | `HOLD`, live follow-on |

All seven trees were older than 24 hours at survey. Age alone never admitted a tree. The fleet was
`codex arms live: 0/4` at each selected-tree launch and again at each deletion gate.

## Reclaim ledger

### SA1

- Original: 13,956 files, 758 directories, 0 symlinks; 24,595,085,369 logical B;
  26,447,839,232 allocated B.
- Manifest:
  `/Volumes/APDataStore/pact/ddm_sa1/SR3_ORIGINAL_MANIFEST.jsonl`, 3,088,238 B,
  SHA-256 `cc93d09c39dd8fc331d8b33360cc0f63563c9d4cdcb1ec2d283bccc655e7dd5d`.
- Archive:
  `/Volumes/APDataStore/pact/ddm_sa1/SR3_ORIGINAL_TREE.tar.zst`, 9,616,134,370 B,
  SHA-256 `bd13d225ec69a969db2523245824f79e7acee74a9ed9dc038ff0aacfa5579d65`.
- Verification receipt:
  `/Volumes/APDataStore/pact/ddm_sa1/SR3_VERIFICATION_RECEIPT.json`, 726 B,
  SHA-256 `2131bab99cddae24589146ffea42b3c42c504087ba13338df66cbe314003e6b5`.
- Reclaim certificate:
  `/Volumes/APDataStore/pact/ddm_sa1/SR3_RECLAIM_CERTIFICATE.json`, status
  `RECLAIMED_VERIFIED`, SHA-256
  `6167a8c394a63fa8d6619b2e09187c9766160bfdc5816b1e86fee187d491f3f4`.
- `df`: 12,911,771,648 B before -> 29,736,042,496 B after; certified delta
  **+16,824,270,848 B**.
- Reconstruction:
  `zstd --long=31 -d -c /Volumes/APDataStore/pact/ddm_sa1/SR3_ORIGINAL_TREE.tar.zst | tar -xf - -C /Volumes/APDataStore/pact/ddm_sa1`.

### B2E

- First attempt at the initial AP denominator stopped fail-closed when live AP free crossed the
  2 GiB abort floor. Its 11,065,622,528 B partial was SHA-recorded as
  `13642a699b7976d3661cd7dd15d28365f402b7e59956e6417fa4a5f78cda5905` in
  `SR3_FAILURE_RECEIPTS.jsonl`, then removed as certified rebuildable scratch. All originals and
  the complete manifest remained. The first receipt names the resulting broken pipe rather than
  the floor cause; the certifier was corrected so subsequent floor-caused pipe failures record the
  causal floor explicitly. This first-attempt shape is closed at the initial-headroom condition.
- After SA1 reclaimed capacity, B2E resumed from the unchanged manifest and completed.
- Original: 1,296 files, 98 directories, 0 symlinks; 29,814,524,961 logical B;
  29,981,671,424 certificate allocated B.
- Manifest:
  `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_ORIGINAL_MANIFEST.jsonl`, 342,357 B,
  SHA-256 `bae895d34ebb7f16267eba25cd6b66d1964dc0c3ef1e1ffb9fb3b9cdd672fed1`.
- Archive:
  `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_ORIGINAL_TREE.tar.zst`,
  11,606,530,604 B, SHA-256
  `37fbfaf43e80a259054004763c9d5964a4726c5bf95d28ef3ecdaf8514bb6e31`.
- Verification receipt:
  `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_VERIFICATION_RECEIPT.json`, 738 B,
  SHA-256 `b1e6ca924715a47912e402c144d8e8ee6b349f2e3409ce255c86a1148cbb1df8`.
- Reclaim certificate:
  `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_RECLAIM_CERTIFICATE.json`, status
  `RECLAIMED_VERIFIED`, SHA-256
  `3a925c10499aac090a08a5e3a290316fd50550919a3e2ef0a922fcdb4415b200`.
- `df`: 29,736,042,496 B before -> 48,109,322,240 B certificate-after; certified delta
  **+18,373,279,744 B**. The later live `df` is 48,109,191,168 B after filesystem metadata.
- Reconstruction:
  `zstd --long=31 -d -c /Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/SR3_ORIGINAL_TREE.tar.zst | tar -xf - -C /Volumes/APDataStore/pact/ddm_b2e_f2_alone_run`.

Both archives use Zstandard CLI v1.5.7, level 15, `--long=31`, and four threads; the certificate
defines deterministic identity over the same manifest, zstd version, settings, and thread count.
No `/private/tmp/ddm_sr3_verify_*` directory remained after either run. The explicit protected
stores remained present and were not opened by the certifier.

The prior-law prediction that these trees would compress at >=3x is **refuted on two measured
trees**: SA1 measured 2.557689x and B2E measured 2.568771x. The stronger charter falsifier
`ratio < 1.5x` did not fire, and closure adjudication did not empty the candidate list.

## RECALL EVIDENCE

Sources searched before adjudication:

- `.omx/research/` memos and receipts using content queries for
  `certify move`, `reclaim certificate`, `content addressed`, `storage preflight`, each of
  `sa1|b2e|ai1|wd2|tv1|wc1|rx2`, and `terminal|closed|folded|superseded|owner|fire trigger`;
- `.omx/state/canonical_task_status.jsonl`, lane/fleet state, `main_hot_state.md`, and the W96B
  build/storage receipt plus sealed fire order;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/spec surfaces, and
  `tools/list_canonical_equations.py --json` with `storage|compress|reclaim|content-address|dedup|zstd`;
- the SR2 certify-and-move implementation/memo and FB2 cleanup certificate exemplars.

Beyond the charter seeds, the search found the 2026-08-26 PC2 `CLOSED-FAMILY` source for SA1, the
B2E memo's explicit same-window no-rerun verdict, current WC1/RX2 follow-on consumers, W96B's
post-CAS measured demand, and the current GB1 frontier. That changed execution order to SA1 first,
then a headroom-conditioned B2E resume, while holding WC1/RX2 and every tree without explicit
retirement evidence. The canonical equation registry did not contain a storage-custody equation
that superseded the manifest/archive/round-trip contract; codec-rate equations found there were
out of domain for filesystem custody.

## Tool and verification surface

`experiments/ddm_sr3_ap_certify_compress_reclaim.py` implements direct-child validation, exact
protected/custody refusals, 24-hour age gating, fleet-idle gates at launch and deletion, every-file
SHA manifests, deterministic tar+zstd, AP floor aborts, local-scratch full extraction, complete
hash equality, crash-resumable pre-certificates, exact top-level removal, and final machine-readable
certificates. `experiments/tests/test_ddm_sr3_ap_certify_compress_reclaim.py` covers deterministic
repeat archives, full manifest/extract equality, protected/custody refusal including a `..` bypass,
exact-target removal, partial-reclaim resume, same-size/same-mtime tamper rejection, and detached
progress pipes. Result: 5 focused tests passed; Ruff passed; both Python files received two fresh
review passes after the final edit.

## Typed MAIN handoff

Disposition: **QUEUED-WITH-A-FIRE-ORDER**.  
Owner: **MAIN/operator**.  
Consumer store: `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`.  
Fire trigger: **GREEN** when live AP free is at least 33,569,378,304 B and the sealed W96B
implementation/config gates remain green. Live AP free is 48,109,191,168 B; use
`SEALED_FIRE_ORDER_W96B.json` and run the two aligned seeds sequentially. SR3 does not launch them.

Canonical ledger receipt: actor `ddm_sr3` consumes #1165's storage input by reopening
`ddm_w96a_aligned_config_renderer_window` from `blocked` to `in_progress`; append-only ledger line 711,
event timestamp `2026-08-26T23:07:27.276980Z`, status `in_progress`, owner `MAIN`, tests `green`.

**GESTALT-DELTA:** lossless custody compression converted 52.554 GiB of terminal allocated retention
into 19.765 GiB of archives plus small manifests/certificates, reclaiming 32.780 GiB across the two
certificate intervals and greening W96B without discarding a payload; the assumed >=3x ratio was
replaced by the measured ~2.56x regime.

Own-vehicle frontier: **GB1 S = 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**, archive SHA-256 `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`; unchanged by SR3.
