# ddm_vr7 — certified deletion plan, 2026-09-10

| Selected bulk [filesystem bytes measured; scorer-free] | DELETABLE files | Logical bytes | GiB |
|---|---:|---:|---:|
| sj1_overlay | 7 | 12,818,433,600 | 11.938097 |
| sj1_parseback | 4 | 14,649,638,400 | 13.643539 |
| tc3_prepare | 600 | 1,729,724,545 | 1.610932 |
| tc3_public_raw | 1 | 3,662,409,600 | 3.410885 |
| tc3_trace | 600 | 2,174,673,405 | 2.025322 |
| **Total** | **1,212** | **35,034,879,550** | **32.628774** |

**Plan complete; inventory deletion is reserved for MAIN. Inventory deleted: 0 B.** All 1,212 destination files freshly matched their retained hashes. The native `plan-retained` run admitted all 1,212 after certificate and reference validation. This meets the charter's ≥30 GiB target. It is a logical-byte reclaim plan, not a measurement of free-space gain; filesystem allocation and concurrent writers can change `df`.

The prior-law prediction is partly falsified, at INSTANCE scope: all 11 SJ1 files certify, but the selected superseded TC3 bulk is only **7.047139 GiB**, below the predicted ≥8 GiB, and the total is below the predicted ≥33 GiB. The entire inventoried pre-rebase TC3 tree was only about 7.52 GiB before excluding retained inputs and metadata. No representative hash mismatch occurred. Do not extrapolate the charter's approximate 10 GiB TC3 estimate into a deletion receipt.

## Rebuild certificates and inheritance

| Representative [macOS-CPU byte identity; n600; scorer-free] | Bytes | Rebuilt and original SHA-256 | Certificate |
|---|---:|---|---|
| pass5 / sj1_parseback | 3,662,409,600 | `f7cd6a31635fcba5fce0053db19e2543c8b1999fe83d2f6f37be4f666a9a8a12` | [rebuild_pass5_certificate.json](ddm_vr7_20260910/rebuild_pass5_certificate.json) |
| multipass / sj1_parseback | 3,662,409,600 | `bfe96bac3066dfec78cf32075582686ac74e6c1756df6008522e589bf0709b4b` | [rebuild_multipass_certificate.json](ddm_vr7_20260910/rebuild_multipass_certificate.json) |
| compose39 / sj1_parseback | 3,662,409,600 | `c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5` | [rebuild_compose39_certificate.json](ddm_vr7_20260910/rebuild_compose39_certificate.json) |
| pass5 / sj1_overlay | 1,831,204,800 | `494484d210cc91ac33ff5107ea40616509b3a6536f541aa9845fc0c8164f80cc` | [overlay_pass5_certificate.json](ddm_vr7_20260910/overlay_pass5_certificate.json) |
| multipass / sj1_overlay | 1,831,204,800 | `4560927f432a20db598d028dadf90614967d6f11a54125803d87e686306e2859` | [overlay_multipass_certificate.json](ddm_vr7_20260910/overlay_multipass_certificate.json) |
| compose39 / sj1_overlay | 1,831,204,800 | `37ea3842982a7fe7d1c4c06216fde20e5bc4d1b6c15841d71981085f0fe2c653` | [overlay_compose39_certificate.json](ddm_vr7_20260910/overlay_compose39_certificate.json) |

The three parse-backs execute the original `experiments/ddm_sj1_joint_admission.py parseback` from each retained sealed archive/runtime with four CPU threads. All original seals, archives, source code, runtime file sets and receipts remain in custody. Each certificate pins its immutable config, original output, actual argv and measured fresh output. Scratch was removed only after twin hashes and a durable certificate. The three overlays execute the same SJ1 field loader and JG1 batch-one renderer through a scorer-free adapter; each runs all 600 pairs. No SegNet/GT table was loaded by the overlay adapter.

**Overlay correction:** the final archive alone cannot reproduce every pre-admission overlay. The descriptor also pins the exact retained field named by that overlay's original `OVERLAY.json`. Rendering from the sealed renderer plus that field reproduced the recorded SHA. No archive-only overlay reproducibility claim is made.

Six SJ1 files are directly rebuilt; the other five inherit a representative only within the **same original SJ1 root and output kind**. Instance archive/field values can differ and are pinned separately, with each original output receipt and MOVED destination individually checked. Every row declares this scope in `inheritance`; these five were not individually rerun. `descriptors.json` and the seven `overlay_row_*.json` configs carry exact effective restore argv per file, output destinations and input hashes.

The raw certificate wrapper's historical v1 source is preserved as `rebuild_worker_v1_source.txt`; the later overlay helper version is preserved as `rebuild_worker_v2_source.txt`, with identity receipts. The final wrapper adds interrupted-run recovery and a lint-only conditional simplification. Original SJ1/JG1 numerical code is unchanged and hash-pinned. Do not claim the final wrapper itself produced the older raw certificates.

## TC3: superseded base and proof boundary

All 1,201 selected TC3 files are under the old **move39** root; its INPUTS archive is `8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4`. The field is `4aa519a25e4b02afb564498025b366cb9007ea663093c60bf8ce90d079dc8791`. The selected set is exactly 600 `trace/frames/frame_*.npz`, 600 `prepare_v2/frames/frame_*.npz`, and `public_identity/output/0.raw`. No separate move37 bulk is claimed in this selected inventory. The live `rebase_move40/` tree is categorically excluded, including its newer exact-row data.

Trace admission joins the retained completed n600 twin-byte receipt, each original per-frame receipt, historical input/code/native-library pins, and the fresh 600-file SHA census. Prepare admission joins the retained full-n600 receipt, each frame receipt and fresh 600-file SHA census. The ordinary old CLI refuses because it reads the now-advanced pointer. The historical replay adapter replaces only input loading and output routing; original `trace.run` and `price.prepare` execute unchanged, against pinned retained move39 inputs.

The final restore configuration was actually verified on **frame 0 of both stages**, matching their original NPZ hashes (`992b7ed0…` trace, `e9f5bd52…` preparation). This is a command-mechanism check, not a new full600 replay or statistical quality claim. Full600 evidence is retained historical evidence, not relabeled new measurement. `tc3_replay_current_certificate.json` binds the exact current config; `--stop-after 600 --retain-output` is the full restoration command. Frames are saved independently and resume from disk. Missing current original bulk after MAIN applies is allowed only when the rebuilt bytes match retained per-file receipts.

The old public raw was freshly full-file twin-hashed against the retained cold-stored rp1 source raw: 3,662,409,600 B, SHA `fd4b08e6aa0e967cc1453b9363c5f5ad243cb3bb57417f0f383cdb4bbfbfb2e8`. It is certified against the retained source-runtime parse-back receipt and exact identity, **not** a new raw decode. `tc3_raw_identity.json` records both full current measurements. Source raw custody stays outside this deletion set.

The first historical replay stopped because a retained `rank_control.envelope` had not been copied into owned scratch; it did not expose a numerical mismatch. The corrected command passed. Failed scratch bulk was removed only after successful-replay twin comparison and `failed_tc3_scratch_cleanup.json`; failure metadata is retained.

## Admission and apply boundaries

The planner accepts explicit data descriptors rather than spoofing old family certificate statuses. Legacy VR3/VR5 admission remains intact. The new target exception is structurally limited to the eleven named SJ1 cold-store files and old move39 frame/raw paths. Symlinks, unexpected runtime entries, missing dependencies, MOVED/source reappearance, archive/seal/code/config/proof drift and missing representative proof fail closed. Config dependency pins are checked as part of admission and rechecked on apply.

Reference recall initially found 64 mentions across 11 SJ1 rows in 53 files. The reviewed exclusions are exact hash-pinned historical observations: MOVED, completed render/parse-back/score-shard receipts, completed launch logs and MAIN's move ledger. The EB2 MOVED resolution and completed partition-cache memo record custody/cache consumption; the memo states no raw decode. No executable consumer is excluded. `reference_classification.json` records these decisions; `closures.json` pins the exact files. The final configured repository/live-tree scan found no remaining hits. This is bounded absence in the planner's configured scopes, not proof that no possible future consumer exists.

Apply rechecks the complete certificate, current stat and full SHA, owner command and cwd census, open descriptors and references immediately before deletion, writing the existing fsynced journal. The sandbox refused `ps` with Operation not permitted for both owners. Accordingly **process safety was not measured here**; `process_visibility.json` records the refusal. MAIN must run with full process/cwd/lsof visibility and must retain every row whose gate refuses. A DELETABLE plan row is not authorization to bypass those runtime gates.

Protected and untouched: `upstream/`, all original seals, archives, candidate runtimes, source receipts/ledgers, pointer tree, rp1 round2, eb2 and TC3 `rebase_move40`. The six representative bulk outputs and replay copies were owned scratch, auto-cleaned under the charter. At final scratch census only 367,116 B of launch/certificate/progress metadata remained. All launchers used `--nice-best-effort`; at most two measurement workers overlapped. Maximum concurrent raw/overlay allocation was below the 8 GiB own-scratch cap. Zero scorer jobs, GPU dispatches or scores were run.

## Exact MAIN fire order

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN**. Consumer store: `.omx/research/ddm_vr7_reclaim_plan_20260910.jsonl`; resulting journal: `.omx/research/ddm_vr7_apply_journal_20260910.jsonl`. Fire trigger: harvest this plan, land the reviewed source, then execute with full process visibility and passing current gates. The command below is recorded, **not executed**. Cwd `/Users/adpena/Projects/pact`.

```sh
.venv/bin/python -B experiments/ddm_vr3_certified_raw_reclaim.py apply --ledger .omx/research/ddm_vr7_reclaim_plan_20260910.jsonl --expected-ledger-sha256 fbff257ddab7dac5e3160ea3e4b667a3046f28d66b53eea5dde109cc48146394 --journal .omx/research/ddm_vr7_apply_journal_20260910.jsonl --target-bytes 35034879550
```

Machine-readable command, exact ledger hash and fire order: `ddm_vr7_20260910/apply_fire_order.json`. All expected restore bytes are retained in originals until MAIN applies. The deletion ledger SHA above must match exactly; do not regenerate the plan silently after a drift.

## Validation and landing

Ruff passes on all seven Python files. Targeted VR3, VR5 and VR7 tests: **95 passed, 2 skipped** (the existing live-process tests require visibility unavailable in this sandbox). Tests exercise legacy paths, descriptor admission/apply, archive/code/MOVED/argv/proof/runtime drift refusals, protected paths and symlinks, exact observation pins, real direct-script entrypoints and interrupted own-scratch recovery. Both visible review passes marked every Python file through `tools/review_tracker.py`; per-file content hashes and tracker receipts are retained in `review_pass1.json` and `review_pass2.json`. Actual integration evidence is the six full representative matches, fresh complete inventory hash census, final two-stage frame replay and actual all-admitted native plan run.

Landing must use one serializer attempt with post-edit hashes, `[no-triality] [p0-ledger-ok]`, no attribution trailer. The shared staged index was empty at start and is preserved. Final serializer or denied-write fallback status is recorded separately in `ddm_vr7_20260910/landing_status.json`; inspect it before applying. If denied, MAIN owns the reviewed `landing.patch` landing before the apply fire order.

## RECALL EVIDENCE

Read the charter and common contract in full, PROGRAM, CLAUDE/AGENTS, operating handoff manual and live board. Own content recall searched `.omx/research/`, docs/configs, canonical research index, DAG FEEDs, task and lane surfaces for `retained.reproducer|certify.or.block|superseded.{0,30}trace|MOVED.json`; the receipts are `recall_0.json` through `recall_2.json`. Ran `tools/list_canonical_equations.py --json`: 483 entries, one incidental PA1 amplitude-law custody match, no directly applicable storage equation in that result; retained full output and summary.

Beyond charter seeds, `ddm_dk2_apdatastore_cold_store_sweep_20260904.md` showed a prior sevenfold bulk-class forecast miss and disk saturation with seven/eight movers beside live training. This changed the work to exact inventory arithmetic and at most two read/rebuild workers, with no inventory moves. `ddm_cs1_consolidation_debt_20260909.md` distinguishes exact blob custody from mere reachability; this reinforced current runtime/source pins and real replay checks. EB2's completed cache memo resolved an otherwise ambiguous reference as a pinned historical observation. Checked mv1 Git history: its helper was not landed in the searched path history, so MOVED manifests were read directly. Memory recall reinforced preserving the shared index and the one-attempt serializer-denial fallback; it supplied no live bytes or score facts.

## LIVE-HYPOTHESES

- The admitted batch can release up to 32.629 GiB of logical file contents after MAIN's live gates clear. Plausible because every file is present, freshly hashed and admitted; actual `df` gain and absence of live consumers remain unmeasured here.
- Same-root/same-kind restoration of the five inherited SJ1 instances is plausible from six successful full representatives and individually pinned inputs/receipts; those five are explicitly not independently rebuilt in this charter scope.

## DEAD-ENDS

- ≥8 GiB selected TC3 / ≥33 GiB total: closed for this inventory by measured byte sums; do not repeat the approximate charter forecast.
- Archive alone as an overlay reproducer: closed for pre-admission overlays; retain and use each recorded supplemental field.
- Ordinary historical TC3 CLI against the current pointer: closed as a restore route; it refuses advanced-base inputs. Use the verified pinned historical adapter.
- Cold-store presence or custody-text mentions as deletion proof: closed; only the complete certificate plus live apply gates admits deletion, and only reviewed exact observations are excluded.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `ddm_vr7_20260910/landing_status.json`; trigger harvest: land the reviewed scoped change using the serializer receipt or `landing.patch` if Git writes were denied.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `.omx/research/ddm_vr7_reclaim_plan_20260910.jsonl` and apply journal; trigger landed code plus full process visibility: execute the exact pinned apply command and report actual deleted bytes/free-space change; retain every refused row.

Live own-vehicle frontier, observed from current pointer and recomputed from its retained components: **S = 0.13758600733559048 @ 180,154 B [contest-CUDA T4 n600]**, `ddm_tc3_t4_lane_predictor_tail_20260910`, archive `299a8201…`. Components: seg 0.010636 + pose 0.0069928534948188355 + rate 0.11995715384077166. This arm did not move it; sub-0.12 remains unachieved.
