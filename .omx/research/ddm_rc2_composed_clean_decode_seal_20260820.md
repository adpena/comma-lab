# ddm_rc2 — clean ship object and composed upgrade decode-proven, sealed

## Verdict

The charter amendment's two objects have different dispositions.

- **Object A — SHIPPABLE.** The exact clean tree
  `/Volumes/VertigoDataTier/pact/ddm_rr8/candidate_runtime_jg5_native_corrector`
  already executed its real `inflate.sh` at full n600 scope with the unchanged jg5 archive
  `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`
  (180,625 B). Its retained output is 3,662,409,600 B with SHA-256
  `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`, exactly
  the retained local instrumented-tree and jg5-baseline digest. The run reports
  `free_corrector=NativeFreeCorrector`. Under Amendment 1, this closes the only open
  clean-tree behaviour question: Object A is a shippable sub-0.15 candidate and its
  existing exact authority result remains **S=0.14839100138338618 @ 180,625 B
  [contest-CUDA T4, n600]**. No new T4 score fire is required to establish that score.
- **Object B — DECODE-PROVEN AND SEALED; AUTHORITY ROW QUEUED.** Two independent full-n600
  executions of the composed tree's real `inflate.sh` returned rc=0, compiled and used
  `NativeFreeCorrector`, consumed the RR5-marked archive, and each emitted the same local raw
  digest as jg5. The one candidate content identity is frozen in axis-specific CUDA and CPU
  seal envelopes for a paired, sequential fire. No Modal dispatch or scorer ran here.
  Consequently **S=0.14827847122030854** remains a **contest-CUDA hypothesis**, not a row.

## Per-item execution and measurement

| item | what EXECUTED at what scope | measured result | verdict |
|---|---|---|---|
| 1A — clean tree behaviour | Retained `advisory_native_r3`: the clean tree's real `inflate.sh`, full n600, macOS arm64 CPU advisory. This arm re-read the producer log and independently joined its retained output manifest against the instrumented-tree `advisory_composed_r1` and jg5 `advisory_final` manifests. The settled full run was not duplicated. | Clean runtime: `decode_and_render_seconds=682.0396792090032`; outer inflate wall `682.8109928751364 s`; raw `3,662,409,600 B`, SHA `7246a4ff…f2de7`; decoded-token SHA `cc10a7b0…6efb`; `free_corrector=NativeFreeCorrector`. All three local manifests report the same raw byte count, raw SHA, and aggregate SHA `23fc14a6…22020`. `[macOS-CPU advisory, n600]` | **PASS / Object A SHIPPABLE.** Clean behaviour is byte-neutral against the instrumented tree and baseline on the measured local axis. Per Amendment 1, the already-landed jg5 T4 authority result attaches to the unchanged archive; no new T4 score is needed for Object A. |
| 2A — composed real receiver, run 1 | `/usr/bin/time -p bash <composed>/inflate.sh <extracted> <inflated> <public_test_video_names.txt>` through `tools/launch_detached_process.py`, full n600, macOS arm64 CPU advisory. | rc=0; launcher `1768.097 s`; receiver `decode_and_render_seconds=1681.777882708935`; raw `3,662,409,600 B`, SHA `7246a4ff…f2de7`; token checkpoint `117,964,800 B`, SHA `cc10a7b0…6efb`; `free_corrector=NativeFreeCorrector`. `[macOS-CPU advisory, n600]` | **PASS.** Real compile/decode path completed and matched retained jg5 local bytes. Timing is a contended observation because run 2 overlapped it; it is not a clean-alone or CI-wall estimate. |
| 2B — composed real receiver, determinism repeat | `bash <composed>/inflate.sh <extracted> <inflated> <public_test_video_names.txt>` through the same governed launcher, independent full n600 run, macOS arm64 CPU advisory. | rc=0; launcher `1283.395 s`; receiver `decode_and_render_seconds=1250.8826307908166`; raw and token checkpoint byte counts and SHAs exactly match run 1; `free_corrector=NativeFreeCorrector`. `[macOS-CPU advisory, n600]` | **PASS.** Independent same-axis receiver repeat is byte-identical. This timing is also contended and does not answer the 464-second clean/composed transfer question. |
| 2C — RR5 rider engagement | Read the exact retained archive member `p` through the composed runtime parser, then exercised the full receiver in 2A/2B. Scope: exact 180,456-byte archive; no parser substitute used for the behavioural verdict. | `p`: `180,356 B`, SHA `83fa979c…1cdf3`; header `magic=RX1M`, `version=1`, `reserved=10`, so `reserved & 0x08 != 0`; parsed `carrier_blob=22,316 B`, `token_stream=113,847 B`, compressed models `66,413 B`. The exact source dispatches `restore_carrier_body` when that flag is set, and both full runs returned the expected raw bytes. | **PASS.** The rider branch and native port ran in the same real-receiver executions. A parse-only claim is not being substituted for the two full decodes. |
| 3 — seal one candidate content identity | `tools/make_candidate_seal.py` measured the exact Object B runtime and archive from disk and ran the consumer validator. Because the fire path binds a seal to one contest axis, the same one content identity was frozen in CUDA and CPU envelopes for paired execution. No hand-typed SHA populated either seal; `--verify-archive-sha` was only a mismatch guard. | Both seals: archive `180,456 B`, SHA `df7fd266…e2080`; runtime `37` shippable files, `666,709 B`, content SHA `749ce030…f5225`; validator `SEAL_VALID`. CUDA disk-file SHA `b90d8300…35cb`, embedded seal SHA `2e32079c…7005`. CPU disk-file SHA `0651dd69…c934`, embedded seal SHA `a7c915e0…8917`. | **PASS / QUEUED-WITH-A-FIRE-ORDER.** Object B is content-frozen for a sequential CPU/CUDA pair. MAIN owns dispatch; ddm_rc2 made no provider call. |

## Object A authority boundary

The clean local run's scorer fields are not used as contest evidence: its environment is explicitly
`[env-mismatch advisory]` and its GT lineage is the CPU/PyAV branch. The authority score stated above
comes only from the exact retained jg5 contest-CUDA T4 row, recomputed from its components rather than
from the receipt's rounded `final_score=0.15` display field. Clean-vs-instrumented equality is proved
on the local axis by output bytes; the amendment expressly makes that the remaining shippability gate
for unchanged Object A.

The 464.558564563-second figure remains one instrumented T4 instance. The clean local Object A wall is
682.8109928751364 seconds at the outer inflate wrapper, and no number is transferred between those
regimes.

## Object B composition conclusion

The prior-law prediction survived twice at full local n600 scope. Both full receivers produced
`7246a4ff…f2de7`, matching the retained local jg5 raw. Thus Object B's local semantic byte identity is
**MEASURED**, not inferred. Its native corrector compiled and ran (`NativeFreeCorrector`), and its
RR5 flag requires the carrier-body restoration branch that precedes the successful render.

The cross-axis boundary remains binding. The retained jg5 T4 raw is
`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`, not the macOS digest.
Object B has not produced a T4 or Linux contest-CPU raw. Therefore unchanged T4 `d_seg=0.00020139`
and `d_pose=0.00000637`, and the rate-only arithmetic

`0.14839100138338618 - 25 * 169 / 37,545,489 = 0.14827847122030854`

are a falsifiable **contest-CUDA hypothesis** until the fresh authority row lands. They are not a
pointer-moving result.

## Timing boundary

The two Object B receiver observations were intentionally retained, but their overlap means neither
is an uncontended wall benchmark. The measured receiver totals are 1681.777882708935 seconds and
1250.8826307908166 seconds `[macOS-CPU advisory, n600]`; the corresponding governed launcher totals
are 1768.097 and 1283.395 seconds. They show two successful within-budget local decodes, not a clean
estimate of composed T4 wall time and not evidence that 464.559 seconds transfers.

## Payload custody

Nothing materialized by ddm_rc2 was discarded.

- Machine-readable summary: `/Volumes/APDataStore/pact/ddm_rc2/COMPOSED_DECODE_RECEIPT.json`,
  `5,652 B`, SHA `afa2dc86e73ebb2ed6c3f56f259caa1bfeac10353f4251e2bffd499baa9518fc`.
- Run 1 root: `/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r1/`; final raw
  `3,662,409,600 B`, SHA `7246a4ff…f2de7`; retained token checkpoint `117,964,800 B`,
  SHA `cc10a7b0…6efb`; log `2,538 B`, SHA `09bba6d0…ba60`.
- Run 2 root: `/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r2/`; final raw
  `3,662,409,600 B`, SHA `7246a4ff…f2de7`; retained token checkpoint `117,964,800 B`,
  SHA `cc10a7b0…6efb`; log `2,505 B`, SHA `73c8bfb6…2e78d`.
- The first launch's premature failure diagnosis is preserved as
  `FAILED_LAUNCH_RECEIPT.json`; its append-only `FAILED_LAUNCH_RECEIPT_SUPERSESSION.json` withdraws
  that diagnosis. The governed launch receipt is authoritative: rc=0. No evidence was overwritten.
- Existing clean/instrumented/baseline raw payloads remain under their cited APDataStore/Vertigo
  roots. The charter's Vertigo-full rule was respected for all new bulk.

## Sealed candidate and exact MAIN fire-order

One candidate content identity is sealed twice only because `fire_modal_auth_eval.py --seal` derives
and locks the contest axis from the envelope.

1. CUDA seal:
   `/Volumes/APDataStore/pact/ddm_rc2/CANDIDATE_SEAL_rc2_composed.json`, `4,062 B`, disk-file
   SHA `b90d830065243181ff7804437235dcad0fa7973b469575bd9f2ee141dc1335cb`, embedded seal SHA
   `2e32079c5de2cff9e2c2e6788eb74e8152127273aa0f977cf11cb302a3547005`.
2. CPU seal:
   `/Volumes/APDataStore/pact/ddm_rc2/CANDIDATE_SEAL_rc2_composed_cpu.json`, `3,426 B`, disk-file
   SHA `0651dd694cd3f94e9b5d3195904d28ac8ac938d728c483863b6f38b3c8f7c934`, embedded seal SHA
   `a7c915e0a85d93ce47f681e88dd683ec6c3d7bfb6934055ef01bf02d85108917`.

MAIN must first verify `claim_lane_dispatch.py summary` has no active full-n600 Modal/scorer claim,
both seals still validate against disk, and neither pointer baseline has moved. Then execute this
paired group **sequentially**: CUDA claim/fire/harvest/terminal-close first, CPU only after CUDA is
terminal.

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_rc2_composed_cuda_20260820 \
  --platform modal \
  --instance-job-id modal:ddm_rc2_composed_cuda_r1 \
  --agent MAIN \
  --status active_paid_dispatch \
  --notes 'ddm_rc2 Object B paired-axis CUDA leg; pair_ddm_rc2_composed_cpu_cuda_20260820; seal b90d830065243181ff7804437235dcad0fa7973b469575bd9f2ee141dc1335cb'

.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_rc2/CANDIDATE_SEAL_rc2_composed.json \
  --output-dir /Volumes/APDataStore/pact/ddm_rc2/t4_row_r1 \
  --lane-id lane_ddm_rc2_composed_cuda_20260820 \
  --instance-job-id modal:ddm_rc2_composed_cuda_r1 \
  --claim-agent MAIN \
  --pair-group-id pair_ddm_rc2_composed_cpu_cuda_20260820 \
  --claim-policy require_active
```

After the CUDA job is harvested and its claim is terminally closed:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_rc2_composed_cpu_20260820 \
  --platform modal \
  --instance-job-id modal:ddm_rc2_composed_cpu_r1 \
  --agent MAIN \
  --status active_paid_dispatch \
  --notes 'ddm_rc2 Object B paired-axis CPU leg; pair_ddm_rc2_composed_cpu_cuda_20260820; seal 0651dd694cd3f94e9b5d3195904d28ac8ac938d728c483863b6f38b3c8f7c934'

.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_rc2/CANDIDATE_SEAL_rc2_composed_cpu.json \
  --output-dir /Volumes/APDataStore/pact/ddm_rc2/cpu_row_r1 \
  --lane-id lane_ddm_rc2_composed_cpu_20260820 \
  --instance-job-id modal:ddm_rc2_composed_cpu_r1 \
  --claim-agent MAIN \
  --pair-group-id pair_ddm_rc2_composed_cpu_cuda_20260820 \
  --claim-policy require_active
```

Adjudication must compare the T4 raw to `6bf8acf8…e79883`, require
`free_corrector=NativeFreeCorrector`, confirm both receipts bind archive `df7fd266…e2080` and runtime
content `749ce030…f5225`, compute the score from exact archive bytes plus reported components, and compose
the two-row report-8dp error bound. Any divergence localizes first to the failed axis and then to the
port leg versus rider leg; do not transfer one axis to the other.

## RECALL EVIDENCE

The recall census searched `.omx/research/`, arm receipts on both SSD roots,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, current hot state, canonical pointer, lane
registry, and task/dispatch ledgers using the content queries
`clean port|native corrector|instrumented|restore_carrier_body|RR5|composed candidate|decode identity|candidate seal|dual-axis|pair-group-id|464.559|7246a4ff|6bf8acf8|df7fd266`.
It also ran `tools/list_canonical_equations.py --json` and filtered for
`decode|identity|composition|rate|seal|axis|receiver`, and read the actual evaluator, seal producer,
seal consumer/fire path, launcher, rc1/rr5/rr6/rr8/rv16/CD1 source memos, research index/DAG, and live
pointer rather than relying on the charter seeds alone.

Beyond the seed list, recall found the already-complete clean-tree full-n600 `advisory_native_r3`
execution and its retained raw manifest. It also found the sibling local instrumented run and baseline
manifest, all with the same raw digest. This changed the plan materially: Object A was closed by
re-deriving the retained evidence instead of wasting another 3.66 GB decode, consistent with the
already-settled/no-signal-loss rule. Recall also exposed that the current fire tool is paired-by-default
and an individual seal is axis-bound; that changed the handoff from an ambiguous “T4 fire” to two
same-content axis envelopes and a sequential shared pair-group. The live pointer superseded the stale
frontier paragraph in the common contract: it remains jg5 at 0.14839100138338618 [contest-CUDA T4,
n600].

No further cheaper or already-executed Object B authority row was found in the searched scopes. The
two new full composed decodes therefore paid the actual remaining behavioural debt; no parse-only or
proxy result was promoted.

## Measurement boundary and mission status

ddm_rc2 measured real full-n600 local receiver behaviour and preserved every output. It did **not**
measure Object B on contest-CUDA T4, Linux contest-CPU, either scorer, or an uncontended composed wall.
It did not run Modal and did not move the pointer. Object B's 169-byte rate credit is real archive
arithmetic; its score is conditional on the unmeasured same-axis semantic identity.

OWN-VEHICLE FRONTIER: unchanged at **S=0.14839100138338618 @ 180,625 B [contest-CUDA T4, n600]**,
archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` (Object A / jg5).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/APDataStore/pact/ddm_rc2/{t4_row_r1,cpu_row_r1}/` followed by `.omx/state/main_hot_state.md`; fire trigger: no active full-n600 Modal/scorer claim, both axis seals above remain `SEAL_VALID` against the same archive/runtime content, and their bound pointer baselines have not moved; action: execute the exact shared-pair-group CUDA-then-CPU claim/fire order above, retaining and harvesting every payload before adjudicating Object B.

## LIVE-HYPOTHESES

- Object B will reproduce `6bf8acf8…e79883` on T4 because the only archive change is the round-trip-proven RR5 carrier-body recode, the native-port merge is line-disjoint, and two full local real-receiver runs reproduced their same-axis jg5 raw. T4 receiver identity is still untested.
- If T4 raw identity holds, Object B will preserve jg5's `d_seg` and `d_pose`, making the 169-byte rate-only result **S=0.14827847122030854** plausible. This is not authority until recomputed from the fresh receipt.
- The composed T4 wall will likely remain comfortably below 30 minutes because the instrumented native-port sibling completed in 464.559 seconds on T4 and both composed local runs completed below 30 minutes even while contending. Cross-regime timing transfer remains forbidden; only the fresh row can test it.

## DEAD-ENDS

- Re-running Object A's unchanged clean tree locally is closed: a retained full-n600 real-`inflate.sh` execution already proves local byte identity and native-corrector use. Reopen only if the archive or clean runtime content changes.
- Treating rc1's parse proof as composed execution is closed: two full receiver runs now exercised native token correction, RR5 restoration, render, and final raw hashing together.
- Using a hand-typed digest as a seal is closed: both axis envelopes were computed from disk and passed the consumer validator; typed archive SHA was used only as a refusal guard.
- Transferring the 464.559-second instrumented T4 wall to the clean or composed trees is closed. The new local timings are measured but contended and cross-regime.
- Treating the initial run-1 process-visibility failure as a child failure is closed: the append-only supersession withdraws it, and the governed launcher plus done receipt record rc=0.
- Promoting Object B from local raw identity or rate arithmetic is closed: no Object B contest-CUDA or contest-CPU score exists yet, and the canonical pointer is unchanged.
