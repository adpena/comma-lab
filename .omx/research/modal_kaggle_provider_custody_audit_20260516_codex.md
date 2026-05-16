# Modal/Kaggle provider custody audit - Codex 2026-05-16

Tag: `research_only=true`. No score claim. No remote/GPU dispatch. No raw
result JSON edits. No lane-claim edits. This ledger is the only write from this
audit.

## Scope

Audit target: current Modal/Kaggle/provider result and lane-claim custody
surface, specifically:

- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/lane_registry.json`
- `.omx/state/modal_call_id_ledger.jsonl`
- `experiments/results/_modal_harvest_summary.json`
- Kaggle PR106 latent-sidecar harvest evidence under
  `reports/raw/kaggle_ingested/`
- D1 paired Modal auth-eval JSONs under `experiments/results/modal_auth_eval*`

Out of scope by operator instruction: code edits, raw experiment/result JSON
edits, raw claim edits, commits, and any remote/GPU dispatch.

## Preflight

Read/checked:

- `CLAUDE.md`
- `AGENTS.md`
- `PROGRAM.md`
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md`
  top entries
- `.omx/state/lane_registry.json`
- `.omx/state/active_lane_dispatch_claims.md`
- recent `.omx/research/*_directive_*` files dated 2026-05-14:
  all-design-decisions, harness-rigor, holistic-engineering, journal-grade,
  DP1 streamer, zen-floor council, per-trainer automatic wire-in, D1 NVDEC
  recovery anchor, and recovery no-signal-loss directive.

Repo state at audit:

| Element | Value |
|---|---|
| UTC audit time | `2026-05-16T00:52:57Z` |
| Branch | `main` |
| HEAD | `041ebec8807aff777b95258220f2e806b307d8ed` |
| Lane registry sha256 | `0d3e9459797a4f289e2c2e566c93bd831be4dee8db381fc81a51922c538c4078` |
| Active claims sha256 | `248a2d9ef09fb56b5ed4f1e0a940122330a2022ebc6c2c0ce84add56a7422321` |
| Modal harvest summary sha256 | `460c48bd5af6ddaac9afbf3bd812b3ee818c567da37add624f666977017a5e94` |
| Modal call-id ledger sha256 | `948dfc5f1dcc37ee13e5a415f26a36415656cb5af9bf1a0880b755f98e5796e3` |

Dirty tree note: `experiments/results/_modal_harvest_summary.json` was already
modified before this audit. During the audit, partner worktree state changed
from Time-Traveler harness files to Ballé BRV2-related files
(`src/tac/substrates/balle_renderer/*`). None of those files were touched.

## Findings

### F1 - D1 paired auth eval is terminal, but per-axis claims remain active

`tools/claim_lane_dispatch.py summary --ttl-hours 24` reports 8 active latest
jobs and 0 stale nonterminal rows. Two active rows are D1 per-axis Modal
auth-evals:

- `lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cuda` /
  `d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cuda`
- `lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cpu` /
  `d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cpu`

But the result JSONs already exist and the pair-level claim row says D1 paired
complete. The issue is closure granularity: the terminal row was written under
`lane_d1_paired_cpu_cuda_modal_dispatch_20260515` /
`d1_paired_modal_auth_20260515T194530Z`, not under the exact per-axis
`(lane_id, instance/job_id)` keys that the claim helper uses to close active
rows.

D1 CUDA JSON:

- path: `experiments/results/modal_auth_eval/d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cuda/modal_cuda_auth_eval_result.json`
- axis/evidence: `contest_cuda` / `contest-CUDA`
- score recomputed: `0.23128813459939868`
- components: `avg_segnet_dist=0.00066299`, `avg_posenet_dist=0.00017103`
- archive: `c4f40c055e5e1ba721dcc492f05008abd39e9220e6f3d93700ed436716f17c6e`, `185675` bytes
- `score_claim=true`, `promotion_eligible=false`, `adjudication_required=true`
- formula recomputation matched JSON exactly.

D1 CPU JSON:

- path: `experiments/results/modal_auth_eval_cpu/d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cpu/modal_cpu_auth_eval_result.json`
- axis/evidence: `contest_cpu` / `contest-CPU`
- score recomputed: `0.19778968855773263`
- components: `avg_segnet_dist=0.00056029`, `avg_posenet_dist=0.00003286`
- same archive SHA and byte count as CUDA
- `score_claim=false`, `promotion_eligible=false`, `adjudication_required=true`
- formula recomputation matched JSON exactly.

Next close-claim action, no re-eval needed:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cuda \
  --platform modal \
  --instance-job-id d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cuda \
  --agent codex:provider_custody_closeout \
  --status completed_contest_cuda_modal_auth_eval_recovered \
  --force \
  --notes "D1 per-axis closure only; result_json=experiments/results/modal_auth_eval/d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cuda/modal_cuda_auth_eval_result.json; archive_sha=c4f40c055e5e1ba721dcc492f05008abd39e9220e6f3d93700ed436716f17c6e; archive_bytes=185675; score_recomputed=0.23128813459939868; axis=contest_cuda; evidence_grade=contest-CUDA; hardware_substrate=linux_x86_64_t4; score_claim=true; promotion_eligible=false; adjudication_required=true"

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cpu \
  --platform modal \
  --instance-job-id d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cpu \
  --agent codex:provider_custody_closeout \
  --status completed_contest_cpu_modal_auth_eval_recovered \
  --force \
  --notes "D1 per-axis closure only; result_json=experiments/results/modal_auth_eval_cpu/d1_paired_smoke_archive_paired_modal_auth_20260515T194515Z_cpu/modal_cpu_auth_eval_result.json; archive_sha=c4f40c055e5e1ba721dcc492f05008abd39e9220e6f3d93700ed436716f17c6e; archive_bytes=185675; score_recomputed=0.19778968855773263; axis=contest_cpu; evidence_grade=contest-CPU; hardware_substrate=linux_x86_64_cpu; score_claim=false; promotion_eligible=false; adjudication_required=true"
```

### F2 - Active Modal rows need harvest/closure before TTL expiry

The claim helper's active latest rows are all Modal:

1. D1 per-axis CPU/CUDA auth evals, already terminal by result JSON but missing
   exact-key terminal rows (F1).
2. D4 Wyner-Ziv current smoke/full-paired chain:
   - `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C`
   - active rows under both registered `lane_d4_wyner_ziv_frame_0_substrate_20260514`
     and unregistered alias `lane_d4_wyner_ziv_paired_full_modal_dispatch_20260515`
   - ETA rows are already past at audit time.
3. Z3-G1 full CUDA sunk-cost dispatch:
   - `fc-01KRPMDRH2VNVKFFAYTWD4X0SD`
   - registry explicitly says `research_only=true`; if harvested, tag as
     `DIRECT_RESIDUAL_Z3HV2_REPRODUCTION`, not G1 evidence.
4. Z4 cooperative receiver loss current smoke:
   - `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`
5. Z4-V2 Wunderkind E1 pending smoke placeholder:
   - `z4_v2_wunderkind_e1_pending_smoke_20260515T194341Z`
   - status `eval`, no ETA.

Next recovery action:

```bash
.venv/bin/python tools/harvest_modal_calls.py --from-ledger --repo-root . --execute --get-timeout-seconds 60
```

Then re-run the read-only summary:

```bash
.venv/bin/python tools/claim_lane_dispatch.py summary \
  --claims-path .omx/state/active_lane_dispatch_claims.md \
  --ttl-hours 24
```

If the harvester closes only the registered D4 training lane, also close or
`stale_superseded_*` the unregistered paired alias rows with the same
`instance/job_id`, preserving the call_id and harvested result path in notes.

### F3 - Active lane IDs are not all registered

Comparing active claims to `.omx/state/lane_registry.json`:

- registered: `lane_d4_wyner_ziv_frame_0_substrate_20260514`
- registered: `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515`
- registered: `lane_z4_cooperative_receiver_loss_step2_20260514`
- registered: `lane_z4_v2_wunderkind_e1_tier1_modal_paired_20260515`
- missing from registry:
  `lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cpu`
- missing from registry:
  `lane_d1_paired_cpu_cuda_modal_dispatch_20260515_contest_cuda`
- missing from registry:
  `lane_d4_wyner_ziv_paired_full_modal_dispatch_20260515`

Risk: the claim helper accepts syntactically valid `lane_*` IDs, but registry
coherence and anti-duplication review cannot see these per-axis/pair aliases as
lanes. For future paired CPU/CUDA dispatches, either pre-register these as real
L0 lanes or use registered lane IDs plus an axis field in notes/result JSON.

No registry edit was made in this audit.

### F4 - Kaggle latent-sidecar rows are closed; the result is proxy-only and not dispatch-ready

Kaggle claim rows for `lane_pr106_latent_sidecar` are terminal:

- `kaggle_pr106_format0c_latent_score_table_repair_20260515T203529Z`
  failed with missing source bundle:
  `FileNotFoundError: required source bundle 'pact_pr106_latent_source_bundle.tar.gz' not found...`
- `kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z`
  failed at builder composition:
  `ValueError: cannot compose selected score-table correction for pair 0: base dim 26 and selected dim 11 would require two latent dimensions in a one-dim sidecar grammar`.

The second run still produced useful proxy signal:

- score-table shape `[600, 113]`
- candidate count `113`
- strict improvement pair count `570`
- best improvement mean `0.0014046559808775783`
- score table SHA `cbea7c7578a275044f162ff72b2fb502cac5ef4aa0b63eff7627da553bf8e935`
- manifest flags: `score_claim=false`, `ready_for_exact_eval_dispatch=false`,
  `promotion_eligible=false`, `proxy_authority_forced_false=true`

Next action is not another identical Kaggle dispatch. Choose one:

- implement a two-latent-dimension sidecar grammar for pairs where best dim
  differs from the base dim; or
- constrain the selector/materializer to one latent dimension per pair and
  preserve only compatible improvements.

Only after a byte-closed archive exists should it enter paired contest-axis
CUDA/CPU adjudication.

### F5 - Modal harvest summary and modal call-id ledger are out of sync

`experiments/results/_modal_harvest_summary.json` has 114 entries:

- 111 with `status="already_harvested"`
- 3 current entries with `status=null` but terminal claims embedded
- 72 entries with `terminal_claim.appended=true`
- 42 legacy harvested entries with no terminal-claim object/status in the
  summary

The read-only call-id ledger view reports 11 unharvested call IDs. At least four
of those are already present in `_modal_harvest_summary.json` with terminal
claim metadata:

- `fc-01KRNQVJWJH5AFV5W3TCGGTEFG`
- `fc-01KRNYZ6HN53HGHGC44XGAGJZ3`
- `fc-01KRNRMD29XM4M1Q467D8S1HXV`
- `fc-01KRNRTDNVNJHWRKGYYJV3S2XF`

Risk: `tools/harvest_modal_calls.py --from-ledger` may overstate unharvested
work when the append-only call-id ledger lacks a terminal event even though the
summary and active-claim table have terminal evidence. Treat `--from-ledger`
as a discovery queue, not as sole custody truth, until reconciled against
summary rows and claim rows.

Next action: run the canonical harvester once before TTL expiry and then
inspect whether it appends terminal call-id-ledger events for already
summarized rows. If not, add a separate reconciliation/backfill task so the
call-id ledger, `_modal_harvest_summary.json`, and active-claim latest-state
views converge.

### F6 - Score-axis labeling risks remain in active/predicted notes

Good examples:

- D1 pair row labels both `0.23128813 [contest-CUDA Modal T4 1to1]` and
  `0.19778969 [contest-CPU Modal CPU container 1to1]`.
- Modal auth-eval JSONs carry `score_axis` and `evidence_grade`.
- Kaggle manifests force `score_claim=false` and `promotion_eligible=false`.

Risk examples:

- `lane_z4_v2_wunderkind_e1_tier1_modal_paired_20260515` active note says
  `predicted dS [-0.020, -0.005] vs A1 0.1928` without an inline axis label.
- `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515` registry notes are
  explicit now, but its active full CUDA row still lacks ETA and axis-specific
  expected-use fields.

Next action: every terminal closeout note for active Modal rows should include
`axis=...`, `evidence_grade=...`, hardware substrate, result JSON path,
archive SHA/bytes, `score_claim=...`, `promotion_eligible=...`, and whether
the row is rank/kill eligible. Predictions should be labeled as
`[predicted; axis=...]` or kept non-scalar.

## Priority recovery order

1. Close the two D1 per-axis active claims using the existing result JSONs.
   This is bookkeeping only; do not re-run eval.
2. Harvest live Modal call IDs before cache TTL loss:
   `fc-01KRPJZ9FY7N1HJH6HMEK6TX6C`,
   `fc-01KRPJVEMQ5S7Q8EKGKQWKCS93`, and
   `fc-01KRPMDRH2VNVKFFAYTWD4X0SD`.
3. After D4 harvest, close both the registered D4 training lane row and the
   paired-dispatch alias row, or terminal-stale the alias if it is only a
   parent orchestration claim.
4. Resolve the Z4-V2 `eval` placeholder: attach it to a real call_id if one
   exists, otherwise terminal-stale it with notes.
5. Reconcile `modal_call_id_ledger.jsonl` terminal events against
   `_modal_harvest_summary.json` and active claims so `--from-ledger` does not
   keep already-harvested calls in the unharvested queue.
6. For Kaggle PR106 latent sidecar, do grammar/selector repair before any
   further dispatch. The latest signal is useful but proxy-only.

## Commands run

Read-only or audit commands used:

```bash
pwd && git rev-parse --abbrev-ref HEAD && git status --short
rg -n "Modal|Kaggle|provider|lane-claim|dispatch claim|active_lane|custody|exact.*eval|score-axis|axis" /Users/adpena/.codex/memories/MEMORY.md
find .omx/research -maxdepth 1 -type f -name '*_directive_*' -mtime -2 -print | sort
wc -l CLAUDE.md AGENTS.md PROGRAM.md .omx/state/lane_registry.json .omx/state/active_lane_dispatch_claims.md experiments/results/_modal_harvest_summary.json
rg --files -g 'MEMORY.md'
sed -n '1,220p' PROGRAM.md
sed -n '1,260p' CLAUDE.md
sed -n '1,260p' AGENTS.md
rg -n "Modal|Kaggle|provider|claim|dispatch|axis|custody|score" CLAUDE.md AGENTS.md
sed -n '1,220p' .omx/state/active_lane_dispatch_claims.md
jq 'type, (if type=="array" then length else keys end)' experiments/results/_modal_harvest_summary.json
jq 'type, (if type=="array" then length else keys end)' .omx/state/lane_registry.json
date -u '+%Y-%m-%dT%H:%M:%SZ'
jq -r '[.[].status] | group_by(.)[] | "\(.[0])\t\(length)"' experiments/results/_modal_harvest_summary.json
.venv/bin/python tools/claim_lane_dispatch.py summary --claims-path .omx/state/active_lane_dispatch_claims.md --ttl-hours 24
.venv/bin/python tools/claim_lane_dispatch.py summary --claims-path .omx/state/active_lane_dispatch_claims.md --ttl-hours 24 --format json
jq queries over D1 Modal auth-eval result JSONs
python3 formula recomputation for D1 CPU/CUDA result JSONs
find reports/raw/kaggle_ingested -maxdepth 2 -path '*kaggle_pr106_format0c_latent_score_table_repair*' -type f
jq queries over Kaggle ingest manifests and summaries
.venv/bin/python tools/harvest_modal_calls.py --from-ledger --repo-root . --get-timeout-seconds 1
git diff --stat -- experiments/results/_modal_harvest_summary.json
git status --short
git rev-parse HEAD
shasum -a 256 .omx/state/lane_registry.json .omx/state/active_lane_dispatch_claims.md experiments/results/_modal_harvest_summary.json .omx/state/modal_call_id_ledger.jsonl
```

No tests were run because this was a no-code, no-mutation custody audit.
