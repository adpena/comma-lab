# ddm_mx2b Row-2 Adapter Prep Receipt

Date: 2026-08-06
Axis: `[scorer-free training prep]`
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## Verdict

COMPLETED for the tq1c Row-2 adapter-prep gate. The PR130 pose trainer can now
consume the required target cache and tq1c master-surface cache, and the sibling
wrapper has a true full-state resume sidecar. This did not run a full n600
scorer/evaluate job and did not move the pointer.

## Derived Schemas

From `src/tac/pr130_lift/pose/lifted/train_pose_carrier_full.py`:

| cache | required keys | required shape / guard |
|---|---|---|
| target cache | `seg`, `pose` | `seg.shape[0] == 600`, `pose.shape == (600, 6)` |
| master cache | `source_checkpoint`, `masters` | `source_checkpoint == str(--master-checkpoint.resolve())`, `masters.shape == (600, 3, 874, 1164)`, `masters.dtype == torch.uint8` |

The tq1c master cache is an adapter cache: the frames come from the tq1c parent
inflated raw, while `source_checkpoint` is the PR130 trainer's cache-identity
guard.

## Built Artifacts

| artifact | bytes | sha256 | notes |
|---|---:|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt` | 117981301 | `0eae6dab35331bfacebd787548b901553bdcf373abe3d88371a723989fb65d68` | Decompressed from PR130 official Ada cache; `seg` uint8 `(600,384,512)`, `pose` fp32 `(600,6)` |
| `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt` | 1831206837 | `3a9792136823046eb89d3b7d808d07e5a1186cbef6ec78f58d260a5472b709b4` | Extracted frame 1 from each tq1c parent RGB pair into PR130 master-cache layout |
| `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/cache_validation.receipt.json` | 864 | `73018684c0a686fb920178bc765de838fca79d64a659c09049063bd091c85b6a` | Cache validation PASS |
| `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/launch_ticket.json` | 4111 | `ea0435476ee963cc527fecc80a4468ecb5b26d420409cc2f84dde92e80d7a5bf` | SSD mirror of `.omx/research/ddm_mx2b_20260806/LAUNCH_TICKET.json` |
| `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/canonical_equations_mx2b_20260806.json` | 3127049 | `73452be589e6d8429e427418b8ca34801758d417fd792b768a9fca43b845eaac` | Recall snapshot, 424 entries |

Source receipts:

| source | bytes | sha256 |
|---|---:|---|
| PR130 `gt_cache_600_official_ada.pt.xz` | 526820 | `233884c672eff22258376cf9532bb69a52017980000a2615bbd917ba7a8ec3dc` |
| PR130 official cache uncompressed payload | 117981301 | `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195` |
| tq1c parent inflated raw `0.raw` | 3662409600 | `82de098f5b97e6c61c7a53b4180f425117ea2e3c89e6ab435e7aea423f81291a` |
| PR130 selected semantic checkpoint | 282352 | `3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647` |

## Resume Wrapper

New wrapper: `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py`.

Borrowed substrate accounting: PR130 owns the pose carrier model, losses,
quantization, optimizer semantics, cache schema, and recipe defaults. This arm
adds only wrapper-owned full-state sidecar checkpointing plus an explicit
`--smoke-pairs` validation scope reduction. The vendored trainer file was not
edited.

Full-state sidecar contains: current step, args snapshot, basis, all coefficient
rows, both optimizer states, both scheduler states, generator state, sampling
order, cursor, hard-mining weights, history, best basis/coeff, and active pair ids.

## Smoke

Command shape: CPU, `--smoke-pairs 4`, `--steps 2`, first run stopped after
step 1, second run resumed from `carrier.step000001.full_state.pt` to step 2.
This is a scope-reduced smoke only, not a family verdict.

| item | result |
|---|---|
| cache acceptance | PASS: target cache and master cache loaded through trainer path |
| full-state load | PASS: resume emitted `full_state_loaded`, `resume_from_step=1`, `start_step=2` |
| full-state save | PASS: wrote step 1 and step 2 sidecars |
| one resumed step | PASS: quantized smoke mean changed `83.71189880371094 -> 83.69911193847656` |
| result JSON | `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/smoke_tq1c_pose_resumable/resumed_step2.json`, sha256 `c902aa4a9b3c9dcf65048c71e422b9be9c1d274a4da0259da11ea5ac873a3026` |

Full-state sidecars:

| artifact | sha256 |
|---|---|
| `carrier.step000001.full_state.pt` | `bf52c08bfcd707f6fcc516cddf9f1299384322df6435eddbd87aecfd007ac2a0` |
| `carrier.step000002.full_state.pt` | `a88fda4d3c65eecef650ea5e09de9748730ec5b6275c1ad043b5984b6e17b473` |
| `carrier.full_state.latest.pt` | `a91bd4217f0614846a17c292402bc2d2d815c3b363a062e7c2c04ac330c422dd` |

## Launch Ticket

`LAUNCH_TICKET.json` carries:

- Row-2 tq1c-arm command: fireable after lane claim and ET4/scorer-slot boundary.
- Row-2 mx1-arm placeholder: waits for Row-1 output path and a selected MX1
  renderer surface.
- Resume addition: append `--resume-state .../carrier_int6_coefftail_tq1c.full_state.latest.pt`.

The tq1c command preserves the PR130 pose hard-mining scheduler horizon
(`--steps 4000`) and selected boundary (`--stop-after-step 750`) and adds
`--state-save-every 250`.

## RECALL EVIDENCE

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Memory registry | `rg -n "mx2b|codex_runs|common_contract|main_hot_state" /Users/adpena/.codex/memories/MEMORY.md` | No mx2b-specific memory hit; current relevant memory emphasized live hot-state authority and source-verified queue ownership. | Used live board and RR6 artifacts rather than treating the common-contract frontier line as current. |
| Governing files | `mx2b_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | ET4 owns the scorer slot; common-contract live frontier was stale versus main hot state. | Stayed scorer-free except n<=4 CPU smoke and preserved pointer-unmoved language. |
| RR6 receipt | `.omx/research/ddm_rr6_20260806/ROUND6_FINDINGS.md` | Confirmed exact missing adapter outputs and the trainer cache guard. | Built the named target/master caches first and did not launch the long pose fit. |
| PR130 recipe | `scripts/train.sh`, `recipe/TRAINING.md`, `recipe/artifacts.json`, `build_gt_cache_official.py` | Official target cache builder requires CUDA/DALI; retained boundary table says pose hard-mining uses a 4000-step horizon with selected step 750. | Used the PR130 official Ada cache as source provenance and set launch ticket to `--steps 4000 --stop-after-step 750`. |
| Trainer source | `rg`/`sed` over `train_pose_carrier_full.py` | Source trainer saves latest/best deployable weights but no optimizer/scheduler/RNG state. | Added sibling full-state wrapper; left vendored trainer unchanged. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` | 424 entries; no PR130-lift equation superseded authority/axis separation. | Kept this as scorer-free prep, not a score row. |

## Verification

| command | result |
|---|---|
| `.venv/bin/python -m py_compile tools/build_mx2_pose_adapter_caches.py src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py ...` | PASS |
| `.venv/bin/python -m pytest src/tac/pr130_lift/tests/test_mx2_pose_adapter_caches.py src/tac/pr130_lift/tests/test_mx2_pose_resumable_state.py -q` | PASS, 2 tests |
| `tools/build_mx2_pose_adapter_caches.py all` | PASS; wrote target/master caches and validation receipt |
| `train_pose_carrier_full_resumable --smoke-pairs 4` step 1 then resume step 2 | PASS; cache accept, state save/load, resumed step |
| `pgrep` / `ps` process check | BLOCKED by sandbox: `sysmond service not found` / `operation not permitted`; no full scorer/evaluate job was launched |
| `git diff --cached --name-status` before receipt | empty; staged index untouched |

## Follow-On Disposition

| follow-on | disposition |
|---|---|
| Row-2 tq1c pose arm | QUEUED-WITH-A-FIRE-ORDER in `LAUNCH_TICKET.json`; claim lane before dispatch and use resume sidecar on interruption |
| Row-2 mx1 pose arm | QUEUED-WITH-A-FIRE-ORDER after Row-1 output path exists; build a matching master cache first |
| Exact score / pointer movement | NOT RUN; no archive, no `upstream/evaluate.py`, no contest-CPU/CUDA claim |

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
Contest pointer unchanged: `S = 0.1910828242` borrowed.
