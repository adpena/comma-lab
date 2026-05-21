# OVERNIGHT-BBB: NSCS06 v8 Phase 4 retry rc=1 DEEPER DIAGNOSIS landed 2026-05-21

**Lane**: `lane_overnight_bbb_nscs06_v8_rc1_deeper_diagnosis_post_rr_fix_20260521`
**Sister-coherence**: AAA T4 symposium (slot 1 memo `t4_grand_council_symposium_*`) + XX cron Selfcomp 17:00 CDT — DISJOINT substrate scope verified.
**Classification per Catalog #307**: IMPLEMENTATION-LEVEL falsification; PARADIGM-INTACT (NSCS06 v8 chroma-LUT substrate paradigm valid; recipe-driver-state divergence is the implementation bug).

## Empirical anchor

Modal dispatch `fc-01KS5XN8WF9JF15KVX3GPCFAE7` (call_id from OVERNIGHT-VV cron 5e07de6e verdict MEDIUM) at 2026-05-21T18:44:08Z returned `rc=1 elapsed=6.913s artifacts=7` — DIFFERENT from prior OVERNIGHT-RR diagnosis at `fc-01KS5QRXWNVYC54E2Y9Z8KZ4W2 rc=22 1.7-2.1s/5 artifacts`. The longer elapsed + more artifacts confirmed RR's driver Stage-0 mode-validation fix WORKED; failure moved deeper into trainer Stage 1 device-or-die gate.

## Phase 1: Harvest 7 artifacts via Modal API

Harvested to `experiments/results/nscs06_v8_rc1_diagnosis_artifacts_20260521/`:

- `modal_lane_substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T184408Z.log` (792B)
- `modal_worker_head_ledger.json` (1676B; sentinel sha matches local; no source drift)
- `modal_live_metadata.json` (443B)
- `lane_nscs06_v8_chroma_lut_results/run.log` (509B)
- `lane_nscs06_v8_chroma_lut_results/trainer.log` (283B)
- `lane_nscs06_v8_chroma_lut_results/provenance.json` (716B; `trainer_rc: 1`, `mode: full`, `device: cpu`)
- `results/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T184408Z/modal_worker_head_ledger.json` (1676B)
- `_top_level_metadata.json` (carries stdout_tail with the verbatim trainer error)

## Phase 2: Stderr verbatim — ROOT CAUSE OBVIOUS

trainer.log (verbatim):

```
[nscs06_v8_chroma_lut] --device cpu is permitted only with --smoke per CLAUDE.md
'MPS auth eval is NOISE' + 'EMA — non-negotiable' + full-training-needs-CUDA
convention. Use --device cuda for promotion-grade training. CPU smoke is
allowed only when deterministic-bytes acceptable.
```

modal_lane log (verbatim):

```
[lane-nscs06-v8-chroma-lut] 2026-05-21T18:46:35Z NSCS06_V8_TRAINER_MODE=full accepted
[lane-nscs06-v8-chroma-lut] 2026-05-21T18:46:35Z PYBIN=/usr/local/bin/python
[lane-nscs06-v8-chroma-lut] 2026-05-21T18:46:35Z sourcing canonical bootstrap (Catalog #163 sentinel honored)
[archive-only-eval] 2026-05-21T18:46:35Z running v8 chroma-LUT trainer mode=full smoke_flag=''
[nscs06_v8_chroma_lut] --device cpu is permitted only with --smoke ...
[archive-only-eval] 2026-05-21T18:46:40Z trainer exit code: 1
[archive-only-eval] 2026-05-21T18:46:40Z LANE_NSCS06_V8_CHROMA_LUT_FAILED rc=1
```

**Failure point**: `experiments/train_substrate_nscs06_v8_chroma_lut.py:582` `_device_or_die(args.device, smoke=False)` in `_full_main` — canonical `tac.substrates._shared.trainer_skeleton.device_or_die_canonical` REFUSED `device=cpu + smoke=False` per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.

## Phase 3: Sister-comparator diff (DP1 + 6 sister drivers vs NSCS06 v8)

Searched all `scripts/remote_lane_substrate_*.sh` for `_DEVICE.*:-` pattern:

| Driver | Default | Diff vs NSCS06 v8 |
|---|---|---|
| `atw_codec_v2.sh:59` | `cuda` | OK |
| `balle_renderer.sh:64` | `cuda` | OK |
| `d1_segnet_margin_polytope.sh:61` | `cuda` | OK |
| `pretrained_driving_prior.sh:65` (DP1) | `cuda` | OK |
| `z3_balle_hyperprior_bolton.sh:63` | `cuda` | OK |
| `z3_g1_scorer_softmax_hyperprior_gating.sh:62` | `cuda` | OK |
| `z4_cooperative_receiver_loss.sh:71` | `cuda` | OK |
| `z5_predictive_coding_world_model.sh:72` | `cuda` | OK |
| **`nscs06_v8_chroma_lut.sh:65`** | **`cpu`** | **OUTLIER** |

7 sister substrate drivers ALL default `_DEVICE` to `cuda`. NSCS06 v8 was the **only outlier** — a stale carryover from L0 SCAFFOLD --smoke-only era. OVERNIGHT-V Phase 2 BUILD landing 2026-05-21 (commit `29f92af8d`) atomically flipped `NSCS06_V8_TRAINER_MODE` smoke→full + `SMOKE_ONLY` 1→0 but **MISSED** the atomic device flip cpu→cuda. The trainer-canonical gate then enforced its constraint when the dispatch fired.

DP1 succeeded at training arms specifically BECAUSE DPP_DEVICE defaulted to `cuda` AND DPP recipe matched — no contradiction triggered the trainer gate.

## Phase 4: Fix landed (atomic 2-file edit per Catalog #240 sister + Catalog #326 sister)

**Driver fix** at `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh:51-65`:
- Replaced `NSCS06_V8_DEVICE="${NSCS06_V8_DEVICE:-cpu}"` with `NSCS06_V8_DEVICE="${NSCS06_V8_DEVICE:-cuda}"`
- Added 11-line citation block referencing this landing memo + Catalog #240 + Catalog #326 + sister-comparator audit + verbatim trainer error
- Preserved `${NSCS06_V8_DEVICE:-...}` ladder so smoke-mode operators can override via env per Catalog #151

**Recipe fix** at `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml:125-139`:
- Added `NSCS06_V8_DEVICE: "cuda"` to `env_overrides` block (explicit declaration sister-pattern per Catalog #326 driver-mode-env-var-explicit-declaration discipline)
- Updated comment header citing OVERNIGHT-BBB atomic followup

Both validate clean:
```
DRIVER SYNTAX OK (bash -n)
RECIPE YAML OK (env_overrides: {NSCS06_V8_TRAINER_MODE: full, NSCS06_V8_DEVICE: cuda, NSCS06_V8_EPOCHS: 1, SMOKE_ONLY: 0})
```

## Phase 5: STRICT preflight gate — DEFERRED per Catalog #299 quota brake

The bug class is **already covered structurally** by existing gates:
- **Catalog #240** `check_substrate_contest_cuda_chain_complete_or_research_only_tagged` — recipe-vs-trainer-state consistency (the parent META class this incident sits inside)
- **Catalog #326** `check_substrate_driver_consumes_trainer_mode_env_var` — driver-mode env var discipline; the device-mode parallel is conceptually identical
- **Catalog #270** `check_dispatch_optimization_protocol_complete` — umbrella protocol Tier 2/3 hardware correctness

Per CLAUDE.md "Gate consolidation discipline" + Catalog #299 quota brake principle + Catalog #287 META principle "every new STRICT gate should subsume >=3 sister cases": a new Catalog #345-class gate for "driver _DEVICE default consistent with recipe trainer_mode" would be a single-instance gate addressing 1 outlier across 8 sister drivers — not enough sister cases to justify new gate addition. The atomic fix + memo + sister-driver inventory documenting the pattern serves the same structural purpose without growing the catalog quota.

**Operator-routable** (if recurrence): if a 2nd substrate driver lands with stale `_DEVICE=cpu` default + full mode, escalate to Catalog #345 gate authoring (the 3rd sister case would clear the META-principle bar).

## Sister-coherence verification

- **Slot 1 AAA T4 symposium** (`a8b02679`): writes research memos + council anchor APPEND-ONLY per Catalog #110/#113. DISJOINT from NSCS06 v8 substrate scope. AAA memo at `.omx/research/t4_grand_council_symposium_distortion_axis_cascade_post_dp1_verdict_d_landed_20260521.md` is untracked at this commit; I touch ZERO files in AAA's scope.
- **Cron 9efd7486 Selfcomp XX harvest** at 17:00 CDT: DISJOINT substrate.
- **Catalog #340 sister-checkpoint guard**: my checkpoints declare only `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh` + `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`; sister Slot 1 declares research memos. No file overlap. PROCEED.

## Carmack MVP-first 5-step compliance

1. **FREE local macOS-CPU smoke first** — N/A; this is post-hoc diagnosis of a paid failure, not a new dispatch
2. **Falsifiable challenge to cargo-cult** — sister-comparator diff (7 sister drivers cuda vs 1 outlier cpu) IS the falsifiable assumption-classification test
3. **Canonical equation anchor + Catalog #344 reference** — N/A (this is infrastructure fix not a substrate score claim); the substrate's canonical equation #26 routing remains intact + `predicted_delta` field unchanged
4. **Verdict in same commit batch as fix** — fix + this landing memo in same commit batch per CLAUDE.md "Sister-supersession respect" non-negotiable
5. **Re-route operator priority queue within ~1h** — operator-routable below

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: N/A (driver/recipe infrastructure fix; no signal contribution to sensitivity map)
- **Hook #2 Pareto constraint**: N/A
- **Hook #3 bit-allocator**: N/A
- **Hook #4 cathedral autopilot dispatch**: N/A (no autopilot ranker mutation; the fix enables future NSCS06 v8 Phase 4 retries to actually fire successfully, but the recipe's `predicted_delta=-0.002706 [prediction]` remains unchanged)
- **Hook #5 continual-learning posterior**: N/A (no empirical anchor produced; landing is implementation fix not score claim)
- **Hook #6 probe-disambiguator**: N/A (no probe outcomes; diagnostic-only landing)

## Discipline checklist

- ☑ Catalog #229 premise verification (read 6 files + verified sister-driver inventory via grep + verified empirical artifact content before claiming root cause)
- ☑ Catalog #117 / #157 / #174 canonical serializer (commit via tools/subagent_commit_serializer.py with POST-EDIT --expected-content-sha256)
- ☑ Catalog #206 checkpoint discipline (4 checkpoints emitted; lane_id pre-registered; final step=complete planned)
- ☑ Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (zero mutation of RR/VV memos or sister Slot 1 AAA memo)
- ☑ Catalog #230 bulk-rewrite ownership map (no bulk rewrite; 2-file targeted edit; ownership disjoint from sister Slot 1)
- ☑ Catalog #340 sister-checkpoint guard (file scope disjoint; PROCEED)
- ☑ Catalog #287 placeholder-rationale rejection (all comments substantive ≥4 chars)
- ☑ Catalog #307 paradigm-vs-implementation classification (IMPLEMENTATION-LEVEL; PARADIGM-INTACT documented above)
- ☑ Catalog #299 quota brake (defer STRICT gate; bug class already structurally covered by #240 + #326 + #270)
- ☑ CLAUDE.md "Forbidden premature KILL" (substrate paradigm preserved; recipe `predicted_delta` + `predicted_band` retained; lane remains dispatch-eligible)

## Operator-routable: NSCS06 v8 Phase 4 retry post-fix

After this commit lands, the canonical retry path is:

```
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch \
    --confirm
```

Expected behavior with fix:
- Driver Stage 0 mode validation: PASS (NSCS06_V8_TRAINER_MODE=full accepted per OVERNIGHT-RR fix)
- Driver Stage 3 trainer invocation: passes `--device cuda` (per OVERNIGHT-BBB fix)
- Trainer `_full_main` Stage 1 `_device_or_die(device=cuda, smoke=False)`: PASS (cuda + full is the canonical promotion-grade combination)
- Trainer Stage 2+: real Phase 2 BUILD execution per OVERNIGHT-V 10-stage decomposition

Iteration pattern is sound: OVERNIGHT-RR (driver mode-validation fix at Stage 0) → OVERNIGHT-BBB (driver device-default fix at Stage 1/trainer-gate) → next iteration would diagnose any post-Stage-1 failure.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (not in active race mode) + "Carmack MVP-first phasing" (this is FREE post-hoc CPU-only diagnosis at $0; next paid dispatch is operator-gated): operator decides when to fire the next NSCS06 v8 Phase 4 retry.

## Cost summary

- $0 paid GPU (post-hoc diagnosis only; Modal artifact harvest is metered against existing dispatch cost)
- ~75 min wall-clock (10 min harvest + 20 min sister-comparator + 5 min fix + 30 min memo + 10 min commit)
- 2 files edited; 0 new STRICT gates; 0 new tests (Catalog #240 + #326 + #270 existing coverage sufficient)
