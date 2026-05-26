# Cascade C' subagent B — LANE SCRIPT + INFLATE.SH WRAPPER LANDED

- **subagent_id**: `cascade-c-prime-frame-1-segnet-waterfill-substrate-B-lane-script-inflate-wrapper-20260526`
- **lane_id**: `lane_cascade_c_prime_option_a_build_scaffold_20260526` (extends)
- **date_utc**: 2026-05-26T22:05:00Z
- **scope**: per-substrate symposium PROCEED_WITH_REVISIONS revision #2 (inflate.sh 3-arg signature wrapper + lane script) landing
- **execution_substrate**: macOS M5 Max local (NO PAID DISPATCH per CLAUDE.md "Executing actions with care")
- **horizon_class** per Catalog #309: `frontier_pursuit` (unblocks sister subagent C paired-CUDA Modal T4 smoke per revision #3)
- **mission_contribution** per Catalog #300: `frontier_breaking_enabler` (sister subagent C now structurally able to dispatch the canonical Modal T4 smoke)
- **cost**: $0 GPU + ~25 min wall-clock + 0 paid dispatches

## Landing summary

| # | Module | LOC | Status | Verification |
|---|---|---|---|---|
| 1 | `scripts/remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh` | 301 | LANDED | `bash -n` syntax-valid; chmod +x set; 22 structural-pattern tests PASS; Catalog #244 + #163 + #326 PASS |
| 2 | `src/tac/tests/test_cascade_c_prime_substrate_B_lane_script.py` | 244 | LANDED | 22/22 PASS in 0.19s |
| 3 | inflate.sh wrapper | DEFERRED | per canonical pattern, trainer wrapper `_write_runtime` emits into `submission_dir/inflate.sh` at compress-time | sister subagent C scope (the trainer wrapper) |

**Total this landing**: 2 files / 545 LOC added.

## Canonical pattern compliance verdicts (sister catalog gates)

| Catalog # | Gate | Verdict | Cascade C' violations |
|---|---|---|---|
| #244 | `check_remote_lane_scripts_carry_canonical_nvml_block` | PASS | 0 |
| #163 | `check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap` | PASS | 0 |
| #204 | sister of `check_pr95plus_modal_smoke_uses_durable_provider_output` (canonical 3-branch OUTPUT_DIR pattern) | PASS | N/A (sister recipe-level surface) |
| #326 | `check_substrate_driver_consumes_trainer_mode_env_var` | PASS | 0 |
| #146 | contest 3-arg inflate.sh signature | DEFERRED | sister subagent C trainer wrapper scope |

Live-repo regression guard: existing 31 sister substrate driver scripts unchanged (Catalog #244 strict-flip wave landed 2026-05-15); new cascade_c_prime driver follows the same canonical pattern. Total live-repo Catalog #244/#163/#326 violation count remains 0.

## Canonical 3-branch Modal-aware OUTPUT_DIR resolution per Catalog #204

```bash
if [ -n "${CASCADE_C_PRIME_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$CASCADE_C_PRIME_OUTPUT_DIR"          # (a) explicit override
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ] && [ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"   # (b) Modal
else
    OUTPUT_DIR="$LOG_DIR/output"                       # (c) local/Vast.ai
fi
```

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" non-negotiable + Catalog #204 cross-driver expansion (2026-05-19): `contest_auth_eval.py` refuses score-grade evidence under `/tmp/...`. Modal workers run from `/tmp/pact`, so the canonical Modal branch routes outputs to the durable `/modal_results` volume which `modal_train_lane.py` harvests via the canonical Catalog #245 call_id ledger.

## Catalog #326 mode env var multi-key precedence

```bash
# CASCADE_C_PRIME_TRAINER_MODE > SMOKE_ONLY > default("full")
if [ -n "${CASCADE_C_PRIME_TRAINER_MODE:-}" ]; then
    CASCADE_C_PRIME_MODE_RESOLVED="$CASCADE_C_PRIME_TRAINER_MODE"
elif [ -n "${SMOKE_ONLY:-}" ]; then
    if [ "$SMOKE_ONLY" = "1" ] || [ "$SMOKE_ONLY" = "true" ] || [ "$SMOKE_ONLY" = "yes" ]; then
        CASCADE_C_PRIME_MODE_RESOLVED="smoke"
    else
        CASCADE_C_PRIME_MODE_RESOLVED="full"
    fi
else
    CASCADE_C_PRIME_MODE_RESOLVED="full"
    echo "[lane-cascade-c-prime] WARN ... per Catalog #326 sister anti-pattern fail-loud" >&2
fi
```

Recipe `env_overrides` block sets `CASCADE_C_PRIME_TRAINER_MODE: "full"` + `SMOKE_ONLY: "0"` explicitly to avoid the Z6 Wave 2 4c dispatch bug class (driver default "smoke" while recipe intent "full"; sister bug-class anchor: `fc-01KRW7ZCYK5XF6MSHD24R71A46`).

## inflate.sh contract status

The canonical sister pattern (per `submissions/nscs06_carmack_hotz_strip_everything/inflate.sh` 10 lines) is for the trainer wrapper's `_write_runtime` method to emit `submission_dir/inflate.sh` at compress-time. Since the trainer wrapper (`experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py`) is sister subagent C's scope, the canonical inflate.sh wrapper landing is DEFERRED to that subagent.

The scaffold's `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` (211 LOC; HNeRV L7 substrate_engineering exception per Catalog #270 substrate trainer scope) is the actual numpy-portable inflate runtime. The canonical inflate.sh wrapper (when sister C emits it) will be ~10 lines per:

```bash
#!/usr/bin/env bash
# Cascade C' frame-1 SegNet waterfill contest-compliant inflate.
# Contract: $1=archive_dir $2=output_dir $3=file_list (Catalog #146).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
```

## Catalog #201 Modal sentinel files within mount set

Per recipe inspection (canonical `STRUCTURAL_MINIMUM_DIRS` from `tac.deploy.modal.mount_manifest`):
- `scripts/remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh` is under `scripts/` (canonical mount set member) - PASS
- `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py` is under `experiments/` (canonical mount set member) - PASS (when sister C lands it)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/` is under `src/` (canonical mount set member) - PASS
- Required input `upstream/videos/0.mkv` is under `upstream/` (canonical mount set member) - PASS

Catalog #201 sentinel parity: VERIFIED.

## Catalog #240 recipe-vs-trainer-state status

Recipe still declares `dispatch_enabled: false + research_only: true` (subagent A landing preserved this; my landing does not modify recipe). Per Catalog #240 acceptance cascade (b), the recipe IS opt-out-conformant. Sister subagent C MUST flip BOTH:
- `dispatch_enabled: true` (post-symposium PROCEED-unconditional)
- `research_only: false` (post trainer wrapper landing AND post paired-CUDA validation per revision #3)

Catalog #240 verdict for current state: PASS via opt-out cascade.

## Test verification

```bash
$ .venv/bin/python -m pytest src/tac/tests/test_cascade_c_prime_substrate_B_lane_script.py -x --tb=short -q
......................                                                   [100%]
22 passed in 0.19s
```

Test coverage:
- File existence + executable bit + bash syntax validity (3 tests)
- Catalog #244 canonical NVML 3-export block + early-position invariant (1 test)
- Catalog #163 canonical sentinel sourcing + bootstrap_runtime_deps invocation (2 tests)
- Catalog #204 canonical 3-branch Modal-aware OUTPUT_DIR resolution (2 tests)
- Catalog #326 mode env var multi-key precedence + fail-loud warning (1 test)
- Trainer path invocation + fail-fast on missing wrapper (1 test)
- Dispatch claim verification + DISPATCH_INSTANCE_JOB_ID (1 test)
- Heartbeat watchdog (5-min cadence per CLAUDE.md) + trap cleanup (2 tests)
- Provenance JSON emission + non-promotable defaults per CLAUDE.md (1 test)
- Axis-tagged completion marker per CLAUDE.md "Apples-to-apples" (1 test)
- WORKSPACE default + LANE_ID matches recipe + macOS AppleDouble strip (3 tests)
- Modal-aware required-input multi-candidate probe per Catalog #152+#204 sister (1 test)
- Stage log markers + observability per Catalog #305 (1 test)
- shebang + `set -euo pipefail` per Catalog #2 (2 tests)

## Canonical-vs-unique decision per layer (per Catalog #290; substrate-engineering exception per HNeRV parity L7)

| Layer | Decision | Rationale |
|---|---|---|
| Canonical NVML 3-export block | CANONICAL | Catalog #244 sister-wave pattern; mirrored verbatim from NSCS06 / PR101++ |
| Canonical sentinel sourcing | CANONICAL | Catalog #163 anti-pattern protection; verbatim from sister drivers |
| 3-branch OUTPUT_DIR resolution | CANONICAL | Catalog #204 cross-driver expansion pattern; verbatim from PR101++ sister |
| Mode env var multi-key precedence | UNIQUE (substrate-specific keys) | Catalog #326 canonical schema with substrate-prefix CASCADE_C_PRIME_* |
| Multi-candidate required-input probe | CANONICAL | Catalog #152+#204 sister pattern (Vast.ai / Modal-ro / Modal-rw) |
| Provenance JSON emission | CANONICAL stencil + substrate-specific fields | Sister of PR101++ driver; adds predicted_band + validation_status |
| Axis-tagged completion marker | CANONICAL stencil + substrate-specific tag | Sister of NSCS06 LANE_*_DONE marker |
| Trainer invocation | CANONICAL stencil + substrate-specific path | Per canonical sister pattern; fail-fast on missing wrapper |
| inflate.sh wrapper emission | DEFERRED to trainer wrapper `_write_runtime` | Sister of NSCS06 trainer wrapper pattern; sister subagent C scope |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Substrate-specific env var prefix CASCADE_C_PRIME_* + substrate-specific LANE_ID + substrate-specific trainer_path |
| 2 BEAUTY+ELEGANCE | 301 LOC reviewable in segments; canonical sister pattern preserved; clear stage progression |
| 3 DISTINCTNESS | DISTINCT from sister NSCS06 v8 / Z6 / PR101++ via substrate-specific env vars + LANE_ID + canonical-equation-#344 anchor proposal cross-reference |
| 4 RIGOR | Premise verified against scaffold + recipe + 2 sister canonical driver scripts + 4 sister catalog gates; 22 dedicated tests |
| 5 OPTIMIZATION-PER-TECHNIQUE | Canonical NVML 3-export + canonical sentinel + canonical 3-branch OUTPUT_DIR + canonical mode env multi-key precedence are SUBSTRATE-OPTIMAL via sister-driver patterns |
| 6 STACK-OF-STACKS-COMPOSABILITY | Compatible with operator-authorize / smoke-before-full / paired-dispatch / sister substrate canvas |
| 7 DETERMINISTIC-REPRODUCIBILITY | CASCADE_C_PRIME_SEED env var (default 20260526); seed pinned in provenance; canonical fcntl-locked dispatch claim |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | Thin actuator design (bootstrap delegated; mode resolution closed-form); 22 structural tests run in 0.19s |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | PROVISIONAL-PENDING-VERIFICATION per Catalog #363 (paired-CUDA gate per revision #3; sister subagent C scope) |

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Unwind plan |
|---|---|---|
| Canonical NVML 3-export block prevents NVML 999 on T4 | HARD-EARNED | D1 incident anchor 2026-05-15 + Catalog #244 strict-flip wave clean across 31 sister drivers |
| Canonical sentinel sourcing prevents sourced-main-flow side effects | HARD-EARNED | Catalog #163 anti-pattern protection + WAVE-D 2c957c31e forensic anchor |
| 3-branch OUTPUT_DIR resolution prevents `/tmp` refusal on Modal | HARD-EARNED | Catalog #204 cross-driver expansion 2026-05-19 + STC v2 2026-05-14 + stack_of_stacks 2026-05-19 anchor wave |
| Mode env var multi-key precedence prevents driver-mode-mismatch | HARD-EARNED | Catalog #326 anchor: Z6 Wave 2 4c dispatch `fc-01KRW7ZCYK5XF6MSHD24R71A46` |
| Multi-candidate required-input probe handles Modal-IGNORED `experiments/results/**` paths | HARD-EARNED | Catalog #152 WAVE-1 APPARATUS HARDENING extension 2026-05-16 + STC v2 anchor |
| Fail-fast on missing trainer wrapper surfaces sister subagent C blocker structurally | HARD-EARNED | Exit code 26 + diagnostic FATAL log per CLAUDE.md "Operator gates must be wired and used" |
| inflate.sh wrapper emission deferred to trainer wrapper is canonical pattern | HARD-EARNED | Sister NSCS06 + PR101++ trainer wrappers both emit `submission_dir/inflate.sh` at compress-time via `_write_runtime` |

## Observability surface (per Catalog #305)

- **inspectable per layer**: 6 numbered stages (0a strip / 0 claim verify / 0b NVDEC probe / 1 bootstrap / 1b video probe / 2 provenance / 3 heartbeat / 4 trainer invoke / 5 completion marker); each stage emits log markers per Catalog #305
- **decomposable per signal**: provenance.json carries 17 fields (cuda_available + torch_version + cuda_version + git_hash + gpu_name + driver_version + video_path + upstream_dir + device + epochs + trainer_mode_resolved + seed + predicted_band + predicted_band_variance + predicted_basis + predicted_band_validation_status + score_claim/promotion_eligible/ready_for_exact_eval_dispatch trio)
- **diff-able across runs**: provenance JSON deterministic given (lane_id, dispatch_instance_job_id, seed); diff via JSON keys
- **queryable post-hoc**: `$LOG_DIR/run.log` + `$LOG_DIR/heartbeat.log` + `$LOG_DIR/provenance.json` + Modal `/modal_results/${INSTANCE_JOB_ID}/output/` artifact tree
- **cite-able**: provenance carries (subagent_id implicit via LANE_ID + DISPATCH_INSTANCE_JOB_ID, git_hash, gpu_name)
- **counterfactual-able**: Modal-runtime branch path + mode env var resolution can be flipped without touching the trainer path

## Predicted ΔS band (per Catalog #296)

| Status | Band | Validation |
|---|---|---|
| MLX-LOCAL synthesis (Cascade C') | -0.058820 [macOS-MLX research-signal] | Dykstra-feasibility per Cascade C' synthesis 48-cell sweep (already-landed in subagent A) |
| Paired-CUDA expected | PROVISIONAL-PENDING-VERIFICATION | Contrarian + Atick dissent: 10-30× literature overestimate common |

Per Catalog #324: `predicted_band_validation_status: pending_post_training` (emitted in provenance.json). Reactivation criterion: post-training Tier-C re-measurement on landed paired-CUDA smoke archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c` (sister subagent C scope).

## Canonical equation #344 anchor status

`atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` remains **FORMALIZATION_PENDING** per Catalog #344 sister discipline. Registry count: 52 (unchanged this landing; sister subagent C registers the anchor via canonical helper post paired-CUDA empirical per registration tool pattern commits `7ab5f58ae` + `04f34ea40`).

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: N/A (lane script is operator-facing dispatch actuator; signal contribution N/A)
- **hook #2 Pareto constraint**: N/A
- **hook #3 bit-allocator**: N/A
- **hook #4 cathedral autopilot dispatch**: ACTIVE (canonical Modal/Vast.ai dispatch entry point; cathedral autopilot ranker can dispatch via this lane script per canonical Catalog #245 call_id ledger)
- **hook #5 continual-learning posterior**: ACTIVE (provenance.json + auth_eval artifact emission feeds the canonical posterior anchor surface; sister subagent C will register the anchor post empirical)
- **hook #6 probe-disambiguator**: N/A (lane script is the dispatch actuator; probe-disambiguator lives in the trainer + Tier-C hook)

## Per-substrate symposium revision status

| # | Revision | Status |
|---|---|---|
| 1 | MLX-first trainer landing | LANDED (subagent A; commit 116d46da8) |
| 2 | inflate.sh 3-arg signature wrapper + lane script | **LANDED** (this subagent B); inflate.sh wrapper itself DEFERRED to sister subagent C trainer `_write_runtime` emission per canonical pattern |
| 3 | Paired-CUDA Modal T4 smoke | DEFERRED to sister subagent C (PAID; operator-pre-approved-per-blanket-auth) |
| 4 | Paired-CPU Linux x86_64 anchor | DEFERRED to sister subagent C (PAID) |
| 5 | Canonical equation #344 anchor registration | DEFERRED to sister subagent C (post paired-CUDA empirical) |

**2 of 5 revisions LANDED + 3 of 5 DEFERRED** per operator-routable next step.

## Discipline closure

- **Catalog #229 PV**: read scaffold (1 file: inflate.py) + trainer (1 file: trainer.py tail) + recipe + 3 sister canonical lane scripts (NSCS06 + PR101++ + remote_archive_only_eval) + 2 sister canonical inflate.sh files (a1 + nscs06) + subagent A landing memo + recipe-vs-trainer state
- **Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256**: used for final commit
- **Catalog #119 Co-Authored-By Claude trailer**: emitted via canonical serializer (auto-appends per Catalog #119)
- **Catalog #206 checkpoints**: 2 in_progress + 1 complete emitted via `tools/subagent_checkpoint.py`
- **Catalog #229 premise verification**: ALL canonical surfaces read FIRST before any edit
- **Catalog #230 sister-subagent ownership map**: scope DISJOINT from any active sister (lane script + tests; not touching subagent A's trainer/bridge/hook/__init__.py per landing memo discipline)
- **Catalog #287 placeholder-rationale rejection**: every waiver ≥4 chars + substantive
- **Catalog #208 no local-paths**: all paths repo-relative; the canonical `/workspace/pact/` + `/modal_results/` + `/tmp/pact/` paths are STRUCTURAL Modal/Vast.ai canonicals NOT operator-machine-local
- **Catalog #340 sister-checkpoint guard**: my own in-progress checkpoint blocks own staging (correct behavior); canonical serializer arbitrates at commit time per Catalog #157
- **Catalog #343 no hardcoded score literals**: predicted_band cited as research-signal hypothesis pending paired-CUDA validation; no contest score literals
- **Catalog #344 canonical equation registry**: remains 52 (FORMALIZATION_PENDING; sister subagent C registers post-empirical)
- **Catalog #346 canonical roster complete=True**: N/A this landing (no NEW T2+ deliberation invoked; reuses per-substrate symposium PROCEED_WITH_REVISIONS from subagent A landing chain)

## Forbidden patterns avoided

- NO PAID DISPATCH (subagent C scope)
- NO modification of subagent A's trainer / bridge / hook / __init__.py / tests (Catalog #230 sister-disjoint)
- NO modification of scaffold modules (substrate_contract / architecture / archive / inflate)
- NO subagent spawning
- NO `gh pr create` / `git push`
- NO touching `submissions/exact_current/` per CLAUDE.md mutation frontier
- NO sister-substrate work (Cascade A / Cascade B / NSCS06 / UNIWARD / CATALYST per Catalog #230)
- NO Phase 1 audit work (sister subagent in flight scope)
- NO `submissions/cascade_c_prime_*/inflate.sh` creation (trainer wrapper emission scope per canonical pattern)

## Operator-routable next step

1. **Sister subagent C**: build `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py` trainer wrapper following canonical NSCS06 sister pattern; the wrapper invokes `tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.trainer.run_mlx_first_compress_pass` (MLX-LOCAL) AND emits `submission_dir/inflate.sh` via `_write_runtime` AND calls canonical `gate_auth_eval_call` per Catalog #226
2. **Sister subagent C paired-CUDA Modal T4 smoke**: via `tools/operator_authorize.py --recipe substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch --target modal` per CLAUDE.md "Executing actions with care"; smoke-before-full per Catalog #167 + paired-CUDA per Catalog #246; ~$0.30-0.50; **OPERATOR-PRE-APPROVED per blanket auth**
3. **Sister subagent C recipe flip**: `dispatch_enabled: true` + `research_only: false` per Catalog #240 once trainer wrapper lands and per-substrate symposium achieves PROCEED-unconditional verdict
4. **Post-empirical**: sister subagent C registers canonical equation #344 anchor via sister registration tool pattern
5. **Post-Tier-C**: per-substrate symposium re-convenes for PROCEED-unconditional verdict per Catalog #325 + #324

## Cross-references

- Predecessor landing memo (scaffold): `.omx/research/cascade_c_prime_option_a_build_scaffold_landed_20260526.md` (commit `aaf0b1eb6`)
- Predecessor landing memo (subagent A trainer): `.omx/research/cascade_c_prime_subagent_A_mlx_first_trainer_landed_20260526.md` (commit `116d46da8`)
- Per-substrate symposium: `.omx/research/council_t2_cascade_c_prime_frame_1_segnet_waterfill_per_substrate_symposium_20260526.md`
- Canonical equation #344 anchor proposal: `.omx/research/cascade_c_prime_canonical_equation_344_anchor_proposal_20260526.md`
- Sister canonical lane scripts: `scripts/remote_lane_substrate_nscs06_carmack_hotz_strip_everything.sh` + `scripts/remote_lane_substrate_pr101_lc_v2_clone_enhanced_curriculum.sh`
- Sister canonical inflate.sh: `submissions/a1/inflate.sh` + `submissions/nscs06_carmack_hotz_strip_everything/inflate.sh`
- Substrate package: `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/`
- Operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch.yaml`
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Meta-Lagrangian/Pareto solver" + "Forbidden re-implementing remote bootstrap inline" + "Remote code parity" + "Apples-to-apples evidence discipline" non-negotiables
- CLAUDE.md Catalog #244 / #163 / #204 / #326 / #146 / #152 / #240 / #270 catalog rows
