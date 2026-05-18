---
schema: codex_routing_directive_v1
directive_id: codex_routing_directive_rate_attack_vector_1_f1_hydra_dims_7_12_20260518
target_subagent: codex_019de465
routing_date: "2026-05-18"
parent_design_memo: rate_attack_vector_1_f1_hydra_dims_7_12_design_memo_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: F1
priority: TOP-1
council_verdict: PROCEED_WITH_REVISIONS_PROBE_REQUIRED
binding_revisions:
  - "Contrarian: F1 PROCEED conditional on probe `tools/probe_hydra_dim_7_12_score_invariance.py` PASSING across 600 pairs CPU+CUDA"
  - "Fridrich: PoseNet adversarial-blind-spot assumption MUST be PROBED before substrate dispatch"
operator_approved_gpu_budget_usd: 5.0
operator_approved_gpu_budget_phase_breakdown:
  phase_1_probe_local: 0.00
  phase_1_probe_modal_smoke: 0.30
  phase_2_substrate_build_modal_smoke: 3.00
  phase_3_full_run_modal_a100: 5.00
write_scope_for_codex: |
  src/tac/substrates/rate_attack_f1_hydra_dims_7_12/  (NEW package)
  experiments/train_substrate_rate_attack_f1_hydra_dims_7_12.py  (NEW trainer)
  submissions/rate_attack_f1/  (NEW submission)
  tools/probe_hydra_dim_7_12_score_invariance.py  (NEW probe tool)
  scripts/remote_lane_substrate_rate_attack_f1_hydra_dims_7_12.sh  (NEW driver)
  .omx/operator_authorize_recipes/substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml  (NEW recipe)
  src/tac/tests/test_rate_attack_f1_hydra_dims_7_12.py  (NEW tests)
write_scope_excludes:
  - "Anything in PRIMARY research subagent scope (.omx/research/* files this wave landed)"
  - "Anything in ADVERSARIAL sister subagent scope (.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md if exists)"
---

# Codex Routing Directive — Rate-Attack Vector 1: F1 Hydra Dims 7-12 Substrate

**Target Codex subagent**: `019de465` (the src/tac/*.py source code owner per CLAUDE.md "Subagent coherence-by-default" disjoint-scope rule)

**Operator directive context**: TOP-PRIORITY rate-attack wave per operator 2026-05-18 message "all of the rate attacks sound amazing and that subagent and its work should take the top priority". F1 is TOP-1 per master memo §0 TOP-5 selection — highest EV (predicted ΔS / engineering LOC ratio).

**META-paradigm**: STRUCTURAL INFORMATION NOT SHIPPED (SINS). Hydra dims 7-12 are STRUCTURALLY SCORE-INVARIANT per `upstream/modules.py:84` first-6-dims slice. Encode bits in dims 7-12; ship as side channel for FREE.

## 0. PRE-FLIGHT (MANDATORY per CLAUDE.md "Subagent coherence-by-default" + Catalog #206)

1. Read `/Users/adpena/Projects/pact/CLAUDE.md` (FULL); honor every NON-NEGOTIABLE marker
2. Read `/Users/adpena/Projects/pact/AGENTS.md`
3. Read `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` (top 50)
4. Read this routing directive in full
5. Read parent design memo: `.omx/research/rate_attack_vector_1_f1_hydra_dims_7_12_design_memo_20260518.md`
6. Read parent master memo: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
7. Read META-paradigm memo: `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
8. Read source anchors:
   - `upstream/modules.py:26` (HEADS = [Head('pose', 32, 12)])
   - `upstream/modules.py:84` (compute_distortion `[..., : h.out // 2]`)
   - `upstream/modules.py:67-79` (Hydra + PoseNet forward path)
9. Read sister CLAUDE.md non-negotiables: "HNeRV / leaderboard-implementation parity discipline" / "UNIQUE-AND-COMPLETE-PER-METHOD" / "EMA" / "eval_roundtrip" / "Strict scorer rule" / "Submission auth eval — BOTH CPU AND CUDA"
10. Lane registry check: `lane_rate_attack_f1_hydra_dims_7_12_substrate_20260518` already pre-registered by primary at L0
11. Predecessor checkpoint: `python3 tools/subagent_checkpoint.py read --subagent-id codex_rate_attack_f1_20260518`
12. Predecessor probe outcome check per Catalog #313: `python3 tools/check_predecessor_probe_outcome.py --substrate rate_attack_f1_hydra_dims_7_12 --recipe substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml` (expect NO_PRIOR_VERDICT)

## 1. Phase 1 — PROBE (FREE; $0; LOCAL)

**Goal**: Verify the structural cargo-cult "encoder can freely set dims 7-12 without affecting forward pass" is HARD-EARNED-VERIFIED-EMPIRICALLY.

**Deliverable**: `tools/probe_hydra_dim_7_12_score_invariance.py` (~150 LOC)

**Pseudocode**:
```python
#!/usr/bin/env python3
"""
Probe per Catalog #313: empirically verify upstream/modules.py:84 first-6-dims slice
means dims 7-12 of pose head output are SCORE-INVARIANT.

Method: take PR101 frontier archive (sha 6bae0201...); run inflate.sh + upstream/evaluate.py;
capture baseline pose values + score; mutate pose dims 7-12 to random bits; re-run evaluate.py;
confirm score IDENTICAL to 1e-9 precision across 600 pairs on BOTH --device cuda + --device cpu.
"""
import argparse, hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PR101_FRONTIER_ARCHIVE_SHA_PREFIX = "6bae0201"

def run_evaluate(archive_dir, device):
    # Runs upstream/evaluate.py --device <device> on the archive_dir
    # Returns full per-pair pose values + aggregate score
    ...

def mutate_pose_dims_7_12(pose_array, seed=42):
    # In-place mutation of dims 6-11 (0-indexed) of pose_array
    # Uses deterministic random per seed
    rng = np.random.default_rng(seed)
    pose_array[..., 6:12] = rng.standard_normal(pose_array[..., 6:12].shape, dtype=np.float32)
    return pose_array

def probe(archive_path, device):
    baseline = run_evaluate(archive_path, device)
    mutated = mutate_pose_dims_7_12(baseline.pose_values.copy())
    # Re-run evaluate.py with mutated pose values
    ...
    return {
        "device": device,
        "baseline_score": baseline.score,
        "mutated_score": mutated_score,
        "score_diff_abs": abs(baseline.score - mutated_score),
        "verdict": "PASSED" if abs(baseline.score - mutated_score) < 1e-9 else "FAILED",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(REPO_ROOT / "submissions" / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean" / "archive.zip"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = []
    for device in ("cpu", "cuda"):
        results.append(probe(args.archive, device))

    overall_verdict = "PASSED" if all(r["verdict"] == "PASSED" for r in results) else "FAILED"

    # Register probe outcome per Catalog #313
    from tac.probe_outcomes_ledger import register_probe_outcome
    register_probe_outcome(
        probe_id=f"f1_hydra_dim_7_12_score_invariance_{int(time.time())}",
        substrate_id="rate_attack_f1_hydra_dims_7_12",
        recipe_path="substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml",
        verdict="PROCEED" if overall_verdict == "PASSED" else "DEFER",
        rationale=f"Probe per-device results: {results}",
        status="blocking" if overall_verdict == "FAILED" else "advisory",
        event_type="adjudicated",
    )

    print(json.dumps({"overall": overall_verdict, "per_device": results}, indent=2))
    sys.exit(0 if overall_verdict == "PASSED" else 1)

if __name__ == "__main__":
    sys.exit(main())
```

**Acceptance criteria**: probe outputs `overall=PASSED` AND `per_device[*].verdict=PASSED` AND `score_diff_abs < 1e-9` AND probe outcome registered in `.omx/state/probe_outcomes.jsonl`.

**Gates**:
- If PASSED → Phase 2
- If FAILED → re-classify F1 to substrate-engineering with auxiliary loss path; cost increases $1-3 → $3-8

## 2. Phase 2 — SUBSTRATE BUILD + SMOKE ($1-3; MODAL T4)

**Goal**: Implement F1 substrate; Modal T4 100-epoch smoke; paired Linux x86_64 [contest-CPU] anchor.

### 2.1 Substrate package `src/tac/substrates/rate_attack_f1_hydra_dims_7_12/`

Files:
- `__init__.py` — public API + register_substrate(SubstrateContract(...)) per Catalog #241/#242
- `architecture.py` — F1 encoder with side-channel emission
- `archive.py` — F1 grammar (side-channel section + main payload section)
- `score_aware_loss.py` — `score_pair_components` adapter + auxiliary `L_side_channel = MSE(pose_out[..., 6:12], target_bits)` loss
- `inflate_runtime.py` — side-channel extraction + payload reassembly helper
- `tests/` — unit tests

### 2.2 Trainer `experiments/train_substrate_rate_attack_f1_hydra_dims_7_12.py`

- ADOPT `tac.substrates._shared.trainer_skeleton::device_or_die` per Catalog #178
- ADOPT `tac.differentiable_eval_roundtrip::patch_upstream_yuv6_globally` + `load_differentiable_scorers` per CLAUDE.md "Forbidden scorer load before patch"
- ADOPT `score_pair_components` per Catalog #164
- Trainer's `_full_main(args)` MUST be implemented (not raise NotImplementedError per Catalog #240)
- `TIER_1_OPERATOR_REQUIRED_FLAGS` dict per Catalog #151 declares `--enable-autocast-fp16` (Catalog #172) + `--enable-mp4-codec-sim` + sister Tier 1 flags

### 2.3 Submission `submissions/rate_attack_f1/`

- `inflate.py` (~180 LOC) — ADOPT canonical inflate device per Catalog #205 + F1 side-channel extraction
- `inflate.sh` — ADOPT canonical 3-arg signature per Catalog #146
- NO scorer load at inflate per CLAUDE.md "Strict scorer rule"

### 2.4 Operator-authorize recipe `.omx/operator_authorize_recipes/substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml`

```yaml
substrate_id: rate_attack_f1_hydra_dims_7_12
platform: modal
gpu: T4
min_vram_gb: 14  # T4 has 16 GB; reserve 2 GB headroom
min_smoke_gpu: T4
video_input_strategy: per_dispatch_local_copy
pyav_decode_strategy: cpu_thread_async_upload
target_modes:
  - contest_exact_eval
  - contest_generalized
canary_status: independent_substrate
predicted_band: "[-0.012, -0.004]"
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "post-training Tier-C re-measurement per Catalog #324 + paired Linux x86_64 [contest-CPU] anchor"
research_only: false
dispatch_enabled: true
cost_band:
  epochs: 100  # smoke
  estimated_cost_usd: 0.50
trainer_path: experiments/train_substrate_rate_attack_f1_hydra_dims_7_12.py
lane_script: scripts/remote_lane_substrate_rate_attack_f1_hydra_dims_7_12.sh
required_input_files:
  archive_pr101_frontier:
    flag: "--pr101-frontier-archive"
    default_path: "submissions/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean/archive.zip"
    required_input_file: true
env_overrides:
  Z6_TRAINER_MODE: "full"  # not smoke; per Catalog #326 — never bias to smoke without explicit recipe override
  SMOKE_ONLY: "0"
TIER_1_EXTRA_MOUNT_PATHS:
  - "submissions/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean/archive.zip"  # per Catalog #152 Modal-IGNORED extra-mount
```

### 2.5 Driver `scripts/remote_lane_substrate_rate_attack_f1_hydra_dims_7_12.sh`

Per Catalog #244 canonical 3-export NVML env block:
```bash
#!/bin/bash
set -euo pipefail

# Catalog #244: canonical NVML env block
export DALI_DISABLE_NVML=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Sentinel: REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents main flow side-effects per Catalog #163
REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source scripts/remote_archive_only_eval.sh

bootstrap_runtime_deps  # per CLAUDE.md "Forbidden re-implementing remote bootstrap inline"

# Catalog #326: support Z6_TRAINER_MODE / SMOKE_ONLY env var; default full
MODE="${Z6_TRAINER_MODE:-${SMOKE_ONLY:-full}}"

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=100
else
  EPOCHS=1000
fi

# Canonical multi-candidate path resolution per Catalog #152 Wave 2 driver extension
resolve_required_input_modal_aware() {
  local rel_path="$1"
  for root in "$WORKSPACE" "/workspace/pact" "/tmp/pact"; do
    if [[ -f "$root/$rel_path" ]]; then
      echo "$root/$rel_path"
      return 0
    fi
  done
  echo "FATAL: required input $rel_path not found in any candidate root" >&2
  exit 25
}

ARCHIVE_PATH=$(resolve_required_input_modal_aware "submissions/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean/archive.zip")

uv run --with-requirements pyproject.toml python experiments/train_substrate_rate_attack_f1_hydra_dims_7_12.py \
  --enable-autocast-fp16 \
  --pr101-frontier-archive "$ARCHIVE_PATH" \
  --epochs "$EPOCHS"
```

### 2.6 Dispatch via canonical `tools/operator_authorize.py`

```bash
python3 tools/operator_authorize.py \
  --recipe substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml \
  --confirm
```

This routes through ALL canonical gates per CLAUDE.md "Operator gates must be wired and used":
- Catalog #151 (Tier-1 flags wire-in)
- Catalog #152 (required-input-file validation)
- Catalog #243 (local pre-deploy harness)
- Catalog #271 (codex pre-dispatch review)
- Catalog #270 (canonical dispatch optimization protocol)
- Catalog #167 (smoke-before-full pattern)
- Catalog #313 (predecessor probe outcome check)
- Catalog #191 (sentinel files threaded per Catalog #166)

### 2.7 Paired Linux x86_64 [contest-CPU] anchor

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable:
1. Vast.ai CPU instance (Linux x86_64) OR Modal CPU container OR GitHub Actions CI
2. Run `upstream/evaluate.py --device cpu` on the SAME archive bytes
3. Tag result `[contest-CPU]`

## 3. Phase 3 — FULL ($5; CONDITIONAL on Phase 2 smoke validating)

**Goal**: Modal A100 1000-epoch full + paired Linux x86_64 [contest-CPU] anchor; submit PR.

Per Catalog #324: predicted_band [-0.012, -0.004] validated post-training when full-run + paired CPU anchor confirm.

If validated → submit PR per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + "Submission PR gate" non-negotiables.

## 4. Discipline (NON-NEGOTIABLE)

- Catalog #229 premise verification BEFORE each section
- Catalog #287 evidence tags on ALL numeric claims
- Catalog #206 checkpoint discipline every ~10 tool uses via `tools/subagent_checkpoint.py`
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha for EVERY file
- Catalog #126 lane already pre-registered by primary
- Catalog #314 absorption avoidance: YOUR scope is ONLY the source files listed in `write_scope_for_codex`
- Sister subagents disjoint scope per CLAUDE.md "Subagent coherence-by-default":
  - Primary owns `.omx/research/rate_attack_*.md` + `.omx/research/structural_information_not_shipped_*.md` + `.omx/research/codex_routing_directive_rate_attack_*.md`
  - Adversarial sister owns `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md`
  - YOU own source code per write_scope_for_codex

## 5. Routing Outcomes

When Phase 1 PROBE completes:
1. Register probe outcome per Catalog #313
2. Update master memo §10 op-routables status
3. Emit checkpoint per Catalog #206

When Phase 2 SUBSTRATE BUILD + SMOKE completes:
1. Register Modal call_id per Catalog #245
2. Append cost-band anchor per Catalog #175/#177
3. Update Catalog #316 frontier ledger if NEW frontier
4. Emit council deliberation anchor per Catalog #300

When Phase 3 FULL completes:
1. Submit PR per CLAUDE.md "Submission PR gate"
2. Update reports/latest.md FRONTIER section

## 6. Cross-References

- Parent design memo: `.omx/research/rate_attack_vector_1_f1_hydra_dims_7_12_design_memo_20260518.md`
- Parent master memo: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
- META-paradigm: `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
- Source anchors: `upstream/modules.py:26,67-79,84`
- CLAUDE.md non-negotiables: all from primary lane
- Catalog gates: #125 / #146 / #151 / #152 / #163 / #164 / #166 / #167 / #172 / #175 / #177 / #178 / #191 / #205 / #229 / #240 / #241 / #243 / #245 / #270 / #271 / #287 / #292 / #294 / #296 / #300 / #303 / #305 / #313 / #316 / #324 / #325 / #326
