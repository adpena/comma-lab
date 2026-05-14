# Harness engineering / rigor / deterministic reproducibility directive 2026-05-14

**Operator directive verbatim 2026-05-14**: *"harness engineering and rigor and deterministic reproducibility and everything will be very important"*

**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within the last 24 hours. **EVERY future subagent + nested spawn + in-flight Grand Council MUST honor.**

## Why this directive

The across-class campaign push pre-authorized by the operator (commit `2d32c8ac1` recursive NO-SIGNAL-LOSS + `6fcad3105` journal-grade standard) requires harness-grade execution. Without rigor + reproducibility, parallel dispatch produces UNAUDITABLE results: a $50 multi-stage Tier 2 anchor with no exact-command provenance cannot be re-derived by another agent in 6 months, cannot be re-run when the operator asks "what if X?", and cannot be submitted to the contest as evidence. This directive codifies the eight harness pillars all future subagents inherit.

## The 8 harness pillars

### Pillar 1 — Canonical pipeline routing

Per CLAUDE.md "Canonical pipeline standard — non-negotiable", every experiment routes through `experiments/pipeline.py --profile <name>`. Anti-pattern: hand-crafted SSH + bash one-liners. The profile IS the experiment definition; no CLI flag overrides for architecture params. Required: profile from `src/tac/profiles.py` registered + pinned.

### Pillar 2 — Seed + determinism pinning

Every training/eval subagent MUST:
- `torch.manual_seed(profile.seed)`
- `numpy.random.seed(profile.seed)`
- `random.seed(profile.seed)`
- `torch.use_deterministic_algorithms(True)` where supported
- `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` (per Catalog #204)
- TF32 enforcement via canonical `tac.substrates._shared.trainer_skeleton.device_or_die` (Catalog #178)
- Refuse MPS for any authoritative axis (CLAUDE.md "MPS auth eval is NOISE" non-negotiable)

### Pillar 3 — Sentinel-file mtime + worker source parity

Modal dispatches MUST honor Catalog #165 (mount-mtime stability) AND Catalog #166 (worker source-parity ledger). NEVER bypass via `TAC_MODAL_MTIME_STABILITY_DISABLED=1` — per CLAUDE.md "Bugs must be permanently fixed AND self-protected against", the gate is the protection. The empirically-validated parallel-wave dispatcher-vs-editor serialization pattern (`feedback_parallel_wave_dispatch_vs_editor_serialization_pattern_20260514.md`) is mandatory: Phase 1 editors-first, Phase 2 dispatchers-second.

### Pillar 4 — Exact dependency closure

Per Catalog #203 + #224: every Modal training image MUST include hard runtime deps (`brotli`, `constriction`, `pyppmd`) declared in `pyproject.toml` AND propagated via `tac.deploy.modal.runtime` (`DALI_DISABLE_NVML=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Driver-version pinning per CLAUDE.md "Forbidden uv torch install without driver-version pin": auto-detect `driver_major < 580` → `torch==2.5.1+cu124` else `torch==2.11.0`.

### Pillar 5 — Durable provider output paths

Per Catalog #204: Modal smoke + full output writes to `/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output`, NOT `/tmp/`. The `experiments/contest_auth_eval.py` refuses temp-storage evidence by design. Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact (the transient-evidence trap)" non-negotiable.

### Pillar 6 — Smoke-before-full discipline

Per Catalog #167: every operator-authorize substrate dispatch routes through `tools/run_modal_smoke_before_full.py`. Smoke validates integration ($0.30-1.00) BEFORE the full ($5-50) fires. Recipe declares `min_smoke_gpu` (Catalog #215) + `canary_status` (Catalog #173). Same-line `# SMOKE_BEFORE_FULL_OK:<reason>` waiver for established trainers with ≥3 successful anchors.

### Pillar 7 — Custody validator + posterior locked writes

Every empirical anchor MUST route through:
- `tac.continual_learning.ContestResult.validate_custody(...)` (Catalog #127 / #130) returning `CustodyVerdict` with explicit refusal class
- `tac.continual_learning.posterior_update_locked(...)` (Catalog #128) under `fcntl.flock(LOCK_EX)`
- `tac.cost_band_calibration.append_anchor(outcome=...)` (Catalog #175 + #177)
- Canonical auth-eval helper `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call(...)` (Catalog #226)

Bare `save_posterior(...)` outside the lock is FORBIDDEN per Catalog #131. Bare `subprocess.run(['contest_auth_eval.py', ...])` outside the gate is FORBIDDEN per Catalog #226.

### Pillar 8 — Append-only HISTORICAL_PROVENANCE

Per Catalog #110 + #113: every long-lived artifact classifies as LIVE_STATE / HISTORICAL_PROVENANCE / LIVE_RECIPE / DERIVED_OUTPUT. Recovery metadata is append-only (no timestamp-only mutation). Public-PR intake clones stay pristine (Catalog #109). Status JSONs emit `generated_at: <utc>` + `from_state_hash: <sha>` (Catalog #111). Rebuild commands avoid baked timestamps (Catalog #112).

## The deterministic reproducibility table (mandatory per landing)

Every landing memo + campaign ledger + dispatch report MUST embed a table like:

| Element | Value | Verification |
|---|---|---|
| HEAD commit | `<sha>` | `git rev-parse HEAD` |
| Lane registry sha | `<sha>` | `sha256sum .omx/state/lane_registry.json` |
| Archive sha256 | `<sha256>` | `sha256sum <path/to/archive.zip>` |
| Archive bytes | `<N>` | `stat -c%s <path>` |
| inflate.sh sha | `<sha>` | `sha256sum <inflate.sh>` |
| Runtime-tree sha | `<sha>` | declared in build manifest per Catalog #105 |
| Modal call_id | `<fc-XX>` | `modal app logs <call_id>` |
| Vast.ai instance | `<N>` | `vastai show instance <N>` |
| Provider rate | `<USD/hr>` | per provider quote |
| Wall-clock | `<sec>` | run.log start/end |
| Hardware substrate | `linux_x86_64_<gpu>` | `tac.deploy.detect_hardware_substrate` (Catalog #190) |
| Axis tag | `[contest-CUDA T4]` / `[contest-CPU GHA Linux x86_64]` / `[macOS-CPU advisory only]` | per Catalog #127 |
| Score | `<S>` | `experiments/contest_auth_eval.py --archive <path> --json-out <out>` |
| Score evidence | `<json path>` | `cat <out> | jq .score` |
| Predicted vs empirical Δ | `<float>` | `predicted - empirical` |
| Reproduce command | exact | including all flags |

## The 12 forbidden harness anti-patterns

Per CLAUDE.md FORBIDDEN_PATTERNS section, this directive adds:

1. **Forbidden hand-crafted SSH** instead of canonical pipeline + canonical dispatchers
2. **Forbidden missing seed pin** in any training/eval module
3. **Forbidden mtime bypass** via `TAC_MODAL_MTIME_STABILITY_DISABLED=1`
4. **Forbidden bare subprocess** `contest_auth_eval.py` (must use `gate_auth_eval_call`)
5. **Forbidden bare posterior write** (must use `posterior_update_locked`)
6. **Forbidden `/tmp/` durable evidence** path
7. **Forbidden missing `archive_sha256` + bytes + axis-tag** in any score claim
8. **Forbidden bare Modal dispatch** without `--sentinel-files` + `--require-clean-head` (Catalog #191)
9. **Forbidden bare auth-eval CLI flag invention** (must grep argparse first per Catalog #12)
10. **Forbidden missing custody validator** routing (Catalog #127 / #130)
11. **Forbidden missing reproduce command** in any landing memo
12. **Forbidden missing predicted-vs-empirical Δ** in any anchor row

## Harness verification recipe (per landing)

Before declaring a subagent landing complete, run:

```bash
# 1. Test suite green
.venv/bin/python -m pytest src/tac/tests/ -q --tb=short 2>&1 | tail -20

# 2. Preflight strict
.venv/bin/python tools/preflight_hook.py --strict --scope dev 2>&1 | tail -20

# 3. Lane registry consistent
.venv/bin/python tools/lane_maturity.py validate 2>&1 | tail -5

# 4. Substrate trainer --help parse (no dead flags per Catalog #12)
.venv/bin/python experiments/train_substrate_<name>.py --help > /dev/null

# 5. Recipe yaml schema validation
.venv/bin/python tools/validate_dispatch_required_inputs.py \
    --trainer experiments/train_substrate_<name>.py
```

All 5 MUST exit 0. Any failure refuses the landing per CLAUDE.md "Operator gates must be wired and used" non-negotiable.

## Cross-tier coordination (per the parallel-wave serialization pattern)

For the grand council's tiered parallel plan:
- **Tier 0** (immediate $0-$2): may include 1-2 DISPATCHERS once Catalog #165 quiescence reached
- **Tier 1** (short $2-$15): dispatchers fire EITHER (a) in Phase 2 of the editor-first/dispatcher-second pattern, OR (b) when sister Tier 0 editors complete
- **Tier 2** (mid $15-$50): long-running multi-stage runs; their editors land in Phase 1; their dispatchers fire in Phase 2 of subsequent waves
- **Tier 3** (long $50-$500): the most expensive; coordinated to land BEFORE other tiers' editors disrupt mount stability

Concretely: the in-flight Grand Council subagent should produce a TIERED PLAN whose execution waves follow the dispatcher-vs-editor serialization pattern automatically.

## Operator-visibility checkpoints

Per the grand council's tiered plan, the operator MUST be able to ask at any time:

1. "What's tier 0 status?" → answer from `.omx/state/lane_registry.json` + recent landing memos
2. "What's tier 1 progress?" → answer from `tools/subagent_checkpoint.py read --latest-incomplete` per tier-1 lane
3. "What's the cost-band posterior look like for $0?" → `tac.cost_band_calibration.summarize_posterior()`
4. "What anchors have we landed today?" → `git log --since '<today UTC>' --oneline | grep '[contest-CUDA\|contest-CPU]'`
5. "What's the current frontier?" → `cat reports/latest.md`

## Cross-refs (verbatim)

- CLAUDE.md "Canonical pipeline standard — non-negotiable" (Pillar 1)
- CLAUDE.md "Deployment version checklist — non-negotiable" (Pillar 4)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE — NON-NEGOTIABLE" (Pillar 5)
- CLAUDE.md "Forbidden /tmp paths in any persisted artifact" (Pillar 5)
- CLAUDE.md "MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS" (Pillar 2)
- CLAUDE.md "Operator gates must be wired and used — NON-NEGOTIABLE" (Pillar 7-8)
- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE" (the orchestration-without-orchestrator pattern this directive applies)
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against — NON-NEGOTIABLE" (Pillar 7, custody validators)
- CLAUDE.md "Beauty, simplicity, and developer experience — non-negotiable" (the rigor + reproducibility doctrine)
- CLAUDE.md "Deterministic packet compiler — non-negotiable" (the canonical contract for native ports)
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (NO-SIGNAL-LOSS R1-R7)
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` (recursive R1-R4)
- `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` (11-element journal-grade)
- `feedback_parallel_wave_dispatch_vs_editor_serialization_pattern_20260514.md` (Phase 1 / Phase 2 dispatcher-vs-editor)

## Effective immediately

In-flight subagents (HARVEST-AND-Z1, CATALOG-226-REFACTOR, F3-GTSCORERCACHE-WIRE-IN, OSS-RELEASE-V0.2.0-RC1, GRAND-COUNCIL `adeeeaa8`) MUST honor this directive on their next checkpoint cycle via mandatory `.omx/research/*_directive_*` last-24-hours pre-read. Their landings MUST include the deterministic reproducibility table + the 5-step harness verification recipe + tag every score claim with the axis label per Pillar 7.

Tagged `research_only=true`. NO score claims. NO GPU spend by this directive. Effective for all subagents from this directive's commit forward.
