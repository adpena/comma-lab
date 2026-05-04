# Silent default masquerading as a negative result — 2-in-2-days bug class

**Date**: 2026-04-27 (Lane F + Lane G recurrence — second occurrence in 2 days)
**Severity**: BINDING NON-NEGOTIABLE — preflight check `check_no_silent_auto_discovery_with_warn` enforces.

## The pattern

A CLI flag is missing from a script. The script auto-discovers from a list of N hardcoded fallback paths. None exist. The script prints a `[WARN] ...` line and proceeds with a silent default (`None` / zero / empty). The operator sees the script "succeed" but the produced artifact was trained against the wrong inputs. The result then enters council deliberation as if it were a real negative result, leading to "this lane is dead" misjudgments.

```python
# FORBIDDEN — exact pattern from qat_finetune.py pre-fix:
poses = None
for poses_path in [
    Path("experiments/results/gt_poses.pt"),
    Path(cfg.upstream_dir) / "gt_poses.pt",
]:
    if poses_path.exists():
        poses = torch.load(poses_path)
        break
# poses is silently None here if neither exists → trains with zero poses,
# then archive bundles real poses → 23x PoseNet degrade.
print("[WARN] Renderer has pose_dim>0 but no poses_path provided — will use zero poses")
# ... trainer runs to completion, "succeeds", produces wrong artifact
```

## Real-world incidents (both 2026-04-27, within 24h)

1. **Lane F v1** (`experiments/qat_finetune.py`) — auto-discovered `gt_poses.pt` from 2 paths, neither existed, trained renderer with zero poses, deployed against real poses. Reported as "FP4 quantization is dead" (auth 2.73 vs 2.29 baseline = +0.44 regression). Council audit revealed the renderer had a PoseNet degrade of +58% from the conditioning-input mismatch alone, NOT from FP4 quant noise. The actual experiment was NEVER measured. Cost: ~$2 GPU + 1 day council bandwidth + a falsified "FP4 is dead" claim that almost killed a viable lane.

2. **Lane G v1** (`kl_distill_weight=5e-6` default with `reduction="batchmean"`) — silent over-weighting by 5000× because batchmean under-divides per-pixel by H×W=196,608. Reported as "KL distill killed PoseNet." Same class: a silent bad default produced an invalid result that read as a clean negative. Cost: ~$3 GPU + multi-round council deliberation.

## The structural fix — preflight gate

`check_no_silent_auto_discovery_with_warn` (29th meta-bug check, `src/tac/preflight.py`):

- AST-walk every `.py` in `src/tac/`, `experiments/`, `scripts/`.
- Detect functions containing BOTH:
  - (a) a `for x in [<list of >=2 Path-like literals>]:` loop body with `<x>.exists()` checks, AND
  - (b) somewhere later in the same function, an unguarded `print/log/warn(...[WARN]...)` call with no following `raise` / `sys.exit`.
- Either RAISE on missing input (preferred), or annotate the loop/function with `# AUTO_DISCOVERY_OK:<reason>` to opt out (only when the fallback is genuinely safe).

Wired into `preflight_all` strict at module load. Live count after Lane F fix: **0**.

## Operator rules (mandatory)

1. **Never invent a CLI fallback.** If a flag is required, fail loud. Don't print `[WARN] ... using default` and proceed.
2. **explicit > implicit** — same principle as the FORBIDDEN PATTERNS section in CLAUDE.md (mirror of `feedback_default_to_convenience_trap`).
3. **The operator must SEE the failure.** A silent fallback is worse than a crash because it pollutes the experimental record.
4. **If a fallback is genuinely safe** (e.g., an optional resource), document with `# AUTO_DISCOVERY_OK:<reason>` so the next reviewer can verify.
5. **Pair this rule with the dead-flag rule** (`feedback_dead_flag_wiring_pattern`): every CLI flag passed in a subprocess must be argparse-grep'd against the target. Together these two rules close the silent-corruption gap.

## Why it keeps happening

Both Lane F and Lane G failed the same review process: each had a bug at a boundary that no caller checked, and the WARN line was buried in 100s of lines of training output. The council reasoned about the result (high distortion → "the technique is dead"), not the apparatus (the renderer was never trained against the right inputs). The structural fix MUST be in preflight (caught at script-load time), not in the council protocol (caught after $2 of GPU has burned).

## References

- `findings.md` "## 2026-04-27 Council audit: Lane F regression — bugged or dead?"
- `findings.md` "## 2026-04-27 Council forensics: Lane G — really dead, or bugged?"
- Memory: `project_baseline_poses_load_bearing` (the 23× PoseNet degrade fingerprint)
- Memory: `feedback_default_to_convenience_trap` (cousin bug: device defaults)
- Memory: `feedback_dead_flag_wiring_pattern` (cousin bug: invented CLI flags)
- Test: `src/tac/tests/test_preflight_meta_bugs.py::TestNoSilentAutoDiscoveryWithWarn` (10 tests)
- Test: `src/tac/tests/test_qat_finetune.py::test_qat_finetune_raises_on_missing_poses_for_pose_dim_gt_0`

## Verification

```bash
# Live count must be 0 across the codebase after the qat_finetune.py fix.
.venv/bin/python -c "from tac.preflight import check_no_silent_auto_discovery_with_warn; \
    v = check_no_silent_auto_discovery_with_warn(strict=False, verbose=False); \
    print(f'LIVE COUNT: {len(v)}')"
# → LIVE COUNT: 0

# Strict mode raises on regression.
.venv/bin/python -c "from tac.preflight import preflight_all; preflight_all(verbose=False)"
# → preflight_all PASS
```
