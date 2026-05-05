---
name: Silent-default override bug class — 3 real bugs in train_renderer.py + audit hardening
description: 2026-04-29 PM. Sherlock Holmes audit chain landed. Found 3 REAL silent-default override bugs in train_renderer.py (commit 256c5e42), and hardened the audit tool to filter 246 noisy findings down to actionable ones (commit 4eeb6452). The bug class is the same as the original KL distill bug — argparse default that's not None silently overrides profile values when the resolver doesn't special-case it.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug class (definition)

**Pattern**: argparse `add_argument("--X", default=Y)` where Y is not None AND profile dict has key `X` with value Z ≠ Y. When user runs with `--profile XYZ` and does NOT pass `--X` on CLI, the profile's Z is SILENTLY ignored — args.X holds Y.

**Original incident**: Quantizr council CRITICAL 2026-04-26 — `--kl-distill-weight` had default=0.0, profile set 0.05, every production training run had KL distill DISABLED silently.

**Why this happens**: argparse's default-value mechanism is "set if user didn't pass." A profile-resolver typically checks `if args.X is not None: return args.X` — but if argparse already set args.X to the default Y, that check passes for Y, profile never consulted.

## How to detect (the audit)

Run `tools/audit_silent_defaults.py`. It scans every `add_argument(...)` in `experiments/` and `src/tac/experiments/` and flags non-None defaults whose flag name matches a key in `tac.profiles.PROFILES`.

After the 2026-04-29 PM hardening:
- File-level filter: only scripts that BOTH declare `--profile` AND `from tac.profiles import` are at risk (skips informational-only --profile flags like `train_imp_cycle.py:114`)
- File-level filter: scripts using `_user_provided_flags`/`_apply_profile`/`_resolve(args.X)` mechanisms are exempt (already handle override correctly)
- Per-arg filter: `action="store_true"` with `default=False` + `action="store_false"` with `default=True` are STRUCTURALLY correct, not flagged

Pre-hardening: 246 CRITICAL (noisy). Post-hardening: 0 CRITICAL (live repo clean after the 3 fixes).

**KNOWN GAP** in current audit: per-file `has_override_mechanism` is broad. A file with SOME `_resolve(args.X)` calls and SOME un-resolved non-None defaults passes the file-level check but contains real bugs in un-resolved flags. The 3 real bugs caught this loop in `train_renderer.py` were exactly this shape — caught only by manual review.

## How to fix (the canonical pattern)

For each silent-default flag:

```python
# BEFORE (silent override):
p.add_argument("--my-flag", type=float, default=0.5,
               help="...")

# AFTER:
p.add_argument("--my-flag", type=float, default=None,
               help="...")
# In the profile-resolve block:
args.my_flag = _resolve(args.my_flag, "my_flag", 0.5)
```

The `_resolve` helper is the canonical:
```python
def _resolve(cli_val, profile_key, default):
    if cli_val is not None:
        return cli_val
    return profile_vals.get(profile_key, default)
```

## 3 real bugs fixed today (commit 256c5e42)

1. **`--fp4-codebook`** default `'default'` silently overrode profile's `'residual'` in **14 profiles** (1 set 'default', 14 set 'residual'). The 'residual' codebook is denser-near-zero, designed for 4× better small-magnitude weight preservation. Every halfframe / Quantizr-replica / MAE-V / J-JBL training run was silently using the WRONG codebook.

2. **`--grad-clip`** default 1.0 silently overrode profile's 10.0 in 2 profiles. Effective gradient clipping was 10× too aggressive on those profiles.

3. **`--wall-clock-timeout`** default 0 (no limit) silently overrode profile's 39600 (11h) in 2 profiles. Modal lanes configured for 11h cap could silently run forever, burning budget.

Verified via:
```bash
.venv/bin/python -c "
import sys
sys.argv = ['train_renderer.py', '--profile', 'dilated_h64_half_frame', '--tag', 't', '--epochs', '1']
from tac.experiments.train_renderer import parse_args
args = parse_args()
assert args.fp4_codebook == 'residual'  # was silently 'default' before fix
"
```

## Sister bug class: "fix lands in helper but never callsite"

Same root cause: a function gets a new safety kwarg (e.g., `baseline_poses`), but the callers don't pass it, so the dangerous default still wins. Lane GP `fit_pose_gp.py:33` was this — `reconstruct_poses(model, n)` without `baseline_poses=baseline` even after Fix-A "landed". Fixed in commit `8746793e`.

The audit doesn't currently catch this class. A static-analysis check could: walk all callers of helpers with safety-flag kwargs and verify each caller passes the kwarg. Future work.

## Permanent prevention

To make this bug class structurally extinct:

1. ✅ `tools/audit_silent_defaults.py` exists and is hardened (this memory)
2. 🟡 STRICT preflight check that calls the audit and fails on CRITICAL>0 — pending
3. 🟡 Per-flag precision in audit — pending (current heuristic is per-file)
4. 🟡 Static analysis for "helper-callsite kwarg parity" — pending
5. 🟡 PR review checklist item: any new --X with non-None default + matching profile key requires `_resolve()` wiring

## Cross-refs

- tools/audit_silent_defaults.py (the audit, hardened in commit 4eeb6452)
- reports/silent_defaults.md (auto-generated, currently 0 CRITICAL)
- src/tac/experiments/train_renderer.py (3 real bugs fixed in commit 256c5e42)
- experiments/fit_pose_gp.py (sister bug class, fixed in commit 8746793e)
- CLAUDE.md "Default override antipattern" non-negotiable
