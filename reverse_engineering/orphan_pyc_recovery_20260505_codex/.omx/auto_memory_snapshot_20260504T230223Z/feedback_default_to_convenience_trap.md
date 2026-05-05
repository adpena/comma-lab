---
name: NEVER default to convenience over correctness — especially MPS fallbacks
description: Multiple times this session I've defaulted to MPS-fallback or proxy-as-measurement out of "available locally" convenience, even with CLAUDE.md non-negotiables saying MPS is NOISE. The pattern is automatic — it has to be manually disabled at design time, not noticed in review.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Binding rule:** When writing any script that touches the eval pipeline, NEVER default to "use whatever's available." Default to the CONTEST-CORRECT device (CUDA) and FAIL LOUD if it's not available. Convenience defaults are how the MPS-noise non-negotiable keeps getting bypassed.

**Why** (2026-04-27 incident):

Writing `experiments/build_baseline_archive.py` (canonical script to rebuild the 0.9001 baseline's masks). My first draft had:
```python
if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
```

This is the SAME pattern that's been documented as forbidden in CLAUDE.md's "MPS auth eval is NOISE" non-negotiable for weeks. On any local machine without CUDA (i.e. the M5 Max), it would silently fall back to MPS — producing SegNet outputs that drift 2x from CUDA, producing a different masks.mkv, producing a different archive bytes, producing a different score.

The user caught it: *"are you respecting deterministic reproducibility"* → audit → *"you almost made the MPS mistake again before I stepped in."*

**How to apply:**
1. **Default device for any eval/data-prep script must be CUDA-REQUIRED.** Raise SystemExit with explicit reasoning if CUDA is not available. Provide an explicit `--device cpu` opt-in for development smoke tests, with a banner that the bytes WILL differ.
2. **Never write `device = "cuda" if cuda.is_available() else "mps"`** — even as a "smart" fallback. This is the trap that bypasses the MPS non-negotiable.
3. **Audit existing scripts** for this pattern: grep for `mps.is_available()` and ensure any falls-through go to CPU not MPS, AND the script either refuses or warns loudly.
4. **Pattern check**: any time I'm about to write "if X is available else Y is available else Z" for device selection, STOP. Ask: "is the user submitting their work to a contest that runs on Y or Z?" If not, refuse the fallback.

**Cost of this trap (cumulative, this session):**
- ~6 hours debugging "phantom baselines" that were partially MPS-vs-CUDA drift
- Multiple memory entries (`feedback_mps_cuda_drift_critical`, `feedback_proxy_auth_math_useless`) all written AFTER I made this mistake
- The user has spent emotional energy correcting me on the SAME pattern multiple times

**The meta-meta-lesson:** documenting a non-negotiable in CLAUDE.md is not enough. The non-negotiable must be ENFORCED at the code-write moment, not the code-review moment. If I'm writing a function and the "convenient default" is forbidden, the convenient default doesn't exist — refuse it before typing.
