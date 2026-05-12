# Design — Trainer Flag Manifest: closing the "landed-but-not-wired" gap and making features easy to compose

**Date**: 2026-05-12
**Author**: Claude (this session, operator-directed)
**Status**: DESIGN — awaiting council + operator approval before Catalog #151 lands
**Scope**: Operator-asked meta-questions
  1. "how can we permanently fix and protect against the landed but not yet wired and integrated"
  2. "how can we make things as easy to extend and wire up and compose"

These are two faces of the same problem. Section 1 frames the bug class; Section 2 enumerates options; Section 3 recommends one; Section 4 is the strict-preflight that locks the choice; Section 5 is the council ask.

---

## 1. The bug class

**Definition**: a trainer/codec/script lands a new CLI flag, but the downstream wrapper / dispatcher / operator-authorize script that should drive it never gets updated. The feature is "landed but not wired." Dispatches run without it. The score floor doesn't move because the speedup/quality win never actually fires.

**Concrete instances in this session (2026-05-12)**:

| # | Landed | Not wired in | Time-to-discovery | Cost-of-miss |
|---|---|---|---|---|
| 1 | trainer `--enable-autocast-fp16` (commit b0ef91a3) | `scripts/remote_lane_t1_balle_endtoend.sh` + `scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh` | fresh-eyes subagent (NF1) | every Phase-1 dispatch ran fp32, ~2× slower than necessary |
| 2 | trainer `--enable-mp4-codec-sim` (same commit) | same two scripts | same | proxy-auth gap on Phase-1 worse than it had to be |
| 3 | trainer default `--segmentation-surrogate=soft_cosine` (commit 3aecb9b8) | remote driver hardcoded `${SEG:-sinkhorn}` override | same | every score-domain dispatch ran the O(N²) surrogate |
| 4 | trainer `--batch-size` default 16→32 | remote driver hardcoded `${BATCH_SIZE:-16}` | same | every dispatch ran at half the intended throughput |

All four are the same pattern. All four would have been caught by **the same single STRICT preflight check** if it had existed.

**Symmetric problem (the user's second question)**: when you add a *new* feature behind a flag, the friction of "now go update the wrapper + the dispatch script + the env-var ladder + the docs" is high enough that people skip it. The bug class is the *exhaust* of this friction.

**Therefore**: the design that makes wire-up effortless is the *same* design that prevents the bug class. We're not building two systems; we're building one.

---

## 2. Design options

All four options share a common substrate: the trainer declares its operator-tier flags somewhere; the wrapper consumes that declaration somehow; a STRICT preflight refuses drift.

The options differ in *where the declaration lives* and *how much code is auto-generated*.

### Option A — module-level constant in the trainer (lightweight, fast to land)

The trainer exposes a module-level constant:

```python
# experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--enable-autocast-fp16": {"env": "T1_ENABLE_AUTOCAST_FP16", "rationale": "fp16 throughput on A100/4090"},
    "--enable-mp4-codec-sim": {"env": "T1_ENABLE_MP4_CODEC_SIM", "rationale": "shrinks proxy-auth gap by ~30%"},
    "--enable-t20-kl-pose-distill": {"env": "T1_ENABLE_T20_KL_POSE_DISTILL"},
    "--enable-t22-temporal-consistency": {"env": "T1_ENABLE_T22_TEMPORAL_CONSISTENCY"},
    "--segmentation-surrogate=soft_cosine": {"env": "SEGMENTATION_SURROGATE", "default_value": "soft_cosine"},
}
```

**Strict preflight Catalog #151** (`check_operator_wrapper_threads_trainer_tier_required_flags`):
1. Walk every `scripts/operator_authorize_*.sh` + `scripts/remote_lane_*.sh`
2. For each one, AST-grep which trainer module-paths it references (via `$WORKSPACE/experiments/<trainer>.py` substring)
3. Import the referenced trainer (or AST-walk it, no import side-effects); read `TIER_1_OPERATOR_REQUIRED_FLAGS`
4. For each declared flag, verify the wrapper *either* hardcodes it *or* threads the env-var (`if [ "${ENV:-0}" = "1" ]; then` block visible)
5. Same-line waiver: `# TIER_REQUIRED_FLAG_WAIVED_OK:<flag>:<reason>`
6. STRICT-flip after live-count drops to 0

**Pros**: minimal new code; declaration lives next to the flag it declares; no auto-codegen.
**Cons**: still requires hand-editing the wrapper; the protection catches drift, doesn't *eliminate* the friction.
**Effort**: ~150 LOC + 30 tests + 1 commit-batch. ~3h.

### Option B — auto-codegen of env→CLI block from manifest (medium, eliminates friction)

Same manifest as Option A, **plus** a code-generator:

```bash
python tools/regenerate_operator_wrapper_envcli_block.py \
    --trainer experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py \
    --wrapper-tier 1 \
    --emit-block-into scripts/remote_lane_t1_balle_endtoend.sh \
    --emit-section-marker "BEGIN_TIER_1_AUTOWIRE / END_TIER_1_AUTOWIRE"
```

The generator emits the entire `if [ "${T1_ENABLE_X:-0}" = "1" ]; then TRAIN_CMD+=(--enable-x); fi` block between section markers. The STRICT preflight check additionally verifies the emitted block is byte-identical to the canonical output (idempotent regenerate → no diff).

Adding a new flag becomes:
1. Add the argparse flag to the trainer + add the entry to `TIER_1_OPERATOR_REQUIRED_FLAGS`
2. Run `python tools/regenerate_operator_wrapper_envcli_block.py --all`
3. Commit both

The wrapper is now *generated*, not hand-edited. The strict check refuses any wrapper whose autowire block diverges from the canonical regenerate.

**Pros**: eliminates the wire-up friction entirely; trainer is the single source of truth.
**Cons**: more code; the codegen mechanism itself is a wire-in surface; section-marker discipline is required.
**Effort**: ~500 LOC + 60 tests + a codegen tool + the strict check. ~10h.

### Option C — typed-Python dispatcher replaces shell wrappers (heavy, future-proof)

Replace the shell layer entirely. Operator-authorize wrappers become:

```python
# scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.py
from tac.operator_authorize import OperatorAuthorize, Tier1Profile
from experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend import TIER_1_OPERATOR_REQUIRED_FLAGS

OperatorAuthorize(
    profile=Tier1Profile(
        trainer="experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
        required_flags=TIER_1_OPERATOR_REQUIRED_FLAGS,  # auto-discovered
        cost_band_usd=(0.50, 1.50),
        platforms=("modal", "vastai"),
    )
).run_with_confirmation()
```

The shell layer's job (ENV_OVERRIDES, platform dispatch, dispatch-claim coordination, modal/vastai backend selection) is encoded as a typed class. Required flags are auto-discovered from the trainer module. Wire-up becomes a single typed import.

**Pros**: composition becomes trivial (Tier1Profile × Tier2Profile = combined dispatch); typed contract; no codegen drift class; richer error messages.
**Cons**: large migration; existing shell wrappers all need to be rewritten; bash → python transition friction.
**Effort**: ~2000 LOC across `tac.operator_authorize` + migration of ~8 wrappers + ~100 tests. ~30-40h.

### Option D — Hybrid: Option A now, Option B in the next session, Option C if pattern recurs

Land Option A this session as the immediate Catalog #151 protection (~3h, low risk). After that protection is strict-flipped at 0, evaluate whether the "easy to extend" goal is met: if the next 2-3 trainer feature-lands take <30 min each to wire (with the protection catching forgetters), Option A is enough. If wire-up friction is still high enough to slow feature velocity, escalate to Option B in the following session.

**Pros**: incremental, evidence-driven, low-risk first step; preserves option value.
**Cons**: doesn't deliver the full composition story up front.

---

## 3. Recommendation

**Option D (hybrid)**. Specifically:

- **This commit-batch** (NF1 fix + this design memo): land the immediate NF1 wire-fix. Land THIS memo as the operator-routable design. Do NOT land Catalog #151 yet (it's a council-grade design decision per CLAUDE.md "Design decisions — non-negotiable").
- **Next commit-batch** (operator-approved): land Option A — `TIER_1_OPERATOR_REQUIRED_FLAGS` constant on `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py` + Catalog #151 strict check, warn-only initially, strict-flip at 0.
- **Following session**: re-evaluate. If wire-up friction is still high, design Option B's codegen tool with council review.

This honors:
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — the protection (Catalog #151) is identified and scoped, not silently deferred.
- CLAUDE.md "Design decisions — non-negotiable" — the multi-option choice gets operator/council review before implementation.
- CLAUDE.md "Don't add features … beyond what the task requires" — Option C is correctly deferred until evidence demands it.
- CLAUDE.md "Subagent coherence-by-default" — the manifest IS a coherence primitive; declaring it now extincts a recurring bug class.

---

## 4. Catalog #151 sketch (Option A's strict-preflight)

```python
# src/tac/preflight.py
def check_operator_wrapper_threads_trainer_tier_required_flags(strict: bool = False) -> None:
    """Refuses operator-authorize/remote-lane scripts that target a trainer
    declaring TIER_<N>_OPERATOR_REQUIRED_FLAGS without threading every
    declared flag (either hardcoded in the TRAIN_CMD array or via the
    env-var name declared in the manifest).

    Same-line waiver: `# TIER_REQUIRED_FLAG_WAIVED_OK:<flag>:<reason>`.

    Discovery:
    1. Walk scripts/operator_authorize_*.sh + scripts/remote_lane_*.sh.
    2. For each, parse out python invocations (literal `experiments/<x>.py`).
    3. For each referenced trainer file, AST-walk to find module-level
       Assign of `TIER_*_OPERATOR_REQUIRED_FLAGS = {...}`.
    4. For each flag in the dict, scan the shell file for one of:
       - literal flag in a TRAIN_CMD+=(...) block
       - `if [ "${<ENV>:-...}" = "1" ]; then TRAIN_CMD+=(--<flag>); fi` block
       - `--<flag> "${<ENV>:-<default>}"` substitution
    5. Refuse if any tier-required flag is missing AND no same-line waiver.

    Bug class: 2026-05-12 four-instance burst —
    --enable-autocast-fp16, --enable-mp4-codec-sim,
    --segmentation-surrogate (default-override), --batch-size
    (default-override) all landed in trainer but not in wrapper or driver.

    Sister of Catalog #12 (preflight_arity — caller flags must be a
    subset of target argparse, the dead-flag detector). This is the
    inverse: target-required flags must be a subset of caller-threaded
    flags. Together they close the bidirectional gap.
    """
```

Roughly 150 LOC implementation + 30 dedicated tests. Wire into `preflight_all()` warn-only; strict-flip after live-count = 0.

---

## 5. Council ask + next-step gate

**Operator/council decision OD-WIRE-1**: which option (A/B/C/D)?

**Operator/council decision OD-WIRE-2**: if Option A or D, who lands Catalog #151?
- Same Claude session via subagent? (~3h follow-up work after this commit-batch)
- Codex subagent? (cleaner adversarial pass on the design)
- Defer to next operator-decision sweep?

**Operator/council decision OD-WIRE-3**: scope — do we also retrofit the same manifest pattern to non-trainer surfaces (e.g., `tac.packet_compiler` codec-flag wiring, `tools/cathedral_autopilot_*.py` ranking-flag wiring)? Same bug class, different surface.

**No Claude-side autonomous decision on any of these.** This is a structural design choice per CLAUDE.md "Design decisions — non-negotiable" (>$1/hr potential impact + multiple alternatives with non-trivial preference).

---

## 6. Cross-refs

- The four bug instances this session: `feedback_adversarial_review_fixup_pass_1_landed_20260512.md` (F2/F5 autocast smoke), the in-flight fresh-eyes NF1 finding (this commit)
- Sister Catalog: #12 `preflight_arity` (the dead-flag detector this would mirror)
- CLAUDE.md non-negotiable: "Bugs must be permanently fixed AND self-protected against"
- CLAUDE.md non-negotiable: "Design decisions — non-negotiable" (requires council approval)
- CLAUDE.md non-negotiable: "Subagent coherence-by-default" (the manifest IS a coherence primitive)
- CLAUDE.md FORBIDDEN_PATTERNS: "forbidden_default_to_convenience_trap", "forbidden_dead_flag_wiring_pattern" — both are dual to this proposed protection
