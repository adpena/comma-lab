# Pre-deploy adversarial sweep — canonical 10-point review

**Date:** 2026-04-28
**Trigger:** user observed "we basically wasted an entire day without
running any experiments" — TIER-1 lanes had been authored all day but
none cleared deploy because nobody had run the extreme adversarial
review across the FULL queue.

## The pattern this prevents

Without a periodic full-queue review, lanes accumulate in three states:
1. **Truly ready** — author shipped, smoke proof passed, structure clean
2. **Looks ready, secretly broken** — Lane RM-d archetype: trains 3.5h, then
   inflate.sh tries to read 0.mkv at archive eval → silent zero score.
3. **Superseded by an active variant** — V1 sitting on disk while V2 is
   already deployed; would burn $2-5 if dispatched.

A subagent that just "deploys whatever lane has a fresh smoke proof" hits
all 3 buckets indiscriminately. The adversarial sweep separates them.

## The canonical 10-point checklist (per lane)

1. E2E smoke proof fresh (<7d) + PASS (`.omx/state/lane_e2e_smoke_proofs.json`)
2. Profile sanity — referenced profile exists in `src/tac/profiles.py`
3. Loss mode validity — in train_renderer.py validator allowlist (Check 49)
4. Anchor reuse correctness — file exists; tarball auto-discovery includes (Check 43)
5. Dead flag scan — every `--flag` exists in target argparse
6. Auth eval path — contest_auth_eval invoked; no `0.mkv` class bug (Check 63)
7. Hardware compat — FP4/FP8/QAT properly gated (Check 40)
8. Recovery readiness — heartbeat.log + provenance.json present
9. Architecture sanity — orphan modules referenced exist on disk
10. Predicted band justified — band derived from documented mechanism

## Triage taxonomy for never-deployed lanes

After scanning `scripts/remote_lane_*.sh` vs `vastai_active_instances.json`:

- **SUPERSEDED** (~70% of never-deployed): V1 sits on disk because V2/V3 is
  the active variant. SKIP. Sample evidence: lane name has `_v2_` or `_v3_`
  suffix matching a deployed instance.
- **RESEARCH-ONLY** (10-15%): no inflate-time decoder; archive shipped is
  the anchor (e.g. Lane AC, Lane EA, Lane SQ). auth_eval just re-measures
  the anchor. SKIP unless the inflate decoder is also being shipped this cycle.
- **TIER 1** (15-20%): real new science not previously deployed. RUN.

## Time investment vs payoff

- **75 min** to audit 32 never-deployed lanes (~2.3 min/lane via smoke + checklist + categorization).
- **6 lanes cleared READY** for $15.80 / max 14h wallclock.
- **Per crash avoided**: $1.50-3.00 GPU + 6-14h wallclock + 1-2 day delay.
- Break-even: 2 crashes prevented. Today's frontier: ~5 crashes/week class-wise.

## When to run this sweep

- **Trigger 1**: end-of-day, before issuing any new deploy roster
- **Trigger 2**: after any new preflight check lands (verify it doesn't break the canonical lanes)
- **Trigger 3**: weekly cadence even without new lanes (catches stale anchors)
- **NOT a trigger**: every single dispatch — too granular; preflight + Check 64 cover the per-deploy case

## Top-3 highest-EV lanes from THIS sweep (for parent-shell dispatch)

1. **Lane J-NWCS-EC stack** — pred [0.78, 0.92] @ $1.50 / 6h. **First sub-1.0 stack candidate.**
2. **Lane Ω-Hessian-QAT** — pred [0.70, 1.05] @ $2.50 / 8h. Per-WEIGHT bit allocation moonshot.
3. **Lane MAE-V** — pred [0.85, 1.10] @ $4.00 / 14h. First sub-1.0 single-lane on G v3 anchor.

## The healthy-system signal

When the sweep completes with **zero fixes committed**, that is the
expected outcome of a healthy preflight architecture. The adversarial
review is the **verification** step, not the **discovery** step. Discovery
happens at code-write time via the 63+ static + 1 E2E smoke + tarball
anchor parity checks. The sweep just confirms the system worked.

If the sweep DOES find a fix needed, that is a process failure — a check
should have been added that catches the class. Don't ship the fix without
also extracting the meta-pattern into a new preflight check.

## References

- Audit doc: `.omx/research/pre_deploy_adversarial_sweep_20260428.md`
- Smoke registry: `.omx/state/lane_e2e_smoke_proofs.json`
- Active fleet: `.omx/state/vastai_active_instances.json`
- Launcher: `scripts/launch_lane_on_vastai.py`
- Related memory:
  - `feedback_canonical_local_e2e_smoke_landed_20260428.md` (Check 64)
  - `feedback_check_43_tarball_anchor_parity_20260428.md` (Check 43)
  - `feedback_codex_sandbox_blocks_vastai_dns_20260428.md` (parent-only dispatch)
