---
council_tier: T1
council_attendees: [Claude-subagent-cable_b1_e7_e8_dispatch]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "T4 14.56GB capacity sufficient for SegNet stride-2 stem at 384x512 batch=16"
    classification: CARGO-CULTED
    rationale: "Empirical first dispatch OOM'd at SegNet BatchNorm forward at 141s; recipe's min_vram_gb=14 declared barely-equal to T4 capacity left zero headroom; VQ codebook activation pushed past the threshold; Catalog #218 D4 OOM sister anchor confirmed same bug class"
  - assumption: "VQ K-sweep at K=16 default is informative for the Wave 2A K=2 hypothesis"
    classification: HARD-EARNED
    rationale: "Council T3 Finding 1 directive specifies K in {2, 4, 8, 16, 32, 64, 256} sweep; K=16 default is mid-range smoke; if A10G retry completes successfully, single K=16 anchor is a one-point smoke (not the full sweep — sweep requires 8 separate dispatches per env_overrides parameterization)"
council_decisions_recorded:
  - "op-routable #1: T4 -> A10G GPU upgrade via MODAL_GPU=A10G env override (50% more memory headroom 22GB vs 14.56GB)"
  - "op-routable #2: per Catalog #313 sister-probe-alternative-reducer (#308), DEFER probe outcome from T4 OOM is RATIFIED-SUPERSEDED by PROCEED probe outcome for A10G retry"
  - "op-routable #3: this dispatch is a SINGLE K=16 anchor at lambda=1.0; full 8-K sweep requires 7 additional dispatches (~$2.10 incremental on A10G)"
  - "op-routable #4: after A10G K=16 verdict lands, fan-out remaining K in {2, 4, 8, 32, 64, 128, 256} per council T3 Finding 1 directive"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_breaking
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 2.10
finding_canonical_path: experimental
---

# E.7 VQ-VAE K-sweep dispatch verdict (Cable B1)

**Status:** PROCEED_WITH_REVISIONS pending A10G retry completion + 7-dispatch fan-out per council T3 Finding 1

**Cable B1 lane:** `lane_cable_b1_e7_e8_combined_dispatch_20260519` (operator-frontier-override ratified per Catalog #300)

## Empirical anchors

### Dispatch attempt 1 (T4): `fc-01KRZC53Y0D28B6BYEQ1MRG347` — FAILED OOM

- 2026-05-19T05:42:44Z — Modal T4 dispatch fired
- 2026-05-19T05:45:10Z — trainer_invoke_begin (100 epochs at batch_size=16, K=16, lambda=1.0)
- 2026-05-19T05:47:25Z — rc=1 at 141s elapsed
- Failure: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 37.81 MiB is free`
- Failure site: SegNet (smp.Unet EfficientNet-B2) stride-2 stem BatchNorm forward at 384x512x16
- Root cause per Catalog #218 sister: SegNet decoder pre-saturates T4 14.56GiB capacity at 384x512 batch=16 BEFORE VQ codebook activation; min_vram_gb=14 declared barely-equal left zero headroom
- DEFER probe outcome registered: `vq_vae_k_sweep_dispatch_attempt_t4_oom_at_segnet_batchnorm_20260519`

### Dispatch attempt 2 (A10G): `fc-01KRZCX15GAF5Z5E3E568Q60FF` — RUNNING (status pending)

- 2026-05-19T05:55:56Z — Modal A10G dispatch fired (via MODAL_GPU=A10G env override)
- Per Catalog #308 sister-probe-alternative-reducer: same trainer, same recipe, same K=16, same lambda=1.0 — only GPU class changed (A10G 22GB vs T4 14.56GB)
- PROCEED probe outcome registered: `vq_vae_k_sweep_dispatch_a10g_retry_ratified_20260519`
- Pending: verdict to be appended when terminal status reached

## Per CLAUDE.md "Forbidden premature KILL" verdict

The T4 OOM is INFRASTRUCTURE-LEVEL falsification (specific GPU memory class), NOT PARADIGM-LEVEL falsification (van den Oord 2017 VQ-VAE intact). Per Catalog #307 paradigm-vs-implementation classification.

## Council T3 Finding 1 directive recap

Per `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md` op-routable #1:

> "dispatch K-sweep paired-comparison smoke ($1-3 Modal T4) on D4 or sane_hnerv: K in {2, 4, 8, 16, 32, 64, 256} at lambda=1.0 to empirically validate the Pareto pole"

The PREP subagent's recipe is parameterized for SEQUENTIAL 8-dispatch fan-out via `VQ_VAE_CODEBOOK_SIZE` env var sweep. The current single dispatch is K=16 (smoke default). After A10G K=16 verdict lands:

- IF K=16 succeeds → fan out K ∈ {2, 4, 8, 32, 64, 128, 256} via 7 additional `MODAL_GPU=A10G VQ_VAE_CODEBOOK_SIZE=<K>` dispatches (~$0.30 each on A10G = ~$2.10)
- IF K=16 fails again → investigate trainer-level memory bug (mini-batch reconstruct per Catalog #218; or batch_size reduction; or gradient checkpointing)

## Per-substrate symposium continuity

The Cable B1 operator-frontier-override per Catalog #300 ratified the symposium DRAFT to PROCEED. The T4 OOM + A10G retry is a per-dispatch infrastructure verdict, NOT a symposium-level deferral. The symposium ratification remains valid; per Catalog #325 14-day window the next dispatch (and the 7-fan-out follow-on) are admissible.

## Continual-learning anchor

Per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2 frontmatter, this T1 working-group anchor will be persisted via `tac.council_continual_learning.append_council_anchor` so the downstream autopilot ranker / Rashomon ensemble / next-iteration council see this verdict.

## Cross-references

- Operator-frontier-override memo: `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- E.8 sister verdict: `.omx/research/e8_sgld_convergence_dispatch_verdict_20260519T060000Z.md` (DEFER)
- T4 DEFER probe outcome: `vq_vae_k_sweep_dispatch_attempt_t4_oom_at_segnet_batchnorm_20260519`
- A10G PROCEED probe outcome: `vq_vae_k_sweep_dispatch_a10g_retry_ratified_20260519`
- Catalog #218 sister anchor: D4 OOM fix mini-batch (same bug class at sister substrate)
- Catalog #178 TF32: VQ trainer routes through trainer_skeleton.device_or_die per phase 1b commit 220c207ed
- Council T3 Finding 1: `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`
- Wave 2A R-D analytical anchor: commit `8b987215a` rows #2+#3 (K=2 Pareto-frontier optimum hypothesis)
- HNeRV parity L7 (substrate-engineering vs bolt-on): VQ K-sweep is substrate-engineering (architectural sweep), bolt-on size budget does not apply

## Mission-alignment classification

**`predicted_mission_contribution: frontier_breaking`** — IF A10G K=16 succeeds AND K=2 sweep arm validates the Wave 2A R-D Pareto-pole hypothesis, this is a class-shift signal (per-cell bit rate = 1-8 bits across the sweep; K=2 = 1 bit per cell IS the minimum-rate canonical extreme). The Pareto-pole validation would inform VQ-VAE substrate architectural decisions across the contest.

— Cable B1 E.7+E.8 combined dispatch subagent
2026-05-19T06:00:00Z (pre-A10G verdict; will be appended when terminal status reached)
