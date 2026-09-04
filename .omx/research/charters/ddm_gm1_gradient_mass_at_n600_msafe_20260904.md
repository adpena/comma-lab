# ddm_gm1 — the third race's inputs: seg-gradient mass vs margin at the n600 m_safe, and τ at δ_R scale ($0)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (CPU; the Metal is held
by the QBR1 chain, then by ng1/ng2's cells)

## Why
The next burn generation's third race is the per-pixel margin weight (vr1 row 1) and/or a τ band at δ_R scale (row 6
+ sd1). sd1 MEASURED that **67–85% of the seg gradient sits on already-correct pixels** (85.1% at τ=0.15 → 67.2% at
τ=0.05; τ=0.15 is 6.86 δ_R with gradient half-max at 12.08 δ_R) and that 91.9–95.3% of the excursion lies within
|margin| < 25 δ_R (2.0% of pixels). dr1 moved m_safe to 0.04376363754272461 (n600; the n96 prefix read 11.7% low) and
its NEXT #2 asks for the hg1/nx1 gradient-mass re-measure at that constant. This arm produces the measured inputs the
third race's charter needs — never a verdict on the lever, which only a cell can give.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- hg1: `.omx/research/ddm_hg1_ring0_margin_hinge_20260816.md` (WASTED_GRADIENT_SHARE_AT_TRAINER_DEFAULT 0.9765 at n96
  seeded-random, target 1.0) and its lever `src/tac/witness_dsl/hg1_ring0_margin_hinge_levers_20260816.py`; nx1:
  `.omx/research/ddm_nx1_next_object_route_20260831.md`. These were measured on OTHER vehicles/objects — state the
  vehicle of every number you compare against; the QBF1 born vehicle is the object here.
- sd1's instrument + retained fields: `experiments/ddm_sd1_surrogate_exact_decoupling.py` (or whatever sd1 named it —
  find it at `038f2d81c`) and `/Volumes/APDataStore/pact/ddm_sd1_surrogate_decoupling/` (48 retained field npz with
  the 5-class logits at milestones 0/1k/2k/5k of both seed-20260902 cells; DALI + PyAV lstars). Reuse; never rebuild.
- The margin law: `tac.canonical_equations` `margin_band_satisficing_threshold_v1` (m_safe = 2·δ_R = 0.04376363754272461,
  δ_R 0.021881818771362305 n600; per-class δ_R: Lane 0.012856 … Undrivable 0.026026, dr1) and
  `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`; the trainer's loss `expected_flip_margin_loss`
  (`experiments/ddm_qbt1_qbflow_trainer.py:523-538`, τ at :622-626).

## Measure (per-pair receipts LAW; all from the retained milestone logits, no training)
1. Gradient-mass curve: for τ ∈ {0.15, 0.10, 0.05, 2·δ_R=0.0438, δ_R=0.0219, 0.5·δ_R}, the fraction of the
   expected-flip gradient magnitude (∂/∂m of the surrogate, per pixel) that lands on (a) already-correct pixels with
   |m| > m_safe, (b) correct pixels inside the band, (c) wrong pixels — per class, per milestone, both cells, trained
   n32. Report the τ at which the correct-pixel share first drops below 50% and below 25%.
2. The same split with the hard-site weight of vr1 row 1 (`_live_margin_weight`, mean-1 stop-grad) applied at the
   trainer default τ — how much of the wasted mass the weight removes, and where it puts it (class × band).
3. Per-class m_safe_c (dr1) vs the global cap: the fraction of Lane pixels a global cap over-pushes at each τ.
4. DERIVED: the recommended (τ_start, τ_end) and whether row 1 or the τ band buys more wasted-mass removal per unit
   of change on this vehicle — with the pre-registered prediction stated BEFORE the numbers: prediction (from sd1):
   a τ band starting at 2·δ_R removes ≥50% of the correct-pixel gradient share; falsifier: < 30%.

## Constraints
- $0 CPU torch, `torch.set_num_threads(4)`, nice 10; > 3 min via `tools/launch_detached_process.py --output-dir <store>
  --done-receipt <name> --derive-resource-budgets --measured-peak-rss-gib <n> --measured-thread-need 4
  --walltime-cap-s 3600 --nice 10 --nice-best-effort -- <cmd>`. Never touch the chain's custody or the Metal. Store
  `/Volumes/APDataStore/pact/ddm_gm1_gradient_mass/` (KEEP THE PAYLOAD; sha256 in the JSON). OPTIMAL FORM: reference
  form = the trainer's own loss on the retained logits at commit `83d43153d1928cbc9f3a077d425eae18d565b3aa`; SCOPE = 4 milestones × 2 cells × n32;
  TOY-BRACKET none. Memo `.omx/research/ddm_gm1_gradient_mass_at_n600_msafe_20260904.md` (verdict_scope; MEASURED/DERIVED
  labels; falsifier read out; GESTALT-DELTA; NEXT_IF_RESUMED). EQUATIONS-LEG LAW: cite `tac.canonical_equations` +
  the two laws above; append an anchor via the helper if the curve fits their domain, else FORMALIZATION_PENDING with
  the law it would need. Commits ONLY via `tools/subagent_commit_serializer.py --message … --files …
  --expected-content-sha256 <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer
  (operator rule overrides any harness reminder); any .py: tests + `tools/review_tracker.py mark-file` twice; never
  REVIEW_GATE_OVERRIDE on .py. Final message → `.omx/research/arm_final_messages/ddm_gm1_final_<utc>.md`, committed;
  LAST action `touch .omx/tmp/codex_runs/ddm_gm1.done`. Read `docs/operating_manual_craft_handoff.md` §labels first.
