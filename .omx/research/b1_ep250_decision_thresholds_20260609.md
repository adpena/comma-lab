# B1 ep-250 first-exact-score decision thresholds (2026-06-09)

UTC: 2026-06-09 · agent HARVEST-AUTOMATION · lane `lane_harvest_automation_20260609`.
Binding companion to `.omx/research/b1_baseline_discipline_operator_directives_20260609.md`
(the 8 binding points) + `.omx/research/b1_muon_curriculum_verification_criteria_20260609.md`
(the KILL/RESTART conditions). This memo is the POLICY for interpreting the
FIRST exact backend-only HiNeRV score the moment the ep-250 checkpoint is
harvested by `tools/watch_and_harvest_b1_checkpoint.py`.

Operator spine: **"the first exact score beats all optimizer speculation."**
The ep-250 harvest exists to turn that score into a real number FAST, then act on
the TREND — not to over-react to a single early-curriculum measurement.

## Evidence-axis discipline (read first)

- The harvester's B2 score is **`[macOS-CPU advisory]`** — it proves the
  export→inflate→evaluate pipeline runs end-to-end AND gives the first real
  number AND the d_seg/d_pose/rate decomposition AND the trend. It is **NOT
  authoritative** and **NOT promotable** (`score_claim=false`).
- The **authoritative** score requires **Linux-x86_64 CPU (`[contest-CPU]`,
  the public-leaderboard axis)** AND **NVIDIA T4 CUDA (`[contest-CUDA]`, the
  promotion axis)** on the SAME archive bytes (paid + deferred to operator;
  recipe: `tools/run_hi_nerv_backend_only_b2_exact_eval.py --print-dual-axis-recipe`).
- Frontier to beat = the canonical pointer's LOCAL FRONTIER (read via
  `tools/refresh_canonical_frontier.py`; do NOT hardcode the literal — as of
  this writing the pointer's local frontier is the 0.19199 `[contest-CPU]`
  value the operator referenced; the leaderboard ranks by contest-CPU). All
  ep-250 advisory comparisons are macOS-CPU-vs-pointer and are directional only
  until the paid Linux-CPU axis lands.

## Why ep-250 is EARLY (do not over-react)

The pilot runs the PR95 8-stage curriculum (29,650-epoch full schedule)
**compressed into a 3000-epoch reduced pilot** (`--research-curriculum-total-epochs
3000`). At ep-250 the run is ~8% through the reduced schedule — **early stage-1
(ce_seg) territory**. Concretely, at ep-165 the live PROXY decomposition was
`seg≈1.11, pose≈6.19, archive_bytes≈0.0` (rate not yet charged — QAT / byte
pressure are LATER stages). Therefore:

- **The ep-250 archive is very likely PRE-QAT and PRE-byte-pressure.** A high
  byte count or high score at ep-250 is EXPECTED and is NOT a kill signal — the
  rate-shaping stages (QAT / C1a / σ / λ sweeps) have not run yet.
- The exact score at ep-250 establishes (a) the pipeline works, (b) the
  d_seg/d_pose/rate STARTING decomposition, (c) the SegNet-chamber-entry signal
  (is d_seg finite and improving, or is SegNet collapsed?).

## The decision table (what to do per ep-250 outcome)

Priority order: **pipeline correctness > gate violations (HARD FAIL) > score
trend interpretation**. Use the EXACT contest score `100*d_seg +
sqrt(10*d_pose) + 25*bytes/37_545_489`, NEVER a linear seg/pose tradeoff.

### A. Pipeline FAILS (export / inflate / evaluate errors)

- **Symptom:** harvester writes `harvest_status_ep250.json` state
  `failed_harvest_exception`, OR the B2 out-row verdict is
  `contest_auth_eval_failed_*` / `inflate_*` error, OR the result JSON's
  `first_exact_score_advisory` is null with a non-OK verdict.
- **Action:** PATCH the B2 bridge / inflate runtime / export path. **Keep B1
  running** if its heartbeat is healthy — a pipeline bug is NOT a model failure.
  Common culprits (de-risked in P3): dep-closure (`brotli+torch+numpy` — already
  satisfied in `.venv`), inflate.sh `${PYTHON}` wiring, num_pairs!=600 (the
  archive carries 600 pairs so this should not fire), member-name (`x` vs
  `0.bin`). Re-run the harvester after the patch (idempotent: it re-exports +
  re-evals because the result JSON does not yet exist).
- **Rationale:** "first exact score" is blocked by tooling, not by the model;
  fixing tooling is highest-EV and does not waste the running pilot.

### B. `sidecar_exported == true` without `pays_rent == true` → HARD FAIL

- **Symptom:** the exported archive has >1 member OR the export/telemetry shows a
  sidecar that did not pass the pay-rent gate. (Manifest contract:
  `sidecar_export_enabled=false`, `pay_rent_gate_active=true`. The export fn
  `export_hi_nerv_mlx_archive` drops unproven sidecars at the boundary, so this
  should NOT happen — if it does, the gate regressed.)
- **Action:** **KILL/RESTART** per the muon-curriculum verification criteria
  (`hard_stop: sidecar_exported_without_pays_rent`). Patch the export gate; the
  archive MUST ship backend-only. This is the canonical extinction of the
  2026-06-08 harmful-sidecar incident.
- **Harvester check:** the result JSON's `export.zip_members` MUST be `["x"]`
  and `export.member_magic` MUST be `"HIV1"`. Anything else is this failure.

### C. `muon_active == true` before stage 8 → HARD FAIL

- **Symptom:** telemetry rows in stages 1-7 show `muon_active=true`, OR the
  stage-8-only Muon partition leaked into earlier stages. (Manifest:
  `stage8_use_muon_flag=true`, `stage8_muon_status=WIRED_AND_VALIDATED`. PR95 is
  an 8-stage curriculum ENDING in Muon, NOT a black-box global Muon train.)
- **Action:** **KILL/RESTART** per `b1_muon_curriculum_verification_criteria`
  (`muon_active==true in ANY of stages 1-7 → KILL/RESTART`). Note: at ep-250
  (early stage 1) Muon MUST be inactive; `muon_active` should be `false`/`None`.
  If the latents / entropy / QAT params show up in the Muon group, that is a
  PATCHABLE stage-8 partition fix (safe — stage 8 is last; ~the rest of the run
  remains to patch; the run is resumable), NOT a restart.

### D. Catastrophically bad exact score (e.g. >> frontier) → DO NOT auto-kill

- **Symptom:** `first_exact_score_advisory` is large (e.g. > 1.0, or many× the
  frontier).
- **Action:** **DO NOT auto-kill.** ep-250 is early-curriculum + likely pre-QAT.
  Instead INSPECT the d_seg / d_pose / rate decomposition in the B2 out-row:
  - **SegNet collapsed** (d_seg saturated high, e.g. the C6-IBPS-class
    `score_seg` dominating ~85%+): this is the HiNeRV hard blocker
    (target-region class birth surviving uint8/resize/parse-back). If d_seg is
    collapsing AND the proxy `seg` trend in telemetry is NOT descending across
    ep-250→ep-500, that is a SOFT-STOP review signal (per
    `soft_stop_review: proxy_score_not_improving_after_stage1`), not an auto-kill.
  - **rate huge but pre-QAT:** expected; ignore until the QAT/byte-pressure
    stages run (re-harvest at a later checkpoint to see post-QAT bytes).
  - **Compare to the PROXY trend:** the live proxy decomposition (via
    `tools/read_b1_pilot_status.py`) should be descending. If exact >> proxy
    suggests, suspect an mlx↔torch scorer-parity drift
    (`soft_stop_review: mlx_torch_scorer_parity_drift`) — a measurement axis
    issue, not necessarily a model failure.
- **Rationale:** CLAUDE.md "Forbidden premature KILL" — a single early-config
  exact number is DEFERRED-pending-research, never a kill.

### E. d_seg improves but d_pose worsens → decide by EXACT ΔS (not pose fear)

- **Symptom:** between checkpoints (or vs a baseline) SegNet improves while
  PoseNet regresses.
- **Action:** compute the EXACT contest-score delta using the **nonlinear
  sqrt-pose term** `sqrt(10*d_pose)`, NOT a linear seg/pose tradeoff. Per the
  manifest's `marginal_flip_note` + CLAUDE.md "SegNet vs PoseNet importance":
  at LOW pose (`pose_avg < ~2.5e-4`) the pose marginal `d/d(pose) = 5/sqrt(10*pose_avg)`
  can EXCEED the seg marginal (constant 100). So:
  - If the net exact ΔS (seg gain − sqrt-pose loss) is NEGATIVE (score drops):
    GOOD, continue (`continue: pose_regression_within_exact_score_value`).
  - If net exact ΔS is POSITIVE (score rises): SOFT-STOP review
    (`soft_stop_review: seg_proxy_improves_but_pose_proxy_worsens_beyond_exact_score_tradeoff`).
  - At ep-250's operating point (pose_avg likely still high, ~0.x), the seg term
    typically dominates — but VERIFY with the exact decomposition; do not assume.
- **Rationale:** the contest objective is the exact formula; pose fear without
  the sqrt-term math is cargo-culted.

### F. Bytes too high → inspect rate / C1a / QAT stage state

- **Symptom:** `25*bytes/37_545_489` term is large; `export.archive_bytes` near
  or above the `hard_byte_ceiling` (300000).
- **Action:** check whether the checkpoint is PRE-QAT (ep-250 likely is). The
  byte-shaping levers (coder-QAT, C1a entropy weight, σ/λ sweeps, decoder
  pruning/quant-noise) run in LATER stages. If pre-QAT, high bytes are EXPECTED
  — re-harvest a later checkpoint (ep-500/750/...) to observe post-QAT bytes.
  Only if bytes remain high AFTER the QAT/rate stages is this a real rate
  blocker → inspect the C1a entropy weight + decoder-codec + waterfill plan.
- **Reference:** the export uses `decoder_codec=int8_mixed`,
  `hard_byte_ceiling=300000` (the pilot defaults). A fresh-init export probe
  measured ~265 KB; a trained-but-pre-QAT export will differ.

### G. Promising (exact score near or below frontier trend) → continue + prep paid axes

- **Symptom:** `first_exact_score_advisory` is in a plausible trajectory toward
  the pointer's local frontier (read via `tools/refresh_canonical_frontier.py`),
  with d_seg finite + descending and d_pose controlled.
- **Action:** **continue to ep-3000** (the pilot is healthy; let the curriculum
  run) AND **prepare the authoritative dual-axis replay** for the END archive
  (and optionally the best-EMA checkpoint): the paid Linux-x86_64 CPU
  (`[contest-CPU]`) + T4 CUDA (`[contest-CUDA]`) recipe via
  `--print-dual-axis-recipe`, gated on operator approval + a lane claim. Only
  the paid axes can claim a frontier move.
- **Rationale:** advisory-promising is the green light to spend the paid axis,
  not a frontier claim by itself.

## The burning question this harvest answers

Does the ep-250 trend show the 229K PR95-faithful curriculum on a trajectory to
**beat the pointer's local frontier** (read live; ~0.19199 `[contest-CPU]`), or is
the binding constraint one of:

1. **rate** — bytes dominate (but pre-QAT at ep-250, so re-check post-QAT);
2. **Pose** — `sqrt(10*d_pose)` term large (decide by exact ΔS, not pose fear);
3. **SegNet-chamber-entry** — d_seg collapsed / not entering the target-region
   birth chamber (the HiNeRV hard blocker)?

The intervention BEFORE the 17h pilot completes follows the binding rule:
**pipeline bugs → patch + keep running; gate violations (sidecar / early-Muon)
→ KILL/RESTART or patch-the-partition; bad-but-early score → DEFER + inspect
the decomposition; promising → continue + prepare the paid dual-axis replay.**
The first exact number + its trend — not optimizer speculation — drives the next
move.

## Operator-facing commands

```bash
# Live pilot status (heartbeat, epoch, stage, sec/epoch, ETA, gates, harvest):
.venv/bin/python tools/read_b1_pilot_status.py \
    --run-dir /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z

# Harvester status + result (after ep-250 lands; auto-written by the watcher):
cat /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/harvest_status_ep250.json
cat /Volumes/VertigoDataTier/pact/b1_229k_pilot_20260609T055851Z/hi_nerv_backend_only_ep250_exact_eval.json

# Authoritative paid dual-axis recipe (deferred; operator-gated):
.venv/bin/python tools/run_hi_nerv_backend_only_b2_exact_eval.py --print-dual-axis-recipe

# Current canonical frontier (do NOT hardcode the literal):
.venv/bin/python tools/refresh_canonical_frontier.py
```

## Provenance / wire-in

- Harvester: `tools/watch_and_harvest_b1_checkpoint.py` (durable, detached,
  idempotent, fail-closed, memory-aware, disk-hygienic). Tests:
  `src/tac/tests/test_watch_and_harvest_b1_checkpoint.py` (20 NO-FAKE).
- Status reader: `tools/read_b1_pilot_status.py`. Tests:
  `src/tac/tests/test_read_b1_pilot_status.py` (12 NO-FAKE).
- B2 bridge (read-only consumer): `tools/run_hi_nerv_backend_only_b2_exact_eval.py`
  + runbook `docs/b2_hi_nerv_backend_only_exact_eval_bridge_runbook.md`.
- This memo is `research_only=true` (a decision-policy bridge artifact);
  it changes the NEXT operator action on the ep-250 harvest, satisfying the
  "results must become system intelligence" wire-in via the harvester's
  machine-readable result+status JSON that this policy consumes.
```
research_only: true
evidence_grade: [macOS-CPU advisory] (harvest) / policy memo
lane_id: lane_harvest_automation_20260609
```
