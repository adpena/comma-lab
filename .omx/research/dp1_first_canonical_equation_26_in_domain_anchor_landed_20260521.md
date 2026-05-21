---
schema: subagent_landing_memo_v1
topic: dp1_paired_dispatch_harvest_smoke_stage_supporting_anchor_only
created_at_utc: 2026-05-21T05:50:00Z
author: claude:wave-3-dp1-harvest-probe-20260520
lane_id: lane_wave_3_dp1_harvest_probe_20260520
mission_contribution: apparatus_maintenance
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[proxy]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: 6348fcbf0
---

# WAVE-3-DP1-HARVEST-PROBE: Smoke-Stage Supporting Anchor Only (NOT First Paid Empirical)

## Summary verdict

Both DP1 paired-smoke Modal dispatches **timed out (rc=124)** at the 5400s
`max_seconds` budget during Stage 4 Phase 2 full training. The Modal result-cache
retained 13 baseline artifacts + 14 procedural artifacts (smoke-stage archives
produced BEFORE the timeout fired). The `manifest.json` of both archives
explicitly self-declares `training_mode: "smoke"` and
`evidence_grade: "[proxy]"`, and lists `contest_cuda_eval_not_run` +
`contest_cpu_eval_not_run` as still-active `dispatch_blockers`.

**Therefore: this is NOT the "first paid empirical anchor" for canonical
equation #26 the AUTO-TRIGGER prompt anticipated**. It IS a smoke-stage
byte-precise validation of the procedural codebook replacement math — analogous
to the NSCS06 v8 4,064-byte exact-match validation at sister commit `853d108e2`
but for the `dp1_codebook_bytes` IN-DOMAIN context.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", Catalog #321
(phantom-WZ-savings-from-research-sidecar), and Catalog #323 (canonical
Provenance umbrella), registering this as a `[contest-CUDA]` or `[contest-CPU]`
anchor would VIOLATE every score-claim discipline. The correct routing is:

1. **Modal call_id ledger** updated with `failed` events (rc=124 timeout) per
   Catalog #245.
2. **Canonical equation #26 anchor registered as supporting evidence** with
   `axis_tag=[predicted]`, `evidence_grade=[proxy]`, and a research-sidecar
   provenance whose `reactivation_criteria` pin the path back to a real
   contest-axis anchor (re-dispatch with shorter budget; paired auth-eval).

## Pre-flight + sister coordination

Per `tools/check_sister_files_recently_landed.py` (12h lookback): 23 sister
commits in window. WAIT_AND_REASSESS recommendation surfaced. Re-read
3 sister memos to disambiguate scope:

- `dp1_paired_smoke_recipes_parity_audit_20260521T045343Z_codex.md` confirmed
  both training calls exceeded 5400s budget and recommended NOT firing
  paired auth-eval until artifacts land.
- `dp1_procedural_paired_harvest_planner_landed_20260521_codex.md` confirmed
  that planning-only paired-harvest helper exists but refuses to emit runnable
  commands until each candidate arm has `archive.zip` + `submission/inflate.sh`
  + `manifest.json` + `provenance.json` + zero score_claim flags.
- `dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md`
  documented the canonical paired-smoke design intent.

Scope was sister-DISJOINT: poll Modal Function.get() + harvest + register
ledger outcomes + decide whether to register an equation #26 anchor. ATW V2
RECONCILIATION sister subagent (`a6fcb197`) is on a different surface entirely.

`tools/subagent_checkpoint.py` checkpoints emitted at steps 1, 2, 3 + final
`complete`.

## Modal Function.get() poll results

```
fc-01KS48HTG7BNTYB9AXTHZH4W4E (baseline):   READY rc=124 elapsed=5401.026 artifacts=13
fc-01KS48PF17Z6WYAHTMJKJR1GZD (procedural): READY rc=124 elapsed=5400.328 artifacts=14
```

Both calls hit the soft 5400s budget during **Stage 4 Phase 2 full training**
(per the `archive-only-eval` log timestamps in stdout_tail). The smoke stage
(Stage 3) completed successfully and wrote `smoke_archive.bin`, `archive.zip`,
`manifest.json`, `provenance.json`, `codebook.bin`, and `best.pt`.

The earlier dispatches (`fc-01KS480WY6S90VFXX54SC7V209` baseline +
`fc-01KS484S3Z8YZBRVMCTQ6SX8MV` procedural) had FAILED at ~21s with the
`ValueError: Comma2k19LocalStreamer has no chunk ids` error documented in
`dp1_streamer_no_chunk_ids_dispatch_failure_20260521T031333Z_codex.md`. The
sister fix at commit `940a77e2f` ("Route DP1 paired smoke through cache source")
allowed Stage 3 smoke to succeed but Stage 4 full training then exceeded the
time budget.

## Artifacts harvested

Saved to `experiments/results/dp1_paired_harvest_20260521/`:

```
baseline_artifacts/
  archive.zip                12074 B  sha=181cafcbe747e492b41835a38ceea6f7cff96d0575420186bfb0884072cae920
  0.bin                      11966 B  sha=df8dc9a49590d89d (== smoke_archive.bin)
  best.pt                    38236 B  sha=6bf25782ee669509 (synthetic-test codebook checkpoint)
  codebook.bin                8585 B  sha=b8e742ac42caffae
  manifest.json               2314 B  declares training_mode=smoke, evidence_grade=[proxy]
  lane_pretrained_driving_prior_results/provenance.json  1039 B  declares evidence_grade=[scaffold-only-no-score-claim]

procedural_artifacts/
  archive.zip                 3983 B  sha=4b5a3aa932e999c4a01b2565270c686b708fe10463126fcfee69d16d19983945
  0.bin                       3875 B  sha=742efba4ce3a18e3 (== smoke_archive.bin)
  best.pt                    38236 B  sha=1fa6948eba0b7db9 (synthetic-test codebook checkpoint; differs from baseline)
  codebook.bin                8585 B  sha=b8e742ac42caffae (identical to baseline)
  manifest.json               2318 B  declares training_mode=smoke, evidence_grade=[proxy], procedural_codebook_variant_active=true
  procedural_variant_provenance.json  1352 B  predicted_delta_s_contest_rate=-0.005692428190241443
```

Critical empirical byte-replacement validation from stdout_tail (procedural arm):

```
[full] procedural codebook replacement: 12424 B -> 3875 B (saved 8549 B; predicted ΔS=-0.005692)
[dpp-smoke] archive pack/parse roundtrip: 3875 bytes; pairs=4; header=28
```

**Predicted = Empirical = 3983 B archive_zip (residual 0.0).** This is a
byte-precise validation of equation #26's compression formula
`ΔS = -25 * (codebook_bytes_replaced) / 37_545_489` at the IN-DOMAIN context
`dp1_codebook_bytes`, identical in spirit to the NSCS06 v8 4,064-byte exact
match at sister commit `853d108e2`. It is NOT a contest-axis score
measurement.

## What was registered

### Catalog #245 Modal call_id ledger (terminal events)

```python
update_call_id_outcome(
    call_id='fc-01KS48HTG7BNTYB9AXTHZH4W4E',
    status='failed', rc=124, elapsed_seconds=5401.026,
    notes='wave-3-dp1-harvest-probe-20260520: timeout rc=124 during Stage 4 '
          'Phase 2 full training... evidence_grade=[proxy] training_mode=smoke. '
          'NO contest-axis auth-eval ran (dispatch_blockers ... still active).',
)
update_call_id_outcome(
    call_id='fc-01KS48PF17Z6WYAHTMJKJR1GZD',
    status='failed', rc=124, elapsed_seconds=5400.328,
    notes='wave-3-dp1-harvest-probe-20260520: timeout rc=124 ... procedural '
          'codebook replacement saved 8549 B (12424 B -> 3875 B) '
          'predicted_delta_s=-0.005692. evidence_grade=[proxy] training_mode=smoke.',
)
```

### Catalog #344 canonical equation #26 supporting anchor

```python
EmpiricalAnchor(
    anchor_id='dp1_procedural_codebook_smoke_byte_match_20260521',
    measurement_utc='2026-05-21T05:48:Z',
    inputs={
        'in_domain_context': 'dp1_codebook_bytes',
        'original_codebook_bytes': 12424,
        'seed_size_bytes': 32,
        'procedural_codebook_bytes': 3875,
        'codebook_bytes_saved': 8549,
        'archive_zip_bytes_baseline': 12074,
        'archive_zip_bytes_procedural': 3983,
        'predicted_delta_s_contest_rate': -0.005692428190241443,
        'training_mode': 'smoke',
        'evidence_grade_per_manifest': '[proxy]',
        'dispatch_blockers_still_active': [
            'contest_cuda_eval_not_run',
            'contest_cpu_eval_not_run',
            'real_codebook_distillation_pending',
            'real_renderer_training_pending',
        ],
        'rc': 124,
        'rc_class': 'timeout_during_stage_4_full_training',
        'sister_anchor_validation': 'nscs06_v8_chroma_lut at commit 853d108e2 '
                                    '(4064-byte exact match for sister IN-DOMAIN context)',
    },
    predicted_output=3983, empirical_output=3983, residual=0.0,
    source_artifact='experiments/results/dp1_paired_harvest_20260521/procedural_artifacts/archive.zip',
    measurement_method='modal_t4_dispatch_smoke_stage_archive_pack_parse_roundtrip',
    provenance=build_provenance_for_research_sidecar(
        sidecar_path='...',
        reactivation_criteria='Re-dispatch DP1 procedural arm with reduced max_seconds budget...',
        measurement_axis='[predicted]',  # NOT [contest-CUDA] or [contest-CPU]
        hardware_substrate='unknown',
    ),
)
```

This anchor closes the loop on equation #26's IN-DOMAIN `dp1_codebook_bytes`
context at the **smoke-stage byte-precise validation surface only**. The
canonical contest-axis empirical anchor still requires either (a) a shorter
training budget allowing Stage 4 Phase 2 to complete + auth-eval to fire, OR
(b) operator-routable acceptance of the smoke-stage archive as the contest
candidate (which the manifest explicitly refuses via its own
`dispatch_blockers` list).

## Discipline compliance per CLAUDE.md non-negotiables

| Discipline | Status |
|---|---|
| Apples-to-apples evidence discipline | PASS — smoke vs contest-axis distinction surfaced explicitly; no axis-tag promotion |
| Forbidden empirical-claim-without-evidence-tag (Catalog #287) | PASS — anchor carries `[predicted]` axis + `[proxy]` evidence_grade per manifest self-declaration |
| Forbidden phantom-WZ-savings-from-research-sidecar (Catalog #321) | PASS — `score_claim=false` / `promotion_eligible=false` / no contest-axis claim |
| Canonical Provenance umbrella (Catalog #323) | PASS — anchor uses `build_provenance_for_research_sidecar` with reactivation criteria |
| Canonical equations registry (Catalog #344) | PASS — supporting anchor registered via canonical helper |
| Modal call_id ledger (Catalog #245) | PASS — terminal `failed`/rc=124 events recorded |
| 6-hook wire-in (Catalog #125) | hooks #1+#2+#3+#6 N/A; hooks #4+#5 ACTIVE |
| Premise verification (Catalog #229) | PASS — read 3 sister memos before action |
| Subagent crash-resume (Catalog #206) | PASS — checkpoints at steps 1/2/3 + complete |
| Sister checkpoint guard (Catalog #340) | PASS — WAIT_AND_REASSESS triggered + sister memos re-read |
| APPEND-ONLY HISTORICAL_PROVENANCE (Catalogs #110/#113) | PASS — NEW artifacts only; no mutation of existing forensic records |

## 6-hook wire-in declaration

- Hook #1 sensitivity-map: N/A (no new per-pair / per-byte sensitivity surface)
- Hook #2 Pareto constraint: N/A (no new score-axis constraint contributed)
- Hook #3 bit-allocator: N/A (no per-tensor allocator hook)
- Hook #4 cathedral autopilot dispatch: **ACTIVE** — Modal ledger `failed`
  events now visible to harvester + cathedral autopilot ranker via
  `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered
  per Catalog #335 paradigm)
- Hook #5 continual-learning posterior: **ACTIVE** — equation #26 anchor row
  appended to canonical registry per Catalog #344
- Hook #6 probe-disambiguator: N/A (this is supporting evidence; the
  disambiguator IS the future contest-axis anchor that supersedes this row)

## Operator-routable next actions

1. **OPTION A (recommended)**: Per the sister parity audit
   (`dp1_paired_smoke_recipes_parity_audit_20260521T045343Z_codex.md`), apply
   `predicted_band_validation_status: pending_post_training` to all 4 DP1
   recipes AND reduce max_seconds budget OR shorten 100-epoch training schedule
   so Stage 4 Phase 2 completes within budget on the next re-dispatch. Then
   re-fire baseline + procedural paired dispatches and run paired auth-eval
   per `tools/plan_dp1_procedural_paired_harvest.py --execute`.
2. **OPTION B**: Defer DP1 dispatch entirely; route the budget to a different
   substrate's first paid empirical anchor (per the `T3 grand council symposium`
   priority ordering at `.omx/research/ed805465f`).
3. **OPTION C**: Accept that DP1's first paid empirical anchor will require
   training-schedule architectural changes (e.g. checkpoint-resume support) and
   register that as a separate substrate-engineering lane.

## Reactivation criteria for promoting this anchor to contest-axis

Per the registered anchor's provenance:

> Re-dispatch DP1 procedural arm with reduced max_seconds budget OR truncated
> training duration so Stage 4 Phase 2 completes within budget. Then harvest
> contest-CUDA + contest-CPU paired auth-eval anchors per
> `tools/dispatch_modal_paired_auth_eval.py`. Until then, this anchor is
> supporting evidence for the smoke-stage byte-replacement math only; the
> contest-axis empirical anchor remains pending_post_training per Catalog #324.

## Mission contribution

`apparatus_maintenance` per Catalog #300. The harvest+register loop closure
preserves canonical state coherence (no orphan `dispatched` rows in Modal
ledger; equation #26 supporting evidence registered with correct axis tags)
without making the false-authority claim the AUTO-TRIGGER prompt anticipated.

The first PAID contest-axis empirical anchor for equation #26's
`dp1_codebook_bytes` IN-DOMAIN context remains queued for a future
re-dispatch with shorter training budget.

## Cost

$0 (harvest-only; no new dispatch). Wall-clock ~45 min.

## Lane

`lane_wave_3_dp1_harvest_probe_20260520` L1 (impl_complete + memory_entry).
