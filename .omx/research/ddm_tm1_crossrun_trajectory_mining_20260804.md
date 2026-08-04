# ddm_tm1 cross-run telemetry + trajectory mining - 2026-08-04

Status: MEASURED zero-scorer artifact mining. No training launch, no n600 scorer job, no upstream
mutation. Evidence axis: persisted run telemetry / receipts only; score_claim=false.

## Instrument + Denominator

Reusable apparatus landed:

- `src/tac/crossrun_trajectory_mining.py` - typed frame harvester + smooth reducers.
- `tools/harvest_crossrun_trajectories.py` - thin CLI; can harvest roots or re-analyze existing frames.
- `src/tac/tests/test_crossrun_trajectory_mining.py` - fixture coverage for telemetry, loss terms,
  gates, config, and markdown JSON fences.

Full command:

```bash
.venv/bin/python tools/harvest_crossrun_trajectories.py \
  --root experiments/results \
  --root .omx/research \
  --root /Volumes/VertigoDataTier/pact \
  --frames-jsonl /Volumes/VertigoDataTier/pact/ddm_tm1_20260804/ddm_tm1_crossrun_frames_20260804.jsonl \
  --summary-json .omx/research/ddm_tm1_crossrun_summary_20260804.json \
  --max-bytes-per-file 8000000 \
  --max-records-per-file 50000
```

Denominator:

- Candidate files considered: 26,749.
- Parsed files: 26,687.
- Skipped files: 62, all `too_large`.
- Parsed records: 360,012.
- Bad records: 1,005.
- Frames emitted: 670,186 across 4,818 run IDs.
- Frame artifact: `/Volumes/VertigoDataTier/pact/ddm_tm1_20260804/ddm_tm1_crossrun_frames_20260804.jsonl`,
  534,677,413 B, SHA-256 `c2f0ebd0892a307949e0be617421344251aa31c26c00e1b5349ed52a8f5b1985`.
- Summary: `.omx/research/ddm_tm1_crossrun_summary_20260804.json`, 103,027 B,
  SHA-256 `200ae1e6ac137e27f700d155a2fc08793fb439d9b0d3ec02fff0d99f507e5ead`.
- Bulk custody: `.omx/research/ddm_tm1_full_frame_coldstore_manifest_20260804.json`.

A previous capped/interrupted local frame dump was certified and deleted; the capped SSD artifact is
renamed `ddm_tm1_crossrun_frames_cap10000_20260804.jsonl` and is NOT the evidence basis here. See
`.omx/research/ddm_tm1_interrupted_harvest_cleanup_20260804.json` and
`.omx/research/ddm_tm1_frame_coldstore_manifest_20260804.json`.

## The Eight Questions

| Q | Answer | Consumer disposition |
|---|---|---|
| 1. Per-class / per-edge convergence laws | DATA-INSUFFICIENT for the requested law. The harvest found 163 run IDs with aggregate `d_seg` or `realized_gate_dseg_mean` series and 700 topology-per-class series, but not true per-class `d_seg(t)` or per-edge `d_seg(t)` across the requested v5/v6/v7/v9/TR1/burn/r1c population. | QUEUED-WITH-FIRE-ORDER: add `per_class_d_seg`, edge/component id, and denominator to A1/verdict telemetry under the #420 contract before deriving burn-4 stage-length LawRefs. |
| 2. Lever impulse responses | MEASURED event denominator: `lane_guard` 229, `lane_guard_init` 29, `resume_form_reanchor` 39, `telemetry_v9_port` 27. 210 events had enough before/after loss samples. No lever type is trajectory-proven inert on this corpus. | FOLDED into lever registry activation ledger as `inert_lever_count=0`; QUEUED-WITH-FIRE-ORDER to require pre/post comparable metric windows for future lever events. |
| 3. Warm-vs-fresh basin geometry | MEASURED 46 resume events; 43 had enough post-resume loss samples. Burn-4 windows half-descended in 2, 9, and 2 epochs with exponential-floor fits. This validates fast warm re-descent as a phenomenon, but not #518 beta2 quantitatively because beta2/horizon metadata is not consistently present. | QUEUED-WITH-FIRE-ORDER for gc15: record beta2, horizon, resume source stage, and post-resume objective in the same #420 row. |
| 4. Plateau census | MEASURED A1 classifications: FIRST_GATE 193, COUPLED_DESCENT 253, FLAT 111, A1_REALIZATION_GAP_ALARM 53. Detected 69 near-flat gate-to-gate `realized_gate_dseg_mean` segments at abs(rel delta) <= 0.005; 2 carry a confound alarm, 67 remain unclassified by available joins. | QUEUED-WITH-FIRE-ORDER for na3: join skip-rate/spike-guard/cap-stop ledgers by run_id+epoch so the 67 unclassified plateaus become typed, not prose. |
| 5. Memory envelope law | DATA-INSUFFICIENT for a fit. The harvest found 5,761 memory frames and a config join, but the units are mixed (`mlx_peak_memory_bytes`, `rss_gib`, `peak_rss_bytes`, free-memory floors). This is a measurement corpus, not a governor law yet. | QUEUED-WITH-FIRE-ORDER for #294: standardize `peak_rss_bytes`, `peak_vram_bytes`, batch/config keys, and host axis in the #420 run artifact. |
| 6. Event-timing backcast | MEASURED A1 cadence: 69 runs with `cfg.gate_every` and >=2 A1 gates; 58 match exactly. Nonmatches are dominated by terminal/stage-boundary partial gaps. | FOLDED into burn-4 thresholds: event cadence is trustworthy when terminal partial gaps are excluded; no threshold amendment landed. |
| 7. Stopping-policy validation | DATA-INSUFFICIENT. Found 18 stop/decision events, but no explicit stop point paired with same-object counterfactual continuation after the stop. | QUEUED-WITH-FIRE-ORDER for E2/#848: every stop decision row must carry `stop_epoch`, objective, reason, and a continuation-or-no-continuation receipt. |
| 8. Seg-constancy autopsy | MEASURED, FORMULATION-scoped. 43/163 seg series were constant within 2%. Burn-4 `realized_gate_dseg_mean` stayed in a narrow band despite loss descent: window_01 span 1.33%, window_02 5.87%, window_03 9.50%. But activation-screen runs descended strongly to low floors, so history does NOT support a global actuator-class ceiling. It supports a burn/TR1 gate-level constancy/realization-wall phenomenon. | FOLDED into GC16-R2 routing: distinguish loss descent from realized-gate d_seg movement; aim carrier-vs-burn decisions at rows where realized d_seg bends. |

## Smooth Forms

The reducer fits smooth exponential-floor families of the form
`y = floor + amplitude * exp(-k * (x - x0))`; no lookup table or piecewise schedule is emitted.

Measured examples:

- `ddm_b4s_20260731::window_01`: post-resume `ep_loss` half-descent 2 epochs, `k=0.3316`,
  rmse/span 0.0970.
- `ddm_b4s_20260731::window_02`: half-descent 9 epochs, `k=0.02537`, rmse/span 0.0867.
- `ddm_b4s_20260731::window_03`: half-descent 2 epochs, `k=0.02885`, rmse/span 0.2349.
- Activation screens: FINER/SIREN/WIRE n100 show aggregate `d_seg` dropping by 99.2-99.4% with
  exponential-floor fits; these are not the burn/TR1 wall.

## Canonical Equations / LawRef

No new canonical equation was registered in this landing. Reason: no multi-anchor finding above met all
three gates simultaneously: stable domain, predictive law form, and train-time consumer with the required
inputs already present. The measured warm-resume and seg-constancy patterns are FORMULATION-scoped and
consumer-routed, but the missing beta2/per-class/edge/counterfactual fields would make a registry row
look more authoritative than the evidence. This is an explicit no-register verdict, not a forgotten step.

## Boundaries

- New scorer evaluations: 0.
- Training launches: 0.
- `upstream/`: untouched.
- `/tmp` evidence: none.
- Local bulk: none retained; full frame JSONL is on SSD.
- Own-vehicle frontier: unchanged at S = 0.7541459 @ 358,084 B [macOS-CPU advisory]; contest pointer
  0.1910828242 remains borrowed/harvest-only and unmoved.
