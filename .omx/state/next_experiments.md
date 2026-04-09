# next experiments

## 2026-04-09 queue after PSD faithful proxy resolution

The promoted honest floor is still `1.73` from `long1000_h64`. Two deploy-ready alternates have now been resolved honestly and rejected:

- `pixelshuffle_h64_long1000` -> faithful proxy `1.99`
- `psd_h64_long1000` -> faithful proxy `1.85`

That means the next cycle should stop pretending those families are active promotion candidates in their current form.

## cycle budget

1. Prefer at most **3** serious lanes in flight.
2. Spend authoritative scorer time only after packaging, inflation, shape checks, and a promising faithful proxy.
3. Treat free-tier GPUs as real resources only when the run is durably recorded under the scheduler surfaces.

## current priority order

1. **PF-DILATED-H64 DEPLOY-CORRECT RELAUNCH**
   - Why first:
     - strongest raw local packaged lane now sits at `3.5753838920593264`
     - gap to promoted h64 local best is only `0.0281`
   - Current blocker:
     - saved meta still says `variant: "saliency_weighted"`
     - current artifact is observation-only until relaunched through the repo-side deploy-correct wrapper
   - Action:
     - relaunch via `experiments/train_postfilter_dilated_h64.py`
     - prefer Kaggle or Modal GPU
     - record manifest under `.omx/logs/remote_jobs/` and status under `.omx/status/`

2. **PF-SEGNET CHECKPOINTING RELAUNCH**
   - Why second:
     - SegNet remains the highest-leverage theoretical headroom
     - current live SegNet processes still do not emit rankable `best_*` artifacts
   - Action:
     - launch a fresh checkpoint-saving rerun from the synced repo-side trainer
     - reject any run that only prints pretty logs without writing artifacts

3. **PF-PAIRAWARE**
   - Why third:
     - still the most plausible architecture delta that directly addresses PoseNet pair scoring
   - Action:
     - launch `experiments/train_postfilter_pairaware.py` on Kaggle or Modal
     - do not proxy unless the local best closes materially toward the promoted h64 line

## platform guidance

- **Kaggle**
  - primary free-tier GPU lane for long training jobs
  - use for: deploy-correct dilated, pair-aware, fresh SegNet reruns
- **Modal**
  - secondary GPU lane when Kaggle is blocked or you need a cleaner Python/runtime story
- **Coiled**
  - CPU-side fan-out only
  - use for: fleet snapshots, proxy-gate triage, quantization audits, report rebuilds
  - do not treat Coiled as the default GPU training path under a free-tier-first strategy

## operator surfaces now on disk

- platform registry:
  - `configs/platforms.json`
- manifest/status templates:
  - `configs/run_manifests/kaggle_run_manifest.template.json`
  - `configs/run_manifests/modal_run_manifest.template.json`
  - `configs/run_manifests/coiled_run_manifest.template.json`
  - `configs/run_manifests/run_status.template.json`
- operator notes:
  - `docs/operator_run_manifest_templates.md`
- scheduler:
  - `comma-lab sched status`
  - `comma-lab sched results`
  - `comma-lab sched budget`

## sidecar outputs

- refreshed live fleet snapshot:
  - `reports/raw/2026-04-09-sidecar-analysis/live_fleet_snapshot.json`
- refreshed proxy gate:
  - `reports/raw/2026-04-09-sidecar-analysis/proxy_gate_triage.json`
- resolved PSD proxy evidence:
  - `reports/raw/2026-04-09-psd-h64-best/psd_h64_long1000_proxy_summary.json`
  - `reports/raw/2026-04-09-psd-h64-best/proxy_psd_h64_long1000_best.log`

## queue hygiene

1. Do not reopen `pixelshuffle_h64_long1000` or `psd_h64_long1000` without a material architecture or objective change.
2. Do not proxy deploy-blocked artifacts.
3. Do not claim Kaggle/Modal/Coiled are integrated unless the run is recorded on disk.
4. Leave the next agent a truthful queue, not a chat-memory queue.
