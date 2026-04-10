# next experiments

## 2026-04-09 queue after Kaggle launch saturation

The promoted honest floor is still `1.73` from `long1000_h64`. Two deploy-ready alternates have now been resolved honestly and rejected:

- `pixelshuffle_h64_long1000` -> faithful proxy `1.99`
- `psd_h64_long1000` -> faithful proxy `1.85`

The first real saved SegNet-family artifact has now resolved honestly:

- `segnet_attack_fixed_ste_h32` -> faithful proxy `1.84`
- PoseNet `0.05168364`
- SegNet `0.00543626`
- bytes `864,167`

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
   - Status:
     - Kaggle kernel `adpena/comma-lab-dilated-h64-long1000` is now running
     - manifest: `.omx/logs/remote_jobs/kaggle-dilated-h64-long1000.json`
   - Action:
     - poll Kaggle status until the first artifact lands
     - mirror progress into `.omx/status/kaggle-dilated-h64-long1000.json`

2. **PF-SEGNET CHECKPOINTING RELAUNCH**
   - Why second:
     - SegNet remains the highest-leverage theoretical headroom
     - `segnet_attack_fixed_ste_h32` just proved the family can transfer honestly to `1.84`
     - the metadata gap is now fixed for future reruns, so the next launch can be both rankable and automatable
   - Status:
     - Kaggle kernel `adpena/comma-lab-segnet-attack-fixed-h32` is now running
     - manifest: `.omx/logs/remote_jobs/kaggle-segnet-attack-fixed-h32.json`
   - Action:
     - poll Kaggle status until the first artifact lands
     - confirm the hardened metadata path is actually exercised in the remote run

3. **PF-PAIRAWARE**
   - Why third:
     - still the most plausible architecture delta that directly addresses PoseNet pair scoring
   - Status:
     - Kaggle smoke kernel bundle is ready at `experiments/kaggle_kernels/pairaware_smoke`
     - push attempt was blocked by Kaggle's maximum batch GPU session count of `2`
   - Action:
     - launch on Kaggle when a slot frees up
     - or move the same bundle to Modal if Kaggle remains saturated

## platform guidance

- **Kaggle**
  - primary free-tier GPU lane for long training jobs
  - currently saturated with:
    - deploy-correct dilated
    - fresh SegNet fixed rerun
- **Modal**
  - secondary GPU lane when Kaggle is blocked or you need a cleaner Python/runtime story
  - immediate fallback for pair-aware if the Kaggle quota does not clear
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

1. Do not reopen `pixelshuffle_h64_long1000`, `psd_h64_long1000`, or `segnet_attack_fixed_ste_h32` without a material architecture/objective or packaging-metadata change.
2. Do not proxy deploy-blocked artifacts.
3. Do not claim Kaggle/Modal/Coiled are integrated unless the run is recorded on disk.
4. Leave the next agent a truthful queue, not a chat-memory queue.
