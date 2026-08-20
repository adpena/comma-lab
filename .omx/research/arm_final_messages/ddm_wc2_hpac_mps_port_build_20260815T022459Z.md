Built and verified the HPAC MPS race apparatus, but it is uncommitted and unexecuted due environment limits.

Delivered:

- [MPS trainer wrapper](/Users/adpena/Projects/pact/tools/train_ddm_cl1_hpac_capacity_mps.py:1), importing the hash-pinned CPU trainer and changing only device/epoch admission.
- [Parity comparator](/Users/adpena/Projects/pact/tools/compare_ddm_cl1_hpac_capacity_mps.py:1), measuring trajectory divergence, retaining repeated real IHS1 payloads, and computing race timing from successful receipts.
- Six validated liveness/quality watcher configurations.
- [Full receipt and exact MAIN commands](/Users/adpena/Projects/pact/.omx/research/ddm_wc2_hpac_mps_port_20260814.md:1).

Measured routing remains:

- Profile wall: 2,486.478 seconds.
- Conv2d forward: 1,282.433 seconds, 51.6%.
- Backward: 1,008.073 seconds, 40.5%.
- Peak RSS: 10.278 GiB; measured thread need: six.
- Local Torch 2.12.1 has MPS built but unavailable.
- This sandbox cannot apply niceness 10.

Verification passed: Ruff, compilation, 14 focused tests, two review passes per Python file, zero payload-retention findings, six watcher validations, and three governed launcher dry runs. Those dry-run manifests were retained losslessly under [DRY_RUN_RETENTION_MANIFEST.json](/Volumes/APDataStore/pact/ddm_wc2_hpac_mps_port_20260814/dry_run_validation/DRY_RUN_RETENTION_MANIFEST.json:1); the real launch roots are absent again.

No parity run, MPS kernel proof, speedup, endpoint byte measurement, full race, scorer, archive, or exact evaluation occurred.

The required serializer failed before staging because Git cannot create index/object temporary files here. The staged index remains empty. All eleven repository artifacts are verified but uncommitted.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT** — owner: MAIN in a Git-writable context; consumer store: repository HEAD and the WC2 receipt; fire trigger: the serializer can create Git index/object files while the staged index remains empty.
- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/cpu/`; fire trigger: the commit is landed, niceness 10 verifies, and live RX2 cadence is healthy.
- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-ENVIRONMENT** — owner: MAIN on the Metal host; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/mps/`; fire trigger: the commit is landed, MPS is available, and fallback remains zero.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/parity/comparison/`; fire trigger: both parity receipts report `rc=0`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full/`; fire trigger: repeated IHS1 packs match exactly, the trajectory is accepted as the same instrument, and projected finish margin is positive.

## LIVE-HYPOTHESES

- Torch MPS may exceed 3× CPU throughput because roughly 92% of measured time lies in convolution forward/backward. Actual kernel coverage and speed remain untested.
- CPU/MPS trajectories may remain close because the wrapper imports the same model, optimizer, EMA, seed, and evaluation code. Floating-point accumulation order can still make MPS a different instrument.
- An MLX/custom-Metal port may reduce drift while retaining GPU speed because fixed-point forward and grouped-convolution VJP assets exist. It remains plausible but owes substantial HPAC-specific training adaptation.

## DEAD-ENDS

- CPU vectorization is closed for this measured profile: the dominant cost is inside convolution kernels, not Python dispatch.
- Local MPS execution is closed in this environment: MPS is built but unavailable.
- Local matched CPU parity is closed in this environment: niceness 10 cannot be applied.
- Treating the existing fixed-point scorer kernels as a drop-in HPAC trainer is closed: they do not provide the evolving-weight training semantics required here.
- MPS as score authority is closed; CPU packing and exact contest evaluation remain authoritative.
- Claiming this work as committed is closed: the serializer failed before staging.
- Frontier movement by WC2 is closed because no candidate or evaluation ran. Own-vehicle frontier remains `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`.

