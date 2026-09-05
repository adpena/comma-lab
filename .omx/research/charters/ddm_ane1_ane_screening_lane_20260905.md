# CHARTER ddm_ane1 — saturate the ANE: prove placement, measure per-scorer fp16 drift at n600, and wire an ADVISORY screening lane (never authority)

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (operator 09-05: "ANE as well"; apparatus → Opus). Spawned 2026-09-05 ~13:10Z. Parents:
`ane_unlock_correction_20260713.md` / `ane_unlock_followup_20260713.md` / `a2_ane_fidelity.json` / `a2_ane_latency.json` (07-13: CoreML fp16 on
`CPU_AND_NE` SegNet forward 9.1 ms vs 346 ms CPU-torch-1-thread = 38.03× on (1,3,384,512), RANDOM-INIT latency evidence; on 24 REAL frames with real
weights: argmax flip rate 0.02475 (116,764 / 4,718,592 px), logit cosine median 0.9943; the AUTHORITY bar is flip rate ≤ 3.3e-5 — fp16 SegNet is ~750×
above it; **PLACEMENT NOT PROVED** (CPU_AND_NE requested, never shown to run on the ANE); the full-fp32 CoreML CPU+GPU route is `UNLOCKED_LOCAL_ONLY`
for verdict advisory; no `.mlpackage` retained in-tree), gv1 §5 (closed by recall — "the ANE can never be a d_seg authority"), the MPS train/authority
split in CLAUDE.md (MPS is a gradient device, never a score — the ANE is a SCREENING device under the same law), and the two pointer moves of 09-04
(fs1/fs2: the pose-axis instruments run CPU-torch PoseNet at ~2.3 s/pair for the selector sweep and batch-8 n600 measures — the concrete workload an
ANE screen would accelerate).

## PRIOR-LAW PREDICTION (owed line)
The ANE is fp16-only; SegNet's argmax is decided by sub-1e-3 logit margins at boundaries (m96/md1), so fp16 flips ~2.5% of pixels and can NEVER read
d_seg (authority bar 3.3e-5). PoseNet's output is a 6-dim regression whose contest term is √(10·MSE) with MSE ~6e-6; fp16 perturbs each dim at ~1e-3
relative, so the fp16-vs-fp32 pose MSE delta lands ~1e-8–1e-7 per pair — an order below the per-pair Δd_pose the selector sweep ranks on (fs1 adopted
pairs at ~1e-6–1e-5). PREDICTION: PoseNet fp16 on the ANE is ADMISSIBLE for SCREENING (rank-preserving on ≥ 95% of the sweep's adopted set with a CPU
fp32 confirm on the adopted pairs), at ≥ 10× forward speedup; SegNet fp16 is INADMISSIBLE for any d_seg read (flip rate ≥ 1e-2 at n600) and only a
hybrid (ANE trunk + fp32 recompute of margin-band tiles) could ever be exact — priced, not built. FALSIFIER: placement cannot be proved (then every
"ANE" number is a CoreML-CPU/GPU number and the lane is renamed honestly), or PoseNet fp16 per-pair drift ≥ the sweep's adoption deltas.

## Scope
1. **Placement proof.** Convert the frozen SegNet (smp Unet efficientnet-b2) and PoseNet (FastViT-T12) to CoreML (coremltools; retain the
   `.mlpackage`s with sha256 under `/Volumes/VertigoDataTier/pact/ddm_ane1_ane_screening/`); load with `compute_units=CPU_AND_NE`; PROVE ANE
   placement with `coremltools.models.compute_plan.MLComputePlan` (per-op device) — report the fraction of ops on the ANE per model; if the API is
   unavailable on this OS build, use the latency triad (CPU_ONLY vs CPU_AND_GPU vs CPU_AND_NE, same input, 30 reps) and state that placement is
   INFERRED from latency, not proved. Never sudo/powermetrics.
2. **Fidelity at n600 on REAL data** (the DALI GT cache pairs — the same inputs the scorers see): per pair, SegNet argmax flip rate fp16-ANE vs CPU-torch
   fp32 (1 thread, the authority form) and vs CoreML fp32 CPU; PoseNet 6-dim MSE delta vs CPU-torch fp32 and the resulting Δ√(10·MSE); distribution
   (median, p95, max). Report per scorer against the bars: seg 3.3e-5 (authority) and the fs1 sweep's adoption deltas (screening).
3. **Screening admissibility + wiring (PoseNet only unless 2 says otherwise):** add `--scorer-backend {cpu_torch,coreml_cpu_fp32,ane_fp16_screen}` to
   the pose-axis instrument family (`experiments/ddm_fs1_frame0_selector_reselection.py` sweep/measure; pr1's solve if shared) with the CONTRACT: the
   backend may rank/screen; every ADOPTED pair is re-measured on `cpu_torch` fp32 before any number leaves the instrument; the output JSON records
   both, the backend name, and `score_claim=false`. Replay fs1's 39-point sweep on the ANE backend and report rank agreement + wall-clock vs the
   recorded CPU run (2.3 s/pair). Provenance: the `.mlpackage` sha + coremltools version in every receipt.
4. **The exact-SegNet hybrid, PRICED not built:** from the flip-site census (2), the fraction of pixels inside a margin band that would need fp32
   recompute for bit-exact argmax, and the implied end-to-end speedup vs 1-thread CPU torch; bar for a future build: bit-exact argmax on n600 AND ≥ 3×.
5. Memo `.omx/research/ddm_ane1_ane_screening_lane_20260905.md` (numbers, bars, verdicts, what is wired, what is priced), a runbook line in
   `docs/` if wired, ≥ 15 tests (backend contract; CPU-confirm invariant; provenance fields), lane registered.

## Cost + admission
$0; CPU + ANE; ONE process at a time (the box also runs cl2 on the Metal and md2/hc2 on CPU — do not add parallel fan-outs; the watchdog is
report-only and names the actor). Long steps detached via `tools/launch_detached_process.py --done-receipt <distinct>`; storage on Vertigo (44 GiB) —
declare `--artifact-budget-gib`.

## OPTIMAL FORM
Reference form = the 07-13 lane's measurement discipline (real weights, real frames, flip-rate + cosine, latency triad) lifted to n600 and to BOTH
scorers, plus the MPS train/authority split as the wiring contract. No mechanism reduction: fidelity is measured on real n600 inputs, never a random-init
proxy; screening never replaces the CPU fp32 confirm.

## Rules that bind
NO-FAKE (no "ANE" claim without a placement proof or an explicit INFERRED label); ALWAYS KEEP THE PAYLOAD (mlpackages, per-pair fidelity tables,
receipts with sha256); `upstream/` READ-ONLY (never patch the scorers — convert copies); no Modal, no Metal; commits ONLY via
`tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`; NO co-author
trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool uses (`--subagent-id ddm_ane1`); never
invent flags (grep argparse); no `/tmp` evidence; register the lane first (`lane_ddm_ane1_ane_screening_20260905`); persist records before bulk saves;
label MEASURED/DERIVED/INFERRED; "Equations leg (`tac.canonical_equations`)" line (register the per-scorer fp16 drift as an anchor on the existing
MPS/precision drift law if one exists — grep first; else `scorer_fp16_drift_by_axis_v1`). `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
