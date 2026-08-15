# ddm_hv1_harvest_compose_ep508 — the #1058 harvest-now composition from the selected checkpoint

## CONTEXT (MAIN adjudication 08-15, operator-steered "harvest live signal, don't waste long runs")
The e960 burn is ASYMPTOTED (fit receipt midrun_descent_fit_ep568.json: floor 130,875 B) and the
rfo2 selector (tools/select_hpac_checkpoint.py, landed 5624ef8bdc) identifies the distortion-
protected optimum: **ep508, joint 130,875 B, top1 0.00189660** — argmin of
(25/37545489)·joint_bytes + 100·top1 over retained checkpoints. This arm executes #1058(a):
compose ONE candidate from ep508 WITHOUT waiting for the endpoint and WITHOUT touching the live
run (pid 47772 + watchers + armed closer = SACRED, read-only).

## THE CHAIN (each stage = receipts; ALWAYS KEEP THE PAYLOAD)
(1) SELECT + VERIFY: run tools/select_hpac_checkpoint.py against the live run's log + periodic
    checkpoint dir (/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/
    full_e480b_e960/…) read-only; confirm ep508 (or the selector's current argmin), sha256 the
    selected checkpoint COPY into retention — never operate on the live file in place.
(2) EXPORT + BYTE-CLOSE: build the full archive from the selected checkpoint via the rx2/mc36
    identity-race conventions (experiments/ddm_rx2_mc36_identity_race.py; the e480b v2 build
    chain is the reference — recall its receipts, do not reinvent). Byte-exact section manifest.
(3) MICRO-EDIT RECOMPILE vs the FINAL coder state: re-apply the banked micro-edits — qs2
    (−4.375e-6, w/ its byte-preserved Schur compensation re-solved IN-COMPILE per the qs5 proven
    protocol, never carried cross-lattice) + re1 round-1 (−1.207e-6) — recompiled against THIS
    archive's coder, not transplanted (the cross-regime constant-transfer poison).
(4) ADVISORY n600: CPU-torch d_seg/d_pose + exact byte count on the composed archive, labeled
    [macOS-CPU advisory] honestly; compare vs e480b v2 components (seg 0.029611 / pose 0.0082946 /
    183,502 B). Admission bar: projected net ΔS < 0 vs e480b v2 at the ±3.5e-6 band.
(5) SEALED T4 FIRE-ORDER (MAIN fires, never dispatch Modal yourself): dual repeat, all hashes
    pinned, mirroring the proven r4/js1c dispatch conventions + canonical poller close. Also emit
    the CPU-axis note: pq1's sealed CPU fire-order re-targets to THIS candidate via the landed
    SWAP_PROCEDURE.md (.omx/research/ddm_pq1_submission_packet_prep_20260815/).

## HARD CONSTRAINTS
- LIVE RUN SACRED (no signal/copy-in-place hazards; cp then sha-verify). NO Modal dispatch.
- Retention: /Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/ w/ sha manifest.
- Serializer commits (tools/commit_autosha.sh), [no-triality] [p0-ledger-ok]; .py = 2 review
  passes; memo .omx/research/ddm_hv1_harvest_compose_ep508_20260815.md w/ DEAD-ENDS +
  LIVE-HYPOTHESES; end with the vehicle frontier line.

## OPTIMAL FORM
- Family reference: the e480b v2 composition chain at its landed form (the MC36 promotion +
  e480b build receipts) + qs5's in-compile compensation protocol + re1's dual-axis seal builder
  (9207d5eac0). MECHANISM reductions: NONE — a transplanted (non-recompiled) micro-edit or a
  skipped advisory gate is inadmissible. SCOPE: ONE candidate from the selected checkpoint.
- Provenance pins: selector 5624ef8bdc · fit receipt midrun_descent_fit_ep568.json · e480b v2
  archive sha e3e6f440b45bbb92… · qs2/re1 verdict memos · pq1 swap procedure (dec5402577).
