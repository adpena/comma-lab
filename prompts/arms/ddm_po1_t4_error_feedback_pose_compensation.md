# ddm_po1 — pose leg: compensation against T4-MEASURED error vectors (the PR135 mechanism aimed at our carrier)

**Charter date:** 2026-08-13 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN fires T4 dispatches (~$0.16/round, 2–3 rounds ≈ $0.50, #381).
**Read first:** CLAUDE.md/AGENTS.md + `docs/operating_manual_craft_handoff.md` + `.omx/research/ddm_ps135_pass4_exact_row_harvest_20260812.md` (the refutation this supersedes) + `.omx/research/ddm_js1b_cuda_custody_adjudication_20260813.md` (instrument law) + the js1b worker/dispatcher pair (proven chain).

## The unmeasured cell (m44-checked)

ps135 pass4 solved the pose carrier ON THE LOCAL AXIS and died at T4 (d_pose 1.4674e-4 = 21.3×
cp135's 6.88e-6) — hc1's dead-end: "chasing the CPU pose solve below its CUDA disagreement floor is
closed." But PR135's OWN winning mechanism is quantize-then-COMPENSATE: solve the correction against
errors MEASURED on the shipping axis (they are CUDA-locked by choice). We never ran that cell — every
one of our pose solves optimized against locally-computed pose values. The pose headroom is −0.0083 S
(pose→0 from d_pose 6.88e-6), UNCLAIMED per the ps135 harvest verdict.

## The loop (train-nothing; measure-then-solve)

1. **Worker extension ($0 build):** extend the js1b field-materializer worker to ALSO emit, for all
   600 pairs: PoseNet(decoded pair)[:6] and PoseNet(GT pair)[:6] on T4 (fp32, 600×6×2 ≈ 28.8 KB —
   trivial payload, P0-retained). The scorer already runs there; this is a small additive output.
   **Include an in-dispatch determinism repeat:** run the decoded-pair forward TWICE in the same job
   and emit both — the repeatability of the pose vectors at the 1e-6 scale is the gate for everything
   downstream, and it is free inside one dispatch.
2. **Round-1 dispatch (MAIN fires):** cp135 archive → T4 pose vectors + repeat + (free) seg field
   re-emission. Read: per-pair T4 error vectors e_i = pose(decoded)_i − pose(GT)_i, and the
   repeat-noise floor n_i.
3. **Compensation solve ($0, local):** adjust the EXISTING pose-carrier coefficients in place
   (byte-neutral; any byte delta stated) to cancel the measured e_i through the carrier's LOCAL
   Jacobian — local J as preconditioner, T4 residual as the error signal (the standard split: axis
   drift corrupts VALUES, small-step descent directions survive). Damped step; never chase below n_i.
4. **Round-2 dispatch (MAIN fires):** re-measure. Admission: d_pose strictly down AND seg field
   unchanged (pose-carrier coefficients must not touch seg — verify via the free seg re-emission)
   AND joint ΔS < 0. One more round only if the first step realized ≥50% of its predicted gain.

## Falsifiers (pre-registered; each is a load-bearing routing fact)

- **F1 (instrument floor):** repeat disagreement |n_i| ≳ |e_i| for most pairs → the remaining 6.88e-6
  is AT the T4 noise floor; the pose leg is CLOSED-MEASURED, −0.0083 unclaimable, and sub-0.15 must
  come entirely from seg+rate. Report plainly — this is the most valuable possible negative.
- **F2 (Jacobian-direction failure):** round-2 realizes <20% of predicted gain or inverts → the
  local-Jacobian/T4-residual split fails at this scale; one damped retry, then closed.
- **F3 (seg contamination):** compensation moves the seg field → the carrier is not seg-isolated;
  report the coupling, do not ship.

## Custody
- cp135 archive READ-ONLY; candidate archives are new files. P0 KEEP-THE-PAYLOAD (pose vectors, both
  repeats, all candidates). No Modal/MPS from the arm — build the worker extension + solver + pinned
  dispatch commands; fires stay with MAIN. Land via serializer (post-edit working-tree shas).
