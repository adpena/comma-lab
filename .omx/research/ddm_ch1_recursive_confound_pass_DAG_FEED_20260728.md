# DAG FEED — ddm_ch1 recursive confound pass [no-triality] [p0-ledger-ok]

**2026-07-28 · report-only L3 confound hunt · score_claim=false · pointer 0.1910828242 UNMOVED**

## Node: ch1 — recursive confound pass over the 07-28 stack (fc1/da1/ar1/sc1/sp1/rp1/pp1/fd1r/cs1)

**Question:** does any instrument or same-day verdict-chain link in the 07-28 stack carry a
DEFAULT-HARMFUL×SILENT×MEASUREMENT-CORRUPTING confound? Audit S1–S9 adversarially + orthogonal surfaces.
NO scorer jobs (live arm owns slot); cached-receipt numpy + code inspection only.

**RESULT: 0 CONFIRMED-CONFOUND. The stack is unusually well-controlled.**

- **S1 (plumbing-bug reopener) — REFUTED, decisive.** `realized_advisory_action` diverges from
  `baseline+block_proxy_delta` in the WRONG direction (proxy predicted decrease, reality measured increase)
  → impossible to fabricate from proxy. Code (`_chunked_n600_verdict` L734-771) computes d_pose LIVE via
  canonical `cpu_verdict_d_pose_batch`; receipt folds it into advisory_action (labeled "implied", honest).
  Pose deltas +2.8…+13.1% receipt-true. fd1 S2 ZERO-accept is REAL, not an artifact.
- **S3+S8 (coherent-poisoning worst class) — CLEAN.** gt_n600.npz SHA `cf8d83605d…` = producer-log size
  5,078,017,610 = fd1r `run_identity.target_cache_sha256`; consumed via fail-closed `_verify_regular`
  (recomputes sha, RAISES on mismatch). Content-sane (argmax int64 0-4). rp1 C0 (GT→lstars 0 flips) is a
  scorer↔substrate positive control.
- **S5 (un-centered SVD) — REFUTED.** numpy reproduction: receipt's `e_p_svd_energy_frac[0]=0.998624` ==
  CENTERED SVD exactly (uncentered=0.998089). Rank-1 is genuine motion. SEAM: DC mean dim0=−0.809 (33.5% of
  uncentered energy) must be stored separately in any #741 realization.
- **S4 (pp1 KT closed-form) — SEAM.** 173.6 KB is closed-form KT; round-trip proven at 6 frames w/ Laplace
  (coded/closed=1.0000435), extrapolated to n600 KT. Robust to routing (generic coders bracket 330-660KB;
  2× falsifier margin). Memo honest.
- **S6 (rp1 C2 degeneracy) — CLEAN.** memo self-flags C2 degenerate-by-construction; over-read closed by fd1
  S0 (box-solve band 3.76e-4 ≈ GT 3.63e-4).
- **S2 (positive controls) — most instruments have intrinsic controls (coders: round-trip; rp1: C0; sc1:
  banked_vs_local_tp 0.0156; fd1r S0: raw-sha custody). fd1r S2 lacks a DECLARED canary (validity via
  implicit controls) → SEAM, L3 cure = declare identity canary.
- **S7 — SEAM.** two verdict impls (fd1r wraps canonical base-trainer fns; rp1 rolls own modules.py loop);
  each anchored, total-flip agg + self-consistency assert; not cross-bit-validated (measure different objects).

**S9 blast-radius (routing-flip risk):**
1. **fd1 S2 ZERO-accept → two-rung ladder** (single window, 4-pair block, one warm start; scoped INSTANCE).
   Re-check: 1-2 more GN windows, different block + #383 pose-null projector (~1 hr, = rung-1).
2. **rp1/fd1 CELLS-HOLD → engine-capacity** (double-measured, but not at the GN engine's 0.0702 operating
   point). Re-check: rp1 C1 probe on the W_joint frames (existing tool, ~525 s).

**VERDICT: recursive confound pass CLEAN. No day verdict flips. Cures routed to MAIN (L1: persist
realized_d_pose + DC-mean note; L2: full-n600 KT round-trip + cross-verdict-path vector check; L3: fd1r
declared identity canary). Pointer UNMOVED.**

**Artifacts:** memo `.omx/research/ddm_ch1_recursive_confound_pass_20260728.md`; scratchpad numpy checks
(gt SHA, e_p centering) non-durable. No new tool, no scorer job, no .py.
