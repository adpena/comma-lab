# ddm_dr1 — δ_R (R-chain uint8 margin-noise floor) at n600 — the constant vr1 row 6 needs

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (local CPU)

## Why (gestalt position)
The next QBR1 burn generation races vr1's FOLD-NOW levers one at a time. Row 6 (margin-band satisficing cap,
`m_safe = headroom · δ_R`, equation `margin_band_satisficing_threshold_v1`, DSL `MarginBandSatisficing`) rests on
**δ_R = 0.019590163230895963 measured at n96** — a contiguous PREFIX of a skewed population ([[m88]]: a prefix of a
skewed population is a different population; seg prefixes read 0.95–0.97× the population, pose 2.5–4.2×). Nothing
may race row 6 on an n96 constant. This arm buys δ_R at **n600, all pairs** — no subset, no seed needed.

PRIOR-LAW PREDICTION (pre-registered): δ_R is a p95 of |Δmargin| induced by the uint8 round at camera resolution —
a property of the R operator, weakly dependent on scene content. Prediction: n600 δ_R within **±10%** of the n96
value (0.01763–0.02155). **Falsifier:** outside ±10% ⇒ the n96 constant was prefix-biased and every m_safe
derived from it (fh1 R3, hg1) is re-graded; report per-class-annulus δ_R as well (Lane vs Road vs Movable), since
the cap is applied per pixel and the classes have 2.185× flipdist spread (vr1 row 4).

## Verified at source (VERIFIED-AT-SOURCE LAW)
- Tool: `tools/measure_delta_R_noise_floor.py` (sha16 `1a6cc79029f44c49`, commit `aefb8ca7865be0820f2dae8a72243c11f9d8d4c9`). Argparse (:71-76):
  `--gt-npz` (default gt_n96.npz) · `--upstream` · `--band` (default 1.0, annulus |GT margin| < band) · `--n`
  (default 96, "caps at cache size") · `--out` (default reports/delta_R_noise_floor.json). Docstring :19-33 defines
  the isolation: x_c = bicubic(bilinear(gt_f1→384×512)→874×1164); m0 = margin(segnet(bilinear(x_c→384×512))),
  m1 = same with round/clamp at camera; δ_R := p95 |m1−m0| over annulus pixels. CPU-torch frozen SegNet only.
- GT frames: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` — **PyAV lineage** (20,671 argmax sites off
  DALI; memory `gt_n600_npz_is_pyav_lineage_train_on_dali_20260903`). δ_R uses the FRAMES (gt_f1) and SegNet's
  own margins, not the lstars table, and the DALI cache (`gt_cache_dali.pt`) holds no frames — so this measurement
  is honestly `[PyAV frames · macOS-CPU advisory · NON-PROMOTABLE]`. State that scope; do not launder it.
- Prior receipts to reconcile (read before writing): `.omx/research/ddm_fh1_forces_harvest_20260731.md` (δ_R n96,
  headroom 2.0 DERIVED, m_safe 0.0391803, `headroom_3_status: OPEN_UNMEASURED`) ·
  `.omx/research/ddm_hg1_ring0_margin_hinge_20260816.md` · `ddm_nx1_negative_and_mixed_signal_audit_20260816.md`.

## OPTIMAL FORM
- Reference form: the tool AS IS (never rebuild it — SPEC_v75 §8B ALREADY-SETTLED) at `--n 600` on gt_n600.npz,
  frozen CPU-torch SegNet, band 1.0 (the fh1 form). SCOPE change = n only. No mechanism change.
- Additions that are receipts, not mechanism: per-frame p95 + per-class-annulus p95 rows (PER-PAIR RECEIPTS LAW);
  a second run at band 0.5 if it costs < 15 min (sensitivity of δ_R to the annulus definition). TOY-BRACKET: none.
- Cadence: measure s/frame on the first 8 frames, then detach the full run through
  `tools/launch_detached_process.py --output-dir <store> --done-receipt ddm_dr1_n600 --derive-resource-budgets
  --measured-peak-rss-gib <n> --measured-thread-need 4 --walltime-cap-s 5400 --nice 10 --nice-best-effort -- <cmd>`
  with `torch.set_num_threads(4)` (the QBR1 burn is resident on Metal; ar1 holds 4 CPU threads).

## Deliver
- `reports/delta_R_noise_floor_n600.json` (the tool's own output; commit it) + per-frame/per-class receipts and the
  retained margin arrays under `/Volumes/APDataStore/pact/ddm_dr1_delta_R_n600/` (ALWAYS KEEP THE PAYLOAD; sha256 in
  the JSON). Report n600 vs n96: value, ratio, whether the falsifier fired; derived m_safe (headroom 2.0, DERIVED —
  label it) and per-class m_safe.
- Register an n600 `EmpiricalAnchor` on `margin_band_satisficing_threshold_v1` through the canonical registry
  helper (see how `src/tac/canonical_equations/` modules append anchors; `tools/recalibrate_equation.py
  --equation-id …` if applicable) — never hand-edit the JSONL. If the helper refuses, record the refusal verbatim.
- Memo `.omx/research/ddm_dr1_delta_R_noise_floor_n600_20260904.md`: verdict_scope, MEASURED/DERIVED labels,
  falsifier read out, GESTALT-DELTA line (what row 6 may now race with), NEXT_IF_RESUMED.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Any .py you touch: tests + `tools/review_tracker.py mark-file` twice (two real reads), never
  REVIEW_GATE_OVERRIDE on .py. Final message → `.omx/research/arm_final_messages/ddm_dr1_final_<utc>.md`, committed.
  LAST action: `touch .omx/tmp/codex_runs/ddm_dr1.done`.
- Never write under the burn's `runs/`; never touch `upstream/` or `submissions/semantic_joint_ctxmix/`; no /tmp
  paths in artifacts. Read `docs/operating_manual_craft_handoff.md` §labels first.
