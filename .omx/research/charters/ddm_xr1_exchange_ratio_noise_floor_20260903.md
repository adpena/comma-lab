# ddm_xr1_exchange_ratio_noise_floor — the campaign's meta-instrument nobody measured (#1248, rn1 rank 10): define and MEASURE the noise floor of the byte↔distortion exchange ratio so every ratio-based near-win/closure verdict carries an interval instead of a point

## MANDATE

Operator standing GO. `ddm_rn1_n600_reopen_sweep_20260903.md` (90e560957) rank-10 fire order: "register the
estimand, repeated physical objects, denominator, and acceptance interval; then a scorer-free exact-reencode
instrument may be named." Every closure of the form "row X is 1.14× / 0.94× / 1.66× its bar" (300 near-win
candidates in rn1's screen) inherits an UNMEASURED noise floor. MAIN registers the estimand here; the arm
instruments and measures it.

## THE ESTIMAND (MAIN's definition — binding for this arm)

- **Object:** the exact RC64 token-stream byte count B(F, M, G) of a field F under the shipped model M and
  schedule G, and the realized advisory distortion D(F) = (100·d_seg, sqrt(10·d_pose)) through the frozen CPU
  scorers. The exchange ratio of an edit set E is r(E) = ΔS_rate(E) / ΔS_dist(E) with ΔS_rate = 25·ΔB/37,545,489.
- **Repeat unit (physical):** one complete re-encode of the SAME field under the SAME model. Byte noise floor
  σ_B = the spread of B across physically repeated encodes (determinism: expected 0; MEASURE it — the RXC1
  instrument reported 0 differing bytes on null replays, so σ_B's floor is 0 and any nonzero is a finding).
- **Repeat unit (statistical):** the pair-level bootstrap — B and D are sums over 600 pairs; resample pairs
  with replacement (seeded) to get the sampling interval of ΔB and ΔD for an edit set E confined to a pair
  subset. This is the interval a "near-win" must clear.
- **Denominator:** ΔS_dist in S units (never d_seg alone), at the live operating point (∂S/∂d_pose = 626.5;
  one argmax site = 8.477e-7 S).
- **Acceptance interval:** a near-win is ADMISSIBLE only if its ΔS (rate + distortion) is negative at the 95%
  pair-bootstrap interval's upper edge. Report the interval width for the two canonical edit sets below.

## SCOPE

1. **Instrument:** `experiments/ddm_xr1_exchange_ratio_noise_floor.py` — reuses the RXC1/JG2 exact
   re-encode path (`experiments/ddm_rxc1_restartable_exact_coder.py` 9cf2fd5d8) and JBP1's retained fields
   (`experiments/ddm_jbp1_joint_batch_price.py` 625de245e; custody
   `/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/`): (a) physical repeat of the null encode
   ×3 (σ_B); (b) pair-bootstrap (seeded, 200 resamples) of ΔB for JBP1 row A (5,506 edits across 567 pairs,
   −2,950 B) using the retained per-frame byte ledger; (c) the same bootstrap on the retained FCD3 realized
   d_seg/d_pose (`ddm_wwc1_winwin_cone_sweep_20260831.md`: net +0.00194 S) — the one edit set with BOTH axes
   measured — giving the interval on its ΔS.
2. **Register** the estimand as a canonical equation via `tac.canonical_equations` (the registry is the
   triality's equations leg; use the register_* pattern in tools/), with the measured σ_B and the two
   intervals as EmpiricalAnchors.
3. **Re-grade** rn1's 300 near-win candidates' top 20 by whether their margin clears the measured interval
   (scorer-free where the receipt has per-pair data; UNGRADABLE otherwise — say so).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY. NO scorer runs (use retained
  per-pair distortion receipts only), NO Modal/Metal; CPU re-encodes only (each ~15 min; if >30 min total,
  DETACHED ONLY via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt
  <name> --nice 10 --nice-best-effort -- <cmd...>`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/VertigoDataTier/pact/ddm_xr1_exchange_ratio_noise_floor/`.
- Do not touch gc1/gf2 files or stores.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rn1_n600_reopen_sweep_20260903.md` — rank 10: statistic undefined; every ratio close inherits it.
- `ddm_ww1_walls_that_werent_20260902.md` §3.5 — "exchange-ratio noise floor (#1248) — nobody has measured it".
- `ddm_rxc1_gen3_gate1_verdict_20260901.md` — exact restart: 64/64 byte-identical; 0/32 terminal adaptive
  state reconvergence (prices are suffix-level).
- `ddm_fs3_*` (memory m166): AVERAGE ≠ MARGINAL (2.24×): the bootstrap must resample PAIRS, not sites.
- `prefix_bias_sign_inverts_between_seg_and_pose_20260803` (memory): never a prefix subset.

## OPTIMAL FORM

- Family exemplar: JBP1's exact roster with its per-frame byte ledger, reference
  `experiments/ddm_jbp1_joint_batch_price.py` (commit 625de245e), and the RXC1 exact coder
  `experiments/ddm_rxc1_restartable_exact_coder.py` (commit 9cf2fd5d8); the canonical-equation registration
  pattern `tools/register_*.py` (e.g. `tools/register_adaptive_ceiling_admission_control_equation_20260703.py`).
- SCOPE reductions: 200 bootstrap resamples; ×3 physical repeats (legal). MECHANISM reductions FORBIDDEN:
  no site-level resampling; no synthetic fields; no scorer proxy.
- **PRIOR-LAW PREDICTION (falsifiable):** σ_B = 0 (deterministic coder) and the pair-bootstrap 95% interval
  on JBP1 row A's −2,950 B is narrower than ±600 B, so its rate credit is real but its 7.46%-of-demand
  standing is unchanged; FCD3's net +0.00194 S interval excludes zero (the win-win cone stays refused).
  FALSIFIER: an interval that includes zero for FCD3 — count it plainly; it reopens the cone at n600 with
  a fresh scorer order.

## DELIVERABLE

`.omx/research/ddm_xr1_exchange_ratio_noise_floor_20260903.md` — the registered estimand, σ_B, the two
intervals, the re-graded top-20 near-wins, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS.
Commit via the serializer. Cite `docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
