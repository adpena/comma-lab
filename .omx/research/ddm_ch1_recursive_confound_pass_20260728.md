# ddm_ch1 — RECURSIVE CONFOUND PASS over the 07-28 measurement stack (L3, report-only)

**Pointer honesty first: 0.1910828242 [contest-CPU] UNMOVED (submittable local anchor;
`submitted_pr_number_for_current_frontier=None`). This arm is REPORT-ONLY — it moved no exact score, fixed
nothing, ran NO scorer jobs.** Fresh-eyes confound hunter per CLAUDE.md "Confound self-protection" at the L3
recursive level: I audit the INSTRUMENTS and the day's VERDICT-CHAIN, and I audit the orchestrator's seeded
hypotheses S1–S9 ADVERSARIALLY (claims, not facts). Evidence = cached SSD receipts + committed JSONs + code
inspection + $0 numpy re-derivations. A suspicion without receipt evidence is filed SEAM-NAMED, never
CONFIRMED-CONFOUND.

**Headline:** the 07-28 stack is unusually well-controlled. **ZERO CONFIRMED CONFOUNDS.** S1 (the plumbing-bug
reopener) is DECISIVELY REFUTED. S3+S8 (the coherent-poisoning worst class) are CLEAN with a verified
end-to-end SHA custody chain. S5 (the un-centered-SVD confound) is REFUTED by exact numpy reproduction. The
findings are five SEAM-NAMED reporting/robustness gaps (none of which flips a day verdict) + one genuine
verdict-chain velocity seam (S9 rank-1) whose cheapest re-check is already the campaign's named next rung.

## STORES CONSULTED (recall-first; every cited receipt verified this session):
`CLAUDE.md` + `AGENTS.md` (full read: NO-FAKE #1/#6/#8; Confound self-protection 3-layer immune system;
staleness-confound law; measured-scored-quantity axis; MPS/advisory-never-authority; serializer + post-edit
sha; pointer-only frontier) · `MEMORY.md` current-state (`box_retired_min_s_target_warp_family_closed…`,
`staleness_is_a_named_confound_class_freshness_at_consumption_20260723`, `null_subspace_rate_measure_20260717`,
`realization_is_quantization_gated…`, `verdict_rules_measured_pace_reported_direction_20260725`) · charter
`scratchpad/ch1_charter.md` (S1–S9 + beyond-seeds) · the stack under audit: memos
`.omx/research/ddm_{fc1,da1,ar1,sc1,sp1,rp1,pp1,fd1,cs1}_*_20260728.md` + DAG FEEDs · SSD receipts
`/Volumes/VertigoDataTier/pact/ddm_{fd1,rp1,sc1,pp1,sp1,fc1,da1,ar1}_20260728/` · verdict code
`tools/launch_ddm_joint_descent.py` (`_chunked_n600_verdict` L703, `_verify_regular` L154),
`tools/measure_ddm_rp1_rangeA_cell_probe.py` · producer `tools/build_shared_gt_cache_for_mlx_fleet.py`
(gt_n600 build log) · registry `.omx/state/canonical_equations_registry.jsonl` (band-lemma row).

## TYPED CONFOUND TABLE

| # | surface | class | evidence (receipt / code) | VERDICT | blast radius | cure routing |
|---|---|---|---|---|---|---|
| **S1** | fd1r GN-window pose-delta plumbing | plumbing-bug (proxy-computed vs realized) | `ddm_fd1_20260728/s2_gn_window/fd1_gn_window_receipt.json`; `launch_ddm_joint_descent.py:703-771` | **CLEAN** (refuted) | would have flipped fd1 S2 "ZERO accept" verdict + the whole two-rung routing — but robust | none (plumbing real) |
| S1-seam | `realized_d_pose` not persisted per-attempt | recording gap (non-corrupting) | receipt has `realized_advisory_action`+`realized_d_seg` only; d_pose folded into advisory_action (invertible) | SEAM-NAMED | reader must recompute pose channel; memo already labels it "**implied** Δd_pose" | L1: persist `realized_d_pose` per attempt |
| **S2** | fd1r S2 positive-control | missing declared canary (L3 gap) | receipt has no pre-registered sentinel; validity rests on IMPLICIT controls (5/6 baseline bit-reproduction + advisory-action divergence from proxy) | SEAM-NAMED | fd1r S2 alone lacks a declared control; all other instruments have intrinsic controls | L3: declare identity-canary (re-run baseline == recorded) in window receipt |
| **S3** | content-addressed gt-cache staleness in verdict paths | staleness (default-harmful×silent) | `launch_ddm_joint_descent.py:154-163` `_verify_regular` HARD-raises on sha/byte mismatch; `run_identity.json target_cache_sha256=cf8d83…` = my computed SHA | **CLEAN** | — | none (freshness-at-consumption enforced fail-closed) |
| **S4** | pp1 173.6 KB KT closed-form authority | extrapolation (closed-form==coder) | `ddm_pp1_20260728/r1_direct_partition_n600.json`: round-trip proof `subset_frames=6`, `coded/closed=1.0000435`, **Laplace** model; headline is **KT temporal-context** | SEAM-NAMED | pp1 "partition cheap → wall is realization" routing — robust (generic coders 330-660KB bracket; 2× falsifier margin) | L2: full-n600 round-trip through the actual KT coder OR label headline "closed-form" |
| **S5** | sc1 e_p rank-1 0.9986 centering | un-centered SVD = mean offset not motion | `ddm_sc1_20260728/ep_chunks/*.npz`: my CENTERED SVD = **0.998624** (matches receipt exactly); uncentered = 0.998089 | **CLEAN** (refuted) | #741 dual-use design safe (SVD was centered) | none |
| S5-seam | e_p DC mean offset (dim0 = −0.809, 33.5% of uncentered energy) | realization completeness | numpy: `e_p.mean(0)=[−0.809,0.018,−0.053,…]` | SEAM-NAMED | any #741 realization must store the DC mean separately from the rank-1 direction | L1: note DC term in the dual-use spec |
| **S6** | rp1 C2 "all damage in C1" attribution | non-separated damage (uint8 solve) | `ddm_rp1_…20260728.md:88-94` explicitly flags C2 "degenerate BY CONSTRUCTION"; charter split "collapses" | **CLEAN** | rp1→fd1 routing — and fd1 S0 CLOSED the deferred box-solve check (band invariant 3.76e-4 vs GT 3.63e-4) | none (attribution honest; over-read closed) |
| **S7** | chunked n600 verdict path identity | divergent-copy | `_chunked_n600_verdict` wraps CANONICAL `cpu_verdict_d_seg_argmax_batch`/`_d_pose_batch` + total-flip agg + self-consistency assert (L757); rp1 rolls own modules.py loop | SEAM-NAMED | two impls, each anchored (fd1r: baseline bit-reproduction; rp1: C0 GT→lstars 0 flips); measure different objects, not cross-bit-validated | L2: add a shared-vector cross-check between the two verdict paths |
| **S8** | shared substrate gt_n600.npz | coherent poisoning (worst class) | SHA `cf8d83605d…` = producer-log size 5,078,017,610 + fd1r `run_identity.target_cache_sha256`; lstars int64 0-4, no NaN; built 2026-06-26; rp1 C0 GT→lstars 0 flips = scorer↔substrate control | **CLEAN** | entire day rests on it — verified consistent producer→my-check→consumption | none |
| S8-bound | gt_n600 vs LIVE scorer bit-correctness | (residual, unmeasurable here) | scorer jobs forbidden — rests on 2026-06-26 producer receipt + month of cross-checked lineage | SEAM-NAMED | — | boundary, not a cure |
| O1 | `b/flip` unit reuse across arms | units/normalization | fc1 "0.325 b/flip" = LABEL entropy (0.3248 ✓); sp1/support "3.31-3.49 b/flip" = POSITION | SEAM-NAMED | cross-arm skim could conflate; each memo internally precise | doc: qualify unit as `b/flip(label)` vs `b/flip(support)` |
| O2 | band-lemma registry row | false-authority | registry: `score_claim=False, promotion_eligible=False, evidence_axis="[macOS-CPU advisory]"` | **CLEAN** | — | none |
| O3 | fd1r resume-blocker (copied MAIN pointer into worktree) | config-staleness | `ddm_fd1_…:129-132` "deterministic, value-identical fix"; pointer is provenance-only, not in numeric verdict (v19 rule is candidate-vs-baseline relative) | **CLEAN** | — | none |
| O4 | sc1 `banked_vs_local_tp_maxabs=0.0156` | recompute-consistency | receipt field; 0.0156 vs dim0 std 1.14 = negligible for rank-1 | SEAM-NAMED | banked t_p 0.0156 off local recompute; does not affect any 07-28 verdict | note in any t_p-consuming realization |

**Tally: 0 CONFIRMED-CONFOUND · CLEAN on S1,S3,S5,S6,S8,O2,O3 · 8 SEAM-NAMED (all low/no verdict-flip blast).**

## S1 — the plumbing-bug reopener: DECISIVELY REFUTED (the day's highest-stakes check)

The claim under audit: "pose deltas +2.8–13.1% across d_seg-bit-identical candidates prove the verdict saw
changed frames." The reopen condition: if pose deltas were **computed** (proxy arithmetic), not
**realized-measured**, the plumbing bug returns and fd1 S2's ZERO-accept becomes an artifact.

**Two independent proofs it is realized, not computed:**
1. **Proxy-divergence (receipt arithmetic).** For all 6 candidates `realized_advisory_action` (26.46–27.40)
   DIVERGES massively from `baseline + block_proxy_delta` (22.75–24.31), and diverges in the WRONG
   direction — the proxy predicted a DECREASE (all `block_proxy_delta<0`), reality measured an INCREASE.
   That is impossible to fabricate with proxy arithmetic. Pose deltas reconstructed by inverting
   `S=100·d_seg+√(10·d_pose)+25·bytes/37,545,489` land at exactly **+2.8% … +13.1%** — receipt-true.
2. **Code (definitive).** `launch_ddm_joint_descent._chunked_n600_verdict` (L734-771) computes both
   `d_seg` and `d_pose` LIVE via the canonical `cpu_verdict_d_seg_argmax_batch` /
   `cpu_verdict_d_pose_batch`, then `advisory_action = 100·d_seg + √(10·d_pose) + 25·bytes/…`. **d_pose IS
   realized-measured through the frozen scorer**; it is merely folded into `advisory_action` in the receipt
   rather than persisted as a standalone `realized_d_pose`. The memo already labels it "**implied** Δd_pose"
   — honest. The "5/6 bit-identical" framing is precise (the memo names the 1/6 exception, step-1 mult 1.0,
   Δd_seg +3.3e-5). Independent second-witness confirmation to fd2 (no coordination).

## S9 — VERDICT-CHAIN BLAST-RADIUS RANKING (which single link, if wrong, flips a routing)

Same-day chain with no independent replication: `sc1→W_joint · rp1→fd1 · sp1→route · fd1→(two-rung ladder)`.
Most of it is unusually well cross-validated (sp1's 421 KB is confirmed by fc1+da1+r2s; rp1's CELLS-HOLD is
double-measured by fd1 S0 at both operating points). The two links carrying real deferred risk:

**RANK 1 (highest blast) — fd1 S2 "ZERO accept → route to two-rung ladder (rung 2 = token-grid + trained
renderer)".** The day's TERMINAL routing rests on a SINGLE governed window (2 GN steps, 4-pair block 447-450,
one pose-unsolved warm start). The memo honestly scopes it `verdict_scope=INSTANCE`, but the campaign-level
"realization is binding, build the renderer" inference draws on this one window + ee1's C10 convergence theory
+ PR130's external anchor. If the ZERO-accept is block/warm-start-specific rather than engine-general, rung-1
could descend and "build a renderer" is premature. **Cheapest independent re-check:** 1-2 more GN windows from
a DIFFERENT pair block AND with the memo's own named `#383 seg-null pose-null projector` engaged on the seg
step (tests engine-general vs block-local + removes the pose-collateral gate) — ~2×1550 s ≈ 1 hr, no new
engine code beyond the projector. This IS rung-1 of the ladder.

**RANK 2 — rp1/fd1 CELLS-HOLD → "formulation not broken at uint8; sc1-far 464× is engine-capacity".** Double-
measured (GT 3.63e-4 + box-solve 3.76e-4) so well-controlled at those points. Residual gap: BOTH measured
operating points are ALREADY-SOLVED cells (d_seg ~1e-3); NEITHER measures the range-carrier realization at the
GN engine's ACTUAL warm start (d_seg 0.0702, 56× higher / more margin-eroded). fd1 uses the result correctly
(it attributes the ZERO-accept to cross-pair transfer + pose collateral, NOT to realization), but the
inference chain touches the un-probed 0.0702 operating point. **Cheapest re-check:** run the existing rp1 C1
range-carrier probe on the exact W_joint frames — confirms realization noise there is still ~3.6e-4 and not
the binding term (existing tool, ~525 s, no new code).

Both links share one theme: the terminal "realization-is-binding → build renderer" routing rests on the
single GN window + the cross-operating-point robustness of the realization probe, with the confirming
measurements AT the GN engine's own operating point deferred.

## What I checked and found CLEAN (clean bills are signal too)
- **gt_n600.npz end-to-end custody:** producer log (2026-06-26, bytes 5,078,017,610) → my SHA
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` → fd1r `run_identity.target_cache_sha256`
  → fail-closed `_verify_regular` at load. Content-sane (argmax 0-4, no NaN). Same file all arms.
- **The staleness surface (S3):** the one class most likely to be default-harmful×silent is HARD-gated —
  `_verify_regular` recomputes sha and RAISES on mismatch; symlinks rejected; byte count checked.
- **S5 centering:** exact numpy reproduction of the receipt's SVD energy fractions proves centering was done.
- **S6 C2 degeneracy:** the memo self-flags it as a structural correction, not a hidden gap; the one
  downstream over-read risk (optimistic-bound extrapolation) was CLOSED by fd1 S0.
- **Cross-arm coder agreement:** sp1 421,366 B = fc1 stage2 = da1 per-class sum = r2s support_geometry —
  quadruple-independent; sp1 also caught the phantom 142 KB projection.
- **Registry hygiene (O2):** band-lemma row carries advisory/non-promotable flags; no false authority.

## Honest boundary
- **Report-only.** I fixed nothing; every cure above routes to MAIN as an L1/L2/L3 suggestion.
- **NO scorer jobs (forbidden — a live arm owns the slot).** I therefore could NOT: independently re-run the
  fd1r window to reconfirm ZERO-accept; bit-verify gt_n600 against the LIVE scorer (S8-bound); cross-bit-
  validate the two verdict implementations (S7). These rest on receipt arithmetic + code inspection + the
  arms' own intrinsic controls, which for S1/S3/S5/S8 are decisive and for S7 are strong-but-not-cross-checked.
- **The three orchestrator steers to fd2/fd1r:** I have no access to those steer messages, so I cannot verify
  their numbers directly; I verified every number that appears in the committed receipts (S1 pose deltas,
  S4 6-frame ratio, S5 SVD, S8 SHA) as receipt-true. Filed as a boundary, not a clean bill.
- Every "SEAM-NAMED" is a suspicion WITHOUT confound-grade evidence of corruption; none is asserted as a
  CONFIRMED-CONFOUND. Pointer 0.1910828242 UNMOVED.
