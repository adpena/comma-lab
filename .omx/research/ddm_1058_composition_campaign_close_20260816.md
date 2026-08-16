# #1058 composition campaign — CLOSE: QAT-leg compose trajectory-stopped; mp2 weight-edit family REFUSED; hv1 ep0634 stands

Date: 2026-08-16 · Owner: MAIN · Axis labels inline per row · Frontier UNMOVED this unit:
**hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]** (sha 80d9c8c6…).

STORES CONSULTED: telemetry_quality_poller.jsonl (rx2 trainer, 32 rows) + tools/select_hpac_checkpoint.py
receipt (checkpoint_selection_e960_endpoint.json) + ddm_hv1_harvest_compose.py CANDIDATES table +
three mp2 contest_auth_eval.json rows + MP2_ADVISORY_ADJUDICATION.json (this unit, sha 54228227…) +
base advisory row (ddm_mp2_relay_base_advisory_row_20260815) + wd2 refusal verdict + wd3 n120 family
disposition + mz2 recoding-class verdict + wc1 decode-wallclock verdict + memories
[[same_defect_negatives_masquerade_as_family_convergence_20260805]] +
[[cross-regime-constant-transfer-genus-finishing-stage]].

## 1. QAT continuation leg: compose-from-this-leg TRAJECTORY-STOPPED (dominated)

The 60-epoch QAT continuation (`ddm_rx2_mc36_hpac_cpu_resume`, exit 0 after 19.2 h) plateaued FLAT
from ep40 through ep60 at joint ≈134.5 kB / top1 ≈0.00193. The distortion-protected selector's
argmin over the leg is **ep46: 134,539 B @ top1 0.0019207** — worse than the already-composed
ep0634 row on BOTH selector columns (130,393 B @ 0.0018945), with a flat tail (no crossing
trajectory). Composing from this leg is dominated; the row is NOT fired.
verdict_scope: instance (this 60-epoch QAT continuation at this config). The ep0634 row stands as
the lineage argmin over all 81 retained periodic checkpoints. The early-stop question is moot —
the burn ended cleanly on its own; all checkpoints retained.

## 2. mp2 receiver-close candidates: ALL THREE REFUSED — one mechanism, family scope

All three receiver-closed sub-KB candidates completed their advisory n600 rows
([macOS-CPU advisory, env-mismatch-advisory instrument — same instrument as the base row, never a
score]). Base: d_seg 0.00042714 · d_pose 1.4747e-4 · S_adv 0.202807539287 @ 182,759 B.

| candidate | ΔB | d_pose (ratio) | seg leg | pose leg | rate leg | net ΔS | verdict |
|---|---|---|---|---|---|---|---|
| keep75∖keep87 marginal prune | −25 | 5.5551e-4 (3.8×) | +5.60e-5 | +0.036131 | −1.66e-5 | **+0.036170** | REFUSED |
| keep87 FiLM-row prune | −130 | 6.8390e-4 (4.6×) | +6.40e-5 | +0.044268 | −8.65e-5 | **+0.044274** | REFUSED |
| mixed q3/q4 quantization | −823 | 7.3123e-4 (5.0×) | +1.14e-4 | +0.047110 | −5.48e-4 | **+0.046676** | REFUSED |

Admission bar was net < −3.5e-6. The pose leg alone exceeds it by ~10,000×.

**The mechanism (one finding, measured three times per the same-defect law):** post-hoc edits to
the HPAC renderer's semantic tensors — whether precision coarsening (q3/q4) or FiLM-row pruning
(full or marginal set) — destroy d_pose 3.8–5.0× while seg barely moves and the recovered rate is
negligible. Dose-response is monotone and hopeless: the SMALLEST probe (−25 B) pays +0.0362 S, a
pose tax of ~1.4e-3 S per byte removed against a rate value of 6.66e-7 S/B — three orders of
magnitude over. There is no magnitude at which this family clears the bar.

**Family closure (composed with the sibling verdicts):** wd2 (prune + 60-epoch scorer-aware refit:
8.2× over the seg bar, pose destroyed 0.092) · wd3 (fresh-init distill family negative at n120,
parked with reactivation ladder) · mp2 (post-hoc edits without refit: pose destroyed, this memo).
verdict_scope: family — FAMILY(post-hoc semantic-tensor weight edits without joint re-descent, on
the hv1 ep0634 HPAC base, sub-KB..multi-KB magnitudes). The renderer weights are pose-load-bearing
far beyond their byte value; the weights cannot be touched after training without joint descent.
NOT covered (untested by construction): exact-algebraic carrier rank/refit, token-field edits,
warm-lineage retraining — but see §3 for the derived gate on the first.

Decode-identity bonus: keep75∖keep87's inflated raw sha (c881db66…) matches the wc1 fresh AND
cached decodes exactly — triple cross-instrument confirmation of the advisory decode path.

## 3. Route disposition after this close

1. **rfo2 rung 1 (mixed precision receiver-close): MEASURED DEAD** (this memo, family scope).
2. **rfo2 rung 2 (carrier rank/refit, 22,032 B pool): GATED, not fired.** Any LOSSY rank cut is a
   weight edit and faces the measured pose wall. Fire-condition (derived from §2): a $0
   scorer-free pre-proof must first show the rank-k semantic-field reconstruction error maps to a
   projected pose ΔS within ~2× of the bar before any ~2 h advisory row is bought (the pz4a
   pattern). Exact/lossless recodings are already dead (mz2: all ≥+340 B).
3. **Micro-edit bank union: HELD** — qs2 −4.375e-6 + re1 −1.207e-6 ≈ −5.6e-6 < the 1e-5 naming
   bar (#1044). Unchanged.
4. **The −15,157 B rate rung has NO surviving supplier** on this base. All remaining major routes
   (pose AND seg AND rate) run through the joint line: js8 implicit-joint-distortion-conditioning ·
   trained receiver (#982) · coupled multi-token (#978) — consistent with eu4's pose-dominance
   allocation (69.38% of gap).
5. **rx2 terminal harvest (ep60 neutral 100 B RCF1 table, full n600 real-rc64): OWNED, queued
   LOW** — coder-family pricing (not a weight edit; different risk class), advisory-only, fires in
   a free scorer window; its original consumer (mc36 base) is superseded so the value is the
   transferable coder measurement.

**#1058 is CLOSED.** The composed frontier row (hv1 ep0634) was the campaign's product and it
stands; every measured lever on top of it this cycle was refused or held; the fire-conditions for
the remaining gated rungs are recorded above.

## Receipts

- `MP2_ADVISORY_ADJUDICATION.json` (mp2 workspace, sha prefix 54228227…) — full arithmetic.
- Three `contest_auth_eval.json` rows under `advisory_n600_cpu/<candidate>/attempt_0000/`
  (payloads + work dirs retained per ALWAYS-KEEP-THE-PAYLOAD).
- `checkpoint_selection_e960_endpoint.json` (hv1 compose workspace) — ep46 argmin over the QAT leg.
- rx2 trainer telemetry: `training/telemetry_quality_poller.jsonl` (ep40–60 plateau).
