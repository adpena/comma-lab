# CHARTER — ddm_qs3_saturation_compose (2026-08-13, operator steer "We can still iterate and optimize further")

ITERATE THE COUPLED FAMILY TO A SUPER-BAND COMPOSED CANDIDATE. qs2 won the coding race
(5.67 B/pair, density 0.941 flips/B > 0.785 breakeven, projected −4.37e-6 — R2 dispatch
in flight, verdict will land at
`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/dispatch/ddm_qs2_dual_axis_20260813_r2/QS2_T4_REMOTE_RESULT.json`
— consume it as CALIBRATION when it appears). But −4.37e-6 is SUB-BAND on the evaluate.py
canonical (±3.5e-6/side): a claimable floor move needs |ΔS| > ~1e-5. The three levers
compose multiplicatively — build the candidate that uses ALL of them.

**PRIOR-LAW PREDICTION (m38, composition arithmetic):** at qs1's measured prices, raising
realization efficiency 16.9%→~50% at constant bytes gives ~95 flips @ 34 B ≈ 2.8 flips/B
→ net ≈ −5.8e-5 (13× the R2 projection, SUPER-band). Scaling admitted pairs from 6 toward
the screened js6 bank (200 rows, re-screened at calibrated prices) multiplies further;
shared-codebook coding amortizes BELOW 5.67 B/pair at scale. Prediction: a composed
candidate with worker-verified ΔS ≤ −2e-5 exists. If the post-mortem refutes the
efficiency headroom (e.g. reverted pixels are dominated by the js5 uint8 quantum FLOOR,
not engineering), say so plainly with the measured classification — that ceiling is
itself the finding.

**OPERATOR DOCTRINES:** efficiency + coding costs are IMPROVABLE first-pass numbers,
never constants · "no naive or toy or generic basis ever" · payload law (retain
everything) · byte-closed-row cadence · errata law: sub-1e-5 deltas adjudicate ONLY on
the full-precision worker instrument; canonical rows only for super-band candidates.

## THE WORK (ordered)
1. **Per-pair post-mortem — the data is LOCAL now**:
   `/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/retained_fields/ddm_qs1_dual_axis_20260813_r2/`
   (seg argmax fields + pose 6-vectors + inputs; FINAL_RESULT.json; ~6.5+ GiB; verify the
   download's .done receipt `.omx/tmp/codex_runs/qs1_field_download.done` before trusting
   completeness). NO SegNet rerun. Classify every one of the ~157 reverted pixels:
   {sub-quantum amplitude vs the js5 uint8 quantum floor · AA/resize washout through the
   shared D · tie-margin failure · other}. Deliver the classification TABLE + per-pair
   marginal flips/B waterfill order.
2. **Survival-engineered proposals** (hr1/rvs1 playbook): amplitude above the quantum
   floor, margin-targeted placement, D-aware pre-compensation. Consume ddm_gca1's
   edit-propagation bounds IF its memo has landed (check
   `.omx/research/ddm_gca1_graph_calculus_crosswalk_20260813.md`). Report the efficiency
   CURVE per variant against the post-mortem taxonomy.
3. **Re-screen at calibrated prices**: the full js6 bank (200 rows) + NEW proposals from
   the survival-engineered generator, screened at (calibrated efficiency × measured
   coding cost × Schur pose compensation). Admission bar per pair: marginal flips/B >
   0.785 · (1 + pose_S/rate_S).
4. **Coding at scale**: shared codebook across the admitted pair set (amortize the
   codebook; beat 5.67 B/pair), qs2's dead-zone step-2 (+9 B, pose-improving) re-raced in
   the composed context.
5. **Waterfilled compile**: ONE composed candidate through the HP3/RC64 closure (count
   every byte; retain everything). Target worker-verified |ΔS| ≥ 1e-5; stretch −5e-5.
6. **Sealed dual-axis fire-order** (js6b worker UNCHANGED, single candidate leg per the
   R2-proven contract; worker SELF-CLAIMS its lane; MAIN fires ~$0.16). Pre-encode
   admission: net realized ΔS < 0 on matched worker instruments (base = po1/pz4r
   worker-family pair: 34,970 flips · d_pose 6.885642960696714e-6). If super-band clears
   on the worker → NAME the follow-on canonical evaluate.py fire-order for MAIN (the
   floor move) — do not fire it yourself.

## OPTIMAL FORM
Iteration arm on a PROVEN engine + PROVEN coder. Reference forms (ADAPT, provenance-pin):
qs2's coder stack + closure (commit `d77fb69efc`, store
`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/`) · qs1 compile workspace
(`/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/`) · the js6 bank. Instrument UNCHANGED
(mechanism changes = TOY-BRACKET). SCOPE reductions legal; declare them.

## OUTPUT
`.omx/research/ddm_qs3_saturation_compose_20260813.md` + code/tests + retained store
(`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/`) + sealed (no-)fire-order. Commit via
`tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality] [p0-ledger-ok]`,
no co-author trailer). End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
