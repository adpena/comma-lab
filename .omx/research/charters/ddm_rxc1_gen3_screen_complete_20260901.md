# ddm_rxc1_gen3_screen_complete — resume the rxc1 gate-1 screen to 32/32 + manifest + the preregistered cost adjudication (task #1374 SCMDL gate 1; fire order from arm final message ddm_rxc1_gen2_screen_resume_20260901T143849Z.md and vr2's READY note ddm_vr2_ap_reclaim_round2_20260901.md)

## MANDATE

The recorded gen-2 fire order is now TRIGGERED: vr2's certified reclaim left APDataStore at
52,713,881,600 free bytes (≥1,400,000,000, no concurrent decline — vr2 finished; MOVE_CERT
`6a5173ff…4fa7df`). Gen-2 exited honest `BLOCKED(storage-reserve)` at 26/32 sealed rows
(26/26 byte-identical at strides 200 AND 300; exact-vs-exact correlation correctly ruled
VACUOUS per prereg — no gate authority claimed; SCREEN.json/MANIFEST.json correctly absent;
blocker receipt `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BLOCKER.json` sha
`581a0768…d839b`; memo commit 463d2cbe03). This arm completes the screen, runs the manifest
stage, and performs the preregistered cost adjudication — the campaign's gate-1 verdict on
whether the restartable exact coder is a usable SCMDL pricing instrument.

## SCOPE

1. **Verify the trigger at start**: measure AP free bytes; refuse to start if <1.4e9 or
   declining across two samples 60 s apart (record both samples).
2. **Resume the screen**: `.venv/bin/python experiments/ddm_rxc1_restartable_exact_coder.py
   --stage screen` — resumes from the durable receipts + frame-400 restart point (gen-2's
   live hypothesis: the 26 sealed rows are NOT recomputed; verify that from the run's own
   receipts and refuse to silently recompute). Projected ~6 rows × ~479–714 s/proposal per
   stride ≈ 1–2.5 h → run DETACHED via `tools/launch_detached_process.py` (script path must
   avoid claude/codex tokens — fleet-reaper argv predicate), pidfile + done-receipt, monitor.
3. **Manifest stage**: on screen completion (SCREEN.json present with the n=32 denominator),
   run `--stage manifest`.
4. **Preregistered cost adjudication**: quote the rule from
   `.omx/research/ddm_rxc1_preregistered_harvest_adjudication_20260901.md` VERBATIM, compute
   the adjudicated quantities from SCREEN.json/MANIFEST.json, and state which branch (1/2/3)
   fired — no post-hoc reinterpretation; the prereg text is the authority. Consume gen-2's
   partials honestly: byte-identity 32/32-or-not · median s/proposal per stride · terminal
   adaptive-state reconvergence (gen-2: 0/26; the splice question closes for this instance
   if the remaining 6 confirm).
5. **Route the verdict**: the SCMDL X-alone axis is now CLOSED (sg2b falsifier 3/3, memo
   ddm_sg2b_falsifier_verdict_20260901.md) — gate 2 is purified to G/M-coupled candidates.
   State explicitly what the gate-1 branch means for pricing the 3 xov1-folded candidates
   (born expert · generator-conditioned peel chain · 5,506-record B/H/W support, memo
   ddm_xov1_crossover_pass_20260901.md) and for jc1's own G/M refit.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer runs (the screen is coder-side, $0).
- OWNERSHIP: this arm now owns `experiments/ddm_rxc1_restartable_exact_coder.py` +
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` (gen-2 exited; vr2 finished).
  Do NOT touch sg2b stores or `/Volumes/APDataStore/pact/ddm_xov1_crossover_pass/` (READ-ONLY).
- Preserve the mandatory 1 GiB AP reserve exactly as gen-2 did — stop safely, never breach.
- ALWAYS KEEP THE PAYLOAD: every sealed row's streams/receipts retained; no scalar-only runs.
- Serializer commits w/ post-edit `--expected-content-sha256`; on a .git/objects denial use
  the serializer's auto-bundle fallback (rc=17) and name the bundle for MAIN (the #1293
  mechanism — vr2 hit it this same day).
- DETACHED >30-min compute per the canonical launcher; monitor, never in-session loops.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Gen-2's own DEAD-ENDS bind verbatim: exact-vs-exact correlation is VACUOUS by prereg
  (never gate evidence) · no GATE-1-PARTIAL claims from <32 rows · terminal frame-600
  reconvergence closed 0/26 for this instance · never continue into the storage reserve.
- #1210 stale-precondition genus: verify AP free AT START (constraint 1), never trust this
  charter's numbers as current.
- qs4/#1039 cross-regime constant transfer: every consumed artifact sha-verified at load.

## OPTIMAL FORM

- Family exemplar: gen-2's own run (463d2cbe03) is the reference form — same tool, same
  custody schema, same refusal discipline; this arm is its continuation, not a variant.
  Provenance pins: BLOCKER.json sha `581a0768…d839b` · vr2 MOVE_CERT sha `6a5173ff…4fa7df`
  (memo commit 4b1cf978d3) · prereg memo ddm_rxc1_preregistered_harvest_adjudication_20260901.md
  (quote its own sha in the deliverable) · sg2b verdict memo (this session).
- SCOPE reductions legal: none needed — the remainder is 6 rows + manifest. MECHANISM
  reductions FORBIDDEN: full byte-identity checks on every new row, both strides, real
  receipts; no sampling, no extrapolated medians.
- **PRIOR-LAW PREDICTION (falsifiable):** gen-2's live hypothesis — the remaining 6 rows
  preserve exact byte-identity (32/32) because all 52 sealed comparisons were identical and
  checkpoint restore is proven. FALSIFIER: any new row deviates — then the identity claim is
  PARTIAL-scoped and the adjudication must consume the split honestly.

## DELIVERABLE

`.omx/research/ddm_rxc1_gen3_gate1_verdict_20260901.md` — typed rows: trigger verification
(both AP samples) · screen completion table (32/32, per-stride medians, identity) · manifest
receipt · the VERBATIM prereg rule + computed quantities + the branch that fired · gate-1
verdict + its consequences for the 3 xov1 candidates and jc1's G/M refit · DEAD-ENDS +
denominator. Commit via the serializer (bundle-fallback on denial). End with the own-vehicle
frontier line (S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha
cbb8d928…d405bf25 — UNMOVED unless a fire order lands).
