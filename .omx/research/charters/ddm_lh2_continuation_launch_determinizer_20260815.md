# ddm_lh2_continuation_launch_determinizer — make the local continuation-launch class AUTOMATIC, not ad hoc

## OPERATOR BINDING (2026-08-15, verbatim): "Must harden and polish and make automated instead of ad hoc."
Third recurrence of the determinization directive (08-14 ×2 → ac1/dt1 covered the MODAL side; this
charter covers the LOCAL detached-launch side). dt1's census fire-trigger is MET: five deduplicated
incident receipts share one genus with no strict cure and no active owner.

## THE INCIDENT GENUS (the five e960 fire attempts, all in ONE log — read it first)
Log: /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/launcher/run.log
Attempts 1–5 (2026-08-15, launch counters 19–23): each died to ONE fail-closed trainer gate, learned
sequentially by refusal, cured by hand at the typing moment:
  1. PYTHONHASHSEED=0 env gate → 2. TAC_ADMISSION_ENFORCE=1 env gate (+PYTORCH_ENABLE_MPS_FALLBACK=0,
  all three at reference trainer :920–927) → 3. resume-identity drift (cured by the wrapper
  continuation adapter, commit 6aee906d52) → 4. typed resume_lineage custody (cured by the receipt-file
  fix, commit c156d7fabc) → 5. LIVE (pid 47772). Every attempt: MAIN hand-typed a ~40-line launcher+trainer
  argv and hand-wrote bounded until-loop poll scripts to verify gate-chain survival. THAT is the disease.

## PROVENANCE PINS (sha256 first-16, working tree 2026-08-15)
- tools/train_ddm_cl1_hpac_capacity_mps.py ecc923430c2af084 (sealed wrapper: PORT_MODES, continuation adapter, receipt writer)
- tools/train_ddm_cl1_hpac_capacity.py 8392a9b9f2d30369 (reference trainer, READ ONLY — env gates :920–927, identity gate ~:1106, lineage validator :335)
- tools/launch_detached_process.py c09f2cf95cf2c2dd (the watched launcher; verify-alive window 3s but trainer gates fire at 30–120s — the verification GAP)
- tools/fit_hpac_descent_law.py d0a5084aa9ec6522 (the $0 endpoint refit this class owes at every burn end)
- tools/modal_endpoint_close.py cef5c9b39431ac67 (ac1's closer — THE convention to mirror: typed receipts, detached arm-at-dispatch, NEXT_IF_RESUMED extraction)
- tools/main_hot_state.py 014add6311eefaa2 (POINTER_LINE was hand-edited + stale this morning — the staleness check target)
- Parent manifest: .../gpu_race/full_e480b/launcher/launch_manifest.json (resource budgets, watcher wiring — the values MAIN copied by hand)
- Live-run manifest: .../gpu_race/full_e480b_e960/launcher/launch_manifest.json (counter 23, pid 47772)
- Watcher configs: .omx/research/ddm_wc2_hpac_mps_port_20260814/full_e480b_e960_{liveness,quality}.json (hand-derived bars)
- Memo: .omx/research/ddm_wc2_hpac_mps_port_20260814.md §5j (this morning's fire record — the full attempt-by-attempt anatomy)

## DELIVERABLES (build to admission, each with tests + executed positive controls)
(1) CONTINUATION-LAUNCH COMPOSER: tools/fire_watched_continuation.py.
    Input: parent run dir + sealed wrapper mode (+ overrides). It must:
    a. parse the reference trainer's env-gate raises AT SOURCE (never a hardcoded list — the gate
       set must track the trainer) and set/verify ALL of them pre-launch in ONE pass;
    b. pull resource budgets (peak-rss/thread-need/walltime) from the PARENT launch_manifest.json;
    c. locate the resume checkpoint (newest qat_stage_end_epoch_*.pt under the parent checkpoints
       root; refuse ambiguity);
    d. generate watcher configs from the parent's (bars derived: quality bar = parent endpoint joint
       bytes; bar_start_epoch = continuation_of_epochs+1);
    e. write the SEALED full command to <run_root>/launcher/launch.sh BEFORE firing (reproducible,
       zero hand-typing), then fire via launch_detached_process.py with --arm-watchers + done-receipt.
(2) GATE-CHAIN SURVIVAL VERIFIER (fold into (1) as a post-fire phase or a small sibling):
    poll run.log past the launcher's 3s window with TYPED outcomes:
    GATE_REFUSED(<gate name, extracted from the CL1TrainingError text>) | RECONCILED | RESUMED |
    FIRST_EPOCH_ROW(epoch, joint_bytes) | DEAD(rc). Bounded; replaces MAIN's hand-written until-loops.
    On GATE_REFUSED it must print the gate name + the known cure class (from this charter's genus map).
(3) LOCAL ENDPOINT CLOSER, armed AT LAUNCH (the local sibling of modal_endpoint_close.py — mirror its
    receipt schema conventions): a detached watcher on the done-receipt that, on fire:
    a. runs tools/fit_hpac_descent_law.py on the run's log ($0, scorer-free) → refit receipt;
    b. verifies final checkpoint exists + sha256s it (ALWAYS KEEP THE PAYLOAD);
    c. emits a typed NEXT fire-order file (endpoint → identity race → micro-edit recompile → composed
       T4 row — the #1058 chain) + PushNotification-style terminal note; MAIN adjudicates, closer never
       launches paid/scorer work (CONTAINMENT).
    DOGFOOD: arm it read-only against the LIVE done-receipt rx2_wc2_full_mps_e960 (fires ~6h from now).
(4) HOT-STATE STALENESS CHECK (warn-only): compare the S value in main_hot_state POINTER_LINE against
    .omx/state/canonical_frontier_pointer.json effective_frontier (read-only; file is gitignored
    live-state — never commit it); surface mismatch in the costate digest or a warn-only preflight.
(5) CENSUS: record the 5-attempt genus as ONE deduplicated incident (dt1's store/convention), cure =
    deliverables (1)+(2), so the recurrence trigger is discharged.

## HARD CONSTRAINTS
- THE LIVE RUN IS SACRED: pid 47772 + its watchers must be untouched; nothing you build or run may
  signal, throttle, or kill the trainer (control-plane safety — the molt lesson). Dogfood is READ-ONLY
  against its run dir; the closer arms on the done-receipt only.
- Reference trainer tools/train_ddm_cl1_hpac_capacity.py is READ ONLY (sealed, sha-pinned).
- No scorer, no Modal, no training. Apparatus arm; no score row expected — say so at close.
- Serializer commits w/ post-edit working-tree SHAs, [no-triality] [p0-ledger-ok]; .py = 2 review
  passes; ruff + compile; memo .omx/research/ddm_lh2_continuation_launch_determinizer_20260815.md.

## OPTIMAL FORM
- Family reference: ac1's tools/modal_endpoint_close.py (typed receipts + armed-at-dispatch closer,
  commits 4f4537d835/b4404f9fa3/8a3207e10e/f204c8fcb6) + lh1's watched-launch hardening (#1057).
  This charter is the LOCAL-launch member of that family at the family's landed form — not a sketch.
- SCOPE reduction (legal): first target = the HPAC continuation class (rx2/wc2 lineage) only;
  generalization to arbitrary trainers is a named hook (gate-parse + manifest-pull are the only
  trainer-specific seams), not built now.
- MECHANISM reductions: NONE. Dead-ends inherited from dt1/ac1 (closed, do not reopen): print-only
  runbooks (preserve the manual failure class) · a second poller duplicating the canonical poller ·
  treating local poll deadlines as process failure.
- Provenance pins: see section above (paths + sha16 for every reused component).
