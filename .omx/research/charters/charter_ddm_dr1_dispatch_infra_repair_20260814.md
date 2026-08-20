# CHARTER — ddm_dr1_dispatch_infra_repair (2026-08-14, the 3-receipt repair wave)

THREE MEASURED DEFECTS from today's remote fires (all receipts on disk):
R1. SEAL-vs-SIGNATURE DRIFT: mt1's SEALED_FIRE_ORDER argv predates the
    dispatcher's current signature (--output-dir/--detach/
    --provider-detach-ack now required; fire refused "Missing option").
    Store: .../multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/
    t4_sign_gate_r1/SEALED_FIRE_ORDER.json (seal c9d6d62c…fc VERIFIED).
R2. CONTAINER IMPORT DRIFT: with corrected argv, call fc-01M01AJV3V died
    remotely — ImportError "cannot import name
    'ddm_js1b_modal_cuda_argmax_field_materializer' from 'experiments'
    (/workspace/pact/experiments/__init__.py)" + "No module named
    'experiments.modal_auth_eval'". BOTH modules EXIST on main — the
    package attribute-import chain (experiments/ddm_mt1_modal_multitoken_
    sign_gate.py:26-40 try/except ladder) breaks IN-CONTAINER. Suspect:
    the post-#936 experiments/__init__.py change × Modal mount import
    order. Reproduce the mechanism, do not guess.
R3. CLOSER DEFECT: the f26r CPU closer (tools/modal_endpoint_close.py via
    the detached wrapper) exited rc=1 at 782s with closure_manifest null
    while the call itself completed rc=0 (recover_modal_auth_eval then
    harvested cleanly). Receipt: .omx/tmp/f26r_cpu_closer.done.

## THE TASK (repair + regression-proof, then hand mt1 back to MAIN)
1. R2 FIX FIRST (it blocks the re-fire): diagnose the in-container import
   failure at source (read experiments/__init__.py history around the
   #936 fix 88dc45548f; reproduce with a minimal modal run smoke or local
   sys.path simulation of PYTHONPATH=/root:/workspace/pact/src:
   /workspace/pact/upstream:/workspace/pact). Fix the mt1 module's import
   chain to be container-robust (explicit importlib / absolute module
   path), AND fix the ROOT CAUSE in __init__.py if it breaks the general
   `from experiments import <submodule>` contract (other dispatchers use
   the same pattern — sweep for siblings).
2. R1 FIX: reseal mt1's fire order against the LIVE dispatcher signature
   (regenerate SEALED_FIRE_ORDER argv; seal builder gains a
   signature-check: refuse to seal an argv that the current entrypoint
   cannot parse — the deterministic cure per the 08-14 determinization
   binding). Payloads/SEALED_REQUEST unchanged (hash preserved).
3. R3 FIX: root-cause the closer rc=1@782s (read the full .done tail +
   tools/modal_endpoint_close.py path taken; the call was mid-run at
   782s so the closer should have kept polling — find why it exited).
   Fix + one executed positive control (closer against a completed call).
4. VERIFY: modal run DRY smoke of the mt1 entrypoint import path in a
   container (no T4, no scorer — import + seal parse only, ~$0.01 CPU) →
   green. Then STOP: MAIN fires the real T4 sign gate (do NOT fire it).

## OPTIMAL FORM
PINS: mt1 module experiments/ddm_mt1_modal_multitoken_sign_gate.py ·
seal c9d6d62c8115f6c209576a57d4cbf7e40c2191c542473fa0df33bc82af91dffc ·
#936 commit 88dc45548f · f26r closer receipt .omx/tmp/f26r_cpu_closer.done ·
call ids fc-01M01AJV3VNSZT8V51FCXT4F2G (cancelled) / fc-01M014B5F4DB…
(f26r, harvested). Reference form = the canonical dispatcher/closer pair
(experiments/modal_auth_eval_cpu.py + tools/modal_endpoint_close.py +
tools/modal_harvest_poller.py) — extend, do not fork. MECHANISM-reduction
forbidden: each fix carries a reproduced-then-cured receipt, not a guess.
No T4/scorer spend; the import smoke is the only paid call (~$0.01).
Git-blocked ⇒ memo SHA handoff.

## OUTPUT
Memo .omx/research/ddm_dr1_dispatch_infra_repair_20260814.md: per-receipt
mechanism + cure + positive control · resealed mt1 fire order path ·
sibling-sweep table (other dispatchers with the same import pattern).
Serializer commit, [no-triality] [p0-ledger-ok], no co-author trailer.
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
