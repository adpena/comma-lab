# HR1 durable follow-on backfill — ddm_gc21

Canonical typed provenance: `HR1_ROUTING.jsonl`. This file is a bridge into the
costate-readable arm queue, not a second disposition engine.

## NEXT_IF_RESUMED

- `gc21_rank3_fire_n120_receiver`; disposition=DEFERRED; owner=M1 Metal-lane controller owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=All M1R5 defects cured, fresh 3/3 seal, refreshed mem probe, and passing fire guard.; action=Fire the n120 receiver only after the clean-pass seal; 3250 is forecast only.; source=.omx/research/ddm_gc21_20260808/GC21_CONVOCATION.md (line 262, commit 0df50fcbf2f05bc704a509edda0f3f8432f5d7be, sha256 15f6d2febc23e7eb779ebaa93d902d7470aec612a9a4c6bba54cd9f6de1d06ee).
- `gc21_rank4_renderer_backward_kernel`; disposition=QUEUED-WITH-FIRE-ORDER; owner=WC2 renderer-kernel owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=Parity passes and same-object whole-step speedup is at least 1.25x at a checkpoint boundary.; action=Build renderer-backward kernel in parallel; adopt at checkpoint resume only after parity and whole-step bench.; source=.omx/research/ddm_gc21_20260808/GC21_CONVOCATION.md (line 263, commit 0df50fcbf2f05bc704a509edda0f3f8432f5d7be, sha256 15f6d2febc23e7eb779ebaa93d902d7470aec612a9a4c6bba54cd9f6de1d06ee).
