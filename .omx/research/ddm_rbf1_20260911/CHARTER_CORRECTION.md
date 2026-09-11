## CORRECTION (MAIN, 2026-09-11 ~05:10Z) — the wall-clock slack premise: READ FIRST
mxo1 (`.omx/research/ddm_mxo1_free_decode_time_online_context_mixing_20260911.md`, sha 60b5a34bb6146af8…) MEASURED the strict T4 receiver slack under the contract's 1,260 s t4_direct ceiling: **27.581 s (233.8 ns per
coded symbol at n600)**, not the ~540 s this charter assumed (that figure was derived from the 1,800 s contest budget, the wrong ceiling).
Consequence: any decode-time model or render-time treatment you add must cost ≤ ~27 s at n600 on T4 (projected from this host by the leg's
factor) or it is closed on wall-clock alone; a receiver that exceeds 1,260 s fails the timing gate regardless of bytes. Price wall-clock FIRST,
then value. Also: move 44 already runs a 23-family PAQ-style online adaptive corrector — do not re-derive online adaptation as new.
