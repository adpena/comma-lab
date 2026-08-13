# ddm_cp5v — Compose the 5 n600-VALIDATED events into ONE +≤3 B archive → READY_TO_FIRE the final exact row

**Charter date:** 2026-08-12 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN fires the final contest-CUDA T4 row (~$0.15, #381) vs cp135 S 0.16195513827824176.
**Parents:** ddm_vd1 run-c census (PASSED, fc-01KZWM334XKMRQE4YFJ5PX3574) + ddm_jo1 (commit `3bc2cb557f`, the +≤3 B HP3/RC64 carrier proof). Read `docs/operating_manual_craft_handoff.md` + CLAUDE.md/AGENTS.md first.

## The measured input (n600 authority, per-event exact, T4)

Census: `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/FINAL_RESULT.json`
(sha256 `6c53628184f55722f87fcb7e3dadc8b6c9a70025a804e00cfcbecb6674004973`) + `EVENT_RESULTS.jsonl` (sha `a97400d328…`).

The 5 downstream-selection-eligible events (flip-positive ∩ pose-budget-pass, additive pose 6.54e-09 ≪ the 1.3e-7 global budget):
`ec1_0164_3a4e239de5b9 · ec1_0168_818a3c77af51 · ec1_0004_3bc2b69c706c · ec1_0104_f4e219067530 · ec1_0003_fcb5ca3a4453`
Optimistic additive seg gain: +6 flips = 5.086263020833333e-06 S. The ec1 gen-1 falsifier FIRED on projection (bar 0.000216) — the campaign routes to gen-2 separately. THIS arm's purpose is narrower and still binding (`main_final_exact_row_required: true` in the census contract): close the loop with authority AND calibrate additivity-through-exact-compose — the exact step where js7 died (projection −0.00058, realized +0.00147).

## The job

1. **Thin compose driver on jo1's LANDED machinery** (`experiments/ddm_jo1_joint_probability_object.py`, memo `.omx/research/ddm_jo1_joint_probability_object_20260812.md`): take an explicit event-id list (the 5 above), event store `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`, base = the cp135 composed archive `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip` (sha `6eb1a3b7…`, 186,252 B) → ONE archive via the HP3/RC64 probability-object encoding (jo1 proved 200 events → +3 B; 6 events → +1 B; 5 events must land ≤ +3 B, expect +1 B).
2. **Receiver-close**: decode the composed archive through the ADAPTED runtime's canonical reader (vd1b's proven path, `cpr1/` + Brotli bootstrap — reuse `experiments/ddm_vd1_batch_event_validator_worker.py`'s loader seam). Verify the decoded token plane differs from the base plane (base sha `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`) at EXACTLY the 5 events' token cells and nowhere else. Deterministic repeat → byte-identical archive.
3. **P0 KEEP THE PAYLOAD**: retain the composed archive + decoded token plane sha + per-event diff receipts under `/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/`.
4. **Arithmetic sheet**: expected S = 0.16195513827824176 − 5.086e-06 + 25·Δbytes/37,545,489 (state Δbytes measured). Also state the ADDITIVITY calibration readout the exact row will provide: realized-vs-sum-of-singletons on the 6 affected pairs [7,18,53,73,76,96].
5. **Do NOT dispatch Modal.** Final message = `READY_TO_FIRE` + the pinned exact-eval command through the CANONICAL submission chain (the same `experiments/modal_auth_eval.py` chain that bought cp135/lc2/js7 rows — never a probe script) + archive path/sha/bytes.

## Custody / constraints
- Base archive + adapted_runtime READ-ONLY. Serializer commits with post-edit working-tree shas. No scorer runs, no MPS. Advisory only; the exact row is MAIN's.
- Falsifier: if the 5-event compose exceeds +3 B or the token diff is not exactly the 5 events' cells, STOP and report the mechanism (do not widen the budget).
