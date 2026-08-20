# MC36 Variant C — complete-S promotion fire order (prepared 2026-08-14 by MAIN)

The named component row (net ΔS −1.99799e-5, memo a3a2196128) owes ONE promotion
step: the full `archive.zip → inflate.sh → upstream/evaluate.py --device cuda`
chain on T4 — the same instrument that produced the lc2 authority row.

- submission_dir: `promotion_submission/runtime_stage/` (self-consistent: the
  bundled archive.zip IS the candidate, sha
  f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cad573... [full:
  f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de], 186,269 B;
  inflate chain T4-PROVEN by the dual-axis worker: rc=0, 327.6 s decode, raw sha
  a41ca69d2288d3edd8f009b03404ef070661297a8f962a067e663ff26f7c0e8b).
- Expected: S ≈ 0.16193516 (component arithmetic). Any material disagreement =
  finding, not verdict.
- Fire trigger: mt1 endpoint terminally closed (single-flight) + no other active
  Modal claim.
- Estimated cost: ~$0.25 (T4 inflate ~6 min + evaluate n600 CUDA ~15 min).
- Arm `tools/modal_endpoint_close.py` detached at dispatch (agent=MAIN,
  --allow-legacy-manifest).

## The command (lc2-precedent chain, experiments/modal_auth_eval.py)

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/promotion_submission/runtime_stage/archive.zip \
  --submission-dir /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/promotion_submission/runtime_stage \
  --inflate-sh inflate.sh \
  --expected-archive-sha256 f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de \
  --gpu T4 --scorer-device cuda \
  --output-dir /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/promotion_submission/dispatch_t4 \
  --detach --provider-detach-ack \
  --lane-id ddm_mc36_promotion_evaluate_t4_20260814 \
  --instance-job-id modal:ddm_mc36_promotion_t4_20260814 \
  --claim-agent MAIN \
  --claim-notes "MC36 Variant C complete-S promotion: named component row -1.998e-5 (a3a2196128) -> full evaluate.py authority; expected S ~0.16193516"
```

On harvest: if S < cp135's 0.16195513827824176 on the same axis, this becomes
the new effective_frontier row (pointer refresh via
tools/refresh_canonical_frontier.py) — the first frontier move bought by the
micro-edit campaign.

## FIRED 2026-08-14 (state log, MAIN)

- mt1 T4 sign-gate call fc-01M00MQ1S6ZV0E7AKD5514YR78 CANCELLED after 63 min
  queued-never-started ($0); both ledgers terminal
  (manually_terminated / stopped_cancelled_t4_queue_starvation_requeued).
  mt1 seal + payloads valid; re-fire after this promotion closes.
- The bare ::main fire refused: paired-by-default gate. Re-fired via
  tools/dispatch_modal_paired_auth_eval.py --execute.
- CUDA leg LIVE: fc-01M00RBQM4RMGEG2GXK4H8MEVX, pair_group
  ddm_mc36_promotion_paired_modal_auth_20260814T182512Z, output
  experiments/results/modal_auth_eval/..._cuda/. AC1 closer armed detached
  (pid 17018, deadline 5400 s, done-receipt mc36_promo_closer).
- CPU leg DEFERRED-TO-POST-HARVEST: the concurrent sibling fire cleared the
  modal-single-flight override (rationale quoted) but the claim tool's own
  conflict check refused (rc=5; needs --override threading the CPU wrapper
  does not forward). Sequential pairing satisfies the dual-eval mandate on the
  same archive bytes + pair_group_id; fire the CPU leg (same command shape,
  experiments/modal_auth_eval_cpu.py::main, no --gpu) immediately after the
  CUDA leg's lane closes.
