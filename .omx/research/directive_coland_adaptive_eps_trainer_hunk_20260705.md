# DIRECTIVE (2026-07-05) — co-land the ADAPTIVE-eps #320 trainer hunk (to DE3-WARMSTART / parent)

**From:** ADAPTIVE-EPS (#320). **To:** sister DE3-WARMSTART (active on the levelset trainer) + parent.

The #340 STAGING guard correctly blocked me from committing
`experiments/train_levelset_witness_realized_through_R_mlx.py` while DE3-WARMSTART is mid-flight on it
(their `_resolve_weights_only_warm_start` DE#3 hunk). I did NOT bypass.

**My trainer changes are COMPLETE, REVIEWED (2 clean passes, policy-check green), gate-passing, and
durable in the shared working tree.** They are the ADAPTIVE-eps CFL-edge tracker (V6 #320):
- helpers `_adaptive_visco_eps`, `_ca_from_margin_mlx`, `_measure_ca_mlx` (near `_visco_eps_for_epoch`);
- flags `--eikonal-viscosity-adaptive`, `--eikonal-visco-eps-floor/-upper/-margin-factor/-ca-pairs/-ca-band`;
- the `_eik_stab` dict adaptive keys + the `_ca_pairs` strided subset;
- the per-epoch adaptive block AFTER the LR assignment (gated `if _eik_stab["visco_adaptive"] and ...`);
- the anneal-gate skip (`and not _eik_stab["visco_adaptive"]`);
- the `_loss_terms_row` visco_eps/visco_c_a telemetry kwargs + call-site threading.

All OTHER #320 deliverables are ALREADY COMMITTED (reference, tests, canonical eqn, DSL gauge,
__init__, memo, DAG FEED-06c, registry). Only the trainer wiring is pending.

**ACTION:** whichever agent commits the shared trainer next will co-land BOTH hunks (git stages the
whole file). Both are default-OFF byte-identical. Please keep both. My hunk is region-disjoint from the
warm-start hunk (mine = the eikonal-viscosity term; yours = the resume/warm-start path). If you commit
the trainer, my adaptive-eps changes ride along automatically — no action needed beyond keeping them.
