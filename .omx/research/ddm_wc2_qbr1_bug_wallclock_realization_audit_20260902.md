# DDM WC2 — QBR1 bug, wall-clock, and realization audit

**Status:** ACTIVE AUDIT — mandatory fire alarm issued; remaining static timing,
realization, staged-patch, and harness evidence is being completed.

**Own-vehicle frontier:** UNMOVED. This audit has produced no new byte-closed
archive and no exact `upstream/evaluate.py` row.

## FIRE ALARM

### WC2-F1 — `BURN-INVALIDATING`: the executable EMA law differs from the sealed law

The authorized control config resolves `ema_decay_run_geometry_v1` in its
constant-decay seed-fraction mode: 5,000 updates, target terminal seed fraction
0.01, and resolved decay `0.9990793899844618`. The burn entry point nevertheless
constructs `tac.training.EMA(..., warmup=True)`. Its executable law is
`min(d, (1+t)/(10+t))`, not the registered constant-decay law `d`.

The difference is material and closed-form exact:

- declared terminal initialized-shadow coefficient: `d^5000 =
  0.010000000000000278`;
- executed warmup coefficient: `product(t=1..5000, (t+1)/(t+10)) =
  1.838001854879489e-27`;
- the configured cap is not reached until update 9,767; at update 5,000 the
  effective decay is only `0.9982035928143712`.

Milestones and candidate materialization consume the EMA shadow. The running
cell therefore measures a different intervention from its sealed config. MAIN
must hold all later QBR1 cells at the next durable cell boundary, must not
promote this cell as the sealed constant-decay treatment, and must first apply
the staged pinned-source cure and rerun resume identity. The machine-readable
alarm is retained at
`/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/WC2_FIRE_ALARM.json`.

The audit made no write to the live burn.

## SOURCE-PIN PREFLIGHT

All 20 entries in the authorized config's `source_pins` map matched both their
sealed byte counts and SHA-256 digests before adjudication. This includes the
5,078,017,610-byte `gt_n600.npz` cache. The burn entry point itself matched
48,397 bytes and SHA-256
`0c143eb232b8f8494756310f0c47b001d350c145af7f33568d01a57f23adb66e`.

## RECALL EVIDENCE

Pending consolidation into the completed audit.
