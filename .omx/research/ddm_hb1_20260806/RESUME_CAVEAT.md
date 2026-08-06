# hb1 HPAC chain — RR2-F2 resume caveat (BINDING at harvest)

RR2 measured at source: the PR130 trainer's `.latest.pt` carries WEIGHTS ONLY — resuming via
`--init` warm-starts weights but resets optimizer/scheduler/RNG/epoch. A crash-resumed run is
therefore FORM-DEVIATED from their 60-epoch schedule (their own e2e is single-shot; they never
resume either). The live driver may NOT be edited while running (bash reads incrementally).

DECISION RULE (applies if a crash occurs; otherwise moot): crash before ~epoch 30 → restart
the payload CLEAN (cheap, form-faithful); crash after → accept the warm-start continuation but
label the payload's race row FORM_DEVIATED_RESUME and note total-steps-trained. HARVEST CHECK:
grep driver.log for "resume from latest" — zero hits ⇒ run is form-clean (current state: zero).
Driver-side FORM_DEVIATION logging lands at the next driver start, never mid-run.
