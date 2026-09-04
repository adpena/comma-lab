# Burn-cell fire scripts (git copies of the SSD executing copies)

Policy (operator 2026-08-20, [[m150]]): code lives in git; the SSD holds artifacts only. Each directory here is a byte-copy of the
fire/authorize scripts that MAIN executes from `/Volumes/APDataStore/pact/ddm_<arm>/fire/` (the executing copy stays on the SSD because
a running `bash` reads its script incrementally — never edit an executing copy). Provenance: ng2 (area cap, 2026-09-04), ng3 (τ band,
2026-09-04; inline `vm_stat` admission — superseded), ng4 (continuous objective, 2026-09-04; admission through
`tools/cell_admission.py admit --candidate-peak-gib 45`, the gv1 governor, and a detached wait-then-fire waiter that requires the owed
smoke receipt plus three consecutive admits). The `authorize_*.py` scripts bind claim ids through the chain driver's own
`authorized_config` / `write_or_verify_authorized` — never a hand-edited JSON.
