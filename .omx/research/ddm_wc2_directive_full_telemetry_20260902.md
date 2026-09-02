# DIRECTIVE → ddm_wc2_qbr1_bug_wallclock_realization_audit (operator-routed, 2026-09-02)

Operator verbatim (2026-09-02, AFTER the wc2 charter was sealed): **"Need full telemetry and
signal always."** This directive supersedes the charter where they conflict, per the subagent
pre-flight rule (directives within 24h supersede the original prompt).

## Amendments to the wc2 charter

1. **Default-ON, not default-OFF.** The charter's SCOPE item 2 asked for a timing harness
   "flag-gated, DEFAULT-OFF, byte-identical when off." That default contradicted CLAUDE.md's
   own law ("'Off' is a tracked queue, never a forgotten default"): read-only score-neutral
   telemetry DEFAULTS ON; gate only on genuine compute cost, and then the CADENCE is a
   recorded decision, never a hardcoded switch. Build the harness **default-ON for future
   runs**, with a derived cadence (e.g. every-step cheap counters, every-N full stage split)
   and the cadence reason recorded in the emitted rows themselves. The byte-identity proof
   obligation is UNCHANGED (telemetry must not perturb training or coded bytes); the live
   qbr1 burn's pinned sources remain untouchable mid-burn — the ON default applies from the
   next launch that consumes the harness.
2. **Field-semantics registry is now a DELIVERABLE, not a hypothesis.** Every telemetry field
   the trainer emits (history.jsonl, milestones, and the new timing rows) gets one registry
   row: {field, producer file:line, exact meaning, units, which consumers read it}. This is
   the structural cure for the #1260 reading-semantics genus (two near-identical seg-error
   fields already caused one MAIN misread on 2026-09-02).
3. **External sampler exists — consume it, do not duplicate it.** MAIN landed
   `tools/burn_external_telemetry_sampler.py` (committed, 2 review passes, pos/neg controls
   executed) and it is LIVE on cell 1 via the canonical detached launcher: output at
   `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/external_telemetry/seed_20260902_control_native100_telemetry.jsonl`
   (60 s cadence: completed_steps, loss_total, RSS, milestone count). Use its rows for the
   wall-clock leg's aggregate curve (step-rate over time, milestone-blocking dips) instead of
   building a second sampler; the arm's per-STAGE decomposition remains its own deliverable.
4. **Gap census.** Add one typed table to the findings memo: every telemetry gap found in the
   qbr1 stack — {missing signal, cost to emit, consumer it would serve, ON/OFF recommendation
   with reason}. This feeds the standing "full telemetry and signal always" law forward into
   the wx1 Route 1 builder charter.
