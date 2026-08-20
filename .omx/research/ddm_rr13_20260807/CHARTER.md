# ddm_rr13 — round-13 recursive adversarial review: the ARM-CAP fire sequence (counter 0/3)

**Critical-path clause:** the fire sequence that finally put a Metal training run live (fire1→5)
contains hand-corrections, rewraps, and receipt reuse that the ARM-VEH fire and the n120
dispatch will REPEAT unless reviewed now. Round-13 of the standing review chain; clean-pass
counter starts 0/3. Findings small enough to fix inline get fixed; larger ones get routed with
named owners (mx1g owns ticket-generation cures; do not duplicate its three deliverables).

**Review corpus (read ALL before writing findings):**
- Fire artifacts: `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/arm_cap_fire{,2,4,5}/run.log`
  (+ fire3 if present) — three honest refusals (governor stale-projection · ModuleNotFoundError
  'tools' · mem_probe_receipt_missing) and the successful fire5.
- `tools/mx1_fire_guard.py` — esp. `_validate_config_match` named key set, `_unwrap_safe_run`
  inner-argv binding, freshness window, host fingerprint.
- `experiments/ddm_mx1_pr130_semantic_renderer.py` — the guard-import repo-root sys.path anchor
  (~1943), the RR11-F1 probe-mode-only ticket write (~1844), mx1f chunked allocator (~1355)
  + load-phase telemetry, `run_mlx_train` checkpoint/history persistence (~1420-1465).
- `tools/safe_run.py` post-rr12 (child-pidfile, SIGTERM-safe receipts).
- The v4 ticket `.omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json` vs the
  ACTUAL fire5 argv (corrected outer wrapper `--projected-gib 15 --rss-mb 45000` around the
  ticket's inner argv) — the rewrap pattern itself is a review subject.
- Memory `concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806` legs 1-9.

**Seeded hypotheses (adjudicate each; find beyond them):**
- **F1 (HIGH candidate): guard key-set omits footprint-determining flags.** `microbatch_pairs`
  (mx1f's chunking flag) is NOT in `_validate_config_match`'s comparison set — a fire argv with
  chunking disabled/altered could pass against a chunked probe receipt, re-opening the exact
  load-spike the receipt was supposed to certify. Decide: add to the named key set (fail-closed),
  with a positive-control test (mismatched microbatch → refusal). Any OTHER footprint flag
  similarly omitted? (bits, pairs, caches ARE compared; sweep argparse for what isn't.)
- **F2: the rewrap pattern.** Firing the inner argv under a hand-corrected wrapper is legal
  (guard binds inner argv) but unaudited: the OUTER wrapper's projected-gib is then operator-
  asserted, not derived. Is there a check that the wrapper's projection ≥ receipt peak? Should
  safe_run itself read the receipt? Adjudicate; route the cure to mx1g if it belongs there.
- **F3: stale done-receipt reuse** (fire5 shows fire4's rc=9 while alive) — mx1g owns the fix;
  verify mx1g's cure covers the WATCHER side too (anything that reads `.done` to decide state).
- **F4: kill procedure** — child-only kill by pidfile is now doctrine (leg 9); verify the
  documented procedure/tooling actually uses `--child-pidfile` output and that no instruction
  surface still says `pkill -f`.
- **F5: resume-leg soundness** — appending `--resume-from` passes the named key set; verify
  resume ALSO re-enters through #254 admission + governor + a FRESH probe receipt, and that
  nothing in the entrypoint skips the in-process guard on resume.

**Protocol:** per-round findings CRITICAL/Medium/Low → fix-or-route → next round; 3 consecutive
clean rounds = SEAL. Each round answers the assumption-challenge axis explicitly. Findings doc:
`.omx/research/ddm_rr13_20260807/ROUND13_FINDINGS.md`.

**Discipline:** serializer + POST-EDIT shas; tags `[no-triality] [p0-ledger-ok]`;
review_tracker ×2 per .py; NO Claude/AI attribution or Co-Authored-By trailer — commits are the
operator's alone. Do NOT touch the live run dir or fire anything on Metal.
