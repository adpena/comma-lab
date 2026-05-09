# Expand a00501f9 scope: ONE STRICT preflight check per codex finding (2026-05-09)

<!-- generated_at: 2026-05-09T11:00:00Z, from_state_hash: operator_self_protect_directive -->

## Operator directive (verbatim, 2026-05-09)

> "such bugs must be permanently fixed and self-protected against"

## Implication for in-flight a00501f9

Original prompt: codex round-3 findings fix (4 findings) + Catalog #132 for HIGH 1 only.

**Expanded scope per operator directive**: land ONE STRICT preflight check per finding (not just HIGH 1). Total catalog claims: #132/#133/#134/#135.

### Per-finding STRICT check mapping

- **HIGH 1** (verify_vast_instances stale-timestamp merge bug) → **Catalog #132** `check_locked_writes_preserve_deletions` (already in original prompt)
- **HIGH 2** (preflight Check #130 custody-gate bypass via unrelated blockers/errors) → **Catalog #133** `check_custody_gate_accept_tokens_concrete_only` — refuses any custody-validator-accept-token list that includes generic `blockers` / `errors` tokens; requires concrete validator function names OR archive_sha256/hardware_substrate/axis explicit checks
- **MEDIUM 1** (remote dispatch runbook probes local NVDEC on dry-runs) → **Catalog #134** `check_remote_dispatch_runbooks_no_local_cuda_probe_default` — refuses any `scripts/remote_lane_*.sh` that runs CUDA/NVDEC probes locally without `LOCAL_CUDA_WORKER=1` guard or `DRY_RUN` short-circuit
- **MEDIUM 2** (active_jobs_state silently resets corrupt JSON) → **Catalog #135** `check_state_writers_strict_load_for_mutating_path` — refuses any `update_*_locked` / `_save_*` writer whose load path returns empty on corrupt state without first quarantining + raising

### Strict-flip atomicity

Per CLAUDE.md NEW non-negotiable "Bugs must be permanently fixed AND self-protected against": each Check #132/#133/#134/#135 should land STRICT immediately if live count = 0 (which it should be, since the fix achieves the closure). Don't ship as warn-only-purgatory.

### Catalog # claiming

Use `tools/claim_catalog_number.py claim` per check (atomic claim). Catalog state currently at 132 (per a8bc7e79's prior claim). You'll claim 132, 133, 134, 135 sequentially.

### Tests per check

15-25 dedicated tests per Catalog #, covering:
- Positive (catches violation): synthetic violation MUST raise in strict
- Negative (allows non-violations): canonical fix pattern MUST pass
- Waiver-respect: same-line `# <CHECK_NAME_OK>:<reason>` skips check
- Edge cases: false-positive sources (similar-looking patterns that aren't violations)

### Memory + MEMORY.md

Update the landing memo `feedback_codex_round3_findings_fix_landed_20260509.md` to reflect:
- 4 fixes landed (per original)
- 4 STRICT preflight checks landed (Catalog #132/#133/#134/#135)
- 4 strict-flips applied (assuming live count = 0 across all 4)

### CLAUDE.md catalog table additions

Add 4 rows to the "Meta-bug class catalog (strict-mode preflight)" table:

```
132. `check_locked_writes_preserve_deletions` — refuses `_save_*` / `update_*_locked` functions that reload-and-merge old state instead of transactional replace; ratchets verify_vast_instances stale-timestamp class
133. `check_custody_gate_accept_tokens_concrete_only` — refuses Check #130's custody-validator accept-token list to include generic `blockers`/`errors`; requires concrete validator function calls
134. `check_remote_dispatch_runbooks_no_local_cuda_probe_default` — refuses `scripts/remote_lane_*.sh` that runs CUDA/NVDEC probes locally without `LOCAL_CUDA_WORKER=1` guard
135. `check_state_writers_strict_load_for_mutating_path` — refuses `update_*_locked` / `_save_*` writers whose load path silently returns empty on corrupt state
```

## References

- Original prompt: a00501f9 launch turn
- Operator directive source: parent message 2026-05-09T11:00Z
- CLAUDE.md NEW non-negotiable: "Bugs must be permanently fixed AND self-protected against" (lines after HNeRV parity discipline section)
- Sister: a8bc7e79's #130/#131 META gates (canonical pattern)
