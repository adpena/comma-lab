# ddm_t0r1 terminal-day T0 intake runbook

Status: `REHEARSAL-DERIVED`; do not run until the PS135 terminal `RESULT.json` exists and the
safe-run receipt is terminal. This checklist is scorer-free. It does not authorize JS1 execution.

## 1. Pin the terminal parent

```sh
T0_SOURCE=/Volumes/VertigoDataTier/pact/ddm_ps135_20260810
T0_STORE=/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight
test "$(jq -r '.complete' "$T0_SOURCE/RESULT.json")" = true
test "$(jq -r '.status' "$T0_SOURCE/gen3_resume/leg_a_resume.safe_run.json")" != running
shasum -a 256 "$T0_SOURCE/leg_a/final/archive.zip" "$T0_SOURCE/RESULT.json" "$T0_SOURCE/gen3_resume/leg_a_resume.safe_run.json"
```

Expected: all three files exist, the result is complete, the safe-run is terminal, and the hashes
become the immutable T0 source pins. A running or missing receipt is `REFUSE`.

## 2. Require the terminal producer set

```sh
test -f "$T0_SOURCE/leg_a/final/archive.zip"
test -f "$T0_SOURCE/leg_a/final/carrier.cpr1"
test -f "$T0_SOURCE/leg_a/final/coefficients.int16.npy"
test -f "$T0_SOURCE/leg_a/final/receipt.json"
test -f "$T0_SOURCE/RESULT.json"
test -f /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/lc2_int12_pose_sensitivity_map.npz
test -f /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/STAGE_C_DISPOSITION.json
```

Expected: seven zero-status checks. The final archive supplies the embedded renderer and HPAC
probability object through the same lossless section adapter rehearsed in
`t0_rehearsal_pass03`; `RESULT.json.history` supplies the convergence receipt adapter. Do not
substitute a moving pass's search chunks for the emitted terminal sensitivity map.

## 3. Prove the sensitivity map has the terminal parent

```sh
jq -e '.complete == true and .measured_map.payload.sha256 != null and .measured_map.source_chunks != []' \
  /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/STAGE_C_DISPOSITION.json
jq -r '.final.archive_sha256' "$T0_SOURCE/RESULT.json"
jq -r '.history[-1].archive_sha256' "$T0_SOURCE/RESULT.json"
```

Expected: the disposition is complete and the final two hashes are identical. If they differ,
leave `terminal_sensitivity_map` typed-unresolved; do not issue an m37 map receipt.

## 4. Bind and freshness-check the terminal objects

Use the same `bind_existing_file`, `unresolved_terminal_binding`, and
`assert_same_parent_freshness` calls exercised by the rehearsal manifest at:

```text
/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/30_CONTENT_BINDINGS_REHEARSAL.json
/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/40_M37_FRESHNESS_RECEIPTS.json
```

Write the terminal copies and manifests under `$T0_STORE/content_bindings/`. Required output is
seven `bound` roles, zero unresolved roles, exact bytes and SHA-256 for every role, plus m37
receipts for the archive selector, global-exact selector, coefficient fit, and sensitivity map.
Any unequal or absent parent is `REFUSE`; there is no waiver.

## 5. Join the actual terminal DSL activation set

```sh
.venv/bin/python tools/report_terminal_activation_join.py \
  "$T0_STORE/compiled_terminal_config.json" \
  --ledger .omx/state/lever_activation_ledger.jsonl \
  --output "$T0_STORE/activation_audit/terminal_activation_join.json"
```

Expected: return code 0, `status=PASS`, and `missing_levers=[]`. This command must consume the
newly compiled terminal config, not the v7.5.2 rehearsal control.

## 6. Admit T0 or stop

```sh
jq -e '.bound_role_count == 7 and .unresolved_role_count == 0 and .execution_allowed == false' \
  "$T0_STORE/content_bindings/terminal_content_bindings.json"
jq -e '.status == "PASS" and (.missing_levers | length) == 0' \
  "$T0_STORE/activation_audit/terminal_activation_join.json"
```

Expected: both checks pass. Only then may MAIN recompute terminal gates and continue to T1.
This runbook never changes `execution_allowed` and never launches a scorer or realization arm.

## TRANCHE-STOP ADDENDUM (2026-08-11, operator-approved pass-6 stop — READ BEFORE §2)

The solve now ends by an operator-approved pass-6 TRANCHE STOP (killed at the boundary by
`terminal_watch/boundary_stop_pass6.py`), not by the natural min-8/dry-3 convergence exit.
The runner's convergence-exit artifacts (`leg_a/final/`, `RESULT.json`) therefore DO NOT EXIST
and must NOT be forged. Substitute per this table; everything else in the runbook stands.

| Runbook expectation (natural exit)        | Tranche-stop equivalent (authoritative)                          |
|-------------------------------------------|------------------------------------------------------------------|
| `leg_a/final/archive.zip`                 | `leg_a/passes/pass_06/selected/archive.zip` (also bound in `state.json.current`) |
| `leg_a/final/coefficients.int16.npy`      | `leg_a/passes/pass_06/selected/coefficients.int16.npy`           |
| `leg_a/final/receipt.json`                | `leg_a/passes/pass_06/receipt.json`                              |
| `RESULT.json`                             | `leg_a/TERMINAL_BY_POLICY.json` (policy receipt)                 |
| `RESULT.json.history` convergence adapter | `leg_a/state.json` `history` (last row = pass 6)                 |

§3 sensitivity-map parent check: the emitted map's parent is the PASS-5-selected archive (the
chunks differentiate pass 6's INPUT), so the two hashes WILL differ — this is the runbook's own
"leave typed-unresolved" branch, and it is the EXPECTED terminal outcome, not a defect. Bind the
map PLANNING-BAND with `consumer_status=RETIRED_PER_PZ4A_20260811` (the A10/T7 active-dim recode
was refuted 2026-08-11; no live terminal consumer requires final-parent equality).

§4 required-output amendment: SIX bound roles + `terminal_sensitivity_map` explicitly
`unresolved_by_policy` (with the planning-band label above), and m37 receipts for the archive
selector, global-exact selector, and coefficient fit ONLY — the map is excluded from the m37
receipt set by this addendum. The "zero unresolved roles / no waiver" clause is superseded for
this ONE role by the operator-approved stop; every other role keeps the no-waiver bar.

Resumability note: the solve state stays resumable forever (`--resume`, tranche semantics);
passes 7+ remain available if a future window wants the pose asymptote.
