# ddm_t0r1 — pass-03 T0 intake rehearsal

Tags: `[no-triality]` `[p0-ledger-ok]`  
Axis: `[scorer-free byte/custody apparatus]`  
Verdict scope: `INSTANCE(PS135 pass_03 selected objects)`  
Score claim: false  
Pointer moved: false

## Conclusion

The full T0 intake chain ran against the selected pass-03 object shapes. Six of HR2's seven roles
bind with stream-hashed custody; `terminal_sensitivity_map` correctly remains typed-unresolved.
The pass-03 search chunks are a real map-shaped payload, but they differentiate the pass-03 input
archive `b8c3b1187...`, not the selected pass-03 archive `93f8d7b4b...`. Binding that map to the
selected archive would be a stale-parent fake.

The charter prediction was partly falsified. It predicted at least two roles with no producer.
Only one role is unresolved in this rehearsal, and even that role has an existing terminal producer
in `emit_sensitivity_and_stage_c_disposition`; it is unavailable only because pass 3 was a moving,
non-terminal pass. Convergence receipts are losslessly derivable from the ordered pass receipts.
Renderer and probability-object bytes are losslessly extractable from the selected archive.

## Demand list — 7/7 roles censused

| Role | Rehearsal disposition | Evidence and terminal demand |
|---|---|---|
| `terminal_archive` | BOUND direct | Copied pass-03 selected archive, 187,222 B, SHA `93f8d7b4b668...`. |
| `terminal_renderer` | BOUND through adapter | Lossless CX2 decode + TM1 split + semantic-section extraction, 40,252 B, SHA `9b98360bd569...`. Repeat the adapter on the final archive. |
| `terminal_carrier` | BOUND direct | Copied CPR1 carrier, 23,050 B, SHA `a532057d6c78...`; extracted archive section matched byte-for-byte. |
| `terminal_coefficients` | BOUND direct | Copied `(600,12)` int16 NPY, 14,528 B, SHA `2daec0ae99e8...`. |
| `terminal_probability_object` | BOUND through adapter | Lossless CX2 decode + TM1 split + trailing HPAC-base extraction, SHA `b07fff73fac4...`. Repeat on the final archive. |
| `terminal_convergence_receipt` | BOUND through adapter | Ordered projection of complete pass 1–3 receipts, ending at archive `93f8...`; terminal adapter consumes `RESULT.json.history`. |
| `terminal_sensitivity_map` | TYPED-UNRESOLVED | Retained map shape is `(600,6,12)` Jacobian, `(600,12)` GN update, `(600,3)` active dimensions, but parent is pass-02 archive `b8c3...`. Bind only the terminal-emitted map after its final-parent equality gate passes. |

There is no missing terminal producer to add. MAIN must retain the existing final sensitivity-map
emission and must not move it earlier in the solve. The only required change is intake discipline:
verify the emitted map's source-pass parent equals the final selected archive before binding it.

## Ordered-item receipts

- Role census: **6 bound / 1 typed-unresolved / denominator 7**.
- Binder: executed through `bind_existing_file` with expected bytes and SHA-256 for all six
  admissible roles; no fake path or placeholder was introduced.
- m37: **3/3 PASS** for the selected archive receipt, global-exact selector, and coefficient fit.
  The stale sensitivity map has a separate explicit refusal receipt.
- Activation join: clean CLI invocation passed **9/9** non-default v7.5.2 levers against the live
  ledger, with zero missing rows. This is only the required positive control; terminal day must
  run the same CLI on the actual terminal compiled config.
- T0 runbook: `.omx/research/ddm_t0r1_terminal_day_runbook_20260811.md`.

All retained rehearsal outputs live at
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/`.
The retained tree contains 615,640 bytes across 19 records before its own tree manifest. The selected archive and all
pass-03 source pins were unchanged after the rehearsal. Mutable live `state.json` and safe-run
receipts were copied byte-exactly into `source_snapshots/` so later solve updates cannot stale the
rehearsal record.

## RECALL EVIDENCE

Searched the full `.omx/research/` memo/receipt corpus and the canonical research index/DAG with
`terminal_sensitivity_map`, `terminal_convergence_receipt`, `content binder`, `same-parent
freshness`, `activation-ledger terminal join`, `ps135`, `hr2`, and `ip1`; queried the canonical
equations listing for `ps135|fresh|parent|activation|convergence|sensitivity`; and checked the live
hot state plus canonical task-status/harness surfaces for T0/PS135 ownership.

Beyond the charter seeds, the source runner proved two load-bearing facts that changed the plan:

- `emit_sensitivity_and_stage_c_disposition` already emits the terminal consolidated map, so adding
  a second producer would duplicate and risk drifting from the actual solve.
- Each moving pass's sensitivity chunks are generated from its input state. On pass 3 the input
  archive is pass 2, while the selected archive is new. This changed the proposed map adapter from
  BOUND to typed-unresolved and added the explicit final-parent equality gate.

The canonical equation `radius2_multistart_singleton_escape_v1` independently records the pass-2
PS135 lineage and its advisory-only boundary. No canonical equation supplied a newer T0 binder or
same-parent adapter. The research corpus confirmed that the activation join must use explicit
compiled lever names rather than `never_fired()` summaries.

## Boundaries and non-measurements

- Scorer-free: no SegNet, PoseNet, evaluator, MPS, Modal, GPU, or paid job ran.
- The live solve tree was read only. Pass 4 continued independently; no live file was moved,
  rewritten, locked, or consumed as a terminal object.
- The terminal store was not touched. Every created payload is retained in the separate rehearsal
  store with SHA-256 custody.
- The retained sensitivity diagnostic is not a terminal role, score surface, or candidate.
- No exact score, distortion component, or frontier row was measured.
- No Python source changed, so the `.py` review gate was not invoked. Unrelated worktree and staged
  state were preserved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: PS135 terminal producer; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/`; fire trigger: Leg A reaches its real convergence condition and writes final `RESULT.json`; action: retain the existing terminal map emission and its source-chunk disposition.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 content-binding owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/content_bindings/`; fire trigger: terminal archive, result, safe-run, and sensitivity disposition are all complete; action: execute the terminal-day runbook, bind all seven roles, and refuse unless the sensitivity parent equals the final archive.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/activation_audit/`; fire trigger: the exact terminal DSL config compiles; action: run the activation-join CLI on that config and resolve every missing lever before T1.

## LIVE-HYPOTHESES

- The existing terminal sensitivity producer will bind cleanly after convergence because the stop
  requires dry passes, making the final pass input and selected archive identical; the exact hash
  equality remains untested until terminal emission.
- The renderer and HPAC extraction adapters should remain byte-stable on the final archive because
  PS135 freezes those sections and mutates only the CPR1 carrier; final parse-back must still prove
  this rather than inheriting the pass-03 hashes.
- The actual terminal activation join may expose rows absent from the v7.5.2 positive control
  because JS1 compiles a different active-lever set; this is why the 9/9 rehearsal does not authorize
  terminal execution.

## DEAD-ENDS

- Binding pass-03 sensitivity chunks to the selected pass-03 archive is closed: their producer
  parent is the pass-02 archive, so m37 correctly refuses the join.
- Adding a new sensitivity-map producer is closed: the PS135 runner already emits the correct map at
  terminal, and a second producer would create drift rather than readiness.
- Treating convergence receipts as absent is closed: the ordered complete pass receipts produce a
  distinct, hash-custodied curve without a scorer replay.
- Treating embedded renderer or HPAC bytes as missing is closed: lossless receiver parsing extracted
  both and the semantic/carrier sections matched the selected parse-back receipt.
- Treating the activation positive control as terminal authorization is closed: it proves only that
  the CLI can join a real current config; terminal JS1 must supply its own config.
- Treating this rehearsal as frontier movement is closed: own-vehicle frontier remains **S = 0.16959899569230852 @ 187,226 B** `[contest-CUDA T4, adjudicated, n600]`.
