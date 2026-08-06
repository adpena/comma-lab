# ddm_ty2 checkpoints

## Scope

- Charter: `.omx/tmp/codex_runs/ty2_prompt.md`
- Common contract: `.omx/tmp/codex_runs/_common_contract.md`
- Output directory: `.omx/research/ddm_ty2_20260806/`
- Score authority: none used by TY2
- `score_claim`: `false`

## Inputs Read

- `PROGRAM.md`
- `CLAUDE.md`
- `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/research/ddm_ty1_20260806/RECEIPT.md`
- `.omx/research/ddm_ty1_20260806/TOY_LEDGER.jsonl`
- `.omx/research/ddm_tk1_20260806/RECEIPT.md`
- `.omx/research/ddm_tk1_20260806/semantic_stream_race.json`
- `.omx/research/ddm_tk2_20260806/RECEIPT.md`
- `.omx/research/ddm_sw1_20260806/RECEIPT.md`
- `.omx/research/ddm_dk1_20260806/RECEIPT.md`
- `.omx/research/ddm_et3_20260806/RECEIPT.md`
- `.omx/research/ddm_rw2_20260806/RECEIPT.md`
- `.omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md`
- `.omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json`
- `.omx/research/pr86_pr130_fullstack_intake_20260728.md`
- `.omx/research/codex_findings_ddm_p1_frame0_pose_quotient_carrier_20260725T143303Z_codex.md`
- `.omx/research/codex_findings_ddm_v14_realization_fidelity_20260722_codex.md`
- `.omx/research/scorer_batch_20260804.md`
- `.omx/research/AR9_RECEIPT.md`
- `.omx/research/ddm_cg1_force_class_edge_ledger_20260803.jsonl`
- `.venv/bin/python tools/list_canonical_equations.py --json`

## Invariants

- No scorer/evaluator job launched.
- No `upstream/` edit.
- No protected-file edit.
- No staged index or stash touched outside the serializer commit step.
- No `/tmp` evidence persisted or cited.
- No external PR bytes/weights/constants imported.
- PE3 kept as runtime-survival-unmeasured.

## Artifacts

- `SYNERGY_HYBRID_LEDGER.jsonl`: 10 typed rows, 4 hybrid and 6 synergy.
- `RECEIPT.md`: ranked top rows, recall evidence, boundaries, and outcome.
- `NEXT_IF_RESUMED.md`: fire order and fold/stop rules.

## Validation Run

- `jq -c . .omx/research/ddm_ty2_20260806/SYNERGY_HYBRID_LEDGER.jsonl >/dev/null`: passed.
- Required-field check over `row_type`, `members`, `weakness_coverage`, `mechanism`, `s_stakes`, `byte_accounting`, `fire_order`, and `cost`: passed with 0 missing rows.
- Row count check: passed with 4 hybrid rows and 6 synergy rows.
- Search for forbidden authority claims and durable temporary-directory evidence paths: passed.
- `shasum -a 256 .omx/research/ddm_ty2_20260806/*`: run after edits for serializer input.

This file intentionally does not embed the serializer commit hash to avoid a self-referential commit update. The final handoff should report the commit hash after the serializer succeeds.
