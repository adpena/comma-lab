# ddm_cvg1 - Checkpoints

## Scope

- Read charter: `.omx/tmp/codex_runs/cvg1_prompt.md`.
- Read common contract: `.omx/tmp/codex_runs/_common_contract.md`.
- Read governing context: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` opening
  non-negotiables, `docs/operating_manual_craft_handoff.md`, and
  `.omx/state/main_hot_state.md`.
- Read lw1 fire order and receipt:
  `.omx/research/ddm_lw1_20260806/NEXT_IF_RESUMED.md` and `RECEIPT.md`.

## Commands / Artifacts

- Inspected endpoint n600 instrument:
  `sed -n '1,260p' experiments/ddm_jd4_endpoint_n600_both_bases.py`.
- Inspected jd endpoint receipts and window receipts under
  `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/`.
- Generated structured replay:
  `.omx/research/ddm_cvg1_20260806/replay_table.json`
  (`45,012` bytes, sha256
  `b8a5e374b99c63f26bebf4b587a3ffc46724cc0452f045f1fa935a13cbe66ced`).

No scorer run was launched. No training launch was made. No `upstream/` file was
edited. No `/tmp` persisted evidence was created.

## Result

- `d_seg`: tested controls do not pass lw1 acceptance.
- `d_pose`: apparent correction is not admitted because controls are constant
  across the three rows.
- rate: blocked by absence of same-object population-rate truth.

The replay is a scoped negative for this instance, not a family-level kill.
