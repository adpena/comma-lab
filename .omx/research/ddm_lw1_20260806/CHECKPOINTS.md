# ddm_lw1 - Checkpoints

## Intake

- Read `.omx/tmp/codex_runs/lw1_prompt.md`.
- Read `.omx/tmp/codex_runs/_common_contract.md`.
- Read governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Checked worktree status before edits. The shared tree was already dirty; this
  arm touched only `.omx/research/ddm_lw1_20260806/`.

## Paper

- Fetched/read arXiv abstract: <https://arxiv.org/abs/2608.04312>.
- Deep-read arXiv HTML: <https://arxiv.org/html/2608.04312v1>.
- Scope boundary set before transfer: deterministic Pact n=1 archive overfit is
  out of the paper's direct statistical frame; gate and campaign-layer
  estimators are in scope.

## Recall

- Searched memory registry for `lw1`, `codex_runs`, `lane`, and common-contract
  terms.
- Searched `.omx/research`, docs, reports, code for:
  `control variate`, `variance reduction`, `side information`,
  `non-orthogonal`, `excess risk`, `first-order improvement`.
- Searched subset/gate surfaces for:
  `a1_gate`, `gate36`, `prefix bias`, `stratified`, `od9`, `ffm1`, `tq1b`,
  `m88`, `m96`, `na4`, `selection_mode`.
- Read `ddm_ffm1_20260806/RECEIPT.md`,
  `ddm_na4_20260805/NA4_RECEIPT.md`, `src/tac/subset_selection.py`,
  TQ1/TQ1C receipts, JD1/JD3/JD4 receipts, and
  `experiments/ddm_jd4_endpoint_n600_both_bases.py`.
- Ran `tools/list_canonical_equations.py --json` and targeted searches over
  `CANONICAL_RESEARCH_INDEX_20260629.md` plus `sub015_DAG_*.md`.

## Decisions

- ADOPT only the control-variate replay as a falsifiable $0 retro-test.
- Preserve ffm1's queued strong-consistency replay; lw1 extends it with a
  residual correction rather than duplicating it.
- Do not promote prefix rows; every correction keeps axis and selector caveats.
- Do not retune hyperparameters in this arm; large-scale use is conditional on a
  future positive side-information test.
- Do not register a canonical equation before measured residual shrinkage.

## Verification Boundary

- No scorer forwards.
- No `upstream/evaluate.py`.
- No archive bytes created.
- No exact, contest-CPU, or contest-CUDA row.
- No `/tmp` persisted evidence.
- Markdown-only artifact; review-tracker `.py` passes are not applicable.

Own-vehicle frontier line at checkpoint:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
borrowed/unmoved.
