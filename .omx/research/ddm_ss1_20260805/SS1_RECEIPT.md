---
schema: ddm_ss1_subset_scope_censoring_sweep.v1
date_utc: 2026-08-05
arm: ddm_ss1
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[scorer-free static/source/receipt audit]"
tokens: [no-triality, p0-ledger-ok]
---

# SS1 - subset-default verdict-scope sweep

## Answer First

SS1 landed the #875 instrumentation extension and a live-source inventory. It did
not move a score row, did not run frozen scorer forwards, did not launch, and did
not run `upstream/evaluate.py`.

Existing coverage found before this landing:

- `src/tac/subset_selection.py` already provides the canonical selector and
  provenance surface.
- `src/tac/subset_selection_gate.py` already guards newly added prefix slices in
  staged Python diffs and is wired through `tools/preflight_hook.py`.
- The uncovered gap was verdict/receipt emission: a tool can have an under-sampled
  subset default and still emit a result without structured subset scope fields.

This landing adds `tools/check_subset_default_scope_fields.py` and a warn-only
`tac.preflight.check_subset_default_scope_fields` wrapper. The scanner requires
verdict-emitting subset-default sites to carry all three scope signals: `n` or
population count, selection mode, and a subset/population or axis-bias caveat.

Live-source denominator after excluding archived `experiments/results` bundles:

| field | value |
|---|---:|
| Python files scanned | 6,214 |
| files matched with subset-default sites | 371 |
| subset-default sites | 595 |
| scope-reported sites | 11 |
| silent verdict-like sites | 568 |
| dormant/no-verdict sites | 16 |
| parse errors | 0 |

Kind breakdown:

| status | kind | count |
|---|---|---:|
| silent | CLI subset default | 282 |
| silent | prefix slice | 134 |
| silent | prefix range | 100 |
| silent | assigned subset default | 52 |
| scope-reported | CLI subset default | 7 |
| scope-reported | prefix range | 3 |
| scope-reported | assigned subset default | 1 |
| dormant/no-verdict | all kinds | 16 |

Machine-readable inventory:

| path | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_ss1_20260805/ss1_subset_default_scope_inventory_20260805.json` | 299,768 | `000b8c062161ffcfdbdd2f3e7ed4794a81e9da8ee365b410f94824bb3a5c8651` |
| `.omx/research/ddm_ss1_20260805/ss1_ranked_silent_subset_sites_20260805.json` | 4,618 | `13a7a0a8fc1f8943f221f8d82d136ab73c219509f3f944c2e6bdd1805000d81d` |

## Recall Evidence

Sources searched:

- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter files: `.omx/tmp/codex_runs/ss1_prompt.md`,
  `.omx/tmp/codex_runs/_common_contract.md`.
- Memory registry: `/Users/adpena/.codex/memories/MEMORY.md` for
  `prefix_bias`, `subset_default`, `ddm_na2`, `ddm_na3`, `selection_mode`.
- Prior receipts and state: `.omx/research/ddm_na2_negative_audit_20260803.md`,
  `.omx/research/ddm_na3_20260805/ddm_na3_receipt.md`,
  `.omx/research/ddm_ca1_20260805/CA1_RECEIPT.md`,
  `.omx/state/canonical_anti_patterns_registry.jsonl`,
  `.omx/state/canonical_task_status.jsonl`.
- Targeted live-consumer searches included `p3v2`, `ddm_pz1`,
  `diag_custom_vs_reference_trajectory_n8`, `diag_faithful_vs_muon`,
  `diag_recipe_fix_muon`, `run_taskspace_r10_feature_texture_relay`,
  `ddm_gr1`, `probe_frontier_adapter_dseg_flip`, `selection_mode`,
  `prefix bias`, and `population truth`.

Findings beyond the charter seeds:

- NA3 already registered the two anti-pattern classes
  `prefix_bias_sign_inversion_pose_axis_v1` and
  `subset_default_silent_under_sampling_v1`; SS1 does not duplicate that
  registry work.
- The existing `subset_selection_gate` is a real positive-control guard, but it
  only protects newly added staged prefix slices. It does not inspect emitted
  JSON/receipt scope fields.
- `p3v2` is an actual live-consumer example: its n6/n24 pose rows are cited by
  `ddm_gc17_from_here_gradient_not_coordinates_20260801.md` and AU1, while the
  source default is `--n-pairs=24` with no structured `selection_mode` field.
- `pz1` is partial coverage, not clean coverage. Some PZ1 tools state strided
  pairs and an m88 guard in prose/output, but still lack the structured
  `selection_mode` field the charter asks the scanner to enforce.
- Memory lookup reinforced the existing rule that even strided subset pricing
  must state `selection_mode`; it did not change the code plan beyond keeping
  the scanner field-based rather than prose-based.

## Ranked Silent Rows

The full 568-site class-B inventory is in the JSON sidecar. Manual ranking of
the live-consumer head:

| rank | site | why it matters | disposition |
|---:|---|---|---|
| 1 | `experiments/ddm_p3v2_optimal_form_pose_resolve.py --n-pairs=24` | pose-heavy n24 row cited by `gc17`/AU1; m96 says pose prefixes are false-negative shaped | QUEUED: no population pose claim until n>=120 non-prefix rerun with structured scope |
| 2 | `experiments/ddm_pz1_* --pairs=6/32/40` | live window-solve d_pose gate; partial m88 prose but missing structured `selection_mode` | QUEUED: add structured subset fields before next pz1/gk2 run |
| 3 | `tools/run_taskspace_r10_feature_texture_relay.py --pair-count=2` | current R10 mechanics runner; n2 output can be over-read without scope | QUEUED: scorer-free receipt migration before external consumption |
| 4 | n8 Muon/custom-backward diagnostics | both seg and pose targets on n8; n8 banks no population claim | FOLDED: instance diagnostics only unless rerun with n>=32 non-prefix scope |
| 5 | `experiments/ddm_gr1_granularity_rerace.py --pairs=48` | wrong-level/granularity comparative surface; subset choice can change comparison | QUEUED: rerun or emit subset scope before any formulation verdict |
| 6 | `experiments/probe_frontier_adapter_dseg_flip.py --n-pairs=8` | d_seg-only adapter probe; smaller seg bias but still not population truth | FOLDED: n8 instance evidence only |

No scorer-free rerun was fired in SS1. The high-ranked rows either require
PoseNet/SegNet forwards, are historical/completed consumers needing downgrade,
or need a small receipt-field migration in their owning lane. Firing them here
would violate the common contract's scorer-free and single-owner boundaries.

## Gate Extension

What changed:

- Added `tools/check_subset_default_scope_fields.py`.
- Added focused tests in `tools/tests/test_check_subset_default_scope_fields.py`.
- Added warn-only `tac.preflight.check_subset_default_scope_fields` and wired it
  into `preflight_all`.

Executed controls:

- Positive control: a `--pairs 32` verdict emitter with no subset fields flags.
- Negative control: a NA3-style scoped receipt with `n`, `selection_mode`, and an
  axis-bias caveat does not flag.
- Dormant/no-verdict control: a subset default with no verdict-like output is
  inventoried but not class-B.
- Full-population control: `--num-pairs 600` plus `range(args.num_pairs)` is not
  counted as a subset-default site.
- `experiments/results` exclusion control: archived result bundles do not enter
  the live denominator.

## Verification

Commands run:

| command | result |
|---|---|
| `.venv/bin/python -m pytest tools/tests/test_check_subset_default_scope_fields.py src/tac/tests/test_subset_selection_gate.py src/tac/tests/test_subset_selection.py` | 67 passed |
| `.venv/bin/python -m pytest tools/tests/test_check_subset_default_scope_fields.py` | 7 passed after scanner refinements |
| `.venv/bin/ruff check --isolated --select F821 tools/check_subset_default_scope_fields.py tools/tests/test_check_subset_default_scope_fields.py src/tac/preflight.py` | passed |
| `.venv/bin/python tools/check_subset_default_scope_fields.py --write-baseline .omx/research/ddm_ss1_20260805/ss1_subset_default_scope_inventory_20260805.json` | 6,214 scanned; 595 sites; 568 silent; 0 parse errors |

Review tracker: `ss1-pass1` and `ss1-pass2` were recorded for
`tools/check_subset_default_scope_fields.py`,
`tools/tests/test_check_subset_default_scope_fields.py`,
`tac.preflight::check_subset_default_scope_fields`, and the `preflight_all`
callsite.

The scanner emits one existing Python `SyntaxWarning` from
`src/tac/composition/alaska_inverse_steganalysis_patterns/__init__.py:18`; AST
parsing still completed with zero parse errors.

## Evidence And SHA Table

| path | bytes | sha256 |
|---|---:|---|
| `.omx/tmp/codex_runs/ss1_prompt.md` | 3,820 | `b68c7db2a09ee567954fb1420b98fab18111c0772800bb79034c7de47db21a46` |
| `.omx/tmp/codex_runs/_common_contract.md` | 4,124 | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |
| `tools/check_subset_default_scope_fields.py` | 18,609 | `02061b1977c762163b0589c212f4f87856d53a2a73089612fd1a821727faa1c6` |
| `tools/tests/test_check_subset_default_scope_fields.py` | 5,969 | `edb3b052633cf7cd5203f571d802519e3f5bda7a5da26dc6f3d410369fe0eecc` |
| `src/tac/preflight.py` | 3,970,056 | `bf291b414617b14c200c575b79a25606b8600f9d11a6c13a1a7467530419de65` |
| `src/tac/subset_selection_gate.py` | 18,133 | `6befbbb2e906cb2cd80ae93503288267e0ccdc873543dac795e354a463bf73f6` |
| `src/tac/subset_selection.py` | 33,410 | `074c0485691ce3b649d7585fb0d0e5f1fa11b2896baaef5b44189d8a2abbfb0a` |

## Boundaries

- No frozen scorer forward pass was run.
- No exact archive was built.
- No `upstream/evaluate.py` run was made.
- No contest-CPU/CUDA claim was made.
- No protected file was edited.
- The existing staged index was not used as evidence and was not rewritten.
- `src/tac/preflight.py` contains unrelated concurrent DN1/#904 edits in the
  working tree. SS1's intended preflight change is only the
  `check_subset_default_scope_fields` wrapper and warn-only call.
- The own-vehicle frontier is unchanged.

## NEXT_IF_RESUMED

Run this exact sequence:

1. Re-run `tools/check_subset_default_scope_fields.py --json` and compare the
   denominator to `ss1_subset_default_scope_inventory_20260805.json`.
2. Inspect any count delta before trusting this ranking.
3. Migrate the ranked rows in order: first add structured scope fields to any
   scorer-free receipt emitters; then queue scorer-dependent reruns behind the
   active scorer owner.
4. Do not promote any prefix/n8/n24/n48 subset verdict as population truth until
   it carries `n`, `selection_mode`, seed/stride/strata, and a governing
   subset/population ratio.

```json
{
  "next_if_resumed": [
    "rerun tools/check_subset_default_scope_fields.py --json",
    "compare files_scanned/files_matched/subset_default_sites/silent_verdict_subset_default",
    "migrate ranked scorer-free receipt fields first",
    "queue scorer-dependent reruns behind active scorer owner"
  ],
  "inventory": ".omx/research/ddm_ss1_20260805/ss1_subset_default_scope_inventory_20260805.json",
  "ranked_rows": ".omx/research/ddm_ss1_20260805/ss1_ranked_silent_subset_sites_20260805.json"
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
