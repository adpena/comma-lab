# C6 IBPS1 Parser Alias And Structural MDL Ablation - Codex 2026-05-14

## Scope

Operator Tier 0 items T0.5/T0.6 required the C6 `IBPS1` archive grammar to be
canonical instead of falling back to `whole_blob`. The parser subagent landed the
canonical parser under `tac.substrates.c6_e4_mdl_ibps.archive`; this pass fixed
the remaining operator-facing alias gap so `parser=ibps1` resolves to the
canonical parser name `ibps1_mdl_ibps`.

This is not a score claim and not promotion evidence. It is a parser-conditioned
MDL headroom measurement for the existing 5-epoch C6 smoke archive.

## Code Change

- `src/tac/analysis/hnerv_packet_sections.py`
  - Added alias normalization for `ibps1`, `c6_e4_mdl_ibps`, and `mdl_ibps`.
  - Kept emitted manifest parser name canonical as `ibps1_mdl_ibps`.
- `src/tac/tests/test_scorer_conditional_mdl_cli.py`
  - Changed the C6 CLI coverage to call `parser=ibps1`, matching the Tier 0
    operator spelling instead of relying on auto-detection.

## Verification

Command:

```bash
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_conditional_mdl_cli.py::test_compute_cli_parses_ibps1_sections_without_whole_blob \
  src/tac/tests/test_hnerv_packet_sections.py::test_ibps1_manifest_records_canonical_c6_sections_and_auto_infers \
  src/tac/tests/test_hnerv_packet_sections.py::test_ibps1_manifest_rejects_trailing_schema_drift \
  -q
```

Result: `3 passed in 0.65s`.

Structural MDL command:

```bash
PYTHONPATH=src:upstream .venv/bin/python tools/compute_scorer_conditional_mdl_ablation.py \
  --archive c6_5ep=experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal/harvested_artifacts/lane_substrate_c6_e4_mdl_ibps_results/output/archive.zip,parser=ibps1 \
  --output-dir experiments/results/mdl_ablation_c6_structural_20260514_codex_retry2
```

Artifacts:

- `experiments/results/mdl_ablation_c6_structural_20260514_codex_retry3/scorer_conditional_mdl_ablation.json`
- `experiments/results/mdl_ablation_c6_structural_20260514_codex_retry3/scorer_conditional_mdl_ablation.md`

## Result

Archive:

- label: `c6_5ep`
- archive zip bytes: `224481`
- parser: `ibps1_mdl_ibps`
- parsed sections: `ibps1_header`, `encoder_blob`, `decoder_blob`,
  `latent_blob`, `meta_blob`

MDL layers:

| layer | groups | floor bytes ceil | gap bytes ceil | claim strength |
|---|---:|---:|---:|---|
| `unconditional_payload` | 1 | 225421 | 201 | parser-proven empirical entropy floor |
| `parser_section_conditioned` | 5 | 223464 | 2158 | parser-proven empirical entropy floor |
| `parser_role_conditioned` | 3 | 223517 | 2105 | parser-proven empirical entropy floor |
| `scorer_feature_proxy_conditioned` | 3 | 223517 | 2105 | proxy only, not true scorer conditional entropy |

Interpretation:

The C6 5-epoch archive has only about `2.1 KB` parser-conditioned entropy
headroom under this structural byte model. That is real grammar-level headroom,
but it is not enough by itself to justify a C6 full dispatch. The key blocker is
still scorer evidence: the current structural tool has no penultimate-feature or
component-response byte map, so `scorer_feature_proxy_conditioned` remains a
proxy and cannot prove cooperative-receiver savings.

Next action:

- Let the running Tier B/C CPU ablations finish for a higher-cost byte-level
  signal.
- Do not promote C6 from this artifact alone.
- Prioritize scorer-feature binding or component-response byte maps before
  using C6 MDL results as a dispatch trigger.
