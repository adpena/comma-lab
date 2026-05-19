# PR106 Format0d PacketIR Master-Gradient Boundary - Codex Findings 2026-05-19T052006Z

## Verdict

`pr106_format0d` is now parsed through the canonical PR106 PacketIR consumed-byte
proof instead of a hand-coded fixed-offset approximation.

This is a boundary hardening step, not anchor authority. Master-gradient anchor
emission remains fail-closed for `pr106_format0d` until a real
`pr106_format0d_packet_ir_two_pass_jacobian_projector` exists for the primary
payload plus base/extra ranked-no-op sidecar streams.

## What Changed

- `tools/extract_master_gradient.py` parses format0d by calling
  `parse_pr106_sidecar_packet()` and `pr106_sidecar_consumed_byte_proof()`.
- The parser now verifies byte-identical PacketIR re-emission before accepting
  the layout.
- Archive sections now carry `score_affecting` so consumers can distinguish
  wrapper bytes from score-affecting payload bytes.
- The fail-closed projector name was sharpened from
  `pr106_format0d_primary_payload_projector` to
  `pr106_format0d_packet_ir_two_pass_jacobian_projector`.
- `tac.__version__` now matches `pyproject.toml` (`0.2.0rc1`), closing the
  package identity red test left by the TAC naming pass.

## Live Fixture

Canonical live format0d archive:
`experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`

- archive SHA-256:
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- gradient subject SHA-256:
  `15a5dccba352838df7a8dd190a8782e51afa53b475bcf07921f05ce88c10785e`
- gradient subject bytes: `186776`
- score-affecting sections:
  `pr106_payload`, `base_format0c_sidecar_payload`,
  `extra_pr101_ranked_no_op_payload`, `extra_framing_meta`
- non-score-affecting wrapper sections:
  `magic`, `format_id`, `pr106_len_le_u32`, `extra_payload_len_le_u16`

## Authority Boundary

The parser proves layout and section identity only. It does not prove a usable
score-response gradient for those sections.

Downstream planner/cathedral consumers must continue to treat this grammar as
detection-only:

- `anchor_emission_allowed=false`
- `score_claim_allowed=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The next implementation step is a grammar-aware operator-response projector,
not raw byte-gradient authority.

## Verification

- `PYTHONPATH=src:tools .venv/bin/python -m pytest src/tac/tests/test_extract_master_gradient.py::test_pr106_format0d_parser_fails_closed_on_malformed_packetir src/tac/tests/test_extract_master_gradient.py::test_pr106_format0d_live_archive_packetir_sections_are_pinned src/tac/tests/test_extract_master_gradient.py::test_main_fail_closes_detection_only_grammar_before_codec_import -q`
- `PYTHONPATH=src:tools .venv/bin/python -m pytest src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_master_gradient_archive_parsers.py src/tac/tests/test_package_api_hygiene.py src/tac/tests/test_tac_terminology_guard.py -q`
- `.venv/bin/python -m ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py src/tac/__init__.py`
- `.venv/bin/python tools/audit_master_gradient_wire_in_coverage.py --summary`

## Residual Work

`codex_routing_directive_op_syn_1_master_gradient_six_archive_extension_20260518::OP_SYN_1`
should remain open. This landing improves the PR106 format0d parser and
projection contract, but it does not implement the actual two-pass Jacobian
projector or write format0d master-gradient anchors.
