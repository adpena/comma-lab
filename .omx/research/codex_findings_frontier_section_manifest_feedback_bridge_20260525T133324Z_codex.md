# Codex Findings - Frontier Section Manifest And Feedback Bridge - 2026-05-25T13:33:24Z

## Scope

Continuation of the queue-owned final-rate attack lane against the current
`[contest-CPU]` frontier and PR110/FEC6 control archive. This pass closed two
orphan-signal gaps:

1. `archive_section_entropy_recode_v1` no longer depends only on a manually
   supplied manifest. The frontier bootstrap derives parser-section manifests
   per archive and only schedules section recode when the manifest is tied to
   the same archive/member and contains Brotli-decompressible sections.
2. Frontier materializer sweep feedback can be discovered from sweep artifacts
   and compiled into a queue-owned DQS1 follow-up surface, instead of leaving
   saturated packet-member results as prose-only observations.

## Bugs / Bug Classes Closed

- Current CPU frontier archive parsing failed on the charged `DQS1` tail after
  the PR101/FEC6 selector. `tac.analysis.hnerv_packet_sections` now models the
  DQS1 tail as `selector_dqs1_selective_runtime_tail` after validating the DQS1
  payload. This preserves parser coverage for the actual frontier packet.
- Multi-archive section recode queues previously risked reusing one
  `section_manifest` across different archive SHAs. The frontier bootstrap now
  accepts per-archive section manifests and emits per-archive contexts.
- Section recode is now fail-closed when parser custody exists but no selected
  section is Brotli-decompressible. This keeps `archive_section_entropy_recode_v1`
  executable for compatible archives while correctly refusing current
  PR101/FEC6/DQS1 packets.
- Tensor factorization remains explicitly blocked without a tensor manifest and
  factorization contract/rank. No false generic tensor manifest derivation was
  introduced.
- Frontier packet-member sweep feedback can now flow into DQS1 follow-up queue
  generation via `frontier_rate_attack_feedback_refresh.v1`.

## Live Evidence

- Refreshed queue artifact:
  `.omx/research/frontier_final_rate_attack_20260525_section_manifest_bridge/`
- Current CPU frontier + PR110/FEC6 derived section manifest summary:
  `ready_manifest_count=0`; both archives have parser-proven sections but
  `section_manifest_has_no_brotli_decompressible_sections`.
- Executed queue:
  `frontier_final_rate_attack_20260525_section_manifest_bridge`
- Packet-member results:
  - `packet_member_zip_header_elide_v1`: `rate_positive_count=0`,
    `max_saved_bytes=0`
  - `packet_member_recompress_v1`: `rate_positive_count=0`, `max_saved_bytes=0`
- Feedback compiler artifact:
  `.omx/research/codex_frontier_rate_attack_feedback_compiler_20260525T133356Z/`
- Follow-up DQS1 queue:
  `codex_frontier_rate_attack_feedback_dqs1_followup_20260525`, 4 experiments,
  28 steps, valid, false-authority only.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_bootstrap.py tools/build_frontier_final_rate_attack_queue.py src/comma_lab/scheduler/__init__.py src/tac/analysis/hnerv_packet_sections.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_hnerv_packet_sections.py`
- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_packet_sections.py src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_rate_attack_feedback_refresh.py src/tac/tests/test_frontier_rate_attack_feedback.py src/comma_lab/scheduler/__init__.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_frontier_rate_attack_feedback_compiler_20260525T133356Z/dqs1_followup_queue.json validate`

## Next

1. Execute the feedback-generated DQS1 follow-up queue with bounded local CPU
   concurrency and immediately canonicalize the harvested observations back
   into acquisition/feedback surfaces.
2. Add a frontier bootstrap regression using a real PR106-like archive with
   Brotli sections and run an executable section-recode materializer smoke on
   that compatible fixture.
3. Continue PR95 MLX on the separate reproduction lane: full-size 1-step
   stage 1/5/8 timing smokes, public curriculum/QAT port, scorer-loss bridge,
   MLX-trained export parity, and runtime/full-frame parity.
