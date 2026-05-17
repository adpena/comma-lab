# Master-Gradient Extractor WIP Axis And Rate Guard - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch, score claim, promotion, method kill, or architecture-lock
authority is created by this review.

## Why This Exists

The operator explicitly warned that relevant OMX / Claude signal may sit outside
`.omx/research`, and asked for bug hunting, adversarial review, and no retread.
The widened parent Markdown scan is already preserved in:

- `.omx/research/l5_v2_omx_parent_markdown_scope_refresh_20260517_codex.md`
- `.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`
- `.omx/research/omx_parent_markdown_cargo_cult_and_quantizr_staircase_review_20260517_codex.md`
- `.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`

This follow-up applies that same anti-cargo-cult discipline to the active
master-gradient WIP that appeared in the dirty tree after those scans.

## Reviewed WIP

Reviewed without staging partner files wholesale:

- `src/tac/master_gradient.py`
- `tools/extract_master_gradient.py`
- `src/tac/tests/test_extract_master_gradient.py`
- `scripts/operator_authorize_master_gradient_fec6_modal_cpu.sh`
- `.omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml`
- `tools/cathedral_autopilot_autonomous_loop.py` dirty diff
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
- `.omx/research/canonical_helper_pattern_audit_20260517.md`

The prior false-authority review already blocked raw archive-byte gradients:

- `.omx/research/master_gradient_partner_wip_false_authority_review_20260517_codex.md`

This review adds two additional blockers found in the newer extractor WIP.

## Findings

### F1 - Subset gradient is labeled as full contest axis

`tools/extract_master_gradient.py` defaults to `--n-pairs-used 8`, while the
recipe labels the output axis as `[contest-CPU]`. A subset gradient can be a
diagnostic prior, but it is not a full contest-axis anchor. If the sidecar is
written to `.omx/state/master_gradient_anchors.jsonl` under `[contest-CPU]`,
downstream autopilot consumers can mistake a local 8-pair derivative for a
600-pair contest-axis measurement.

Required unwind:

- Either use a diagnostic axis label for subset gradients, or
- run the full sample count required by the claimed axis, and
- persist `n_pairs_used` / `n_pairs_total` in the authority-bearing anchor.

### F2 - Rate column semantics conflate byte value with archive byte count

The WIP projects an `(N_archive_bytes, 3)` array and fills the rate column with
`1 / 37,545,489` for every byte. `MasterGradient.predict_delta_s` then accepts
`{byte_idx: delta}` and multiplies by the closed-form rate marginal.

That shape confuses two different things:

- changing the numeric value of an existing byte; and
- changing the charged archive byte count.

The contest rate term changes with archive length, not with the value stored in
an existing byte. Rate must be carried as a separate packet byte-count delta or
measured as a real score-response row after packet rebuild.

Required unwind:

- Use `byte_count_delta` for rate-only packet-size moves.
- Use measured `seg_dist_delta`, `pose_dist_delta`, and `rate_bytes_delta` for
  operator-response rows.
- Do not use a uniform per-archive-byte rate column inside a byte-value
  derivative API.

### F3 - Compressed-stream uniform projection is diagnostic, not a derivative

The extractor notes that Brotli decompressed bytes have no one-to-one mapping to
archive bytes, then uniformly spreads tensor sensitivity over compressed spans.
That may be useful as a section-level diagnostic, but it is not a per-byte
gradient suitable for direct reranking. The valid score-moving object remains a
grammar-aware operator candidate with parser, repack, inflate, and
byte-consumption proofs.

## Guard Landed

Extended `tac.master_gradient_feasibility` with:

- `audit_master_gradient_anchor_authority`
- `MasterGradientAnchorAuthority`

The new guard blocks authority when:

- coordinate system is raw archive byte, ZIP-member payload byte, or compressed
  stream uniform surrogate;
- a subset gradient is labeled as `contest_cpu`, `contest_cuda`, or
  `paired_contest_cpu_cuda`;
- a uniform per-archive-byte rate column is projected over byte-value deltas.

It allows only packet-valid coordinate systems as operator probes:

- `logical_section_parameter_response`
- `grammar_aware_operator_response`
- `codec_symbol_response`

Even allowed anchors remain:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_master_gradient_feasibility.py -q
.venv/bin/python -m ruff check src/tac/master_gradient_feasibility.py src/tac/tests/test_master_gradient_feasibility.py
```

Observed:

- `8 passed`
- `ruff`: all checks passed

## Dispatch Decision

The current `master_gradient_fec6_modal_cpu_dispatch` WIP must not be treated as
provider-ready while it emits a subset/raw-byte/compressed-span sidecar under a
full contest-axis label.

Valid next work is one of:

1. convert the extractor to a diagnostic-only sidecar with explicit subset
   axis and no ranking authority;
2. rewrite it as a packet-valid operator-response builder using
   `tac.master_gradient_operator_plan`; or
3. continue with the already-valid Brotli operator-candidate path and add
   runtime inflate plus byte-consumption proof.

The L5 / Rule #6 priority is unchanged: use master-gradient intuition only on
packet-valid operator rows, not as another raw-byte local-minima apparatus.
