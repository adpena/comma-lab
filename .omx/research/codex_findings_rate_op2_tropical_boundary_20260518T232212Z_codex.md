---
schema: codex_findings_v1
memo_id: codex_findings_rate_op2_tropical_boundary_20260518T232212Z_codex
timestamp_utc: "2026-05-18T23:22:12Z"
agent: codex
task_id: rate_attack_op_2_tropical_argmax_boundary_grammar
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
prediction_only: true
---

# Codex Findings: RATE-OP-2 Tropical Argmax Boundary Grammar

## Verdict

Implemented OP2 as a fail-closed feasibility surface, not as a mutating archive
builder. The first real artifact over the current fec6 CPU-frontier archive
found strong SegNet boundary/stable-interior analysis inputs, but **zero
contest-charged argmax/logit/label-like payload** in the archive layout.

That means OP2 is not a bolt-on byte-savings claim for the current fec6 packet.
It routes to export-first substrate engineering or a future packet grammar that
natively contains charged argmax-cell payloads.

## Concrete Artifacts

- Code: `src/tac/contest_exploits/tropical_argmax_boundary_grammar.py`
- CLI: `tools/build_rate_attack_op2_tropical_argmax_boundary_grammar.py`
- Tests: `src/tac/contest_exploits/tests/test_tropical_argmax_boundary_grammar.py`
- Local planning artifact:
  `experiments/results/rate_attack_op2_tropical_argmax_boundary_20260518T232125Z/feasibility_report.json`
- Cathedral rows:
  `experiments/results/rate_attack_op2_tropical_argmax_boundary_20260518T232125Z/cathedral_autopilot_candidates.jsonl`

Artifact SHA-256:

- `feasibility_report.json`: `b3ac879e6328317c97e3e31b8d0fc3d315f7bc865e4564572e7b0d1d452ec14d`
- `cathedral_autopilot_candidates.jsonl`: `da7fde9264b6d455e381816356dc15ef81be54b2d39881fcc407f3d8973be5f7`
- `boundary_tiers.csv`: `5d61a9577e8b45c3850a3e54389268c8cc301f7e22a7ddbd35b3c11cbde3b40b`
- `tropical_cells.jsonl`: `39152b64aa002b89f970657833e905fda8b1c0376f5e8b7d642bd358979c6b60`

## Authority Findings

The OP2 artifact preserves these hard distinctions:

- Tropical/logit-boundary geometry is a candidate detector, not exact d_seg
  equivalence.
- Existing SegNet boundary and SABOR stable-interior artifacts are analysis
  sidecars, not contest-byte savings.
- The fec6 archive has no detected contest-charged argmax-like section.
- Boundary overhead is charged in the proxy model; for fec6, replacement
  savings are zero and overhead is nonzero.
- All Cathedral rows load as planning-only rows with `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

Key fail-closed blockers encoded in every candidate row include:

- `tropical_op2_contest_charged_argmax_payload_missing`
- `tropical_op2_exact_root_equivalence_unproven`
- `tropical_op2_existing_boundary_baseline_comparison_missing`
- `tropical_op2_smooth_surrogate_baseline_missing`
- `tropical_op2_composition_alpha_missing`
- `tropical_op2_packet_proofs_missing`
- `tropical_op2_no_score_claim_without_exact_eval`

## Verification

Commands passed:

```bash
.venv/bin/python -m pytest \
  src/tac/contest_exploits/tests/test_tropical_argmax_boundary_grammar.py \
  src/tac/contest_exploits/tests/test_stable_orbit_packet_diet.py

.venv/bin/ruff check \
  src/tac/contest_exploits/tropical_argmax_boundary_grammar.py \
  src/tac/contest_exploits/tests/test_tropical_argmax_boundary_grammar.py \
  tools/build_rate_attack_op2_tropical_argmax_boundary_grammar.py \
  src/tac/contest_exploits/__init__.py
```

Result: 5 pytest tests passed; Ruff passed.

## Next Routing

OP2 should remain prediction-only until a future archive grammar binds tropical
cells to charged bytes. The next high-EV branch is either RATE-OP-3 decoy/mosaic
route entropy or an OP2 export-first substrate sketch that explicitly owns a
monolithic `0.bin` grammar, charged boundary overhead, packet mutation proof,
full-frame parity, and exact eval.
