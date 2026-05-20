# Null-Seed Candidate Spec Lowering - 2026-05-20T23:11Z

## Verdict

Implemented the fail-closed lowering bridge from the fec6 null-codebook
replacement plan to typed runtime-adapter candidate specs.

The empirical null-byte plan remains high-signal, but the new spec builder
proves an important blocker: neither top candidate is a direct 8-byte seed
reconstruction of the original parser-visible payload. Both require a runtime
adapter plus full-frame/contest exact eval before any archive shrink or score
language is valid.

## Landed artifacts

- `src/tac/procedural_codebook_generator/null_seed_candidate_spec.py`
- `tools/build_null_seed_candidate_spec.py`
- `src/tac/tests/test_null_seed_candidate_spec.py`
- `.omx/research/null_seed_candidate_spec_fec6_rank1_20260520T230946Z_codex.json`
- `.omx/research/null_seed_candidate_spec_fec6_rank1_20260520T230946Z_codex.md`
- `.omx/research/null_seed_candidate_spec_fec6_rank2_selector_20260520T230946Z_codex.json`
- `.omx/research/null_seed_candidate_spec_fec6_rank2_selector_20260520T230946Z_codex.md`

## Empirical result on fec6 null candidates

| rank | span | original bytes | net saved upper bound | direct seed reconstruction | verdict |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `source_payload+selector_len_hdr+selector_payload` `[162171, 178417]` | 16246 | 16238 | false | blocked until runtime adapter + exact eval |
| 2 | `selector_payload` `[178168, 178417]` | 249 | 241 | false | blocked until runtime adapter + exact eval |

Rank 1 carries `source_payload_seed_substitution_parse_risk` because it spans
the PR101 source payload tail. Rank 2 is the cleaner next adapter target: it is
selector-only, has lower upside, and still needs a seeded selector runtime mode
plus seed-mutation frame-delta proof.

## What this extinguishes

This closes the false-authority path "null-gradient bytes are free bytes."
The builder now forces:

- archive ZIP custody and member SHA verification;
- original span SHA verification against the plan;
- deterministic seed derivation or operator-supplied seed validation;
- direct reconstruction check via `derive_codebook_from_seed`;
- blocked `CandidateModificationSpec`-shaped payload until runtime adapter,
  frame-delta/no-op proof, and contest CPU/CUDA exact eval exist.

## Tests

`36 passed`:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_null_seed_candidate_spec.py \
  src/tac/tests/test_null_seed_replacement_plan.py \
  src/tac/tests/test_procedural_codebook_generator.py \
  src/tac/tests/test_procedural_codebook_candidate_authority.py -q
```

## Next action

Build the selector-only seeded adapter first. It is the safer concrete
runtime surface because it avoids replacing a PR101 source-payload tail. The
adapter should parse a new charged selector-seed payload, derive a 600-code
selector stream, run the existing FEC6 runtime-consumption/no-op proof pattern,
then generate a candidate archive for exact eval.
