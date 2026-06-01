# Archive-runnable section payload grammar optimizer landed

## Context

The generic section payload grammar optimizer was extended from pre-extracted
section files to direct single-member ZIP archive intake. This removes the
manual extraction step from packet/compiler rate diagnostics and gives future
substrates a reusable archive-runnable rate gate before replay or exact auth.

## Landed surfaces

- `src/tac/packet_compiler/section_payload_grammar_optimizer.py`
  - `sections_from_single_member_zip_archive(...)`
  - `SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA`
  - source-manifest propagation into `solve_section_payload_grammar(...)`
- `tools/section_payload_grammar_optimizer.py`
  - `--zip-archive`
  - `--zip-member`
  - `--zip-section NAME:START:LENGTH`
- `src/tac/tests/test_section_payload_grammar_optimizer.py`
  - archive provenance extraction test
  - CLI archive-span test
  - candidate-queue signal-surface consumption guard

## Real artifact

Command class:

```bash
uv run python tools/section_payload_grammar_optimizer.py \
  --zip-archive experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/archive.zip \
  --zip-member x \
  --zip-section decoder_blob:0:170252 \
  --zip-section latent_blob:170252:7906 \
  --zip-section sidecar_blob:178158:0 \
  --output /Volumes/VertigoDataTier/pact/section_payload_grammar_pr101_zip_archive_20260601T192143Z/section_payload_report.json \
  --queue-output /Volumes/VertigoDataTier/pact/section_payload_grammar_pr101_zip_archive_20260601T192143Z/section_payload_queue.json \
  --campaign-id pr101_source_zip_section_payload_grammar \
  --brotli-quality 11
```

Artifact root:

`/Volumes/VertigoDataTier/pact/section_payload_grammar_pr101_zip_archive_20260601T192143Z`

Source:

- archive path: `experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/archive.zip`
- archive ZIP bytes: `178258`
- ZIP member: `x`
- member payload bytes: `178158`
- ZIP overhead bytes: `100`
- section spans:
  - `decoder_blob:0:170252`
  - `latent_blob:170252:7906`
  - `sidecar_blob:178158:0`

## Verdict

PR101/fec6-like section payload grammar remains saturated on the current
competitive archive:

- selected isolated section bytes: `178168`
- baseline isolated section bytes: `178168`
- selected savings vs Brotli baseline: `0`
- selected coders:
  - `decoder_blob`: `brotli`
  - `latent_blob`: `brotli`
  - `sidecar_blob`: `brotli`
- selected over Shannon floor ratio: `1.0003585227776992`
- saturation status: `entropy_saturated`
- next action emitted by diagnostic:
  `stop_format_churn_on_this_tensor_family_without_new_substrate_signal`

This is not a promotion artifact. It is planning-only byte-profile evidence:
the report remains fail-closed with blockers for receiver binding, byte-closed
materialization, runtime proof, full-frame inflate parity, and exact CPU/CUDA
eval.

## Implication for score lowering

The current PR101/fec6-style grammar is close enough to the section entropy
floor that further container/section churn is not the high-EV path to breaking
`0.19`. Long training and new substrate work should be judged by whether they
produce better decoder weights, latent grammar, or scorer-aware allocation
before archive packaging. The archive-runnable optimizer is now the reusable
rate gate that future HPRC, Z8, HNeRV, NeRV-family, and non-NeRV substrate
outputs must pass before local replay or exact auth promotion.

## Verification

- `uv run ruff check src/tac/packet_compiler/section_payload_grammar_optimizer.py tools/section_payload_grammar_optimizer.py src/tac/tests/test_section_payload_grammar_optimizer.py`
- `uv run pytest src/tac/tests/test_section_payload_grammar_optimizer.py -q`

