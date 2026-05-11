# Public PR95+ mechanism index integration (2026-05-11)

## Status

This is a routing and evidence ledger, not a score claim.

- `score_claim=false`
- `promotion_eligible=false`
- `dispatch_attempted=false`
- generated index command:

```bash
.venv/bin/python tools/build_public_pr_mechanism_index.py \
  --min-pr 95 \
  --json-out .omx/research/artifacts/public_pr95_plus_mechanism_index_20260511_codex/index.json \
  --md-out .omx/research/artifacts/public_pr95_plus_mechanism_index_20260511_codex/index.md
```

Generated artifacts are local custody under ignored `.omx/research/artifacts/`.
The reusable parser and CLI are committed code:

- `src/tac/analysis/public_pr_mechanism_index.py`
- `tools/build_public_pr_mechanism_index.py`
- `src/tac/tests/test_public_pr_mechanism_index.py`

## What changed in our understanding

The local public-PR text corpus now has a reusable evidence-only indexer. It
indexed 99 unique PR95+ report/writeup/readme files across the public mirror
roots and records device-labeled eval rows plus mechanism tags. The index is
not a promotion surface; it exists to prevent selective-memory errors while
routing exact score-lowering work.

The high-confidence pattern is not "HNeRV alone." The winning stack is:

`score-aware RGB renderer -> compact latent stream -> correction sidecar -> byte-level packet grammar -> exact device-axis validation`.

HNeRV is the current best public substrate because PR95 supplied a strong
renderer/latent base, then PR98/100/101/102/103/105/106 added small correction
and byte-grammar changes. Non-HNeRV methods are still live, but they need to
enter as consumed residual/sidechannel/compiler passes or fully score-aware
export-first renderers rather than isolated research prototypes.

## Corpus anchors

- PR95 `hnerv_muon` states the 178 KB HNeRV archive uses a 229K-parameter
decoder, 28-d per-frame-pair latents, and an 8-stage curriculum ending in QAT,
C1a, and Muon. Source:
`experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/README.md:1-17`.
- PR101 `hnerv_ft_microcodec` adds schema-driven decoder packing, centered
latent packing, ranked Huffman/no-op sidecar grammar, and reports local CPU
`0.19284`. Source:
`experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/README.md:1-30`.
- PR103 `hnerv_lc_ac` explicitly identifies arithmetic/range coding on dense
weight tensors and latent-hi bytes, hardcoded section lengths, adaptive Brotli,
single-byte ZIP filename, and stream merging as the byte-level win. Source:
`experiments/results/public_pr_archive_release_view/public_pr103_intake_20260505_auto/pr_body.md:24-29`.
- PR106 `belt_and_suspenders` states the per-pair single-dim latent perturbation
was chosen against DALI cu128 ground truth and is GPU-only. Source:
`experiments/results/public_pr_archive_release_view/public_pr106_intake_20260505_auto/source/submissions/belt_and_suspenders/README.md:1-9`.
- The ANR/non-HNeRV training README shows a different full-stack approach:
TokenRendererV62, ShrinkSingleNeRV, HPACMini arithmetic coding, DALI fine-tune,
CPU-FP32 FiLM portability, and explicit DALI version sensitivity. Source:
`experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/jas0xf_adversarial_neural_representation/training/README.md:1-79`.

## Device-axis correction

The PR95+ indexer records public-comment/report device labels when present and
classifies the old "does this require GPU" prompt as `cpu_capable` or
`gpu_required`; it does not treat those labels as exact CUDA evidence.

Fresh-eyes review also found the CPU-better trend is real for several public
CPU/comment rows but not universal. PR106-derived local packets are currently
CUDA-favored; PR106 latent sidecar exact T4 beats its Linux CPU diagnostic, and
PR103-on-PR106 is also CUDA-better in the local paired matrix. Therefore:

- never infer CPU or CUDA superiority globally;
- always record scorer device, inflate device, raw-output aggregate SHA, archive
SHA, runtime tree, and loader path;
- P100/Kaggle, MPS, macOS CPU, and Linux CPU remain separate axes from Modal T4
contest CUDA.

## Score-lowering routing decision

Immediate P0/P1 stays PR106 latent score-table sidecars because it has already
moved exact T4 CUDA from `0.20739428085403283` to
`0.20664588545741508` on the radius-2 archive. The next work is not another
unprincipled bias constant; it is:

1. custody/compliance hardening for the radius-2 archive;
2. paired CPU/CUDA and inflate/scorer-axis matrix;
3. PR101/PR103-style entropy grammar for the sidecar bytes;
4. residual-basis score tables over PR106 decoded outputs.

## Non-HNeRV tracks to keep live

- ANR/HPAC/token renderer: mine HPAC/context-model and CPU-FP32 FiLM portability
lessons for sidecar compression and deterministic runtime design.
- PR85/86/91 HPAC/mask-action/HPM1: reuse byte grammars and context models for
sidecar/table compression; do not replace PR106 base without exact replay.
- Wavelets/foveation/RAFT/ego-motion: generate basis coefficients against PR106
residuals and route through the same score-table -> materialize -> exact-T4 loop.
- Cool-Chic/C3/VQ/coordinate/SIREN: require export-first archive grammar and
runtime consumption before dispatch; treat as Phase 2 residual or renderer lanes.
- Ballé/CompressAI/hyperprior: valuable as a trained end-to-end substrate only
after the runtime/export loop is contest-compliant; current smoke failures are
implementation/training evidence, not broad family death.

## New reusable surface

`tools/build_public_pr_mechanism_index.py` gives the operator and future agents a
fast way to refresh the corpus-level mechanism map. It is intentionally separate
from experiment logic and outputs evidence-only artifacts. Future score-routing
work should call this instead of repeatedly hand-grepping public PR writeups.

## Next actions

1. Add strict custody/compliance packet around
`experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json`.
2. Launch the PR106 R2 four-cell device-axis matrix if no active claim conflicts.
3. Build a sidecar entropy-grammar reducer using PR101 ranked no-op/Huffman and
PR103 measured-positive AC as reusable `tac` primitives.
4. Harvest the currently running PR106 yshift Kaggle job; if it emits a
byte-closed positive, materialize locally and exact-T4 adjudicate. If not,
classify by axis and close the claim.
5. Convert wavelet/foveation/RAFT residual bases into score-table candidates on
PR106 decoded outputs so non-HNeRV signal enters the exact-eval path instead of
remaining research prose.
