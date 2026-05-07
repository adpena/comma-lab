# PR101 entropy-floor ladder and AAC triage - 2026-05-07

## Scope

This ledger hardens the PR101 PMF/AAC tranche into reproducible planning
evidence. It answers the operator prompt to "search for and prove the truly
optimal" by separating:

- model-class entropy floors that are mathematically meaningful,
- packet estimates that include explicit side-information overhead,
- actual score claims, which remain CUDA exact-eval only.

No artifact in this ledger is a contest score claim. No archive was changed.

## Artifacts

- `tools/pr101_provable_optimal_floor.py`
- `tools/pr101_adaptive_arithmetic_coding.py`
- `tools/pr101_pmf_pca_compression.py`
- `tools/pr101_constriction_marginal_floor.py`
- `tools/pr101_constriction_shared_pmf_floor.py`
- `tools/pr101_constriction_quantgauss_hyperprior.py`
- `tools/pr101_context_transform_floor_probe.py`
- `reports/pr101_provable_optimal_floor.json`
- `reports/pr101_adaptive_ac.json`
- `reports/pr101_pmf_pca.json`
- `reports/pr101_constriction_marginal.json`
- `reports/pr101_constriction_shared_pmf.json`
- `reports/pr101_constriction_quantgauss.json`
- `reports/pr101_context_transform_floor_probe.json`
- `reports/pr101_markov1_aac_round_trip.json` (pre-existing context-coder
  round-trip report, used here as negative implementation evidence)

Input state dict for regenerated reports:

`experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`

Each regenerated report now records `input_state_dict_sha256`, evidence grade,
score-claim false, dispatch readiness false, and explicit blockers.

## Entropy-floor ladder

`reports/pr101_provable_optimal_floor.json` computes entropy floors for declared
source models:

| Model class | Payload bytes | Archive bytes | Interpretation |
|---|---:|---:|---|
| Markov-2 per tensor | 98,013 | 114,107 | Oracle context floor; context/model overhead omitted |
| Markov-1 per tensor | 152,106 | 168,200 | Oracle one-symbol context floor; model overhead omitted |
| IID per tensor | 159,822 | 175,916 | Static per-tensor PMF floor; PMF side information not included |
| Conditional on tensor | 159,822 | 175,916 | Same as IID per tensor; tensor id is side information |
| IID pooled | 186,584 | 202,678 | One shared marginal PMF, no tensor identity |

The unrestricted global optimum is not proven by this ladder. What is proven is
conditional: for a declared source model, entropy is the lower bound for that
model class, and arithmetic/range coding can approach it. Higher-context rows
are research targets because they omit the cost of transmitting or learning the
context model.

## AAC triage

`reports/pr101_adaptive_ac.json` computes a theoretical running-count adaptive
AC bound:

- Per-tensor AAC payload: 161,975 bytes.
- Per-tensor AAC archive estimate: 178,181 bytes.
- Brotli+Optuna reference: 178,144 bytes.
- Delta: AAC is 37 bytes worse at archive level.
- Blocker: no actual adaptive coder bitstream and no archive substitution.

The existing Markov-1 AAC round-trip report is an important negative control:

- Naive Markov-1 AAC payload: 183,144 bytes.
- Naive Markov-1 AAC archive estimate: 199,238 bytes.
- Round-trip byte faithful: true.
- Delta: 21,094 payload bytes worse than Brotli payload and 21,094 archive
  bytes worse after equal archive overhead.

This does not falsify Markov/context coding as a family. It falsifies the naive
prefix-adaptive implementation as a route to the oracle Markov-1 floor. The
oracle row uses full empirical transition counts with no model cost; the
deployed adaptive coder pays large cold-start/smoothing cost and does not get
the free transition table.

Important nuance: AAC is 75 bytes better than the reported Brotli payload
reference (161,975 vs 162,050), but the boundary/archive accounting leaves it
37 bytes worse at total archive level. Therefore AAC is not dispatchable yet.
It becomes worth implementing only if a real packet format can remove at least
38 bytes of overhead or if a context transform pushes it materially below the
IID per-tensor basin.

## PMF-side-information probes

Measured packet estimates:

| Probe | Archive bytes | Delta vs 178,144 | Disposition |
|---|---:|---:|---|
| Per-tensor empirical PMF + range coder | 190,718 | +12,574 | PMF overhead dominates |
| Shared PMF + range coder | 203,196 | +25,052 | Cross-tensor distribution mismatch dominates |
| Gaussian/Laplace parametric hyperprior | 205,938 | +27,794 | Parametric shape mismatch |
| PCA PMF side information, best K=15 | 184,837 | +6,693 | Low-rank PMF hypothesis falsified on PR101 |
| Per-tensor AAC theoretical estimate | 178,181 | +37 | Nearly tied, but no real bitstream |
| Naive Markov-1 AAC round trip | 199,238 | +21,094 | Byte-faithful but far from oracle floor |

PCA found Brotli-compressed concatenated PMFs at 3,513 bytes, but the PMF
matrix spectrum is shallow: K=15 is the best measured rank and still loses by
6,693 bytes. The PMFs are not a clean low-rank family on this substrate.

## Categorical and byte slicing

The "AAC subsumes categorical and byte slicing" statement needs a stricter
formulation:

Adaptive coding can subsume those hacks only if the adaptive context sees the
predictive signal they expose. Categorical labels, byte slicing, tensor order,
zigzag maps, and learned transforms still matter because they can lower the
conditional entropy seen by the adaptive coder. They are representation
transforms, not obsolete tricks.

Operational consequence: do not spend much more wall-clock on static PMF
compression. The next useful coder target is a deterministic transform plus
context coder:

`typed symbols -> byte/label transform -> Markov/context/range/ANS coder -> packet`

The floor ladder says why: the gap from IID per-tensor to Markov-1 is 7,716
payload bytes, and the oracle Markov-2 row is far lower. The hard work is
learning a deployable context model whose own description length does not eat
the gain.

`reports/pr101_context_transform_floor_probe.json` adds the first transform
probe:

| Transform | IID payload | Markov-1 payload | Markov-2 payload | Markov-1 archive vs 178,144 |
|---|---:|---:|---:|---:|
| identity | 159,822 | 152,106 | 98,013 | -9,944 |
| signed_zigzag | 159,822 | 152,106 | 98,013 | -9,944 |
| zero_mask_nonzero_value | 159,822 | 152,220 | 99,612 | -9,830 |
| abs_sign_split | 173,797 | 168,859 | 134,062 | +6,809 |
| delta_mod255 | 199,284 | 170,501 | 54,935 | +8,451 |
| nibble_split | 183,072 | 182,071 | 176,699 | +20,021 |
| bitplanes | 225,443 | 225,327 | 225,257 | +63,277 |

Interpretation: signed zigzag and identity are entropy-equivalent under these
oracle models; zero-mask categorical splitting does not improve Markov-1; byte
slicing/nibbles/bitplanes lose under this model. Delta coding creates a very
low Markov-2 oracle row but a worse Markov-1 row, which makes it a research
target for deployable higher-order modeling rather than a ready packer win.

## Next engineering tranche

1. Do not promote the existing naive Markov-1 AAC codec. Keep its round-trip
   report as negative evidence and use it to design the next context coder.
2. Implement a tiny deterministic per-tensor range/ANS/AAC bitstream with
   golden vectors only if the packet can beat 178,144 after headers. Treat it
   as a byte-level proof harness, not a score lane.
3. Build a context-transform probe: byte slicing, zigzag variant, tensor order,
   and categorical labels feeding the Markov-1/Markov-2 floor ladder. Rank
   transforms by reduced conditional entropy after charging transform metadata.
4. Repeat this exact ladder on the PR106/frontier substrate before generalizing
   conclusions.
5. Feed the floor rows into meta-Lagrangian/Pareto as planning atoms with
   `ready_for_exact_eval_dispatch=false` until an actual packet compiler exists.
6. Keep architecture-side work ahead of PMF-side polish: the IID floor is
   already close to Brotli, while structural training and context modeling have
   materially larger headroom.

## Verification

- `uv run ruff check tools/pr101_adaptive_arithmetic_coding.py tools/pr101_pmf_pca_compression.py tools/pr101_constriction_marginal_floor.py tools/pr101_constriction_quantgauss_hyperprior.py tools/pr101_constriction_shared_pmf_floor.py tools/pr101_provable_optimal_floor.py src/tac/tests/test_pr101_entropy_floor_tools.py`
- `uv run --with pytest python -m pytest src/tac/tests/test_codec_op_optuna_search.py src/tac/tests/test_codec_op_cma_search.py src/tac/tests/test_pr101_entropy_floor_tools.py -q`
- `.venv/bin/python -m json.tool reports/pr101_adaptive_ac.json`
- `.venv/bin/python -m json.tool reports/pr101_pmf_pca.json`
- `.venv/bin/python -m json.tool reports/pr101_provable_optimal_floor.json`
- `.venv/bin/python -m json.tool reports/pr101_constriction_marginal.json`
- `.venv/bin/python -m json.tool reports/pr101_constriction_quantgauss.json`
- `.venv/bin/python -m json.tool reports/pr101_constriction_shared_pmf.json`
- `.venv/bin/python -m json.tool reports/pr101_context_transform_floor_probe.json`

Focused tests: 22 passed, 1 pytest-config warning about unknown `timeout`.

## Adversarial review of 19:10 summary claims

Accepted with narrow scope:

- Brotli+Optuna at 178,144 and per-tensor AAC at 178,181 are verified in the
  local reports. AAC is a near tie but not a candidate archive.
- PCA K=15 at 184,837 verifies that this particular linear low-rank PMF basis
  loses on PR101.
- Shared PMF at 203,196 and QuantizedGaussian/Laplace at 205,938 verify that
  those simple non-learned PMF models lose on PR101.
- The naive Markov-1 AAC round-trip at 199,238 is valid negative evidence for
  that implementation.

Rejected or narrowed:

- "Encoder-side ceiling is structurally ~178 KB without ML" is too broad. The
  supported statement is narrower: the tested IID/static/simple-parametric
  PMF and naive adaptive coders are saturated near 178 KB on PR101. It does not
  prove impossibility for all deterministic transforms, table structures,
  grammar coders, or custom low-level packetization.
- "Markov-1 falsified" is too broad. Only the naive prefix-adaptive Markov-1
  implementation is falsified. The oracle Markov-1 row still shows context
  signal; the problem is making that signal deployable after model cost.
- "Oracle Markov-1 + brotli'd table = 209,051" is not accepted without a
  committed artifact showing the 40,851-byte table construction and exact
  accounting. Even if confirmed, it falsifies a full transmitted transition
  table, not sparse, structured, generated, or learned context models.
- "Only NN path" is too strong. Neural hyperpriors are the best current
  hypothesis for making context cheap, but deterministic grammar/codegen/table
  factorization and architecture-side retraining remain live until exact
  artifacts retire them.
- CompressAI/Balle and tiny-NN savings are prediction-band only. They must
  record model bytes, deterministic decode, payload bytes, and archive closure
  before they can affect dispatch priority.

Updated priority: pursue architecture/lower-entropy substrate and learned
context modeling in parallel, while allowing cheap deterministic transform
probes only when they produce new conditional-entropy evidence or a byte-closed
packet.
