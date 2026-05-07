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
- `reports/pr101_provable_optimal_floor.json`
- `reports/pr101_adaptive_ac.json`
- `reports/pr101_pmf_pca.json`
- `reports/pr101_constriction_marginal.json`
- `reports/pr101_constriction_shared_pmf.json`
- `reports/pr101_constriction_quantgauss.json`
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

Focused tests: 20 passed, 1 pytest-config warning about unknown `timeout`.
