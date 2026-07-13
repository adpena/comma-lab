# SFESS OSS reconciliation for frozen-SegNet throughput probe

**OUTCOME:** `NO-GO` for SFESS as a frozen-SegNet throughput replacement remains unchanged.
The ICLR-paper-enriched learned-logit arm tied the in-run clean-room control at
`S=0.19080429731336374`; it did not close the measured
`7.052840218513268e-7 S` gap to exact enumeration. The official-paper `N=32`
arm also tied the control, while consuming 32 objective samples for its only
gradient update. This is a cached, six-bit, single-seed terminal-polish result,
not a family-death verdict for SFESS as a generic discrete optimizer.

**REVIEW STATUS:** the inherited clean-room verdict is `fresh-eyes-reviewed(3)`.
The new measurement receipt is honestly tagged `recovery-written-UNREVIEWED`;
own adversarial round 1 is complete, and the new bytes are
`fresh-eyes-reviewed(0)`. Round 1 found and repaired a borrowed-control flaw by
rerunning the clean-room arm inside the same harness; that repair reset the
clean-pass counter.

**STORES CONSULTED:** research(5715) equations(622) memory(1893) dag(505)
council(277) tasks(96) docs(92). Loaded the unified corpus-query results, the
goldmine correction, frozen-SegNet costate contract, SFESS implementation spec,
landed SFESS receipt, current SFESS code/tests, the canonical task rows, the DAG
FEED, the official GitHub root/README, the ICLR 2025 proceedings paper, the 2024
arXiv precursor, and the first author's publication page. Deliberately did not
load or mutate the live trainer, any live run directory, scorer weights, source
video frames, cloud state, paid providers, or `upstream/evaluate.py`.

## Boundary and source custody

- **MEASURED:** the one shallow clone attempt failed with `Could not resolve
  host: github.com` and rc=128. Browser access exposed the repository root and
  README but not the underlying source files.
- **MEASURED:** the root tree of `github.com/klaswijk/sfess` lists
  `experiments/`, `images/`, `notebooks/`, `samplers/`, `.gitignore`,
  `README.md`, `datasets.py`, `main.py`, and `models.py`; it exposes no
  `LICENSE` file. License status is therefore **UNKNOWN** despite the repository
  being public. No repository code was copied.
- **DERIVED:** this is a publication-equation reconciliation, not a byte-level
  OSS-source diff. The operator's import-eligible correction removed the former
  clean-room policy restriction, but it does not supply absent license terms.
- **MEASURED:** the README identifies the repository as the official PyTorch
  implementation and names upstream code from UCLA-StarAI/SIMPLE, the UvA
  stochastic-kNN notebook, and chendiqian/PR-MPNN. The author's publication page
  lists the later *Differentiable Top-k: From One-Hot to k-Hot* work.
- **INFERRED:** those later pathwise/relaxation routes do not improve this
  black-box exact-through-R SFESS arm without a new biased-estimator admission
  study. No related-repository primitive was imported.

## Three-column reconciliation

| Ours, landed clean-room | Verified official reference | What was missed or deliberately retained |
|---|---|---|
| Exact conditional-Bernoulli sampler; DFT Poisson-binomial PMF; analytic logit score `z-E[z\||z|=k]` | ICLR 2025 equations 2, 3, and 6 compute the conditional-Poisson distribution and DFT normalizer, differentiating the log PMF with autodiff | **MEASURED:** existing DFT matches a polynomial reference within `2e-15`; the analytic logit score has exact finite-support zero-mean tests. No estimator correction was missing. |
| Five-sample leave-one-out control variate with exact outer `1/M` and inner `1/(M-1)` coefficients | ICLR 2025 equation 7 uses the same leave-one-out control variate; final experiments use `N=32` and show a `2,4,8,16,32,64,128` variance ladder | **ADDED:** same-budget `M={2,4,5,8,16,32}` tournament. `M=64` is inadmissible under `B=64` because the counted initial query already consumes one call. |
| Fixed uniform logits; each gradient directly proposes a one-swap candidate | ICLR 2025 Algorithm 2 repeatedly optimizes the subset parameters with variance-reduced gradients until convergence; experiments use Adam at learning rate `1e-4`, `beta1=0.9`, `beta2=0.999` | **ADDED:** iterative learned logits, bias-corrected Adam, deterministic top-k MAP proposal, and an exact separate retention gate. This is independently derived because OSS source bytes were unavailable. |
| Exact recursive conditional-Poisson samples | Official paper uses Gumbel-top-k as approximate conditional-Poisson samples in practice and explicitly says this introduces forward-pass bias | **RETAINED:** exact sampling. Importing the approximation would weaken the exact-through-R estimator contract and requires a separate pre-registered bias/admission study. |
| Fixed-k ladder `k=1..5`; no learned mixture | Official algorithm takes one fixed `k`; no k-mixture control is specified | **NO MISSED PRIMITIVE:** the Pact ladder is already broader. A post-hoc mixture was not invented. |
| Strict exact-S improvement gate at `1e-12 S` | Official algorithm optimizes an expected black-box objective and returns parameters, without Pact's returned-state custody rule | **ADDED/RETAINED:** every proposed state is admitted only after a counted exact objective query improves the incumbent by more than `1e-12 S`; sample values are not silently promoted. |
| No zero-variance event rule | Operator-folded Dr.GRPO/DAPO rule says a near-zero-variance group must skip the update and must not divide by near-zero standard deviation | **ADDED:** event predicate `max(f_i)-min(f_i) <= 1e-12 S` skips Adam and the proposal gate. The estimator performs no standard-deviation division. |

## Imported methods and resolvable citations

- **FROM LITERATURE:** Klas Wijk, Ricardo Vinuesa, Hossein Azizpour (2025),
  *SFESS: Score Function Estimators for k-Subset Sampling*, ICLR 2025,
  OpenReview ID `q87GUkdQBm`. No DOI or arXiv ID for this exact title was found.
  The official proceedings abstract and PDF resolved. Its precursor is Klas
  Wijk, Ricardo Vinuesa, Hossein Azizpour (2024), *Revisiting Score Function
  Estimators for k-Subset Sampling*, arXiv:`2407.16058`; the abstract resolved.
  These support the DFT score estimator, fixed-k formulation, multiple-sample
  control variate, and estimator-cost limitation.
- **FROM LITERATURE:** Manuel Fernández, Stuart Williams (2010), *Closed-Form
  Expression for the Poisson-Binomial Probability Density Function*,
  DOI:`10.1109/TAES.2010.5461658`; the title and DOI resolved. This supports the
  DFT normalizer.
- **FROM LITERATURE:** Diederik P. Kingma, Jimmy Ba (2015), *Adam: A Method for
  Stochastic Optimization*, arXiv:`1412.6980`; the abstract resolved. This
  supports the moment updates. The SFESS paper supplies the task-specific
  `1e-4`, `0.9`, and `0.999` constants.
- **FROM LITERATURE:** Wouter Kool, Herke van Hoof, Max Welling (2019), *Buy 4
  REINFORCE Samples, Get a Baseline for Free!*, OpenReview ID `r1lgTGL5DE`.
  No DOI or arXiv ID exists in the resolved DBLP/OpenReview records. This
  supports the leave-one-out baseline interpretation. The previously guessed
  arXiv:`1904.04998` was explicitly rejected after abstract resolution because
  it is an unrelated monocular-depth paper.
- **LOCAL RULE, NO IMPORTED THEOREM:** the `1e-12 S` zero-variance skip is the
  operator-specified admission rule. No paper result is claimed for that
  threshold.

## Added implementation

- `src/tac/sfess_oss_reconciliation.py`: learned-logit fixed-k SFESS, exact
  conditional samples, RLOO-equivalent SFESS control variate, Adam, deterministic
  MAP proposal, exact descent gate, and zero-variance skip.
- `tools/probe_sfess_oss_reconciliation.py`: same-envelope clean/enriched A/B,
  `k=1..5`, `M={2,4,5,8,16,32}`, exactly 64 counted calls per arm, per-arm atomic
  stage checkpoints, source/input hashes, and false-authority receipt fields.
- `src/tac/tests/test_sfess_oss_reconciliation.py`: positive control that must
  learn the lower-S one-hot state and negative flat-objective control that must
  skip Adam and the exact gate.

The implementation is independent and MIT-marked; no file or code fragment was
imported from the unlicensed-visible repository.

## Remeasurement

Authoritative receipt:
`experiments/results/sfess_oss_reconciliation_20260713T013642Z/measurement_receipt.json`,
SHA-256 `e8ef829e2062ba40a8a8b04c5b37746526f52d9484cfaf992d4699e97755704c`.
The earlier `T012400Z`, `T013600Z`, `T014500Z`, and `T015500Z` directories are superseded
pre-review measurements and carry no verdict authority.

- **MEASURED:** the in-run clean-room control reproduced the sealed result
  exactly: `k=5`, `M=5`, `S=0.19080429731336374`, 64 counted calls.
- **MEASURED:** the best enriched arm was `k=5`, `M=4`,
  `S=0.19080429731336374`, with 12 learned-logit updates, 12 strict gates, and 3
  padding calls. Delta against the in-run clean-room control was exactly `0.0 S`.
- **MEASURED:** the official-paper-calibrated `k=5`, `M=32` arm returned the
  same `S=0.19080429731336374`, with 1 learned-logit update, 1 strict gate, and
  30 padding calls. Delta against the in-run clean-room control was exactly
  `0.0 S`.
- **MEASURED:** exact enumeration remained `S=0.19080359202934188`; the enriched
  result remained `7.052840218513268e-7 S` worse, above the registered
  `1e-12 S` floor.
- **MEASURED:** 9 zero-spread groups occurred across the `M=2` arms; all 9
  updates and their strict gates were skipped. No `M>=4` arm hit the skip.
- **MEASURED:** the receipt records `score_claim=false`,
  `promotion_eligible=false`, `pointer_moved=false`, zero scorer calls, and zero
  paid dispatch.
- **MEASURED:** rerunning the completed output directory returned rc=0 and left
  stage-checkpoint SHA-256 unchanged, proving arm-level resume idempotence.
- **UNKNOWN:** NumPy/Torch/MLX parity for the newly added learned-logit optimizer
  was not measured. The harness is NumPy-only and does not enter a live provider.
  Across-seed variance is also unknown.

## Verdict and control laws

**VERDICT: `NO-GO`, fresh-eyes-reviewed(0), verdict_scope = SFESS as a
frozen-SegNet forward-replacement route for #449 under exact-through-R objective
queries.** The official paper itself identifies multiple `f` evaluations as the
main limitation. Here, every SFESS sample is one exact objective lookup; in a
live frozen-SegNet application that means repeated frozen-teacher forwards, not
replacement of the 95%-dominant forward. `N=32` permits only one update in the
64-call envelope. The enriched estimator therefore does not clear the throughput
bar and does not change the landed Pareto knee.

**VERDICT BOUNDARY:** SFESS remains viable as a terminal cached discrete
optimizer or on objectives whose samples can be batched cheaply. This result
does not kill SFESS globally, does not measure a non-enumerable support, and does
not move any score pointer.

Control laws are closed:

- `k`: fixed ladder `1..5`; no post-hoc mixture.
- `M`: measured ladder `2,4,5,8,16,32`; `M=4` is the smallest tied knee, while
  `M=32` is the final-paper calibration.
- Adam: constant `lr=1e-4`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.
- Admission: event-conditioned exact gate
  `S_proposal < S_incumbent - 1e-12`.
- Zero-variance recess: event-conditioned skip when sampled spread is
  `<=1e-12 S`.
- **FALSIFIER:** any enriched arm had to beat the in-run clean-room arm by more
  than the `1e-12 S` floor at the same 64-call budget; none did.

## Own adversarial review and verification

Round 1 traced each receipt key and unit, source/input hash, sample/gate/padding
count, returned-state rule, and false-authority flag. It found the borrowed
control described above. After repair:

- `ruff check`: green.
- `py_compile`: green.
- focused pytest: `38 passed`.
- restart/resume check: rc=0; stage checkpoint byte hash unchanged.
- positive and negative measurement canaries both green.

The tests would fail if Adam's sign/update were removed or reversed, if the
zero-variance branch still updated or gated, if cardinality changed, or if the
64-call budget were not exact.

## Exact shared-surface handoff

**SUGGESTED TASK-HOOK NOTE** for `sfess_cached_replay_ugc64_20260712`:
`MEASURED OSS-paper reconciliation: in-run clean control and learned-logit
SFESS both returned S=0.19080429731336374 at B=64; official N=32 also tied;
delta=0.0 S; exact gap remains 7.052840218513268e-7 S; #449 forward-replacement
NO-GO unchanged; receipt experiments/results/sfess_oss_reconciliation_20260713T013642Z/measurement_receipt.json;
new bytes own-round-1 reviewed, fresh-eyes-reviewed(0), tests 38 passed.`

**SINGLE DAG FEED ROW:** `FEED-sfess-oss-reconciliation-20260713 — MEASURED:
same-envelope clean control, learned-logit SFESS, and final-paper N=32 all return
S=0.19080429731336374 at B=64 on the sealed UGC64 objective; delta enriched vs
clean=0.0 S and gap vs exact=7.052840218513268e-7 S. Official DFT+LOO pieces were
already present; learned logits, M ladder, and zero-variance skip add no Pareto
movement. VERDICT NO-GO, scope SFESS as #449 frozen-forward replacement only,
fresh-eyes-reviewed(0), score_claim=false, pointer_moved=false; receipt
experiments/results/sfess_oss_reconciliation_20260713T013642Z/measurement_receipt.json.`

Shared DSL, canonical-equation registry, DAG, and task ledger were deliberately
not edited in this subagent lane; the master reconciliation agent owns those
shared-file updates and the serializer landing.
