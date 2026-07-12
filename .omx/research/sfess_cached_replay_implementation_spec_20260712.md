# SFESS cached 64-state replay implementation contract — 2026-07-12

`lane_id=lane_sfess_cached_replay_ugc64_20260712` · `$0` · `score_claim=false` ·
`promotion_eligible=false` · `research_only=true`

## Answer and boundary

Build one clean-room, zero-scorer replay of the SFESS fixed-cardinality estimator on the already
measured six-bit objective. The replay may read only the sealed 64-row exact-enumeration JSONL,
its measurement receipt, and its candidate manifest. It must not import or instantiate a scorer,
renderer, archive repacker, live trainer, Torch, MLX, cloud client, or the protected V9 run.

The result is a throughput/estimator-routing verdict, not a score claim. The submitted
`[contest-CPU]` pointer remains outside this task. The local defensive-bank row also remains
outside this task.

**REVIEW STATUS:** the implementation contract is `pre-registered-only`. The inherited UGC
receipt and its scoped verdict are `fresh-eyes-reviewed(3)`. New SFESS code and results begin as
`recovery-written-UNREVIEWED`; every fix resets the clean-pass counter.

## Settled input and comparison surface

- **MEASURED:** the cached objective source is
  `experiments/results/ugc_terminal_polish_ab_20260712/search_exact_enumeration_accepted_proposals.jsonl`,
  SHA-256 `249c19af0b8c117412de491e944bcacb6194c870c9d9ec57d5c93b5e55f1a979`.
- **MEASURED:** it contains 64 unique masks in little-endian order, where state `i` has bit
  `j = (i >> j) & 1`. Every value is finite. The all-zero state is
  `0.19081182131424618`. The all-one state is the exact minimum,
  `0.19080359202934188`.
- **MEASURED:** the source measurement receipt has SHA-256
  `b2f7a87b43ce5face651da5caf4cd723884445d4fa04f92c007811b40d32b357`.
- **MEASURED:** the candidate manifest has SHA-256
  `fb99a24410d4b7dbb8ccf3d8ecba67c8f2033b732440ba32a623e0dec9d6fce0`.
- **DERIVED:** exact enumeration cannot be strictly beaten at budget 64 on this complete table.
  A stochastic arm can tie it or reach it earlier; it cannot produce a lower legitimate value.
- **MEASURED:** the prior UGC, DisARM, and RLOO arms discard internal estimator-sample values and
  retain only a separately queried one-bit exact-gated proposal. SFESS must use the same returned-
  state rule. Reporting the minimum of its control-variate samples would change the formulation.

## Clean-room derivation and citations

At the point of implementation, use only these equations and independently written NumPy code:

1. **FROM-LITERATURE:** Klas Wijk, Ricardo Vinuesa, and Hossein Azizpour (2024),
   *Revisiting Score Function Estimators for k-Subset Sampling*, arXiv:2407.16058.
   The arXiv abstract and v2 PDF were fetched and matched to the named paper. Its conditional
   Bernoulli score-function identity, DFT normalizer, and multiple-sample leave-one-out control
   variate are the SFESS source.
2. **FROM-LITERATURE:** Manuel Fernández and Stuart Williams (2010), *Closed-Form Expression for
   the Poisson-Binomial Probability Density Function*, DOI:10.1109/TAES.2010.5461658. The DOI,
   authors, title, journal, volume, and pages were resolved in the literature lookup. Its DFT
   Poisson-binomial PMF is the normalizer used by Wijk et al.

No SFESS/YOPO/JRD official licensed implementation is imported. No supporting paper is claimed for
the Pact-specific exact-gated swap controller; that controller is a local derived comparison rule.

For probabilities `p_i=sigmoid(phi_i)` and a mask `z` conditioned on `sum(z)=k`:

`p_phi,k(z) = prod_i p_i^z_i (1-p_i)^(1-z_i) / P_phi(sum b_i=k)`.

The exact logit score is:

`grad_phi log p_phi,k(z) = z - E_phi,k[z]`.

For `M=5` independent exact k-subset samples and objective values `f_j`, the leave-one-out
control-variate estimate is:

`g = (1/M) sum_j score(z_j) * (f_j - sum_{l != j} f_l/(M-1))`.

## Control laws

- **Cardinality ladder:** constant arms `k in {1,2,3,4,5}`. Each arm gets its own matched
  `B=64` cached-lookup budget. `k=0` and `k=6` are deterministic structural controls, not SFESS
  wins. Choosing the best k afterward is explicitly post-hoc exploratory; it is not a 64-call
  mixed-k policy.
- **Sampling:** self-deriving conditional Bernoulli probabilities initialized at `p_i=k/6` and
  sampled exactly from the conditional law. No Gumbel approximation and no objective enumeration
  enter sampling.
- **Control variate count:** constant `M=5`, anchored to Wijk et al.'s registered experimental
  setting. Every sampled objective lookup counts.
- **Proposal:** event-conditioned fixed-k swap. Remove the selected coordinate with largest
  estimated gradient and add the unselected coordinate with smallest estimated gradient; query the
  resulting k-subset once.
- **Acceptance:** event-conditioned strict exact gate. State changes only when the separately
  queried proposal is lower than the incumbent by more than the registered `1e-12 S` comparison
  floor; sampled control-variate states never become the returned state directly.
- **Completion:** each arm stops at exactly 64 counted lookups. Residual calls too small for a
  five-sample-plus-proposal step re-query the current state and are labeled padding.
- **Noise floor:** comparisons within `1e-12 S` are ties. This is anchored to the prior final-mask
  exact-composition verification tolerance; the single-seed across-seed variance remains UNKNOWN.
- **Failure:** any missing/extra/reordered/duplicate/nonfinite state, custody/stat drift, objective
  fingerprint drift, stale decision evidence, or budget overrun refuses the replay. It never opens
  a scorer fallback. Attempted live-gradient use remains `full_teacher` fallback because SFESS
  produces no costate.
- **Resume binding:** a snapshot's `k`, five-sample estimator count, seed, and comparison floor must
  exactly match the compiled policy before its trace can be restored; stage metadata is emitted from
  that checked search state rather than from parallel hardcoded labels.
- **Source-video custody:** hash and byte-count `upstream/videos/0.mkv` directly at compile time and
  recheck its stat fingerprint at every cached lookup. This is a provenance join only; no frame is
  decoded and no scorer is invoked.

## Falsifier and verdict rule

The pre-registered falsifier is: after 64 counted cached lookups, no non-degenerate SFESS arm is
strictly below both the `(1+1)-ES` and exact-enumeration returned states by more than `1e-12 S`.

- `GO`: a non-degenerate k arm beats every registered baseline under that rule.
- `NO-GO`: exact enumeration remains lower or tied; any `k=6` tie is reported as a one-state
  degeneracy, not estimator evidence.
- `NEEDS-MORE`: reserved for a separately registered non-enumerable support where exact enumeration
  cannot consume the entire matched budget. It cannot overwrite this cached-instance verdict.

## Triality and outputs

- **Provider seam:** a separate terminal-objective provider mode in
  `tac.boundary_math.segnet_gradient_replacement`; it must not enter `GradientMode`.
- **DSL:** `tac.witness_dsl.sfess_cached_replay_policy`; all semantics are typed and frozen.
- **Equations:** `tac.canonical_equations.sfess_k_subset_cached_replay_20260712` after measurement.
- **DAG:** one FEED row with the measured ranking, scoped verdict, pointer-unmoved statement, and
  receipt hash.
- **Receipt:** `experiments/results/sfess_cached_replay_ugc64_<UTC>/`, with per-k stage snapshots,
  exact query counts, input hashes, source-video provenance, and zero-scorer import proof.

## STORES CONSULTED

Loaded: `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; operator `MEMORY.md`
top entries; `.omx/research/goldmine_hunt_20260712.md`;
`.omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md`;
`.omx/research/ugc_terminal_polish_ab_20260712.md` and its implementation spec;
`.omx/research/policy_gradient_variance_reduction_survey_20260712.md`; the existing provider,
DSL, equation, probe, finisher, tests, exact-enumeration JSONL, candidate manifest, measurement
receipt, canonical task/lane/subagent ledgers, latest sister findings/session/design/council memo,
Wijk et al. arXiv abstract/PDF, and the Fernández-Williams DOI metadata.

Deliberately not loaded or invoked: the live trainer; the protected V9 run; any scorer weights;
source-video frames; Torch/MLX execution; renderer/repacker code during the actual replay; cloud or
Modal state; paid dispatch; `upstream/evaluate.py`.
