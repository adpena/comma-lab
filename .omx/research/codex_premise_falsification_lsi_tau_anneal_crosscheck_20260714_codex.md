# Codex premise falsification — LSI tau-rate versus #318

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]`, unchanged.  
**Lane:** `lane_lsi_tau_anneal_metric_synthesis_500_20260714`  
**Status:** `PREMISE_FALSIFIED_BEFORE_WIRE_IN`  
**Scope:** the proposed direct numerical comparison, not log-Sobolev, spectral-gap curricula,
eikonal stabilization, or reachable-decision geometry as families.

## Finding

The requested equality test cannot lawfully be run from retained bytes. Four premises fail:

1. The full arXiv:2605.29035 methodology is not present locally and could not be retrieved through
   the available official-source/browser surfaces. The only local paper description is two
   paragraphs saying the fixed-temperature entropy contraction is one half the cycle spectral
   gap. That is not a deep read and does not custody its normalization, hypotheses, or proof.
2. A fixed-temperature contraction rate does not by itself set a cooling rate. For a moving
   equilibrium manifold, the missing term is its path sensitivity. If the tracking error obeys
   `e_dot <= -rho e + beta_tau |tau_dot|`, invariance of an allowed tube `e<=r` gives
   `|tau_dot| <= rho r/beta_tau`. Thus, under the locally stated `rho=lambda_gap/2`,
   `c=r/(2 beta_tau)`, not a universal guessed constant.
3. #318 does not produce `tau_dot`. It derives the static explicit-step inequalities
   `eta lambda_eik c_a^2/(8 epsilon^2)<=1` and
   `eta lambda_eik epsilon^2 k_max^4<=2`. A DE tau-rate requires an additional, separately
   custodied differentiation/continuation law.
4. The retained gap and DE rows are not the same telemetry. The ep100 generic Adam/Hessian proxy
   `lambda_pre=3.66e6` was already measured-falsified as #318's governing group. The retained
   K=128 HVP spectrum is ep650, subset/MLX advisory, and is neither the ep100 state nor a
   reversible-Markov entropy gap.

## Honest verdict

`NO_VERDICT_SOURCE_CUSTODY` for numerical LSI-versus-DE agreement. The only supported statement is
qualitative: both mechanisms demand slower motion as their own relaxation/stability margin closes.
Calling that numerical agreement would collapse different operators, units, states, and laws.

## Optimal-form reformulation queue

1. Recover and hash the full paper; verify the generator, entropy convention, theorem constant,
   finite-cycle assumptions, and whether the result is continuous- or discrete-time.
2. Define an actual reversible diffusion on a named V9 state space; do not call the training
   Hessian a Markov spectral gap by analogy.
3. On one hashed EMA state and one epoch, measure `lambda_gap`, the equilibrium-path sensitivity
   `beta_tau`, a registered tracking tube `r`, and #318's `c_a, epsilon, eta, lambda_eik, k_max`.
4. Derive a DE `tau_per_epoch` cap from the static safe-set boundary, with exact units and the same
   state hash.
5. Only then run the executable agreement test and classify `AGREE`/`DISAGREE` at a preregistered
   tolerance.

## Stores consulted

The paper warm-start summary and V9 intake DAG; #318 differential-equation memo and v6 council;
stepping-instability receipt; ep650 HVP spectrum; current curriculum/metric/basis/RIPO/Bregman
artifacts; canonical pointer, lane, subagent, and live-inbox stores.

