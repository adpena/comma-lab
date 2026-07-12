# SFESS cached 64-state replay — measured verdict (2026-07-12)

`lane_id=lane_sfess_cached_replay_ugc64_20260712` · `score_claim=false` ·
`promotion_eligible=false` · `research_only=true` · `$0` · `scorer_calls=0`

## Answer first

**NO-GO — exact enumeration and `(1+1)-ES` still win this cached instance.**

**MEASURED:** at 64 counted cached-objective lookups per arm, the best non-degenerate SFESS arm
was `k=5` with returned `S=0.19080429731336374`. Exact enumeration and `(1+1)-ES` both returned
`S=0.19080359202934188`. UGC, DisARM, RLOO, and `SFESS k=5` tied at
`S=0.19080429731336374`.

**DERIVED:** the best non-degenerate SFESS arm remained
`7.052840218513268e-7 S` above exact enumeration. That is larger than the registered
`1e-12 S` comparison floor. Fixed-cardinality structure therefore did not change the same-budget
baseline ranking on this six-bit objective.

**MEASURED:** `k=6` returned the global minimum, but its support contains exactly one state.
It is a structural control, not SFESS estimator evidence.

**REVIEW STATUS:** this verdict is `fresh-eyes-reviewed(3)` after three consecutive independent
CLEAN passes on the current post-fix bytes. The immutable receipt retains its honest
`recovery-written-UNREVIEWED` at-measurement tag. The inherited UGC baseline rows are
`fresh-eyes-reviewed(3)`.

## Measured result

Machine receipt:
`experiments/results/sfess_cached_replay_ugc64_20260712T214520Z/measurement_receipt.json`
(SHA-256 `aa296c61fde712f9a2207ff5ecf9298c2506c92e3a48af8ac2af3d9bc83e6c9e`).

**MEASURED provenance:** receipt schema v2 contains five module-alias custody rows covering four
unique repository Python source paths loaded by the replay. Their combined custody-tree SHA-256 is
`ef65258b6c652b040a3ecbc5a8a6e2a3c12b9a30460fcea286ff9421c54b84fb`. Every row records its
path, bytes, SHA-256, base-commit blob identity, and whether the measured bytes matched that base.
The aggregate base-match flag is false because new SFESS sources were not yet tracked at measurement
time; the source hashes, rather than the pre-landing Git identifier, are the code custody authority.
The receipt also records the exact argv, selected environment variables, Python, NumPy, Pydantic,
platform, machine, and processor. Resume refuses any source-tree drift.

**REVIEW STATUS:** a fresh-eyes provenance review found that the earlier v1 receipt recorded only
the pre-landing Git identifier. That receipt is superseded. The v2 receipt above repairs the bug
class by binding the actual execution-source bytes. A later fresh-eyes review found that the
`20260712T211015Z` receipt had eagerly loaded the boundary-math package, including renderer and
SegNet-loader modules; that receipt is also superseded. The current replay moves the executable
core outside that eager package and executes only the isolated SFESS DSL leaf. The clean-pass
counter reset to zero after this repair. Fresh-eyes pass 1 then showed that a self-consistent
foreign snapshot could preserve its own seed and sample count while the stage row reported the
registered tuple, and that source-video provenance was asserted rather than directly joined. The
`20260712T213113Z` receipt is therefore superseded. The current resume path binds `k`, estimator
sample count, seed, and comparison floor to the compiled policy before restoring records; the
current policy also hashes `upstream/videos/0.mkv` and rechecks its stat identity at every lookup.

| Arm | Returned mask | Returned S | Counted lookups | Status |
|---|---:|---:|---:|---|
| exact enumeration | `111111` | `0.19080359202934188` | 64 | inherited measured baseline |
| `(1+1)-ES` | `111111` | `0.19080359202934188` | 64 | inherited measured baseline |
| UGC | `101111` | `0.19080429731336374` | 64 | inherited measured baseline |
| DisARM | `101111` | `0.19080429731336374` | 64 | inherited measured baseline |
| RLOO | `101111` | `0.19080429731336374` | 64 | inherited measured baseline |
| SFESS `k=5` | `101111` | `0.19080429731336374` | 64 | MEASURED cached replay |
| SFESS `k=4` | `101101` | `0.19080513567307628` | 64 | MEASURED cached replay |
| SFESS `k=3` | `100101` | `0.19080614714701477` | 64 | MEASURED cached replay |
| SFESS `k=2` | `100001` | `0.19080778831677525` | 64 | MEASURED cached replay |
| SFESS `k=1` | `000001` | `0.19080943457333555` | 64 | MEASURED cached replay |

The masks are written in the receipt's little-endian list order, not as integer bit strings.

**MEASURED:** every non-degenerate SFESS arm reached the exact within-`k` minimum in the sealed
64-state table. Each used 50 SFESS sample lookups, 10 separately counted strict-gate lookups, one
counted initial lookup, and three padding lookups. Sampled control-variate states were not retained
as candidates.

**MEASURED containment:** all seven replay/control arms have 64 query records and 64 re-derived
policy decisions, for 448 counted lookups and 448 re-derived decisions in total. Every decision
admitted the cached lookup. Every decision refused live-gradient admission and retained the
`full_teacher` fallback. The fresh process loaded only `tac`, the top-level SFESS core, the isolated
SFESS policy leaf, and the probe. It loaded no boundary-math package, Torch, MLX, Modal, scorer,
renderer, trainer, repacker, `mc_finisher`, or upstream evaluator module. The receipt records
`scorer_calls=0`, `archive_repack_calls=0`, `paid_dispatch=false`, and `cloud_dispatch=false`.

**MEASURED resumability:** a completed `--resume-from` execution reproduced measurement-receipt
SHA-256 `aa296c61fde712f9a2207ff5ecf9298c2506c92e3a48af8ac2af3d9bc83e6c9e` byte-for-byte.
Every cardinality has a preserved stage snapshot and stage receipt.

**MEASURED source join:** the replay directly hashed `upstream/videos/0.mkv` as
`2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` with
`37,545,489` bytes. It did not decode video frames. The compiled policy preserved that file's full
SHA-256 plus stat fingerprint and every one of the 448 lookup decisions rechecked the stat identity.

## Clean-room estimator and control laws

**FROM-LITERATURE:** Klas Wijk, Ricardo Vinuesa, and Hossein Azizpour (2024),
*Revisiting Score Function Estimators for k-Subset Sampling*, arXiv:2407.16058
([abstract](https://arxiv.org/abs/2407.16058)). The fetched abstract/PDF matched the title and
authors. This landing independently implements its conditional-Bernoulli score identity and
five-sample leave-one-out control variate; no source code was imported.

**FROM-LITERATURE:** Manuel Fernández and Stuart Williams (2010),
*Closed-Form Expression for the Poisson-Binomial Probability Density Function*,
DOI:10.1109/TAES.2010.5461658
([DOI](https://doi.org/10.1109/TAES.2010.5461658)). The resolved metadata matched the named paper.
Its DFT Poisson-binomial probability is the conditioning normalizer.

For `p_i=sigmoid(phi_i)` conditioned on `sum(z)=k`, the implemented exact logit score is
`z - E[z | sum(z)=k]`. With five independent samples, each sample's score is multiplied by its
objective value minus the mean of the other four values. The result is an unbiased score-function
gradient because each leave-one-out baseline is independent of its own zero-mean score.

The Pact-specific swap controller has no claimed supporting paper. It is a local derived comparison
rule:

- **Constant:** each independent arm fixes `k` to one of `{1,2,3,4,5}` and fixes `B=64`.
- **Constant:** the conditional policy stays uniform with `p_i=k/6`; there is no learning-rate knob.
- **Constant:** each estimate uses five samples, following Wijk et al.'s registered setting.
- **Event-conditioned predicate:** remove the selected coordinate with the largest current estimated
  gradient and add the unselected coordinate with the smallest current estimated gradient.
- **Event-conditioned predicate:** accept only a separately queried proposal lower than the
  incumbent by more than the registered `1e-12 S` comparison floor.
- **Completion guarantee:** stop at exactly 64 lookups; residual calls that cannot fund five samples
  plus one gate re-query the incumbent and are labeled padding.
- **Constant noise floor:** values within `1e-12 S` tie, inherited from the source receipt's exact
  composition verification tolerance.
- **Failure law:** custody, order, finiteness, context, state, age, cardinality, or budget failure
  refuses the cached replay. Attempted live use retains the real full teacher.

## Scope and risk

**UNKNOWN:** across-seed variation is unknown because the registered spine has one seed.

**INFERRED:** the measured result is unsurprising on a support that exact enumeration completely
consumes at the same budget. It does not answer whether SFESS helps on a non-enumerable terminal-edit
support.

**VERDICT SCOPE:** instance/formulation only. The negative applies to the SHA-pinned six-bit table,
seed `396400`, fixed cardinalities `1..5`, the uniform conditional policy, five-sample estimator,
strict returned-state gate, and 64-lookups-per-arm budget. It is not a family death verdict and is
not contest score evidence.

The submitted `[contest-CPU]` pointer is unchanged. The separate defensive bank is unchanged.
`upstream/evaluate.py` was not invoked. The live trainer and protected V9 run were not read or
mutated. No paid or cloud dispatch occurred.

## Triality and system hook

- DSL: `tac.witness_dsl.sfess_cached_replay_policy`; terminal-objective cache lookup only,
  `produces_costate=false`, live fallback `full_teacher`.
- Equation: `sfess_fixed_k_cached_replay_ranking_v1`.
- DAG: `FEED-sfess-cached-replay-ugc64`; instance-scoped NO-GO, exact-enumeration wall preserved.
- Canonical task: `sfess_cached_replay_ugc64_20260712`.

## STORES CONSULTED

Loaded: `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; operator `MEMORY.md`
top entries; `.omx/research/goldmine_hunt_20260712.md`;
`.omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md`;
`.omx/research/ugc_terminal_polish_ab_20260712.md` and its implementation spec;
`.omx/research/policy_gradient_variance_reduction_survey_20260712.md`; the existing gradient
provider, DSL, canonical equation, probe, finisher, focused tests, exact-enumeration JSONL,
candidate manifest, measurement receipt, canonical task/lane/subagent ledgers, latest sister
findings/session/design/council memo, Wijk et al. arXiv abstract/PDF, and Fernández-Williams DOI
metadata.

Deliberately not loaded or invoked: the live trainer; the protected V9 run; scorer weights;
source-video frames; Torch/MLX execution; renderer/repacker code during replay; cloud or Modal state;
paid dispatch; `upstream/evaluate.py`.

## Consumer-leg disposition (triality drift-detector, 2026-07-12)
`src/tac/witness_dsl/sfess_cached_replay_policy.py` lives in the DSL package but is a RESEARCH-PROBE
policy, NOT a campaign `Lever`. VERIFIED: (a) not exported from `witness_dsl/__init__.py`; (b) it is
pydantic custody/fingerprint/objective-context models (`SFESSCacheCustody`, `SFESSObjectiveContext`,
`SFESSCachedReplayPolicy`), not a `Lever` factory; (c) `lever_registry.completeness()` returns NONE for
`sfess` — it is not in the campaign lever registry. Therefore the campaign consumer surfaces
(`tools/dashboard_server.py`, `tools/costate_digest.py`, `src/tac/witness_dsl/schedule_readback.py`,
`src/tac/witness_control/producer_bridge.py`) render campaign levers via the registry and correctly do
NOT render this — NO consumer update is needed. Sole consumers: `tools/probe_sfess_cached_replay.py` +
the sfess provider mode in `segnet_gradient_replacement.py`. Disposition = research-probe-scoped /
out-of-campaign-render. `[consumers-generic]`
