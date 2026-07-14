# Build spec — corrected K=2 n600 economic adjudication

Date: 2026-07-14  
Lane: `lane_p0_backward_closer_20260713`  
Authority: local cached n600 training-signal evidence only; no score, pointer, live run, training,
paid dispatch, or provider activation.

## Defect being closed

The source-sealed v3 replay correctly measures per-state fidelity, exact forward guards, and
accepted/fallback outcomes. Its preregistered diagnostic accounting undercharges a rejected reuse:
after the stale-candidate forward guard fails, rollback restores the pre-step state, so the guard
forward cannot be reused for the required full exact refresh. A rejection pays a full additional
forward plus backward.

For `p=A/600`, `q=1-p`, and diagnostic forward share `alpha`:

```text
two-step baseline cost                   = 2
guarded expected cost                    = 1 + alpha + q = 2 + alpha - p
corrected teacher-slice speedup          = 2 / (2 + alpha - p)
positive diagnostic speedup              iff p > alpha
beat forward-elimination Amdahl ceiling  iff p > 3*alpha
exact-backward call amortization         = 2 / (2 - p)  (unchanged)
exact-backward call reduction            = p / 2        (unchanged)
```

At the sealed `alpha=0.1784755863`, the corrected strict gate is
`p > 0.5354267589`. The superseded gate `p > 0.4344985574` must never authorize a policy.

## Preservation and re-adjudication contract

1. Let the immutable v3 replay finish all 600 rows and its three stage manifests. Do not edit or
   delete any v3 row, manifest, run contract, original receipt, or completion seal.
2. The original v3 economic verdict is labeled
   `SUPERSEDED_INVALID_FALLBACK_CHARGE`; its fidelity rows remain measurements.
3. Emit a new adjacent wrapper, proposed path
   `experiments/results/p0_costate_reuse_k2_n600_v3_20260713/corrected_adjudication_receipt.json`,
   with schema `p0_costate_reuse_k2_corrected_adjudication.v1`.
4. The wrapper recursively binds and revalidates:
   - original `run_contract.json`, `measurement_receipt.json`, and `complete.json` bytes/SHA-256;
   - all three stage-manifest bytes, run-contract hashes, state counts, and tree hashes;
   - exactly 600 unique pair rows, each pair index, bytes, SHA-256, row self-hash, and run-contract hash;
   - original objective/scorer/admission-content hashes and false-authority fields.
5. Recompute only the deterministic aggregate accounting and corrected admission predicates from the
   sealed rows. No scorer call, teacher call, renderer call, or row rewrite is allowed.
6. The corrected wrapper records both the superseded verdict and the corrected verdict, the exact
   formulas above, `alpha`, `p`, `q`, threshold, exact-call economics, all accepted-row fidelity gates,
   and `whole_epoch_speedup=UNKNOWN_IN_LOOP_TIMER_OWED`.
7. Pin only the corrected-wrapper SHA-256 in a code-reviewed allowlist. The DSL must reject the
   original receipt, an arbitrary caller-supplied hash, missing nested bytes, tree drift, or a failed
   corrected gate.

## Code surfaces

- Correct future accounting in `tools/probe_p0_costate_reuse_k2.py` and focused probe tests.
- Add an isolated deterministic adjudicator (new tool or a clearly separate subcommand) that consumes
  the sealed v3 artifacts under the preservation contract above.
- Update `src/tac/witness_dsl/exact_costate_reuse_policy.py` to trust only the corrected-wrapper
  allowlist and recursively validate its nested custody.
- Correct `src/tac/canonical_equations/exact_costate_reuse_k2_20260713.py` and tests.
- Make terminal SPSA/ES equation admission consult the empty reviewed certificate registry; caller
  booleans/hashes cannot self-certify.
- Populate the synthesis memo, standalone DAG FEED, and GO packet with sealed values and explicit
  supersession language.

## Live-trainer boundary

Live trainer integration remains refused. The replay certifies the primary SegNet CE input costate;
the monolithic live gradient also contains PoseNet, additional SegNet paths, and non-scorer terms.
The operator-GO prerequisite must factor the complete supported SegNet-backed scalar, preserve exact
Pose/non-scorer gradients, refuse unsupported microbatch/dual-seed paths, register complete provider
state under the resume registry, and pass exact-provider gradient parity before K=2 activation.

## Acceptance tests

- Corrected algebra unit tests at `p in {0, alpha, 3*alpha, 1}` with strict-boundary behavior.
- A rejection is charged `F_guard + (F+B)_refresh`, never `F_guard+B`.
- Original v3 receipt alone cannot admit the DSL.
- Corrected wrapper fails on any nested byte/tree/row/objective/scorer/hash mutation.
- Corrected wrapper covers exactly n600 and preserves all accepted-row rel-L2/d_seg-regret gates.
- Arbitrary terminal dimension booleans/hashes remain refused while the certificate registry is empty.
- Existing controller/policy/terminal/equation tests, ruff, py_compile, lane validation, and three clean
  adversarial reviews pass.

Pointer delta remains `NONE`. The in-loop component/whole-step timer remains operator-GO owed.
