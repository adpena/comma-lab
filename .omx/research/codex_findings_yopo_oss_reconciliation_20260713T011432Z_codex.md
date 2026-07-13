# Codex findings: YOPO OSS reconciliation for #449

**Date:** 2026-07-13 UTC  
**Review status:** `recovery-written-UNREVIEWED`; own adversarial round 1 completed, no fresh-eyes pass yet  
**Verdict scope:** formulation only; exact inherited scope is recorded below  
**Detailed receipt:**
`experiments/results/yopo_oss_reconciliation_20260713T011432Z/reconciliation.md`

**STORES CONSULTED:** unified research/equations/memory/DAG/council/tasks/docs query; operating
contracts; goldmine/frozen-SegNet/share-ge2 memos; landed provider/harness/equation/tests/final receipt;
canonical task/lane/probe state; official YOPO GitHub root/README; author repository list; official
YOPO arXiv paper; official Seidman PMLR/arXiv follow-up; README-linked C++ repository root. Deliberately
excluded live trainer/run dirs, cloud/provider state, paid dispatch, and `upstream/evaluate.py`.

## Finding

**DERIVED — SOURCE-LEVEL RECONCILIATION BLOCKED; NO VERDICT CHANGE,
recovery-written-UNREVIEWED; own-round-1-completed.** YOPO's accessible official `m`-`n` paper specification does not
contain a cheaper exact validation/refresh scheme. It obtains its speed by running `n` cheap projected
first-layer updates with a frozen costate and no per-inner-step exact teacher validation. That is not
equivalent to Pact's exact through-`R` descent contract. Nested Python implementation details remain
**UNKNOWN** because neither permitted source-access path was available.

**MEASURED — inherited authority, fresh-eyes-reviewed(2):** the current `blocks[0]`, `K={1,2,4}`
matrix remains scoped `NO-GO`. Costate direction was nearly exact, but 402 operational validation
forwards made every complete `K>1` arm slower than `K=1`. Final receipt SHA-256:
`a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3`.

**MEASURED DELTA: UNKNOWN.** No source was imported and no compatible semantic delta existed, so
the settled harness was not rerun under a false "OSS-enriched" label. The derived verdict delta is
none.

## License and source blocker

The one permitted shallow-clone attempt failed with DNS `rc=128`. The official root tree exposes no
repository-level `LICENSE` entry or GitHub license label. Nested headers and upstream commit identity
could not be inspected because nested static pages failed and no in-app browser connection existed.
The task's read-license-before-copy rule therefore blocks import. No source bytes were copied.

## Literature-bound derivation

- Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, and Bin Dong (2019), *You Only
  Propagate Once: Accelerating Adversarial Training via Maximal Principle*, arXiv:1905.00877,
  DOI `10.48550/arXiv.1905.00877`.
- Jacob H. Seidman, Mahyar Fazlyab, Victor M. Preciado, and George J. Pappas (2020), *Robust Deep
  Learning as Optimal Control: Insights and Convergence Guarantees*, arXiv:2005.00616,
  DOI `10.48550/arXiv.2005.00616`, PMLR 120:884-893.

Seidman et al. bound the frozen-costate error and show an oracle-error contribution growing as
`(n-1)^2`, while the finite-inner-optimization term decays with `m*n`. Their theorem assumes smoothness
and local strong concavity that are not established for Pact's realized-through-`R` witness objective.
It therefore explains the need for a finite `n` but does not certify exact descent or replace validation.

## Task and DAG handoff

Keep task `449_yopo_first_layer_costate_probe_20260712` completed/green and its original reactivation
criteria unchanged. Candidate YOPO fragment for the master DAG FEED:

> **YOPO OSS reconciliation:** official YOPO-`m`-`n` confirms banked first-layer costate reuse but
> contains no per-inner-step exact validation; its apparent cheaper cadence deletes that check.
> Seidman 2020 bounds inexact-oracle error and predicts a finite inner-count knee but does not certify
> Pact through-`R` descent. No licensed source import or comparable enriched rerun occurred because
> the repository-level license and nested source could not be resolved after the single clone failed.
> The inherited `blocks[0]`, `K={1,2,4}` `NO-GO` remains formulation-scoped; pointer `0.1880443979880752`
> is unmoved; `score_claim=false`.

Shared task/DAG files were intentionally left untouched for the master agent's collision-safe single
append.

## Own round-1 result

Fresh receipt-to-evidence recomputation matched the receipt SHA, admission status/scope, all work
counts, all three minimum cosine metrics, and the current pointer (`rc=0`). The focused YOPO
provider/equation suite passed 14 tests with 11 deselected in 0.88 seconds (`rc=0`). The conclusion is
still tagged `recovery-written-UNREVIEWED` because this was self-review, not an independent
fresh-context pass.
