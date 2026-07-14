---
title: "Codex post-commit findings — P0 backward closer"
date_utc: "2026-07-14"
reviewed_commit: "26977fb41c94c215baeb11fd33be803a78927267"
lane_id: "lane_p0_backward_closer_20260713"
research_only: true
score_claim: false
pointer_moved: false
---

# Outcome first

Three independent post-commit reviews returned **NOT CLEAN**. The clean-pass counter reset to
`0/3`; `.omx/state/review_counter.jsonl` records all three rounds against exact commit
`26977fb41c94c215baeb11fd33be803a78927267`.

## Findings and closure

1. **HIGH — production measurement authority was caller-mintable.** A caller could hash any
   repo-local regular file, pass the hash as `trusted_receipt_sha256`, and create a production
   `measured` curriculum row. The code now requires an exact candidate/path/SHA entry in a
   code-reviewed trust-root registry *and* a supported typed receipt validator. The default trust
   registry is empty. README bytes, a forged supported-schema receipt, and the former synthetic
   `{"archive_sha256":"authority"}` fixture all fail closed.
2. **MEDIUM — clean-checkout research signals could disappear.** The K2 and sparse-adjoint seed
   rows referenced ignored experiment receipts. They now bind tracked artifacts and exact hashes.
   Missing or changed research custody downgrades the signal to an explicit
   `RECEIPT_CUSTODY_BLOCKED` reformulation row instead of silently erasing it.
3. **MEDIUM — `144` was mislabeled as guard fallback.** The corrected machine surfaces now state
   **MEASURED** `67` actual guard fallbacks, `77` terminal/blocked states, and `144` charged
   nonaccept states. Diagnostic economics are labeled **DERIVED COUNTERFACTUAL / NOT ADMITTED**;
   admitted bulk speedup remains `1.0x` and admitted reduction remains `0%`.
4. **MEDIUM — timing was routed before fidelity.** The current formulation is
   `FIDELITY_BLOCKED_PENDING_NEW_PREREGISTERED_FORMULATION`. An in-loop timer becomes owed only
   after a new preregistered formulation passes a fresh sealed n600 fidelity gate and provider /
   resume parity. The probe ledger carries an append-only superseding row; no current timer GO is
   requested.

## Verdict scope

The authority finding applies to production `measured` admission in
`tac.witness_dsl.curriculum_candidate_pool`. It is not a claim that existing contest receipts are
false; the live curriculum pool contained zero production-measured rows at audit time. The K2
negative remains scoped to the sealed direct raw-input-costate zero-order-hold K=2 formulation.
Sibling costate-provider families remain open.

## Triality and verification

- **DSL:** production admission gate and durable K2/sparse/terminal research signals.
- **Equation:** corrected 67/77/144 accounting and fidelity-before-timing law.
- **DAG:** the canonical probe outcome remains blocking and records the reactivation chain.
- Focused verification before the corrective commit: `161 passed`; Ruff, `py_compile`, and
  `git diff --check` clean.

## STORES CONSULTED

- `.omx/state/review_counter.jsonl`
- `.omx/state/probe_outcomes.jsonl`
- `.omx/state/canonical_equations_registry.jsonl`
- `.omx/state/curriculum_candidate_pool.jsonl`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- tracked K2 compact receipt, sparse-adjoint memo, terminal handoff receipt, synthesis memo, and
  DAG FEED under `.omx/research/`

## Second exact-commit review and closure

Commit `668a2886842759d3867e7e47ecfc4e15e6dea986` passed the independent science/math
re-derivation but remained **NOT CLEAN** on authority and integration. Review-counter rounds 4--6
preserve those exact-commit verdicts. The subsequent closure does not count that isolated science
pass toward a future three-clean seal.

1. **HIGH -- allowlisting was necessary but not sufficient.** The first correction still left a
   future typed production receipt able to self-attest an incomplete evaluator transaction. The
   production validator registry is therefore now deliberately empty: even an allowlisted,
   byte-hash-matching, semantically well-formed v1 receipt is refused, so this lane cannot mint a
   production `measured` row. Re-enablement requires a new reviewed schema that binds the exact
   archive copy actually evaluated, an immutable executed runtime, scorer import origins, the
   canonical n600 names/GT transaction, and their complete pre/post custody. Research-diagnostic
   rows remain separately hash-bound and non-promotable. This is a fail-closed blocker, not a claim
   that the existing general-purpose `contest_auth_eval.py` transaction already supplies those
   missing guarantees.
2. **HIGH -- future producers still exposed obsolete timing routing.** The raw probe now emits
   `PENDING_CORRECTED_ADJUDICATION_NO_GO`. Corrected non-admission emits
   `FIDELITY_BLOCKED_PENDING_NEW_FORMULATION`; even a hypothetical fidelity admission stops at
   `FIDELITY_ADMITTED_PENDING_PROVIDER_RESUME_PARITY_NO_TIMER_GO`. Every current request/grant
   boolean is false. Only a separate provider/resume-parity validator may later make a timer request
   eligible.
3. **HIGH -- immutable v2 resume needed equation-aware compatibility.** Public resume accepts only
   the exact code-pinned legacy receipt/complete/run-contract/objective/scorer/full-run identity.
   It re-derives the historical v2 economics from sealed pair rows, keeps the corrected economics
   separate and non-authoritative, and normalizes only the public control-routing view in memory.
   The real pinned run re-derived historical `1.4538672169368423x` versus corrected diagnostic
   `1.4099643443401577x`, returned no-GO routing, loaded no scorer/trainer, and left all 608 sealed
   regular files byte-identical. Unpinned, resealed, current, and incomplete paths remain on strict
   rejection/current-contract enforcement.

The closure changes no scientific verdict: direct raw-input-costate zero-order-hold K2 remains
`NOT_ADMITTED`; sparse realized saving remains `1.0x`; terminal exact-metric post-training search
remains route-local; composed admitted bulk teacher-cost reduction remains `0%`.
