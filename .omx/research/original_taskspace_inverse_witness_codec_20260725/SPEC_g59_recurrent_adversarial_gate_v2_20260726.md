# G59 recurrent adversarial codec gate v2

Date: 2026-07-26  
Lane: `lane_g59_recurrent_codec_adversarial_gate_20260726`  
Authority: structural enforcement; no score or candidate claim

## Objective

Replace the current retrospective, self-attested G57 linter with a fail-closed
campaign lifecycle that makes representation substitution, stale/caller-chosen
frontiers, uncompetitive promotion, archive switching, and orphaned post-eval
signal mechanically impossible on the governed path.

The v1 code and its four G57 receipts remain useful regression evidence, but
must not be described as an enforced recurrent gate until v2 acceptance below
is satisfied.

## Adversarial findings that v2 must extinguish

1. The competitive frontier is caller supplied. V2 must reopen the current
   `.omx/state/canonical_frontier_pointer.json` through
   `tac.witness_dsl.dynamic_frontier_target`, verify freshness and file
   identity, derive the target, and bind the exact pointer SHA/source/snapshot.
2. Score math is duplicated. V2 must use the canonical contest score helper and
   evaluator operation order; no private denominator/formula copy.
3. Boundaries are independent. V2 must bind one immutable `campaign_id`,
   requested representation, predecessor receipt SHA/body SHA, archive SHA,
   decoded-raw SHA, and exact-eval receipt/report identity across the ordered
   chain. A refused predecessor cannot admit a later live stage. G57's switch
   from requested program-residual at PRE_ENCODE to direct-control later must
   fail.
4. PRE_PROMOTION can currently admit score 39.31 by changing only the axis.
   V2 promotion requires the exact competitive PRE_PUBLIC/POST_EVAL score row
   for the same archive/raw object plus contest CPU/CUDA receiver closure.
5. Evidence is self-attested. Production admission must reopen regular files
   without symlink following, check exact bytes/SHA, parse the relevant strict
   receipts, and validate source relationships. SHA-shaped strings and booleans
   alone are fixture-only and can never admit production.
6. POST_EVAL accepts arbitrary prose as an integration blocker and even accepts
   `not_killed=[]`. V2 requires nonempty narrow scope/not-killed and either
   reopened canonical integration receipts/ledger rows for every hook or a
   structured blocker artifact with exact path/SHA, owner, missing hooks, and
   next executable action. A blocker admits learning custody only, never
   candidate promotion.
7. The gate is not used by the owed program producer. V2 must be called by the
   governed `PROGRAM_RESIDUAL_LAYERED` n600 producer/orchestrator and by the
   promotion path. Existing direct-plane G52/G55 tools remain explicitly
   research-only controls: they may build and stage evidence, but they cannot
   become candidate lineage or promote through G59.

## Lifecycle

```text
CAMPAIGN_SEAL
  -> PRE_ENCODE identity/custody admission
  -> ENCODE exact archive/raw identity
  -> POST_EVAL exact coupled score + scoped learning custody
  -> PRE_PUBLIC_CLOSURE only when score is strictly below live frontier
  -> PRE_PROMOTION same-object public receiver + contest-axis proof
```

It is acceptable for a retrospective audit to inspect any boundary
independently, but the receipt must say `RETROSPECTIVE_ONLY`, must set
`candidate_admission=false`, and must never satisfy a live-launch prerequisite.

## Recurrence contract

Adversarial review is a per-cycle protocol, not a one-time code audit. Every
new encode attempt, resumed attempt whose counted inputs changed, or frontier
refresh starts a new immutable campaign chain. A live receipt is valid only for
the exact campaign id, producer/config bytes, requested representation,
canonical repository root, and predecessor object identities it sealed.

- `PRE_ENCODE` runs before any counted bytes are produced and binds the exact
  producer/config plus selected-preimage custody.
- `POST_EVAL` runs after every complete exact row, including killed and
  noncompetitive rows, and must custody either all canonical learning hooks or
  a structured fail-closed integration blocker.
- Exact-eval staging is evidence preparation, not promotion. It must occur
  before `POST_EVAL` and must not require a receipt that itself depends on the
  exact-eval row.
- `PRE_PUBLIC_CLOSURE` runs after a competitive same-object `POST_EVAL` and
  before public release/promotion closure. It reopens the already evaluated
  archive/raw objects.
- `PRE_PROMOTION` runs immediately before pointer/submission promotion and
  reopens the same archive, raw output, evaluator evidence, public receiver
  evidence, and live frontier.

No receipt may be reused for another config, producer, archive, decode, eval,
campaign cycle, repository root, or refreshed frontier. Lower-level launchers
must reject missing, retrospective, refused, learning-only, stale, cross-cycle,
or object-mismatched receipts. Refused receipts remain durable adversarial
training signal; they never authorize a later stage.

The predecessor chain must be the exact adjacent prefix of the lifecycle.
Matching identities are insufficient: a receipt that links PRE_ENCODE directly
to PRE_PROMOTION, or skips/repeats any boundary, fails closed.

## Production selected-preimage evidence

For `PROGRAM_RESIDUAL_LAYERED`, PRE_ENCODE must consume the actual G58 adapter
custody surface, not user-entered names. It must reopen and verify:

- fresh semantic archive regular file, exact bytes/SHA, source receipt, and
  counted outer byte home;
- G49 selected-preimage packet regular file, exact bytes/SHA, strict parse-back,
  factor partition, and at least one G49-verified behavior-changing factor;
- G46 batch-16 target-custody receipt and full n600/five-stage geometry;
- generic V10 implementation source identity;
- explicit absence of target/label/pose/scorer/historical payload members.

The adapter alone is not an outer archive. If outer embedding is still owed,
v2 must return an exact fail-closed blocker rather than admit the literal
program-residual campaign.

## Acceptance tests

Focused tests must prove at least:

- target `100` plus a fake pointer SHA refuses against the live/synthetic
  canonical pointer;
- canonical contest-score order matches `upstream/evaluate.py` and the shared
  helper at a strict-boundary case;
- G57 PRE_ENCODE refusal prevents all later live admissions;
- representation switch and archive/raw switch refuse;
- the G57 39.31 row cannot promote even if its axis string changes to
  `[contest-CPU]`;
- fabricated SHA-shaped strings/counts cannot admit production;
- symlink, mutated semantic/program/archive/report objects refuse;
- empty `not_killed`, prose-only blocker, or missing integration artifact
  refuses;
- a retrospective G57 audit is clearly non-promotable;
- an actual G58 fixture adapter receipt can pass the identity portion but
  remains blocked until the required outer embedding exists;
- direct research-only build/eval staging remains executable without creating
  candidate authority, while promotion still requires the exact live chain;
- skipped or repeated lifecycle stages fail even when every receipt is
  canonically serialized and all campaign fields match;
- a fresh second campaign cycle cannot reuse any first-cycle receipt, even when
  the human-readable representation name is unchanged;
- a valid program-selected-preimage receipt cannot launch a direct-plane
  config, a different producer/config hash, or a caller-selected repository
  containing a fake frontier pointer;
- exact-eval and public-authority JSON assembled from SHA-shaped strings and
  arbitrary regular files refuses until it is emitted and recursively verified
  by the strict trusted custody path.

Run at minimum:

```bash
.venv/bin/python -m pytest -q \
  src/tac/witness_control/tests/test_taskspace_codec_adversarial_gate_v2.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_preimage_operand_adapter_v1.py
.venv/bin/ruff check \
  src/tac/witness_control/taskspace_codec_adversarial_gate_v2.py \
  tools/audit_taskspace_codec_adversarial_gate_v2.py \
  src/tac/witness_control/tests/test_taskspace_codec_adversarial_gate_v2.py
```

## Do not touch

- Do not mutate the canonical frontier pointer or any candidate pointer.
- Do not run training, full-n600 encoding, public decode, or evaluator jobs.
- Do not weaken G49/G46/G55/G58 custody checks.
- Do not reuse V15/C1 historical payloads.
- Do not commit; root reviews and serializer-commits exact owned files.
