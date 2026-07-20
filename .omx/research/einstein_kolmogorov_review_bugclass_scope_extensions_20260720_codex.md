# Einstein–Kolmogorov recursive-review bug-class scope extensions

Date: 2026-07-20 UTC  
Lane: `lane_einstein_kolmogorov_crux_20260719`  
Scope: apparatus hardening only; no score promotion; no launch; `$0` local  
Pointer: `0.1910828242 [contest-CPU Linux x86_64]`, unchanged

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/meta_bug_class_catalog.md`; the delegated authority prompt;
latest lane/inbox state; commits `d30b243c80` and `03ae758880`; the immutable V3 final and
precleanup receipts; `src/tac/canonical_equations/{equation,registry,einstein_kolmogorov_crux_20260719}.py`;
`src/tac/preflight.py`; `src/tac/v9_provenance_gates.py`;
`tools/measure_v10_free_predictor_floor.py`; and the three fresh-context review verdicts on
`03ae758880`.

## Review reset

`03ae758880` is **INVALIDATED** and contributes zero clean passes. Two independent reviewers
reproduced a cwd-dependent authority upgrade: from a non-repo cwd, semantic validation used
repo-root-resolved producer bytes while the provenance builder received the original relative
path, emitted an all-zero SHA, and the anchor retained `VERIFIED_VIA_EMPIRICAL_ANCHOR`. The
contract reviewer additionally reproduced byte-identical alias laundering and found that three
earlier review fixes lacked the mandatory strict bug-class refusal surface. Clean counter: `0/3`.

The scientific arithmetic and immutable V3 receipts were independently re-derived clean. The
primary blocker remains `NO_COMPLETE_N600_ARCHIVE_WITHIN_TOTAL_SCORE_BYTE_CAP`; no result here
moves the pointer.

## Immediate fixes

1. The Einstein–Kolmogorov builder now constructs each producer path independently of cwd, requires
   its lexical absolute path to equal the exact repo-canonical constant *before* resolving links,
   rechecks resolved identity, passes only that exact path to the provenance builder, restores the
   stable canonical source label only after hashing, and compares the resulting provenance SHA
   against the independent frozen hash. Byte-identical copies, symlinks, hardlinks, and alternate
   lexical paths are not canonical producers.
2. The canonical registry reconstruction preserves `empirical_verification_status`, `noise_floor`,
   and `noise_floor_provenance`. `audit_empirical_anchor_roundtrip_fidelity` compares complete
   anchor JSON before/after reconstruction so any future additive field fails closed without a
   hardcoded allowlist.
3. Historical manifest-less V3 cleanup compatibility remains licensed only for the exact final and
   precleanup path/bytes/SHA pair, exact predecessor mapping, exact loaded final payload hash,
   frozen cleanup tuple, and absent output root. Tuple-shaped/lookalike receipts refuse.

## Mandatory second landings under the Catalog #299 quota

The next catalog number is `407`, while the binding quota forbids a new strict gate after `#400`
without retirement/replacement or operator waiver. Therefore no new number was claimed. Following
the established Catalog #287/#323 scope-extension precedent, the three findings extend existing
strict umbrellas:

- **Catalog #154:** manifest-absence branches across `tools/`, `scripts/`, `experiments/`, and
  `src/tac/` must either refuse or route through an exact final+precleanup identity validator. The
  scanner preserves membership polarity, accepts a direct fail-closed raise without demanding a
  compatibility validator, and requires distinct validation calls bound to the final and
  predecessor mappings.
- **Catalog #344, enforced again through strict-clean #351 custody:** every `EmpiricalAnchor` field
  must be reconstructed from its same-name payload field; additive/defaulted fields must use
  legacy-compatible `.get`; every durable event must be complete-JSON roundtrip exact. The #351
  bridge prevents the unrelated historical #344 memo-formalization backlog from weakening this new
  authority-schema refusal surface.
- **Catalog #351:** parameterized verified producer builders must bind every path argument to an
  exact canonical constant and must compare constructed provenance SHA values to independent
  `SOURCE_*_SHA256` constants.

Fresh live sub-audits: registry structure `0`; 760-event registry roundtrip `0`; canonical producer
identity `0`; manifest-less cleanup identity `0`; combined strict #351 custody extension `0`.

## Verification and triality

Dedicated adversarial scope-extension suites: **72 passed** (`#154`: 25; `#344`: 23; `#351`: 24).
They cover positive and negative signatures, branch polarity, direct fail-closed refusal,
one-to-one preserved-file binding, malformed/placeholder waivers, legacy absence, unknown additive
keys, multi-event aggregation, cwd changes, byte-identical copy/symlink/hardlink aliases,
resolve-only alias laundering, missing SHA rechecks, partial historical tuples, unrouted
manifest-less branches, strict raises, and live-zero regressions. A broader relevant regression run
reported **357 passed** and three unrelated pre-existing live-tree drift failures: one Catalog #207
unguarded-rmtree row and the two known Catalog #344 memo-backlog assertions at 147 rows. Ruff check
is clean for all seven changed/new Python files; Ruff format is clean for the six files other than
the pre-existing non-format-normalized `preflight.py`. A trial full-file format of that hot module
was fully backed out at hunk granularity so the landing carries no repository-wide style churn.

- **DSL:** no new witness lever; apparatus-only change.
- **DAG:** `FEED-EINSTEIN-KOLMOGOROV-20260719` gains a review-reset and bug-class-closure edge.
- **Equation:** `einstein_kolmogorov_crux_action_rate_contract_v1` retains the same numerical law and
  receipts; only producer/authority custody is strengthened.
- **Pointer delta:** exactly `0`.

## Landing boundary

This scope-extension unit requires serializer commit, then a fresh immutable three-clean-pass seal.
Any finding resets the counter. Even after a clean seal, the branch requires independent **MAIN
landing review**; no result is authorized for frontier-pointer promotion by this memo.
