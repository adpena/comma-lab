# Codex review — live DDM costate organ elevation 2

Date: 2026-07-23  
Lane: `lane_ddm_costate_organ_elevation2_20260723`  
Verdict: `THREE_CLEAN_PASSES_MAIN_LANDING_REVIEW_REQUIRED`  
Scope: advisory-only live DDM organ; no score, launch, dispatch, or promotion claim

## Review disposition

| Pass | Review source | Scope | Result |
|---|---|---|---|
| 1 | `ddm-elevation2-r1-postfix-authority-custody` | live-source authority, import boundary, external-atlas custody, default repo resolution | clean after two fixes below |
| 2 | `ddm-elevation2-r2-blind-math` | independently re-derived 8 pair, 40 site, and 10 block rows from raw g3/v19/v19b inputs | clean |
| 3 | `ddm-elevation2-r3-failure-determinism` | missing producer, receipt overwrite, changed-hash resume, scheduler invariants, digest old-lineage exclusion | clean |

The canonical review tracker records all entities in the new law, organ, tool, compatibility
surface, and test module as reviewed by the three independent sources above. The two modified
digest functions were also marked through all three passes.

## Bugs found and fixed

1. Moving the live implementation from `tac.witness_control` to `tac` left its default repository
   root one parent too high. The default is now `Path(__file__).resolve().parents[2]`, and a direct
   default-path test protects it.
2. A byte-freeing v19b block initially used unit byte price. It now uses
   `1 / max(abs(byte_delta), 1)`, while freeing-before-spending remains a separate scheduler law.

The final live import does not initialize `tac.witness_control`; no legacy controller can execute
on the required-producer path. Receipt run IDs must match their containing directory, the full g3
SSD atlas is byte-counted and SHA-verified, and resume fails closed if any of the seven input
hashes changes.

## Verification receipt

- focused DDM plus digest suites: `51 passed`;
- triality detector suites: `77 passed`;
- review-tracker self-test: `8 passed`;
- Ruff, compileall, and `git diff --check`: clean;
- fresh organ build is byte-equal to the durable receipt;
- receipt SHA-256:
  `23835b4026af21dcc384f88a65454af1905c9746b1084978ca13ef1260a50fd0`.

The new lane is internally coherent at L2 with four satisfied gates. The repository-wide
`lane_maturity.py validate` still reports 110 historical missing-evidence paths in other lanes;
that pre-existing global debt is preserved and was not bypassed or weakened.

One broader compatibility run exposed an unrelated existing expectation in
`test_costate_digest_ncde.py::test_section_omitted_on_short_telemetry`: the unchanged NCDE helper
returns `NO-FIRE` for four verdict rows while the test expects omission. Verdict scope:
`PRE_EXISTING_NCDE_HELPER_CONTRACT_OUTSIDE_LIVE_DDM_PATH`; this arm neither edits nor relies on
that helper when the required DDM fleet is available.

## MAIN landing review

MAIN must review the base-to-branch diff, especially the old-lineage exclusion in
`tools/costate_digest.py`, raw-input custody in `src/tac/ddm_costate_organ.py`, and the FEED-603
triality append. This arm does not merge or promote itself.
