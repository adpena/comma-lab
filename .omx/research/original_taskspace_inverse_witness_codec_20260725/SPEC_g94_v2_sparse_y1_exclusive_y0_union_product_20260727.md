# G94 V2 sparse-Y1 / exclusive conditional-Y0 union product

Status: **research-only typed integration implemented and source-backed fixture
verified; final-Y1 G95 refit, public runtime closure, and same-archive n600
authority remain blocked.**

Date: 2026-07-27  
Lane: `lane_g101_g94_v2_typed_union_product_20260727`  
G98 dependency: committed `cf1db441df`  
G95 dependency: committed `f2a20d607d`

This landing makes no candidate, score, pointer, promotion, or full-n600
result claim. It reuses no historical learned payload as a result.

## Smallest closed algebra

G94 V2 is the product

```text
(base PVSA1 * G98 sparse cumulative final-Y1)
    * (G88 conditional-Y0 | (G95 population basis * indexed coefficient chunks))
```

The parenthesized right factor is a closed tagged sum, not two optional
sections. The encoder accepts one `ConditionalY0OwnerV2`; no call surface can
accept both owners. The wire tag fixes owner-section cardinality, every section
has exact length and SHA-256, the member has CRC32 and exact EOF, and strict
parse-back rejects trailing bytes. Consequently a second conditional-Y0 object
cannot be accepted as inert/dead payload.

The owner-independent left factor is an actual counted serialized member:

```text
G94PRE2(base PVSA member bytes, G98 sparse cumulative Y1 operand bytes)
```

That member counts the base and G98 operand once each. Its exact SHA-256 is the
G95 V1 field historically named `g94_product_member_sha256`. This removes the
otherwise circular choice of binding G95 to an outer product that itself
contains G95. The outer product embeds the exact preconditional member once,
then exactly one owner:

- G88: one counted `PopulationConditionalOperandV1`;
- G95: one counted population-global basis and the exact indexed chunk stream.

`byte_homes` exposes every counted object and its exact size/SHA once. A G95
chunk contains only coefficients and foreign keys; it cannot duplicate the
shared basis.

Implementation:
`src/tac/witness_dsl/taskspace_g94_typed_union_product_v2.py`.

## Receiver contract

The decoder is deterministic NumPy/generic receiver code and imports no
SegNet, PoseNet, frozen scorer, target table, or evaluator.

1. Strictly reopen base PVSA and committed G98.
2. Render G98's actual final cumulative preconditional batch.
3. Apply exactly the selected conditional-Y0 owner twice.
4. Require byte-identical double decode.
5. Require selected-owner output Y1 to equal G98 final Y1 byte-for-byte.
6. Emit immutable uint8 camera pairs and exact array/member/state hashes.

For G88, base-member and semantic-P foreign keys must match the left factor.
`COPY_CONDITIONAL_Y1` and the existing role-bounded transition execute through
G88's actual receiver law.

For G95, basis parent/conditioning keys must match the exact left factor. The
owner validates that its chunks cover ordered selectors `0..599` exactly once
with no gap, overlap, basis drift, state drift, or rank drift. Streaming decode
accepts only a selector equal to one indexed chunk. The whole-state proof
streams every G98 batch, hashes the actual complete preconditional population,
checks every chunk-bound state during real decode, and preserves Y1. It returns
success only if the streamed hash equals the basis's whole-state key.

## Outer coding and parse-back

`build_g94_v2_outer_archive` builds both canonical one-member ZIP encodings:
STORE and raw DEFLATE. Both are reopened through the strict outer parser and
the strict G94 V2 parser. Equality is established from exact member bytes
(not NumPy-bearing dataclass equality), and only the smaller exact archive is
selected.

## Fixture scope

The focused source-backed fixture reopens the retained exact G85 public base,
builds one typed G98 Y1 atom, and exercises both branches. The G88 fixture
executes a real conditional-Y0 transition and proves final-G98-Y1 identity.
The G95 fixture provides one global basis plus 38 exact-coverage chunks so the
product and selected pair receiver can execute.

The G95 fixture is intentionally **not** a final-Y1 n600 fit: only its first
chunk is bound to actual fixture preconditional bytes. Its whole-state proof
must fail closed. This is evidence that the blocker is enforced, not a
population result.

Focused tests:

```text
uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g94_typed_union_product_v2.py
```

Formatting, lint, and compile:

```text
uv run ruff format --check \
  src/tac/witness_dsl/taskspace_g94_typed_union_product_v2.py \
  src/tac/witness_dsl/tests/test_taskspace_g94_typed_union_product_v2.py
uv run ruff check \
  src/tac/witness_dsl/taskspace_g94_typed_union_product_v2.py \
  src/tac/witness_dsl/tests/test_taskspace_g94_typed_union_product_v2.py
uv run python -m py_compile \
  src/tac/witness_dsl/taskspace_g94_typed_union_product_v2.py \
  src/tac/witness_dsl/tests/test_taskspace_g94_typed_union_product_v2.py
```

Tests cover:

- one counted base, G98 operand, and selected-owner object home;
- exact STORE/DEFLATE typed parse-back;
- G88 real transition, deterministic replay, and final-Y1 identity;
- G95 P-once basis plus exact indexed `0..599` chunk coverage;
- G95 selected-chunk real transition and final-Y1 identity;
- mandatory failure of an unrefit G95 whole-population state;
- wrong G95 parent rejection;
- invalid multi-owner API input rejection;
- trailing/double-owner payload rejection; and
- scorer-free imports and negative truth labels.

## Explicit blockers and next lawful step

- `G94_V2_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED`
- `G94_V2_SAME_ARCHIVE_UPSTREAM_EVALUATE_PY_N600_AUTHORITY_OWED`
- `G94_V2_G95_FINAL_Y1_WHOLE_STATE_REFIT_AND_CHUNK_REEMISSION_OWED` (G95
  branch)

The G95 branch may advance only by refitting/re-emitting its basis and every
chunk against the exact chosen final G98 whole population, then passing the
whole-state proof. Either branch still requires the same serialized archive
to decode through the public `inflate.sh`/`inflate.py` closure and score
through `upstream/evaluate.py` over all 600 samples on contest-authority
hardware. Until those receipts exist, the product remains
`research_only=true`, `candidate_claim=false`, `score_claim=false`,
`public_runtime_closed=false`.

## Stores consulted

- `PROGRAM.md`, byte-identical `CLAUDE.md` / `AGENTS.md`
- Claude project top-level memory index and current-state hooks
- Codex memory registry hooks for the Pact takeover/live-arm contract
- canonical frontier pointer, active-claim registry, lane registry, and
  subagent checkpoint registry
- last-24-hour directive scan (no matching directive files)
- committed G88, G94 V1, G95, and G98 modules/specifications/tests

Pointer delta: **none; no pointer file was modified.**
