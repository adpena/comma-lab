# Master-Gradient Raw Authority Guard - 2026-05-17

## Context

The parent-scope `.omx` Markdown scan was refreshed outside `.omx/research`.
The current authority surfaces are:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_lane_dispatch_claims.md`

`.omx/notepad.md` is explicitly historical April Track-B/AV1 memory and not
current L5, TT5L, FEC6, Rule #6, PR101, or submission authority.

The live May 17 control plane says the raw master-gradient route is blocked:
archive-byte / bit finite differences over ZIP plus entropy-coded packet bytes
are not valid score derivatives. Valid routing is an operator-response matrix
over packet-valid mutation rows, with packet rebuild, ZIP metadata/CRC refresh,
inflate proof, byte-consumption proof, and axis-labelled result review.

## Finding

Untracked partner WIP currently still contains the raw-authority shape:

- `src/tac/master_gradient.py`
- `tools/extract_master_gradient.py`
- dirty hook additions in `tools/cathedral_autopilot_autonomous_loop.py`

The disallowed surface includes `(N_archive_bytes, 3)` / `(N_bytes, 3)`
gradient sidecars, `finite_difference_bit_flip` method naming,
`gradient_array_path`, raw `{byte_idx: delta}` projection APIs, and
anchor-presence reranking. Those are useful as a warning signal but must not
land as dispatch, rank, kill, or score authority.

## Patch

Added Catalog #318 in `src/tac/preflight.py`:

- `check_master_gradient_raw_byte_authority_not_landed(...)`
- strict call wired into `preflight_all(...)`
- first-80-line waiver:
  `# MASTER_GRADIENT_RAW_AUTHORITY_OK:<rationale>`

The check is deliberately narrow. It blocks authority-bearing raw
master-gradient source surfaces while allowing the already-landed valid route:

- `tac.master_gradient_feasibility`
- `tac.master_gradient_operator_plan`
- grammar-aware `CandidateModificationSpec` / operator response rows

## Verification

Focused tests:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_check_318_master_gradient_raw_authority.py \
  src/tac/tests/test_master_gradient_feasibility.py \
  src/tac/tests/test_master_gradient_operator_plan.py
```

Result: `17 passed`.

Hook-relevant undefined-name lint:

```text
.venv/bin/ruff check --select F821 \
  src/tac/preflight.py \
  src/tac/tests/test_check_318_master_gradient_raw_authority.py
```

Result: `All checks passed` with pre-existing invalid-`noqa` warnings in
`src/tac/preflight.py`.

Broad ruff remains non-actionable for this patch because `src/tac/preflight.py`
has many pre-existing style violations outside the touched Catalog #318 surface.

## Next Route

Do not stage or promote raw-byte master-gradient WIP as authority. The next
score-relevant object is a typed grammar/operator candidate, ideally FEC6 or
Rule #6 aligned:

1. parse the exact anchor packet;
2. emit `CandidateModificationSpec` / `grammar_aware_operator` rows;
3. rebuild packet bytes with updated metadata/CRC;
4. prove inflate and byte consumption;
5. only then run paired CPU/CUDA exact review.
