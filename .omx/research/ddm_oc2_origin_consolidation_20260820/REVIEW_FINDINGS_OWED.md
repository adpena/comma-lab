# ddm_oc2 — review findings owed to MAIN (6 real bugs, all PRE-EXISTING on main)

`date_utc: 2026-08-20` · `owner: ddm_oc2` · reviewer: dedicated code-review pass on the three files
the review gate held back during the sh1 merge.

## Why this file exists

Merging `ddm/sh1_integration_20260727` left three files blocked by the review gate as UNREVIEWED. I
ran a real review rather than marking them reviewed. It found **1 critical and 5 important
correctness bugs**.

**The load-bearing fact: every one of these bugs is already on `main`.** The sh1 deltas are additive
and small — 329 / 36 / 9+5 lines — and none of them introduces a finding below. Verified directly:

| Finding site | Present on `main` before this consolidation? |
|---|---|
| `"partial base coder-frame resume set is unsafe"` (critical #1) | **Yes** — `main:tools/materialize_ddm_pf3_finite_prices.py:642` |
| `assert proof["all_samples_refused_or_changed_decode"] is True` (#5) | **Yes** — `main:src/tac/optimization/tests/test_direct_description_carrier_compose.py:855` |

So landing the sh1 deltas does **not** make main worse. These are debts against code that has been
shipping. They are recorded here as owed work, and the three files are marked `needs_fix` in the
review tracker — **not** `reviewed`. No arm may cite them as review-clean.

## Critical

### C1 — `tools/materialize_ddm_pf3_finite_prices.py:685-688` — a missing optional coder crashes untyped and permanently poisons the resume set

`_load_or_race_base_coders` builds `artifacts` over the full 7-tuple `codecs`, but
`race_same_receiver_object` returns frames **only for codecs that succeeded**.
`src/tac/optimization/ddm_ms7_receiver_edges.py:538-598` emits `"available": False` rows (no frame)
for `E4_BROTLI_Q11`, `CONSTRICTION_ORDER1_CONTEXT_ANS` and `ZSTD19_TRAINED_DICTIONARY` when the
optional dependency is absent, and never produces a `G4_*` frame at all.

`_artifact` (line 171) calls `path.resolve(strict=True)`, so a missing frame raises
`FileNotFoundError`, not `PF3MeasurementError` — **after the whole base race has been paid for**.
`_publish` (line 682) has already written the partial set, so every later invocation trips line 641
(`any(existing.values()) and not all(existing.values())`) and raises "partial base coder-frame
resume set is unsafe" **forever**. The bulk directory cannot recover without manual deletion. The
resumed branch compounds it by hardcoding `"available": True, "parseback_exact": True` for all seven
(lines 650-658), misreporting a codec that never ran.

**Fix:** derive `paths`, `existing` and `artifacts` from the codecs actually present in `frames`
(intersected with the race's `available` rows); record unavailable codecs as typed null rows instead
of expecting a file on disk.

## Important

### I2 — `src/tac/optimization/ddm_runtime_receiver.py:2010` — E5A receipt `coder` decoded from the wrong frame layout

`BLOB_HEADER` is `">5sBBBBQQ32s"`, so index `[2]` is byte offset 6. On an E5A packet that member is
not a `DDE1B` frame — it is the LA1 bundle framed by `E5A_BUNDLE_PREFIX = ">5sBBB"` (line 56), where
byte 6 is `component_count`. The receipt reports `brotli_q11` iff the bundle happens to carry one
component and `lzma1_raw_d1m_lc3_lp0_pb2` otherwise. Both are meaningless: E5A's real coders are the
per-component `E5A_CODECS`. Secondary: `_validate_ws1_manifest` guarantees only
`len(framed) >= E5A_BUNDLE_PREFIX.size` (8 bytes) on that path, so a short bundle raises an untyped
`struct.error` instead of `ReceiverError`, breaking the fail-closed contract.

**Fix:** branch on `manifest["schema"]`. For E5A report the per-component codec census already
available from `_reconstruct_e5a_la1_bundle`'s `consumed` rows; touch `BLOB_HEADER` only when the
member really is a DDE1B frame (guard on `len(framed) >= BLOB_HEADER.size` **and** the magic).

### I3 — `src/tac/optimization/ddm_runtime_receiver.py:2201-2210` — CC3's final-output identity check is self-referential and cannot fail

`_preserved_stage_sequence_identity` digests the concatenated stage files, and that digest is passed
back in as `expected_sha256` to `_assemble_final`, which concatenates the *same* files and compares.
On a fresh assembly the "final raw identity mismatch" guard at line 1611 is dead code. Every other
inflate route (`inflate:2486`, `_inflate_ws1:2006`, `_inflate_cb1:1790`) binds final bytes to the
counted `manifest["output"]["sha256"]`; CC3 has no such binding, so a decoder regression or a
nondeterministic render still yields a receipt claiming verified custody. Only the byte *count* is
independently checked (line 2202).

**Fix:** carry the expected raw-output SHA-256 in the counted CC3 packet (natural home
`manifest/pc1.json`, restored via `restore_extracted_composition`) and compare against that; keep the
stage digest as a cheap pre-check only.

### I4 — `tools/materialize_ddm_pf3_finite_prices.py:1015-1017` — the PF2 assignment table is re-read without its SHA binding

`_load_inventory` reads this file through `_read_bound_json(table_path, str(table_ref["file_sha256"]), ...)`
(lines 295-296), which enforces the SHA and refuses symlinks. `materialize` then re-reads the same
path with a bare `json.loads(...read_bytes())`, dropping both checks. The resulting `probe_rows` feed
`receipt["inventory"]["source_status_counts"]` (line 1206) and the per-bucket
`assigned_coordinate_probe_status_counts` / `blocker` fields (lines 1041-1057), which are then
`_publish`ed as immutable artifacts. Unverified bytes enter published custody records in a tool whose
whole contract is SHA-bound replay.

**Fix:** return the already-bound `table` from `_load_inventory` and reuse it, or repeat
`_read_bound_json` with the same expected SHA.

### I5 — `test_direct_description_carrier_compose.py:890-891` — the fail-closed proof assertion passes on zero samples ⚠ NO-FAKE class 2

`proof["all_samples_refused_or_changed_decode"]` is computed as `refused + changed == len(positions)`
(`direct_description_carrier_compose.py:3298`), which is `0 == 0` → `True` when `positions` is empty.
`positions` is built by walking ZIP local headers and skipping `info.file_size == 0`
(lines 3273-3277); an offset or member-walk regression there silently yields zero mutations and the
test still passes. It is the only assertion in the test, so it **verifies a returned constant rather
than behaviour** — the canonical NO-FAKE forbidden class 2.

**Fix:** also assert `proof["sampled_member_payload_homes"] == len(members)` (≥ 2 for this archive),
`proof["refused"] + proof["changed_decode"] == proof["sampled_member_payload_homes"]`, and
`proof["unique_home_coverage_bytes"] == len(archive)`.

### I6 — `test_direct_description_carrier_compose.py:747-752, 775` — `template_camera_masks` is never asserted to select any pixel ⚠ NO-FAKE class 2

The loop paints the no-op template colour through `lane_mask` into a copy and asserts
`np.array_equal(v14_camera, patched)` — trivially true if every mask is all-`False`. The only other
assertion is a shape check at line 775. **An implementation replaced by
`np.zeros((n, 874, 1164), dtype=bool)` passes the entire test.**

**Fix:** add `assert lane_mask.any()` (and the same for `bank.templates[1]`), and strengthen by
painting a *different* colour through the mask and asserting the camera does change.

## Verified clean (no high-confidence bug)

`_read_exact_ws1_members`, `_read_exact_cc3_members`, `_reconstruct_e5a_la1_bundle`,
`_reconstruct_ws1_state`, `_install_ws1_source_bundle`, `_preserved_stage_sequence_identity`,
`_validate_ws1_manifest`. The parsed-byte arithmetic was checked specifically:
`_reconstruct_e5a_la1_bundle` bounds-checks the entry header and payload, rejects `frame_bytes <= 0`,
enforces strictly-increasing in-range `component_id` and in-range `codec_id`, and refuses trailing
bytes — no out-of-bounds slice is reachable. `_reconstruct_ws1_state` enforces `offset == cursor`,
`bytes > 0`, `stop <= len(archive)` and exact full coverage. `_parse_blob` pins the exact total
length before slicing.

Sub-threshold, recorded but not counted: `ddm_runtime_receiver.py:2264-2266` swallows `RuntimeError`
from `set_num_interop_threads(1)` while the receipt unconditionally claims a fixed thread contract.
Unreachable from the standalone `inflate.py` entry point.

## Disposition

I5 and I6 are the **tests-verify-constants-not-behaviour** class, which CLAUDE.md names as forbidden
class 2 of the NO-FAKE supreme rule. They are on main today. They deserve to be fixed before any
result that leans on `prove_carrier_archive_fail_closed` or `template_camera_masks` is cited as
evidence.
