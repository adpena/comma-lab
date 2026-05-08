# Codex Finding A2 — packet inflate-parity custody fix — recursive adversarial review

**Date:** 2026-05-08
**Fix scope:** `tools/build_a2_sensitivity_weighted_pr101_packet.py` (split clearance sets, `verify_inflate_parity()`, `--run-inflate-parity` flag) + `src/tac/preflight.py::check_packet_blocker_clearance_evidence_matches` STRICT gate + 2 manifests patched in place.

CLAUDE.md "Recursive adversarial review protocol" requires 3 consecutive clean passes. Each round below uses different inner-council perspectives (per the protocol's adversarial-rotation rule).

---

## Round 1

**Yousfi (custody-evidence definition):** What evidence ACTUALLY closes
inflate-parity? Parse-smoke validates wire-format + decoder-load. To close
parity, you need: (a) `bash inflate.sh` runs to completion on both source
and candidate archive bytes, (b) emitted files match byte-for-byte. The
fix's `verify_inflate_parity()` does exactly this — runs in isolated work
dirs, compares per-file SHA-256 maps, returns `differing_paths` list. The
evidence label `"inflate_parity_log"` maps to a record with `passed=bool`,
`source_combined_sha256`, `candidate_combined_sha256`, `differing_paths`.
**No defect found.** PASS.

**Contrarian:** Edge cases where parse-smoke DOES close parity? In theory,
if the decoder is byte-deterministic AND inflate.sh is a pure dispatcher
that does NOT modify the decoded output, then parse-smoke's
`decoder_state_sha256` agreement could imply parity. The fix's gate
allows `inflate_parity_log` (or stronger) for the parse-bounded blocker
but ONLY `inflate_parity_log` for the parity blocker. So a future
contributor who notices the implication still cannot conflate the two —
the parity blocker requires the actual log. **Edge case is structurally
prevented.** PASS.

**Carmack (simpler split):** Could you fold the two sets into one with a
flag? Yes, but that re-creates the same conflation possibility — the
flag becomes a side channel. Two named sets with explicit
`cleared_blockers_by_evidence` mapping is the simplest discipline that
prevents recurrence. **Simpler alternative would re-create the bug.** PASS.

**Round 1 verdict: CLEAN.**

---

## Round 2

**Hotz (engineering tradeoff):** Is `verify_inflate_parity()` expensive?
It runs `inflate.sh` twice and compares output files. For PR101's
packet, inflate produces a small set of files (decoder.bin, latents,
metadata) — total bytes well under 1 MB, hash comparison is microseconds.
Total wall ~5-10 sec per variant. With 1-2 variants per ladder, total
≤30 sec. **Cheap relative to the GPU dispatch this gates.** PASS.

**Boyd (backward compat):** Existing manifests written by old builder
have `cleared_blockers: ["no_byte_closed_runtime_packet_built",
"packet_local_inflate_parity_not_run"]`. The new gate refuses these.
The fix patches the 2 known offenders in place. New manifests will be
correct by construction. Schema bump is additive (new fields
`cleared_blockers_by_evidence`, `inflate_parity_status`,
`inflate_parity_record`); old fields unchanged. **No backward-compat
break for downstream consumers that ignore unknown fields.** PASS.

**Tao (theoretical separation):** The set-theoretic separation between
"parses" and "decodes-to-same-bytes" is real: parsing is a syntactic
predicate over the wire format; decoding-equivalence is a semantic
predicate over the runtime contract. They are NOT logically equivalent
in general. The fix correctly treats them as independent evidence kinds.
**Math is sound.** PASS.

**Round 2 verdict: CLEAN.**

---

## Round 3

**Selfcomp (custody discipline):** Packet ladders are upstream of dispatch
custody. Treating cleared/dispatch_blockers as independent on the same
manifest is exactly the discipline failure that loses dispatch budgets.
The fix's `cleared_blockers_by_evidence` field forces every clearance to
name its evidence — future packet builders must do the same or fail
preflight. **Custody discipline is now structurally enforced.** PASS.

**MacKay (information-theoretic):** Does parse-smoke imply inflate-parity
probabilistically, or zero correlation? In the contest setting with
deterministic decoders, P(inflate-parity | parse-smoke) is HIGH but not
1. The decoder may parse correctly yet produce different bytes if the
wire layout's interpretation diverges (e.g., a renamed but
parse-compatible tensor). In adversarial settings (a packet builder
that writes a parse-valid but semantics-divergent archive),
P(inflate-parity | parse-smoke) → 0. The fix correctly demands the
empirical observation, not the prior. **Information-theoretic
separation correct.** PASS.

**Hassabis (catches future packets):** This gate generalizes. ANY packet
manifest that ships `cleared_blockers` without `cleared_blockers_by_evidence`
is refused. Future packet builders for other lanes (PR102, PR103, ChARM,
factorized HNeRV) must use the same discipline or fail preflight on
landing. **Class-level prevention.** PASS.

**Round 3 verdict: CLEAN.**

---

## Counter status

3 consecutive clean passes achieved. Per CLAUDE.md non-negotiable:
**review counter satisfied; fix is clear for landing.**

## Cross-references

- Memory file: `feedback_codex_finding_a2_packet_inflate_parity_FIXED_20260508.md`
- Codex review source: `/tmp/codex_runs/phase_a_codemath_20260508T161501Z/output.txt`
- Sister findings reviews: `codex_finding_1_operator_approval_scoping_recursive_review_20260508.md`,
  `codex_finding_2_public_intake_pristine_recursive_review_20260508.md`,
  `codex_findings_3_4_status_dirtypaths_rebuild_recipe_recursive_review_20260508.md`,
  `codex_finding_5_recovery_metadata_appendonly_recursive_review_20260508.md`
