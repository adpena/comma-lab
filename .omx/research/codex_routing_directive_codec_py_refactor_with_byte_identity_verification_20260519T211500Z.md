# Codex routing directive — codec.py 480 LOC refactor with empirical byte-identity verification protocol

**Date**: 2026-05-19T21:15:00Z (UTC)
**Authority**: Operator 2026-05-19 verbatim "we should delegate this to our pal codex with a design memo" — for Round 10 deferred item #12 (codec.py internal helper extraction)
**For consumption by**: codex CLI subagent (Pattern A detached BG invocation)

## Operator directive

> "we should delegate this to our pal codex with a design memo"

Re: Round 10 item #12 (codec.py 480 LOC internal extraction refactor). Slot P + Slot K do NOT cover this item because of SAFETY (archive byte-stability risk). This memo + codex delegation is the canonical separate-dispatch path.

## Scope

**Target file**: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py` (480 LOC)

**Goal**: Internal helper extraction to reduce per-helper LOC + improve module-boundary clarity. Each helper should be ≤200 LOC and have a top-of-function purpose docstring.

**Hard constraint**: ZERO change to archive byte-stability OR inflate output bytes. The submission is FROZEN at archive sha256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`; any code change MUST preserve the inflate path's output bytes for the contest evaluator.

## Acceptable refactor patterns

- Extract `pack_*` functions into separate helper modules or sub-functions
- Extract `unpack_*` functions
- Extract entropy-coding helpers (Huffman / arithmetic / fixed-codebook)
- Extract bit-twiddling helpers
- Add type hints to public API surfaces (typed dataclasses for structured returns)
- Add canonical per-function purpose docstrings per Round 9 canonical pattern
- Rename internal helpers for clarity (without changing public API)

## Forbidden refactor patterns

- ANY change to bit-level wire format
- ANY change to ordering of byte-emission operations
- ANY change to dependency imports that triggers different module-init code paths (e.g. adding `import numpy as np` where it wasn't imported)
- ANY optimization that changes float arithmetic ordering (per CLAUDE.md "MPS auth eval is NOISE" — per-kernel fp accumulation order matters)
- ANY change to public API signatures consumed by `inflate.py`
- ANY change to `archive.zip` bytes (the bundled archive is frozen)

## EMPIRICAL BYTE-IDENTITY VERIFICATION PROTOCOL (non-negotiable)

This protocol MUST run pre-refactor + post-refactor. The protocol is BINARY: empty diff = PROCEED + LAND. Non-empty diff = ROLLBACK + REPORT.

### Step 1: Capture baseline (pre-refactor)

```bash
SUBDIR=experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir
UTC=$(date -u +%Y%m%dT%H%M%SZ)
BASELINE=/tmp/codec_refactor_baseline_$UTC
mkdir -p "$BASELINE"

# Hash the archive (should be 6bae0201fb08... — sanity check)
sha256sum "$SUBDIR/archive.zip" | tee "$BASELINE/archive_sha.txt"

# Run inflate.sh against baseline code (pre-refactor)
mkdir -p "$BASELINE/inflate_output"
cd "$SUBDIR"
bash inflate.sh ./ "$BASELINE/inflate_output" upstream/public_test_video_names.txt

# Hash every output file (deterministic if inflate is deterministic)
find "$BASELINE/inflate_output" -type f | sort | xargs sha256sum | tee "$BASELINE/output_shas.txt"
```

If the baseline output files don't sha consistently across runs, FAIL FAST — that means inflate.py has non-determinism that must be addressed BEFORE this refactor can be safely verified. Report to operator.

### Step 2: Apply refactor

- Edit `codec.py` per the acceptable patterns above
- Commit via canonical `tools/subagent_commit_serializer.py` with POST-EDIT --expected-content-sha256
- DO NOT push yet; verify first

### Step 3: Verify byte-identity (post-refactor)

```bash
POST_REFACTOR=/tmp/codec_refactor_post_$UTC
mkdir -p "$POST_REFACTOR/inflate_output"
cd "$SUBDIR"
bash inflate.sh ./ "$POST_REFACTOR/inflate_output" upstream/public_test_video_names.txt
find "$POST_REFACTOR/inflate_output" -type f | sort | xargs sha256sum | tee "$POST_REFACTOR/output_shas.txt"

# THE GATE: empty diff = PROCEED; non-empty = ROLLBACK
diff "$BASELINE/output_shas.txt" "$POST_REFACTOR/output_shas.txt" | tee "$POST_REFACTOR/diff_verdict.txt"
```

### Step 4 (if non-empty diff): ROLLBACK

```bash
git revert HEAD --no-edit
git push  # only push the revert, not the broken refactor
```

Document the rollback reason in the verification memo. Surface to operator with the specific files that differ + speculate on the root cause (e.g. "fp arithmetic ordering changed in pack_residuals helper").

### Step 5 (if empty diff): LAND + verification commit

The refactor is byte-safe. Land it + commit the verification report:

```bash
# Add the verification proof as a separate commit
.venv/bin/python tools/subagent_commit_serializer.py \
    --message "verification: codec.py refactor byte-identity proof" \
    --files .omx/research/codex_codec_py_refactor_verification_$UTC.md \
    --expected-content-sha256 ".omx/research/codex_codec_py_refactor_verification_$UTC.md=<post-edit sha>"

# Push both commits
git push
```

## Verification report contract

Land at `.omx/research/codex_codec_py_refactor_verification_$UTC.md`:

- Operator directive verbatim
- Pre-refactor codec.py LOC count + helper-function inventory
- Post-refactor codec.py LOC count + helper-function inventory (with per-function purpose docstrings)
- Baseline `output_shas.txt` contents
- Post-refactor `output_shas.txt` contents
- `diff` verdict (empty = PASS / non-empty = ROLLBACK)
- Commit SHA of refactor (if landed) OR revert SHA (if rolled back)
- If PASS: byte-identity proof + LOC reduction summary
- If FAIL: per-file diff + root-cause hypothesis

## Codex CLI invocation contract

Codex runs autonomously via Pattern A (`nohup` + detached BG bash + `--sandbox workspace-write`). No operator-judgment loop during the refactor itself; the byte-identity verification is BINARY + empirical.

Codex's responsibilities:
1. Read this memo IN FULL before touching codec.py
2. Execute the baseline capture (Step 1)
3. Apply the refactor per acceptable patterns
4. Execute byte-identity verification (Step 3)
5. ROLLBACK if non-empty diff (Step 4)
6. LAND if empty diff (Step 5)
7. Produce the verification report
8. STOP — do NOT continue to other refactors after this one

## Discipline

- **Catalog #229 PV**: read the actual codec.py BEFORE drafting refactor changes
- **Catalog #117/#157/#174/#235**: canonical serializer with POST-EDIT --expected-content-sha256
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: the verification report is NEW; never mutate prior verification reports
- **Catalog #208 + CLAUDE.md "Public Disclosure Hygiene"**: no local paths leaked into verification memo
- **CLAUDE.md "Executing actions with care"**: codex MAY push the refactor + verification commits to origin/main since both repos are now PUBLIC + operator authorized; codex MUST NOT push if byte-identity verification FAILS (rollback only)
- **CLAUDE.md "MPS auth eval is NOISE"**: fp arithmetic ordering is sensitive; per-kernel accumulation matters

## Cross-references

- Round 10 item #12 deferral rationale in `operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md`
- Round 11 blanket approval + Round 12 re-triage (item 12 explicitly retained as DEFER for SAFETY; this codex delegation is the operator-routed separate-dispatch path)
- Submission archive sha256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` is the canonical frozen anchor

## Expected outcome

- **Likely PASS**: codec.py refactors that decompose `pack_*` / `unpack_*` helpers without changing wire format or arithmetic ordering should preserve byte-identity. Empirical verification confirms.
- **Possible FAIL**: any subtle change to fp arithmetic ordering or dependency-import side effects might shift output bytes. ROLLBACK + report root cause.

If PASS: codec.py becomes more reviewable (≤200 LOC per helper) + maintains submission byte-stability. Slot K's PR body Round 11 narrative still applies.

If FAIL: we learn something about codec.py's sensitivity to refactoring; defer permanently OR pursue with deeper investigation.

— Claude-main 2026-05-19T21:15:00Z (codex routing directive for codec.py refactor + byte-identity verification protocol per operator delegation)
