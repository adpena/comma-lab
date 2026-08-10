# DDM HP3 implementation spec — HPAC section representation and ZIP frame

## Objective

Execute `.omx/research/charters/ddm_hp3_hpac_section_and_zip_frame.md` on the
byte-identical PR130 CPR1 base. Produce retained, deterministic, real candidate
archives for a bounded HPAC representation race; parse and decode the winning
n600 candidate through an owned real receiver; measure the literal ZIP
container bytes; and leave the unavailable exact scorer row in a typed queue
because this charter does not own the scorer slot.

## Authority and pins

- Base archive:
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`,
  exactly 191,052 bytes, SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- Immutable intake:
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo` at
  `e34f31bc4969042c0051ac81aa3c56884419a231`.
- Real HPAC checkpoint SHA-256:
  `0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7`.
- Intake `codec_hpac_integer.py` SHA-256:
  `70632168250cbecc40b9d6de5da5b167adeb56031368311ff936404a1ceba7e0`.
- Existing exact n600 DT1 materialization may be consumed only after its
  manifest and per-chunk hashes pass. Reuse must be named as reuse, not new
  measurement.
- All claims are `[macOS-CPU advisory, scorer-free]`, `score_claim=false`,
  unless an exact evaluator receipt exists. It will not in this arm without
  scorer-slot ownership.

## Optimal form and race

The reference is the real CPR1 `IHS1` HPAC model plus its real Range-coded
n600 token stream. Scope is full n600 for every final race row; no prefix is
admissible. The mechanisms are:

1. `control_ihs1`: exact reserialization control. It must reproduce the
   canonical HPAC raw bytes and canonical Range bytes before any candidate is
   accepted.
2. `factor_frame_delta`: an exact structural factorization of the 600x8 int8
   frame embedding as first row plus modulo-256 temporal residuals. The owned
   receiver must invert it to byte-identical `IHS1`; token bytes stay the
   canonical Range payload.
3. `requant_frame_embed_step2`: round the deployed int8 frame embedding to the
   nearest even integer with explicit deterministic tie/clamp behavior,
   serialize a real candidate model, materialize its exact causal n600 logit
   codes resumably, Range-encode the real n600 target labels, and retain the
   stream. Its measured sensitivity is the net archive delta: model saving
   plus token-stream change.
4. `prune_weight_abs1`: set deployed compressible-module weights with
   `abs(w)<=1` to zero, recompute the minimum legal per-row bit depths,
   serialize, materialize, encode, and retain exactly as above. It is called
   zero-cost only if the final decoded semantic token tensor is exactly equal
   on all 117,964,800 tokens; otherwise it is a failed formulation, not
   zero-cost.

If a candidate is not smaller than the 191,052-byte exact base, it does not
survive. The smallest survivor is rebuilt twice byte-identically, decoded at
n600 through the owned causal receiver, and then inflated through the owned
`inflate.sh` CPU rail to a retained raw file. Semantic and carrier sections
must remain byte-identical to CPR1. The full decoded token SHA-256 must equal
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.

## Container measurement

Parse the base ZIP records, not estimates. Report and retain:

- local header, local filename, member data, central directory header,
  central filename, EOCD, comment, and extra-field byte counts;
- a deterministic stored control archive and repeat;
- a real `ZIP_DEFLATED` candidate archive over the same member bytes;
- the standards lower bound for a one-member extracted path `p`:
  `30+1+46+1+22 = 100` bytes. Do not claim a saving unless `unzip -t` and
  extraction to a file named `p` both succeed.

## Files and ownership

Write only new HP3-owned paths:

- `experiments/ddm_hp3_hpac_section_and_zip_frame.py`
- `src/tac/pr130_runtime/ddm_hp3_runtime/**`
- `src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py`
- `.omx/research/ddm_hp3_20260810/**`
- SSD bulk under `/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/**`

Do not edit `upstream/`, tokens-arm files, semantic-arm files, the dirty
`fx1_runtime_tree`, staged index content, shared state, DAG, DSL, or equations.
The runtime overlay may import/copy the committed owned FX1 receiver at run
time, but must record every borrowed source path, commit, and SHA-256.

## P0 retention and resumability

- Storage preflight must require at least 15 GiB free at the HP3 SSD root.
- Every materialized raw model, compressed model, logit-code chunk, token
  stream, member payload, archive, repeat archive, container candidate, and
  inflated raw file must remain on SSD with path, bytes, and SHA-256 in a
  machine-readable manifest.
- Candidate logit materialization checkpoints every 24 frames (or finer), is
  atomic, and resumes only from a named `--resume-from` state whose provenance
  matches exactly. Earlier stage files are immutable and never overwritten.
- A candidate is complete only when all 600 frames, 117,964,800 labels, and
  all payload hashes are present. Interrupted runs are resumable from the last
  complete chunk.
- Run `tac.payload_retention_gate` in strict mode on every new Python file.

## Deliverables

- `.omx/research/ddm_hp3_20260810/FINAL_REPORT.md` with `## RECALL EVIDENCE`,
  decomposition, full race table, container table, boundaries, dispositions,
  `LIVE-HYPOTHESES`, and `DEAD-ENDS`.
- `.omx/research/ddm_hp3_20260810/FINAL_RECEIPT.json` with every denominator,
  command, path, byte count, SHA-256, axis, provenance pin, candidate verdict,
  receiver result, and scorer queue boundary.
- SSD retained payload tree and its manifest.

## Acceptance commands

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py
.venv/bin/python -m ruff check \
  experiments/ddm_hp3_hpac_section_and_zip_frame.py \
  src/tac/pr130_runtime/ddm_hp3_runtime \
  src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py
.venv/bin/python -m ruff format --check \
  experiments/ddm_hp3_hpac_section_and_zip_frame.py \
  src/tac/pr130_runtime/ddm_hp3_runtime \
  src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py
.venv/bin/python - <<'PY'
from pathlib import Path
from tac.payload_retention_gate import check_no_measure_and_discard_payload
check_no_measure_and_discard_payload(
    repo_root=Path('.'),
    strict=True,
    roots=(
        'experiments/ddm_hp3_hpac_section_and_zip_frame.py',
        'src/tac/pr130_runtime/ddm_hp3_runtime',
        'src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py',
    ),
)
PY
git diff --check -- \
  experiments/ddm_hp3_hpac_section_and_zip_frame.py \
  src/tac/pr130_runtime/ddm_hp3_runtime \
  src/tac/tests/test_ddm_hp3_hpac_section_and_zip_frame.py \
  .omx/research/ddm_hp3_20260810
```

After final edits, mark every new `.py` twice with
`tools/review_tracker.py mark-file ... --status reviewed`. Commit only the
HP3 paths through `tools/subagent_commit_serializer.py`, repeating
post-edit `--expected-content-sha256` for every file, message tags
`[no-triality] [p0-ledger-ok]`, and `--no-co-author`; no attribution trailer.

## Not authorized

- No scorer job while the charter lacks the scorer slot.
- No Modal, CUDA remote, or network dispatch.
- No mutation of the immutable intake or upstream.
- No deletion of retained payloads.
- No use of coder/gauge families already closed by #996 and `113b52fdb1`.
