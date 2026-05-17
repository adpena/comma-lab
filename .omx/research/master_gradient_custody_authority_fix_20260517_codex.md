# Master-Gradient Custody Authority Fix — 2026-05-17

## Context

Adversarial review found that the master-gradient extractor could write an
authority-looking `[contest-CPU]` anchor for an 8-pair macOS/local diagnostic
run, and could identify a one-member contest ZIP by the inner payload bytes
rather than the charged `archive.zip` bytes.

That creates a false-authority path:

- Autopilot joins miss the scored archive SHA or accidentally join on the inner
  payload SHA.
- Subset/advisory tensors can be interpreted as contest-axis evidence.
- Rate is treated like a byte-value derivative, even though contest rate is a
  packet byte-count term.

## Fix Landed

1. `tools/extract_master_gradient.py` now records both identities:
   - `archive_sha256` / `scored_archive_sha256` / `scored_archive_bytes` for
     the charged contest archive.
   - `gradient_subject_sha256` / `gradient_subject_bytes` /
     `gradient_byte_domain` for the differentiated payload domain.

2. Contest-axis anchors now fail closed unless the extraction uses the full
   pair count and non-advisory hardware:
   - `[contest-CPU]` requires `--device cpu`, full `n_pairs_used == n_pairs_total`,
     and non-Darwin/non-advisory hardware tags.
   - `[contest-CUDA]` requires `--device cuda`, full pair count, and
     non-advisory hardware tags.
   - 8-pair/local probes must use `[diagnostic-CPU]`, `[diagnostic-CUDA]`,
     `[macOS-CPU advisory]`, or `[MPS-PROXY]`.

3. The score-response tensor's rate column is now zero for byte-value
   sensitivities. Archive byte-count deltas must come from packet-valid
   `CandidateModificationSpec` / `grammar_aware_operator` response rows after
   rebuilding ZIP metadata and CRCs.

4. `tools/cathedral_autopilot_autonomous_loop.py` now joins master-gradient
   anchors only through structured `CandidateRow.archive_sha256`. It no longer
   scrapes arbitrary hex tokens out of notes or candidate IDs, rejects subset
   anchors as diagnostic-only, and only annotates anchor availability until a
   packet-valid modification spec exists.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_check_318_master_gradient_raw_authority.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q`
  - `177 passed`
- `.venv/bin/ruff check src/tac/master_gradient.py tools/extract_master_gradient.py tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
  - clean
- `.venv/bin/python -m py_compile src/tac/master_gradient.py tools/extract_master_gradient.py tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py`
  - clean

## Remaining Frontier Action

The useful next step is not another raw-byte gradient. It is one packet-valid
operator candidate on the fec6 archive:

1. Materialize a byte-different `archive.zip`.
2. Refresh ZIP headers and CRCs.
3. Prove `inflate.sh` consumes the changed bytes.
4. Run paired CPU/CUDA exact eval only after packet proof passes.
