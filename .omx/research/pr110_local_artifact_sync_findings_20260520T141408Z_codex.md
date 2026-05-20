# PR110 Local Artifact Sync Findings

UTC: 2026-05-20T14:14:08Z

Scope: read-only audit of `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/` local PR #110 artifacts. No submission artifacts were edited.

Live PR reference checked:

- `commaai/comma_video_compression_challenge#110`
- Title: `hnerv_fec6_fixed_huffman_k16`
- Head: `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
- Live files: `inflate.py`, `inflate.sh`, `src/codec.py`, `src/codec_sidecar.py`, `src/frame_selector.py`, `src/model.py`
- Release asset: `fec6-frontier-submission-20260520/archive.zip`

Verification performed:

- Local `archive.zip` SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Local `archive.zip` size: `178517`
- Local ZIP member: single `x`, stored, `178417` bytes, CRC `c4a71a7a`
- GitHub release asset digest/size match local archive: `sha256:6bae0201...`, `178517`
- `adpena/comma-lab` commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` exists, but its `submission_dir` contains only `README.md`, `archive_manifest.json`, `inflate.py`, `inflate.sh`, and `src`; it does not contain `archive.zip`.

## Findings

### 1. README smoke command is broken because the pinned `comma-lab` checkout has no `archive.zip`

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`

Lines: 66-78

Evidence:

- The command clones `adpena/comma-lab` at `b392343d758aba0d3595dd18609f9ca8a8af3e1b`, changes into `.../submission_dir`, then runs `unzip -oq archive.zip -d /tmp/data`.
- GitHub API listing at that commit shows no `archive.zip` in that directory.
- Direct API lookup of `.../submission_dir/archive.zip?ref=b392343d...` returns HTTP 404.

Recommended edit:

- Keep the runtime source pinned, but add an explicit release download before `unzip`, or switch the smoke to the live PR runtime plus release asset.
- Exact replacement intent for lines 71-75:

```bash
git clone https://github.com/adpena/comma_video_compression_challenge.git /tmp/pr110 && \
  git -C /tmp/pr110 checkout ec6cc7f98c16b6ad2db8bc7cde65757bb7993004 && \
  cd /tmp/pr110/submissions/hnerv_fec6_fixed_huffman_k16 && \
  python -m venv .venv && .venv/bin/pip install --quiet torch brotli numpy && \
  mkdir -p /tmp/data /tmp/out && \
  curl -L -o /tmp/archive.zip https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip && \
  shasum -a 256 /tmp/archive.zip && \
  unzip -oq /tmp/archive.zip -d /tmp/data && echo "0.mkv" > /tmp/list.txt && \
  PACT_PYTHON_BIN=.venv/bin/python bash inflate.sh /tmp/data /tmp/out /tmp/list.txt && \
  shasum -a 256 /tmp/out/0.raw
```

### 2. Full-score verification command passes the wrong directory to `evaluate.py`

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`

Lines: 102-118 and 120-127

Evidence:

- `upstream/evaluate.py` computes rate from `(args.submission_dir / 'archive.zip')` and reads inflated tensors from `(args.submission_dir / 'inflated')`.
- The README manually inflates into `/tmp/inflate_out`, then runs `python evaluate.py --submission-dir /tmp/inflate_out`.
- That directory contains raw output files, not `archive.zip` plus `inflated/`, so the command as written will fail or score the wrong layout.

Recommended edit:

- Stage a score directory containing `archive.zip` and `inflated/`, and point `evaluate.py` at that score directory.
- Exact replacement intent for lines 102-117:

```bash
mkdir -p /tmp/archive_dir /tmp/submission_eval/inflated
cp /tmp/archive.zip /tmp/submission_eval/archive.zip
unzip -oq /tmp/archive.zip -d /tmp/archive_dir
cp -r "$RUNTIME"/inflate.sh "$RUNTIME"/inflate.py "$RUNTIME"/src /tmp/archive_dir/
echo "0.mkv" > /tmp/list.txt
bash /tmp/archive_dir/inflate.sh /tmp/archive_dir /tmp/submission_eval/inflated /tmp/list.txt

python evaluate.py \
  --submission-dir /tmp/submission_eval \
  --uncompressed-dir videos \
  --video-names-file public_test_video_names.txt \
  --device cpu \
  --report /tmp/cpu_report.txt
```

- Apply the same `--submission-dir /tmp/submission_eval` correction to the optional CUDA command at lines 121-126.

### 3. CPU thread wording overstates what `num_threads: 2` proves

Files:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`
- Live PR #110 body carries the same wording.

Lines:

- README line 7: `upstream evaluator report shows num_threads: 2` and `matches upstream ubuntu-latest GHA runner family`
- README line 84: `num_threads: 2, matching the GHA runner family`
- `report.txt` line 4: factual `num_threads: 2`

Evidence:

- `report.txt` correctly records the evaluator argument value.
- In `upstream/evaluate.py`, `--num-threads` is an evaluator dataset/worker parameter default, not proof of host CPU thread count or GitHub Actions runner equivalence.
- The auth eval host is Modal Linux x86_64, not GitHub Actions.

Recommended edit:

- Preserve the `num_threads: 2` fact, remove the runner-family inference.
- Replace line 7 note with: `Modal Linux x86_64, Ubuntu, no GPU; upstream evaluator default \`--num-threads 2\` recorded in report.txt`.
- Replace line 84 parenthetical with: `Ubuntu or comparable, no GPU; report.txt records upstream evaluator default \`--num-threads 2\``.

### 4. Runtime dependency closure wording overclaims shared-library closure

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`

Line: 130

Evidence:

- Runtime Python imports found in `inflate.py` and `src/*`: stdlib plus `torch`, `numpy`, and `brotli`.
- The sentence says: `No other Python packages or shared libraries are loaded at inflate time.`
- The Python-package part is supported by import scan; the shared-library part is too strong because `torch`, `numpy`, and `brotli` wheels load transitive native libraries.

Recommended edit:

- Replace line 130 with: `Python import closure: stdlib plus \`torch\`, \`numpy\`, and \`brotli\`. The inflate runtime does not import \`tac\`, \`comma_lab\`, scorer modules, or other project packages; native shared libraries may be loaded transitively by those wheels.`

### 5. `archive_manifest.json` still carries pre-PR D-4/D-5 and stale source-sync blocker language

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`

Lines:

- Line 11: `Remaining: D-4 ... + D-5 ...`
- Line 128: `src/codec_sidecar.py ... NOT present in commit 462f84cdd... local source-sync candidate ... 31c5fa2a9... origin/main ... e0e7d239b... required before publication`
- Line 158: `NOT_SAFE_TO_PR until the remaining compliance blockers are cleared`

Evidence:

- Live PR #110 exists and is updated at head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- Live PR #110 includes `src/codec_sidecar.py`.
- The hosted release exists and its digest/size match the local archive.
- The local README line 43 already identifies the authoritative PR runtime at `ec6cc7...`.

Recommended edit:

- Line 11 should become a current publication-status note, not a D-stage blocker note. Suggested replacement concept:
  - `submission_blocker`: `none_for_current_public_PR110_packet; pre_submission_compliance_check.py --contest-final --strict passed per reports/pr_pre_submission/compliance_report_pr101_fec6_d3_clearance_20260520T032700Z.json; live PR #110 head ec6cc7f98c16b6ad2db8bc7cde65757bb7993004; hosted archive release fec6-frontier-submission-20260520 verified with sha256 6bae0201... and size 178517.`
- Line 128 should delete the obsolete `462f84cdd` / `31c5fa2a9` / `e0e7d239b` source-sync blocker and point at the live PR head:
  - `runtime_tree_note`: `The live PR #110 runtime tree at ec6cc7f98c16b6ad2db8bc7cde65757bb7993004 contains inflate.sh, inflate.py, src/codec.py, src/codec_sidecar.py, src/frame_selector.py, and src/model.py. The runtime tree is not in archive.zip and is not rate-charged; only archive.zip bytes (178517) feed the contest rate term.`
- Line 158 should change from stale `NOT_SAFE_TO_PR` to the current status:
  - `audit_verdict`: `SAFE_FOR_PR110_CURRENT_PACKET_AFTER_D3_CLEARANCE; live PR and hosted release verified; remaining review is maintainer-side evaluation/adjudication, not a local publication blocker.`

### 6. `score_claim: false` conflicts with the artifact now being a public PR score packet

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`

Lines: 8-10, 150-155

Evidence:

- Lines 150-155 carry explicit `[contest-CPU]` and `[contest-CUDA]` scores and hardware-axis disclosure.
- The live PR body reports the same public score observations.
- `score_claim: false` is appropriate for exploratory PacketIR candidates, but stale for this final PR submission manifest unless the field is explicitly scoped to a non-promotional internal queue.

Recommended edit:

- Either:
  - change line 8 to `"score_claim": true` and add an explicit `"score_claim_axes": ["contest-CPU", "contest-CUDA T4"]`, or
  - rename/scope the field to avoid public-score ambiguity, e.g. `"internal_queue_score_claim": false`, while adding `"public_pr_score_claim": true`.
- Keep `promotion_eligible` only if it has an internal queue meaning; otherwise add a note that public PR submission has already occurred and maintainer adjudication is external.

### 7. Manifest revision log still says the current manifest made submission readiness false

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`

Line: 161

Evidence:

- The revision log says v2 added `submission readiness false` and a `source-sync commit blocker`.
- The same manifest now has `ready_for_submission: true` at line 10.
- Live PR #110 is open at `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`, and live PR #110 contains `src/codec_sidecar.py`.

Recommended edit:

- Append a v3 revision-log entry or replace the stale v2 tail with a current entry:
  - `v3: Post-PR110 sync; D-5 PR creation complete; authoritative runtime is adpena/comma_video_compression_challenge submissions/hnerv_fec6_fixed_huffman_k16 at ec6cc7f98c16b6ad2db8bc7cde65757bb7993004; hosted archive release verified; source-sync blocker cleared.`

### 8. Hosted release name calls the packet a generic frontier even though only the CPU axis is frontier

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.hosted_archive_manifest.json`

Line: 7

Evidence:

- Release name is `FEC6 frontier submission - 0.192051 [contest-CPU] / 0.226210 [contest-CUDA T4]`.
- `reports/latest.md` line 43 lists the current CPU frontier as `0.1920513169` for archive `6bae0201fb08`.
- `reports/latest.md` line 44 lists the current CUDA frontier as `0.2053300290` for archive `9cb989cef519`, not this FEC6 packet's `0.226210`.

Recommended edit:

- Replace the release name value with:
  - `FEC6 CPU-frontier submission - 0.192051 [contest-CPU]; paired observation 0.226210 [contest-CUDA T4]`

### 9. Competitive statement says "current leaderboard frontier" but the evidence is the top-CPU PR101 baseline

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.competitive_or_innovative_statement.txt`

Line: 1

Evidence:

- The statement opens with `current leaderboard frontier`.
- The actual support immediately narrows to Modal Linux x86_64 `[contest-CPU]` and the PR #101 GOLD/top-CPU baseline.
- `reports/latest.md` separates CPU and CUDA frontiers; the FEC6 packet is not the current CUDA frontier.
- README line 10 uses safer wording: `current top-CPU submission`.

Recommended edit:

- Replace `current leaderboard frontier` with:
  - `current accepted/top-CPU PR #101 baseline under the PR #108 late-submission gate`

### 10. Build-provenance "before lockdown" wording is stale/vague after PR #110 publication

File: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`

Line: 151

Evidence:

- README line 43 now identifies the authoritative runtime as live PR #110 at `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- The phrase `before lockdown` is not a concrete source state, timestamp, or publication event.

Recommended edit:

- Replace the final sentence with:
  - `The submission archive \`6bae0201...\` was rebuilt deterministically before PR #110 publication/source-sync; the rebuild produced the same SHA-256 byte-for-byte.`

## Clean Items

- Byte counts and hashes are consistent across local archive, `archive_manifest.json`, `report.txt`, hosted release manifest, and live release asset.
- `pre_submission_compliance.hosted_archive_manifest.json` matches the live release digest/size/URL; only the human-facing release name should clarify CPU-frontier vs paired-CUDA-observation.
- `pre_submission_compliance.competitive_or_innovative_statement.txt` is consistent with the score/innovation facts after narrowing the opening "leaderboard frontier" wording to the CPU-axis PR101 baseline.
- README line 43 is consistent with live PR #110 head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- `tac` and `comma-lab` cross-link wording at README lines 161-162 correctly says they are development/research surfaces, not submission runtime dependencies.
