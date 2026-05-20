# PR #110 Final Evidence Pack

UTC: 2026-05-20T14:11:44Z

Scope: read-only verification of live PR #110 and its public runtime/archive
surfaces. No PR body, submission README, manifest, report, `reports/latest.md`,
or source code was edited.

## Commands Run

- `gh api repos/commaai/comma_video_compression_challenge/pulls/110`
- `gh api repos/adpena/comma_video_compression_challenge/git/trees/ec6cc7f98c16b6ad2db8bc7cde65757bb7993004?recursive=1`
- `for n in 95 98 100 101 102 103 108 110; do gh api repos/commaai/comma_video_compression_challenge/pulls/$n --jq '[.number,.user.login,.user.html_url,.head.repo.full_name,.head.ref,.head.sha] | @tsv'; done`
- `curl -L --fail --show-error --silent <release archive URL> -o archive.zip`
- `wc -c archive.zip`
- `shasum -a 256 archive.zip`
- `zipinfo -v archive.zip`
- `zipinfo -l archive.zip`
- `gh api repos/adpena/comma_video_compression_challenge/contents/<runtime file>?ref=ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
- `.venv/bin/python` static import extraction, `py_compile`, and dynamic import probe over the runtime snapshot
- `shasum -a 256` and `wc -c` over the local submission README/report/manifest/archive files
- `curl -L --silent --output /dev/null --write-out '%{http_code}'` over the two comma-lab source-map links

## Results

Live PR:

- PR #110 is open at `https://github.com/commaai/comma_video_compression_challenge/pull/110`.
- Title: `hnerv_fec6_fixed_huffman_k16`.
- Author: `adpena`.
- Head repo/ref/SHA: `adpena/comma_video_compression_challenge`, `hnerv_fec6_fixed_huffman_k16`, `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.

Runtime layout:

- Runtime files are under `submissions/hnerv_fec6_fixed_huffman_k16/`.
- No root `inflate.py`, `inflate.sh`, root `src/`, or root `archive.zip` runtime blob was detected in the public head tree.
- Public head tree contains root `README.md` plus:
  - `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py`
  - `submissions/hnerv_fec6_fixed_huffman_k16/inflate.sh`
  - `submissions/hnerv_fec6_fixed_huffman_k16/src/codec.py`
  - `submissions/hnerv_fec6_fixed_huffman_k16/src/codec_sidecar.py`
  - `submissions/hnerv_fec6_fixed_huffman_k16/src/frame_selector.py`
  - `submissions/hnerv_fec6_fixed_huffman_k16/src/model.py`

Archive:

- Release URL: `https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip`
- `wc -c`: `178517`.
- SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
- ZIP layout: one member `x`, stored uncompressed, `178417` bytes compressed and uncompressed.
- Downloaded release archive is byte-identical to local `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`.

PR author handles:

- #95 `AaronLeslie138`
- #98 `EthanYangTW`
- #100 `BradyMeighan`
- #101 `SajayR`
- #102 `EthanYangTW`
- #103 `rem2`
- #108 `andrei-minca`
- #110 `adpena`

Runtime dependency/import facts:

- Static imports include stdlib plus `torch`, `numpy`, `brotli`, and local modules `codec`, `codec_sidecar`, `frame_selector`, `model`.
- `.venv/bin/python` dynamic import probe succeeded for `torch`, `numpy`, `brotli`, `codec_sidecar`, `model`, `frame_selector`, and `codec`.
- `py_compile` succeeded for all runtime Python files in the snapshot.

Source-map links:

- `https://github.com/adpena/comma-lab/blob/b7f16a081ee381803dd5d917bdaf805453fb81f3/docs/asymptotic_floor_candidate_inventory.md` returned HTTP 200.
- `https://github.com/adpena/comma-lab/blob/b7f16a081ee381803dd5d917bdaf805453fb81f3/docs/full_stack_source_map.md` returned HTTP 200.

Blockers found: none for the requested final evidence pack. One command attempt
used bare `python`, which is unavailable in this shell; the metadata extraction
was rerun with `.venv/bin/python`.

## Files In This Pack

- `README.md`
- `archive.zip`
- `archive_metadata.json`
- `archive_sha256.txt`
- `archive_wc_c.txt`
- `archive_zipinfo_l.txt`
- `archive_zipinfo_v.txt`
- `local_readme_report_manifest_hashes.tsv`
- `local_readme_report_manifest_wc_c.txt`
- `pr110_head_tree_recursive.json`
- `pr110_head_tree_recursive.tsv`
- `pr110_link_urls.txt`
- `pr110_live.json`
- `pr110_live_body.md`
- `pr110_summary.json`
- `pr_author_handles.tsv`
- `release_archive_url.txt`
- `release_vs_local_archive_cmp_status.txt`
- `release_vs_local_archive_sha256.txt`
- `runtime_dependency_import_facts.json`
- `runtime_dynamic_import_probe.txt`
- `runtime_layout_verdict.json`
- `runtime_snapshot/README.md`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/inflate.py`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/inflate.sh`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/src/codec.py`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/src/codec_sidecar.py`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/src/frame_selector.py`
- `runtime_snapshot/submissions/hnerv_fec6_fixed_huffman_k16/src/model.py`
- `runtime_snapshot_files.txt`
- `runtime_snapshot_sha256.txt`
- `runtime_snapshot_wc_c.txt`
- `runtime_tree_relevant.tsv`
- `source_map_link_http_status.tsv`
- `source_map_link_urls.txt`
