# PR #110 body draft v4 — Yousfi template-faithful response to 2026-05-25T21:19:01Z comment

## Status

- **DRAFT-ONLY** — operator copy-paste pending. Not yet pushed to live PR.
- **Trigger**: @YassineYousfi comment on PR #110 at `2026-05-25T21:19:01Z`: *"can you update the pr with the new template, including an easy to understand response to: [competitive or innovative section with HTML comment markers]"*.
- **Supersedes** (for the LIVE PR body, not historical drafts):
  - Live PR body at PR-edit-timestamp `2026-05-20T14:46:27Z` (pre-Yousfi comment)
  - Sister draft `.omx/research/pr_body_corrected_draft_v3_20260520T031530Z.md` (uses different INNOVATION 1/2 structure; not template-faithful per Yousfi's `pull_request_template.md` ask)
- **Canonical upstream template source**: `gh api repos/commaai/comma_video_compression_challenge/contents/.github/pull_request_template.md` fetched 2026-05-25.
- **Operator sole-author** per `user_pr_attribution.md` + `forbidden_claude_attribution_in_public_pr_surfaces.md`: zero `Claude` / `Anthropic` / `Co-Authored` / `claude.com` / `anthropic.com` tokens.

## Discipline applied (Catalog refs)

- **Catalog #229 PV** — read in full before drafting: live PR body, upstream `pull_request_template.md`, Yousfi comment, v3 draft, `pr110_release_body_after_zip_style_pass_*` codex artifact, `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`.
- **Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE** — this is a NEW file; v3 and prior drafts preserved unchanged.
- **Catalog #287** placeholder-rationale rejection awareness; no `<rationale>` / `<reason>` literals.
- **Catalog #208** local-absolute-path discipline; no `/Users/` paths in PR body content (only in this metadata header per `DOCS_LOCAL_PATH_OK:provenance_reference_to_canonical_local_research_directory_per_catalog_110_append_only_discipline`).
- **CLAUDE.md "Forbidden /tmp paths"** — file lands in `.omx/research/`, NOT `/tmp/`.

## Diff vs live PR body (concise)

| Section | Live body | v4 |
|---|---|---|
| `# submission name:` | `hnerv_fec6_fixed_huffman_k16` | same + restored HTML scaffold |
| `# upload zipped archive.zip` | link + SHA-256 + size + ZIP-member counts + Runtime tree link | link only + restored HTML scaffold (template asks for link, not metadata) |
| `# report.txt` | code block + "Full-precision" line + "Paired T4" line | code block only + restored HTML scaffold (paired CUDA moved to `# additional comments`) |
| `# does your submission require gpu` | `no` | `no` + restored HTML scaffold |
| `# did you include the compression script?` | `yes.` + long explanation | tighter `yes.` + pointer to `encoder/README.md` |
| `# is this submission competitive or innovative?` | `Competitive: <numbers>` / `Innovative: <mechanism>` | restored HTML markers + explicit `Competitive: yes.` lead with rate-vs-distortion arithmetic (+259 B / 5.6× payoff) + explicit `Innovative: yes.` lead with training-free + adaptive-quantization framing |
| `# additional comments` | FEC6 + lineage + Research notes | same content tightened; paired CUDA `0.226210` moved here from `# report.txt`; tac + comma-lab links preserved |

## Forbidden-token audit (pre-save)

- `grep -ciE "claude\|anthropic\|co-authored\|claude.com\|anthropic.com"` against the body block below: **0**
- HTML comment markers per upstream template: **7 / 7 sections covered**
- Operator first-person voice: confirmed
- `@`-mention attribution: PR #95 @AaronLeslie138, PR #98 @EthanYangTW, PR #100 @BradyMeighan, PR #101 @SajayR, PR #102 @EthanYangTW, PR #103 @rem2

## v4 PR body (copy-paste to `gh pr edit 110 --repo commaai/comma_video_compression_challenge --body-file -`)

```markdown
# submission name:
<!-- the directory name of your submission -->
<!-- please make sure it matches exactly -->
hnerv_fec6_fixed_huffman_k16

# upload zipped `archive.zip`
<!-- do not check it in the code (it's already ignored in .gitignore) -->
<!-- you can use the upload file feature (drag and drop), make sure curl -L works -->
[archive.zip](https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip)

# report.txt
<!-- do not check it in the code (it's already ignored in .gitignore) -->
<!-- copy the report.txt content here -->
```
=== Evaluation config ===
  batch_size: 16
  device: cpu
  num_threads: 2
  prefetch_queue_depth: 4
  report: /root/modal_auth_eval_cpu_work/eval_work/report.txt
  seed: 1234
  submission_dir: /root/modal_auth_eval_cpu_work/eval_work
  uncompressed_dir: /workspace/pact/upstream/videos
  video_names_file: /workspace/pact/upstream/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00002943
  Average SegNet Distortion: 0.00056029
  Submission file size: 178,517 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00475469
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.19
=== Archive identity ===
  Archive SHA-256: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
  Archive size bytes: 178517
```

# does your submission require gpu for evaluation (inflation)?
<!-- yes|no -->
<!-- this only applies to inflation, you can use gpu for compression but not require gpu for evaluation -->
no

# did you include the compression script? and want it to be merged?
<!-- yes|no -->
yes. `compress.sh` wraps the encoder bundled at `submissions/hnerv_fec6_fixed_huffman_k16/encoder/`. The encoder reproduces this submission's `archive.zip` from PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)'s `archive.zip` plus a precomputed offline scorer-sweep over 31 candidate per-frame transforms. PR #101's archive is not redistributed — fetch from its release. Full recipe in `encoder/README.md`.

# is this submission competitive or innovative? explain why
<!-- competitive: better than top #1 submission -->
<!-- innovative: it has a novel idea that is not on the leaderboard yet, might not be competitive, but has potential -->

**Competitive: yes.** `0.192051 [contest-CPU]` beats current top merged submission PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) `hnerv_ft_microcodec` (`0.192840`) by `-0.000789` on the leaderboard axis. The mechanism is rate-vs-distortion: the FEC6 selector adds +259 bytes (rate penalty `+25 × 259 / 37,545,489 = +0.000172`) but saves `-0.000961` across the distortion terms — a **5.6× payoff** on the bytes spent. The savings come from picking the best-of-31 reconstruction modes per frame against the actual upstream scorer, where PR #101 uses one fixed pipeline for all frames.

**Innovative: yes.** Scorer-aware offline mode selection over a frozen substrate is not on the leaderboard. Every other medal-class PR (#95 / #100 / #101 / #102 / #103) gains from new training. This submission does **no new training** — PR #101's weights are reused byte-identically; the bolt-on is a 16-symbol fixed-Huffman per-frame mode index against an encoder-known/decoder-known codebook (the codebook is not transmitted). The offline sweep (`encoder/frame_exploit_segnet_posenet_sweep.py`) is analogous to adaptive quantization but one abstraction level up — per-frame transform selection rather than per-coefficient quantization. Potential: the K=16 alphabet is a tunable knob (larger K → more flexibility at higher rate cost), and the sweep methodology generalizes to any frozen-substrate PR on the leaderboard.

# additional comments
<!-- anything else you want to share -->
<!-- describe your solution -->
FEC6 selector bolt-on around the public HNeRV lineage. Decoder (`src/model.py`) byte-identical to PR [#95](https://github.com/commaai/comma_video_compression_challenge/pull/95) (@AaronLeslie138). Byte substrate from PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) (@SajayR): compact decoder, latent-payload parsing, Brotli source streams, canonical Huffman on the latent sidecar (all reused unchanged). Member `x` grammar: `FP11 | u32 source_len | source_pr101_payload | u16 selector_len | selector_payload` — PR #101's payload reads verbatim from `source_pr101_payload`; the FEC6 selector is byte-appended outside the Brotli envelope and is not further compressed. Lineage: PR [#98](https://github.com/commaai/comma_video_compression_challenge/pull/98) (@EthanYangTW), PR [#100](https://github.com/commaai/comma_video_compression_challenge/pull/100) (@BradyMeighan), PR [#102](https://github.com/commaai/comma_video_compression_challenge/pull/102) (@EthanYangTW), PR [#103](https://github.com/commaai/comma_video_compression_challenge/pull/103) (@rem2). Sweep compute ran offline on Modal A100 (not in the inflate path). Paired Modal Tesla T4 against the same `archive.zip` bytes returned `0.226210 [contest-CUDA]`; CPU and CUDA are reported separately per dual-axis discipline. Research notes at [`adpena/comma-lab`](https://github.com/adpena/comma-lab); tooling at [`adpena/tac`](https://github.com/adpena/tac).
```
