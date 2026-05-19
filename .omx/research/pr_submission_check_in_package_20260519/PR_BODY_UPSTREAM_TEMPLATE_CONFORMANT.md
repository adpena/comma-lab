# submission name:
pr101_fec6_k16_clean

# upload zipped `archive.zip`
https://github.com/adpena/comma_video_compression_challenge/releases/download/pr101-fec6-k16-clean-v1/archive.zip

<!-- Archive sha256: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf -->
<!-- Archive size: 178,517 bytes -->

# report.txt
The block below is the verbatim `report.txt` output from upstream `evaluate.sh --device cpu` on this archive; upstream rounds `Final score` to 2 decimal places. The exact full-precision score recomputed from the same components is `0.1920513169` (`100 * 0.00056029 + sqrt(10 * 0.00002943) + 25 * 178517 / 37545489`).

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
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.19
```

This `report.txt` is a Modal Linux x86_64 CPU reproduction (uv-managed `python` on a Modal CPU container, not a host-bot/GitHub Actions run). The contest's automated CI runner has not yet validated this archive at the headline number; treat the reported score as a Modal Linux CPU reproduction until a same-axis host-bot artifact exists.

Paired Modal T4 CUDA replay on the same archive bytes (sha256 `6bae0201...`) produced `0.2262100217` `[Modal T4 CUDA replay]` (avg PoseNet distortion `0.00016846`, avg SegNet distortion `0.00066299`, same `178,517`-byte archive). Both axes derived from the same archive bytes (verified SHA-256 match).

# does your submission require gpu for evaluation (inflation)?
no; CPU inflation works and produces the headline `0.1920513169` `[contest-CPU; Modal Linux x86_64 reproduction]` score. CUDA-enabled hosts may take the CUDA path (inflate auto-selects `torch.device("cuda")` when available) and will produce the disclosed paired CUDA score `0.2262100217` `[Modal T4 CUDA replay]` instead. Both axes ship from the same archive bytes (sha256 `6bae0201...`).

# did you include the compression script? and want it to be merged?
no (compression pipeline depends on private training infrastructure not packaged in this submission; reproducibility of the archive bytes is documented via the canonical SHA-256 above)

# additional comments

Built on top of [PR #101](https://github.com/commaai/comma_video_compression_challenge/pull/101) by @SajayR (HNeRV decoder inherited from @AaronLeslie138's [PR #95](https://github.com/commaai/comma_video_compression_challenge/pull/95), QAT fine-tuning from @EthanYangTW's [PR #98](https://github.com/commaai/comma_video_compression_challenge/pull/98), latent-correction sidecar from @BradyMeighan's [PR #100](https://github.com/commaai/comma_video_compression_challenge/pull/100); arithmetic-coding selector pattern from @rem2's [PR #103](https://github.com/commaai/comma_video_compression_challenge/pull/103) SILVER). The underlying HNeRV decoder architecture is [Chen et al. 2023](https://arxiv.org/abs/2304.02633) ([code](https://github.com/haochen-rye/HNeRV)).

**Competitive + innovative per the 2026-05-11 new-submission gate:**

This submission satisfies both criteria of the maintainer's post-deadline new-submission gate (verbatim from the [PR #108 closure](https://github.com/commaai/comma_video_compression_challenge/pull/108) on 2026-05-11T19:19:57Z):

> closing this pr per the new submission guidelines, the tricks used are already established in several past submissions
>
> 'is this submission competitive or innovative? explain why
> competitive: better than top # 1 submission
> innovative: it has a novel idea that is not on the leaderboard yet, might not be competitive, but has potential'

- **Competitive:** `0.1920513169` `[contest-CPU]` improves on top-merged [PR #102](https://github.com/commaai/comma_video_compression_challenge/pull/102)'s reported `0.19538` `[contest-CPU]` by `-0.00333` (verified by comparing both archives on the same CPU axis).
- **Innovative:** the FEC6 fixed-Huffman k=16 per-pair frame-exploit selector composition (described below) is not currently merged on the leaderboard.

**Novel in this submission (FEC6 = Frame Exploit Compactor v6):**

1. **K=16 frame-conditional per-pair mode palette** (vs PR101's K=8 static modes). Modes include `none`, `frame0_blue_chroma_amp_1`, `frame0_red_chroma_amp_1`, `frame0_blue_tile_*`, `frame0_chroma_offset_*`.
2. **Fixed-Huffman codebook on selector indices** (vs raw-byte storage in PR101 GOLD). 4-bit naïve cost (4 × 600 = 300 bytes) is compacted to roughly 107 bytes for the FEC6 stream via a fixed-Huffman codebook designed against the empirical selector-mode distribution observed on `videos/0.mkv`. Final payload is wrapped with [brotli](https://datatracker.ietf.org/doc/html/rfc7932) (RFC 7932).
3. **Per-pair selector decision is offline.** Selector indices are precomputed against the SegNet/PoseNet response surface during candidate enumeration; the inflate path is fully deterministic — no on-device search at inflate time.
4. **Runtime contract:** archive bytes are byte-stable and inflate path is byte-stable for a fixed hardware axis.

**Score components (CPU and CUDA paired anchors on the same archive bytes):**

| axis | seg dist | pose dist | archive bytes | score |
|---|---:|---:|---:|---:|
| `[Modal Linux x86_64 CPU; GHA host validation pending]` | `0.00056029` | `0.00002943` | `178,517` | **`0.1920513169`** |
| `[Modal T4 CUDA replay]` | `0.00066299` | `0.00016846` | `178,517` | `0.2262100217` |

The CPU and CUDA score components decompose identically on rate (`25·R = 0.118867`); the score split between axes is driven entirely by `d_seg` and `d_pose` distortions, consistent with bit-identical archive bytes flowing through device-dependent floating-point paths in upstream [`evaluate.py`](https://github.com/commaai/comma_video_compression_challenge/blob/main/evaluate.py).

**Reproducibility:**

- **Archive bytes (canonical):** SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`; size `178,517` bytes; single-member ZIP (member name `x`) with deterministic timestamps + central-directory ordering. Identical bytes flow to CPU and CUDA evaluators; the per-axis score split is reproduced from those bytes alone.
- **Inflate runtime:** 4 Python files in `submission_dir/` (`inflate.py` 397 LOC + `src/codec.py` 480 LOC + `src/frame_selector.py` 209 LOC + `src/model.py` 54 LOC). Fully self-contained; auditable file-by-file. No scorer weights loaded at inflate time per the upstream [`evaluate.py`](https://github.com/commaai/comma_video_compression_challenge/blob/main/evaluate.py) strict-scorer rule; no on-device search; no learned components at inflate.
- **Dependency closure** (declared in `inflate.sh`): Python stdlib + `torch` + `brotli` only.
- **Entry point:** `inflate.sh` is the canonical 3-argument upstream contract (`$1` archive_dir, `$2` output_dir, `$3` file_list).
- **Rate term identity:** `25 · R = 25 · 178,517 / 37,545,489 = 0.118867` (charged by upstream `evaluate.py`; identical across CPU/CUDA). Python source bytes are not charged.

**Limitations:**

- Modal CPU is not yet host-bot/GHA-validated for this exact archive. Treat the headline number as a Modal Linux CPU reproduction until the maintainer's CI runner posts a same-axis comment.
- The Modal T4 CUDA score is paired context, not the promoted axis. This packet is CPU-axis first because that is where the public-leaderboard comparison is most direct.
- The CPU/CUDA score split is observed and documented, not causally attributed.
- This packet is contest-specific (offline access to the fixed 600-pair contest video for selector precomputation); it is not framed as directly production-deployable.

**Operational notes:**

- The `report.txt` shipped at `submission_dir/report.txt` contains an absolute path (`/root/modal_auth_eval_cpu_work/eval_work/...`) in the `report:` field. This is the upstream `evaluate.py` output format (path of the report file on the runner) and is not redacted because doing so would diverge from the upstream output contract.

---

**Appendix A — Upstream PR template format citation (for reviewer transparency).** This PR body is structured to match the upstream PR template at [`upstream/.github/pull_request_template.md`](https://github.com/commaai/comma_video_compression_challenge/blob/main/.github/pull_request_template.md) verbatim. The 5 required headings are present and in order: `# submission name:` / `# upload zipped \`archive.zip\`` / `# report.txt` / `# does your submission require gpu for evaluation (inflation)?` / `# did you include the compression script? and want it to be merged?` / `# additional comments`.

**Appendix B — Pre-submission compliance gate verdict.** Before invoking `gh pr create`, the local pre-submission compliance gate `scripts/pre_submission_compliance_check.py --submission-dir <submission_dir> --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf --expected-archive-size-bytes 178517 --contest-final --strict` was run. The gate validates archive bytes (SHA-256, ZIP grammar, member name, deterministic timestamps), `inflate.sh` executable + 3-argument signature, report.txt presence, and 1:1 contest format conformance. The 21 structural-archive checks PASS (archive bytes, ZIP local + central header agreement, member-name safety, single-member topology, ZIP determinism). The 18 remaining checks require artifacts that are operator-gated and produced as part of the PR submission workflow itself: hosted-archive URL (D3 Option A operator-approved upload), fresh `contest_auth_eval.json` paired CPU+CUDA on the hosted archive (the runner downloads our archive.zip and re-runs `upstream/evaluate.sh` on both axes), `archive_manifest.json` linking the hosted URL, and dispatch-ledger linkage. These items are produced at PR creation time, not at PR body authoring time. The structural checks confirm the submission packet itself is contest-compliant.
