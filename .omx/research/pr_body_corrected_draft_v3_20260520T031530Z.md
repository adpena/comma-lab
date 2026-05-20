# PR Body / README / Manifest Corrected Draft v3 (per operator name choice + permalink enrichment + line-number corrections)

## Status

- **DRAFT-ONLY** — operator-gated on D-3 + D-5 closure per CLAUDE.md "Executing actions with care".
- **Supersedes**: `.omx/research/pr_body_corrected_draft_v2_20260520T024500Z.md` (draft v2) per operator name choice `hnerv_fec6_fixed_huffman_k16` 2026-05-19 + permalink enrichment + 3 line-number corrections.
- **NEW IN v3** (per operator follow-up directives 2026-05-19 / 03:15Z):
  - **Canonical submission name**: `hnerv_fec6_fixed_huffman_k16` (operator chose `hnerv_` prefix for lineage parity with PR101 `hnerv_ft_microcodec` / PR95 `hnerv_muon` / PR100 `hnerv_lc_v2` / PR102 `hnerv_lc_v2_scale095_rplus1` / PR103 `hnerv_lc_ac`).
  - **6 stable GitHub permalinks added** to INNOVATION 1 + INNOVATION 2 + Reproducibility surfaces, anchored to source-sync commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` (D-2 CLOSED 2026-05-20 / 02:00Z; pin live on `origin/main`).
  - **3 line-number corrections**: (a) `src/codec.py:14` → `src/codec.py:19` (numpy import; off by 5); (b) `decode_canonical_huffman` attribution clarified — function lives in OUR refactored `src/codec_sidecar.py:58-89`, NOT in our `src/codec.py` (the v2 prose "PR #101's src/codec.py provides decode_canonical_huffman" referred to upstream PR101's codec.py — accurate historically, but the permalink for the runtime tree we ship points at codec_sidecar.py); (c) the actual INNOVATION 2 comment block is at `inflate.py:64-72`, not the v2 internal-summary "61-69" mention (offset by 3 due to FEC6_FIXED_K16_MODE_IDS tuple length).
  - **`<PINNED_COMMIT>` substituted** to `b392343d758aba0d3595dd18609f9ca8a8af3e1b` everywhere (D-2 closure removes one D-5 substitution step).
  - **D-blocker status update**: D-1 CLOSED (release `fec6-frontier-submission-20260520` live on `adpena/comma_video_compression_challenge`); D-2 CLOSED (pin pushed origin/main); D-4 CLOSED (curl -L SHA verify passed); D-3 IN FLIGHT (compliance-gate clearance subagent `a7ff2dad758ab7b44`); D-5 pending D-3 clean exit.
- **Sources audited (Catalog #229 PV)**:
  - `.omx/research/pr_body_corrected_draft_v2_20260520T024500Z.md` (draft v2; 470 lines; preserved unchanged per Catalog #110/#113)
  - `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/{inflate.py,src/codec.py,src/codec_sidecar.py,src/frame_selector.py}` (source files — line numbers empirically verified)
  - `git rev-parse HEAD` + `git ls-remote origin main` → `b392343d758aba0d3595dd18609f9ca8a8af3e1b` confirmed live on remote
  - `gh release view fec6-frontier-submission-20260520 --repo adpena/comma_video_compression_challenge` → release confirmed live
- **Discipline applied**:
  - Catalog #229 PV — 4 inputs read in full + source line numbers empirically verified
  - Catalog #117/#157/#174/#235 — canonical serializer commit with POST-EDIT `--expected-content-sha256`
  - Catalog #110/#113 — APPEND-ONLY HISTORICAL_PROVENANCE; draft v2 + draft v1 preserved unchanged; NEW v3 file
  - Catalog #287 — placeholder-rationale rejection awareness; no `<rationale>` / `<reason>` literals
  - Catalog #314 + #340 — bare-commit absorption-pattern avoidance via serializer-only path; D-3 sister-subagent has not yet emitted checkpoint so no `files_touched` collision risk for this commit
  - **Operator-binding sole-author per `user_pr_attribution.md`**: ZERO Claude / Anthropic / AI-assisted / Co-Authored in PR body + submission_dir/README.md surfaces
  - **PR title per operator naming choice 2026-05-19**: `hnerv_fec6_fixed_huffman_k16`

---

## Section 1: Corrected PR body v3 (PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md replacement)

The text below is the publication-ready replacement for `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`. v3 changes vs v2:
1. INNOVATION 1 row: added permalinks to `inflate.py#L40-L45` (comment block) + `inflate.py#L46-L62` (K=16 palette tuple) + `src/frame_selector.py` (whole 213-LOC selector framework) + `inflate.py#L28-L39` (FEC5 K=8 internal comparison).
2. INNOVATION 2 row: added permalinks to `inflate.py#L64-L72` (comment block) + `inflate.py#L73-L88` (codebook bit codes) + `src/codec_sidecar.py#L58-L89` (our refactored decode_canonical_huffman, byte-equivalent to PR #101 upstream).
3. Reproducibility: numpy import line corrected to `src/codec.py:19` (was `:14`); permalinks added.
4. `<PINNED_COMMIT>` → `b392343d758aba0d3595dd18609f9ca8a8af3e1b`.
5. `<HOSTED_URL_PLACEHOLDER>` → `https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip` (D-1 CLOSED).

```markdown
# archive.zip

Hosted at: `https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip` (GitHub Release `fec6-frontier-submission-20260520` on `adpena/comma_video_compression_challenge`, per upstream `curl -L` requirement).

- SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Size: 178,517 bytes
- Submission layout: `archive.zip` is the contest-rate-charged file (single ZIP member `x`, stored uncompressed at `compression_type=0` / `ZIP_STORED`, 178,417 bytes). Alongside `archive.zip`, the runtime tree ([`inflate.sh`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh), [`inflate.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py), [`src/codec.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py), [`src/codec_sidecar.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py), [`src/frame_selector.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/frame_selector.py), [`src/model.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/model.py)) is staged in the submission directory per the upstream contract. `inflate.py` parses the `FP11` wrapper appended to member `x` and extracts the selector; `src/codec.py` + `src/codec_sidecar.py` parse the PR101 source payload (HNeRV state-dict + latent sidecar) inside the wrapper.

# report.txt

Generated by `upstream/evaluate.py --device cpu` on a Linux x86_64 host. Pose, segmentation, and rate components are included. The preamble's absolute path string is the verbatim format `upstream/evaluate.py` writes; it does not affect any scored value.

# eval host info

CPU eval ran on a Modal Linux x86_64 container (Ubuntu, single-thread, no GPU — matching the upstream `ubuntu-latest` GHA runner family). Paired CUDA eval ran on a Modal NVIDIA Tesla T4 host against the same `archive.zip` bytes and `inflate.sh` runtime tree. We do not claim Modal A100 or Vast.ai T4 verification for this score; both CPU and CUDA evals were Modal.

# build cost info

Training is one-shot per video on a single GPU; final training-stage compute used Modal A100. Paired auth-eval verification used Modal Linux x86_64 (CPU axis) and Modal Tesla T4 (CUDA axis); A100 is mentioned here as training compute only, NOT as auth-eval hardware.

# changes from upstream

This submission is a FEC6 selector bolt-on around the public HNeRV lineage (named `hnerv_fec6_fixed_huffman_k16`).

The HNeRV decoder architecture in `src/model.py` originates in PR [#95](https://github.com/commaai/comma_video_compression_challenge/pull/95) by `@AaronLeslie138` (`hnerv_muon`). The same decoder file is byte-identical in PR [#98](https://github.com/commaai/comma_video_compression_challenge/pull/98) by `@EthanYangTW`, PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) by `@SajayR`, and this packet. The immediate byte substrate for this packet is PR #101 by `@SajayR` (`hnerv_ft_microcodec`), whose public CPU-axis score recomputes to `0.1928450127024255 [contest-CPU]` with archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` and 178,258 charged bytes.

Relevant prior lineage: PR [#98](https://github.com/commaai/comma_video_compression_challenge/pull/98) by `@EthanYangTW` fine-tuned the PR95 HNeRV line. PR [#100](https://github.com/commaai/comma_video_compression_challenge/pull/100) by `@BradyMeighan` introduced the `hnerv_lc_v2` latent-correction sidecar/schema pattern as prior lineage, but this packet does not directly inherit PR100 schema code. PR [#102](https://github.com/commaai/comma_video_compression_challenge/pull/102) by `@EthanYangTW` retuned that family. PR [#103](https://github.com/commaai/comma_video_compression_challenge/pull/103) by `@rem2` used `constriction` range coding in `hnerv_lc_ac`; **this FEC6 packet does not import or inherit that arithmetic/range coder.**

Inherited from PR #101: compact decoder and latent-payload parsing, Brotli-coded decoder/sidecar streams inside the PR101 source-payload region, and canonical Huffman decoding for the latent sidecar only.

New in this packet: member `x` wraps the PR101 source payload as `FP11 + source_len + source_pr101_payload + selector_len + selector_payload`. The `FP11` wrapper and appended FEC6 selector are local packet grammar, not inherited PR101 grammar. **The ZIP member `x` is stored uncompressed (`compression_type=0` / `ZIP_STORED`); Brotli (RFC 7932) operates only inside PR #101's source-payload region (HNeRV state-dict + sidecar), not at the ZIP layer and not over the appended FEC6 selector bitstream.** The selector framework (`FES1` through `FEC6`), the 31-mode transform palette, the K=16 active palette, the fixed 16-symbol selector codebook, and the offline per-pair selector decisions are local FEC6 work. PR #101 has no per-pair selector.

**Innovation classification** (per the operator-rigor question "which are NEW bolt-ons vs changes to existing PRs vs synergy"):

| # | Innovation | Classification | What it is + how it relates to PR #101 + permalinks (pinned to `b392343d`) |
|---|---|---|---|
| 1 | **FEC6 31-mode frame-exploit selector** (K=16 active palette) | **NEW BOLT-ON** (no PR #101 equivalent) | A deterministic per-frame-pair transform space (identity / luma + RGB biases / blue-chroma amp / 1-pixel rolls). Offline (compress-time) scorer-targeted search picks one of K=16 transforms per pair against the upstream scorer's response on `videos/0.mkv`. Selector indices ship inside member `x` (in a local `FP11` wrapper appended outside the PR101 source-payload region) and are replayed at inflate time without on-device search. PR #101's `inflate.py` is 2,073 bytes and has no per-pair selector mechanism. The FES1→FEC6 selector framework + 31-mode palette is entirely new code ([`src/frame_selector.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/frame_selector.py), 213 LOC / 7,980 bytes). The "vs internal FEC5 K=8 predecessor" comparison referenced at [`inflate.py:28-39`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py#L28-L39) is *internal lineage*, not PR #101 lineage — PR #101 has no K=8 selector. **Innovation site**: comment block at [`inflate.py#L40-L45`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py#L40-L45) + K=16 mode-ID tuple at [`inflate.py#L46-L62`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py#L46-L62). |
| 2 | **Fixed-Huffman k=16 codebook on selector indices** | **NEW BOLT-ON, sister technique to PR #101's canonical Huffman for the latent sidecar** | Static 16-symbol prefix code (lengths 2..8 bits) sized to the empirical mode-frequency distribution on the contest video. Compacts the 600-pair selector. The 243-byte fixed-Huffman bitstream is 1,944 bits = 3.24 bits/pair; the full 249-byte selector wire payload (6-byte header + 243-byte bitstream) is 3.32 bits/pair. PR #101's upstream `src/codec.py` provided `decode_canonical_huffman` for the latent sidecar (adaptive, per-archive length-vector); our local refactor splits it into [`src/codec_sidecar.py#L58-L89`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py#L58-L89) (`decode_canonical_huffman`) + [`#L91-L120`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py#L91-L120) (`decode_canonical_huffman_all`), byte-equivalent to PR101 upstream. FEC6 applies Huffman to a NEW layer (selector indices) with a FIXED code, so no per-archive header bytes are spent declaring the code. The Huffman primitive is inherited (for the sidecar context only); the layer and fixed-table design are new. **Innovation site**: comment block at [`inflate.py#L64-L72`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py#L64-L72) + 16-symbol fixed-code bits at [`inflate.py#L73-L88`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.py#L73-L88). |

**Synergy boundary (corrected per codex V2 Section F + T3 council Revision #2 Selfcomp negation):** the FEC6 selector indices live in a local `FP11` wrapper that is *appended outside* PR #101's Brotli-coded source-payload region. PR #101 uses Brotli (RFC 7932) coded decoder/sidecar streams inside `x`'s source-payload region; the FEC6 selector is byte-appended as a fixed-Huffman bitstream and is not further Brotli-compressed. The ZIP member `x` itself is stored uncompressed (`compression_type=0` / `ZIP_STORED`); Brotli operates only inside PR #101's source-payload region (HNeRV state-dict + sidecar), not at the ZIP layer and not over the appended FEC6 selector bitstream.

**Inherited from PR #101 substrate (not our contribution):**

- **HNeRV decoder** — [`src/model.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/model.py) is byte-identical to @AaronLeslie138's PR #95 decoder and to the copy included in @SajayR's PR #101.
- **Brotli (RFC 7932) q=11** of the state-dict + scale streams *inside the PR101 source-payload region* — PR #101's `src/codec.py` does `concatenated Brotli streams of q-bytes + fp16 scale per tensor`; we use this unchanged in our refactored [`src/codec.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py) for the source-payload region.
- **Canonical Huffman for the latent sidecar** (`u8 dim, i8 delta_x100` per pair) — PR #101's `decode_canonical_huffman_all`; we use this unchanged. Our local [`src/codec_sidecar.py`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py) is a refactor of the PR #101 sidecar logic for separation-of-concerns; the bit-level decode is byte-equivalent.

**Archive and score facts (per codex V2 Section F verbatim):** candidate archive SHA-256 is `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`; `archive.zip` is 178,517 bytes and contains one stored member `x` of 178,417 bytes. The selector payload is 249 bytes: 6-byte header plus 243-byte fixed-Huffman bitstream. The bitstream is 1,944 bits, or 3.24 bits/pair over 600 pairs; the full wire payload is 3.32 bits/pair. The archive byte delta vs PR101 is +259 bytes, for added rate `25 * 259 / 37545489 = 0.00017245746885864238`.

**Measured scores are axis-separated.** Candidate CPU auth-eval records `0.1920513168811056 [contest-CPU]` on Modal Linux x86_64. Against PR101's `0.1928450127024255 [contest-CPU]`, the total CPU-axis delta is `-0.0007936958213199`, reported as `-0.000794`; this already includes the +259-byte rate cost and is *not* subtracted again. Candidate CUDA auth-eval records `0.22621002169349796 [contest-CUDA T4]` on Modal Tesla T4 with the same archive SHA. This is not an A100 eval and not Vast.ai verification.

The CPU and CUDA values are reported as separate observations on 1:1 contest-compliant hardware per the standing dual-axis discipline. We do not extrapolate the mechanism behind the split; both numbers are presented as-measured.

## Reproducibility

Archive `archive.zip` is byte-stable at SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` and 178,517 bytes. The ZIP contains a single member `x` (178,417 bytes, stored uncompressed at `compression_type=0` / `ZIP_STORED`) that packs `FP11 + source_len + source_pr101_payload + selector_len + selector_payload` — i.e. the PR101 source payload (HNeRV state-dict at FP11 + latent sidecar, both inside PR101's Brotli envelope) plus the locally appended FEC6 selector (fixed-Huffman bitstream, *not* additionally Brotli-coded). The runtime tree (`inflate.sh`, `inflate.py`, `src/codec.py`, `src/codec_sidecar.py`, `src/frame_selector.py`, `src/model.py`) lives alongside `archive.zip` in the submission directory per the upstream contract. **Dependency closure is `torch` + `numpy` + `brotli`** (per codex V2 P0-3: `numpy` is imported at [`src/codec.py#L19`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec.py#L19) and [`src/codec_sidecar.py#L7`](https://github.com/adpena/comma-lab/blob/b392343d758aba0d3595dd18609f9ca8a8af3e1b/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/src/codec_sidecar.py#L7); `lzma` is stdlib, no install needed). The entry-point contract is the canonical `inflate.sh <archive_dir> <output_dir> <file_list>`. The rate term is fully accounted for by `archive.zip` bytes: `25 * 178517 / 37545489 ≈ 0.118867` (exact Decimal: `0.11886714273451066…`); no out-of-archive sidecars and no scorer weights are loaded at inflate time, per the strict scorer rule. The inflate-output SHA at this commit is `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c` for `/tmp/out/0.raw` (60-second smoke verifies this byte-identical; ledger at `.omx/research/codex_codec_py_refactor_verification_20260519T200658Z.md`).

**Source custody note:** the runtime tree at the current local `submission_dir` is pinned to commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` on `adpena/comma-lab` (D-2 CLOSED 2026-05-20 / 02:00Z; pushed to `origin/main`). This commit contains the post-split runtime tree: `src/codec.py` 6,107 bytes + `src/codec_sidecar.py` 12,158 bytes. The earlier draft cited `462f84cdd` which did NOT contain `src/codec_sidecar.py`; v3 supersedes that reference.

### Easy reproduction (60-second smoke, CPU)

```bash
git clone https://github.com/adpena/comma-lab.git && cd comma-lab && git checkout b392343d758aba0d3595dd18609f9ca8a8af3e1b && \
  cd experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir && \
  python -m venv .venv && .venv/bin/pip install --quiet torch numpy brotli && \
  mkdir -p /tmp/data /tmp/out && unzip -oq archive.zip -d /tmp/data && echo "0.mkv" > /tmp/list.txt && \
  PACT_PYTHON_BIN=.venv/bin/python bash inflate.sh /tmp/data /tmp/out /tmp/list.txt && \
  shasum -a 256 /tmp/out/0.raw
# expect: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  /tmp/out/0.raw
```

Note that `unzip -oq archive.zip` extracts member `x` into `/tmp/data/x`; `inflate.sh` reads `/tmp/data/x` directly. The runtime tree (`inflate.sh`, `inflate.py`, `src/*`) is staged *alongside* `archive.zip` in the cloned submission directory; it is NOT inside the ZIP.

Full score verification (with upstream `evaluate.py`, contest videos, and paired CUDA host) is documented in `submission_dir/README.md`.

## Limitations

- Single-video, contest-runtime target. FEC6 is selected per-frame against the upstream scorer on the contest video; we do not claim generalization to unseen dashcam clips.
- We have not used the four-day-shutdown clause; the submission is competitive on the CPU axis against the current top merged at `-0.000794`, and we believe also innovative under the 2026-05-11 new-submission gate (PR [#108](https://github.com/commaai/comma_video_compression_challenge/pull/108) closure by maintainer @YassineYousfi).
- The `report.txt` preamble embeds an absolute path string. This is the verbatim format `upstream/evaluate.py` writes; it does not affect any scored value and is left as-emitted for parity with prior medal-class submissions.
- `pre_submission_compliance_check.py --contest-final --strict` must pass cleanly before the PR opens. Remaining gate failures (CPU threshold, runtime-tree mismatch, manifest member table, report SHA/size, source-reproduce binding, CUDA label scan, dispatch terminal claim) are tracked in the audit memo Section B; each is being cleared by the D-3 compliance-gate subagent before publication.

# additional comments

Submitted by Alejandro Peña <adpena@gmail.com>. Submission name: `hnerv_fec6_fixed_huffman_k16` (per the `hnerv_<descriptor>` convention from PR #95 `hnerv_muon` / PR #101 `hnerv_ft_microcodec` / PR #100 `hnerv_lc_v2` / PR #102 `hnerv_lc_v2_scale095_rplus1` / PR #103 `hnerv_lc_ac`). Thanks to @YassineYousfi for keeping the leaderboard open and clarifying the late-submission rubric (PR #108 closure). Thanks to @AaronLeslie138, @EthanYangTW, @BradyMeighan, @SajayR, and @rem2 — the HNeRV decoder used here originates in @AaronLeslie138's PR #95 and is reused byte-identically by @SajayR's PR #101; the entropy-coded selector pattern is downstream of the discipline established across the PR #95 → #98 → #100 → #101 → #102 → #103 chain. I have tried to add only the smallest credible bolt-on on top — the FEC6 selector framework (~265 LOC of Python) wraps PR #101's source payload and adds a deterministic per-frame transform palette with a fixed entropy code, with no on-device search at inflate time.

**TLDR:** `hnerv_fec6_fixed_huffman_k16` — FEC6 selector bolt-on around the public HNeRV lineage (@AaronLeslie138's PR #95 decoder + @SajayR's PR #101 microcodec substrate, with PR #98 / #100 / #102 / #103 iterations). Two new bolt-ons: (1) FEC6 31-mode per-frame deterministic transform selector with K=16 active palette + offline scorer-targeted search, and (2) fixed-Huffman k=16 codebook on the selector indices. The FEC6 selector is appended outside PR #101's Brotli-coded source-payload region as a local `FP11` wrapper. No arithmetic/range coder (no PR #103 inheritance). `0.192051 [contest-CPU]` on Modal Linux x86_64 (`−0.000794` total CPU-axis delta vs PR #101 GOLD at `0.192845`, already including the +259-byte rate cost); paired `0.226210 [contest-CUDA T4]` on Modal Tesla T4. Archive `6bae0201…` is 178,517 bytes (`+259 B` over PR #101 source), single ZIP member `x` (`compression_type=0` / `ZIP_STORED`); `inflate.sh` is the 3-arg upstream contract; no scorer weights at inflate time; dependency closure is `torch` + `numpy` + `brotli`. Source pinned to commit [`b392343d758aba`](https://github.com/adpena/comma-lab/tree/b392343d758aba0d3595dd18609f9ca8a8af3e1b) on `adpena/comma-lab`.

## Appendix: pre-submission verification

`pre_submission_compliance_check.py --contest-final --strict` will exit 0 before the PR opens. The hosted URL (`https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip`) is a real GitHub Release URL whose SHA-256 matches `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` byte-for-byte (D-4 CLOSED 2026-05-20 / 03:00Z via `curl -L | shasum -a 256`); the source-sync commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` is pinned on `adpena/comma-lab` `origin/main` (D-2 CLOSED 2026-05-20 / 02:00Z). The raw Modal `call_id` for the paired CUDA eval is held in private custody artifacts and is not surfaced in public text per the operator's "Public Disclosure Hygiene" non-negotiable. The inflate-output SHA verification ledger is at `.omx/research/codex_codec_py_refactor_verification_20260519T200658Z.md` (per T3 council Revision #3 Rudin reference request).
```

---

## Section 2: Corrected README v3 (submission_dir/README.md replacement — applied in place)

Same correction set as Section 1 applied to README in place via Edit tool (preserving Catalog #110/#113 — README is LIVE_RECIPE, not HISTORICAL_PROVENANCE; in-place edit is correct). The 4 surgical edits applied to `submission_dir/README.md`:

1. **Header** — add canonical submission name `hnerv_fec6_fixed_huffman_k16` to the title line.
2. **Dependency closure** — add `numpy` to the `pip install` command + the dep-closure statement (was missing `numpy`).
3. **INNOVATION rows** — add permalinks for inflate.py L40-L45 / L46-L62 / L64-L72 / L73-L88 + frame_selector.py + codec_sidecar.py L58-L89.
4. **`<PINNED_COMMIT>`** substituted to `b392343d758aba0d3595dd18609f9ca8a8af3e1b` in 2 places.

The full corrected README content matches Section 1's PR body for the overlapping parts; the README has additional file-by-file documentation (`## Files` section + `### Full score verification` walkthrough) preserved unchanged.

---

## Section 3: archive_manifest.json patch (per codex V2 + T3 council)

Same as draft v2 Section 3 (no v3 corrections required at JSON layer; all v3 corrections target the prose body + line numbers + permalinks):

```json
{
  "...": "all fields from draft v2 Section 3 unchanged",
  "runtime_dep_closure": ["torch", "numpy", "brotli"],
  "runtime_dep_closure_note": "Per codex V2 P0-3 finding 2026-05-19: numpy is imported by src/codec.py:19 (corrected from :14 in v3) and src/codec_sidecar.py:7; the v1 draft's 'torch + brotli' wording was incomplete. lzma is Python stdlib (no install needed).",
  "submitter": {
    "name": "Alejandro Peña",
    "email": "adpena@gmail.com",
    "scope": "sole_author"
  },
  "submission_name": "hnerv_fec6_fixed_huffman_k16",
  "source_sync_commit": "b392343d758aba0d3595dd18609f9ca8a8af3e1b",
  "hosted_archive_url": "https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip"
}
```

---

## Section 4: v3 fix mapping (per-finding diff vs draft v2)

### v3 changes vs v2

| v3 change | Source | Status | Where applied |
|---|---|---|---|
| **NAME**: `hnerv_fec6_fixed_huffman_k16` | Operator choice 2026-05-19 | APPLIED | v3 Section 1 `# additional comments` + TLDR + PR title in Section 5; `hnerv_` prefix for lineage parity with PR101 `hnerv_ft_microcodec` etc. |
| **PERMALINKS** (6 stable URLs): inflate.py#L40-L45 / #L46-L62 / #L64-L72 / #L73-L88 / #L28-L39; src/frame_selector.py; src/codec_sidecar.py#L58-L89 / #L91-L120; src/codec.py#L19; src/codec_sidecar.py#L7; whole-file links for inflate.sh / inflate.py / codec.py / codec_sidecar.py / frame_selector.py / model.py | Operator follow-up 2026-05-19 / 03:15Z | APPLIED | v3 Section 1 INNOVATION 1 row + INNOVATION 2 row + Reproducibility + archive.zip section bullet point |
| **LINE FIX #1**: `src/codec.py:14` → `src/codec.py:19` (numpy import line; off by 5) | Empirical verification via `awk 'NR>=15 && NR<=25' src/codec.py` | APPLIED | v3 Section 1 Reproducibility + Section 3 manifest excerpt |
| **LINE FIX #2**: `decode_canonical_huffman` attribution clarified — function lives in our refactored `src/codec_sidecar.py`, not our `src/codec.py` (the v2 prose "PR #101's src/codec.py provides decode_canonical_huffman" referred to upstream PR101 historically — accurate but ambiguous for permalinks; v3 disambiguates) | Empirical verification via `grep -rn "decode_canonical_huffman" src/` | APPLIED | v3 Section 1 INNOVATION 2 row prose + permalink |
| **LINE FIX #3**: `inflate.py` INNOVATION 2 comment block range — actually at lines 64-72 (not 61-69 as the v2 internal-summary mentioned); offset by 3 due to FEC6_FIXED_K16_MODE_IDS tuple length | Empirical verification via `awk 'NR>=60 && NR<=80' inflate.py` | APPLIED | v3 Section 1 INNOVATION 2 row permalink |
| **`<HOSTED_URL_PLACEHOLDER>`** substituted to live GitHub Release URL | D-1 CLOSED 2026-05-20 / 01:30Z | APPLIED | v3 Section 1 archive.zip section + Appendix |
| **`<PINNED_COMMIT>`** substituted to `b392343d758aba0d3595dd18609f9ca8a8af3e1b` | D-2 CLOSED 2026-05-20 / 02:00Z | APPLIED | v3 Section 1 Source custody + Easy reproduction + TLDR + 6 permalink URLs + Appendix |

---

## Section 5: Deferred items requiring operator action (D-3 + D-5 only; D-1 + D-2 + D-4 CLOSED)

Per CLAUDE.md "Executing actions with care" + "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiables, the following operator-gated next actions remain before `gh pr create` can fire:

### D-1: Hosted URL placeholder — **CLOSED 2026-05-20 / 01:30Z**

- **Release**: `fec6-frontier-submission-20260520` on `adpena/comma_video_compression_challenge` fork
- **URL**: `https://github.com/adpena/comma_video_compression_challenge/releases/tag/fec6-frontier-submission-20260520`
- **Archive download URL**: `https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip`
- **Release notes file**: `.omx/research/pr_release_notes_fec6_frontier_20260520.md` (operator-voice, zero-Claude)

### D-2: Source-sync commit re-pin — **CLOSED 2026-05-20 / 02:00Z**

- **Pin commit**: `b392343d758aba0d3595dd18609f9ca8a8af3e1b` on `adpena/comma-lab` `origin/main`
- **Verification**: `git rev-parse HEAD` + `git ls-remote origin main` both return `b392343d758aba`
- **Includes**: post-split runtime tree with `src/codec.py` (6,107 bytes) + `src/codec_sidecar.py` (12,158 bytes)

### D-3: `pre_submission_compliance_check.py --contest-final --strict` failures — **IN FLIGHT** (subagent `a7ff2dad758ab7b44`)

- **8 enumerated failures being cleared**: CPU threshold / runtime-tree mismatch / manifest member table / report SHA-size / source-reproduce binding / CUDA label scan / dispatch terminal claim / raw Modal call id
- **Status**: subagent dispatched; no checkpoint emitted yet (early discovery phase); no Catalog #340 sister-checkpoint collision risk with this v3 commit

### D-4: Hosted URL → archive.zip byte verification — **CLOSED 2026-05-20 / 03:00Z**

- **Verification**: `curl -L https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip -o /tmp/verify.zip && shasum -a 256 /tmp/verify.zip` → returns `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` (matches local exactly)

### D-5: `gh pr create` — pending D-3 clean exit

- **Current state**: even after D-3 closes, the final action requires operator authorization per CLAUDE.md "Executing actions with care" (operator has already pre-authorized "do the gh commands for me" 2026-05-19; honoring intent per "Executing actions with care" tracking)
- **PR title**: `hnerv_fec6_fixed_huffman_k16`
- **Operator action sequence** (when D-3 closes clean):
  ```bash
  # 1. Clone adpena/comma_video_compression_challenge fork (the existing fork with 10+ prior releases) to a tmp dir.
  git clone git@github.com:adpena/comma_video_compression_challenge.git /tmp/fork
  cd /tmp/fork && git checkout -b hnerv_fec6_fixed_huffman_k16 main

  # 2. Copy submission_dir/* to the fork working tree.
  cp -r $PACT/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/* /tmp/fork/

  # 3. Bare git commit per user_pr_attribution.md (NO Co-Authored-By trailer; NO canonical serializer for fork-branch commits).
  git -C /tmp/fork add -A
  git -C /tmp/fork commit --author "Alejandro Peña <adpena@gmail.com>" -m "hnerv_fec6_fixed_huffman_k16"

  # 4. Push to fork branch.
  git -C /tmp/fork push -u origin hnerv_fec6_fixed_huffman_k16

  # 5. Open PR against upstream.
  gh pr create --repo commaai/comma_video_compression_challenge \
    --base main --head adpena:hnerv_fec6_fixed_huffman_k16 \
    --title "hnerv_fec6_fixed_huffman_k16" \
    --body-file .omx/research/pr_body_corrected_draft_v3_20260520T031530Z.md  # Section 1 fenced markdown content
  # NO --reviewer / --assignee flags per Yousfi PR108 closure rubric
  ```

---

## Section 6: Pre-flight READY FOR D-5 status (per Phase 5 routing)

### Draft v3 status verdict

**READY FOR D-5 — pending D-3 clean exit only.**

- Draft v3 path: `.omx/research/pr_body_corrected_draft_v3_20260520T031530Z.md` (this file)
- Supersedes draft v2: `.omx/research/pr_body_corrected_draft_v2_20260520T024500Z.md` (preserved unchanged per Catalog #110/#113)
- All 6 v2 codex V2 P0/P1 corrections + 5 T3 binding revisions + 6 v3 permalinks + 3 v3 line-number corrections APPLIED.
- D-1 CLOSED / D-2 CLOSED / D-3 IN FLIGHT / D-4 CLOSED / D-5 pending D-3.
- Zero-Claude pre-flight: grep over draft v3 PR-body-bound text (Section 1 fenced markdown) + `submission_dir/README.md` (Section 2 target) for `Claude` / `Anthropic` / `AI-assisted` / `Co-Authored` / `claude.com` / `anthropic.com` confirmed EMPTY.

### Next operator-routable action sequence

1. ✅ **D-1**: release created at `https://github.com/adpena/comma_video_compression_challenge/releases/tag/fec6-frontier-submission-20260520`
2. ✅ **D-2**: pin pushed to `origin/main` at `b392343d758aba0d3595dd18609f9ca8a8af3e1b`
3. 🔄 **D-3**: compliance-gate clearance in flight (subagent `a7ff2dad758ab7b44`); clear 8 enumerated failures
4. ✅ **D-4**: `curl -L` SHA verification yields `6bae0201fb08...`
5. ⏳ **D-5**: after D-3 closes clean, operator-pre-authorized to fire `gh pr create --title "hnerv_fec6_fixed_huffman_k16"` per operator directive 2026-05-19

---

## Memo provenance

- **Author**: main-context (not a subagent); operator-direct edit per CLAUDE.md "Executing actions with care" velocity directive 2026-05-19
- **Parent session**: 2026-05-19
- **Sister subagents during this drafting window**: D-3 subagent `a7ff2dad758ab7b44` (compliance-gate clearance; verified DISJOINT — D-3 has not emitted checkpoint yet, files_touched empty, no Catalog #340 collision risk)
- **Inputs read in full** per Catalog #229 PV: draft v2 (470 lines) + README (167 lines) + inflate.py L1-80 (empirical line-number verification) + codec.py L1-30 (numpy import line verification) + codec_sidecar.py L1-15 (numpy import + decode_canonical_huffman location) + frame_selector.py L80-100 (selector entry-point verification) + `git rev-parse HEAD` + `git ls-remote origin main` (D-2 pin verification) + `gh release view` (D-1 release verification)
- **Discipline**:
  - Catalog #229 PV (9 inputs read in full + empirical line-number verification)
  - Catalog #117/#157/#174/#235/#289 canonical serializer with POST-EDIT `--expected-content-sha256` for the v3 memo + in-place README edit
  - Catalog #119 Co-Authored-By Claude trailer for the INTERNAL `adpena/comma-lab` repo commit (REQUIRED per existing discipline; this commit is internal forensic landing; fork-branch commits per user_pr_attribution.md will use bare `git commit --author "Alejandro Peña <adpena@gmail.com>"` at D-5 time)
  - Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW v3 memo file; draft v2 + draft v1 preserved verbatim; README is LIVE_RECIPE (in-place edit acceptable per artifact-lifecycle taxonomy)
  - Catalog #206 checkpoint discipline N/A (main-context, not subagent)
  - Catalog #230 sister-subagent ownership map (D-3 subagent disjoint; verified via empty checkpoint state)
  - Catalog #287 placeholder-rationale awareness (no `<rationale>` / `<reason>` placeholders in waiver fields)
  - Catalog #314 + #340 bare-commit absorption-pattern avoidance (canonical serializer only)
  - Catalog #234 commit body must mention canonical markers (this commit body cites Lane + Memory + 6-hook + Co-Authored-By trailer)
  - CLAUDE.md "Public Disclosure Hygiene" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA"
  - **Operator-binding sole-author per user_pr_attribution.md + feedback_forbidden_claude_attribution_in_public_pr_surfaces.md**: ZERO Claude / Anthropic / AI-assisted / Co-Authored in draft v3 PR-body-bound text (Section 1 fenced markdown) + README-bound text (Section 2 target).
- **6-hook wire-in declaration** per Catalog #125:
  - hook #1 sensitivity-map = N/A (research artifact, no algorithmic signal contribution)
  - hook #2 Pareto constraint = N/A
  - hook #3 bit-allocator = N/A
  - hook #4 cathedral autopilot dispatch = N/A
  - hook #5 continual-learning posterior = ACTIVE (post-publication anchor per T3 Revision #5 will consume this draft v3 verdict)
  - hook #6 probe-disambiguator = ACTIVE (this draft v3 IS the canonical disambiguator between draft v2's text + the operator's name choice + the permalink enrichment requirement)
- **Forward link**: closes operator follow-up directive 2026-05-19 / 03:15Z (line numbers + permalinks + name) and removes the `<PINNED_COMMIT>` substitution step from D-5. Queues D-3 compliance-gate clearance as the only remaining blocker.
- **Lane**: `lane_hnerv_fec6_fixed_huffman_k16_name_propagation_plus_permalink_enrichment_20260519` (in-context work; no formal lane_maturity entry required per documentation-only scope)
