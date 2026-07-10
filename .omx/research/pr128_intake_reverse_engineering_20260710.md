# Public-frontier intake — PR #128 `rhnerv_latent_polish` (a12dongithub, claimed 0.188532 → head-repo 0.187992 [contest-CPU external, WINDOWS-CPU caveat])

**Date:** 2026-07-10
**Subagent:** `pr128-intake` (checkpointed in `.omx/state/subagent_progress.jsonl`)
**Custody:** `/Volumes/VertigoDataTier/pact/public_pr128_intake_20260710_claude/` (detached clone `pr128_repo` @ PR head + `artifacts/archive.zip` + `artifacts/archive_v1.zip`). NOT checked out into the shared worktree.
**STORES CONSULTED:** `.omx/research/public_pr112_frontier_beat_intake_20260610.md` (our PR112 intake — PR112 = lossless ctx-recode of OUR PR110 payload) · `.omx/state/canonical_frontier_pointer.json` (our CPU pointer 0.19109982, CUDA 0.20533) · CLAUDE.md L14–L32 (PR95/101/103 canonical grammar) · `.omx/research/carrier_sota_online_survey_20260611T042151Z.md` · grep `.omx/research` for latent-polish/rhnerv priors.

---

## Headline verdict

**YES it is HNeRV-family — in fact it is OUR OWN LINEAGE, untouched, with zero training.**
PR #128 = frozen PR #95/#101 HNeRV decoder (byte-identical weights) + **our PR #110 (@adpena) FEC6 K=16 selector + codec grammar (byte-identical)** + PR #112 (mattneel) constriction ctx range coder (unchanged) + **the sole contribution: exact-score-gated discrete latent "click" polish** (±1/±2 steps on the stored 8-bit latent codes, each candidate scored through the EXACT inflate chain + exact evaluator + real re-encoded byte count, kept only if the exact contest score strictly improves), plus **folding PR #101's 607-byte latent sidecar into the base latent codes** (−605 archive bytes at zero distortion cost).

**How it beats the frontier:** vs PR #112 (0.191126, = our decoded video): d_seg 0.00056032 → 0.00053309 (score −0.00272), bytes 177,136 → 176,531 (score −0.00040), d_pose ≈ unchanged (0.00002943 → 0.00002937). Net ≈ **−0.0031 → 0.187992** on the CPU axis. No architecture change, no gradient step, no new model — "which 8-bit codes to store, chosen one verified click at a time."

---

## 1. PR record (SOURCE-VERIFIED unless tagged)

| field | value |
|---|---|
| PR | **#128** `rhnerv_latent_polish (0.188532)` — OPEN, no maintainer eval comment yet |
| author | **a12dongithub** (Samarth Singhal; DTU 2k19 batch; ML/vision background — cell vision, deepfake detection; "looking for internship"; no compression repos beyond the contest fork) |
| URL | https://github.com/commaai/comma_video_compression_challenge/pull/128 |
| head SHA | `3eb39cac8261075888b1c562e9d9c2a7f1c7aebf` |
| created / updated | 2026-07-09T09:29:14Z / 2026-07-10T09:01:21Z |
| claimed score (PR body) | 0.188532 — recomputed from components 100·0.00053838 + √(10·0.00002941) + 25·176531/37545489 = 0.053838+0.017150+0.117545 = **0.188533 ✓ internally consistent** — `external-claim` |
| head-repo `report_cpu.txt` | seg 0.00053309, pose 0.00002937, 176,531 B → recomputed **0.187992** — `external-claim`; **generated on WINDOWS CPU** (`C:\Users\samar\...` paths, batch 16, 2 threads, seed 1234) — NOT the ubuntu-x86_64 contest CI axis |
| archive URL | `github.com/a12dongithub/comma_video_compression_challenge/releases/download/rhnerv-latent-polish-20260709/archive.zip` |
| downloaded archive | **176,531 bytes, sha256 `cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395`** ✓ VERIFIED download; **≠ PR-body claimed sha `fae8d338…`** — the release asset was REPLACED 2026-07-10T09:01:39Z (same minute as the PR update). `compress.py` at head asserts the CURRENT sha (`cfd941de…`, member `8f7b808e…`), so head repo ↔ current asset are consistent; the PR body text is stale (an earlier polish iteration: body says ~1,565 clicks, METHOD.md says 1,802 changed codes, compress.py docstring says 2,162 — the author kept polishing). v1 asset also archived (`ab732593…`). |
| ZIP member table | single member `x`, ZIP_STORED (method 0), 176,431 B, CRC 341ecef1, fixed 1980-epoch timestamp, deterministic rebuild via `compress.sh` (asserts member+archive SHA) |
| CUDA axis | **NO claim made** (inflate pins device to CPU; "Optimized for CPU run specifically"). Never infer CUDA from CPU. |

Both claimed scores beat our pointer **on the CPU axis only** (ours 0.19110 contest-CPU). All scores `external` until exact replay (§8).

## 2. Bit-level anatomy (SOURCE-VERIFIED by parsing the downloaded bytes)

ctx container (PR #112 format), 7-byte header `u8 (version<<4 | coder-id bits) | u24 dec_len | u24 lat_len`, coder_ids=(1,1,1)=ctx for all three sections:

| section | coded bytes | byte entropy | content |
|---|---|---|---|
| decoder | **161,104** | 7.9989 bits/B | 229,014 raw weight bytes (28 tensors, uint8 codes × fp16 scale; 7 streams) range-coded with per-tensor adaptive 256-ary models, geometric-primed priors, 2-byte/model header (M/inc/eps grids). **Frozen = byte-identical to #101/#95.** |
| latent | **15,072** | 7.9883 | 16,912 raw bytes (112-B fp16 min/scale header + 600×28 uint8 codes), temporal-delta + per-dim causal AR prediction (AR(1)+lag2+up to 4 cross-dims, integer-quantized LS coeffs) + discrete-Gaussian residual models (Q_TABLE). **The polished section.** |
| selector | **248** | 7.0594 | PR #110 FEC6 K=16 fixed-Huffman per-pair frame-0 transform codes, re-wrapped by an adaptive 16-ary range model (decodes back to the exact 249-B FEC6 wire payload). **Frozen = byte-identical to #110.** |

All sections sit at the byte-entropy ceiling (≈8 bits/B) — no residual generic-compressor slack. No trailing sidecar: **PR #101's 607-B per-pair single-dim correction sidecar (our L27) is GONE — folded into the base codes.** Deviations from the PR101/103 canonical grammar (L20–L32): monolithic 4-section brotli/LZMA grammar replaced by PR112's 3-section ctx range-coded container (L23/L24 superseded by constriction adaptive models); L27 sidecar eliminated; everything else (L18 decoder, L19 28-d pair latents, L21 byte maps, L22 storage perms, L25 temporal delta, L28 channel biases, L29 fp16 scales) intact and verbatim.

`inflate.py` (232 LOC) + `inflate.sh` (36 LOC): deps numpy + torch + constriction (all in harness base env, same as merged PR #112); device **pinned CPU**; exact #110/#112 decode chain — HNeRVDecoder (PixelShuffle + bilinear-skip + sin, 6 stages 6×8→384×512, channel taper [36,36,36,27,20,18,18], dilated-conv refine + dual sigmoid RGB heads = PR95 verbatim per `model.py` docstring), 16-pair batches, bicubic 874×1164 align_corners=False, #98 channel biases (f0 R−1, f0 B−1, f1 G−1), clamp/round, per-pair FEC6 transform, uint8 NHWC stream-write.

## 3. Family classification

**HNeRV-family, PR95→101→110→112 direct descendant** (per tensor schema: 28 tensors, stem Linear(28→36·6·8), 6× Conv(in,out·4,3×3)+PixelShuffle(2)+bilinear-skip+sin, `TENSOR_SCHEMA` numels match PR101 exactly; `DECODER_STORAGE_ORDER`/`CONV4_STORAGE_PERMS`/`DECODER_BYTE_MAPS`/`LATENT_DIM_ORDER` verbatim from OUR PR110 `codec.py`, credited "@adpena, MIT" in the source). Not a new class — a **search-layer** contribution on a frozen stack.

## 4. Mechanism deltas vs PR101/103/110/112 (the crux)

1. **Exact-score-gated discrete latent coordinate descent** (THE contribution; d_seg axis, −0.0027). Key structural exploit: pair-locality — pair p's frames depend only on z[p], the metric is a mean over pairs ⇒ (a) exact per-candidate effect from re-rendering ONE pair, (b) accepted clicks on different pairs add EXACTLY (no interaction). **"Diagonal batching":** one 600-pair render scores 600 independent candidates exactly (every pair gets its own copy of the same (dim,δ) click) ⇒ full sweep of 600×28×4 = 67,200 candidates in 112 renders. Rounds: baseline → sweep → per-candidate byte-delta estimate (only approximate ingredient) → accept-if-net-improving, ≤1 click/pair/round → exact re-verify whole set → bank shippable archive each round → plateau when a full sweep accepts nothing. ~1,565–2,162 net code changes across iterations, 577/600 pairs touched. Genetic-algorithm framing in PR body is marketing; it is greedy exact-gated coordinate descent.
2. **Sidecar folding** (rate axis, −605 B → −0.00040): PR101's 607-B sidecar corrections are representable on the base 8-bit grid ⇒ fold and delete the section. (±0.5-step moves thereby cease to exist; author measured re-adding a sidecar costs more bytes than it's worth.)
3. **CPU-axis-native selection** (axis-integrity): clicks selected on GPU lost ≈0.0009 of their gain when re-scored on CPU (bicubic LSB flips borderline judge-pixels) ⇒ final sweeps run fp32 CPU with fixed batch layout; ±2 steps added because some pairs want a 2-click move whose 1-click midpoint is worse (non-convexity along a dim). Empirically confirms our CPU/CUDA-separate-axes discipline.
4. **Measured negatives (published, high value):** (a) zero-shot 7-bit deadzone re-quantization of weights: saves 1.5–4 KB but costs 16–25× in seg — payload sits at a sharp optimum from PR95's QAT; (b) **gradient fine-tuning (entropy-regularized QAT, straight-through rounding): surrogate improved while TRUE seg metric DOUBLED** — systematic drift, not step size (their words: "per-pixel-margin surrogates optimize the wrong thing near this optimum"); (c) weight-code clicks (same exact-gated search on 229K weight codes, gradient-ranked, bisected to single clicks): EVERY one rejected — the decoder is at a strict discrete local optimum. The latents were the only section with residual slack.
5. Context vs rivals: PR #125 `hnerv_qlp` (0.190946) and PR #127 `qlp_exactgrid` (ryanli0070, 0.190506, x86-CI 0.190503) are the GRADIENT variants of latent polish (PR127: boundary seg loss `sigmoid(−margin/τ)`, τ annealed — our margin/level-set surrogate math appearing publicly — with bit-exact straight-through pack-grid quantizer). **The discrete exact-gated search (0.18799) beats both gradient variants by ~0.0025** on the same frozen stack. Near this optimum: exact discrete search > margin-surrogate gradients.

Do WE have the primitives? Exact evaluator + the payload codec: yes (our PR110 is the substrate; PR112 coder published MIT, intake done 2026-06-10). The click-search loop itself: **not built** (no lane found in `.omx/research` for exact-gated discrete payload polish). Margin surrogate: ours (witness). Pair-local diagonal batching: not yet exploited by us as an exact-search actuator.

## 5. Author repo mining

29 public repos; nothing compression-related beyond the contest fork (branches: `master`, `submission/rhnerv_latent_polish`). **No training stack exists** — the submission trains nothing. The search pipeline itself is NOT published; only the deterministic rebuild (`compress.py` re-runs the PR112 coder on `encoder/{decoder_streams.bin 229,014 B, polished_latent_raw.bin 16,912 B, selector_payload.bin 249 B}` — git-LFS objects in the fork) plus METHOD.md's precise algorithm description (§4 above), which is sufficient to reimplement. Background: DTU 2019, vision/ML projects (CellViT, deepfake detection, exam proctoring). `estimated`: solo contestant, no lab.

## 6. Compliance-risk scan

- Deps (numpy/torch/constriction) identical to MERGED PR #112 — precedent-clean. CPU-only inflate; no GPU required.
- No scorer weights, no sidecars, no hidden members: single ZIP_STORED member, deterministic rebuild with SHA asserts; `expected_output.sha256` (`8c5774c3…`) ships for decode verification.
- Rule-118 / data-in-code: schema constants (storage order, perms, byte maps, Huffman tables, Q_TABLE = precomputed exp grid) live in code uncounted — same as merged PR #101/#110/#112 precedent; the video-derived payload (weights/latents/selector choices) is all inside archive.zip. LOW risk.
- Flags for a maintainer: (a) PR-body archive sha is STALE vs the live release asset (author swap post-submission — eval must pin the asset it scores); (b) report_cpu.txt is Windows-CPU, not contest CI (PR127's analog showed ~3e-6 Windows↔CI drift, so material risk is low but the number is not contest-axis); (c) MIT-licensed lineage properly attributed (THIRD_PARTY_NOTICES.md lists #95/#98/#101/#110/#112).

## 7. Draw-from list, ranked (cross-checked against our stores)

1. **Exact-score-gated click polish on OUR frontier payload — READY.** Our CPU pointer (0.19110, sha `b4689726…`) is the same payload family (PR110-lineage). Reimplementing the §4 search (we have the exact evaluator, codec, payload, and 128GB CPU box; ~112 renders/sweep × few rounds) should reproduce ≈−0.003 → ≈0.188 `[contest-CPU]`. NO-FAKE #7 note: borrowed-mechanism-on-borrowed-substrate ⇒ a **defensive bank**, not innovation — but it is the fastest known path to a lower exact pointer. Next step: build `tools/latent_click_polish.py` against our PR110 payload + n600 exact CPU eval loop (byte-delta estimator from the PR112 coder statistics).
2. **Diagonal batching as an exact-search actuator for the WITNESS — needs-build.** The witness is also per-pair (FiLM per-pair mod, per-pair dxi). Any per-pair discrete stored code admits 600-exact-candidates-per-render sweeps through the real byte-close + frozen scorer. This is a post-training polish stage that composes with v7.5/#205 output; it needs our witness byte-close as its substrate. Next step: register as a DSL-visible post-train stage once #202-lineage byte-close is live.
3. **Sidecar folding audit — needs-measure.** Principle: any stored correction expressible on the base quantization grid should fold into base codes (rate ↓, distortion unchanged). Check our R1 7.2KB dxi section and any future sidecar-shaped sections for foldability into carrier grids. Next step: one-off measurement on the R1 byte-close artifact.
4. **The published negatives as priors — READY (knowledge).** (a) near a QAT-sharp optimum, gradient polish with margin surrogates systematically diverges from the true argmax metric (their surrogate improved while true seg doubled) — direct empirical support for our NO-FAKE #8 surrogate-≠-authority law and a caution for any witness post-hoc gradient polish; (b) PR95-family decoder weights are at a strict discrete local optimum — do not spend on weight-code search of that family; (c) GPU-selected discrete moves lose ~30% of gain on CPU — select on the authority axis.
5. **PR112 ctx coder for our own sections — READY, already intaken 2026-06-10** (`public_pr112_frontier_beat_intake_20260610.md`); worth ~−0.0009 on any PR110-shaped payload and adaptable to witness sections.

## 8. Exact-replay plan (PLAN ONLY — not executed; a live ~59 GiB training arm owns this host, no inflate/decode run locally)

Paired axes per `tools/plan_dual_device_auth_eval.py` conventions, on the EXACT downloaded bytes (`/Volumes/VertigoDataTier/pact/public_pr128_intake_20260710_claude/artifacts/archive.zip`, sha `cfd941de…`, 176,531 B):

```bash
# 1) claim lane
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id public_pr128_intake_replay_20260710 \
    --notes "PR128 rhnerv_latent_polish exact replay, archive sha cfd941de"
# 2) emit paired commands (source runtime = PR head submissions/rhnerv_latent_polish/)
.venv/bin/python tools/plan_dual_device_auth_eval.py \
    --archive /Volumes/VertigoDataTier/pact/public_pr128_intake_20260710_claude/artifacts/archive.zip \
    --runtime-dir /Volumes/VertigoDataTier/pact/public_pr128_intake_20260710_claude/pr128_repo/submissions/rhnerv_latent_polish \
    --expected-archive-sha256 cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395
# 3) CPU axis: Linux x86_64 (Modal CPU / GHA), upstream/evaluate.py --device cpu, 600 samples
# 4) CUDA axis: T4/4090, upstream/evaluate.py --device cuda (inflate is CPU-pinned; evaluate on CUDA)
# 5) harvest → refresh canonical pointer as EXTERNAL leaderboard row (never our_local_frontier)
```

Until step 3 lands, every PR128 number stays `external-claim` (Windows-CPU at that).

## Pointer honesty

Our exact pointer is **UNMOVED** (contest-CPU 0.19110). This intake is MEANS: it names the highest-probability next exact row (draw-from #1, expected ≈0.188 CPU on a borrowed-substrate defensive bank) and hands the witness two composable primitives (#2, #3).
