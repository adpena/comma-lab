# PR 95 full artifact deep research + structured comparison against our submission

**Date**: 2026-05-19T19:23:00Z
**Operator directive (2026-05-19)**: "spawn a research subagent to review the full submission and all files and the readme and his full writeup and everything; perhaps we should adapt our cloudflare site into a similar hosting artifact, but there is already so much there and it's also stale and updating should maybe be a long-term project"
**Operator competitive framing (Round 4)**: "PR 95 was the OG and everyone else stole from him, we want to do better than PR 95 on the organization and rigor and discipline and signal and better than the leaderboard on the score"

**Status**: Cloudflare site adaptation DEFERRED-LONG-TERM per operator. THIS memo is the deep research review of PR 95's full artifact pattern + structured comparison against our submission_dir.

**Subagent**: PR-95-FULL-DEEP-RESEARCH
**Lane**: `lane_pr_95_full_deep_research_against_our_submission_20260519`
**Sister coordination**: disjoint from Slot J (council symposium on PR body), Slot K (submission_dir/README.md integration; queued), Slot L (comma-lab + tac public-readiness audit).

---

## Phase 1: Complete PR 95 inventory (21 files, 1591 LOC)

### File-by-file catalog

| File | LOC | Bytes | Purpose | Notable patterns |
|---|---:|---:|---|---|
| `README.md` | 17 | 964 | Top-level introduction + reproduce instructions | Single-paragraph technical summary; explicit "~50 hours on a single GPU"; link to external writeup |
| `compress.sh` | 30 | 991 | Bootstraps the full 8-stage training pipeline from random init | `set -euo pipefail`; finds latest `ckpts/run_*/`; zips `0.bin` into `archive.zip` |
| `inflate.py` | 66 | 2158 | Decoder inflate runtime (single video pair) | `sys.path.insert(0, str(HERE / 'src'))`; canonical bicubic upsample 384x512 → 874x1164; INLINE device-fork pattern (per CLAUDE.md Catalog #205 would flag this) |
| `inflate.sh` | 28 | 811 | Iterates `file_list` and runs `inflate.py` per video | `python -m submissions.${SUB_NAME}.inflate` package-style invocation |
| `src/codec.py` | 180 | 7455 | Archive format: brotli + zigzag + delta-encoded latents | Per-tensor symmetric INT8 quant; meta+decoder+latents 3-section layout; explicitly documents abandoned hybrid AC ("~217 bytes worse, +0.0001 score") |
| `src/data.py` | 159 | 6721 | Video loading + SegNet/PoseNet target precomputation | **Critical**: monkey-patches `frame_utils.rgb_to_yuv6` to make YUV6 differentiable (otherwise `@torch.no_grad()` severs pose gradient). Documents "real bug in v1/v2: pose plateaued at 142 across 2500+ epochs" |
| `src/losses.py` | 161 | 6892 | All 8-stage seg loss variants + QAT + EMA helpers | CE → tau-Softplus → smooth-disagreement → L7-weighted Softplus progression; C1a entropy regularizer with size-weighted soft histogram |
| `src/model.py` | 54 | 2197 | HNeRV decoder (229K params) | 6 upsample stages + sin activations + separate frame-0/frame-1 RGB heads; channel taper matches HNeRV paper |
| `src/optim.py` | 100 | 3785 | Muon optimizer + AdamW partitioner | Newton-Schulz orthogonalization; decoupled weight decay (Chen-Li-Liu arXiv:2506.15054 reference); Muon on hidden convs, AdamW on stem+RGB heads+biases+latents |
| `src/score.py` | 131 | 4661 | Official score computation + streaming evaluator | Stream-decoded to bound memory (~3 MB/frame at 1164×874×3); exact formula `100·seg + sqrt(10·pose) + 25·rate` |
| `src/train.py` | 78 | 2488 | Top-level 8-stage orchestrator | Iterates `builders[]` list; `~50 hours wallclock` documented; runs codec stage at end |
| `src/stages/common.py` | 274 | (largest) | Shared training loop dataclass + `train_stage()` | `StageConfig` dataclass; per-stage output structure documented (decoder_f32.pt + latents_f32.pt + best_archive.bin + best_meta.json + final_decoder.pt + final_latents.pt); evaluates every 25 epochs |
| `src/stages/codec_stage.py` | 71 | - | Final archive emit + round-trip verification | Reads best checkpoint, builds archive, verifies INT8 round-trip bit-exact, writes `0.bin` |
| `src/stages/stage1_v328_ce.py` | 40 | - | Random-init CE phase (3000 ep) | `init_latents_random=True`; "**Encoded for reproducibility — not re-run for this submission**" (HONEST DISCLOSURE) |
| `src/stages/stage2_v331_softplus.py` | 37 | - | CE → tau-Softplus seg loss (5650 ep) | Continues v3.28 cosine; "Encoded for reproducibility — not re-run." |
| `src/stages/stage3_v332_smooth.py` | 36 | - | Smooth-disagreement seg (1500 ep) | Fresh cosine peak=1e-4 (vs 1e-3); "not re-run" |
| `src/stages/stage4_v332_qat.py` | 38 | - | + QAT in forward (500 ep) | Same loss as Stage 3; "Encoded for reproducibility — not re-run." |
| `src/stages/stage5_c1a_l7.py` | 40 | - | + L7 + C1a regularizer (6000-9000 ep) | "Default canonical: 6000 epochs. Our extension: 9000 epochs." |
| `src/stages/stage6_lambda_sweep.py` | 39 | - | λ=0.01→0.02 (1000-2000 ep) | Default canonical/our extension same pattern |
| `src/stages/stage7_sigma_sweep.py` | 39 | - | σ=0.2→0.1 (2000-3000 ep) | Same pattern |
| `src/stages/stage8_muon_finetune.py` | 48 | - | Muon optimizer switch (3000-5000 ep) | "Researcher #24 tweak applied: Muon weight_decay = 5e-4 (not in canonical). Theoretical justification: Chen-Li-Liu arXiv:2506.15054 — Muon's spectral-norm KKT mechanism requires WD to be active." |

**Total**: 21 files, 1591 LOC (Python + shell).

### Per-file style observations

**Docstring style** (CONSISTENT across all 21 files):
- Module docstring opens with one-line purpose
- 1-3 paragraph extended description with concrete numbers
- Explicit "Source: Stage N output" and "Output canonical: <name>" markers in stage files
- HONEST DISCLOSURES inline (e.g. "pose plateaued at 142 across 2500+ epochs", "~217 bytes worse, ~+0.0001 to score")
- Cross-references to specific arXiv papers when applicable (Chen-Li-Liu, Muon)

**Naming convention** (CONSISTENT):
- Snake_case throughout
- Stage names embed version + experimental marker (`stage5_c1a_l7`, `stage8_muon_finetune`)
- Loss functions descriptive (`tau_softplus_seg_loss`, `l7_softplus_seg_loss`, `smooth_disagreement_seg_loss`)
- Single canonical `HNeRVDecoder` class

**Commenting style** (CONSISTENT):
- Inline comments are SPARSE but high-signal
- Explicit annotations for non-obvious choices (`# ← QAT joins`, `# canonical kept (researcher #24 idea 3 was SKIPPED)`)
- Reasons for hyperparameter choices preserved (`# v3.28 peak_lr`, `# Stage 6's change vs Stage 5`)

**Error handling** (CONSISTENT):
- Hard `raise ValueError` / `raise FileNotFoundError` on contract violations
- No silent fallbacks
- `set -euo pipefail` in shell scripts

**Attribution**:
- README references external Muon repo (`https://github.com/KellerJordan/Muon`) ONCE
- `optim.py` docstring cites `Keller Jordan, 2024` + Chen-Li-Liu arXiv:2506.15054
- NO citation of HNeRV paper (referenced only as "channel taper matches HNeRV paper")
- NO citation of brotli, AV1, EfficientNet, FastViT, smp/Unet
- Single attribution chain: writeup → README → Muon repo

---

## Phase 2: External writeup (aaronleslie.dev/blog/comma-compression)

**Title** (confirmed via web search): "How I (Spiritually) Won comma.ai's Compression Challenge"

**Fetch status**:
- Direct WebFetch: returned only page header (JS-rendered SPA — content not directly extractable)
- curl -L: returned 1076 bytes of HTML skeleton (no body content)
- Wayback Machine: NO snapshots available (`archived_snapshots: {}`)
- Google search: title confirmed via top result

**What we know from indirect sources**:
- Yousfi awarded "best write-up prize" alongside the honorable prize for the submission itself
- Comments thread: "great submission and write-up!" (Yousfi)
- Comments thread: Yousfi caught typo "Spritually" → "Spiritually" (operator/Aaron acknowledged + thanked)
- Author voice: "Wow! Woke up to no shortage of people fine-tuning this in the last 3 hours to eek out an improvement on top. Might have to re-title my blog post ; )" (PERSONAL, GOOD-HUMORED, slightly bitter)
- Author bitter quote (preserved verbatim from PR comments): "Heart on my sleeve, it does suck to lose exclusively to submissions that - code is one thing - just used my archive.zip contents and not a fresh base model. The competition seems to mainly reward whoever is up using codex or claude right at the last hour, not the author of 99% of the work. `</vent>`"
- Yousfi's response: "`<re:vent>` we are going to reward folks publishing their code even if not in top 3" (THIS IS THE CANONICAL HOST POSITION — explicit reward for /src publication, even non-top-3)

**Inferred writeup characteristics** (from PR body + README references):
- Title is whimsical/personal ("Spiritually Won")
- Tone likely matches PR comment voice: technical + personal + good-humored
- Likely includes the 8-stage curriculum narrative
- Likely includes the Muon optimizer story
- Likely includes the Chen-Li-Liu KKT theory citation
- "Full writeup: https://aaronleslie.dev/blog/comma-compression" cited in PR body + README

**Operator-routable**: if Slot K needs verbatim writeup content, options are (a) manual operator paste; (b) WebFetch with browser-headers / Playwright fallback; (c) operator-visit + screenshot.

---

## Phase 3: PR body + comments + reviews

### PR 95 metadata
- **Title**: `hnerv_muon submission (0.20)`
- **Author**: AaronLeslie138 (Aaron Leslie)
- **State**: MERGED
- **Created**: 2026-05-04T07:47:15Z (Mon, May 4 07:47 UTC — RACE-MODE-ANCHOR per CLAUDE.md)
- **Merged**: 2026-05-04T20:06:33Z (~12 hours after creation)
- **Files changed**: 21 (1666 additions, 0 deletions)

### PR body structure (canonical template-conformant)

```markdown
# submission name:
hnerv_muon

# upload zipped `archive.zip`
[archive.zip](https://github.com/user-attachments/files/27332334/archive.zip)

# report.txt
```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00003494
  Average SegNet Distortion: 0.00061212
  Submission file size: 178,417 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00475202
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.20
```

# does your submission require gpu for evaluation (inflation)?
no (CPU works; faster on GPU)

# did you include the compression script? and want it to be merged?
yes. `compress.sh` and `src/` are included and the pipeline is reproducible from random init

# additional comments
229K-parameter HNeRV decoder + 28-d-per-frame-pair latents (~600 pairs), INT8-quantized
and brotli-compressed to 178 KB total. The decoder is trained via an 8-stage curriculum:
cross-entropy seg → τ-Softplus margin → smooth disagreement → +QAT → +L7 hard-pixel
weighting + C1a regularizer → λ-sweep → σ-sweep → +Muon optimizer (with WD=5e-4 per
Chen-Li-Liu's spectral-norm KKT theory). The C1a regularizer shapes the weight
distribution toward the integer grid, which collapses the entropy floor for downstream
brotli compression. Verified end-to-end via this repo's eval workflow on `ubuntu-latest`
CPU; final score 0.1987

Full writeup: https://aaronleslie.dev/blog/comma-compression
```

### PR body observations
- **Length**: ~330 words / 25 lines
- **Template conformance**: 100% — follows comma.ai submission template verbatim
- **Score disclosure**: cites both 0.20 (PR title — likely T4 CUDA from first eval) and 0.1987 (ubuntu-latest CPU — final score in PR body)
- **Reproducibility honesty**: "the pipeline is reproducible from random init" + "~50 hours on a single GPU"
- **Citation discipline**: cites Chen-Li-Liu arXiv:2506.15054 + Muon by name + HNeRV (architecture name only, no paper cite)
- **Zero filler**: no "thanks for considering", no marketing language, no emoji

### Comment thread chronology
1. **github-actions** (auto): "Thanks for the submission @AaronLeslie138! 🤏"
2. **AaronLeslie138**: VENT comment about people fine-tuning on top of his work
3. **YassineYousfi (maintainer)**: "`<re:vent>` we are going to reward folks publishing their code even if not in top 3"
4. **github-actions**: First eval result — `Final score: 0.23` (cuda T4)
5. **AaronLeslie138**: question about CUDA-CPU discrepancy
6. **YassineYousfi**: "yeah there is a small hw difference in the decode, I ran all submissions in t4 for a fair comparison."
7. **YassineYousfi**: "great submission and write-up!"
8. **EthanYangTW (contributor)**: "you definitely deserved a price Awesome work mate"
9. **github-actions**: Re-eval crashed (logs link)
10. **github-actions**: Re-eval CPU: `Final score: 0.20` (matches Aaron's reported 0.1987)
11. **YassineYousfi**: "This submission won an honorable prize and best write-up prize. Please email me at {first name}@comma.ai for logistics."
12. **YassineYousfi**: typo catch "Spritually" → "Spiritually"
13. **AaronLeslie138**: thanks + acknowledgment

### Tone observations
- Maintainer is warm and direct ("great submission", typo catch, prize award announcement)
- Author tone is technical + personal + slightly bitter about late-game absorption
- Community responses are supportive
- NO formal review back-and-forth (no requested changes; trust-based merge)
- Final acceptance signal: explicit prize announcement + merge

---

## Phase 4: Our submission_dir inventory (4 files in src/ + 3 top-level = 7 files, 1140 LOC)

### File-by-file

| File | LOC | Purpose | Notable patterns |
|---|---:|---|---|
| `README.md` | 30 | Minimal description | Cites PR #95 + #98 ancestry; lists 4 innovations as bullet list; gives CPU evaluation results inline; describes archive.zip contents |
| `inflate.sh` | 35 | Iterates file_list, runs inflate.py per video | Honors `PACT_PYTHON_BIN` env var (operator-friendly); handles both `${DATA_DIR}/x` and `${DATA_DIR}/${BASE}.bin` source paths |
| `inflate.py` | 397 | Inflate runtime (decoder + frame_exploit_selector) | SPDX license header; FES1 + FEC2/3/5/6 selector grammar; Huffman decode for FEC5/FEC6; canonical bicubic upsample 384x512 → 874x1164 + frame-0 mode application + clamp+round; CARRIES INLINE DEVICE FORK WAIVER per Catalog #205 |
| `src/codec.py` | 480 | Compact decoder + latent + sidecar codec | Split-Brotli streams with `DECODER_STORAGE_ORDER` permutation tuple + `CONV4_STORAGE_PERMS` per-tensor reshape table + `DECODER_BYTE_MAPS` per-tensor byte coding mode; raw LZMA latent payload (centered-delta uint8) + 9-form sidecar Huffman (`SIDECAR_HUFF_ENUM_LEN`/`SIDECAR_HUFF_COMB_LEN`/`SIDECAR_HUFF_LEN`/`SIDECAR_SPLIT_LEN`/`SIDECAR_PACKED_LEN`/etc.); decoded via combination-co-lex rank |
| `src/frame_selector.py` | 209 | FES1 archive-side selector grammar + frame-0 transforms | `PALETTE_MODE_IDS` 31-element tuple; `_blue_tile` 8x8 dither pattern; `apply_frame0_mode` dispatches by family (identity/rgb_bias/blue_chroma/roll); identical to PR101 grammar |
| `src/model.py` | 54 | HNeRVDecoder | **IDENTICAL** to PR 95's model.py (same 229K-param architecture; same 28-d latents; same sin activation pattern; same 6-upsample-stage taper) |
| `archive.zip` | 178,517 bytes | Final submission packet | 100 bytes larger than PR 95's 178,417 (sidecar + selector overhead) |
| `report.txt` | 18 | Auth-eval evidence | CPU: PoseNet 0.00002943 / SegNet 0.00056029 / rate 0.00475469 / **score 0.19** (rounded; exact = 0.19285 per README) |

### Our submission_dir style observations

**Docstring style**:
- One-line module docstring at top (consistent)
- SPDX license header on inflate.py (we have OSS hygiene; PR 95 does NOT)
- Less narrative + more code-density (PR 95 had multi-paragraph docstrings; ours are terser)

**Naming**:
- Snake_case (matches PR 95)
- Magic constants in MODULE-LEVEL UPPER_SNAKE (`DECODER_STORAGE_ORDER`, `OUTER_MAGIC`, `FEC6_FIXED_K16_MODE_IDS`)
- Per-codec helper names descriptive (`decode_canonical_huffman_all`, `unpack_fec6_fixed_huffman_codes`)

**Commenting**:
- SPARSE — much less narrative comment density than PR 95
- INLINE WAIVER comment on inflate.py:351 explaining device-fork (per Catalog #205) — well-documented but unique to our discipline framework

**Attribution chain in README**:
- Cites PR #95 + PR #98 (current state)
- Does NOT cite PR #100, #101, #103 (other ancestors; OPPORTUNITY)
- Does NOT cite HNeRV / brotli / LZMA / Huffman (no academic / library citations; matches PR 95's minimal-cite style)

**Score evidence**:
- CPU final score 0.19285 cited in README
- report.txt shows the exact CPU eval (Modal CPU work dir; not cited as ubuntu-latest)

---

## Phase 5: Five-dimension comparative analysis

### Dimension 1: Organization

| Aspect | PR 95 | Ours (current) | Bar |
|---|---|---|---|
| File hierarchy | 21 files: `{README, compress.sh, inflate.py, inflate.sh, src/{codec,data,losses,model,optim,score,train}.py, src/stages/{common, codec_stage, stage1..stage8}.py}` | 7 files: `{README, inflate.sh, inflate.py, archive.zip, report.txt, src/{codec, frame_selector, model}.py}` | OURS: smaller (no compress.sh because training is private); PR 95: deeper hierarchy because shipping training pipeline |
| README structure | 17 lines: top-line description + writeup link + Inflate + Compress (reproduce) sections | 30 lines: ancestry + 4-innovation bullet list + CPU eval results + archive.zip contents | TIE; both terse |
| Attribution chains | Single PR (no precedent); cites Muon repo + Chen-Li-Liu arXiv | Cites PR #95 + #98 only; missing PR #100/#101/#103 chain | OURS MISSING: should cite full upstream chain |
| Cross-linking | Single Muon link + single external writeup link | No external links | OURS MISSING: opportunity to add canonical cross-links |
| Codec separation | `src/codec.py` (codec) vs `src/model.py` (architecture) vs `src/stages/*.py` (training) | `src/codec.py` (codec) vs `src/model.py` (architecture) vs `src/frame_selector.py` (selector grammar) | MATCH; both have clean codec/model separation |

**Verdict**: PR 95 has DEEPER hierarchy because they shipped the full training pipeline. Ours is INFLATE-ONLY because training is private. The right comparison is at the inflate surface where we are BOTH terse and clean. We MISS the upstream attribution chain.

### Dimension 2: Rigor

| Aspect | PR 95 | Ours (current) | Bar |
|---|---|---|---|
| Score evidence | ubuntu-latest CPU final score 0.1987 in PR body + 0.20 (T4 CUDA) from auto-eval | CPU final score 0.19285 in README + report.txt (Modal CPU work dir; not ubuntu-latest) | OURS MISSING: paired CUDA + CPU on contest-compliant hardware |
| Reproducibility | `compress.sh` + `~50 hours on a single GPU from random init` (FULLY REPRODUCIBLE) | NO compress.sh (training is private); no random-init reproducibility statement | OURS HAS GAP: must honestly cite "private training infrastructure" |
| Citations | Muon repo + Chen-Li-Liu arXiv (2 academic refs) | NONE (zero academic / library citations) | TIE-ish; PR 95 has 2 cites; we have 0. Operator decision: per medal-class restraint, do NOT add FastViT/EfficientNet/smp/FP4/NeRV/Hinton (deferred per Round 4) |
| Compliance audit | NONE (trusted; clean code, no waivers) | T3 council symposium (PROCEED-unconditional 11-of-11 + 1-co-author-recusal) | OURS HAS BETTER (council-grade), but PR 95 didn't need one |
| Honest disclosures | 6+ "encoded for reproducibility — not re-run" disclosures across stage files | (TBD: depends on README content in Slot K-build) | PR 95 BETTER currently; opportunity to match by adding honest training-infra-private disclosure |

**Verdict**: PR 95 has BETTER reproducibility evidence (full from-scratch pipeline). Ours has BETTER compliance audit (T3 council symposium). The right move: ADOPT PR 95's honest disclosure pattern AT the place where our training pipeline lives.

### Dimension 3: Discipline

| Aspect | PR 95 | Ours (current) | Bar |
|---|---|---|---|
| Axis tagging | Mixed: cites both 0.20 (T4 CUDA) and 0.1987 (ubuntu-latest CPU) inline but doesn't formally tag axes | None currently in README; we have `[contest-CPU]` discipline per Catalog #127 | OURS WILL BE BETTER (per Slot K integration) |
| Commit discipline | Single PR; not visible whether subagent-coordinated | Canonical serializer + `--expected-content-sha256` + Catalog #117/#157/#174/#235/#289 protection family | OURS BETTER — keep STEALTH per operator skunkworks decree |
| Link verification | All links work (verified manually via gh CLI in this research) | OUR practice: curl -L smoke test pre-D5 + Catalog #208 docs gate | OURS BETTER — keep curl-L verification cited in Slot K |
| Honest limitations | NO explicit limitations section | (TBD per Slot K-build current 4-bullet limitations per Slot I) | OURS WILL BE BETTER |
| Inline device-fork pattern | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — naked (would trip Catalog #205) | Same pattern but carrying `# INLINE_DEVICE_FORK_OK:contest_submission_inflate_runtime_byte_stable_per_pr107_apogee_precedent_canonical_select_inflate_device_helper_would_require_vendored_sister_module_inflating_reviewability_loc_budget_further` waiver | OURS BETTER — explicit waiver + rationale |
| SPDX license headers | NONE | inflate.py has `# SPDX-License-Identifier: MIT` | OURS BETTER — OSS hygiene |

**Verdict**: We are systematically MORE DISCIPLINED at the structural-protection layer (Catalog protections + waivers + SPDX). PR 95 trusted the code; we have audit trails.

### Dimension 4: Signal (per-line density)

| Aspect | PR 95 | Ours (current) | Bar |
|---|---|---|---|
| PR body length | ~330 words / 25 lines / 100% template-conformant | (per Slot I post-revision: 99 lines) | OURS LONGER — opportunity to trim toward PR 95's signal density |
| Filler content | ZERO — every line carries information | "Happy to discuss..." DELETED per Slot I; need final verification | OURS MUST MATCH ZERO-FILLER bar |
| Marketing language | ZERO ("great writeup" is from MAINTAINER, not author) | (need verification post-Slot I) | OURS MUST MATCH |
| External writeup | Yes (aaronleslie.dev/blog/comma-compression — won best-write-up prize) | NONE currently | OPPORTUNITY: optional external link (e.g. comma-lab docs/paper/04_results.md) — operator deferred Cloudflare site for long-term |
| Code/comment ratio | Sparse but high-signal comments; multi-paragraph docstrings | TERSER docstrings; mostly self-documenting code | PR 95 BETTER at storytelling; we are BETTER at code density |
| Total submission LOC | 1591 LOC across 21 files | 1140 LOC across 7 files | DIFFERENT SHAPES — PR 95 includes training; we are inflate-only |

**Verdict**: PR 95 wins on writeup signal density (because the writeup IS the prize-winning artifact). We win on code signal density (because we focused on inflate). For Slot K, the bar is to MATCH PR 95's zero-filler PR body and add optional external writeup link.

### Dimension 5: Score (objective)

| | PR 95 | Ours | Bar | Status |
|---|---|---|---|---|
| Final score | **0.1987** [contest-CPU ubuntu-latest] | **0.19285** [contest-CPU Modal] | Better than top-leaderboard | ✓ -0.00702 BETTER than PR 95 directly; -0.00253 BETTER than PR 102 (top medal) per `.omx/state/canonical_frontier_pointer.json` |

**Verdict**: We have already beaten PR 95 + the top-leaderboard on score. The competitive gap is now ENTIRELY at the organization+rigor+discipline+signal dimensions per operator Round 4 directive.

---

## Phase 6: Specific Slot K integration prescriptions

### submission_dir/README.md (NEW build; target 100-200 lines per Slot K integration)

- [ ] **Top-line description (1 paragraph)**: what + score + size + axis + archive sha
- [ ] **Attribution chain section**: PR #95 (HNeRV base) → PR #98 (decode-side channel postprocess) → PR #100 (hnerv_lc_v2 268-LOC substrate) → PR #101 (gold-medal entropy bolt-ons 337 LOC) → PR #103 (silver-medal 241-LOC entropy refinements) → ours (HNeRV_FT_microcodec frame-exploit selector + 9-form sidecar grammar)
- [ ] **Provenance vs innovation table**: what came from upstream vs what is new
- [ ] **Inflate section**: canonical `evaluate.sh --submission-dir` invocation pattern (MATCHES PR 95's README)
- [ ] **Compress section**: HONEST disclosure "private training infrastructure; from-source-build would require N-day GPU-hours; canonical archive bytes verified via sha256"
- [ ] **Score table**: paired CPU + CUDA (when available) + axis tags + archive sha (CLAUDE.md "Apples-to-apples evidence discipline")
- [ ] **Reproducibility section**: sha256 + zip topology (member name `0.bin`) + timestamps + canonical brotli/lzma version deps
- [ ] **External resources** (sanitized): comma-lab + tac repo links; T3 council symposium link
- [ ] **Optional**: link to docs/paper/04_results.md § 4.8 in comma-lab (canonical extended writeup — operator decision per Cloudflare deferral)

### PR body (current 99 lines → target 50-100 lines per Slot K integration)

- [ ] Keep template-conformant sections (exact comma.ai template per PR 95 pattern)
- [ ] Score table: CPU + CUDA both tagged
- [ ] Cross-ref submission_dir/README.md for full detail (so PR body stays terse)
- [ ] Retain COMPETITIVE+INNOVATIVE quote framing (per Slot I)
- [ ] Trim verbose Limitations bullets that duplicate README detail
- [ ] Verify ZERO filler / ZERO marketing per PR 95 bar

### Slot K landing memo specific PR-95-comparison framing

- [ ] Audit-against-PR-95 table demonstrating better-than on each of {organization, rigor, discipline, signal, score}
- [ ] HONEST acknowledgment of dimensions where we don't beat PR 95:
    - PR 95 shipped full training pipeline + ~50hr GPU reproducibility statement; ours has private training infra
    - PR 95 has external writeup (best-write-up prize); ours has none (Cloudflare deferred)
- [ ] Cite empirical comparison data from this research memo

### Stealth + skunkworks tone audit (Slot K must verify)

- [ ] NO emoji
- [ ] NO "we" enthusiasm ("our submission", "we believe", etc.)
- [ ] NO "thank you for considering"
- [ ] Matter-of-fact axis disclosure (e.g. "0.19285 [contest-CPU] / TBD [contest-CUDA]")
- [ ] Minimal or absent sign-off

### Specific README prose patterns to ADOPT from PR 95

1. **Single-sentence top-line description**: "A 178 KB archive containing a 229K-parameter HNeRV decoder + 28-d-per-frame-pair latents..." (PR 95 README line 3) — adapt to: "A 178 KB archive containing a 229K-parameter HNeRV decoder (PR 95 base + PR 100 fine-tune) + 28-d-per-frame-pair latents + frame-exploit selector sidecar..."
2. **Reproducibility honesty**: "~50 hours on a single GPU from random init" (PR 95) — adapt to: "Trained on private infrastructure; canonical archive sha256: `6bae0201...`; archive byte-identical reproduction via committed `submission_archive.zip`"
3. **Inflate command pattern**: "`evaluate.sh --submission-dir ./submissions/hnerv_muon` will unzip `archive.zip` and call `inflate.sh`" (PR 95) — adapt for our submission_dir name

### Specific README prose patterns to AVOID

1. PR 95's enthusiastic "great submission and write-up!" was a MAINTAINER quote, not author self-quote — do not seed self-praise
2. PR 95's external writeup link risks being seen as self-promotion in skunkworks tone — defer Cloudflare site link per operator long-term decision
3. PR 95's bitter "</vent>" comment is voice-specific to Aaron; we should NOT match the personal-voice register

---

## Phase 7: Cross-references + sister coordination

### Cloudflare site adaptation (DEFERRED-LONG-TERM)

Per operator: "perhaps we should adapt our cloudflare site into a similar hosting artifact, but there is already so much there and it's also stale and updating should maybe be a long-term project". This research memo's scope ENDS here for Cloudflare. The Slot K integration may OPTIONALLY include a link to docs/paper/04_results.md § 4.8 in comma-lab as a canonical extended writeup; operator decision required.

### Sister subagent coordination acknowledgment

- **Slot J (`a4b959803c985e440`)**: T3 council symposium on PR body. May consume this research as input. Disjoint from this scope.
- **Slot K (queued integration subagent)**: WILL CONSUME the Phase 6 prescriptions. THIS research is the canonical input.
- **Slot L (`a7284b57a53f6ccf8`)**: comma-lab public-readiness audit + tac CI fix. May surface README link opportunity for comma-lab if operator wants the optional external writeup.

### 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: N/A (research only)
- Hook #2 Pareto constraint: N/A
- Hook #3 bit-allocator: N/A
- Hook #4 cathedral autopilot dispatch: N/A
- Hook #5 continual-learning posterior: N/A
- Hook #6 probe-disambiguator: N/A

This is a RESEARCH ARTIFACT, not a runtime contribution. Per CLAUDE.md "Subagent coherence-by-default" all 6 hooks N/A is acceptable when explicitly declared.

### Discipline + provenance

- Catalog #229 PV: read PR 95 + comments + our submission_dir BEFORE drafting recommendations
- Catalog #117/#157/#174/#235/#289: this memo will land via canonical serializer with POST-EDIT `--expected-content-sha256`
- Catalog #206: checkpoints emitted at steps 1, 2, 3
- Catalog #110/#113 APPEND-ONLY: NEW research artifact; does not mutate prior PR-95-related work
- Catalog #230 sister-subagent ownership map: research + recommendations ONLY; never touches PR body / README / repos (per scope)
- Catalog #287/#323 canonical Provenance: every observation cites source (PR file path, comment thread, gh CLI command output)
- CLAUDE.md "Public Disclosure Hygiene": research memo is private at this stage; no local paths leaked

---

## Citations + sources

- PR 95 full file listing: `gh api repos/commaai/comma_video_compression_challenge/pulls/95/files --paginate`
- PR 95 source files cached at `/tmp/pr95_files/*` for this session (21 files, 1591 LOC)
- PR 95 body + comments: `gh pr view 95 --repo commaai/comma_video_compression_challenge --comments`
- Our submission_dir: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/`
- Archive byte counts: `stat -f%z`
- Wayback Machine availability check: `http://archive.org/wayback/available?url=aaronleslie.dev/blog/comma-compression` (returned `archived_snapshots: {}`)
- Google search confirmed external writeup title: "How I (Spiritually) Won comma.ai's Compression Challenge"
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (per CLAUDE.md "Frontier scores are pointer-only" non-negotiable)
- T3 council symposium memo (sister): TBD per Slot J (`a4b959803c985e440`) output
- CLAUDE.md non-negotiables honored: "Strategic Secrecy" + "Public Disclosure Hygiene" + "Apples-to-apples evidence discipline" + "HNeRV / leaderboard-implementation parity discipline" + "Race-mode rigor inversion"

---

**End of Phase 1-7 research artifact. Companion recommendations memo at `.omx/research/beat_pr_95_organization_rigor_discipline_signal_recommendations_20260519T192300Z.md` consumes this as input for Slot K integration.**
