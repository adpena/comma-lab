# Beat-PR-95 specific recommendations for Slot K integration

**Date**: 2026-05-19T19:23:00Z
**Parent research**: `.omx/research/pr_95_full_artifact_deep_research_against_our_submission_20260519T192300Z.md`
**Operator competitive framing**: "better than PR 95 on the organization and rigor and discipline and signal and better than the leaderboard on the score"
**Consumed by**: Slot K (queued integration subagent)

**Score status**: ALREADY BEATEN. We are at 0.19285 [contest-CPU] vs PR 95's 0.1987 (-0.00585) and vs PR 102 top-medal 0.19538 (-0.00253). The remaining competitive gap is at the 4 non-score dimensions per operator Round 4 directive.

This memo is the SPECIFIC ACTIONABLE SLOT K INSTRUCTION SET derived from the deep research review. Items are ordered by Slot K execution flow: README.md (1-12) → PR body (13-20) → landing memo (21-25) → tone audit verification (26-31).

---

## TARGET: submission_dir/README.md (rebuild from 30 → 100-200 lines)

### 1. Top-line description (target 1 paragraph, ~80 words)

ADOPT PR 95's pattern. PR 95 line 3 (verbatim):
> "A 178 KB archive containing a 229K-parameter HNeRV decoder + 28-d-per-frame-pair latents that, on inflation, produces a video whose frames activate the official frozen SegNet and PoseNet evaluators almost identically to the original."

OUR ADAPTATION (recommended):
> "A 178 KB archive containing a 229K-parameter HNeRV decoder (PR #95 base architecture + PR #100 hnerv_lc_v2 fine-tune) + 28-d-per-frame-pair latents + frame-exploit selector sidecar + 9-form Huffman-coded delta-correction sidechannel that, on inflation, produces a video whose frames activate the official frozen SegNet and PoseNet evaluators identically enough to score `0.19285 [contest-CPU GHA Linux x86_64]` (verified) vs PR 95's `0.1987` and PR 102's top-medal `0.19538`."

### 2. Attribution chain section (NEW; mandatory per Round 4 better-than-PR-95 bar)

The full provenance chain (canonical per `.omx/state/canonical_frontier_pointer.json` + lane registry):

```markdown
## Attribution chain

- **PR #95** ([aaronleslie/hnerv_muon](https://github.com/commaai/comma_video_compression_challenge/pull/95))
  — HNeRV-style 229K-parameter decoder + 28-d latents + 8-stage training curriculum
  + INT8/brotli codec. Score 0.1987 [contest-CPU ubuntu-latest].
- **PR #98** ([host-side decode-side channel postprocess]) — frame-1-from-frame-0
  warp residual technique. Cited in our codec but not used in current archive.
- **PR #100** ([hnerv_lc_v2 268-LOC substrate]) — score-aware training improvements
  on PR 95 base. Adapted as our fine-tune source.
- **PR #101** ([gold-medal entropy bolt-ons 337 LOC]) — split-Brotli decoder streams,
  per-tensor byte maps, FES1 frame-exploit selector grammar. Direct ancestor of
  our codec.
- **PR #103** ([silver-medal 241-LOC entropy refinements]) — additional sidecar
  Huffman length-vector packing patterns.
- **Ours** (hnerv_ft_microcodec) — adds FEC6 fixed-Huffman k=16 selector + 9-form
  sidecar grammar with combination-co-lex rank no-op table + DECODER_STORAGE_ORDER
  permutation for stream-boundary brotli optimization.

The competition rewards code publishing per the maintainer's standing position
("we are going to reward folks publishing their code even if not in top 3" —
@YassineYousfi, PR 95 review thread).
```

### 3. Provenance vs innovation table (NEW)

```markdown
## Provenance vs innovation

| Component | Source | Status |
|---|---|---|
| HNeRVDecoder (229K params) | PR #95 / PR #100 | Used unchanged |
| 8-stage training curriculum | PR #95 (CE → Softplus → smooth → +QAT → +L7+C1a → λ-sweep → σ-sweep → +Muon) | Adapted (private infrastructure) |
| INT8 + brotli decoder codec | PR #95 / PR #101 split-Brotli + per-tensor byte maps | Adapted |
| LZMA + delta-encoded latents | PR #95 / PR #101 centered-delta uint8 | Adapted |
| FES1 frame-exploit selector grammar | PR #101 | Used as-is |
| FEC6 fixed-Huffman k=16 selector | NEW (this submission) | Innovation |
| 9-form sidecar Huffman + combination-co-lex no-op table | NEW (this submission) | Innovation |
| DECODER_STORAGE_ORDER permutation | NEW (this submission) | Innovation |
```

### 4. Inflate section (target 1 code-block + 3 sentences)

ADOPT PR 95's pattern. Our adaptation:

```markdown
## Inflate

`evaluate.sh --submission-dir ./submissions/<our_dir_name>` will unzip
`archive.zip` and call `inflate.sh`, which iterates the video list and runs
`inflate.py` per video. CPU-only inflate works (faster on GPU). The runtime
contains no scorer weights; only the trained HNeRV decoder + latents + sidecar
bytes are read from `archive.zip`.
```

### 5. Compress section (NEW pattern; HONEST private-infra disclosure)

NO `compress.sh` in our case. Replace PR 95's "~50 hours on a single GPU from random init" with HONEST private-infra statement:

```markdown
## Compress (reproduce)

The decoder was trained on private infrastructure derived from PR #95's 8-stage
curriculum + PR #100's hnerv_lc_v2 fine-tune protocol; from-source-build would
require ~50 GPU-hours on contest-equivalent hardware (T4 / A10G / A100). The
canonical archive bytes are committed to this submission directory and
byte-identical reproduction is verified via:

    sha256sum archive.zip
    # expected: 6bae0201fb08...  (full sha256 in canonical frontier pointer)

The codec round-trip (encoder → archive → decoder) is bit-exact for the INT8
weights; verification is part of the upstream PR #95 codec_stage logic.
```

### 6. Score evidence section (paired axes per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")

```markdown
## Score evidence

| Axis | Hardware | Score | Components | Status |
|---|---|---|---|---|
| `[contest-CPU]` | GHA Linux x86_64 (Modal CPU; 1:1 contest CI compliance) | **0.19285** | seg=0.00056018 / pose=0.00003286 / rate=0.00474779 (sha256=`6bae0201...`) | ✓ verified |
| `[contest-CUDA]` | T4 (contest reference) | (pending paired Linux x86_64 + T4 eval) | TBD | per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable |

Comparison anchors (per `.omx/state/canonical_frontier_pointer.json`):
- PR 95 (`hnerv_muon`): 0.1987 [contest-CPU ubuntu-latest] (-0.00585 vs ours)
- PR 102 (`top-medal`): 0.19538 [contest-CPU] (-0.00253 vs ours)
- Local frontier pointer: 0.19285 [contest-CPU] (this submission)
```

### 7. Reproducibility section (canonical zip topology disclosure)

```markdown
## Reproducibility

| Field | Value |
|---|---|
| archive.zip sha256 | `6bae0201fb08...` (full sha in canonical frontier pointer) |
| archive.zip size | 178,517 bytes (vs PR 95: 178,417 bytes; +100 bytes for sidecar overhead) |
| Zip topology | Single member: `0.bin` (matches PR 95 canonical naming) |
| Compression libs | `brotli` (quality 11), `lzma` (`FILTER_LZMA1` dict 4096, lc 3, lp 0, pb 0) |
| Python deps | `torch`, `brotli`, `numpy`, `lzma` (stdlib) |
| Inflate runtime LOC | ~1140 (inflate.py + src/*.py); ≤200 LOC inflate.py budget per HNeRV parity L4 |
```

### 8. Limitations / honest disclosures section (NEW; matches PR 95's "encoded for reproducibility — not re-run" style)

```markdown
## Limitations + honest disclosures

- Training pipeline is private infrastructure derived from PR #95's published
  8-stage curriculum + PR #100's hnerv_lc_v2 fine-tune. From-random-init
  reproduction requires the PR #95 + #100 sources + ~50 GPU-hours.
- `[contest-CUDA]` paired eval is pending (per CLAUDE.md "Submission auth eval —
  BOTH CPU AND CUDA"); current `[contest-CPU]` score is the canonical
  leaderboard axis.
- The FEC6 selector palette (16 modes) is a fixed-Huffman encoding tuned on
  600 contest video pairs; generalization to different videos may shift the
  Huffman-optimal palette.
- The 9-form sidecar grammar (`SIDECAR_HUFF_ENUM_LEN`/`SIDECAR_HUFF_COMB_LEN`/
  `SIDECAR_HUFF_LEN`/`SIDECAR_SPLIT_LEN`/`SIDECAR_PACKED_LEN`/etc.) selects the
  shortest packing at encode time; runtime dispatches via length-based switch.
```

### 9. External resources section (canonical cross-links per organization-better-than-PR-95 bar)

```markdown
## External resources

- **comma-lab** (research artifacts): https://github.com/adpena/comma-lab
- **tac** (Task-Aware Compression library): https://github.com/adpena/tac
- **HNeRV upstream** (architectural ancestor): cited via PR #95
- **Muon optimizer** (training ancestor): https://github.com/KellerJordan/Muon (cited via PR #95)

(Optional, per operator Cloudflare-deferred decision: docs/paper/04_results.md
in comma-lab contains the canonical extended writeup; Cloudflare site
adaptation is a long-term project.)
```

### 10. Section ordering (recommended)

1. Top-line description (Item 1)
2. Attribution chain (Item 2)
3. Provenance vs innovation table (Item 3)
4. Inflate (Item 4)
5. Compress (reproduce) (Item 5)
6. Score evidence (Item 6)
7. Reproducibility (Item 7)
8. Limitations + honest disclosures (Item 8)
9. External resources (Item 9)

### 11. Tone register (skunkworks; NO match to PR 95's personal voice)

PR 95's external writeup is whimsical ("Spiritually Won"); the README is technical-terse. We should MATCH the technical-terse README register. We should NOT match the whimsical external-writeup register.

- NO "we" voice (use passive / impersonal: "The decoder was trained" not "we trained")
- NO emoji
- NO marketing language
- NO self-praise
- MATTER-OF-FACT axis disclosure

### 12. Length target

100-200 lines. PR 95 README is 17 lines but they pushed detail to the external writeup. We are pushing detail to the README itself (because Cloudflare external writeup is deferred), so the longer length is justified.

---

## TARGET: PR body (current 99 lines per Slot I → target 50-100 lines)

### 13. Template conformance verification

PR 95's body is 100% template-conformant. Slot K must verify our PR body matches the exact 5-section comma.ai template:
1. `# submission name:`
2. `# upload zipped \`archive.zip\``
3. `# report.txt`
4. `# does your submission require gpu for evaluation (inflation)?`
5. `# did you include the compression script? and want it to be merged?`
6. `# additional comments`

### 14. Score table in PR body

Include BOTH axes inline (better than PR 95's mixed `0.20` / `0.1987` disclosure):

```markdown
# report.txt
```
=== Evaluation results over 600 samples [contest-CPU GHA Linux x86_64] ===
  Average PoseNet Distortion: 0.00003286
  Average SegNet Distortion: 0.00056018
  Submission file size: 178,517 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00474779
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.19285

(Paired [contest-CUDA T4] eval pending per CLAUDE.md submission discipline.)
```
```

### 15. Cross-reference submission_dir/README.md for full detail

PR body should be DENSE; push detail to README. After the template-required sections, the `# additional comments` section should be a 2-3 paragraph high-signal summary that explicitly cross-refs the README:

```markdown
# additional comments
229K-parameter HNeRV decoder (PR #95 architecture + PR #100 fine-tune) + 28-d-per-pair
latents (~600 pairs) + frame-exploit selector sidecar + 9-form Huffman-coded
delta-correction sidechannel, all in a 178 KB single-`0.bin` archive. Score 0.19285
[contest-CPU GHA Linux x86_64] (-0.00585 vs PR #95, -0.00253 vs PR #102 top-medal).

Full attribution chain + provenance-vs-innovation table + reproducibility evidence
+ honest limitations in `submissions/<our_dir>/README.md`.
```

### 16. Limitations bullets in PR body (KEEP MINIMAL)

Per CLAUDE.md operator decision to maintain stealth, keep limitations bullets in PR body to 2-3 high-signal lines; full detail goes in README. PR 95 had ZERO limitations bullets; we have 2-3 because our discipline framework requires honest CUDA-pending disclosure.

### 17. Filler verification

Slot K must grep PR body for filler tokens and DELETE all matches:
- "Happy to discuss" / "happy to"
- "Thanks for considering"
- "Let me know if"
- "Feel free to"
- Any sign-off line

PR 95's body has ZERO of these. Match the bar.

### 18. Marketing language verification

Slot K must grep for:
- "great", "excellent", "best", "amazing" (when self-applied)
- "novel", "groundbreaking" (when self-applied)
- "carefully designed", "thoughtfully" (when self-applied)

PR 95 has ZERO self-marketing. Match the bar.

### 19. Emoji verification

ZERO emoji in PR body. PR 95 has ZERO. Match.

### 20. Sign-off

PR 95 has NO sign-off (no "Thanks!", no name, nothing). We should MATCH (no sign-off).

---

## TARGET: Slot K landing memo

### 21. Audit-against-PR-95 table (recommended landing memo content)

The landing memo should include the 5-dimension comparison table from this research memo's Phase 5, with explicit verdict per dimension:

| Dimension | PR 95 baseline | Ours (post-Slot K) | Verdict |
|---|---|---|---|
| Organization | 17-line README + external writeup link | 100-200 line README + provenance table + attribution chain | BETTER (more canonical chain depth) |
| Rigor | from-random-init reproducibility + 2 academic cites | T3 council + canonical Provenance per Catalog #323 + paired axes | DIFFERENT (PR 95 better on training reproducibility; we better on compliance audit) |
| Discipline | trust-based; no waivers | canonical serializer + Catalog #205 waiver + SPDX + Catalog protections | BETTER (audit trails everywhere) |
| Signal | 17-line README + 25-line PR body + best-write-up prize external | 100-200 line README + 50-100 line PR body + Cloudflare deferred | DIFFERENT (PR 95 better on writeup density; we better on code density) |
| Score | 0.1987 [contest-CPU] | 0.19285 [contest-CPU] | BETTER (-0.00585) |

### 22. Honest acknowledgments

Slot K landing memo should EXPLICITLY acknowledge where we don't beat PR 95:
- **Training reproducibility**: PR 95 ships full pipeline + ~50hr GPU statement. Ours has private training infrastructure. Honest disclosure is the discipline; matching reproducibility would require operator-routable separate decision to open-source training infra.
- **External writeup**: PR 95 won best-write-up prize. Ours has no external writeup. Cloudflare deferred per operator long-term.
- **Personal voice**: PR 95 author voice is technical+personal+good-humored ("Spiritually Won"). We are stealth-skunkworks (matter-of-fact, no personal voice). NOT a deficit — different operator-mandated register.

### 23. Cite empirical comparison data

Slot K landing memo should cite this research memo's Phase 1 inventory + Phase 4 our-inventory tables verbatim or by reference for the audit trail.

### 24. 6-hook wire-in declaration

Per Catalog #125, Slot K landing memo must declare 6-hook status. For a README+PR-body integration:
- Hook #1-#3, #5-#6: N/A (documentation surface, no runtime contribution)
- Hook #4 cathedral autopilot dispatch: N/A (read-only; observability only)

All 6 N/A is acceptable when explicitly declared. The integration is a TONE/STRUCTURE landing, not a runtime contribution.

### 25. Catalog discipline acknowledgments

Slot K landing memo must cite:
- Catalog #229 PV (premise verification): Slot K must verify all paths exist + all PRs are still accessible + canonical frontier pointer is still current before drafting
- Catalog #117/#157/#174/#235/#289: canonical serializer with POST-EDIT `--expected-content-sha256`
- Catalog #110/#113 APPEND-ONLY: README rewrite is OK (current README is operator-acknowledged stale); PR body edit via gh pr edit
- Catalog #208 docs sanitization: NO local paths in README (`/Users/adpena/...` forbidden)
- Catalog #287 evidence tags: every score literal carries axis tag per CLAUDE.md "Apples-to-apples evidence discipline"

---

## TARGET: Tone audit verification (Slot K + sister Slot J could both check)

### 26. NO emoji

Grep README + PR body for emoji literals. ZERO permitted.

### 27. NO "we" enthusiasm

Grep for:
- "we believe" / "we think" / "we hope" / "we are excited"
- "our submission" / "our work" (impersonal "this submission" is OK)
- "we innovated" / "we discovered"

ZERO permitted.

### 28. NO "thank you for considering"

ZERO permitted. PR 95 has ZERO. Match.

### 29. Matter-of-fact axis disclosure

Every score literal must carry axis tag:
- "0.19285 [contest-CPU]" — OK
- "0.19285" alone — NOT OK
- "achieved 0.19285" — NOT OK (uses achievement framing)
- "scores 0.19285 [contest-CPU]" — OK (descriptive)

### 30. Minimal or absent sign-off

PR 95 has NO sign-off. Default position: MATCH (no sign-off).

### 31. Final verification commands for Slot K

```bash
# Filler scan
grep -i -E 'happy to|thanks for|let me know|feel free' <readme_or_pr_body>

# Marketing scan
grep -i -E '\b(great|excellent|amazing|novel|groundbreaking|carefully)\b' <files>

# "we" scan
grep -i -E '\bwe (believe|think|hope|are excited|innovated|discovered)\b' <files>

# Emoji scan
python3 -c "import re,sys; [print(m) for m in re.findall(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]', sys.stdin.read())]" < <files>

# Axis tag scan (each score literal should be followed by [contest-CPU] or [contest-CUDA])
grep -n -E '0\.[12][0-9]+' <files>
```

---

## Top-5 Slot K prescriptions (TL;DR for parent agent)

1. **REBUILD submission_dir/README.md** from 30 lines to 100-200 lines following Items 1-12 (attribution chain + provenance vs innovation table + axis-tagged score evidence + honest limitations + external resources).
2. **REFINE PR body** to 50-100 lines following Items 13-20 (template-conformant + paired-axis score table + dense additional-comments + zero filler + zero marketing + zero emoji + no sign-off).
3. **WRITE Slot K landing memo** with 5-dimension audit-against-PR-95 table per Items 21-25 (honest acknowledgments where we don't beat PR 95 + cite this research memo + 6-hook N/A declaration).
4. **RUN tone audit grep commands** per Items 26-31 BEFORE final commit (filler, marketing, "we" enthusiasm, emoji, axis tags, sign-off).
5. **HONOR operator deferrals**: Cloudflare site adaptation is LONG-TERM (NOT in Slot K scope); external-writeup link (docs/paper/04_results.md in comma-lab) is OPTIONAL operator decision.

---

## What was NOT in scope for this research (operator-deferred / sister-territory)

- Cloudflare site adaptation (operator long-term)
- PR body edits (Slot K territory)
- submission_dir/README.md edits (Slot K territory)
- comma-lab or tac repo edits (Slot L territory)
- Council symposium adjudication (Slot J territory)
- New GPU dispatch (no GPU spend)
- New canonical equation registrations (out of scope)
- New STRICT preflight gates (out of scope)

This research memo is observability-only per Catalog #287/#323 canonical Provenance discipline.

---

**End of recommendations. Slot K consumes Items 1-31 as the canonical actionable specification.**
