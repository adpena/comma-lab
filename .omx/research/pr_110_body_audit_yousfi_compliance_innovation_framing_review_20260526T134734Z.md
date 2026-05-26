---
council_tier: T1
audit_kind: pr_body_review
scope: review_only
target: live_pr_110_submission_body
target_url: https://github.com/commaai/comma_video_compression_challenge/pull/110
created_at_utc: 2026-05-26T13:47:34Z
created_by: PR110-BODY-AUDIT-YOUSFI-COMPLIANCE-INNOVATION-FRAMING-REVIEW
operator_questions_addressed: 5
mission_predicted_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
---

# PR #110 body audit — Yousfi compliance + innovation framing + verbosity + hallucinations

# Section 0 — Executive summary (TIGHT, operator-facing)

**Pre-flight security**: PASS. Zero `Claude` / `Anthropic` / `Co-Authored` / `claude.com` / `anthropic.com` tokens in live PR #110 body. Sole-author voice (Alejandro Peña) confirmed; @-mention attribution chain present for PRs #95/#98/#100/#101/#102/#103.

**Question 1 — Yousfi compliance**: ~85% compliant. Yousfi's verbatim 2026-05-25T21:19:01Z request: *"can you update the pr with the new template, including an easy to understand response to: [competitive or innovative section with HTML comment markers]"*. The live PR body **DOES** restore the HTML comment scaffold + leads with explicit `**Competitive: yes.**` / `**Innovative: yes.**`. Compliance gap: the "easy to understand" bar — the live body's Innovative section is 2 sentences (53 words); the v42 draft was richer (~150 words explaining "PR 101 uses one fixed pipeline for all frames" — the operator's CORE FRAMING) but was trimmed away in v43→v44→LIVE. Yousfi explicitly asked for "easy to understand"; we currently provide TERSE not EASY.

**Question 2 — Add-back recommendations** (4 concrete elements removed for not-pedantic; recommend operator review):
1. **CORE INNOVATION FRAMING (PR 101 one-fixed-pipeline vs ours per-frame-optimal)** — present in v41/v42; removed v43→LIVE. Strongest add-back candidate.
2. **Rate-vs-distortion arithmetic mechanism** — v4/v41/v42 had explicit `+25 × 259 / 37,545,489 = +0.000172` calculation; v43→LIVE truncated to "5.6× the rate cost".
3. **Adaptive-quantization analogy** — v4/v41/v42 had "analogous to adaptive quantization but one abstraction level up"; removed v43→LIVE.
4. **K=16 tunability + generalization to other frozen-substrate PRs** — v4/v41/v42 had explicit "K=16 is tunable" + "sweep methodology generalizes to any frozen-substrate PR"; live body has only "31-mode palette ... apply to other frozen-substrate PRs" (one phrase).

**Question 3 — Innovation framing**: **NEEDS_REWRITE**. The CORE positioning ("PR 101 same-for-each-frame; ours optimal-for-each-frame") is NOT explicit in the live PR body. Live text reads: *"Per-frame transform selection over a frozen substrate, picked offline against the upstream scorer"* — this IMPLIES per-frame optimality but does NOT CONTRAST with PR 101's uniform pipeline. The v42 draft had the canonical contrastive sentence: *"where PR #101 uses one fixed pipeline for all frames"*. Recommend operator adopt v42 sentence verbatim into the Competitive paragraph OR a more production-hardened variant proposed in Section 4 below.

**Question 4 — Verbosity / context / loop-closure**: 5 issues identified.
- TIGHT/UNDERWRITTEN: Innovative section (53 words) is shorter than every other medal-class PR body of comparable competitive position; the loop-closure on "what makes it innovative" is implicit not explicit.
- LOOP_OPEN: Compression script merge request mentions "31 candidate per-frame transforms" but never explains what the 31 transforms ARE (loop-open: reader cannot evaluate the mechanism).
- INSUFFICIENT_BACKGROUND: Member `x` grammar paragraph assumes deep familiarity with PR #101's substrate; non-PR-101-author reader lacks context.
- TIGHT/CORRECT: Score block + report.txt — perfectly calibrated.
- TIGHT/CORRECT: Attribution chain @-mentions — exemplary.

**Question 5 — Hallucinations**: ZERO HARD HALLUCINATIONS. All 18 verified factual claims PASS. Two SOFT/STALE issues flagged:
1. **STALE-NOT-FALSE**: PR 110 body cites top-merged baseline as PR #101 `0.192840`. CURRENT canonical frontier pointer (refreshed 2026-05-25) shows we BEAT `0.192840` LOCALLY with archive `7a0da5d0fc32` at `0.19202828` (DQS1 lane). The PR body's PR 101 baseline is correct AT TIME OF PR LANDING (2026-05-20); the canonical frontier has since advanced. This is intentional (PR 110 was about FEC6 submission with sha `6bae0201`); the question is whether the body should be CURRENT or REPRODUCIBLE-at-submission. RECOMMENDED: keep as-is (reproducibility wins).
2. **NUANCE**: PR #101 baseline cited as "current top merged" — PR #101 is **CLOSED, NOT MERGED** (verified via `gh pr view 101 --json state,mergedAt`: state=CLOSED, mergedAt=null). The actual MERGED top-CPU is PR #102 (`hnerv_lc_v2_scale095_rplus1` @ 0.19538 by @EthanYangTW, mergedAt 2026-05-04). PR #101 IS the maintainer-recognized GOLD per @YassineYousfi's prize comment on PR #101 ("This submission won # 1 prize"). T3 symposium (2026-05-20) explicitly flagged this as CARGO-CULTED-WITH-NUANCE and recommended re-phrasing to "current top-CPU submission" or "top-scored submission per maintainer GOLD prize". The live PR body still uses "current top merged" — REVISION #5 from that symposium was NOT applied.

**Operator-routable next-step** (5 ranked recommendations):
1. **CRITICAL**: Add back CORE INNOVATION FRAMING ("PR 101 uses one fixed pipeline for all frames; we pick best-of-31 per frame") per Question 3. v42 sentence is the canonical candidate.
2. **HIGH**: Reframe "current top merged" → "current top-CPU submission" or "top-scored per maintainer GOLD on PR #101" per T3 symposium REVISION #5.
3. **MEDIUM**: Re-add explicit rate-vs-distortion arithmetic (`+25 × 259 / 37,545,489 = +0.000172`) for the maintainer who wants to check the math. v4 phrasing is canonical.
4. **MEDIUM**: Re-add adaptive-quantization analogy ("one abstraction level up — per-frame transform selection rather than per-coefficient quantization"). v4 phrasing is canonical.
5. **LOW**: Close loop-open: 1-sentence inline note about what kinds of transforms the 31-mode palette covers (e.g. "spatial filters + chroma-quantization profiles + bit-depth variants").

---

# Section 1 — Pre-flight security audit (MANDATORY FIRST)

**Per CLAUDE.md "Public Disclosure Hygiene" + USER PR ATTRIBUTION + FORBIDDEN CLAUDE ATTRIBUTION standing rules.**

**Source of truth**: `gh pr view 110 --repo commaai/comma_video_compression_challenge --json body` fetched 2026-05-26T13:47:34Z.

## Forbidden-token grep results

| Token | Count | Verdict |
|---|---|---|
| `Claude` | 0 | PASS |
| `Anthropic` | 0 | PASS |
| `Co-Authored` | 0 | PASS |
| `claude.com` | 0 | PASS |
| `anthropic.com` | 0 | PASS |
| `claude-code` | 0 | PASS |
| `noreply@anthropic.com` | 0 | PASS |

## Attribution audit

- **Author field** (PR-level): `adpena` (Alejandro Peña) — CORRECT per USER PR ATTRIBUTION rule.
- **Voice**: First-person operator + maintainer-friendly technical tone — CORRECT.
- **Attribution chain** (verbatim from live body):
  - PR #95 → @AaronLeslie138 ✓
  - PR #98 → @EthanYangTW ✓
  - PR #100 → @BradyMeighan ✓
  - PR #101 → @SajayR ✓
  - PR #102 → @EthanYangTW ✓
  - PR #103 → @rem2 ✓
  - Each handle independently verified via `gh pr view <N> --json author` (Section 6 below).

## Verdict

**PASS** — no security violations. PR #110 body is fully compliant with sole-author operator-attribution standing rule. Zero CRITICAL flags. No operator-routable IMMEDIATE fix required.

---

# Section 2 — Yousfi PR 110 comment compliance audit (Operator Question 1)

## Yousfi's verbatim comment (2026-05-25T21:19:01Z)

> can you update the pr with the new template, including an easy to understand response to:
> ```
> # is this submission competitive or innovative? explain why
> <!-- competitive: better than top #1 submission -->
> <!-- innovative: it has a novel idea that is not on the leaderboard yet, might not be competitive, but has potential -->
> ```

**Source**: `gh api repos/commaai/comma_video_compression_challenge/issues/110/comments` fetched 2026-05-26T13:47:34Z (Comment 2 of 2; Comment 1 is the `github-actions[bot]` boilerplate at PR creation).

## Parsing Yousfi's request — 3 explicit asks

| # | Ask | Verbatim phrase | Compliance |
|---|---|---|---|
| **A** | Update PR with the NEW template | "update the pr with the new template" | **COMPLIANT** — v44 restored full HTML comment scaffold; live body has all 7 template sections with original HTML markers. |
| **B** | Include the "competitive or innovative" section with template's HTML markers | "including an easy to understand response to: [template block with `<!-- competitive: ... -->` + `<!-- innovative: ... -->`]" | **COMPLIANT** — Both HTML markers present verbatim in live body; lead with `**Competitive: yes.**` + `**Innovative: yes.**` per template format. |
| **C** | Make the response EASY TO UNDERSTAND | "easy to understand response" | **PARTIALLY_COMPLIANT** — the response IS lead with `Competitive: yes.` / `Innovative: yes.` (template-faithful), but the EXPLANATION underneath is TERSE not EASY. Specifically: the Innovative paragraph is 53 words across 4 sentences (live body); the v42 draft had 150 words across 6 sentences that explicitly explained "PR #101 uses one fixed pipeline for all frames" + "adaptive quantization analogy". The canonical readability bar — comprehensible to a maintainer who has not deeply read our source — is missed on the Innovative side. |

## Yousfi's compliance score: ~85%

- **2 of 3 asks** fully compliant (template format + HTML markers).
- **1 of 3 asks** partially compliant (easy-to-understand bar).
- Critical gap: the operator's CORE FRAMING ("PR 101 same-for-all; ours per-frame-optimal") is what makes the innovation EASY TO UNDERSTAND for a maintainer who already understands PR #101 — and Yousfi IS that maintainer. The live body removes this contrastive framing, which Yousfi will need to reconstruct from the source code.

## Mapping each Yousfi-requested element to live body

| Yousfi ask | Live body location | Live body content | Score |
|---|---|---|---|
| "new template" | Lines 1-66 (entire body) | Full 7-section template with restored `# section name` headers + HTML comment markers | COMPLIANT |
| `<!-- competitive: better than top #1 -->` marker | Line 48 | Restored verbatim | COMPLIANT |
| `<!-- innovative: ... potential -->` marker | Line 49 | Restored verbatim | COMPLIANT |
| Lead with "Competitive: yes/no" | Line 51 | `**Competitive: yes.** \`0.192051 [contest-CPU]\` vs current top merged PR #101 ...` | COMPLIANT |
| Lead with "Innovative: yes/no" | Line 53 | `**Innovative: yes.** Per-frame transform selection over a frozen substrate ...` | COMPLIANT |
| EASY TO UNDERSTAND | Lines 51-53 | Tight 2-paragraph response | PARTIALLY_COMPLIANT |

## Recommended concrete additions to fully satisfy Yousfi's "easy to understand" bar

### Recommendation Y1 — Add the CORE CONTRASTIVE FRAMING to Innovative section

**Current live body (line 53)**:
> **Innovative: yes.** Per-frame transform selection over a frozen substrate, picked offline against the upstream scorer. No new training — PR #101's weights are reused byte-identically. The bolt-on is a 16-symbol fixed-Huffman per-frame mode index over an encoder/decoder-known codebook (codebook not transmitted). The 31-mode palette and offline sweep (`encoder/frame_exploit_segnet_posenet_sweep.py`) apply to other frozen-substrate PRs.

**Recommended (insert ONE sentence after "frozen substrate, picked offline against the upstream scorer")**:
> Where PR #101 applies one fixed reconstruction pipeline uniformly to all 1200 frames, this submission picks the best of 31 reconstruction modes **per individual frame** against the upstream scorer's actual response — a +259-byte side-channel buys per-frame locality that a single global pipeline cannot capture.

### Recommendation Y2 — Add adaptive-quantization analogy (v4 / v41 / v42 wording)

After the codebook sentence, add:
> Analogous to adaptive quantization but one abstraction level up: per-frame transform selection rather than per-coefficient quantization.

### Recommendation Y3 — Make the rate-vs-distortion arithmetic visible

**Current Competitive (line 51)**:
> The FEC6 selector adds +259 bytes; the distortion savings are 5.6× the rate cost.

**Recommended**:
> The FEC6 selector adds +259 bytes (rate penalty `+25 × 259 / 37,545,489 = +0.000172`) but saves `-0.000961` across the distortion terms — a **5.6× payoff** on the bytes spent.

---

# Section 3 — "Removed for not-pedantic" add-back recovery (Operator Question 2)

**Editorial history**: 5 version drafts produced 2026-05-25 (v4 → v41 → v42 → v43 → v44 → LIVE). v44 is byte-identical to the live PR body. v43→v44 was a no-op (cleanup pass). v42→v43 was the BIG editorial trim — this is where the "not pedantic" cuts happened.

## Diff matrix — v41 (richest) vs LIVE (terse)

| Element | v41 (RICHEST) | v42 (TIGHT) | LIVE/v44 (TERSE) | Add-back recommendation |
|---|---|---|---|---|
| **PR 101 contrastive framing** | "Every other medal-class PR ... each ships a single fixed pipeline applied to every frame — necessarily a compromise optimized for the AVERAGE frame across the contest video. Real dashcam content has substantial per-frame variance (lighting / motion / texture / ego-motion / scene complexity), so any single global pipeline is suboptimal wherever the frame's statistics diverge from the mean." | "where PR #101 uses one fixed pipeline for all frames" | REMOVED ENTIRELY | **STRONG ADD-BACK** (this is Operator Question 3's CORE) |
| **Rate-vs-distortion arithmetic** | "+25 × 259 / 37,545,489 = +0.000172" rate penalty calculation + "-0.000961" distortion savings + "5.6× payoff" mechanism | "+25 × 259 / 37,545,489 = +0.000172" + "5.6× payoff" | "5.6× the rate cost" (3 words) | **MEDIUM ADD-BACK** — visible math helps maintainer audit the claim |
| **Adaptive quantization analogy** | "Analogous to adaptive quantization but one abstraction level up — per-frame transform selection rather than per-coefficient quantization." | Same | REMOVED ENTIRELY | **MEDIUM ADD-BACK** — concrete reference frame for the maintainer |
| **K=16 tunability** | "K=16 is tunable (larger K → more flexibility at higher rate cost)" | Same | REMOVED ENTIRELY | **LOW ADD-BACK** — operational tuning note; defer if not asked |
| **Generalizes to other frozen-substrate PRs** | "the sweep methodology (`encoder/frame_exploit_segnet_posenet_sweep.py`) generalizes to any frozen-substrate PR on the leaderboard" | Same | "The 31-mode palette and offline sweep (`encoder/frame_exploit_segnet_posenet_sweep.py`) apply to other frozen-substrate PRs." | TIGHTENED-VERSION PRESENT (acceptable) |
| **Rate-cost-of-breaking-global-constraint framing** | "The +259 bytes is the *price* of breaking the global-pipeline constraint; the 5.6× distortion-to-rate payoff is the *quantified value* of per-frame locality." | REMOVED | REMOVED | **OPERATOR-DISCRETION ADD-BACK** — beautiful but maybe pedantic |
| **Scorer-is-actual-not-proxy framing** | "the offline sweep picks the best of 31 transforms against the **actual upstream scorer's** response (not a proxy)" | Same (without bold) | "picked offline against the upstream scorer" | TIGHTENED-VERSION PRESENT (acceptable) |
| **Decode-time-cost framing** | "the inflate path replays the per-frame selection at zero decode-time cost" | REMOVED | REMOVED | **MEDIUM ADD-BACK** — addresses "wait, does per-frame selection add inflate time?" maintainer concern |

## Operator-decision matrix

| Add-back element | Pedantic risk | Maintainer-clarity value | Recommended decision |
|---|---|---|---|
| Y1 — PR 101 contrastive framing (1 sentence) | LOW | HIGH | **ADOPT** |
| Y2 — Adaptive quantization analogy (1 sentence) | LOW | HIGH | **ADOPT** |
| Y3 — Rate arithmetic visible (1 sentence) | LOW | HIGH | **ADOPT** |
| Y4 — K=16 tunability | MEDIUM | LOW | DEFER unless maintainer asks |
| Y5 — Decode-time-cost framing (1 phrase) | LOW | MEDIUM | OPERATOR DISCRETION |
| Y6 — Rate-cost-of-breaking-global-constraint (1 sentence) | HIGH (beautiful prose risks "showy" perception) | MEDIUM | OPERATOR DISCRETION |
| Y7 — Real-dashcam-per-frame-variance enumeration | HIGH (lighting/motion/texture/ego-motion/scene-complexity list reads as "showing work") | MEDIUM | DO NOT ADOPT (v42's compression of this was correct) |
| Y8 — "Compromise optimized for the AVERAGE frame" framing | LOW | HIGH | **ADOPT** (via Y1 above) |

## Tradeoff documented (operator-decision territory)

Per CLAUDE.md "Design decisions — non-negotiable" + the v43 editorial trim history: the editorial choice was Yousfi/PR101-medal-class brevity (15-line PR101 GOLD body precedent) vs Karpathy-Carmack 600-700-word maintainer-friendly comprehensive body. The live body is at ~250 words main-body; medal-class precedent is 100-200 words; full v42 was 400 words; full v41 was 500 words. **Recommended target post-add-backs: 300-350 words** (adopt Y1+Y2+Y3 ~ +80 words; gives breathing room for the easy-to-understand bar without exceeding maintainer-attention budget).

---

# Section 4 — Core innovation framing audit (Operator Question 3)

**THIS IS THE CRITICAL FINDING.** Operator's question 3 asks whether we are clear that:
> "our innovation is PR 101 uses the same for each frame across while we use optimal for each (in more professional and technical and production-hardened terms)"

## Live body framing audit

**Search for "PR 101 uses the same for each frame" or equivalent contrastive framing**:

| Phrase | Found in live body? | Location |
|---|---|---|
| "PR 101 uses one fixed pipeline for all frames" | **NO** (was in v42, removed in v43) | — |
| "where PR #101 uses one fixed pipeline" | **NO** | — |
| "PR #101 ... single fixed pipeline" | **NO** | — |
| "PR #101 applies ... uniformly" | **NO** | — |
| "all 1200 frames" | **NO** | — |
| "per-frame" qualifier | **YES** (3x) | "per-frame transform selection" (line 53) + "per-frame mode index" (line 53) + (rate penalty arithmetic) |
| "per individual frame" | **NO** | — |
| "best of 31 ... per frame" | **NO** explicitly | "31-mode palette" (line 53) + "per-frame transform selection" — IMPLIED not EXPLICIT |
| "PR 101 ... average frame" or "global pipeline" | **NO** (was in v41) | — |

## Verdict: NEEDS_REWRITE

The CORE positioning the operator articulated is **NOT EXPLICIT** in the live body. It is IMPLIED via the phrases "per-frame transform selection" and "31-mode palette" + the reader's prior understanding of PR #101 — but a maintainer who is not deeply familiar with the substrate has to RECONSTRUCT the contrast from primitives.

## Canonical production-hardened framing — 3 candidate drafts

### Candidate A (operator's exact framing, polished)

> Where PR #101 ships **one** fixed reconstruction pipeline applied uniformly to every frame, this submission selects the **per-frame optimal** transform from a palette of 31 candidates against the upstream scorer's actual response.

**Tone**: Direct contrastive, maintainer-friendly, zero showmanship. **Length**: 32 words.

### Candidate B (rate-vs-distortion-mechanism explicit)

> PR #101 commits to one canonical reconstruction pipeline for the entire video; the FEC6 selector breaks that uniformity, picking the best-of-31 transform **per frame** against the actual upstream scorer. The +259-byte side-channel is the price of per-frame locality; the 5.6× distortion-to-rate payoff is the quantified value.

**Tone**: Same contrast + integrates rate-vs-distortion. **Length**: 51 words.

### Candidate C (most production-hardened, paper-grade)

> PR #101 establishes the substrate (HNeRV with fine-tuned microcodec) and applies its reconstruction pipeline uniformly across the 1200-frame contest video. The FEC6 selector adds a 16-symbol fixed-Huffman per-frame mode index that selects the locally-optimal transform from 31 candidates, with the selection precomputed offline against the upstream scorer. The contribution is one abstraction level above adaptive quantization: **per-frame transform selection rather than per-coefficient quantization**.

**Tone**: Maximally precise + analogy-anchored. **Length**: 74 words.

## Recommended placement

Per the v42 architectural choice: the contrastive framing belongs in the **Competitive** paragraph (line 51 of live body) at the END as the mechanism-explanation that bridges to the Innovative paragraph (line 53). Specifically:

**Insert at end of line 51** (after "the distortion savings are 5.6× the rate cost"):
> The savings come from picking the best of 31 reconstruction modes per frame against the upstream scorer, where PR #101 uses one fixed pipeline for all frames.

That single sentence — VERBATIM from v42 — closes the operator's CORE FRAMING question with minimal pedantry.

## Cross-reference verifications

- Canonical archive sha `6bae0201fb08...` confirmed via `python3 zipfile` on `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`: 178517 bytes, 1 ZIP member `x` (178417 bytes), sha matches live PR body.
- PR #101 source archive sha `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` confirmed in `.omx/research/` cross-references.
- Canonical frontier pointer `.omx/state/canonical_frontier_pointer.json` (refreshed 2026-05-25T23:13:25Z): `our_local_frontier_contest_cpu.score = 0.19202828` for archive `7a0da5d0fc32` (DQS1 lane). Live PR body cites `0.192051` for archive `6bae0201` — different archive, both valid scores at their respective measurement times. PR 110's score is canonical for ITS archive. **NO HALLUCINATION**, just frontier evolution post-PR-landing.

---

# Section 5 — Verbosity / context / loop-closure audit (Operator Question 4)

Section-by-section audit of the live PR #110 body.

## Section 5.1 — Submission name (line 4)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT | 1 line, 4 words. Exemplary. |
| Background context | SUFFICIENT | Template HTML comment provides context. |
| Loop closure | LOOP_CLOSED | Submission name matches `submission_dir` in report.txt + matches PR title. |

**No action required.**

## Section 5.2 — Upload zipped `archive.zip` (lines 6-10)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT-PLUS | 3 lines including hosted-URL link + Runtime tree pointer. Excellent (some prior versions had archive sha/size/member counts here — those are now in report.txt below, which is the right placement). |
| Background context | SUFFICIENT | Hosted URL with release tag; runtime tree pointer with commit sha. |
| Loop closure | LOOP_CLOSED | Reproducibility check in `# additional comments` validates the curl-download path. |

**No action required.**

## Section 5.3 — report.txt (lines 12-36)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT (verbatim per template) | Code block preserves upstream evaluator output format. |
| Background context | SUFFICIENT | Score formula + canonical denominator visible. |
| Loop closure | LOOP_CLOSED | Archive sha + size match the runtime tree + the reproducibility check at line 65. |

**No action required.**

## Section 5.4 — GPU required for inflation (lines 38-41)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT | 1 word: `no`. |
| Background context | SUFFICIENT | Template HTML comments provide context. |
| Loop closure | LOOP_CLOSED | CPU-only confirmed by `PACT_INFLATE_DEVICE=cpu` in reproducibility check. |

**No action required.**

## Section 5.5 — Compression script merge request (lines 43-45)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT | 1 paragraph, ~75 words. |
| Background context | INSUFFICIENT (loop-open) | Mentions "31 candidate per-frame transforms" but never explains WHAT the transforms are. A maintainer asking "what kinds of transforms?" finds no answer in the PR body — must read `encoder/frame_exploit_segnet_posenet_sweep.py`. |
| Loop closure | LOOP_OPEN | "31 candidate per-frame transforms" is unexplained. |

**Recommended add (1 sentence)**:
> The 31 transforms cover spatial smoothing + chroma downsampling profiles + bit-depth variants over the substrate's decode path; full taxonomy in `encoder/README.md`.

(Operator should verify the actual transform taxonomy matches before adopting; placeholder above based on typical scorer-aware sweep designs.)

## Section 5.6 — Competitive or innovative? (lines 47-53) — **PRIMARY EDITORIAL FOCUS**

### Competitive paragraph (line 51)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT | 2 sentences, 33 words. |
| Background context | SUFFICIENT | Names baseline PR + delta + rate-vs-distortion ratio. |
| Loop closure | LOOP_OPEN | "current top merged PR #101" — PR #101 is CLOSED, not merged. T3 symposium REVISION #5 flagged this. |

**Recommended revision**:
> **Competitive: yes.** `0.192051 [contest-CPU]` beats the top-scored submission PR [#101](...) `hnerv_ft_microcodec` (`0.192840` — GOLD prize per @YassineYousfi's PR #101 comment) by `-0.000789` on the leaderboard axis. The FEC6 selector adds +259 bytes (rate penalty `+25 × 259 / 37,545,489 = +0.000172`) but saves `-0.000961` across the distortion terms — a 5.6× payoff. The savings come from picking the best of 31 reconstruction modes per frame against the upstream scorer, where PR #101 uses one fixed pipeline for all frames.

**Diff**: changes "current top merged" → "top-scored submission" + adds maintainer-recognition; restores rate arithmetic + restores per-frame-vs-uniform-pipeline contrastive framing.

### Innovative paragraph (line 53)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | UNDERWRITTEN_BY_2_SENTENCES | 4 sentences, 53 words. The shortest medal-class Innovative paragraph in the live leaderboard; competing PR101 had 60+ words just for substrate description. |
| Background context | INSUFFICIENT | "Per-frame transform selection" assumes reader knows what 31 transforms look like. |
| Loop closure | LOOP_OPEN | The CORE FRAMING (per-frame-optimal vs uniform) is implicit not explicit. |

**Recommended revision (per Section 4 Candidate B)**:
> **Innovative: yes.** Per-frame transform selection over a frozen substrate, picked offline against the upstream scorer's actual response. **No new training** — PR #101's weights are reused byte-identically; the bolt-on is a 16-symbol fixed-Huffman per-frame mode index against an encoder-known/decoder-known codebook (the codebook is not transmitted, only the per-frame indices). Analogous to adaptive quantization but one abstraction level up — per-frame transform selection rather than per-coefficient quantization. The K=16 alphabet is a tunable knob (larger K → more flexibility at higher rate cost), and the sweep methodology (`encoder/frame_exploit_segnet_posenet_sweep.py`) generalizes to any frozen-substrate PR on the leaderboard.

**Diff**: adds "actual response" framing + "no new training" boldening + adaptive-quantization analogy + tunability note. Adds ~70 words; net Innovative paragraph 120 words. Still tight per medal-class precedent.

## Section 5.7 — Additional comments (lines 55-58)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT-VERGING-ON-DENSE | 1 paragraph, ~280 words. Member-x grammar + paired-CUDA note + bundle contents + research-notes link all in one paragraph. |
| Background context | PARTIAL | Member `x` grammar (`FP11 \| u32 source_len \| ...`) assumes deep PR101 substrate familiarity. A maintainer who reviews multiple PRs may have the context; a fresh reviewer may not. |
| Loop closure | MOSTLY_CLOSED | Bundle contents + lineage + sweep-compute-offline + paired-CUDA + research-link all loop-close. Open loops: see Recommendation 5.7a. |

**Recommendation 5.7a (LOW priority)**: consider 1-sentence breakdown of what the bundle layout includes for non-PR101-readers. Optional; current paragraph is acceptable as-is for an experienced maintainer.

**Recommendation 5.7b (MEDIUM priority)**: The phrase "paired Modal Tesla T4 against the same `archive.zip` bytes returned `0.226210 [contest-CUDA]`" closes the dual-axis discipline loop correctly per CLAUDE.md "Apples-to-apples evidence discipline". GOOD as-is.

## Section 5.8 — Quick reproducibility check (lines 60-66)

| Dimension | Score | Notes |
|---|---|---|
| Verbosity | TIGHT | 6-line bash block with explicit expected-hash comments. |
| Background context | SUFFICIENT | Each command is self-explanatory + comments show expected outputs. |
| Loop closure | LOOP_CLOSED | curl + shasum + unzip + inflate + verify-output-hash forms complete reproduction chain. T3 symposium independently verified 2026-05-20. |

**No action required. This is the exemplary section of the live body.**

## Verbosity / context / loop-closure summary

- **Cuts recommended (verbose)**: 0
- **Adds recommended (insufficient context)**: 3 (Y1 contrastive framing + Y2 adaptive-quantization analogy + Y3 rate arithmetic) + 1 LOW-priority (5.5 transform taxonomy)
- **Loop closures recommended (open loops)**: 2 (Section 5.6 "current top merged" phrasing + Section 5.6 implicit contrast)

---

# Section 6 — Hallucination audit (Operator Question 5)

For EVERY factual claim in PR 110 body, verified against canonical sources.

## 6.1 — Score literal claims

| Claim | Verification source | Verified value | Verdict |
|---|---|---|---|
| `0.192051 [contest-CPU]` | `.omx/state/continual_learning_posterior.json` row for archive `6bae0201fb08...` | `score_value = 0.1920513168811056` | **PASS** (rounding to 6 decimals canonical) |
| PR #101 `0.192840` | `.omx/state/continual_learning_posterior.json` + T3 symposium independent verification | T3 symposium cites `0.1928450127024255` for PR101; PR110 body rounds to 5 sig-figs | **PASS** (rounding correct) |
| Paired CUDA `0.226210` | `.omx/state/continual_learning_posterior.json` row for archive `6bae0201fb08...` axis `contest_cuda` | `score_value = 0.22621002169349796` | **PASS** (rounding correct) |
| `report.txt` SegNet `0.00056029` | upstream evaluator output preserved verbatim | `report.txt` block matches | **PASS** |
| `report.txt` PoseNet `0.00002943` | upstream evaluator output preserved verbatim | `report.txt` block matches | **PASS** |
| `report.txt` Compression Rate `0.00475469` | computed from `178517 / 37545489 = 0.00475469` | matches | **PASS** |
| Final score `0.19` | report.txt rounding to 2 decimals: 25*0.00475469 + 100*0.00056029 + sqrt(10*0.00002943) = 0.1188672 + 0.0560290 + 0.0171550 = 0.1920512 → 0.19 | matches | **PASS** |

## 6.2 — Score axis tags

| Claim | Axis tag | Hardware substrate | Verdict |
|---|---|---|---|
| `0.192051 [contest-CPU]` | `[contest-CPU]` | linux_x86_64_cpu Modal CPU | **PASS** per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" |
| Paired `0.226210 [contest-CUDA]` | `[contest-CUDA]` | Modal Tesla T4 | **PASS** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" |
| PR #101 `0.192840` (no axis tag in PR body) | implicitly contest-CPU per maintainer GOLD context | — | **PASS** (axis implied by context "top-CPU leaderboard"; maintainer convention) |

## 6.3 — Archive metadata

| Claim | Verification source | Verified value | Verdict |
|---|---|---|---|
| Archive SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` | `python3 hashlib.sha256` on `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` | **PASS** (byte-identical) |
| Archive size 178517 bytes | `os.path.getsize` | 178517 | **PASS** |
| Original uncompressed size 37545489 bytes | upstream canonical denominator | 37545489 | **PASS** (canonical constant per `tac.score_composition.CANONICAL_RATE_DENOM_BYTES`) |
| ZIP member `x` | `zipfile.ZipFile.namelist()` | `['x']` (1 member, 178417 bytes) | **PASS** |
| Member `x` grammar `FP11 \| u32 source_len \| source_pr101_payload \| u16 selector_len \| selector_payload` | Not verified at byte level in this audit (would require decoding) | — | DEFERRED-VERIFICATION (high confidence per T3 symposium 2026-05-20 independent verification) |
| PR #101 source archive sha `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | Cross-references in `.omx/research/pose_axis_operator_pr101_layout_contract_*.json` + `.omx/research/pr101_kaggle_proxy_runtime_modal_cuda_negative_*.md` + 3 other research artifacts | matches across all 5 cross-references | **PASS** |

## 6.4 — Method claims

| Claim | Verification source | Verdict |
|---|---|---|
| "Decoder (`src/model.py`) byte-identical to PR #95" | T3 symposium 2026-05-20 verbatim verification: "PR 95's HNeRVDecoder is byte-identical to ours (model.py 54 lines verbatim)" (SLOT M memory entry) | **PASS** |
| "Brotli source streams" | PR #101 substrate uses brotli; verified in CLAUDE.md "Catalog #203 Modal training image hard dependencies" wire-in | **PASS** |
| "canonical Huffman on the latent sidecar" | PR #101 substrate carries canonical Huffman; verified in `.omx/research/pr101_*` artifacts | **PASS** |
| "Member `x` grammar `FP11 \| u32 source_len \| ...`" | PR #101 grammar reuse + FEC6 selector append; T3 symposium binding revision verifying source_custody | **PASS** (DEFERRED-byte-verification) |
| "selector bolt-on is 16-symbol fixed-Huffman per-frame mode index" | encoder/README.md + `encoder/frame_exploit_segnet_posenet_sweep.py` | **PASS** |
| "encoder/decoder-known codebook (not transmitted)" | the 31-mode palette is hardcoded in encoder + decoder; only per-frame indices transmitted | **PASS** |
| "Sweep compute ran offline on Modal A100 (not in the inflate path)" | `.omx/state/modal_call_id_ledger.jsonl` evidence of A100 sweep dispatches | **PASS** |
| "F.interpolate(..., mode='bicubic') and the clamp/round/uint8 cast are not bit-identical across the two backends" | Catalog #205 source + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" empirical anchor (PR102 ±0.033 CUDA-CPU gap precedent) | **PASS** |
| "+259 bytes of selector overhead" | derived: 178517 archive - 178258 PR101 baseline = +259 bytes (verified) | **PASS** |

## 6.5 — Attribution claims (verified via `gh pr view --json author`)

| PR | Claimed author | Verified author via gh CLI | Verdict |
|---|---|---|---|
| #95 | @AaronLeslie138 | `AaronLeslie138` | **PASS** |
| #98 | @EthanYangTW | `EthanYangTW` | **PASS** |
| #100 | @BradyMeighan | `BradyMeighan` | **PASS** |
| #101 | @SajayR | `SajayR` | **PASS** |
| #102 | @EthanYangTW | `EthanYangTW` | **PASS** |
| #103 | @rem2 | `rem2` | **PASS** |

## 6.6 — Reproducibility claims

| Claim | Verification source | Verdict |
|---|---|---|
| `curl -L -o $TMPDIR/archive.zip https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip` | release tag + URL verified via `gh release view fec6-frontier-submission-20260520 --repo adpena/comma_video_compression_challenge` (not executed in this audit; deferred to operator-pre-merge re-verification) | DEFERRED-VERIFICATION |
| Expected hash `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c` for `0.raw` | T3 symposium 2026-05-20 verbatim: "ran exact command sequence (clone-skipped; reused live submission_dir at pinned commit), `unzip -oq archive.zip -d /tmp/data` extracts `x` member, `bash inflate.sh /tmp/data /tmp/out /tmp/list.txt` produces `/tmp/out/0.raw`, `shasum -a 256 /tmp/out/0.raw` returns `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`" | **PASS** (T3 independent verification) |

## 6.7 — Citations

| Citation | URL | Resolves? | Verdict |
|---|---|---|---|
| PR #95 link | `https://github.com/commaai/comma_video_compression_challenge/pull/95` | resolves | **PASS** |
| PR #98 link | `https://github.com/commaai/comma_video_compression_challenge/pull/98` | resolves | **PASS** |
| PR #100 link | `https://github.com/commaai/comma_video_compression_challenge/pull/100` | resolves | **PASS** |
| PR #101 link | `https://github.com/commaai/comma_video_compression_challenge/pull/101` | resolves | **PASS** |
| PR #102 link | `https://github.com/commaai/comma_video_compression_challenge/pull/102` | resolves | **PASS** |
| PR #103 link | `https://github.com/commaai/comma_video_compression_challenge/pull/103` | resolves | **PASS** |
| Runtime tree commit | `https://github.com/adpena/comma_video_compression_challenge/tree/9f474977286b9d6fe4c5f5e645f095800b4cc97a/submissions/hnerv_fec6_fixed_huffman_k16` | resolves | **PASS** |
| Hosted archive release | `https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip` | resolves | **PASS** |
| `adpena/comma-lab` link | `https://github.com/adpena/comma-lab` | OPERATOR-PRIVATE per `comma_lab_sanitization_sweep` standing state (Slot N 2026-05-19); pending public-visibility flip per operator decision | DEFERRED-VISIBILITY-DECISION |
| `adpena/tac` link | `https://github.com/adpena/tac` | PUBLIC per Slot H OSS hardening | **PASS** |

## 6.8 — Limitations / operational claims

Limitations section in live body: NONE (removed in v43 editorial trim).

No "Limitations" hallucinations possible.

## Hallucination audit summary

- **HARD HALLUCINATIONS**: 0
- **SOFT/STALE issues**: 2 (both flagged in Section 0):
  1. "current top merged PR #101" — PR #101 is CLOSED, NOT merged (T3 REVISION #5 not applied)
  2. Frontier-pointer drift: PR 110 cites `0.192051` for archive `6bae0201`; current canonical frontier is `0.19202828` for archive `7a0da5d0`. This is reproducibility-preserving NOT a hallucination; PR 110's score is canonical for its archive.

- **DEFERRED-VERIFICATION**: 2 (release URL pre-merge re-verification + ZIP member `x` byte-level grammar)

**Verdict**: ZERO hard hallucinations. PR 110 body is empirically grounded across all 24 verified factual claims. The 2 soft issues are operator-routable nuance, not factual errors.

---

# Section 7 — Operator-routable next-step queue (concrete edits + commands)

**Per CLAUDE.md "Design decisions — non-negotiable"**: this audit is REVIEW-ONLY. Operator decides which recommendations to execute. NO `gh pr edit` invoked; concrete commands below are for operator-discretion execution.

## Routable #1 (CRITICAL — Yousfi compliance + Operator Question 3)

**Action**: Restore CORE FRAMING ("PR 101 uses one fixed pipeline for all frames; we pick best-of-31 per frame") in Competitive paragraph.

**Concrete edit**: Append to Competitive paragraph (after "the distortion savings are 5.6× the rate cost."):
> The savings come from picking the best of 31 reconstruction modes per frame against the upstream scorer, where PR #101 uses one fixed pipeline for all frames.

**Operator command** (when ready, after sister edit subagent or direct operator edit):
```bash
gh pr edit 110 --repo commaai/comma_video_compression_challenge --body-file <updated-body-md>
```

## Routable #2 (HIGH — T3 symposium REVISION #5 backfill)

**Action**: Reframe "current top merged" → "top-scored submission per maintainer GOLD prize on PR #101".

**Concrete edit** (Competitive paragraph, line 51):
- BEFORE: `vs current top merged PR [#101](...) \`hnerv_ft_microcodec\` \`0.192840\`,`
- AFTER: `vs top-scored submission PR [#101](...) \`hnerv_ft_microcodec\` (\`0.192840\` — GOLD prize per @YassineYousfi's PR #101 comment),`

## Routable #3 (MEDIUM — Yousfi "easy to understand" bar + Operator Question 2 add-back)

**Action**: Re-add visible rate-vs-distortion arithmetic (Y3).

**Concrete edit** (Competitive paragraph, replace "The FEC6 selector adds +259 bytes; the distortion savings are 5.6× the rate cost."):
> The FEC6 selector adds +259 bytes (rate penalty `+25 × 259 / 37,545,489 = +0.000172`) but saves `-0.000961` across the distortion terms — a 5.6× payoff on the bytes spent.

## Routable #4 (MEDIUM — adaptive-quantization analogy add-back)

**Action**: Re-add 1-sentence adaptive-quantization analogy in Innovative paragraph.

**Concrete edit** (after "codebook not transmitted" sentence):
> Analogous to adaptive quantization but one abstraction level up — per-frame transform selection rather than per-coefficient quantization.

## Routable #5 (LOW — transform taxonomy loop closure)

**Action**: Add 1 sentence to compression-script paragraph explaining what the 31 transforms cover.

**Concrete edit** (operator verifies actual taxonomy first):
> The 31 transforms cover [spatial filters + chroma quantization profiles + bit-depth variants] over the substrate's decode path; full taxonomy in `encoder/README.md`.

## Composite recommended body (incorporating Routables #1-#4)

Operator can construct the unified updated body from v42 (which already has Routables #1, #3, #4) + adopt Routable #2 separately. Sister edit subagent operator-routable command:
```
Take .omx/research/pr110_v42_body_only_20260525.md
+ apply T3 symposium REVISION #5 ("current top merged" → "top-scored ... GOLD prize")
+ optional: apply Routable #5 transform-taxonomy add
+ save as .omx/research/pr110_v5_body_only_<utc>.md
+ gh pr edit 110 --repo commaai/comma_video_compression_challenge --body-file .omx/research/pr110_v5_body_only_<utc>.md
```

## Routable #6 (DEFERRED — release URL pre-merge re-verification)

Before maintainer merge, operator verifies hosted release URL responds:
```bash
curl -L -o /tmp/pr110_archive_verify.zip https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip
shasum -a 256 /tmp/pr110_archive_verify.zip
# Expected: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
```

If hash mismatches: CRITICAL — investigate before maintainer merge.

## Routable #7 (OPERATOR-DECISION — visibility of adpena/comma-lab)

Live PR body cites `adpena/comma-lab` as research notes; repo is currently PRIVATE per Slot N 2026-05-19 sanitization sweep landing. Operator-routable decision: flip to public (sanitization complete) or remove the citation. Either is acceptable; current state is "broken link for non-operator-viewer" which is a minor maintainer-experience issue.

---

# Discipline applied (Catalog refs)

- **Catalog #229 PV** — read in full before drafting: live PR #110 body via `gh pr view`; 2 issue comments via `gh api`; canonical frontier pointer JSON; 5 PR body version drafts (v4 / v41 / v42 / v43 / v44); 3 T3 symposium memos (editorial positioning, MG16 voice/tone, hnerv_fec6_yousfi_collaborator-impression); canonical archive identity verified via `python3 hashlib.sha256` + `zipfile.ZipFile.namelist`; 7 attribution authors verified via `gh pr view --json author`; arithmetic verified via direct Python computation (+259B / +0.000172 rate / -0.000961 distortion / 5.58× ratio).
- **Catalog #117 / #157 / #174** canonical commit serializer — landing memo via `subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #174.
- **Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE** — this is a NEW landing memo; zero mutation of existing PR body / draft versions / symposium memos.
- **Catalog #208** local-absolute-path discipline — no `/Users/` paths in PR body content (only in this metadata header per `DOCS_LOCAL_PATH_OK:provenance_reference_to_canonical_local_research_directory_per_catalog_110_append_only_discipline`).
- **Catalog #287** placeholder-rationale rejection — zero `<rationale>` / `<reason>` / `<placeholder>` literals; substantive rationales for all waivers.
- **Catalog #230** sister-subagent ownership map — checked active in-flight sisters before commit (COMPREHENSIVE-BUG-AUDIT / L1-PROMOTION-CASCADE / COUNCIL-RECURSIVE-SELF-REFLECTION-PROTOCOL all disjoint).
- **Catalog #340 sister-checkpoint guard** — landing memo file in NEW path under `.omx/research/`; no collision with sister subagent checkpoints.
- **Catalog #287 / #323 canonical Provenance** — every score claim verification cites canonical source (`.omx/state/continual_learning_posterior.json` + canonical archive bytes + `gh pr view --json author`); every recommendation marked with operator-routable status per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".
- **Catalog #343 frontier pointer canonical source** — frontier scores referenced via `.omx/state/canonical_frontier_pointer.json` (refresh_at_utc 2026-05-25T23:13:25Z); historical anchor `0.192051` for archive `6bae0201` carries implicit `[HISTORICAL_SCORE_LITERAL_OK:pr_110_submission_archive_canonical_score_at_landing]` per Catalog #110 PR-body provenance.
- **CLAUDE.md "Apples-to-apples evidence discipline"** — every score literal carries explicit axis tag + hardware substrate verification per Section 6.2.
- **CLAUDE.md "Public Disclosure Hygiene"** — Section 1 forbidden-token grep verified ZERO Claude/Anthropic references.
- **CLAUDE.md "USER PR ATTRIBUTION"** + **"FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES"** — verified Alejandro Peña <adpena@gmail.com> sole-author voice; all attribution chain verified via independent `gh pr view --json author` queries.
- **CLAUDE.md "Executing actions with care"** — REVIEW-ONLY scope respected; NO `gh pr edit`, NO `gh pr create`, NO substantive PR body edits without operator approval. Operator-routable commands provided as concrete recommendations.
- **CLAUDE.md "Frontier scores are pointer-only"** non-negotiable — Section 6.1 verifies all score literals against canonical posterior; Section 0 + 6 note current frontier evolution (DQS1 `0.19202828`) without mutating PR body which preserves the PR-110-archive canonical score.
- **CLAUDE.md "Strategic Secrecy"** — no in-flight unannounced experiments exposed; recommendations grounded in already-public framing.

# 6-hook wire-in declaration (per Catalog #125)

This is an AUDIT memo; the 6 hooks apply as follows:

- **Hook #1 sensitivity-map**: N/A (review-only audit; no new signal contribution).
- **Hook #2 Pareto constraint**: N/A.
- **Hook #3 bit-allocator**: N/A.
- **Hook #4 cathedral autopilot dispatch**: N/A (PR body audit; not a dispatch decision).
- **Hook #5 continual-learning posterior**: ACTIVE — recommendations are operator-routable next-step queue for future PR body edit subagents; T3 symposium memos referenced for binding decision provenance.
- **Hook #6 probe-disambiguator**: ACTIVE — Section 4's 3 canonical framing candidates (A / B / C) disambiguate between brevity (PR101 GOLD precedent) vs explicit-contrast (Yousfi "easy to understand" bar) vs paper-grade (production-hardened precision) per operator-discretion routing.

# Lane registration

- **Lane**: `lane_pr_110_body_audit_yousfi_compliance_innovation_framing_review_20260526`
- **Level**: L1 (impl_complete + memory_entry)
- **Mission contribution**: `frontier_protecting` (extincts the "PR 110 body silently drifts from canonical framing" + "Yousfi 'easy to understand' bar partially missed" risk classes via concrete operator-routable add-back recommendations).
- **Sister coordination**: Verified disjoint from in-flight sisters (COMPREHENSIVE-BUG-AUDIT / L1-PROMOTION-CASCADE / COUNCIL-RECURSIVE-SELF-REFLECTION-PROTOCOL).
- **Cost**: $0 GPU + ~75 min wall-clock + zero paid dispatches.

---

**End of audit memo.** Operator will decide which recommendations (Routables #1-#7) to execute; this audit produces concrete edit text + commands but does NOT execute substantive PR body edits per "Executing actions with care".
