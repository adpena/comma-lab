# PR128 latent-click DIFFERENTIAL FORENSICS — the author's exact click set, its structure, and transfer to OUR table

**Date:** 2026-07-12 · **Subagent:** `clickforensics` · **Axis:** all rows `[macOS-CPU advisory]` NON-PROMOTABLE (n600 or not evidence)
**Operator ask (2026-07-12):** *"compare our binary blob archive against the author's highly optimized and find what specifically clicks he found because there's likely a pattern and signal there."*
**STORES CONSULTED:** `.omx/research/pr128_intake_reverse_engineering_20260710.md` (the description-level intake) · `.omx/state/canonical_frontier_pointer.json` (pointer 0.19108282 [contest-CPU] UNMOVED) · `experiments/results/click_polish_399_run1/` (our round-1 accepted-click ledger) · `src/tac/click_polish.py` + `src/tac/canonical_equations/click_polish_byte_neutral_slack_20260711.py` (the #399 machinery, reused library-mode).
**Method:** byte-exact latent decode of 6 archives → reconstructed q-grid (600×28 uint8 latent codes, the physically meaningful click coordinate) via each family's own container unpack (PR128 `codec_ctx`, PR112 sidecar-strip, our FP11-wrapper strip) + the shared `LATENT_DIM_ORDER`/cumsum reconstruction (byte-identical across PR128/PR112/ours). Scripts: scratchpad `extract_latent_raw.py`, `click_forensics.py`, `mine_pattern.py`, `cross_check.py`, `n600_chunk.py`, `aggregate.py`, `build_priors.py`.

---

## Headline (5 verdicts)

1. **BASE-EQUALITY = TRUE (decisive).** PR128's baseline latent table (= merged PR112) and OUR PR110-lineage base latent table are **byte-identical in q-space: 0 of 16 800 cells differ** (and the 16 912-byte `latent_raw` blobs hash-match: both `c760cab8…`). PR112 is a *lossless* ctx-recode of our PR110 payload, confirmed at the code level. **⟹ the author's clicks live in the EXACT SAME coordinate system as ours and apply VERBATIM to our table.** (Decode validated independently: our decoded PR128 `latent_raw` hashes `a7eba972…` = the git-LFS oid the author stored.)

2. **The click set — exact counts (q-space diff vs the shared base):** PR128 **final = 2 656 clicks**, PR128 **v1 = 2 031**, our #399 run-1 = 45 (= 8 incumbent n8-frontier clicks on pairs 0-7 + the 37 ledger clicks; reconciled, no anomaly). The intake's "~1 565–2 162 net code changes" was the author's own accounting; the q-space diff against the pre-sidecar base is **2 656** (includes the folded PR101 sidecar corrections).

3. **THE PATTERN (the operator's ask) — the slack is ISOTROPIC, not concentrated.** The hoped-for "5 dims carry 80%" concentration is **FALSIFIED**: clicks distribute **near-uniformly across all 28 latent dims** — hottest dim 26 = 5.6%, coldest dim 23 = 2.1%, coefficient of variation **0.238**, and it takes **21 of 28 dims** to reach 80% of clicks (top-5 only 24%). **99.7% of pairs (598/600) are touched**, mean 4.44 clicks/pair. Sign is balanced (47% + / 53% −). **⟹ there is NO few-dim / few-pair shortcut: the residual quantization slack after PR95's QAT is well-mixed across the whole latent code. A full 28-dim × 600-pair sweep is mandatory — which is exactly why the author leaned on diagonal batching (600-exact-candidates-per-render) rather than a targeted subset.**

4. **Magnitude is the actionable prior — extend beyond ±2.** |δ| distribution: **±1 = 86.3%, ±2 = 10.7%, ±3 = 2.7%, ±4 = 0.3%, ±5 = 0.08%** (roughly geometric decay). Our #399 run-1 capped at |2|; **13.7% of the author's accepted clicks are |≥2| and 3.1% are |≥3| up to |5|** — a real, un-harvested tail. **Temporal coherence is ABSENT** (adjacent-same-dim clicked cells 464 vs random-expected 440 = 1.05×; same-sign 51% = chance) ⟹ clicks are per-pair-INDEPENDENT quantization slack, **not** a smooth ego-motion trajectory. Pair-locality (independent-add) is the ONLY exploitable structure; there is no temporal prior.

5. **TRANSFERABILITY — measured, high.** (a) Where our independent round-1 search and PR128 touched the SAME (pair,dim) cell (14/37 of our clicks, **2.4× above the 5.9 chance baseline**), the two searches agree on **SIGN 93% (13/14)** and on the exact δ 50%. (b) The IMPORT candidate — PR128's final table spliced onto our base packet, sidecar dropped, byte-closed — scores **advisory S 0.188070** vs our base **0.191110** (**ΔS −0.00304**: d_seg −2.6e-5, bytes −605), reproducing PR128's published 0.187992 to within 8e-5 (cross-axis macOS-vs-Windows CPU + our +33 B FP11-grammar overhead). **⟹ the author's entire d_seg gain transfers essentially fully to our substrate** — as base-equality predicts. **DEPTH, not coverage, is our gap:** PR128 accepted **215 clicks in pairs 0-47** where our single round found 37 (**5.8×**); the method is depth-limited (many greedy rounds), and 592 of our 600 pairs are still only n8-polished.

**Pointer honesty:** our exact pointer is **UNMOVED at 0.19108282 [contest-CPU]**. Every number here is `[macOS-CPU advisory]` and NON-PROMOTABLE; the import candidate is a **NO-FAKE #7 defensive bank** (borrowed clicks on borrowed-lineage substrate), not innovation. The exact contest-CPU row is MODAL-HOLD (no paid dispatch this unit).

---

## 1. Provenance table (byte-exact custody)

| tag | archive | archive sha (12) | member `x` | latent_raw sha (12) | q-clicks vs base |
|---|---|---|---|---|---|
| shared base | PR112 `pr112_archive.zip` | dd4f3899b91f | 177,036 | **c760cab847e4** | 0 (origin) |
| shared base | OURS PR110 `pr110_payload_entropy_recode…/archive.zip` | b46897267ded | 177,069 | **c760cab847e4** | 0 (byte-identical to PR112) |
| PR128 FINAL | `…_claude/artifacts/archive.zip` | cfd941de10e5 | 176,431 | a7eba9722beb (=LFS oid ✓) | **2 656** |
| PR128 V1 | `…_claude/artifacts/archive_v1.zip` | ab73259395f9 | 176,431 | 5b2b421acb12 | 2 031 |
| OURS n8 (pointer) | `clickpolish_pr110_20260710/n8_validation/candidate_archive.zip` | ad02b0124cbb | 177,069 | 031f8c63cce5 | 8 |
| OURS run-1 | `click_polish_399_run1/candidate_archive.zip` | 0872086672e7 | 177,069 | f9d5e8a312b2 | 45 (= 8 + 37) |

## 2. Per-dim click density (PR128 final, n=2 656) — the isotropy evidence

Ranked %: dim 26 **5.6** · 24 5.1 · 20 4.9 · 0 4.5 · 14 4.3 · 12 4.3 · 22 4.3 · 4 4.3 · 6 4.2 · 18 3.8 · 10 3.7 · 3 3.5 · 21 3.5 · 7 3.4 · 2 3.4 · 19 3.4 · 13 3.4 · 16 3.1 · 11 3.0 · 17 2.9 · 8 2.9 · 27 2.9 · 25 2.9 · 1 2.8 · 15 2.8 · 5 2.4 · 9 2.4 · 23 **2.1**. CoV 0.238; 21/28 dims for 80%. (v1 is nearly identical — same isotropic profile at n=2 031.)

## 3. v1 → final trajectory (the author's improvement path)

kept **1 905** · revised **49** · dropped **77** · **added 702**. The second published iteration is overwhelmingly *accumulative* (kept 94% of v1, added 702 new clicks); only 126 v1 clicks were revised/dropped as the greedy base shifted. ⟹ the method converges by monotone greedy accumulation over many rounds; more compute = more accepted clicks = lower S; plateau approached (not yet fully hit) by 2 656.

## 4. The negatives (author-published, re-affirmed as priors)

- Weight-code clicks: **every one rejected** — PR95 decoder is at a strict discrete local optimum; do NOT spend on weight-code search of that family.
- Gradient/QAT latent polish with margin surrogates: **surrogate improved while TRUE seg doubled** — direct support for our NO-FAKE #8 surrogate-≠-authority law; caution for any witness post-hoc gradient polish.
- GPU-selected clicks lose ~30% on CPU: select on the authority axis. (Our harness already scores on CPU.)

## 5. Import-candidate advisory result (n600, chunked serial foreground, ≤213 s/half)

| | d_seg | d_pose | bytes | S `[macOS-CPU advisory]` |
|---|---|---|---|---|
| base (Q0 + sidecar) — reproduce | 0.0005599 | 2.942e-5 | 177,169 | **0.1911103** |
| import (PR128 table, sidecar dropped) | 0.0005337 | 2.937e-5 | 176,564 | **0.1880699** |
| Δ | −2.62e-5 | −5e-8 | **−605** | **−0.00304** |

Base reproduces the incumbent (~0.19109 contest-CPU) ✓. Import artifact: `experiments/results/pr128_click_import_forensics_20260712/import_candidate_archive.zip` sha `196acd18e4ca…`, 176,564 B. The −605 B is the PR101-sidecar fold; the +33 B vs PR128's own 176,531 is our FP11 wrapper + FECa re-wrap + DQS1 tail overhead.

## 6. What is owed (means → the exact row)

1. **Exact contest-CPU (Linux x86_64) row** on the import candidate — MODAL-HOLD, operator GO. Advisory 0.188070 predicts a real sub-0.189 defensive-bank pointer move; the pointer moves ONLY through it.
2. **Feed the priors into #399** (`click_search_priors_pr128.json`): extend the per-cell step search to ±3 (rarely ±5), keep the full 28-dim × 600-pair sweep (no subset), run MANY greedy rounds (depth is the gap, 5.8×), exploit ONLY pair-locality (no temporal prior). This is the fastest path past our own n8/run-1 to ≈ the author's 2 656.
3. **Note:** our import candidate ALREADY equals the author's table on d_seg; the remaining ~+33 B is pure grammar overhead — a byte-audit of our FP11/FECa/DQS1 wrapper vs PR128's raw ctx container is a separate ~−33 B rate lane.

## Deliverables (this unit)

- Forensics memo (this file) · priors `experiments/results/pr128_click_import_forensics_20260712/click_search_priors_pr128.json` · import candidate + `n600_advisory_result.json` (own results dir, #399 files untouched) · canonical equation `pr128_latent_click_structure_isotropic_geometric_v1` (EmpiricalAnchor ×2) · DAG FEED-pr128forensics.
