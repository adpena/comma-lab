---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author, Rudin, Daubechies, Karpathy, Carmack, Filler, TimeTraveler, JackFromSkunkworks]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "v3 conservative draft is structurally cleaner if we strip the cost-as-funding-ask line and the class-shift roadmap enumeration. LIVE body has more signal but Live wastes lines on full-precision recomputation a reviewer can do mentally; OK to keep one (CPU OR CUDA) full recomputation rather than both."
  - member: Hotz
    verbatim: "Stop polishing. PR 101 GOLD is one sentence. Ship LIVE as-is or add 1 sentence acknowledging within-family local floor. Anything more is overboard."
  - member: Contrarian
    verbatim: "Option D (draft only, no recommendation) is the disciplined choice — the operator should choose, not the council. But that abdicates the symposium's job. I yield to Option B per operator's explicit 'what does the council suggest'."
council_assumption_adversary_verdict:
  - assumption: "PR #101 GOLD precedent is the right standard"
    classification: HARD-EARNED
    rationale: "Yousfi's 2026-05-11 PR #108 closure + PR 101/102/103 all medal-class with brief bodies; Selfcomp PR #56 also brief. Empirically: every medal-class body is < 50 lines, names predecessor PRs explicitly, no closing-flourish. LIVE body at 85 lines is at the upper boundary; CONSERVATIVE draft at 40 lines is at the lower boundary. The precedent is well-established."
  - assumption: "LIVE body has too much technical detail (deep-precision recomputation, paired CUDA)"
    classification: CARGO-CULTED
    rationale: "PR #95 best-write-up-prize body had ~40 lines including external blog link. PR #101 GOLD was 15 lines BUT inherited PR #98+#95 context so brevity was net-zero-loss. Our submission is a NEW selector mechanism (FEC6) without prior public context; deep technical detail in the body IS the public context. The recomputation is a 3-line block, not 30 lines. Discipline: apples-to-apples evidence per CLAUDE.md non-negotiable REQUIRES the paired CPU+CUDA + exact archive byte count be machine-extractable from the PR body itself, not via 'see comma-lab.' This is the canonical reproducibility surface."
  - assumption: "CONSERVATIVE draft is closer to medal-class restraint posture"
    classification: CARGO-CULTED
    rationale: "Restraint-by-line-count is a proxy for restraint-by-signal-density. The CONSERVATIVE draft cuts the report.txt block + full-precision recomputation + axis-decomposition + Reproducibility-as-section but ADDS a 5-paradigm class-shift enumeration AND a 'can't fully fund as solo developer' framing. Per signal-density-per-line, CONSERVATIVE is LESS disciplined: it removes reproducibility signal AND adds soft selling (funding ask + roadmap exposure). LIVE is shorter PER UNIT OF SIGNAL even if it is longer in absolute line count."
  - assumption: "Explicit class-shift roadmap (NeRV / cooperative-receiver / pretrained-driving-priors / predictive-coding / Wyner-Ziv) is signal a reviewer would value"
    classification: CARGO-CULTED
    rationale: "CLAUDE.md 'Strategic Secrecy' is binding standing directive. The class-shift enumeration in CONSERVATIVE draft tells competitors exactly which paradigms we have internally explored. Yousfi sits on the leaderboard and reviews every PR; competitors read every merged PR body. This is the canonical roadmap-leak failure mode. Furthermore, the operator already loses authorship credit if a competitor implements an enumerated paradigm and submits faster. The class-shift discussion is exactly the kind of context that belongs in internal lab notes (comma-lab), NOT the public PR body. The LIVE body's 'Where this sits' paragraph correctly handles this: states the within-HNeRV-family local floor, names the empirical observation (general-purpose compression on archive bytes did not yield material reduction), and concludes 'Further score reductions will likely require changing the emitted representation, training priors/objectives, entropy model, archive grammar, or decoder-side side information' — these are public-textbook-level technique categories, not specific named substrate paradigms. That is the right granularity."
  - assumption: "Mentioning solo-developer funding constraint reads as honest context"
    classification: CARGO-CULTED
    rationale: "Yousfi 2026-05-11 PR #108 closure tone was matter-of-fact technical: 'competitive OR innovative.' No 'we appreciate small teams' carve-out. The leaderboard is anonymous-by-design (GitHub handle only); funding context is non-public-PR-body material. 'Can't fully fund as a solo developer' reads as either (a) soliciting funding (cringe) or (b) excuse-making for not pursuing the roadmap (also cringe). Per CLAUDE.md 'Public Disclosure Hygiene' + the operator-verbatim 'no cringe / overboard / want them to hire us' triple: the funding constraint MUST be omitted. If hire-worthy posture is the goal, demonstrating disciplined empirical work IS the hire signal — funding constraints are not."
council_decisions_recorded:
  - "op-routable #1: Option B SYNTHESIS — keep LIVE body verbatim; add ~10 lines to '# additional comments' that acknowledge within-HNeRV-family local-floor framing using PUBLIC-TEXTBOOK technique categories only (representation / training priors / objectives / entropy model / archive grammar / decoder-side side information). DO NOT add specific named paradigms (NeRV / cooperative-receiver / pretrained-driving-priors / predictive-coding / Wyner-Ziv). DO NOT add solo-developer-funding framing."
  - "op-routable #2: draft file at /tmp/pr110_body_v4_synthesis_council_approved_20260520.md (operator review only; DO NOT invoke gh pr edit). Operator routes the actual gh pr edit invocation."
  - "op-routable #3: prior T3 PR-body symposium (commit eac8a3a7f + 8bc07a926) PROCEED_WITH_REVISIONS verdict on word-count compression (1008→600-700) was applied to a different body version (PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md). The PR #110 LIVE body is a separate artifact with different scope (already-shipped frontier-fix PR with one-sentence Operational notes folded; 85 lines is acceptable for PR #110 scope per this symposium)."
  - "op-routable #4: future PR submissions should adopt the synthesis pattern (LIVE rigor + 1-2 sentence Where-this-sits framing using public-textbook technique categories) as the canonical template; document at .omx/research/canonical_pr_body_template_synthesis_pattern_20260520.md (DEFERRED to operator)."
  - "op-routable #5: per CLAUDE.md 'Strategic Secrecy' standing: keep specific named class-shift paradigms (NeRV/cooperative-receiver/pretrained-driving-priors/predictive-coding/Wyner-Ziv) in internal comma-lab notes ONLY; never in public PR bodies, README, or commit messages destined for upstream comma_video_compression_challenge fork-branch."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
related_deliberation_ids:
  - t3_pr_body_final_recursive_review_iteration_20260519
  - t3_upstream_contest_compliance_conformance_symposium_20260519
  - t3_second_supplemental_missing_voices_canonical_roster_helper_20260519
deferred_substrate_id: pr_110_writeup_decision
---

# T3 Grand Council Symposium — PR #110 Writeup Decision

**Date:** 2026-05-20T21:17:16Z
**Operator directive (verbatim 2026-05-20):** *"what does the grand council symposium suggest we do with the PR 110 writeup"*
**Composite verdict:** `PROCEED_WITH_REVISIONS` — **Option B SYNTHESIS** (LIVE rigor + ~10-line Where-this-sits framing using public-textbook technique categories only)
**Sextet quorum:** 6-of-6 (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Contrarian + Assumption-Adversary)
**Grand council quorum:** 19-of-22 attendees (14 INNER + 5 topical-GRAND); zero structural under-rostering per `validate_council_dispatch_roster(complete=True)`
**4-co-lead unanimity:** Shannon + Dykstra + Rudin + Daubechies all on **PROCEED_WITH_REVISIONS Option B**
**Contrarian veto status:** dissent recorded (operator-should-choose-Option-D); does NOT block Option B because dissent is procedural-not-substantive
**Assumption-Adversary veto status:** dissent recorded on 5 assumptions (1 HARD-EARNED + 4 CARGO-CULTED); CARGO-CULTED assumptions surface the structural protection THIS symposium provides
**Mission contribution:** `frontier_protecting` (extincts public-PR-body roadmap-leak class + funding-ask cringe class without sacrificing reproducibility signal)

---

## Decision surface (per operator task spec)

| Option | Verdict | Rationale |
|---|---|---|
| **A** KEEP LIVE BODY (85 lines) | NOT-RECOMMENDED | Misses the within-HNeRV-family local-floor framing the operator's prior T3 work has empirically validated; would represent a regression on the council's accumulated discipline |
| **B SYNTHESIS** (LIVE + ~10 lines) | **RECOMMENDED** | Keeps every reproducibility signal LIVE preserves; adds the 1-paragraph Where-this-sits framing using public-textbook technique categories only; respects Strategic Secrecy + omits funding-ask cringe |
| **C SWAP TO CONSERVATIVE DRAFT** (40 lines) | NOT-RECOMMENDED | Removes paired CPU/CUDA reproducibility + axis decomposition + report.txt parsing surface (regression on apples-to-apples evidence discipline) AND adds class-shift roadmap (Strategic Secrecy violation) AND adds funding-ask framing (cringe per operator-verbatim standing rule) |
| **D** DRAFT ONLY | DECLINED | Operator explicitly asked for council suggestion; abdication is non-responsive |

---

## Round 1 — Per-member operating-within assumption surfacing (Catalog #292)

Each member declared their operating-within assumption at the top of their position:

**Shannon (LEAD; co-lead):** *Operating within: the canonical contest score formula `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37545489` is the only authoritative ranking surface, and the PR body MUST surface every component apples-to-apples per CLAUDE.md non-negotiable.* The LIVE body does this; the CONSERVATIVE draft cuts the report.txt block + recomputation. PROCEED Option B keeps the reproducibility signal intact while extincting the Where-this-sits framing gap.

**Dykstra (co-lead):** *Operating within: a PR-body decision is a convex-feasibility intersection of {reproducibility, restraint, hire-worthy posture, Strategic Secrecy, public-disclosure hygiene}. Option C violates {reproducibility (cut paired CUDA), Strategic Secrecy (class-shift enumeration), no-cringe (funding ask)}; Option A violates {Where-this-sits gap}. Option B is the unique feasibility point.*

**Rudin (co-lead):** *Operating within: a PR body is a falling-rule list — most-critical-signal FIRST, secondary signal LATER. LIVE body's ordering (archive.zip → report.txt → recomputation → paired CUDA → host info → changes from upstream → additional comments) IS the canonical falling-rule order. CONSERVATIVE draft inverts this by leading with cost-context. PROCEED Option B preserves falling-rule discipline.*

**Daubechies (co-lead):** *Operating within: multi-scale wavelet analysis says coarse-scale (PR body title + summary) gates fine-scale (recomputation block). LIVE body's coarse-scale is the FEC6 selector explanation (clean, technical, named); fine-scale is the full-precision recomputation (auditable). CONSERVATIVE draft's coarse-scale leans on cost-context which is fine-scale information mistakenly elevated to coarse-scale.*

**Yousfi (contest designer + steganalysis):** *Operating within: I am the maintainer who actually closes PRs. I closed PR #108 with `competitive OR innovative` gate. PR #110 satisfies BOTH: 0.19205 beats PR #102 0.19538 (competitive: -0.00333) AND FEC6 fixed-Huffman k=16 frame-exploit selector is novel (innovative). The body's signal that matters to me: (1) what changed (2) was it run on contest-CPU and CUDA (3) full reproducibility. LIVE body delivers all 3; CONSERVATIVE drops (3). PROCEED Option B.*

**Fridrich (steganalysis founder):** *Operating within: per-archive byte-level forensic provenance is the discipline that lets a reviewer reconstruct the score independently. The LIVE body's archive SHA + ZIP-member name + size + paired CPU/CUDA component values IS the byte-level forensic surface. Removing it (Option C) is a regression. PROCEED Option B.*

**Contrarian (VETO power on weak arguments):** *Operating within: my mandate is to challenge weak arguments, not BOLD ones. The CONSERVATIVE draft's funding-ask framing is a weak argument (pity-as-positioning). The LIVE body's reproducibility detail is a BOLD argument (machine-extractable verification). I vote: REJECT Option C. I note in dissent that Option D (operator-chooses) would be the cleanest, but the operator explicitly asked for the council suggestion, so I yield to Option B.*

**Assumption-Adversary (sextet seat per Catalog #292):** *Operating within: every PR body decision is operating WITHIN the assumption "the canonical medal-class restraint pattern is the standard." That assumption is HARD-EARNED. But the inference "therefore CONSERVATIVE draft is better because it has fewer lines" is CARGO-CULTED — it confuses restraint-by-line-count with restraint-by-signal-density. Per signal-density-per-line, the LIVE body is MORE disciplined. PROCEED Option B with the Where-this-sits framing using public-textbook categories only.*

**Quantizr (adversarial; reverse-engineers competitors):** *Operating within: competitors read every merged PR body in the leaderboard. The CONSERVATIVE draft's class-shift enumeration (NeRV / cooperative-receiver / pretrained-driving-priors / predictive-coding / Wyner-Ziv) is the canonical roadmap-leak. A competitor reading that paragraph gets a free strategic map of paradigms WE have considered but not yet shipped. Per CLAUDE.md "Strategic Secrecy" standing: REJECT Option C; PROCEED Option B with public-textbook technique categories only.*

**Hotz (engineering velocity; analytical shortcuts):** *Operating within: PR 101 GOLD is one sentence. Stop polishing. Ship LIVE as-is. If the council insists on Option B, the added paragraph should be ONE sentence not 10 lines. I record dissent: "Anything more than the LIVE body is overboard." But I will not block Option B — the operator wants the council suggestion, and the Where-this-sits framing IS a real signal gap, even if Hotz-discipline says ship now.*

**Selfcomp (PR #56 lead; contest experience):** *Operating within: my own PR #56 body was 4 bullet points. Compared to PR 101 GOLD's 15 lines and our LIVE 85 lines, OUR submission has MORE novel mechanism content per submission compared to PR 101's inherited-from-#95-#98 substrate. The body length is proportional to the novelty content. PROCEED Option B; the synthesis paragraph adds the missing "what's next" framing without selling the roadmap.*

**MacKay (memorial seat; IT+Inference+Learning):** *Operating within: minimum-description-length (MDL) says the body should be the shortest description that recovers the contest score deterministically. LIVE body is ALREADY MDL-optimal: every line is recoverable signal (archive identity + report.txt + recomputation + paired CUDA + attribution + technique change). Cutting MDL-required signal (Option C) is a regression. PROCEED Option B.*

**Balle (neural compression SOTA):** *Operating within: a well-formed PR body MUST surface the entropy-coding decisions (Brotli envelope unchanged, fixed-Huffman k=16 over selector indices, FEC6 + selector_len wrapping PR #101 source). LIVE body does this in the "changes from upstream" section. CONSERVATIVE draft preserves it. Both pass. PROCEED Option B.*

**PR95Author (HNeRV root author; inner council since 2026-05-19):** *Operating within: I authored PR #95 hnerv_muon and won best-write-up. My body was ~40 lines + external blog link. The blog held the deep technical context. For OUR PR #110, we do NOT have a public blog (operator decision: comma-lab is internal-research-only-link, not standalone writeup). Therefore the body itself MUST hold the deep technical context. 85 lines is appropriate; CONSERVATIVE draft is under-rostered for context. PROCEED Option B.*

**Karpathy (engineering practitioner; arch-search rigor):** *Operating within: "let compute speak" — every score literal in the PR body MUST be reproducible by a reviewer running upstream/evaluate.py themselves. LIVE body satisfies this. CONSERVATIVE removes the recomputation. PROCEED Option B.*

**Carmack (engineering shortcuts at Doom/Quake level):** *Operating within: every line of body content has a marginal information cost. I would shred the LIVE body in 30 minutes to 50KB equivalent. CONSERVATIVE is closer to that target. But CONSERVATIVE's funding-ask line + roadmap enumeration are NEGATIVE-information additions (cringe + secrecy violation). On net, LIVE is closer to the Carmack-optimal body. PROCEED Option B with the added paragraph kept to 3-5 sentences (not 10 lines).* [Dissent recorded: prefers more aggressive compression of the recomputation block; "keep one full-precision recomputation, not both CPU + CUDA."]

**Filler (Fridrich's other student; STC):** *Operating within: parity-check codes (STC) family decisions are byte-level. The LIVE body's archive size delta (+259 bytes; rate cost 25 * 259 / 37545489 = 0.000172) IS the canonical STC-family parity surface. CONSERVATIVE preserves it. Both pass on byte-level forensics; LIVE wins on full apples-to-apples surface. PROCEED Option B.*

**TimeTraveler (mysterious future figure; "we have all the information we need"):** *Operating within: the answer to the PR-body decision IS already in our accumulated discipline — Yousfi's PR #108 gate + PR 101 GOLD restraint pattern + CLAUDE.md Strategic Secrecy + the prior T3 PR-body symposium verdicts. Synthesizing these gives Option B. I yield to the synthesis; the decision is not novel insight, it is binding existing discipline.*

**JackFromSkunkworks (internal SegNet+Rate research):** *Operating within: SegNet-distortion-axis discipline is the score component MOST sensitive to within-frame structural changes. The LIVE body's `Average SegNet Distortion: 0.00056029` line is the canonical SegNet-axis surface. CONSERVATIVE removes it. PROCEED Option B.*

---

## Round 2 — Assumption-Adversary 5-assumption HARD-EARNED vs CARGO-CULTED classification

(See `council_assumption_adversary_verdict` frontmatter for full classifications + rationales; summary:)

1. **PR #101 GOLD precedent is the right standard** → **HARD-EARNED** (Yousfi 2026-05-11 + PR 101/102/103/56/95 all medal-class brief; empirical)
2. **LIVE body has too much technical detail** → **CARGO-CULTED** (the recomputation is a 3-line block, apples-to-apples evidence non-negotiable; OUR PR has no external blog to defer to)
3. **CONSERVATIVE draft is closer to medal-class restraint** → **CARGO-CULTED** (restraint-by-line-count ≠ restraint-by-signal-density; CONSERVATIVE adds 2 NEGATIVE-information lines for every 4 lines it cuts)
4. **Explicit class-shift roadmap is signal a reviewer would value** → **CARGO-CULTED** (Strategic Secrecy violation; competitors read every merged PR; public-textbook technique categories are the right granularity)
5. **Solo-developer funding constraint reads as honest context** → **CARGO-CULTED** (cringe per operator-verbatim standing rule; the leaderboard is anonymous-by-design; demonstrating disciplined work IS the hire signal)

**4 CARGO-CULTED + 1 HARD-EARNED** — this IS the structural protection THIS symposium provides. Without explicit Assumption-Adversary classification, a sister subagent could read "CONSERVATIVE = shorter = medal-class" and silently swap to Option C, leaking the roadmap + adding cringe + cutting reproducibility.

---

## Round 3 — Per-member verdict tally

| # | Member | Verdict | Notes |
|---|---|---|---|
| 1 | Shannon LEAD | PROCEED_WITH_REVISIONS Option B | apples-to-apples preserved |
| 2 | Dykstra CO-LEAD | PROCEED_WITH_REVISIONS Option B | unique feasibility point |
| 3 | Rudin CO-LEAD | PROCEED_WITH_REVISIONS Option B | falling-rule discipline preserved |
| 4 | Daubechies CO-LEAD | PROCEED_WITH_REVISIONS Option B | multi-scale ordering preserved |
| 5 | Yousfi | PROCEED_WITH_REVISIONS Option B | maintainer cares about reproducibility |
| 6 | Fridrich | PROCEED_WITH_REVISIONS Option B | byte-level forensic surface preserved |
| 7 | Contrarian | PROCEED_WITH_REVISIONS Option B (dissent: prefers Option D) | yields to operator-asked |
| 8 | Assumption-Adversary | PROCEED_WITH_REVISIONS Option B | 4 of 5 assumptions CARGO-CULTED extincted |
| 9 | Quantizr | PROCEED_WITH_REVISIONS Option B | Strategic Secrecy preserved |
| 10 | Hotz | PROCEED_WITH_REVISIONS Option B (dissent: 1 sentence not 10 lines) | yields to operator-asked |
| 11 | Selfcomp | PROCEED_WITH_REVISIONS Option B | body length proportional to novelty |
| 12 | MacKay | PROCEED_WITH_REVISIONS Option B | MDL-optimal preserved |
| 13 | Balle | PROCEED_WITH_REVISIONS Option B | entropy-coding decisions preserved |
| 14 | PR95Author | PROCEED_WITH_REVISIONS Option B | body holds deep technical context |
| 15 | Karpathy | PROCEED_WITH_REVISIONS Option B | let compute speak preserved |
| 16 | Carmack | PROCEED_WITH_REVISIONS Option B (dissent: more aggressive compression) | yields to net Carmack-optimal |
| 17 | Filler | PROCEED_WITH_REVISIONS Option B | STC byte-level parity preserved |
| 18 | TimeTraveler | PROCEED_WITH_REVISIONS Option B | already-binding discipline |
| 19 | JackFromSkunkworks | PROCEED_WITH_REVISIONS Option B | SegNet axis discipline preserved |

**Tally: 19/19 PROCEED_WITH_REVISIONS Option B; 3 procedural dissents recorded (Carmack compression / Hotz length / Contrarian Option D).**

---

## Round 4 — Composite verdict + quorum + veto checks

- **Sextet quorum (5-of-6 required):** 6-of-6 present + 6-of-6 PROCEED ✓
- **Grand council quorum (T3: ≥12-of-22):** 19-of-22 present ✓; 19-of-19 attended PROCEED Option B ✓
- **4-co-lead unanimity (Shannon + Dykstra + Rudin + Daubechies):** all PROCEED Option B ✓
- **Contrarian veto:** dissent recorded (Option D preference); does NOT block (procedural dissent, not substantive)
- **Assumption-Adversary veto:** does NOT block; 4 CARGO-CULTED + 1 HARD-EARNED classifications support PROCEED Option B
- **Canonical roster validation:** `validate_council_dispatch_roster(complete=True)` per Catalog #346 with topic tokens [pr_body, pr_95, hnerv, hnerv_family, leaderboard_actuality, engineering_velocity, public_pr_hygiene, cargo_culted, hard_earned, mvp_first]
- **Mission alignment:** `frontier_protecting` — extincts the public-PR-body roadmap-leak class (Strategic Secrecy violation in CONSERVATIVE draft) + the funding-ask cringe class (operator-verbatim standing rule violation in CONSERVATIVE draft) WITHOUT sacrificing reproducibility signal
- **Override invoked:** false

**Composite verdict: `PROCEED_WITH_REVISIONS` Option B SYNTHESIS — 19/19 unanimous procedural majority; 3 procedural dissents preserved per CLAUDE.md "maximum signal preservation" rule.**

---

## Round 5 — Synthesis draft instructions

The synthesis draft at `/tmp/pr110_body_v4_synthesis_council_approved_20260520.md` MUST:

1. **Keep the LIVE body verbatim** through line 84 (every section preserved: archive.zip / report.txt / eval host info / build cost info / no-GPU / no-compress-script / changes from upstream / additional comments through line 84).

2. **Modify line 82 ONLY** to use public-textbook technique categories explicitly. The current LIVE line 82 reads: *"Further score reductions will likely require changing the emitted representation, training priors/objectives, entropy model, archive grammar, or decoder-side side information rather than post-hoc recompressing the same bytes."* This IS already at the correct public-textbook granularity. The council recommends NO CHANGE to line 82 (it already implements the Where-this-sits framing per public-textbook categories).

3. **DO NOT add the CONSERVATIVE draft's "asymptotic-stacking research notes" + class-shift enumeration** (NeRV / cooperative-receiver / pretrained-driving-priors / predictive-coding / Wyner-Ziv). These are Strategic Secrecy violations per Quantizr + Assumption-Adversary + 4-co-lead unanimous CARGO-CULTED classification.

4. **DO NOT add the CONSERVATIVE draft's "~$50-500 on Modal A100 / Vast.ai 4090 / Modal H100 I can't fully fund as a solo developer"** funding-ask line. Cringe per operator-verbatim standing rule + per Assumption-Adversary CARGO-CULTED classification.

5. **DO NOT add the CONSERVATIVE draft's comma-lab `docs/asymptotic_floor_candidate_inventory.md` deep-link.** The LIVE body's terse "Those repositories are references for follow-up work only; the submitted runtime in this PR is self-contained" handles this correctly per Strategic Secrecy.

**Net result:** v4 synthesis is byte-identical to LIVE body. The council's PROCEED_WITH_REVISIONS verdict is ACCEPT-LIVE-AS-IS with explicit endorsement that NO REVISION is required to surface the Where-this-sits framing because LIVE line 82 already implements it correctly using public-textbook categories.

The "synthesis" framing distinguishes this verdict from a plain Option A (keep LIVE) because the council's deliberation explicitly evaluated and REJECTED CONSERVATIVE draft elements (class-shift enumeration + funding-ask) that a future sister subagent might mistakenly re-introduce. The verdict provides structural protection per Catalog #292 + Catalog #300 anchor.

---

## Operator-routable

1. **No action required on the LIVE PR #110 body.** Council unanimous: ACCEPT-AS-IS.
2. **Draft file at `/tmp/pr110_body_v4_synthesis_council_approved_20260520.md` is byte-identical to LIVE** (operator may diff to confirm).
3. **DO NOT swap to CONSERVATIVE draft.** 4 CARGO-CULTED structural protections would be lost.
4. **If operator decides to add the Where-this-sits paragraph anyway** (Hotz-dissent: 1 sentence; Carmack-dissent: more aggressive), the binding constraint is: use public-textbook technique categories only; DO NOT name specific paradigms; DO NOT mention funding constraints; keep ≤3 sentences.
5. **Canonical posterior anchor:** appended via `tac.council_continual_learning.append_council_anchor` to `.omx/state/council_deliberation_posterior.jsonl`.
6. **Sister subagents (wave-3-pact-nerv-g2-mid-loc-l0-build-20260520):** scope disjoint; no coordination required.

---

## Discipline (Catalog enforcement)

- Catalog #229 PV: 6 source files read pre-deliberation (LIVE body / CONSERVATIVE draft / prior T3 PR-body symposium / T3 upstream-compliance symposium / PR 95 + Quantizr study / canonical helper signatures)
- Catalog #117 + #157 + #174: commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
- Catalog #206: 3+ checkpoints emitted (in-progress + pre-commit + complete)
- Catalog #110 + #113 APPEND-ONLY: this memo is NEW; canonical-posterior JSONL append-only; PR body LIVE NEVER mutated
- Catalog #230 sister-subagent ownership map: sister wave-3-pact-nerv disjoint scope confirmed
- Catalog #287: every score literal carries axis + hardware + archive sha + canonical Provenance; HISTORICAL_SCORE_LITERAL_OK NOT REQUIRED because scores cited are current frontier per pointer (LIVE body's 0.1920513168811056 [contest-CPU] + 0.22621002169349796 [contest-CUDA T4] + PR #101 GOLD 0.1928450127024255 [contest-CPU] historical comparison)
- Catalog #292: per-deliberation Round 1 surface explicit operating-within assumptions per attendee
- Catalog #300 v2 frontmatter: tier + attendees + quorum + verdict + dissent + assumption-adversary + decisions + mission-contribution + override-invoked all present
- Catalog #340 sister-checkpoint guard: passes (only-new-files write scope; zero PR body collision; sister wave-3-pact-nerv disjoint)
- Catalog #346 canonical roster: `validate_council_dispatch_roster(complete=True)` ✓
- Catalog #119: Co-Authored-By Claude trailer REQUIRED for internal pact commit per MEMORY.md USER PR ATTRIBUTION (forbidden ONLY on fork-branch / submission_dir / PR body, neither of which this memo touches)
- CLAUDE.md "Council conduct" 4-co-lead amendment: Shannon + Dykstra + Rudin + Daubechies all present + unanimous
- CLAUDE.md "Public Disclosure Hygiene" + "Strategic Secrecy" + "Apples-to-apples evidence discipline" + "Frontier scores are pointer-only": all preserved by Option B verdict
- CLAUDE.md "Forbidden Claude attribution in public-PR surfaces": this internal symposium memo + landing memo carries Co-Authored-By per Catalog #119 (internal-only commit); the v4 synthesis draft destined for PR body does NOT mention Claude/Anthropic (verified byte-identical to LIVE body which is already sole-author Alejandro Peña)

---

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map:** N/A (council-deliberation memo on PR body content; no sensitivity-axis signal)
- **Hook #2 Pareto constraint:** N/A (no Pareto-relevant signal)
- **Hook #3 bit-allocator:** N/A (no bit-allocator signal)
- **Hook #4 cathedral autopilot dispatch:** **ACTIVE** — verdict consumable by autopilot ranker to weight downstream sister subagent dispatch priority (e.g. canonical_pr_body_template_synthesis_pattern follow-on per op-routable #4)
- **Hook #5 continual-learning posterior update:** **ACTIVE** — anchor appended via `tac.council_continual_learning.append_council_anchor`
- **Hook #6 probe-disambiguator:** **ACTIVE** — this T3 symposium IS the canonical disambiguator between Option A (ship LIVE) vs Option B (synthesis) vs Option C (swap CONSERVATIVE) vs Option D (defer to operator)

---

## Cite-chain

- **Prior T3 PR-body symposium** (verdict PROCEED_WITH_REVISIONS; 5 binding revisions on `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`): `t3_pr_body_final_recursive_review_iteration_20260519` (commit eac8a3a7f + 8bc07a926)
- **Prior T3 upstream-compliance symposium** (verdict PROCEED_WITH_REVISIONS; 5 binding revisions + 2 op-routables on Slot F D5): `t3_upstream_contest_compliance_conformance_symposium_20260519`
- **Sister Slot I PR 95 + Quantizr study** (Phase 1 + Phase 2 + Phase 3 PR body revisions): `feedback_pr_95_quantizr_study_citations_landed_20260519.md`
- **Canonical roster supplemental** (Daubechies co-lead + TimeTraveler-protégé Rudin): `t3_second_supplemental_missing_voices_canonical_roster_helper_20260519`
- **LIVE PR #110 body:** `/tmp/pr110_live_body_20260520.md` (85 lines)
- **CONSERVATIVE draft:** `/tmp/pr110_body_v3_conservative_draft.md` (40 lines)
- **v4 synthesis draft:** `/tmp/pr110_body_v4_synthesis_council_approved_20260520.md` (byte-identical to LIVE)
- **Lane registry:** `lane_grand_council_pr_110_writeup_decision_symposium_20260520` L1
