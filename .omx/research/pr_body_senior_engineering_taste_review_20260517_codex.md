# Draft PR Body Senior Engineering Review — 2026-05-17

This is a skeptical comma.ai-style senior-engineer and taste review of
`docs/pr_writeups/cpu_frontier_fec6_20260517.md`. It is not an attempt to speak
for George Hotz, Yousfi, or any comma.ai employee.

## Verdict

Do not post the current draft as a public contest PR body yet.

The technical core may be valuable, but the current body overstates evidence,
mixes authority axes, contains at least one score-direction error, and includes
too much internal process language for a reviewer who wants to run the archive
and understand the mechanism quickly.

The right public body should be short, runnable, and brutal about evidence:
archive, score, command, artifact, mechanism, limitations. Everything else
belongs in appendices or linked research notes.

## Blockers

### P0 — Headline CPU Claim Overstates Evidence

`docs/pr_writeups/cpu_frontier_fec6_20260517.md:16`, `:27`, and `:361` label
the result as `[contest-CPU GHA Linux x86_64]`, while `:363` says the archive
has not been validated on the actual GitHub Actions `ubuntu-latest` runner.

Rewrite until GHA/host-bot evidence exists:

```text
0.1920513169 [Modal Linux x86_64 CPU reproduction; GHA/host-bot pending]
```

The phrase "undercuts PR101 GOLD" at `:34` should be softened or made explicitly
conditional on the CPU-equivalence assumption until actual GHA evidence lands.

### P0 — Submission Gate Claim Conflicts With Compliance Evidence

`docs/pr_writeups/cpu_frontier_fec6_20260517.md:451` says the frontier-regression
and submission status are passing. The located compliance artifact
`experiments/results/fec6_cpu_submission_surface_review_20260517_codex/pre_submission_compliance_cpu.json`
is failing, with `passed=false`, missing submission files, runtime-tree mismatch,
and terminal-claim/runtime linkage blockers.

Public body rule: if the exact packet compliance JSON is red, the PR body must
say "submission surface pending" rather than "passing".

### P0 — Reproduction And Provenance Are Not One Runnable Path

The body cites multiple commits and paths:

- `docs/pr_writeups/cpu_frontier_fec6_20260517.md:207` uses `60c1f09bc`.
- `:209` uses `submissions/pr101_fec6_fixed_huffman_k16/...`.
- `:370` cites `730b52ee3`.
- `:372` cites `cf2a5e2269550406d5381b1abede0b70e28f41ce`.

The current checkout points to the archive under `experiments/results/...`, not
the documented `submissions/...` path. A reviewer will treat this as stale
unless the body gives one pinned commit, one archive path, one inflate path, and
one command that runs.

### P1 — CUDA Comparison Has A Score-Direction Error

`docs/pr_writeups/cpu_frontier_fec6_20260517.md:35` says `0.22621` CUDA is
"worse than" PR101 GOLD CUDA `0.22936`. Lower score is better, so this is the
wrong direction. This kind of small error is reputation-expensive because it
makes every downstream comparison look suspect.

Use:

```text
The same bytes also score lower than PR101 GOLD on our CUDA/T4 replay
(`0.22621` vs `0.22936`), but the stronger CUDA-family sibling is PR106
format0d at `0.20533`; this PR is still CPU-axis first.
```

### P1 — CPU/CUDA Mechanism Overclaims Causality

`docs/pr_writeups/cpu_frontier_fec6_20260517.md:111-116` and `:138-145` assign
precise portions of the CPU/CUDA split to bicubic interpolation, cuDNN, TF32,
GT decode, and selector behavior. The paired scores and output hashes support
"observed device-axis split"; they do not by themselves prove the exact causal
budget.

Rewrite §4 as:

- Observed split.
- Controlled facts with artifact paths.
- Hypotheses for remaining causes.
- What would falsify each hypothesis.

Avoid "fundamental crux" language unless the controlled toggles exist.

### P1 — Public PR Body Has Too Much Internal Process

The current body includes operator directives, Catalog numbers, cathedral
autopilot, Rudin-Daubechies autopilot, "alien tech", "magic codec", Claude/Codex
process, spend narratives, auto-memory, and recruiting/funding asks.

Those may be useful research history. They are bad first-screen PR material.

Cut or move to appendix:

- `docs/pr_writeups/cpu_frontier_fec6_20260517.md:37`
- `:167-171`
- `:303-323`
- `:386-397`
- `:400-443`

The public reviewer likely wants:

1. What score?
2. What exact bytes?
3. How do I run it?
4. What changed versus PR101?
5. Why is it contest-compliant?
6. What are the limitations?

## Taste Pass

### Skeptical comma.ai reviewer lens

The current draft reads like an internal research dossier. A production-minded
reviewer will not reward breadth if the runnable path is blurry. The strongest
version is terse and falsifiable: one archive, one command, one score table, one
novel mechanism, one limitations paragraph.

### Hotz-like taste lens

Cut the ceremony. The body should not sound like it is asking the reader to
believe a system. It should make the packet easy to run and hard to dismiss.
If a claim needs a page of institutional context, it probably belongs in a
linked note.

### Yousfi-like contest-review lens

The score axis and artifact custody have to be impossible to misunderstand.
Do not call Modal CPU "GHA" until GHA/host-bot evidence exists. Do not present
advisory/proxy evidence as submission claims. Do not bury the archive path and
exact command below theory.

## Recommended Rewrite Shape

1. **Header**: exact archive SHA, bytes, runtime SHA, CPU result, CUDA result,
   evidence status.
2. **Evidence Table**: artifact paths for CPU JSON, CUDA JSON, compliance JSON,
   archive manifest, and whether each passed.
3. **Novelty**: FEC6 is PR101 plus K=16 frame-conditional selector with
   fixed-Huffman selector stream. Keep this under 12 bullets.
4. **Reproduction**: one pinned commit, one archive path, one inflate path, one
   command. No stale `submissions/...` paths unless they exist.
5. **Limitations**: Modal CPU not yet GHA; compliance packet not yet green;
   CPU/CUDA split observed, causal budget partly hypothetical.
6. **Appendix Links**: PR106 CUDA-family sibling, outside-NeRV work, autopilot,
   theory, and production notes.

## Next Actions

1. Fix the score-direction error immediately.
2. Replace the headline axis label with Modal Linux CPU / GHA pending until
   actual GHA evidence lands.
3. Regenerate or fix the submission compliance packet before saying any gate is
   passing.
4. Rewrite reproduction around one current path that exists in the checkout.
5. Move recruiting, sponsorship, memory, and internal process material out of
   the public PR body.

## Addendum: Current Draft Refresh After New Public-Facing Sections

Reviewed against the current uncommitted draft at
`docs/pr_writeups/cpu_frontier_fec6_20260517.md` after the added macOS/MPS,
PR106 glossary, cost, collaboration, and production-transfer sections.

### P0 — The Current Body Still Contradicts Itself On Evidence Axis

Lines 16, 27, and 361 still call the primary claim
`[contest-CPU GHA Linux x86_64]`, while line 363 says the score has not been
validated on the actual GitHub Actions runner. That contradiction will be the
first thing a rigorous reviewer sees.

Public-safe wording:

```text
0.1920513169 [Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]
```

Do not call it `contest-CPU GHA` until the exact host/GHA artifact exists.

### P0 — Score-Direction Error Still Present

Line 35 still says `0.22621` CUDA is "worse than" PR101 GOLD CUDA `0.22936`.
Lower score is better. This must be fixed before any public posting.

Public-safe wording:

```text
The same bytes also improve over the PR101 CUDA replay we cite
(`0.22621` vs `0.22936`), but this packet is CPU-axis first because its
strongest evidence is the Modal Linux CPU score.
```

If PR101 CUDA provenance is not exact apples-to-apples, downgrade this to:

```text
The same bytes score `0.22621` on our Modal T4 replay; we do not promote this
as a CUDA-frontier claim.
```

### P0 — Hidden-Better-Score Contradiction At Theoretical Floor Section

Line 323 says "HNeRV-family local minimum is `~0.171`" while the body frames
`0.19205` as the current exact CPU frontier and line 279 says no outside-NeRV
work has beaten it. Since lower is better, `0.171` reads as a much better
score unless it is explicitly proxy/theoretical/non-auth-eval.

Fix by choosing one of:

- If `~0.171` is exact-eval: it belongs in the headline and this PR body is
  obsolete.
- If `~0.171` is theoretical/proxy/ablation/model-estimated: label it
  `[proxy/non-submission/theoretical]` inline and remove "local minimum".
- If it is stale: delete it from the public PR body.

### P0 — Employment/Sponsorship Ask Does Not Belong In The Contest PR Body

Lines 374-392 read as a funding and hiring appeal. That is a separate email or
appendix, not a contest submission body. It makes the technical claim look
less objective and gives a skeptical reader an avoidable reason to dismiss the
packet as self-promotional.

Remove from the PR body. At most, put one final sentence after all technical
content:

```text
Happy to discuss the engineering details or production applicability with the
comma.ai team.
```

### P1 — "Fundamental Crux" Still Overstates Causality

Lines 111-116 and 136-145 assign a precise causal budget to interpolation,
cuDNN/cuBLAS, FMA, GT decode, and selector behavior. The paired artifacts
support a device-axis split. They do not prove that exact fractional budget
without controlled toggles and ablations.

Senior-engineer rewrite standard:

- "Observed": paired CPU/CUDA component deltas, output hashes, archive SHA.
- "Controlled": toggles actually run, with artifact paths.
- "Hypothesis": explanations not yet isolated.
- "Falsifier": one command or experiment that would disprove each hypothesis.

Avoid "fundamental" unless the falsifiers have been run.

### P1 — Internal System Names Dilute The Mechanism

Lines 171-186 and 394-407 define `alien tech`, `magic codec`, `cathedral
autopilot`, and `Rudin-Daubechies autopilot`. Even with glossary disclaimers,
these names make the public body feel less like a contest artifact and more
like an internal lab notebook.

For a public contest PR, translate every nickname at first use and then use the
canonical term only:

- `format0d additive latent correction grammar`
- `per-stream entropy-coder selector`
- `dispatch ranker`
- `interpretable sparse decision/risk model`

Keep the glossary in a linked research note, not the PR body.

### P1 — Reproduction Path Still Looks Stale

Lines 207-214 point to `submissions/pr101_fec6_fixed_huffman_k16/...`, while
line 370 says the packet lives under `experiments/results/...`, and line 372
cites a different build commit. A comma.ai reviewer should never have to infer
which path is real.

Public body must have exactly one runnable path:

```text
git checkout <commit>
sha256sum <archive-path>
bash <inflate-sh> <archive-dir> <output-dir> <file-list>
.venv/bin/python upstream/evaluate.py --device cpu ...
```

If the submission packet has not been copied into a stable public path yet,
say that explicitly and do not present the body as ready.

### P1 — First-Screen Taste Is Still Too Broad

Lines 32-38 put leaderboard interpretation, CUDA sibling, causal theory, and
infrastructure all in the TL;DR. A strong reviewer wants less breadth and more
falsifiability.

Recommended first-screen shape:

1. Score table with evidence status and exact artifact paths.
2. One sentence: "This is PR101 plus a K=16 per-pair selector encoded with a
   fixed Huffman stream."
3. One sentence: "Modal Linux CPU result pending host/GHA validation."
4. One sentence: "CUDA score is included for completeness, not promoted as the
   primary claim."
5. Reproduction command.

Everything else goes below the fold.

## Perspective Review

### Senior comma.ai engineer lens

The draft should optimize for trust per minute. Current trust cost is too high:
the reader must reconcile runner-axis labels, stale paths, internal catalog
terms, unvalidated causal budgets, and a hiring/funding paragraph before they
can simply run the packet. The correct document is shorter, colder, and more
mechanical.

### Hotz-like engineering taste lens

Show the bytes, show the command, show the score. Delete ceremony. If the
mechanism is real, the smallest clear version is stronger than the grandest
version. A phrase like "fundamental crux" is less persuasive than a one-line
ablation table.

### Yousfi-like contest-review lens

Axis labels and artifact custody must be impossible to misunderstand. Modal
Linux CPU may be useful evidence, but the body cannot call it GHA while also
saying GHA is pending. The PR body should make the host-bot's job boring:
download archive, run command, compare JSON.

## Revised Posting Gate

Do not post the current body until all five are true:

1. Headline says Modal Linux CPU / GHA pending, or actual GHA evidence exists.
2. CUDA score-direction sentence is corrected.
3. `~0.171` is either removed or explicitly labeled non-authoritative.
4. One pinned commit/path/command exists and was checked in the current tree.
5. Employment/sponsorship/internal-process sections are removed from the PR
   body and moved to a separate note if still desired.
