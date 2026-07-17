# Submission PR draft — level-set task-space witness (v9 line)

**Status: DRAFT — no PR until every `${...}` placeholder is a value MEASURED through the final frozen
`inflate.py` on this archive's SHA, and the Final Gate passes.** Supersedes the stale Apogee-era
`docs/submission_template.md` for the witness line (that template's "GPU required: yes" and
PR100-adapter framing do NOT apply here).

**Context (binding): the contest is CLOSED (2026-07-06). This is an OPEN-SOURCE DISCLOSURE, not a
live race.** So the honest primary claim is INNOVATIVE (a novel non-HNeRV vehicle, open-sourced),
NOT competitive. "Competitive vs the leaderboard best" is a SECONDARY, conditional claim — and at
current measured state it is expected FALSE (see the score reality below). Do not lead with it and
do not let it launder a means (the method) into achieved goal-progress.

**Score reality at draft time (do not delete — it gates the competitive claim):** our SUBMITTABLE
pointer is 0.19108 [contest-CPU], which is WORSE than the public best 0.187946 (#128 author-local;
~0.187961 on the x86 CI bot — use the CI number for the actual competitive comparison, per the Final
Gate; either way 0.19108 loses). The only corpus number below the public best is the banked 0.18804
splice (PR128-on-PR110) — that is BORROWED substrate,
explicitly NON-SUBMISSION per NO-FAKE #7, and MUST NEVER appear as the witness's score. A submittable
witness row below 0.187946 does not exist yet; it is produced only by the v9c2 → byte-close → dual
exact-eval step still ahead. Until that row exists and is measured, this submission is innovative-only.

Internal rules for the public body (binding):
- PR authored/attributed to the operator (adpena). NO Claude/AI attribution anywhere — public body,
  commit messages, OR branch name (memory L62/L63). This is ENFORCED, not just stated: see the
  commit-trailer gate in the Final Gate (this repo's default commit trailers WOULD leak
  `Co-Authored-By: Claude` + a `Claude-Session:` URL if not stripped).
- No private paths, fleet IPs, provider names, or `.omx`/internal task-number internals in the body.
  Links only to the hosted release asset + (optionally) a sanitized public writeup.
- The `report.txt` field is the LITERAL `upstream/evaluate.py` output block, verbatim (the
  `=== Evaluation results over 600 samples ===` header + the exact component labels + the
  `Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = …` line) — this is what every
  accepted PR (#95/#101/#102/#103/#128) pastes. Then add ONE full-precision recompute line beneath it.
- Every score is recomputed from `upstream/evaluate.py` component output on the EXACT hosted archive
  bytes, on 1:1 contest hardware (Linux x86_64 for CPU; T4-class for CUDA). macOS/M5 CPU rows are
  `[macOS-CPU advisory]` and are FORBIDDEN in the body. Reference the CI axis explicitly (Yousfi's bot
  evals on x86 `ubuntu-latest`, `device: cpu`).
- CITATION POLICY (operator 2026-07-17, refined): cite ONLY what we actually use.
  * @AaronLeslie138's #95 is credited as a PIVOT POINT, not an ancestor: it showed what a tiny
    learned decoder could do here and revealed the limits of its own approach; our model, objective,
    and architecture were then developed originally from a different formulation (Kolmogorov/
    shortest-program framing, not Shannon rate-distortion). ONE honest carve-out (operator
    2026-07-17): the training SCHEDULE did start from #95's published staging as a base and was
    adapted stage-by-stage to the level-set witness — say so plainly; a reader of compress.sh can
    see the staging ancestry, so claiming full curriculum originality would be checkable-false.
  * @Quantizr's #55 ONLY: the stored-target conditioning concept we genuinely draw on. (#56 is
    @szabolcs-cs's selfcomp — NOT Quantizr's, and not used; do not cite it. Verified via gh.)
  * commaai's own scaffold + openpilot/comma10k priors — keep.
  * Do NOT cite later submissions (#125/#127/#128/#129 etc.) unless we SPECIFICALLY use their code.
  * BYTE-SHAVING (conditional): at byte-close we may or may not adopt others' byte-shaving methods
    depending on how the tests go. IF adopted (their code/method used), add the specific credit at
    fill time; if not adopted, cite nothing. This is a `${...}`-gated decision, not a default credit.
  Keep the tone plain and concrete — an engineer explaining, not an abstract.

## Why this draft exists now (process signal from #125/#127/#128/#129, 2026-07-12)

1. **Leaderboard placement happens on EVAL, independent of merge.** The CI bot posts full 600-sample
   eval results on PRs regardless of merge state; #125/#127/#128 were all added to the leaderboard
   though closed without merge. (The "refused for overlap" read is CONFIRMED for #129 — its own note —
   and for #128 — maintainer comment: "too much overlap with code that is already merged... we won't
   merge it unless you refactor"; inferred for #125.)
2. **Merge requires refactor-to-reuse.** #129 is #127 resubmitted importing `rhnerv_comma` modules
   instead of copying them. Our vehicle shares no code with any merged submission — so "want it
   merged: yes" is honest AND the merge should not hit the overlap concern (VERIFY via the
   code-provenance audit in the Final Gate; do not assert it un-audited).
3. **Attribution is rewarded.** #129 ships a `THIRD_PARTY_NOTICES.md` naming PRs #95/#98/#101/#110/
   #112/#125 and credits concurrent work. We do the same (list below).
4. **Hosting pattern that works:** GitHub release asset on the operator's fork, `curl -L`-able, with
   SHA-256 + byte count + real ZIP member layout stated inline.

## Final Gate (ALL must pass before `gh pr create`)

Every claim in the public body is VERIFIED against the final frozen `inflate.py` + this archive's
SHA — never asserted from the template.

- [ ] **Byte-close on the FINAL checkpoint:** `tools/levelset_byte_close_and_eval.py` full n600,
      archive bytes frozen, SHA-256 recorded.
- [ ] **`compress.sh` reproduces byte-for-byte (the public claim is gated, not asserted):** run
      `compress.sh` from scratch in a clean checkout and confirm the rebuilt `archive.zip` SHA-256
      equals `${ARCHIVE_SHA256}`. The body claims "same inputs → same bytes" and merge is a goal
      (maintainer-checked) — this must be VERIFIED, not asserted.
- [ ] **All score components recomputed from THIS archive SHA** via `upstream/evaluate.py` — seg,
      pose, rate each from the printed components of a run on the exact hosted bytes. FORBID copying
      any component (esp. d_pose) from a prior/banked run (the #205 lesson: the naive pose carrier's
      true d_pose was not reproducible through the decode — a banked `0.001610` pasted here would be
      a borrowed-number NO-FAKE #8).
- [ ] **Competitive claim is vehicle-bound and CI-to-CI:** the witness CPU number (`${WITNESS_BYTECLOSED_CPU}`)
      is the byte-close row for THIS archive SHA — never the 0.18804 splice or any borrowed row.
      Compare it against #128's **CI-recomputed** reference (~0.187961 on the x86 bot), NOT #128's
      author-local 0.187946. If the witness row does not beat the CI reference, the body ships
      innovative-only (drop the Competitive sentence).
- [ ] **Re-verify the leaderboard best at submit time** (`gh pr list` on the contest repo) — do not
      print a stale "best" if a lower score landed since.
- [ ] **Dual-axis, 1:1 hardware:** exact `[contest-CPU]` AND exact `[contest-CUDA]` (T4-class —
      Modal), both on the exact bytes, neither inferred from the other. No macOS/M5 CPU row in the
      body. **The BODY-QUOTED CPU row (and any competitive comparison) MUST come from a GHA
      `ubuntu-latest` run (operator-fork CI)** — the axis the body labels and the leaderboard ranks;
      Modal CPU (Linux x86_64) is a pre-check only, never the quoted number.
- [ ] **Determinism claim matches the SHIPPED forward:** the shipped `inflate.py` is same-host
      deterministic; cross-host portability is via the fp64 forward. VERIFY the fp32 forward is OFF in
      the shipped packet (fp32 breaks cross-host bit-identity via argmax-borderline drift; for scale,
      the benign #128 author-local-vs-CI delta measured d_seg +1.6e-7 / score +1.5e-5 — fp32 drift can
      exceed that class). The body must NOT assert cross-host bit-identity; it asserts "same-host
      deterministic; cross-host portable via fp64."
- [ ] **Device pin:** the shipped `inflate.py` pins `device="cpu"` unconditionally (no
      `cuda if available` fallback that would run on GPU on the T4 runner and produce different bytes).
- [ ] **Dependency closure (clean-container):** run `inflate.sh` in a minimal container matching the
      contest CPU image (CPU 4 / 16 GB). The real import set is `numpy, brotli, torch, scipy` — VERIFY
      every dep is installed BY THE PACKET (scipy is NOT guaranteed on the contest box; a missing
      `scipy.ndimage` = ModuleNotFoundError = non-self-contained disqualification). Record the exact
      final dep list to fill the body.
- [ ] **Inflate wall-clock < 30 min** on contest-class hardware, measured on the PORTABLE (fp64)
      packet actually shipped — not the faster fp32 path (#214 established the sub-30-min path; re-time
      the fp64 packet).
- [ ] **Forbidden-scorer-section audit (HARD):** enumerate every `0.bin` section with byte counts and
      an explicit "video-derived-but-not-scorer-OUTPUT" justification per section. PROVE no GT-argmax
      table, stored margin map, SegNet/PoseNet tensor, or GT-label table is present (any of these is a
      NO-FAKE forbidden class, not merely a rate cheat — a "per-class seed" that encodes the SegNet
      argmax partition would fail here).
- [ ] **Code-provenance audit (HARD):** diff the shipped `inflate.py` + `compress.sh` + archive coders
      against every merged AND open submission's code; record no-overlap evidence. This backs BOTH the
      "shares no code" originality claim AND merge-eligibility (the exact ground #125/#127/#128 were
      closed on) — do not assert it un-audited.
- [ ] **Real ZIP layout stated, not idealized:** `zipfile`-inspect the exact hosted `archive.zip` and
      write whatever the member/compression actually is (the live packet member `0.bin` is DEFLATED,
      not STORED). Do NOT re-pack to STORED to match prose — that inflates bytes and raises S.
- [ ] **CLEAN-DIFF PACKAGING (merge-ready with no edits or comments — the #127-rejection lesson):**
      the PR diff contains ONLY `submissions/taskspace_witness/**` — zero files outside it (no
      top-level edits, no README changes, no stray artifacts). The file set matches the MERGED
      exemplar shape (#112): `inflate.sh`, `inflate.py`, `compress.sh` (+ `compress.py`/`src/` if
      split), `LICENSE`, `README.md`, `THIRD_PARTY_NOTICES.md`, `method.md`,
      `expected_output.sha256` — and NOTHING else (no `archive.zip`, no `report.txt` — both
      gitignored upstream; no `__pycache__`, no dev/test scripts, no dotfiles). Every shipped file
      is reviewable in 30 seconds. No file duplicates ANY merged submission's code (the exact ground
      #127 was closed on; we have nothing to import from them either — new decoder family). Branch
      cut from upstream master; ONE tidy commit; trailer-free. Verify with
      `git diff --stat upstream/master...HEAD` before `gh pr create`.
- [ ] **`expected_output.sha256` produced from the shipped decode:** generate it the way the merged
      submissions do (hash of the inflated output), from THIS archive's decode — not hand-typed.
- [ ] **Companion files present + consistent:** `THIRD_PARTY_NOTICES.md` and `method.md` exist in the
      shipped submission dir (from `docs/submission_third_party_notices_draft.md` /
      `docs/submission_method_draft.md`); NOTICES matches the body's credit list AND the internal
      accounting table; the submission NAME field exactly matches the submission directory name.
- [ ] **Archive-contents sentence filled from the audit:** the body's per-section byte figures
      (`${W_BYTES}`/`${DXI_BYTES}`/`${SEED_BYTES}`) come from the forbidden-scorer-section
      enumeration above — same numbers, same audit, no hand-typed values.
- [ ] **Commit-trailer / attribution gate:** submission commits carry trailer-free messages; verify
      `git log --format='%B' <branch>` contains no `Claude` / `Co-Authored-By` / `claude.ai` before
      `gh pr create`. Verify the branch name carries no AI attribution.
- [ ] **`scripts/pre_submission_compliance_check.py --contest-final --strict`** with
      `--expected-archive-sha256` + `--expected-archive-size-bytes` + the auth-eval JSON.
- [ ] **Archive hosted** as a release asset on the operator's fork; `curl -L` round-trip verified,
      downloaded SHA matches.
- [ ] **5-turn consecutive clean-pass adversarial review** of the score + packet (CLAUDE.md
      "Submission PR gate" — stricter than the 3-pass greenup).

## The PR body (fill placeholders from MEASURED values, then paste verbatim)

FILL NOTE for `${COMPETITIVE_SENTENCE_OR_EMPTY}`: IF the measured witness GHA-CI row beats the
re-verified CI leaderboard best, it becomes exactly: `It is also competitive: ${WITNESS_BYTECLOSED_CPU}
(x86 CI) vs the leaderboard best ${CI_BEST} (#N).` OTHERWISE it resolves to EMPTY (delete the line).

````markdown
# submission name:
taskspace_witness

# upload zipped `archive.zip`
Hosted as a release asset (`curl -L` works):
${RELEASE_ASSET_URL}
sha256:${ARCHIVE_SHA256}, ${ARCHIVE_BYTES} bytes, single ZIP member `${ZIP_MEMBER}` (${ZIP_METHOD},
${MEMBER_BYTES} bytes). Rebuilt byte-for-byte by `compress.sh` (the build asserts the member and
archive SHA-256).

# report.txt
```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: ${DPOSE}
  Average SegNet Distortion: ${DSEG}
  Submission file size: ${ARCHIVE_BYTES} bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: ${RATE}
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = ${FINAL_SCORE_AS_PRINTED}
```
Exact score from the CPU report components: `${WITNESS_BYTECLOSED_CPU}` (x86 `ubuntu-latest` CI).
T4 CUDA, same archive bytes: `${SCORE_CUDA}`.

# does your submission require gpu for evaluation (inflation)?
no — inflation is CPU-only (device pinned to CPU). Same-host deterministic; cross-host portable via
the fp64 forward. Deps: ${AUDITED_DEP_LIST}, all installed by the packet. Inflation runs in
${INFLATE_MINUTES} min on the contest CPU runner.

# did you include the compression script? and want it to be merged?
yes — `compress.sh` rebuilds `archive.zip` byte-for-byte (seeded, deterministic; same inputs → same
bytes) and asserts the archive SHA-256. It's a new decoder family with its own archive grammar and
coders, so it shares no code with previously merged submissions (verified by a code diff).

# is this submission competitive or innovative? explain why
Innovative. The idea in one sentence: the decoder never learns to reproduce the video — it learns
only what the two frozen scorers measure, so its output just has to land in the same scorer decisions
as the source, and every byte goes to the SegNet class boundary and pose-relevant structure instead
of picture quality. Concretely it is a small coordinate network (an INR) with no RGB reconstruction
loss anywhere — not an HNeRV/NeRV variant. The framing is Kolmogorov rather than Shannon: `inflate.py`
is the program (free under the rules, so all the generic deterministic code lives there), and
`archive.zip` is the shortest seed we could find for it — the network weights, the per-pair ego
motion `ξ` (quantized + entropy-coded, ~${DXI_BYTES} B), and a few small per-class seeds. Training
backpropagates through the exact inflate chain (bicubic→874×1164, uint8 straight-through, the
shipped packing grid), so gradients are taken on the same bytes the evaluator will score. The
boundary loss is a margin-based surrogate from the level-set formulation. Pose isn't reconstructed
from pixels — the render is conditioned on the stored `ξ`, in the spirit of @Quantizr's #55
stored-target conditioning.
${COMPETITIVE_SENTENCE_OR_EMPTY}

# additional comments
Lineage & credit (full list in `THIRD_PARTY_NOTICES.md`):
- Contest scaffold (unchanged): `upstream/evaluate.py`, the frozen SegNet/PoseNet, the `inflate.sh`
  convention.
- What we learned from (no code or weights reused): @AaronLeslie138's #95 was the pivot point — it
  showed what a tiny learned decoder can do here, and studying it (including where its approach tops
  out) pushed us to develop our own model, objective, and architecture from a different formulation.
  Our training schedule did start from #95's published staging as a base, adapted stage-by-stage to
  what the level-set objective needs. @Quantizr's #55 stored-target conditioning is the one design
  concept we directly adapted.
${IF_BYTE_SHAVING_ADOPTED_AT_PACKAGING: add one bullet crediting the specific byte-shaving
method/code actually used, with its PR/author; OMIT entirely if none adopted.}
- Train-time priors only (never shipped, never counted): openpilot lane/camera geometry and comma10k
  class conventions.
- No code, weights, latents, or archive sections from any merged or open submission are included.
- Archive contents (per-section audit): decoder weights ${W_BYTES} B + per-pair ξ ${DXI_BYTES} B +
  per-class seeds ${SEED_BYTES} B. No scorer weights, no stored masks or labels, and no lookup table
  of scorer outputs anywhere in the archive.
Reproduce: `compress.sh` (deterministic, seeded); evaluate via the standard
`archive.zip → inflate.sh → upstream/evaluate.py` path on CPU.
${OPTIONAL_WRITEUP_URL}
````

## Internal borrowed-substrate accounting (NO-FAKE #7 — stays internal, backs the public claims)

Three-way classification: **OURS** (original, no public prior art) · **INDEPENDENT-CONVERGENT**
(ours, but comparable public prior art exists — credited, no priority claimed) · **BORROWED**
(concept or artifact from elsewhere — attributed).

| Component | Class | Notes |
|---|---|---|
| Decoder (coord-INR level-set witness) | OURS | `lever_b_generator` lineage; no NeRV/HNeRV code |
| Archive grammar (length-prefixed monolithic sections) | OURS | own layout; brotli entropy backend (public lib) |
| ξ pose coder (`xi_pose_coder`, derive-H) | OURS | #257; sidecar-SHAPED, joint-descent VALUES |
| Curriculum (event-gated witness-native) | BORROWED-BASE, ADAPTED | staging started from #95's published schedule as base (credited publicly per operator 2026-07-17); stages rederived/adapted to the level-set energy (#302/#430) — the gates, events, and values are ours |
| Boundary surrogate (level-set margin) | INDEPENDENT-CONVERGENT | related `sigmoid(-margin/τ)` exists in public PRs (#127/#129); per the citation policy we neither cite nor claim priority publicly — no code used, no comparison raised |
| Exact-chain training (train/pack-gap closure) | INDEPENDENT-CONVERGENT | our eval_roundtrip discipline is a long-standing CLAUDE.md non-negotiable; comparable exact-grid framing exists publicly (#125/#127/#129) — internal record only, not cited publicly |
| Stored-pose conditioning CONCEPT | BORROWED (concept) | Quantizr #55/#56 concept; our implementation + joint descent |
| openpilot/comma10k priors | BORROWED (commaai public) | train-time only, rule-118 free side |
| Scorers/eval harness | BORROWED (contest upstream) | unmodified (pinned snapshot) |

**Verdict this table supports:** the innovation claim rests on the OURS rows — the vehicle itself (a
non-HNeRV task-space witness), its archive grammar, its ξ coder, its witness-native curriculum. The
INDEPENDENT-CONVERGENT rows claim NO priority over the public PRs and are not load-bearing for
originality (the vehicle stands without them). The BORROWED rows are concepts + the contest scaffold,
all attributed. This is the opposite pole from the banked 0.18804 splice (PR128-on-PR110 = borrowed
substrate, NEVER submitted as ours).
