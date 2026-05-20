# xhigh PR #110 public-surface adversarial review after freeze guard

UTC: 2026-05-20T14:24:17Z
Author: Codex
Scope: read-only review of live PR #110, hosted release surface, evidence pack, freeze guard, local artifact-sync/doc-audit memos, and PR-linked `comma-lab` docs pinned at `b7f16a081ee381803dd5d917bdaf805453fb81f3`.
Write scope honored: this new memo only. No PR #110 branch/body, `comma-lab` docs, live README, release assets, local submission files, or existing ledgers were edited.
Subagents: three xhigh read-only adversarial reviewers inspected independent surfaces; their results are integrated below, with main-thread verification where findings were material.

## Verdict

Freeze is mostly protecting the right thing: archive bytes, release digest, runtime layout, and dependency closure are not the weak points. The weak points are public wording/custody surfaces.

I recommend **one narrow freeze-break package** only if the operator is willing to touch the PR body after freeze:

1. paste the actual CPU `report.txt` block / component values into the PR body;
2. correct the false `~0.0008` HNeRV-cluster statement;
3. change "alongside `archive.zip`" to "runtime in PR; archive hosted as release asset";
4. remove or retarget the broad `comma-lab` doc links unless the transitive Claude/Codex/persona-process exposure is intentionally public;
5. remove the `num_threads: 2` -> GHA-runner-family inference.

If the operator does not want any public mutation, preserve this memo and use the response-template path below if maintainers ask.

## Sources inspected

- Live PR #110 via `gh api repos/commaai/comma_video_compression_challenge/pulls/110`: open, mergeable clean, head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`, updated `2026-05-20T14:00:48Z`, six added runtime files under `submissions/hnerv_fec6_fixed_huffman_k16/`.
- PR comments/review comments: only the submission bot comment; no review comments.
- Hosted release `fec6-frontier-submission-20260520`: asset `archive.zip`, size `178517`, digest `sha256:6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
- Evidence pack `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/`.
- Required memos:
  - `.omx/research/pr110_transitive_doc_audit_20260520T141423Z_codex.md`
  - `.omx/research/pr110_local_artifact_sync_findings_20260520T141408Z_codex.md`
  - `.omx/research/pr110_live_freeze_guard_20260520T141520Z_codex.md`
- PR-linked docs at `b7f16a...` via GitHub raw/blob API and `/tmp/comma-lab-pr110-doc-clean/docs`.
- Upstream PR template and comparison PR bodies/comments for #95/#98/#100/#101/#102/#103/#108.

## Freeze-break required

### HIGH - PR body has a false/overbroad HNeRV CPU-cluster claim

The PR body says PR #95 / #98 / #100 / #101 / #102 / #103 / PR #110 "cluster within ~0.0008 on the CPU axis." Public PR values do not support this. The `~0.0008` figure only matches PR #101 vs PR #110.

Verified public CPU-axis values / public report values:

| PR | Public CPU score signal |
|---|---:|
| #95 | ~0.1987 |
| #98 | ~0.1966 |
| #100 | ~0.1954 |
| #101 | `0.1928450127024255` |
| #102 | `0.1953791765` from rounded CPU report components |
| #103 | ~0.19487 / ~0.19488 |
| #110 | `0.1920513168811056` local claimed CPU-axis anchor |

Risk: public math rigor. This is not a runtime issue, but it is a factual claim in the PR body and linked docs (`asymptotic_floor_candidate_inventory.md:35`) that can be challenged immediately.

Recommended public correction: replace the sentence with a narrower one: "The top HNeRV-family CPU frontier moved from PR #101 `0.192845` to this packet `0.192051` (`-0.000794 [contest-CPU]`); the broader #95/#98/#100/#102/#103 family spans roughly `0.1928`-`0.1987` on reported CPU values."

### HIGH - PR body does not include the upstream-template `report.txt` block

The upstream PR template says to copy `report.txt` content into the PR body. Medal-class HNeRV PRs #95/#101/#103 include report component lines. PR #110 currently says only that `report.txt` was generated and components are included, while the linked public release exposes only `archive.zip`, and the PR submission directory contains only `inflate.py`, `inflate.sh`, and `src`.

Local report evidence exists:

```text
Average PoseNet Distortion: 0.00002943
Average SegNet Distortion: 0.00056029
Submission file size: 178,517 bytes
Original uncompressed size: 37,545,489 bytes
Compression Rate: 0.00475469
Archive SHA-256: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
Archive size bytes: 178517
```

Risk: public evidence custody and template compliance. Maintainers can still run their bot eval, but the PR body currently asks readers to trust a local/internal report not pasted or linked publicly.

Recommended public correction: paste the report block into `# report.txt` or add a short public comment containing the exact block and archive identity. If editing the body, do it with the cluster-claim correction in the same freeze-break.

### HIGH - PR-linked docs transitively expose Claude/Codex/persona-process material

Direct PR body and runtime files do not mention Claude/Anthropic/Codex. However, the PR body links pinned `comma-lab` source-map and inventory docs. Those docs link first-level public docs that explicitly discuss Claude Code, Codex, subagents, Anthropic persona-vectors, `.omx`, `CLAUDE_PUBLIC.md`, and personal/operator narrative:

- `asymptotic_floor_candidate_inventory.md:373` links `ai_assisted_inverse_steganalysis_persona_council.md`.
- `comma_lab_overview.md:87` links the same methodology memo.
- `ai_assisted_inverse_steganalysis_persona_council.md:70-82` names "Claude Code + Codex + subagents".
- `ai_assisted_inverse_steganalysis_persona_council.md:98-110` discusses Claude/Codex roles, `.omx`, `CLAUDE_PUBLIC.md`, and sister repos.
- `ai_assisted_inverse_steganalysis_persona_council.md:157-167` adds broad personal/process narrative.

Risk: public tone and attribution hygiene. This is transitive rather than in the PR body, but the PR body invites maintainers to read these links as "broader research context." That can distract from the narrow packet and conflicts with the clean sole-author public-submission posture.

Recommended public correction: either remove the two `comma-lab` links from the PR body or retarget them to a sanitized commit whose linked graph does not expose AI-process/persona material. Because `b7f16a...` is immutable, later doc cleanup alone will not change the currently linked commit.

## Future-comment / response-template

### MEDIUM - "alongside archive.zip" creates source-of-truth ambiguity

The PR body says the runtime tree is staged in the PR under `submissions/...`, "alongside `archive.zip`." Live PR files contain six runtime files only; `archive.zip` is release-hosted, not in the PR tree. The evidence pack's `runtime_layout_verdict.json` correctly states this.

Use if asked: "The runtime tree is in the PR at `ec6cc7f...`; `archive.zip` is hosted as the GitHub Release asset with SHA-256 `6bae0201...` and size `178517`. The archive is not a PR-tree blob."

### MEDIUM - public release body has stale reproduction wording

The release body still says "The companion PR will be opened" and its 60-second smoke clones `adpena/comma-lab` at `b392343d...` then runs `unzip -oq archive.zip`; GitHub API confirms that pinned `submission_dir` has no `archive.zip` and direct lookup returns 404. The live local README has since been corrected, but the public release body remains stale.

This does not affect the direct release asset URL or evaluator path. If someone follows the release page instead of the PR body, provide the current smoke using PR head `ec6cc7f...` plus the release asset download.

### LOW - PR #108 / leaderboard-open wording may need narrowing

PR #110 thanks Yousfi for "keeping the leaderboard open and clarifying the late-submission rubric (PR #108 closure)." Upstream README supports the challenge still being open; PR #108 directly supports the competitive/innovative rubric. If challenged, split those citations rather than attributing both to PR #108.

### LOW - local-floor / recompression language should stay scoped to tested artifacts

"Within-HNeRV-family local floor" and "no deliverable rate-term gain" should be read as tested-packet/repack evidence, not a universal compression claim. The linked inventory caveats help, but a response should explicitly say "for my tested HNeRV-family packet/repack attempts and matching evidence axes."

## Local-only fix

### MEDIUM - local artifact-sync findings memo is partly stale after local mirror fixes

The required local artifact-sync memo correctly identified broken local README commands and stale manifest language at `14:14Z`, but current local modified `submission_dir/README.md` has already corrected the smoke and full-score commands to use PR head `ec6cc7f...` plus the release asset. Current `archive_manifest.json` has also cleared several blocker strings.

Treat `.omx/research/pr110_local_artifact_sync_findings_20260520T141408Z_codex.md` as historical, not current authority. Remaining local mirror cleanup:

- `archive_manifest.json` still has `score_claim: false` while also carrying public PR score axes; rename/scope this field or add `public_pr_score_claim: true`.
- `audit_provenance.manifest_revision_log` still says v2 added "submission readiness false" and a "source-sync commit blocker"; add a v3 post-PR110 revision entry.
- Local README and PR body still infer GHA-family equivalence from `num_threads: 2`; keep `num_threads: 2` factual, remove the runner-family inference.
- Local README says no other shared libraries are loaded; static import evidence supports Python import closure only. Native libraries may load transitively through `torch`, `numpy`, or `brotli`.

## Paper / comma-lab cleanup

### MEDIUM - linked docs contain count drift and exact-count false authority

Examples from the pinned public docs:

- `comma_lab_overview.md:9` claims `52 substrate packages`, `47 cathedral consumers`, `235 strict preflight gates`, `11 canonical equations`.
- GitHub tree count at `b7f16a...` found `56` top-level substrate directories and `47` cathedral consumers.
- Unique `Catalog #N` tokens in `src/tac/preflight.py` counted `215`, while multiple docs use `235` or `~300`.
- `canonical_equations_tour.md:5` says six initial equations, while overview says 11 equations. This may be historical-vs-current wording, but public readers will read it as inconsistent.

Fix later by making these dated/generated snapshots or using softer wording.

### MEDIUM - inverse-steganalysis lineage claims over-attribute organizer intent

The technical analogy is useful, but several docs state organizer/scorer intent too strongly:

- `ai_assisted_inverse_steganalysis_persona_council.md:11`
- `standout_undersold_candidates_spotlight.md:62`
- `standout_spotlight_extensions_operator_pinned_20260520.md:24-27`

Fix later by saying the contest "technically resembles inverse steganalysis" unless a direct public organizer quote supports stronger intent language.

### MEDIUM - candidate spotlight claims read stronger than scaffold status

The inventory caveats are generally good, but spotlight docs make assertive claims around scaffolded/design-only lanes:

- Mamba-2 efficiency claim in `standout_undersold_candidates_spotlight.md:42`.
- RAFT/LAPose "math is clean" and `~50 KB` pose-byte elimination claim at line 46.
- SIREN "one-tenth parameter count" and `50-100KB` target at line 58.
- Lane Omega-W V2/V3 improvement/launch-ready wording in `standout_spotlight_extensions_operator_pinned_20260520.md:72-78`.

Fix later by placing "design hypothesis / not contest-anchor validated" beside each performance-like claim, not only in top/bottom caveats.

### LOW - public commit message contains internal-process wording

The second public PR commit message says it removed internal `.omx/research/...` citation because it was "operator-private state not accessible to reviewers." This is visible in commit history but not the PR body/runtime/release. Do not rewrite history solely for this; avoid similar wording in future public commits.

## Checked OK

- No direct `Claude`, `Anthropic`, `Codex`, `Co-Authored`, `/Users`, `.omx`, `tac`, or `comma_lab` dependency tokens found in the live runtime files, except harmless comments mentioning "scorer".
- Runtime static imports are stdlib plus `torch`, `numpy`, and `brotli`, with local modules `codec`, `codec_sidecar`, `frame_selector`, `model`.
- Archive SHA/size/member facts are consistent across evidence pack and live release: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`, `178517` bytes, single member `x`, `178417` bytes, ZIP_STORED, CRC `c4a71a7a`.
- Rate term math is consistent: `25 * 178517 / 37545489 = 0.11886714273451066...`.
- FEC6 selector byte math is internally consistent: source archive `178258` bytes to PR #110 `178517` bytes is `+259`; selector wire payload `249` bytes, bitstream `243` bytes / `1944` bits / `600` pairs = `3.24` bits/pair bitstream and `3.32` bits/pair wire payload.
- Live PR has only the bot comment and no maintainer review comments yet.
- No legal/compliance blocker found in the core PR runtime/release/archive custody: no hidden network fetch, no scorer load at inflate time found in static/dynamic import evidence, archive URL is curl-compatible, and author attribution in direct PR/release surfaces is Alejandro Pena / `adpena@gmail.com`.

## Minimal freeze-break text patch concept

If a public PR body edit is authorized, keep it surgical:

1. Replace `# report.txt` section with the CPU report block from local `report.txt`, including archive identity lines.
2. Replace "Runtime tree ... alongside `archive.zip`" with "Runtime tree is staged in this PR ...; `archive.zip` is hosted at the release URL above and is not a PR-tree blob."
3. Replace "cluster within ~0.0008 on the CPU axis" with "PR #110 improves PR #101's top CPU anchor by `-0.000794 [contest-CPU]`; the broader HNeRV lineage spans roughly `0.1928`-`0.1987` across public CPU reports."
4. Replace `num_threads: 2, matching the ubuntu-latest GHA runner family` with `report.txt records upstream evaluator default num_threads: 2`.
5. Either remove the `comma-lab` links or retarget them to a sanitized public-doc commit.
