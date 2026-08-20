Completed the breadth intake for PR133, PR134, and PR136; PR135 remains delegated to its sister depth intake.

The durable memo is [ddm_pi136_leaderboard_breadth_intake_20260810.md](/Users/adpena/Projects/pact/.omx/research/ddm_pi136_leaderboard_breadth_intake_20260810.md).

Key conclusions:

- [PR133](https://github.com/commaai/comma_video_compression_challenge/pull/133) is the only direct transfer. Its useful mechanism is full-n600 joint coefficient re-solving; coarse basis quantization alone contributes only a narrow secondary improvement. Exact archive custody: 190,212 bytes, SHA-256 `051baf40…cf3fab0`.
- [PR134](https://github.com/commaai/comma_video_compression_challenge/pull/134) is a genuinely different AV1-plus-sidechannel family, but the candidate is dominated. Its best exact-grid/disjoint-support idea is already implemented in Pact. Exact archive custody: 464,856 bytes, SHA-256 `9cb7c817…e9d7d`.
- [PR136](https://github.com/commaai/comma_video_compression_challenge/pull/136) reuses the PR112 adaptive coder on a PR95 HNeRV vehicle. The coder is already ours, and its checked-in producer invokes the wrong submission and is explicitly non-resumable.
- The bounded census found no PR above 136 and no new non-PR leaderboard row since the prior intake.
- No scorer or training run was launched. PR133 and PR134 archive anatomy was measured; all scores remain external bot/source evidence. PR136’s archive was unavailable.

The requested main-repository commit was blocked because the managed sandbox makes `.git` object writes read-only. The main index remains empty and the memo is untracked. A serializer-verified fallback custody commit was created:

- Commit: `00650d3a6b`
- Custody repository: `/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/ddm_pi136_commit_custody.git`
- Memo SHA-256 in both commit and worktree: `acd6a7972db2699bfb461075c437dd8b4315594ab700d27d046b5fb142188111`

## NEXT_IF_RESUMED

- **Main-repository landing — disposition: QUEUED-WITH-A-FIRE-ORDER; owner: operator or next agent with writable Git objects; consumer store: `/Users/adpena/Projects/pact` main branch; fire trigger: a clean landing window with writable `.git`.** Verify the untracked memo against the recorded SHA, safely set aside the identical copy, fetch fallback commit `00650d3a6b`, and cherry-pick it.
- **PR133 joint re-solve — disposition: FOLDED; owner: #995 roadmap owner; consumer store: #995’s governed current-base joint screen/freeze/frame-0 re-solve successor; fire trigger: the PR135/current-base producer is byte-closed, resumable, checkpointed, and owns the scorer slot.** Apply coefficient re-solving jointly with basis quantization and retain every candidate payload.
- **PR136 archive custody — disposition: QUEUED-WITH-A-FIRE-ORDER; owner: next public-intake custody owner; consumer store: `/Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/pr136/archive/` and the intake memo; fire trigger: the release binary becomes reachable or an operator supplies a mirror.** Retain, hash, and parse it without firing a scorer.

## LIVE-HYPOTHESES

- PR133’s unfinished coordinate search may still improve the current vehicle because both matched eight-pass arms were still accepting moves.
- Joint coefficient re-solving after selected basis quantization may expose a small rate/pose Pareto gain; direct quantization alone does not.
- PR134’s receiver remains useful as an independent parity reference for Pact’s existing exact-grid actuator.
- Recovering PR136’s archive could resolve its conflicting coder-size reports and decode cost, though it is unlikely to alter the transfer verdict.

## DEAD-ENDS

- Independent PR133 coarse-basis quantization is closed at the tested formulation: its atom-9 control increased pose error about 29× without coefficient re-solving.
- Treating CBQ as PR133’s principal gain is closed: the matched control attributes only 828 bytes and roughly 2.55% pose improvement to it.
- Building PR134 as a current candidate is closed at this instance/formulation: it is 277,630 bytes and 0.7686 score worse than lc2.
- Porting or retraining PR136 is closed: its coder was already absorbed, its vehicle is a PR95 reskin, and its producer is broken and non-resumable.
- This intake did not move the pointer. The own exact frontier remains lc2 at **S=0.16959899569230852, 187,226 bytes `[contest-CUDA, n600]`**.