# Codex Findings — Procedural Generation Compliance Authority

date: 2026-05-19
author: codex
status: source-authority memo + implementation-routing guidance
related code: `src/tac/procedural_codebook_generator/authority.py`

## Executive Verdict

Procedural generation from a seed, charged weights, or other deterministic
archive state is promising and contest-compatible when the score-affecting
datum is charged through `archive.zip` or derived from an already charged
archive member.

The safe default is:

1. **Archive seed mode**: seed bytes live in `archive.zip`; inflate expands them
   deterministically; dispatch authority requires self-contained replay,
   scorer-free inflate, and no untracked inflate-time side inputs.
2. **Weight-derived mode**: seed is derived from a charged archive member such
   as renderer weights or a compressed latent stream; the source member SHA and
   derivation scope must be frozen.
3. **Script literal mode**: a score-affecting seed literal in `inflate.py` is a
   separate probe-only variant when it is per-video payload. Generic decoder
   constants may affect score through normal decoding, but remain proof-gated
   on self-contained scorer-free replay and exact eval.

This preserves the high-upside path while avoiding the known script-side
payload-smuggling loophole class.

## Upstream Rule Surface

Primary upstream rule text is in `upstream/README.md`:

- Submissions include a download link to `archive.zip`, `inflate.sh`, and
  optional compression/code/model assets.
- Official evaluation runs `evaluate.sh --submission-dir ... --device cpu|cuda|mps`.
- Official inflation has a 30 minute time limit.
- External tools do not count toward compressed size unless they use large
  artifacts; the README explicitly says this applies to PoseNet and SegNet.
- Compression may use anything, including models and the original video.
- Final ranking uses the public leaderboard; no hidden/private test set.

Source: `upstream/README.md` and
https://github.com/commaai/comma_video_compression_challenge#rules

## Host / Maintainer Comment Authority

### 1. One-video overfitting is allowed

Issue #34 asked whether final judging uses only `0.mkv`, all public test
videos, or a private holdout. YassineYousfi answered that only `0.mkv` is used
for final ranking and overfitting to `0.mkv` and the nets is part of the
challenge.

Authority implication: `contest_one_video_replay` is not disallowed by itself.
Procedural seeds and per-frame/per-pair streams derived from the scored video
are in-family if self-contained and exact eval validates them.

Source: https://github.com/commaai/comma_video_compression_challenge/issues/34#issuecomment-4196041941

### 2. Full scorer use at inflate requires charged weights

PR #35 (`tensor_inversion`) and PR #54 (`pixel_oracle`) optimized pixels
through frozen SegNet/PoseNet at inflate. The maintainer replied by citing the
rule that large artifacts, including PoseNet and SegNet, must be included in
the archive.

Authority implication: a full scorer receiver is not free just because the
evaluation environment has scorer weights. Any scorer-like or learned receiver
must be charged if it uses large learned artifacts. A tiny distilled transducer
can be plausible only if its weights/tables are charged and it is not a generic
PoseNet/SegNet substitute.

Sources:

- https://github.com/commaai/comma_video_compression_challenge/pull/35#issuecomment-4198642595
- https://github.com/commaai/comma_video_compression_challenge/pull/54#issuecomment-4274999328

### 3. Script-side payloads are known loophole territory

PR #36/#38 (`loophole_test`) read the original video from the repo path while
shipping a tiny archive. PR #68 (`loophole_v2`), PR #70 (`mask_decoder`),
and PR #78 (`qzs3_script_payload_r147`) moved score-affecting payloads into
script/source rather than `archive.zip`. PR #87 (`100_bytes`) used a
tiny/dummy archive and script-side content; YassineYousfi commented that
scoring-script gaming is easy and would be fixed in the next round.

Authority implication: a score-affecting `inflate.py` literal seed is risky
when it substitutes for archive bytes. It must not be treated as equivalent to
an archive seed. Keep it as a probe-only variant, with `score_claim=false`,
`promotion_eligible=false`, and an explicit compliance-ruling blocker.

Sources:

- https://github.com/commaai/comma_video_compression_challenge/pull/36
- https://github.com/commaai/comma_video_compression_challenge/pull/38
- https://github.com/commaai/comma_video_compression_challenge/pull/68
- https://github.com/commaai/comma_video_compression_challenge/pull/70
- https://github.com/commaai/comma_video_compression_challenge/pull/78
- https://github.com/commaai/comma_video_compression_challenge/pull/87#issuecomment-4367870339

### 3b. Original-video side input is a separate forbidden class

PR #36/#38 are not just "small archive" cases; they read `videos/0.mkv`
directly from the repository during inflate. That is distinct from allowed
compression-time overfitting. It turns the original video into an uncharged
inflate-time side input.

Authority implication: a procedural generator may be trained from the video at
compression time, but the submitted inflate runtime must not read the original
video, scorer files, local caches, network URLs, or undeclared sibling payloads
as score-affecting inputs.

### 4. Public PRs are the evaluation format

Issue #28 asked whether private submissions were acceptable. YassineYousfi said
the intended format is public Pull Requests and no private submissions.

Authority implication: hidden local sidecars, private model stores, host caches,
and untracked data are not contest authority. Procedural generation must be
reviewable from the PR packet and the external `archive.zip` URL.

Source: https://github.com/commaai/comma_video_compression_challenge/issues/28#issuecomment-4189310730

### 5. Heavy PR assets may be hosted outside the repo, but score-affecting data
still must be in the archive

Maintainer comments on PRs #67/#71/#74/#86/#102 asked submitters to host large
assets or zip files outside the repository to keep the repo lightweight. This
does not mean score-affecting data can live outside the submitted archive for
evaluation; it means the archive is linked from the PR instead of committed.

Authority implication: external hosting is transport, not a rate escape. The
scored `archive.zip` bytes remain the charged payload.

Example sources:

- https://github.com/commaai/comma_video_compression_challenge/pull/86#issuecomment-4367887967
- https://github.com/commaai/comma_video_compression_challenge/pull/102#issuecomment-4372708131

## Procedural Generation Authority Ladder

### Mode A — `archive_member_seed`

Verdict: canonical promotion path.

Required proof stack:

- seed member exists inside `archive.zip`;
- archive SHA-256 and byte count recorded;
- seed selection scope recorded;
- no original-video, local cache, network, scorer file, or undeclared sibling
  payload is read at inflate time;
- mutating seed bytes changes generated bytes;
- runtime consumes the seed member;
- full-frame inflated output changes under distinguishing seed mutation;
- exact auth eval validates the final packet.

Implementation hook: `classify_procedural_seed_authority("archive_member_seed")`.

### Mode B — `archive_member_weight_derived`

Verdict: canonical promotion path when the source member is already charged.

Required proof stack:

- source member exists inside `archive.zip`;
- source member SHA-256 is frozen;
- derivation scope is recorded;
- no-new-members/no-new-bytes proof exists for the derivation;
- no untracked inflate-time side input is consumed;
- runtime consumes the source member;
- full-frame output proof exists;
- exact auth eval validates the final packet.

Implementation hook:
`classify_procedural_seed_authority("archive_member_weight_derived")`.

### Mode C — `inflate_py_literal_seed`

Verdict: pursue as a separate probe, not as promotion authority.

If the literal is score-affecting, it is in the same risk family as PR #68/#87
script-side payload smuggling. It can be researched because it may clarify the
boundary, but it must carry:

- `research_only=true`;
- `score_claim=false`;
- `promotion_eligible=false`;
- `ready_for_exact_eval_dispatch=false`;
- paired archive-seed control;
- explicit compliance-ruling blocker.

If the literal is genuinely generic decoder code and not per-video payload, it
has a lower-risk classification even if it affects the output score through
normal decode semantics. It still needs self-contained archive/fixed-code proof,
scorer-free inflate proof, no-external-state proof, packet-compiler target
declaration, and exact eval before any score claim.

Implementation hook:
`classify_procedural_seed_authority("inflate_py_literal_seed", score_affecting=...)`.

## Recommended Next Prototype

Build both versions behind one probe-disambiguator:

1. `archive_seed` candidate: `archive.zip` contains `seed.bin`; inflate expands
   a small codebook or byte transducer deterministically.
2. `script_seed` candidate: `inflate.py` contains the same seed literal; mark
   it probe-only and blocked on compliance ruling.

Compare:

- archive bytes and SHA-256;
- generated codebook SHA-256;
- mutated seed output-delta proof;
- full-frame inflate output aggregate SHA;
- exact eval only for the archive-seed candidate unless the script-seed
  compliance blocker is cleared.

This lets us pursue the operator's promising seed/procedural path aggressively
without losing contest authority.

## Patch Landed With This Memo

`src/tac/procedural_codebook_generator/authority.py` now encodes the split:

- archive seed and weight-derived seed are default promotion carriers once
  runtime proof + self-contained/scorer-free/no-external-state proof + exact
  eval exist;
- score-affecting per-video `inflate.py` literal seed is probe-only;
- generic decoder constants have a separate, lower-risk authority class even
  when they affect score through normal decode semantics.

Focused tests: `src/tac/tests/test_procedural_codebook_generator.py`.
