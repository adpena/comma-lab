# Contest Compliance Authority

This document records how the repository establishes contest-compliance
authority for archive design, procedural generation, deterministic packet
compilation, and scorer-aware inflate paths.

It is not a score ledger. It is a reading protocol for the upstream contest
rules and public PR precedent.

## Authority Ladder

| Rank | Source | Repository use |
|---|---|---|
| 1 | Upstream README submission format and rules | Primary rule text for what a submission contains and what must count toward rate. |
| 2 | Upstream `evaluate.sh` / `evaluate.py` behavior | Operational meter for exact auth eval, not a license to exploit omissions in metering. |
| 3 | Maintainer comments on public PRs | Clarifies ambiguous rule applications, especially scorer/model use during inflate. |
| 4 | Public PR outcomes and withdrawals | Negative precedent for loopholes and payload relocation patterns. |
| 5 | Local `tac` / `comma_lab` strict policy | Fail-closed guardrail used before score claims, releases, or public writeups. |

When these conflict, use the stricter interpretation until a new upstream
maintainer ruling or exact auth-eval rule change supersedes it.

## Upstream Rule Reading

The upstream README says a submission includes a download link to `archive.zip`
and an `inflate.sh` that converts the extracted `archive/` into raw video
frames. It also says external libraries and tools do not count toward compressed
size unless they use large artifacts such as neural networks, meshes, or point
clouds; those artifacts should be included in the archive and counted toward
compressed size. The README explicitly says this applies to PoseNet and SegNet.

Local consequence:

- `archive.zip` is the charged payload.
- `inflate.sh` / `inflate.py` are runtime code, but not a hiding place for
  score-bearing payloads.
- Compression-time use of the original video, scorers, and other assets is
  allowed by the upstream rules.
- Inflate-time use of large artifacts, including PoseNet or SegNet weights,
  requires those artifacts to be in the archive and counted.
- Exact auth eval is necessary but not sufficient; a PR can run under the meter
  and still be a loophole pattern that should not be promoted.

## Public PR Precedents

| PR | Pattern | Authority signal |
|---|---|---|
| [#35 tensor_inversion](https://github.com/commaai/comma_video_compression_challenge/pull/35#issuecomment-4198642595) | Inflate through frozen evaluation networks. | Maintainer repeated the README rule that large artifacts, including PoseNet and SegNet, must be included in the archive. |
| [#54 pixel_oracle](https://github.com/commaai/comma_video_compression_challenge/pull/54#issuecomment-4274999328) | Test-time pixel optimization through frozen SegNet/PoseNet. | Maintainer again pointed to the same rule. |
| [#32 gradient_optimized_av1](https://github.com/commaai/comma_video_compression_challenge/pull/32#issuecomment-4191834778) | Initial plan used SegNet during inflate. | Submitter accepted that model weights must be bundled, then moved scorer-derived labels into archive bytes instead. |
| [#36 loophole_test](https://github.com/commaai/comma_video_compression_challenge/pull/36) / [#38 loophole_test](https://github.com/commaai/comma_video_compression_challenge/pull/38) | Read original video from repo at inflate time. | Closed loophole pattern; exact low score from evaluator mechanics is not contest-faithful authority. |
| [#68 loophole_v2](https://github.com/commaai/comma_video_compression_challenge/pull/68) | Moved compressed payload into `inflate.py` instead of `archive.zip`. | Explicit joke/proof-of-concept payload relocation; closed and should not be used as a leaderboard pattern. |
| [#78 qzs3_script_payload_r147](https://github.com/commaai/comma_video_compression_challenge/pull/78#issuecomment-4365772376) | Script-side payload relocation with tiny archive. | Submitter withdrew as a rules-interpretation payload relocation submission. |
| [Issue #33 PoseNet sensitivity](https://github.com/commaai/comma_video_compression_challenge/issues/33#issuecomment-4193346639) | Scorer sensitivity discussion. | Maintainer encouraged using scorer behavior to advantage; this supports scorer-aware compression, not uncharged scorer/model payloads. |

## Procedural Generation Rule

Procedural generation is contest-relevant and should remain a first-class TAC
design path. The compliant question is where the information lives.

Use two explicit modes:

| Mode | Payload location | Local authority |
|---|---|---|
| `archive_seeded` | Seed, weights, lookup tables, and score-bearing parameters live in `archive.zip`. | Preferred for score claims; charged bytes are explicit. |
| `runtime_constant` | Tiny constants or code-generation logic live in `inflate.py`. | Allowed only for decoder logic or negligible implementation constants; not allowed for relocating score-bearing payloads. |

If a seed, table, distilled transducer, generated code blob, or model parameter
set materially determines the reconstructed frames, treat it as score-bearing
unless proven otherwise. For score claims, score-bearing information must be charged through `archive.zip`. The safest route is to put it in `archive.zip`,
meter it, and make `inflate.py` a deterministic interpreter for the charged
payload.

## Deterministic Packet Compiler Rule

The deterministic packet compiler path is valid when it remains byte-closed:

1. Inputs that determine output frames are in `archive.zip` or are public,
   allowed runtime tools.
2. Runtime code is deterministic and self-contained.
3. Any trained-model behavior distilled into tables, byte transducers,
   generated code, or per-frame/per-pair streams is either archive-charged or
   justified as decoder logic in a compliance memo.
4. Exact auth eval validates the exact archive/runtime pair.
5. The candidate carries archive SHA-256, runtime tree SHA, inflated output
   custody when available, and CPU/CUDA axis labels.

This rule keeps aggressive specialized decoders available while blocking the
known payload-relocation loopholes.

## Local Promotion Gate

Before a procedural, scorer-aware, or deterministic compiler candidate can be
used as a score-bearing row, require:

- `tools/check_tac_terminology.py --strict`
- `scripts/pre_submission_compliance_check.py --contest-final --strict ...`
- exact archive bytes and SHA-256
- runtime tree SHA
- `[contest-CPU]` and `[contest-CUDA]` labels kept distinct
- compliance note that names one of `archive_seeded` or `runtime_constant`
- explicit statement that no uncharged large artifact, original-video read,
  hidden script payload, or unmetered scorer weight is used at inflate time

If any item is missing, the row can be research/probe evidence, but not a
leaderboard or submission-ready claim.

## References

- Upstream rules: <https://github.com/commaai/comma_video_compression_challenge#submission-format-and-rules>
- Upstream evaluation entry point: `upstream/evaluate.py` and `upstream/evaluate.sh`
- Local terminology authority: `docs/terminology_and_boundaries.md`
- Local package boundary: `src/tac/README.md` and `src/comma_lab/README.md`
