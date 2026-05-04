# Public Leaderboard Follow-Up Intake - 2026-05-04

Scope: follow-up public GitHub intake only. No remote dispatch was performed in
this pass. Current internal confirmed A++ anchor remains PR95 stemperm:
`0.23089404465634825`, archive bytes `178277`, archive SHA-256
`e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`.
PR98 and PR99 exact T4 replays are already queued separately and are the score
truth for those public claims.

## Source Artifacts

Local follow-up intake directory:

- `experiments/results/leaderboard_intel_20260504_followup_codex/open_prs.json`
- `experiments/results/leaderboard_intel_20260504_followup_codex/latest_all_prs.json`
- `experiments/results/leaderboard_intel_20260504_followup_codex/pr90_99_intake_summary.json`
- `experiments/results/leaderboard_intel_20260504_followup_codex/public_score_claims_recomputed.json`
- `experiments/results/leaderboard_intel_20260504_followup_codex/pr95_99_static_runtime_archive_intake.json`
- `experiments/results/leaderboard_intel_20260504_followup_codex/pr95_98_99_0bin_segment_profile.json`

GitHub sources queried:

- https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=open&per_page=100&sort=updated&direction=desc
- https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=all&per_page=20&sort=created&direction=desc
- https://github.com/commaai/comma_video_compression_challenge/pull/98
- https://github.com/commaai/comma_video_compression_challenge/pull/99
- https://github.com/commaai/comma_video_compression_challenge/pull/97
- https://github.com/commaai/comma_video_compression_challenge/pull/96
- https://github.com/commaai/comma_video_compression_challenge/pull/95

As of the refresh, the newest PR is `#99`; there is no public PR100+ yet.
Final lightweight refresh at `2026-05-04T09:48:52Z` showed #99 retitled from
`hnerv_muon_lc submission (0.20)` to `hnerv_muon_lc submission (0.19667)`;
the maximum public PR number remained #99.

## Current Public Threat Table

| PR | Public claim | Recomputed from body | Bytes | Archive SHA-256 | Runtime shape | Risk / note |
| --- | ---: | ---: | ---: | --- | --- | --- |
| #98 `hnerv_muon_finetuned_from_pr95` | `0.1963` | `0.19625777542725248` | `178392` | `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb` | HNeRV, one stored `0.bin`, PR95-style `meta+decoder+latents`, CUDA inflate claimed required | Highest threat. Metric fine-tune plus fixed decode postprocess: frame0 red -1, frame0 blue -1, frame1 green -1. Needs exact T4 replay because GPU inflate and public score is from submitter log. |
| #99 `hnerv_muon_lc` | title/text `0.19667` | `0.19668072586615531` | `178546` | `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb` | HNeRV, one stored `0.bin`, schema-driven decoder, fp16 scales, latent correction sidecar | Highest threat if T4-valid. Body claims CPU, runtime selects CUDA when available and pip-installs brotli if absent. Contest-compliance review should focus on dependency closure and runtime tree hash. |
| #97 `vibe_coder_final_boss` | `0.23` | `0.22878228922005306` | `197160` | `6785a84879d3e3395bbf990b980fe32182fca7255c5b8559dcdaac9da7516642` | one stored `p`; arithmetic/range mask C++ helper compiled at inflate; FP4 model; pose/warp sidecar | Near our anchor but not below PR98/99 claims. Compliance risk is runtime C++ compilation and dependency/toolchain closure. Existing duplicate exact replay should resolve score truth. |
| #96 `rem2_HNeRV` | `0.21` | `0.20567121179282477` | `186631` | `2ecbd2118bebdb5566f719ed538a89c4608ccab19c9edc7ae7a6de778bd42b46` | three-member ZIP: `decoder.bin`, `latents.bin`, `p`; HNeRV decoder | Threat is below our A++ anchor by public claim, but above PR98/99. It is structurally simpler and should be used as a fallback HNeRV anatomy baseline. |
| #95 `hnerv_muon` | text `0.1987` | `0.1987048012202245` | `178417` | `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a` | HNeRV, one stored `0.bin`, PR95-style `meta+decoder+latents` | Public source family for our stemperm anchor and PR98/99. Existing internal stemperm repack is A++ at `0.23089404465634825`, so public PR95 exact behavior differs from our stemperm runtime/bytes and must not be conflated. |

Other open PRs remain behind these threats by public claim: #92 recomputes to
`0.2587078229986317`, #91 exact text/recompute `0.24879480490416128`, #90
recomputes `0.2788721801656914`, #93 recomputes `0.3204744681076375`, and
#94 is MPS-local `0.33425141289817706`.

## Archive And Runtime Anatomy

PR95/PR98 both use a single stored `0.bin` with:

- `meta_brotli_len=80`
- PR95 decoder `162349`, latent `15868`
- PR98 decoder `162343`, latent `15849`
- same reported model meta: `n_pairs=600`, `latent_dim=28`,
  `base_channels=36`, native eval size `384x512`

PR98's public win over PR95 is therefore not a large packer change. It is
mostly scorer-aligned fine-tuning plus tiny decoder/latent byte shifts and the
three fixed channel postprocess operations in inflate.

PR99 changes the `0.bin` grammar:

- `decoder_brotli=161883`
- `scale_fp16=56`
- `latents_brotli=15868`
- `correction_brotli=615`

This is the immediate deconstruction target: it saves about 466 bytes in the
decoder segment, pays 615 bytes for a latent correction sidecar, and claims a
net score gain from single-dimension per-pair latent nudges.

PR97 is a different family: fixed single `p` payload with arithmetic/range mask
decode, FP4 model, optimized pose stream, and frame-1 warp sidecar. A prior
static intake split it as mask/model/pose/sidecar; it remains useful as a
stacking-design source, not the immediate score leader.

## Compliance Risks To Carry Into Replay

- PR98: GPU inflation required by submitter; public score must be T4 replayed.
  The decode-side channel postprocess is charged in fixed runtime code, not in
  bytes, so runtime tree hash must be preserved exactly for any comparison.
- PR99: body says no GPU is required, but runtime uses CUDA when available.
  `inflate.sh` and `inflate.py` try to install `brotli` if absent; final
  compliance packet must prove dependency closure or vendor the dependency via
  accepted contest environment, not rely on network during evaluation.
- PR97: runtime C++ compilation is a toolchain dependency. Treat as
  contest-current public forensics until exact replay proves the toolchain and
  runtime budget are acceptable on T4.
- PR96: simpler HNeRV replay surface; lower compliance risk than PR97, but
  public archive uses multiple members rather than PR95-style single `0.bin`.

## Top Actionable Deconstruction Steps

1. Harvest queued PR98 and PR99 T4 exact replays before doing any speculative
   work. If either validates below `0.23089404465634825`, immediately rebuild
   final packet from a clean sanitized runtime tree and exact archive bytes.
2. For PR98, isolate the score contribution of the three fixed channel
   postprocess operations by building deterministic variants:
   no postprocess, each single channel, each pair of channels, and all three.
   This is a tiny finite policy space and directly answers whether the trick
   is scorer-side color bias or the fine-tuned HNeRV weights.
3. For PR99, port the latent-correction sidecar grammar into our profiler and
   run local byte/parity deconstruction against PR95/PR98. The key question is
   whether per-pair one-dimensional latent deltas are an additive water-fill
   surface we can re-optimize on top of PR98 rather than merely copy.
4. Compare PR98 and PR99 decoded outputs pair-by-pair once exact replay artifacts
   are available. Target hard pairs where PR99 improves PoseNet over PR98 while
   preserving SegNet, because that suggests cheap latent atom corrections.
5. Keep PR97 as the orthogonal stack source: range-mask arithmetic coding,
   FP4 model packing, pose bits, and warp sidecar. Do not spend score-truth GPU
   on PR97-derived variants until PR98/PR99 replay determines the new anchor.

## Verification

- Mirrored public PR metadata, file lists, commit check-runs, raw runtime files,
  and public archive ZIPs for PR95-PR99 into the scoped follow-up directory.
- Recomputed public score claims from PR body components.
- Statically inspected ZIP member names, member sizes, SHA-256s, duplicate
  names, hidden files, and zip-slip indicators.
- Compiled mirrored Python runtime files with `py_compile`; removed generated
  `__pycache__` directories afterward to keep artifacts clean.
- No remote GPU jobs were dispatched in this pass.
