# Top Submission Anatomy And Contest-Faithful Recommendations - 2026-05-03

Scope: latest local-plus-web reverse-engineering update for public PR75, PR77,
PR65, and nearby exploit/quarantine PRs. No remote jobs were dispatched. No
`.omx/state` files were read or modified. No runtime code was changed.

Evidence grade: mixed.

- Exact local score evidence is A++ only where a local T4
  `archive.zip -> inflate.sh -> upstream/evaluate.py` replay exists.
- GitHub PR bodies, public leaderboard rows, archive byte anatomy, and local
  raw-output parity are forensic/planning evidence only.

## Retrieval

Refresh time: `2026-05-03T11:57:35Z`.

Web/source URLs used:

- Official leaderboard: https://comma.ai/leaderboard
- PR75: https://github.com/commaai/comma_video_compression_challenge/pull/75
- PR77: https://github.com/commaai/comma_video_compression_challenge/pull/77
- PR65: https://github.com/commaai/comma_video_compression_challenge/pull/65
- PR73: https://github.com/commaai/comma_video_compression_challenge/pull/73
- PR78: https://github.com/commaai/comma_video_compression_challenge/pull/78
- PR70: https://github.com/commaai/comma_video_compression_challenge/pull/70
- PR75 archive URL:
  https://github.com/EthanYangTW/comma_video_compression_challenge/releases/download/qpose14-r55-segactions-minp/archive.zip
- PR77 archive URL: https://github.com/user-attachments/files/27314022/archive.zip
- PR65 archive URL:
  https://github.com/henosis-us/comma_video_compression_challenge/raw/henosis_qz_n3z_r25_clean_submit/submissions/henosis_qz_n3z_r25_clean/archive.zip
- PR78 archive URL:
  https://github.com/nick-neely/comma_video_compression_challenge/releases/download/qzs3-script-payload-r147-archive/archive.zip
- PR70 archive URL: https://github.com/user-attachments/files/27271580/archive.zip

The official visible leaderboard currently lists the top faithful accepted rows
as rounded `0.32` for PR67 `qpose14_qzs3_filmq9g_slsb1_r55`, PR65
`henosis_qz_n3z_r25_clean`, and PR63 `qpose14`; PR75 and PR77 are not visible
leaderboard rows at this refresh.

## Current Exact Frontier

C091 PR75 public replay exact T4 is still the local frontier:

- Archive: `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
- Exact eval:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.json`
- Bytes: `276481`
- SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- T4: `Tesla T4`, `gpu_t4_match=true`
- Runtime tree SHA-256:
  `c816476e31c17ed237644d554801c29964d4f564243021706dbe775a319c3472`
- Samples: `600`
- PoseNet: `0.00049371`
- SegNet: `0.00060804`
- Pose contribution: `0.07026450028285976`
- Seg contribution: `0.060804000000000004`
- Rate contribution: `0.18409725`
- Canonical score: `0.31516575028285976`

Gap to `0.3140000000000000` is `0.0011657502828597566`, equal to
`1750.7465768743152` archive bytes at unchanged components.

The current PR75 body reports a better recomputed score
`0.3143809189609485` from PoseNet `0.00048657`, SegNet `0.00060529`, and
`276481` bytes, but the exact local T4 replay above is the score truth for this
workflow. Treat the public body as hardware/runtime drift signal, not as a
promotion target.

## Public Archive Custody

Downloaded or locally mirrored public archives checked in this pass:

| Source | Status | Bytes | SHA-256 | Strict use |
|---|---:|---:|---|---|
| PR75 `qpose14_r55_segactions_minp` | closed, unmerged | `276481` | `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` | faithful exact T4 frontier |
| PR77 `qzs3_tile_delta_r147` | open | `276551` | `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af` | faithful public replay, exact T4 negative vs C091 |
| PR65 `henosis_qz_n3z_r25_clean` | closed, unmerged | `284425` | `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68` | faithful component-basin source only |
| PR78 `qzs3_script_payload_r147` | closed, withdrawn | `193` | `dc23e2e2516834c5b43803cee1c32ea08d4bf2248f36800f10a9a47a5ef4ad65` | quarantine, script-payload relocation |
| PR70 `mask_decoder` | open | `57329` | `d5046b9b64c0982adb1bd8edf35f25bd3eb7fa0180f6744ce3bbd8c139abb142` | quarantine, author admits payload moved into `inflate.py` |

PR73 `emir_flatpack` is not a usable archive source today: the PR body has no
archive link, a maintainer asked for the zip, and the copied report recomputes
to `0.3684959880089771` from PoseNet `0.00105745`, SegNet `0.00077926`, and
`281948` bytes. Its flat-model/static-schema idea is a packer signal for older
qpose14, not an immediate C091 exact-eval candidate.

Superseded but still relevant custody note: older PR67/PR75 body text named
archive SHA `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd`
at `276741` bytes. The current PR75 release asset is the `03a2...` archive at
`276481` bytes; do not conflate those bodies or scores.

## Stream Anatomy

### PR75 / C091

ZIP: one stored member `p`, payload `276381` bytes, ZIP overhead `100` bytes.

Fixed-slice stream contract:

| Stream | Charged bytes | Charged SHA-256 | Decoded bytes | Decoded SHA-256 | Contract |
|---|---:|---|---:|---|---|
| `masks.mkv` | `219472` | `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87` | `223385` | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` | Brotli AV1/OBU mask stream |
| `renderer.bin` | `55756` | `e892539adec2406f87c824accc0effc80911f160ca8d324429c5d2bac175f2cf` | `59288` | `30159b6ace27a4013d1516c340d58f6d683e6847429fd3d6303a2c650aa2abef` | Brotli QZS3 renderer |
| `seg_tile_actions.bin` | `255` | `e2f8a113ee9e4d009448fec2497602ef67cabb8a13ef8d3fa9e11d69ff8ceed6` | public wire `281`, runtime raw4 `432` | runtime `5af557cdf4c8c4c3747b06c1daabfe34581b62cb9f317d41593b836c6727427a` | SG2 grouped tile/frame delta actions |
| `optimized_poses.qp1` | `898` | `7d7c35f4e7b0eb7022e56aaa76cad111b6c2e536b68080f10a536b2cb418a082` | `1140` | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` | QP1, 600 rows |

PR75 action grammar:

- Runtime records: `108`
- Unique pairs: `106`
- Pair range: `33..598`
- Unique tiles: `21`
- Dominant tiles: `88` (`30` records), `86` (`29` records), `89` (`11`)
- Unique action IDs: `60`
- Runtime applies deltas to fake frame 2 before upscale over 32x32 renderer-grid
  tiles.

Local raw-output parity between the public PR75 runtime and `robust_current`
matched selected pairs `33`, `104`, and `598` exactly after actions. All-600
parity was not completed in that artifact, but exact T4 replay has already
superseded the selected-pair parity question for score custody.

### PR77

ZIP: one stored member `p`, payload `276451` bytes, ZIP overhead `100` bytes.

PR77 shares PR75's mask, renderer, and QP1 pose encoded stream SHAs. Only
`seg_tile_actions.bin` changes:

| Stream | Charged bytes | Charged SHA-256 | Decoded/runtime bytes | Decoded/runtime SHA-256 |
|---|---:|---|---:|---|
| `masks.mkv` | `219472` | `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87` | `223385` | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `renderer.bin` | `55756` | `e892539adec2406f87c824accc0effc80911f160ca8d324429c5d2bac175f2cf` | `59288` | `30159b6ace27a4013d1516c340d58f6d683e6847429fd3d6303a2c650aa2abef` |
| `seg_tile_actions.bin` | `325` | `d8c75e4f3725bbcf608434f0a78f5b37a9ce86bd8177c71092fd727d7e2af75a` | wire `371`, runtime raw4 `588` | runtime `8ac9a01caad973096c58b42daf2b1a8e476ad68cf285d443baa4ac94fdb42255` |
| `optimized_poses.qp1` | `898` | `7d7c35f4e7b0eb7022e56aaa76cad111b6c2e536b68080f10a536b2cb418a082` | `1140` | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |

PR77 action grammar:

- Runtime records: `147`
- Unique pairs: `121`
- Pair range: `11..599`
- Unique tiles: `24`
- Tile range: `82..140`
- Unique action IDs: `73`
- Relation to PR75 is not a pure superset: previous local comparison found
  only `56` exact action-record overlaps and `67` pair/tile overlaps.

Exact T4 replay:

- Exact eval:
  `experiments/results/lightning_batch/exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/contest_auth_eval.json`
- Bytes/SHA: `276551` /
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
- PoseNet/SegNet: `0.00049588` / `0.00060816`
- Score: `0.31537874750377204`
- Delta vs C091: `+0.00021299722091228047` worse total, with
  `+0.00016624722091226085` worse component score.

PR77 is therefore a useful negative and action-policy training signal, not a
new frontier.

### PR65 / Henosis

ZIP: one stored member `x`, payload `284325` bytes, ZIP overhead `100` bytes.

`x` layout: 30-byte 24-bit length header, then
`mask/model/pose/post/shift/frac/frac2/frac3/bias/region/randmulti`.

Key charged streams:

| Stream | Bytes | SHA-256 | Decoded bytes | Contract |
|---|---:|---|---:|---|
| `mask` | `219472` | `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87` | `223385` | same mask stream as PR75/C091 |
| `model` | `57074` | `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc` | `61590` | Henosis renderer bundle |
| `pose` | `1487` | `2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c` | `1806` | `P1D1` pose representation |
| `post` | `1400` | `c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575` | `2400` | qpost-family |
| `shift` | `226` | `48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4` | `603` | qpost-family |
| `frac` | `106` | `a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992` | `179` | qpost-family |
| `frac2` | `149` | `3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b` | `603` | qpost-family |
| `frac3` | `154` | `d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5` | `603` | qpost-family |
| `bias` | `223` | `8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b` | `603` | qpost-family |
| `region` | `273` | `4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392` | `603` | qpost-family |
| `randmulti` | `3731` | `dddf4b4fec32190b219d3e916e05bba5d1711d4208d16966929fa09cbc39349e` | `6265` | qpost-family |

Current PR65 body recomputes to `0.3196824276891214` from PoseNet
`0.00035283`, SegNet `0.00070896`, and `284425` bytes. Component-wise, that is
a strong PoseNet clue but a high SegNet/rate penalty. Existing exact T4 qpost
transfer attempts near this family were negative, including:

- `exact_eval_pr65_qpost_ix_lagtop67_p6_bias_top040_t4_20260503T1035Z`:
  score `0.3156657157109626`
- `exact_eval_pr65_qpost_ix_lagtop67_p6_bias_top080_t4_20260503T1024Z`:
  score `0.3157349872667322`
- `exact_eval_pr75_qpost_microstack_bias032_t4_20260503T1050Z`:
  score `0.3156217237878922`

PR65 should be mined for learned pose/postprocess priors, not transplanted
wholesale and not screened via the already-negative qpost mechanisms.

## Exploit/Quarantine Update

PR78 is explicitly withdrawn by its author as a payload relocation submission.
Its archive is only `193` bytes with a single `note.txt` member, while its PR
adds a `4555`-line `inflate.py`. It is not contest-faithful signal.

PR70 remains quarantine. The PR body says the author realized bytes were moved
from the archive into `inflate.py`; the PR adds a `1472`-line `inflate.py` and
the archive is only `57329` bytes. Even though the latest downloaded ZIP parses
as a single member `m`, its result depends on an unmetered script payload and
must not guide our faithful path except as validator-hardening input.

PR73 is not an exploit by inspection, but it is not a current top source: no
archive is linked, public report fields are worse than the claimed expected
score, and the implementation targets older PR63 torch/pickle overhead rather
than the current QZS3 PR75 frontier.

## Implementation Recommendations

### 1. Prioritize a C091-native action atom learner, not another PR77 replay

Why:

- PR75 actions are cheap: `255` charged bytes, `108` runtime records.
- PR77 expands action coverage to `325` charged bytes and `147` runtime
  records, but exact T4 gets worse than C091.
- PR75 public actions-only and PR77 public replay are now measured negatives in
  the C091 basin, so prefix/superset copying is exhausted.

Concrete implementation:

- Use C091 as the fixed baseline and build an offline action-atom table over
  `(pair, tile, action_id)` using exact component traces where available.
- Search non-prefix policies and small learned action dictionaries jointly.
- Require a manifest field for `changed_raw_output=true`, record overlap with
  PR75/PR77 public records, and reject duplicate `(pair,tile)` conflicts unless
  the runtime order has raw-output parity.
- Dispatch only if the predicted component score improvement is at least
  `0.0011` after rate, because sub-0.314 needs `1751` byte-equivalent from
  C091.

Immediate exact eval: no new one from this update. The active PR77 action plus
C089 pose fixed-slice job is already the relevant live probe; do not launch
another action replay unless that result returns positive component evidence.

### 2. Make QZS3 renderer self-compression the main byte lever

Why:

- C091 needs about `1751` bytes at unchanged components.
- The PR75/PR77 renderer slice is `55756` charged bytes, so a
  component-preserving `3.15%` renderer shrink is enough in principle.
- Generic nested compression probes do not beat current entropy-coded slices,
  so this is not a zip/brotli trick.
- PR73's flatpack idea removes old qpose14 `torch.save()` overhead, but C091's
  QZS3 renderer is already a custom compact wire format. The transferable idea
  is static schema plus learned/block-level quantization, not PR73's old-model
  repack as-is.

Concrete implementation:

- Continue QZS3/QZS4 or block-FP renderer shrink lanes against the C091 exact
  runtime, not PR63 qpose14.
- Before exact eval, require `preflight_trained_renderer_transplant` and
  `preflight_renderer_transplant_pose_safety` against exact source and
  candidate archive SHAs.
- Reject any renderer candidate without raw-output delta classification,
  runtime tree SHA custody, and a score-neutral fallback path.

Immediate exact eval: no. This needs a candidate that saves at least about
`1.8KB` or produces a measured component improvement while passing pose-safety
preflight.

### 3. Rebuild PR65 pose/postprocess as a learned bounded manifold, not qpost

Why:

- PR65's current body has much better PoseNet than C091, but worse SegNet and
  much worse rate.
- Existing qpost transfers on T4 are negative, so the PR65 streams are not
  directly stackable in their public encoding.
- The useful signal is the pose/postprocess basin: `P1D1` pose, `post`,
  `shift`, `frac*`, `bias`, `region`, and `randmulti`.

Concrete implementation:

- Decode PR65 and C091 pose trajectories into a bounded active subspace, then
  propose small QP1-compatible C091 perturbations with explicit component
  break-even math.
- Train or fit postprocess atoms against C091 hard pairs rather than copying
  PR65's public qpost stream. Encode only selected atoms as charged typed
  members, with no-op guards and per-pair raw-output proof.
- Treat any future PR65-derived candidate as non-dispatchable until it shows
  positive local component-trace support and avoids the already-negative qpost
  exact families.

Immediate exact eval: no. The only plausible future PR65-derived screen is a
  pose-only active-subspace candidate after local component evidence; current
  qpost candidates should not be relaunched.

## Dispatch Recommendation

No additional candidate from this update warrants immediate exact eval.

Rationale:

- C091 PR75 public replay is exact T4 frontier but still needs `1751` bytes or
  equivalent components.
- PR77 full public replay is exact T4 negative versus C091.
- PR75 body improvements do not reproduce on exact T4 replay.
- PR65 wholesale and qpost-derived transfers are worse or already negative.
- PR73 has no public archive link and its report is not frontier.
- PR78/PR70 are not contest-faithful.

The correct next action is to wait for the already active PR77-action plus C089
pose fixed-slice T4 job. If it returns positive component evidence, fold that
specific delta into the C091-native action learner. If it is negative, stop
public action replay and spend the next exact-eval budget on a preflight-clean
renderer shrink or learned pose-manifold candidate.
