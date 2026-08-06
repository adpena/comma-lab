# ddm_ed2 #866 entropy-descent d_seg price receipt

Date: 2026-08-05. Axis: `[macOS-CPU frozen-scorer advisory]`.
`score_claim=false`; `promotion_eligible=false`; contest-CPU/CUDA pointer
unmoved. This receipt measures the d_seg price of the `ddm_rg5` entropy-descent
token-rate step on the live qo1 own-vehicle archive; it does not promote an
archive.

## RECALL EVIDENCE

- Read `.omx/tmp/codex_runs/ed2_prompt.md` and
  `.omx/tmp/codex_runs/_common_contract.md`.
- Read `PROGRAM.md`, `HANDOFF.md`, `SYSTEM_MAP.md`, `reports/latest.md`,
  `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`.
- Verified `CLAUDE.md` and `AGENTS.md` have matching sha256
  `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c`.
- Searched the local stores for `ddm_ed2`, `#866`, `ddm_rg5`, entropy, qo1,
  fz4, pu2, and sb1. Consumed `ddm_rg5` section 6, the rg5 row JSONL, fz4/qo1
  prompt context, and the sb1 qo1 n600 receipt.
- Attempted `tools/list_canonical_equations.py --json`; it failed before
  listing equations because this environment lacks `scipy.ndimage`, imported by
  `tac.boundary_math.lane_headstart`.
- Host preflight before receipt finalization: psutil memory available 73.55 GB;
  `/Volumes/VertigoDataTier/pact` free 97.93 GiB. Scratch stayed under
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/`.

## Base Row

Baseline is qo1 `sub_auto_pairbit`:

- archive:
  `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
- archive sha256:
  `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`
- archive bytes: 357,836
- sb1 reported n600 components: `d_seg=0.00431179`, `d_pose=0.00071459`,
  rate term `0.00953073`, `S=0.7539807296911207`
- exact SegNet-decomp baseline count used below: 508,640 errors over
  117,964,800 sites, `d_seg=0.004311794704861111`

## Candidate Construction

The intended object was the rg5 section-6 entropy-descent step at `alpha=0.25`.
The direct MLX gradient path refused on this host with no Metal device, so I
used a CPU NumPy analytic derivative of the same soft-histogram entropy formula
on the recovered qo1 IX2 token lattice-center field, then requantized and
re-encoded the token frame. Non-token archive sections were preserved
byte-identically.

Candidate:

- directory:
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025`
- archive sha256:
  `4df0ec7cea34a2e57824b2e9d3e940c44a66ee4644cf71f7a81bd5b2c9f3f852`
- payload sha256:
  `1bd99939591ecc352d69c48ee2b0d38fb699743eef1dfe8ccb73918df4a3e0e3`
- materialization receipt sha256:
  `56ac6426997b3fcbf0f204c46c516fef204eb45e2877bfcaa09f58e6e3fd1f12`

Byte-side facts:

| field | value |
|---|---:|
| base archive bytes | 357,836 |
| candidate archive bytes | 350,130 |
| archive delta | -7,706 B |
| base token member bytes | 341,295 |
| candidate token member bytes | 333,589 |
| token member delta | -7,706 B |
| base entropy bits | 3.775008231295824 |
| moved entropy bits | 3.681890308206825 |
| entropy delta | -0.09311792308899891 |
| changed token values | 79,113 |
| changed pair-cells | 70,169 |
| pairs touched | 600 |
| inactive token changes | 0 |

## Receiver And Scorer

The `inflate.sh` wrapper smoke failed in this managed shell because `python` was
not on `PATH`. I therefore used the exact copied receiver module directly with
`.venv/bin/python` to inflate the archive. This is enough for the advisory
d_seg/pose price measurement, but it is one reason this row is not promotion
eligible.

Receiver evidence:

- 10-pair timing receipt:
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025/receiver_timing_10pair.json`
- 10-pair elapsed: 4.333311 s; projected n600 including init: 259.998665 s
- full raw:
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025/inflated/0.raw`
- full raw sha256:
  `1baf064bdd9a42c32b43904f225ee2719fc80e4dc44452eaa0b5d6468dab4565`
- full raw bytes: 3,662,409,600

Full n600 scorer command:

```sh
.venv/bin/python upstream/evaluate.py \
  --device cpu \
  --submission-dir /Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025 \
  --uncompressed-dir upstream/videos \
  --video-names-file upstream/public_test_video_names.txt \
  --report /Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025/evaluate_report.txt \
  --batch-size 16 \
  --num-threads 2 \
  --prefetch-queue-depth 4
```

`evaluate_report.txt` sha256:
`e765526dd34be012ea709b6e79e9f881b4b347edf8c1b7cdb1b918a20c8caa20`.
The command completed rc=0 in `8:19.49 total`.

Reported n600 components:

| field | value |
|---|---:|
| n | 600 |
| Average SegNet Distortion | 0.00449912 |
| Average PoseNet Distortion | 0.01071092 |
| archive bytes | 350,130 |
| rate term | 0.00932549 |
| rounded final score | 1.01 |

Using the exact SegNet count from the decomposition plus the reported PoseNet
component and exact archive bytes gives `S ~= 1.010325639339858`, up from
`0.7539812001772319` on the same basis.

## Exact d_seg Decomposition

SegNet-only decomposition receipt:
`.omx/research/ddm_ed2_20260805/ed2_seg_decomp.json`
sha256 `de20416e65bc645cda6542ed158936ab739ce97243c570232489520787d62e8b`.

| field | value |
|---|---:|
| total sites | 117,964,800 |
| base errors | 508,640 |
| candidate errors | 530,739 |
| net delta errors | +22,099 |
| fixed errors | 50,886 |
| introduced errors | 72,985 |
| base d_seg | 0.004311794704861111 |
| candidate d_seg | 0.004499130249023438 |
| delta d_seg | +0.0001873355441623264 |
| d_seg score cost | +0.01873355441623264 |

Per-class net error deltas:

| class | net delta errors | fixed | introduced |
|---|---:|---:|---:|
| Road | -8,372 | 27,042 | 18,670 |
| Lane markings | +11,622 | 9,742 | 21,364 |
| Undrivable | +866 | 9,611 | 10,477 |
| Movable | +15,562 | 3,479 | 19,041 |
| MyCar | +2,421 | 1,012 | 3,433 |

## Break-Even Arithmetic

The rg5 section-6 fixed comparison was `-10,441 B`, with
`W = 1.27310821533 B/flip`, so its stated break-even is:

```text
10,441 / 1.27310821533 = 8,201.188142748421 flips
```

The realized qo1 IX2 byte win here is smaller:

```text
7,706 / 1.27310821533 = 6,052.902579065160 flips
```

The measured net d_seg damage is +22,099 errors. That exceeds the fixed rg5
break-even by 13,897.811857 errors and exceeds the realized-byte break-even by
16,046.097421 errors. In score units:

```text
d_seg delta:       +0.01873355441623264
rate delta:        -0.00513110909275945
d_seg + rate net:  +0.01360244532347319
pose term delta:   +0.24274199383915289
total delta:       +0.25634443916262608
```

The R8 pose-bank guard also fails hard: `d_pose` moves from reported
`0.00071459` to `0.01071092`, `delta_d_pose=+0.00999633`.

## Verdict

**FOLDED / formulation-instance negative** for the qo1 IX2 discrete
entropy-descent step at `alpha=0.25`. The byte win is real, but the candidate
adds far more SegNet errors than the byte win can pay for and destroys the pose
bank. No archive promotion, no contest-axis claim, no pointer move.

Scope label: this folds only this `alpha=0.25` discrete entropy step on the qo1
IX2 token-lattice-center surface. It does not kill rate-aware training, the
rg5 gradient-sign finding, SMEVR-aware surrogate training, or smaller/joint
entropy schedules.

Follow-on disposition:

- FOLDED: spend no further n600 scorer on this exact candidate.
- QUEUED: if #866 is reopened, do it after the jd5 pose-base boundary and begin
  with a byte-only alpha grid plus a stratified n>=120 CPU scorer screen before
  any new n600 spend.
- FOLDED: no attempt to patch or promote the `inflate.sh` wrapper for this
  candidate because the measured row is already score-negative.
