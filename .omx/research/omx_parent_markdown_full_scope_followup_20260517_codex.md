# OMX parent Markdown full-scope follow-up - 2026-05-17

## Scope

Operator concern: relevant L5-v2 / cargo-cult / frontier-escape signal may
live outside `.omx/research`.

This pass widened the scan to every Markdown file under `.omx`, including
ignored and historical surfaces. It used:

- `find .omx -name '*.md' -type f`
- `rg --hidden --no-ignore -g '*.md' ... .omx`
- focused reads of the non-research hits that can change current routing.

Inventory at scan time:

| Surface | Markdown files |
| --- | ---: |
| `.omx/research` | 1763 |
| `.omx/auto_memory_snapshot_20260504T230223Z` | 562 |
| `.omx/context` | 28 |
| `.omx/state` | 22 |
| `.omx/tmp` | 16 |
| `.omx/plans` | 4 |
| `.omx/specs` | 1 |
| `.omx/interviews` | 1 |
| `.omx/notepad.md` | 1 |
| `.omx/release_manifest_v0.2.0-rc1.md` | 1 |

Total `.omx/**/*.md`: `2399`; outside `.omx/research`: `636`.

## Current authority remains narrow

The non-research scan adds no new score, dispatch, or promotion authority.
Current routing remains anchored by:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_lane_dispatch_claims.md`
- dated `.omx/research/*_20260517*.md` ledgers

Historical parent-scope notes are control-plane inputs and no-signal-loss
provenance, not current frontier claims.

## Newly explicit non-research surfaces

Prior parent-scope summaries named state, tmp, notepad, release, and the
auto-memory snapshot. This follow-up explicitly adds the old direct parent
subtrees:

- `.omx/context/*.md`
- `.omx/plans/*.md`
- `.omx/specs/*.md`
- `.omx/interviews/*.md`

These are mostly April Track-B / AV1-era working packets. They are stale as
frontier authority but important as bug-class memory.

## Signals preserved for current L5 / Rule #6 work

### 1. Official scorer evidence beats plausible local intuition

The `.omx/context` and `.omx/plans` files repeatedly enforce the same rule:
do not claim a win without official scorer-backed measurement, preserve the
current floor unless a measured promotion packet justifies replacing it, and
record package/smoke evidence separately from score evidence.

Routing impact: TT5L dry-run, route, doctor, architecture-lock, and source
manifest packets must remain planning/custody surfaces until paired CPU/CUDA
harvested exact eval exists.

### 2. Film-grain and preprocessing are scorer-sensitive, not cosmetic

`.omx/notepad.md` and the April context packets preserve a concrete pattern:

- removing AV1 film grain (`film-grain0`) saved bytes but caused a large
  PoseNet regression;
- changing upscale kernel to Lanczos improved score at unchanged bytes;
- explicit `bt709/tv` encode tags plus explicit `rgb24(pc)` decode improved
  the historical Track-B floor.

Routing impact: FEC6 / Rule #6 / L5 packets that touch pre/postprocessing,
color range, synthetic texture, or frame reconstruction must carry component
movement rows. Byte savings alone are insufficient; PoseNet can treat
apparently cosmetic texture/range changes as task signal.

### 3. ROI / "dynamic bucket" ideas must charge metadata and keep the main ROI

`.omx/context/dynamic-main-roi-20260405T050528Z.md`,
`.omx/context/track-b-smart-roi-prototype-20260404T150000Z.md`, and
`.omx/plans/prd-smarter-segmentation-main-roi.md` preserve an earlier version
of the "dynamic water bucket" lesson:

- naive fixed ROI failed badly despite lower bytes;
- the central/main ROI must remain mandatory;
- auxiliary ROI metadata must be bounded and charged honestly;
- official scorer path decides promotion.

Routing impact: any L5-v2 or Rule #6 side-info allocation scheme must expose
metadata bytes, component deltas, and a "main signal not dropped" invariant.
Do not treat dynamic allocation as automatically score-lowering.

### 4. Half-frame / Quantizr tricks are joint train-inflate contracts

`.omx/auto_memory_snapshot_20260504T230223Z/feedback_half_frame_breaks_posenet.md`
corrects an easy cargo-cult trap: half-frame is not globally good or globally
bad. It works only when the renderer architecture and training distribution
match the inflate-time half-frame contract. Retrofitting it onto a mismatched
warp/motion architecture broke PoseNet catastrophically.

Routing impact: Quantizr-style and L5 side-info reductions require
train/inflate distribution parity. "The public trick worked once" is not
evidence that a transplanted runtime contract is valid.

### 5. Arithmetic coding must be terminal and stream-aware

`.omx/auto_memory_snapshot_20260504T230223Z/project_codec_stacking_composition_canonical_orders_20260429.md`
and
`.omx/auto_memory_snapshot_20260504T230223Z/feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md`
preserve the same hard rule from opposite directions:

- compression order is representation / prediction / quantization /
  hyperprior / arithmetic / pack;
- arithmetic coding first is a no-op trap;
- generic zero-order arithmetic can lose to Brotli when Brotli exploits
  LZ77/context structure.

Routing impact: Rule #6 entropy-stack work must begin with section-conditioned
entropy and consumed-byte proof. Do not route "replace Brotli with arithmetic"
as a default score-lowering move.

### 6. Non-HNeRV frontier seed exists in `.omx/tmp`

`.omx/tmp/lane_g_v3_fork_clone/submissions/jas0xf_adversarial_neural_representation/training/README.md`
describes a TokenRenderer + slave NeRV + HPAC entropy stack:

- token-to-RGB master renderer;
- separate first-frame slave renderer;
- HPAC autoregressive arithmetic coding for token maps;
- CPU FP32 FiLM table for cross-GPU determinism;
- probability quantization for portable arithmetic decode.

Routing impact: this is not a current 0.192 target, but it is a real
non-HNeRV / non-PR106 local-basin design pattern worth clean reverse-engineering
intake if the L5-v2 and Rule #6 paths need a next non-HNeRV donor.

## Practical routing update

1. Keep L5-v2 / TT5L as P0, but judge rate-only side-info packets against the
   closed rate bound unless they prove component movement.
2. Keep Rule #6 A1/FEC6 bolt-ons focused on byte-closed, consumed packets with
   component rows, not selector-byte polish below the charged-byte threshold.
3. Treat film-grain, color/range, ROI, and half-frame mechanics as scorer
   contracts. Each needs train/inflate parity or a direct component-response
   proof.
4. Treat generic arithmetic-coder substitutions as suspect until section-aware
   entropy and byte-consumption evidence prove otherwise.
5. Do a clean ANR/HPAC intake note before copying any `.omx/tmp` code or
   letting the detached fork become source of truth.

## Decision

The "outside `.omx/research`" concern is valid. The missing parent-scope docs
do not change current authority, but they preserve exactly the failure modes
that can keep the project stuck in local minima: scorer-preprocess sensitivity,
unpaid metadata, train/inflate distribution mismatch, arithmetic-coder cargo
culting, and stale detached-clone signal.

This ledger supersedes any wording that implied the parent-scope scan only
covered `.omx/state`, `.omx/tmp`, `.omx/notepad.md`, release manifest, and the
auto-memory snapshot.
