# DDM PF3B — 52-probe joint-improving hunt

**Axis:** `[macOS-CPU frozen-scorer advisory]`
**Authority:** research-only, local `$0`, no launch, no RG5, no campaign fire
**MAIN landing review:** REQUIRED

## Outcome

**MEASURED:** The sealed support-positive RG3 alphabet contains a strict
joint-distortion-improving physical edge. The first improvement in the fixed
52-probe order was rank 2:

`rg3.finer_event.pair523.class0_4.boundary.transient.band04.fine01.mag1`
with `NEGATIVE_ONE_QUANTUM`.

At the V19C n600 endpoint under the real receiver/composite-R/uint8 and frozen
batch-16 scorer geometry:

- Seg errors: `-127`
- `delta_D_seg = -0.00010765923394097222`
- Pose6 SSE: `-0.00968782698873838`
- `delta_D_pose = -3.3321040859846107e-7`
- `delta_D_joint = -0.00010799244434957068`

The exact fixed-owner E4 parseback price is `+860` counted bytes:

- rate delta: `25 * 860 / 37,545,489 = +0.0005726386996850674`
- physical-edge total: `delta_S = +0.00046464625533549665`
- distortion reduction per byte:
  `1.2557260970880312e-7 < 6.658589531221714e-7` break-even

Therefore this is a **real joint-distortion gain but not a total-score gain**.
It does not move the competitive pointer and is not campaign-fire eligible.
No upstream exact contest score is claimed.

## Ranked direction and spill evidence

All 52 formerly open support-positive signs received immutable pre-score
checkpoints. The order uses:

1. actual event-error direction from a sealed MS6 batch-one replay over exact
   PF2 event IDs;
2. changed SegNet argmax cells outside that event union;
3. the separate composite-R actuation footprint;
4. the AT1x n600 camera-input Pose Gram trace; and
5. deterministic event-mass and identity tie-breaks.

For the winner, the ranking replay measured 271 corrected and 22 broken event
cells (`event_delta_errors = -249`), 150 additional changed argmax cells, 638
composite-R support cells, and AT1x pose-null rank `349/600`. These are ranking
facts only. The batch-16 Seg/Pose result above is the joint verdict authority.

The first-ranked mag2 negative neighbor corrected more ranked event cells but
worsened realized joint distortion (`+0.000029303107398940636`), showing why the
full joint replay—not the ranking proxy—controls admission.

## Required neighborhood

The full sign/magnitude neighborhood is measured across PF3 and PF3B:

| Magnitude | Sign | `delta_D_joint` | E4 bytes | total `delta_S` |
|---:|---|---:|---:|---:|
| 1 | negative | `-0.00010799244434957068` | `+860` | `+0.00046464625533549665` |
| 1 | positive | `+0.00023522888273438033` | `+853` | `+0.0008032065697475925` |
| 2 | negative | `+0.000029303107398940636` | `+860` | `+0.000601941807084008` |
| 2 | positive | `+0.0004578043913604625` | `+883` | `+0.0010457578469673398` |

The mag2-positive row is predecessor PF3 custody; the other three are new PF3B
checkpoints. The stop rule left 49 lower-ranked signs unmeasured because the
delegated question was answered. No composed pair was attempted.

## Catch-and-fix receipt

The first ranking formulation incorrectly subtracted PF2/MS6 event cells from
composite-R support cells. Those are different causal coordinates: the latter
is an actuation footprint, while SegNet receptive fields can expand argmax
changes beyond it. No authority measurement had run. The three ranking
checkpoints and storage receipt were preserved and explicitly invalidated in
`.omx/research/ddm_pf3b_rank_v1_invalidation_20260725.json`; v2 restarted under
a new typed-config SHA and SSD root. No bytes were silently overwritten or
deleted.

## Verdict scope

`INSTANCE`: the SHA-bound V19C endpoint and the sealed 68 support-positive
signed RG3 probes only. The result is not an RG3-family or paradigm verdict,
not an RD1 edge price, and not a contest score. It proves that the existing
alphabet is **not** joint-distortion exhausted; it simultaneously proves this
first improving edge is rate-dominated in its current E4 representation.

No new canonical price or direction law was discovered. The run reuses
`cgauge_master_action_v1`; no canonical-equations edit is warranted.

## STORES CONSULTED

- delegated wrapped authority prompt, SHA
  `c6020b30cd22594a3954df316729623d3cb9c8756f43f0d39ba1ebe3e184082d`
- `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, and
  `docs/operating_manual_craft_handoff.md`
- `.omx/research/council_coherent_optimal_path_routing_20260725.md` §§4–6
- PF3 typed config, receipt, 16 immutable candidate checkpoints, and preserved
  E4 base frame
- RG3 assignment, PF2/MS6 assignment table, 68 source checkpoints, and event
  artifacts
- AT1x tracked receipt and full n600 gaze-contraction atlas
- V19C base archive and endpoint receipt
- frozen scorer config, SegNet/PoseNet bytes, and `gt_n600.npz`
- operator broadcast `2026-07-25T19:52:29Z` (official displayed `0.172`
  competitive frontier; `0.1910828242` is custody-local only)

## Durable evidence

- `.omx/research/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z/receipt.json`
- `.omx/research/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z/ranked_inventory.json`
- `.omx/research/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z/DAG_FEED.json`
- SSD v2 root:
  `/Volumes/VertigoDataTier/pact/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z_v2`

Pointer delta: **UNCHANGED**. Goal progress: **NOT ACHIEVED** because the exact
competitive score did not decrease.
