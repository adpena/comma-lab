# Current Focus — 2026-04-10 01:35 CDT

## Floor
- **Promoted honest floor**: `1.51` from `robust_current-standard-h64-long2500-promoted-cpu-2026-04-10`
- **Previous floor**: `1.73` from `long1000_h64`
- **Public target to beat**: last verified public first was `1.89`

## Newly confirmed winner
- `standard_h64_long2500`
  - authoritative current_workflow score: `1.51`
  - PoseNet `0.01229283`
  - SegNet `0.00579903`
  - bytes `864,167`
  - saved best local scorer continued improving to `3.443498338063558` at epoch `1303`
  - evidence root:
    - `reports/raw/2026-04-10-standard-h64-long2500-authoritative/`

## Active remote lanes
- Kaggle:
  - `adpena/comma-lab-segnet-attack-fixed-h32` is the only live Kaggle kernel
  - status: `.omx/status/kaggle-segnet-attack-fixed-h32.json`
- Modal:
  - `modal-dilated-h64-long1000` is running as app `ap-oe1x7fZOSx1lQ2R4WTt51O`
  - status: `.omx/status/modal-dilated-h64-long1000.json`

## Blocked queue
- Kaggle `dilated-h64-long1000`
  - `CANCEL_ACKNOWLEDGED`
  - queued for repush when Kaggle frees a slot
- Kaggle `pairaware_smoke`
  - `NOT_PUSHED`
  - still blocked by the 2-session GPU quota

## Infra reality
- the old helper-file Kaggle bundle path is dead
- direct code-file kernels are the right Kaggle execution model
- the baseline archive is staged through private dataset `adpena/comma-lab-private-assets`
- Modal is now a real fallback path
- one-shot Kaggle watchdog cycles are working and writing durable state; the long-lived daemon process remains unreliable

## Next real moves
1. Monitor the live Kaggle SegNet kernel for first checkpoint/artifact outputs.
2. Monitor the live Modal dilated fallback for first checkpoint/artifact outputs.
3. Repush the dataset-backed Kaggle dilated kernel when Kaggle actually frees the slot.
4. Push `pairaware_smoke` only after the slot/quota situation changes or move it to Modal.
