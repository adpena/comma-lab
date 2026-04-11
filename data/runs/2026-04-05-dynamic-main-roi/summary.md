# dynamic main ROI summary

## measured result

- Status: rejected
- Packaging view: `current_workflow`
- Device: `cpu`
- Final score: **`4.47`**
- Archive bytes: `2660388`
- Rule-faithful estimate: `4.488615891076854` at `2694567` bytes

## candidate config

- base floor: `432x324 / medium / crf23 / keyint48 / bframes4 / ref4 / lanczos+lanczos`
- dynamic main ROI enabled
- metadata windows: `200` frames
- sample step: `10` frames
- tile grid: `12x9`
- base CRF delta: `+1`
- main ROI CRF delta: `-1`
- auxiliary ROI: enabled in config, but not triggered by metadata on this run

## interpretation

The dynamic main-ROI experiment preserved the central corridor more intelligently than the earlier fixed rectangle, but it still lost badly to the uniform `3.33` floor. The archive grew to about `2.66 MB`, and both SegNet and PoseNet distortion remained worse than the promoted floor.

## conclusion

- Main ROI protection by itself is not enough.
- Segment/window overhead is expensive.
- The segmentation lane should not be promoted again unless the side information gets much sparser or the protected stream becomes much cheaper.
