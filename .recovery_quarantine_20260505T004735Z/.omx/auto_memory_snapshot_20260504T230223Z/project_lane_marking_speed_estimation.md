---
name: Lane marking displacement → vehicle speed → zoom scalars (zero archive cost)
description: Lane markings (SegNet class 1, MUTCD standard 3m×15cm) are in every frame. Their inter-frame displacement encodes vehicle speed via camera intrinsics. Zoom scalars COMPUTABLE at inflate time from masks alone — zero extra archive bytes. Paper-worthy innovation.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The Core Insight

Lane markings are standardized physical objects:
- US highway: ~3m long, ~15cm wide, ~9m spacing (MUTCD standard)
- Camera: fx=910px, 20fps, principal point (582,437) native / (256,192) scorer
- SegNet class 1 = lane markings (0.58% of pixels, but geometrically precise)

Inter-frame lane displacement directly encodes vehicle speed:
  speed = (displacement_pixels / fx) * distance_to_lane * fps
  zoom_scalar = displacement_radial / distance_to_FoE

## The Power: Zero Archive Cost

The zoom scalars could be COMPUTED at inflate time from masks, not stored:
1. Extract lane marking centroid from mask_t1 (class 1 pixels)
2. Compute radial distance from FoE (256, 174)
3. Compare across pairs → displacement → zoom scalar
4. Apply zoom warp via grid_sample

Archive cost: ZERO. The masks already encode lane positions implicitly.
This eliminates the 1.2KB of stored zoom scalars entirely.

## Measured Data (from actual CRF50 masks, verified)

Lane marking coverage: mean=0.58%, present in ALL 1200/1200 frames.

| Pair | Centroid t→t1 | Radial Δ | Lane pixels |
|------|---------------|----------|-------------|
| 0 | (234,216)→(225,224) | +0.245 | 1081/1359 |
| 100 | (272,227)→(251,219) | -0.192 | 1220/1079 |
| 200 | (276,211)→(275,217) | +0.104 | 1109/1373 |
| 300 | (262,213)→(251,222) | +0.215 | 881/1087 |
| 400 | (237,236)→(253,233) | -0.085 | 1536/1314 |
| 500 | (276,221)→(312,208) | +0.297 | 1318/1046 |

Sign flips = lateral movement (lane changes, curves).
Magnitude = forward speed. Range 0.08-0.30.
Centroid-based is crude; endpoint tracking would be more precise.
Noisy: the centroid shifts laterally too, not just radially.

## Three Application Options

1. **Compute at inflate time**: zero archive bytes. Derive zoom from masks.
   Risk: noisy centroid method, needs robust lane tracking in inflate.py.
   Reward: eliminates 1.2KB zoom scalars from archive entirely.

2. **Warm-start for gradient optimization** (RECOMMENDED FIRST):
   Use mask-derived estimate as initialization for gradient descent through PoseNet.
   50-100 gradient steps to polish (vs 500 from zero init).
   Faster convergence, more robust, still needs PoseNet at compress time.

3. **Validation tool**: compare mask-derived speed against learned zoom scalars
   to verify physical consistency. If they disagree, something is wrong.

## Paper Significance

"Mask-derived ego-motion estimation for neural video compression" — the semantic
segmentation masks, already required for SegNet fidelity, implicitly encode the
camera's ego-motion through lane marking displacement. This dual-use of the mask
channel (SegNet appearance + PoseNet motion) is a novel contribution.

## Connection to openpilot

The competition video is from comma EON running openpilot. openpilot's lane
detection and path planning models understand this exact road geometry.
At compress time (unlimited compute), openpilot models could provide:
- Sub-pixel lane boundary detection (better than argmax centroids)
- Lane polynomial fitting (parametric, smooth)
- Vehicle trajectory estimation (speed, heading, curvature)
All contest-compliant since compress time has no restrictions.

**How to apply:**
1. Start with option 2 (warm-start + polish) — validates the concept
2. If mask-derived zoom matches gradient-optimized zoom, prove option 1 works
3. For the paper: demonstrate dual-use of mask channel (appearance + motion)
4. For production: mask-derived zoom enables ZERO-cost motion at inflate time
