---
name: Fridrich inverse steganalysis — how to fool the scorer (4 actionable principles)
description: Fridrich/Yousfi published work on detector-informed embedding, UNIWARD distortion hiding, square root law, and CNN blind spots. All directly applicable to our renderer.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Fridrich's Own Work Tells Us How to Beat the Scorer

### 1. UNIWARD: Hide errors in texture, never in smooth regions
Holub & Fridrich 2014. Changes in textured/noisy regions are undetectable.
Changes in smooth sky/road areas are instantly caught.
**→ Our renderer loss should weight errors by local texture complexity.**
**→ The hinge loss already focuses on class boundaries, but we should ALSO**
**   weight by a texture map (high-frequency energy).**

### 2. Detector-Informed Embedding = TTO with scorer gradients
Yousfi, Dworetzky & Fridrich, IH&MMSec 2022. Using detector feedback to
guide changes produces "substantial security gains."
**→ Our TTO (test-time optimization through frozen scorers) is EXACTLY this.**
**→ This is the Fridrich-approved approach. We're doing it right.**

### 3. Square Root Law: Spread small errors, don't concentrate large ones
Ker, Filler & Fridrich 2008-2009. Detectability scales with max error, not mean.
Spreading tiny errors across many pixels is fundamentally harder to detect.
**→ Our renderer should minimize peak pixel error (L∞), not just mean (L1/L2).**
**→ Consider adding a max-error penalty term to training loss.**

### 4. CNN Blind Spots: EfficientNet has systematic weaknesses
Yousfi & Fridrich 2020. CNNs fail on simple statistics that hand-crafted
features capture. Conversely, CNNs have blind spots.
**→ The SegNet scorer (EfficientNet-B4) has blind spots in:**
**   - Texture-rich regions (high local variance)**
**   - Regions where directional filter response is already high**
**   - Changes that preserve local Markov chain statistics**

### How to apply
1. Add texture-aware loss weighting (UNIWARD principle) to training
2. Continue using TTO with scorer gradients (already doing this)
3. Add L∞ regularization to encourage spread errors over concentrated ones
4. Focus optimization budget on smooth regions (where detector is sensitive)
