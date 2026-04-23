# Hard-Pair Analysis: Mathematical Framework for Pair Difficulty Prediction

## 1. Definitions and Notation

The challenge video contains 1200 frames at 20 fps. PoseNet evaluates 600 non-overlapping consecutive pairs $(f_{2k}, f_{2k+1})$ for $k \in \{0, 1, \ldots, 599\}$. SegNet evaluates only the even-indexed frame $f_{2k+1}$ (the second frame in each pair). The scoring formula is:

$$S = 100\bar{d}_\text{seg} + \sqrt{10\bar{d}_\text{pose}} + 25r$$

where $\bar{d}_\text{seg}$ and $\bar{d}_\text{pose}$ are averages over all 600 pairs. A single hard pair contributes $1/600$ to each average. The pair-level decomposition is:

$$\bar{d}_\text{pose} = \frac{1}{600}\sum_{k=0}^{599} d_\text{pose}^{(k)}, \qquad \bar{d}_\text{seg} = \frac{1}{600}\sum_{k=0}^{599} d_\text{seg}^{(k)}$$

**Definition 1 (Pair difficulty).** The *difficulty* of pair $k$ under a renderer $G_\theta$ is:

$$\delta^{(k)}(\theta) = 100 \cdot d_\text{seg}^{(k)}(\theta) + \sqrt{10 \cdot d_\text{pose}^{(k)}(\theta)}$$

This is the pair's marginal contribution to the score (excluding rate, which is global). The total distortion score is $\frac{1}{600}\sum_k \delta^{(k)}$.

**Definition 2 (Pair hardness prior).** The *hardness prior* $h^{(k)}$ is a scalar predictor of pair difficulty computable from mask topology, camera geometry, and GT frame statistics --- without running the scorers at inflate time. We seek $h^{(k)}$ such that $\text{rank}(h^{(k)}) \approx \text{rank}(\delta^{(k)})$.

## 2. What Makes a Pair Hard for PoseNet

PoseNet is a convolutional 6-DOF ego-motion estimator. It consumes YUV 4:2:0 pairs at 192x256 (6 channels per frame, 12 total) and outputs $[\hat{t}_x, \hat{t}_y, \hat{t}_z, \hat{r}_x, \hat{r}_y, \hat{r}_z]$. Distortion is:

$$d_\text{pose}^{(k)} = \frac{1}{6}\sum_{j=1}^{6}\left(\hat{p}_j^{(k)}(\text{compressed}) - \hat{p}_j^{(k)}(\text{GT})\right)^2$$

### 2.1 Ego-motion magnitude and PoseNet sensitivity

PoseNet is most sensitive to errors when the ego-motion is large, because large motions produce large optical flow, and small rendering errors get amplified through the flow field.

Given the camera intrinsics $K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$ with $f_x = 400.3$, $f_y = 399.5$, $c_x = 256$, $c_y = 192$ (scorer resolution), a point at depth $Z$ with image coordinates $(u, v)$ has ego-motion-induced optical flow:

$$\begin{pmatrix} \dot{u} \\ \dot{v} \end{pmatrix} = \frac{1}{Z}\begin{pmatrix} -f_x & 0 & u - c_x \\ 0 & -f_y & v - c_y \end{pmatrix}\begin{pmatrix} t_x \\ t_y \\ t_z \end{pmatrix} + \begin{pmatrix} \frac{(u-c_x)(v-c_y)}{f_y} & -f_x - \frac{(u-c_x)^2}{f_x} & (v-c_y) \\ f_y + \frac{(v-c_y)^2}{f_y} & -\frac{(u-c_x)(v-c_y)}{f_x} & -(u-c_x) \end{pmatrix}\begin{pmatrix} r_x \\ r_y \\ r_z \end{pmatrix}$$

**Key observation:** The flow magnitude at pixel $(u, v)$ is inversely proportional to depth $Z$. Near objects (vehicles at $Z \approx 15\text{m}$) produce $2\times$ more flow than far road surface ($Z \approx 30\text{m}$) and $67\times$ more flow than sky ($Z \approx 1000\text{m}$). This means pairs with large near-field objects are harder for PoseNet because any rendering error gets amplified by the $1/Z$ factor.

### 2.2 Computable PoseNet hardness features

For pair $k$ with masks $(m_{2k}, m_{2k+1})$, define:

**Feature P1: Inter-mask change area.**
$$A_\Delta^{(k)} = \frac{1}{HW}\sum_{i,j}\mathbb{1}[m_{2k}(i,j) \neq m_{2k+1}(i,j)]$$

Large $A_\Delta$ means significant scene change between frames, requiring the motion predictor to model complex flow. Empirically, highway driving at 20 fps has $A_\Delta \in [0.001, 0.15]$.

**Feature P2: Near-field object fraction (depth-weighted).**
$$F_\text{near}^{(k)} = \frac{1}{HW}\sum_{i,j}\frac{1}{Z_\text{class}(m_{2k}(i,j))}$$

where $Z_\text{class}$ is the depth prior per class: road=30m, lane=30m, vehicle=15m, sky=1000m, background=20m. Higher $F_\text{near}$ means more near-field content and larger expected flow magnitudes.

**Feature P3: Vanishing point occupancy mismatch.**
The vanishing point at $(256, 174)$ in scorer coordinates is where PoseNet is most sensitive to $t_z$ (forward translation). Define the VP region as a Gaussian-weighted disk:

$$V^{(k)} = \sum_{i,j} w_\text{VP}(i,j) \cdot \mathbb{1}[m_{2k}(i,j) \neq m_{2k+1}(i,j)]$$

where $w_\text{VP}(i,j) = \exp\left(-\frac{(j - 256)^2 + (i - 174)^2}{2 \cdot 40^2}\right)$. A mask change near the VP directly corrupts PoseNet's $t_z$ estimate.

**Feature P4: Flow field complexity (analytical).**
Given the per-pair affine parameters $\theta^{(k)} = [a_{11}, a_{12}, t_x, a_{21}, a_{22}, t_y]$ (stored in the LearnableAffineFlow at 3.6KB for all 600 pairs), the expected flow magnitude is:

$$\|\mathbf{f}\|^{(k)} = \frac{1}{HW}\sum_{i,j}\sqrt{(a_{11}x_j + a_{12}y_i + t_x)^2 + (a_{21}x_j + a_{22}y_i + t_y)^2}$$

This is computable at compress time and directly measures how much inter-frame motion PoseNet must resolve. Larger flow = more sensitivity to rendering errors.

## 3. What Makes a Pair Hard for SegNet

SegNet is an EfficientNet-B4 U-Net that predicts 5-class segmentation at 384x512. Distortion is argmax disagreement:

$$d_\text{seg}^{(k)} = \frac{1}{HW}\sum_{i,j}\mathbb{1}\left[\arg\max_c s_c^{(k)}(i,j;\text{comp}) \neq \arg\max_c s_c^{(k)}(i,j;\text{GT})\right]$$

where the sum is over the even-indexed frame only ($f_{2k+1}$).

### 3.1 Computable SegNet hardness features

**Feature S1: Boundary density.**
$$B^{(k)} = \frac{1}{HW}\sum_{(i,j) \in \text{4-neighbors}} \mathbb{1}[m_{2k+1}(i,j) \neq m_{2k+1}(i',j')]$$

Class boundaries are where argmax disagreements concentrate. A frame with many fine boundaries (complex road geometry, multiple lane markings, scattered vehicles) has more opportunities for SegNet misclassification.

**Feature S2: Rare class prevalence.**
$$R^{(k)} = \sum_{c \in \{1, 2\}} \frac{|\{(i,j) : m_{2k+1}(i,j) = c\}|}{HW}$$

Lane markings (class 1) and vehicles/undrivable (class 2) are typically minority classes. SegNet has higher per-pixel error rates on rare classes because the class boundary is proportionally longer relative to class area.

**Feature S3: Horizon band boundary concentration.**
The horizon band (rows 155--195 in scorer coordinates) contains the sky/road boundary where SegNet gradients are strongest. Define:

$$H_B^{(k)} = \frac{\text{boundary pixels in rows } [155, 195]}{\text{total boundary pixels}}$$

High $H_B$ means boundaries are concentrated where SegNet is most sensitive.

**Feature S4: Softmax entropy of GT segmentation.**
At compress time, we can run SegNet on GT frames and compute the per-pixel softmax entropy:

$$E^{(k)} = \frac{1}{HW}\sum_{i,j} H\left(\text{softmax}(s^{(k)}_\cdot(i,j;\text{GT}))\right)$$

where $H(p) = -\sum_c p_c \log p_c$. High entropy means SegNet is uncertain even on the GT frame, so small perturbations from compression can flip the argmax. This is the single most predictive feature for SegNet difficulty.

## 4. Composite Difficulty Score

Combining PoseNet and SegNet hardness, the composite difficulty prior for pair $k$ is:

$$h^{(k)} = \alpha_1 A_\Delta^{(k)} + \alpha_2 F_\text{near}^{(k)} + \alpha_3 V^{(k)} + \alpha_4 \|\mathbf{f}\|^{(k)} + \beta_1 B^{(k)} + \beta_2 R^{(k)} + \beta_3 H_B^{(k)} + \beta_4 E^{(k)}$$

The coefficients $\alpha_i, \beta_j$ are fit at compress time via simple linear regression against the true per-pair difficulties $\delta^{(k)}$ (computed from scorer forward passes at compress time, which is allowed since compress has unlimited compute).

### 4.1 Expected distribution

From the scoring formula structure and typical driving video statistics:

- **Easy pairs** ($\sim$60%): Highway driving, minimal scene change, road dominates, sky/road boundary is smooth. Expected $\delta < 0.5$.
- **Medium pairs** ($\sim$30%): Lane changes, curve entry, vehicles entering/exiting frame. Expected $0.5 < \delta < 2.0$.
- **Hard pairs** ($\sim$10%): Sharp turns, close vehicle following, complex urban intersections, multiple class boundaries near VP. Expected $\delta > 2.0$.

The hard 10% (60 pairs) contribute disproportionately to the score. Because $\bar{d}$ is a mean, reducing the worst pair's distortion by 1.0 saves $1.0/600 \approx 0.0017$ on the average, which maps to $100 \times 0.0017 = 0.17$ score points from SegNet alone. Across 60 hard pairs, the total hard-pair contribution to SegNet distortion is potentially $0.17 \times 60 = 10$ score points.

### 4.2 Connection to the scoring formula

The sqrt on PoseNet creates a specific interaction with hardness. For PoseNet, $\partial S / \partial d_\text{pose}^{(k)} = \frac{1}{600} \cdot \frac{\sqrt{10}}{2\sqrt{\bar{d}_\text{pose}}}$. At $\bar{d}_\text{pose} = 0.003$ (current renderer level), this is $\frac{1}{600} \cdot \frac{3.16}{2 \times 0.055} = 0.048$. For SegNet, $\partial S / \partial d_\text{seg}^{(k)} = \frac{100}{600} = 0.167$. So at current operating point, SegNet improvement is $3.5\times$ more valuable per unit than PoseNet. This means the difficulty prior should weight SegNet features more heavily.

However, there is a subtlety: PoseNet distortion for hard pairs can be orders of magnitude larger than for easy pairs (e.g., $d_\text{pose}^{(\text{hard})} \sim 0.01$ vs $d_\text{pose}^{(\text{easy})} \sim 0.0001$). The sqrt nonlinearity means hard pairs dominate the PoseNet score contribution quadratically: the pair with $100\times$ worse distortion only contributes $10\times$ more to the sqrt term. This is an inherent compression of the hard tail by the scoring formula.

## 5. Practical Computation at Compress Time

All features are computable from:

1. **Masks** $(m_0, \ldots, m_{1199})$: already in the archive, 5-class integer tensors at 384x512.
2. **Camera intrinsics** $K$: fixed, known constants (400.3, 399.5, 256, 192).
3. **Depth priors**: fixed per class (road=30m, lane=30m, vehicle=15m, sky=1000m, bg=20m).
4. **GT scorer outputs**: computed once at compress time via frozen scorer forward passes.
5. **Learned affine flow params**: from LearnableAffineFlow, stored at 1.8KB in the archive.

Total compute: one forward pass per pair through PoseNet + SegNet on GT frames ($\sim$30 seconds on T4), plus simple mask statistics ($\sim$1 second). The difficulty scores $h^{(k)}$ are 600 floats = 2.4KB uncompressed, or $\sim$1.2KB at FP16.

## 6. Usage in the Renderer Pipeline

The difficulty prior enables three mechanisms:

### 6.1 Difficulty-weighted training loss

Instead of uniform averaging over pairs, weight the loss by difficulty:

$$\mathcal{L} = \sum_{k=0}^{599} w^{(k)} \cdot \delta^{(k)}(\theta), \qquad w^{(k)} = \frac{h^{(k)}}{\sum_j h^{(j)}}$$

This allocates more gradient signal to hard pairs, forcing the renderer to spend capacity where it matters most.

### 6.2 Hard-pair capacity allocation

The renderer has 287K parameters shared across all 600 pairs. If we can identify the 60 hardest pairs at compress time, we can:

- Store per-pair conditioning vectors in the archive (60 pairs x 16 floats = 3.8KB at FP16)
- The renderer receives an additional conditioning input for hard pairs, giving it more expressiveness where needed
- Easy pairs use the default conditioning (zero vector), so no archive cost

### 6.3 Selective TTO distillation

TTO frames are the gold standard but cost 500 gradient steps per pair at inflate time (forbidden by the "no scorer at inflate" rule). Distillation trains the renderer to match TTO outputs. The difficulty prior tells us where to focus distillation effort:

- **Easy pairs**: renderer already produces acceptable output, minimal distillation needed
- **Hard pairs**: renderer struggles, TTO provides the largest improvement, concentrate distillation epochs here
- **Curriculum**: train on easy pairs first (fast convergence), then progressively add hard pairs (hard-frame curriculum)

## 7. Empirical Difficulty Distribution (2026-04-23)

We computed the full per-pair PoseNet distortion map for our 103K-param renderer
with CRF50-matched poses on all 600 pairs:

| Statistic | Value |
|-----------|-------|
| Mean pose_d | 0.124 |
| Top 20% mean | 0.611 |
| Bottom 80% mean | 0.003 |
| Max (pair 73) | 11.56 |
| Top 20% / Bottom 80% ratio | **227x** |

The distribution is extremely heavy-tailed. The top 20% of pairs contribute
>98% of the PoseNet average. A single pair (pair 73, pose_d=11.56) contributes
more to the average than the bottom 400 pairs combined.

**The sqrt exploit**: fixing the top 20% from mean 0.611 to 0.001 reduces
the PoseNet contribution from sqrt(10*0.124)=1.115 to sqrt(10*0.002)=0.153 ---
a 0.962 point improvement. At a postfilter cost of 0.030 rate (46KB), this is
a net +0.932 points. This is the single highest-leverage optimization available.

The difficulty map is saved as `difficulty.pt` (2.4KB, 600 float values) and
can be bundled in the archive to enable per-pair adaptive processing at inflate time.

## 8. Fridrich Loss Ablation (2026-04-23)

Head-to-head experiment with identical training configurations (Float + EMA + CRF50 masks +
103K DSConv + FiLM + CLADE) on separate 4090 GPUs, differing only in Fridrich losses:

| Epoch | No-Fridrich best | Fridrich best | Fridrich advantage |
|-------|-----------------|---------------|-------------------|
| Phase 1 ep 875 | 0.540 | 0.513 | +5% |
| Phase 2 ep 1050 | 1.155 | 1.249 | -8% (no-Fridrich ahead) |
| Phase 2 ep 1100 | ~1.10 | 1.006 | +9% (Fridrich catches up) |

**Finding**: Fridrich losses (UNIWARD texture weighting + L-infinity penalty) improve
Phase 1 pixel regression by 5-7% consistently. In Phase 2 (scorer-guided), they initially
hurt (competing with scorer gradients for capacity) but catch up by epoch 1100.

**Recommendation**: Apply Fridrich losses in Phase 1 and Phase 3 only. Disable in Phase 2
where the scorer itself provides texture-aware gradient signal.

## 9. Validation Protocol

The difficulty prior is validated by computing Spearman rank correlation between $h^{(k)}$ and the true per-pair difficulty $\delta^{(k)}(\theta^*)$ on the best renderer checkpoint:

- $\rho_s > 0.7$: strong predictor, use directly for loss weighting
- $0.4 < \rho_s < 0.7$: moderate predictor, use for curriculum but not loss weighting
- $\rho_s < 0.4$: weak predictor, features need redesign

Expected: $\rho_s \approx 0.6$--$0.8$ based on the observation that mask boundary density and near-field fraction empirically correlate with scorer distortion in our training logs.
