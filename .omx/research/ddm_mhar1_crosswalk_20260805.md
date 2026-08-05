# ddm_mhar1 — arXiv 2607.27230 "Multi-Head Attention Residuals" crosswalk vs the live TR1 vehicle

**Operator drop 2026-08-05:** `https://arxiv.org/abs/2607.27230` (the established drop pattern:
identify + deep-read + crosswalk vs live surfaces, honest transfer rows, no fake adoption).
Executed by MAIN directly (codex fleet DOWN until 2026-08-10). Axis of every number below:
paper-reported unless tagged otherwise; nothing here is a score claim.

## 1. Paper identification (WebFetch, v2 2026-07-31)

**"Multi-Head Attention Residuals"** — Cheng Luo, Zefan Cai, Junjie Hu (cs.AI; v1 2026-07-22,
v2 2026-07-31). Mechanism: the transformer residual stream's read from DEPTH HISTORY is split
into H per-subspace ROUTING HEADS (block-diagonal over the model dim) — **zero added
parameters**; each subspace routes its own weighted mixture over prior-layer outputs instead of
one shared mixture. Findings the crosswalk consumes:

1. **U-shaped in H** with a FLAT optimum at H = 4–8; H = 16 over-splits and gives the gains back.
2. **Probe-the-driver methodology:** the claimed mechanism (per-subspace disagreement in what
   depth-mixture each subspace wants) is MEASURED on TRAINED queries before being credited —
   subspace-wise gradient/attention disagreement is the driver, verified, not asserted.
3. **Identity-preserving conversion:** delta-form attention residuals initialize the new
   routing to reproduce the OLD architecture exactly, enabling MID-TRAINING retrofit at 8B
   scale (+3.2 GSM8K / +3.1 GPQA reported) — the architecture lever enters without a loss jump.

## 2. Live-vehicle scoping (why direct adoption is INAPPLICABLE)

The shipped renderer `render_frame1_float` (`src/tac/optimization/ddm_tr1_runtime.py:1300-1319`)
is a plain sequential conv2d→gelu_exact stack — **no residual stream exists**: ~3–4 layers,
width 24, renderer ≈3.3 KB counted (LOTTO). The trainer's `build_module`
(`experiments/train_tr1_partition_renderer_mlx.py:534`) matches. MHAR's object of study (the
depth-history residual read) is structurally absent from this vehicle. Grafting attention
residuals onto a 3-layer width-24 conv stack is a FAKE adoption (mechanism-name transfer
without the mechanism's substrate) — refused per NO-FAKE #7.

## 3. Honest transfer rows

| # | Row | Grade | Consumer + fire-order |
|---|---|---|---|
| 1 | **Identity-preserving conversion protocol** — enter ANY TR1 architecture lever at a window boundary initialized to reproduce the incumbent exactly (delta/zero-init on the new path), so the lever A/B starts from the incumbent's loss, not a re-birth. This is the same physics as our warm-start law ([[m72]]) lifted to ARCHITECTURE changes; en1's margin-flag restoration + w4m staging already follow the spirit (identical resume, single variable). | **ADOPT_CLASS** (protocol, $0) | Every future TR1 architecture lever (depth-mixing, per-class heads, width changes) enters via zero-init-delta at a window boundary. Standing entry rule; recorded here + hot-state. |
| 2 | **Probe-the-driver: per-class gradient-subspace disagreement probe** — before building v8-lite per-class carriers/heads, MEASURE whether the classes actually fight over the shared trunk: per-class masked-CE gradients (classes 0–4, ~8 pairs) through the real R+SegNet path via the trainer's own `pair_loss` closure (`:3883`), pairwise cosine per parameter tensor. High disagreement on trunk tensors = the per-class-split premise has measured support; low = shared trunk suffices and the split is dominated. | **ADOPT_METHOD — QUEUED-WITH-FIRE-ORDER** | **Owner MAIN. Fires at the w4/w4m ep1363 boundary adjudication turn, against the WINNING endpoint checkpoint** (trained queries per the paper's own methodology; same turn feeds jd1 #366 regeneration). Method: probe mode reusing main()'s setup (the closure resists standalone extraction — ~700 lines of setup; building it into the sealed live trainer mid-burn was refused). |
| 3 | **U-shape racing method → pose-basis rank sweep** — the paper's "race the split count, expect a flat optimum then over-splitting" is the right shape for the DERIVED PoseNet-Jacobian basis race (naive-audit cure, consumer jd1): sweep basis rank the way MHAR sweeps H, expect flat-optimum-then-collapse, and stop at the knee via `adjudicate_tail_slope` rather than a fixed rank. | **ADOPT_METHOD** (folds into the queued basis race) | The eg1-generic vs derived-Jacobian terminal race (naive-audit queue) inherits the rank-sweep protocol. |
| 4 | **Learned depth-mixing over conv blocks** — the nearest structural analogue: let later conv blocks read a learned mixture of earlier block outputs (a DenseNet-lite routing, per-class-subspace optional). | **RACED-CANDIDATE, deferred** | Deferred with reason: width-24 × 3-layer means depth history is 2–3 tensors — the driver MHAR exploits (many-layer history) is minimal here; renderer bytes are LOTTO-priced and any routing params are counted. Fire only if row-2's probe measures HIGH cross-class trunk disagreement (then per-class routing and depth-mixing race together via row-1 entry). |
| 5 | **NOT TRANSFERRED:** attention residuals themselves; H-head routing over a residual stream; any LLM-benchmark-derived constant. No residual stream exists on the vehicle (§2); constants-are-poison ([[m21]]) bars borrowing H=4–8 as anything but the *shape* of the race in row 3. | — | — |

## 4. Disposition

The drop is CONSUMED: identified, deep-read, scoped against the shipped renderer at source,
five rows with named consumers/fire-orders, zero fake adoptions. The single deferred build
(row 2 probe) exits QUEUED-WITH-FIRE-ORDER per the follow-on law
([[follow_on_work_fires_immediately_or_it_is_orphan_poison_20260803]]) — fire event is the
ep1363 boundary (~4 h out, done-receipts armed), owner MAIN, and the measurement is *better*
there (winner's trained trunk).

Pointer: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer
borrowed/unmoved by this memo (crosswalk = means, not progress).
