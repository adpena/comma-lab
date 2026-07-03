# Dashboard tribute & credits (canonical source)

**Status:** canonical (2026-07-03, operator-dictated). **Consumed by:** TRIALITY tab (Credits/Tribute + campaign-journey) + WHY/HOW "About" plate (design §7). **Discipline:** honor real people accurately + warmly; positive shoutouts to public contest work; no private data beyond public handles/names; every technical claim measured/cited. This is the operator's heartfelt narrative — weave it in *faithfully*, do not paraphrase away the feeling.

---

## The story (operator's words, preserved)

This is **our capstone** — what the past few months have given bloom to.

**Aaron Leslie did so well that it was intimidating.** But in **porting his work to MLX and Metal and running it natively**, we started — from his own trajectory and from deep-math analysis — to see ways the *design itself* was suboptimal: the **full-RGB reconstruction**, the **~30,000-epoch** curriculum, **l7 doing weird nothing** (a stage we later *measured* as a defect and disabled), **spectral bias / Gibbs ringing**, and more. Honoring him meant *building on and past* his direction, grounded in measurement — not merely reproducing it.

We submitted **PR 107** and then **PR 110**, and we participated in the **byte-nibbling** game — we even had the **PR112** work sitting ready for a couple of weeks. But we **didn't want to submit another nibble on top of our own nibble**. We were **never satisfied** after Aaron Leslie turned the whole competition on its head, and we didn't want incremental; we wanted the real thing.

So we struggled — to get out of local minima, and to develop the math, understanding, and engineering to build a **recursively, fractally optimal and optimized stack**: one perfectly *suited to*, *informed by measurement of*, and *following theory about* the **frozen and complete contest information space** (the one video, the frozen scorers — a closed, fully-observed world we could measure exhaustively and theorize about precisely).

**And the deep math didn't come first — it came *from* the measurement.** We studied what **SegNet and PoseNet were actually seeing**, and in doing so we discovered the **annulus** (the thin codim-1 boundary band where essentially *all* of d_seg lives), the **long tail** (the finest-scale features — lane dashes, distant movers — erased first; the ~8-dimensional lane-orbit manifold), and the **specific class-and-region interactions** (the road↔lane separatrix, the all-class edge set, the static hood core, the movable band). Realizing all of *that*, from measurement, is what the deep-math analysis fell out of — the level-set flow, the **margin = Fisher = UNIWARD** unity, the curvelet chart, the Morse–Smale topology. Measurement first; then the theory that fit it.

Thankfully, our work and belief and passion and untiring curiosity have yielded **something we are proud of** — and that we hope is **interesting and useful to the very people who introduced us to such an interesting and fulfilling problem.**

---

## Credits & shoutouts (weave into TRIALITY + the WHY/HOW About plate)

### Aaron Leslie — PR95, the HNeRV cathedral
The author of **PR95** and its HNeRV "cathedral." He **turned the whole competition on its head** and showed what was possible — his result was **so good it was intimidating**. He **taught us about schedule and curriculum** and, more than any single technique, **drove us to obsessively dig into the math and geometry of the video**, starting from what we'd gleaned from openpilot. The honest twist — and the highest form of respect we could pay him: **porting his work to MLX + Metal and analyzing his own trajectory + the deep math let us *see past* the design** (full-RGB, ~30k epochs, the inert l7 stage, spectral bias), which is exactly what pointed us to the task-space level-set direction. We stand on the cathedral he built. *(Our inner-council "PR95Author" seat is him.)*

### Quantizr (Jimmy) — the earlier mind-opening
He **opened our minds even earlier** to what was possible. Beyond his 0.33 HNeRV result, it was his **spirit** we carry: **experimentation, openness, curiosity, playfulness, competitiveness, confidence.** He **affirmed some of our earliest intuitions** — the ones that made us feel we were **biting off more than we could chew**, except we **loved the flavor and became obsessed.** *(Our inner-council "Quantizr" seat is him.)*

### Yassine Yousfi & Jessica Fridrich — the detection game
Yousfi built the SegNet/PoseNet scorer (from comma10k) and framed the whole thing as **inverse steganalysis**; Fridrich's **UNIWARD** (DDE Lab) is the cost that turns out to *equal* the scorer's own sensitivity metric (measured Pearson 0.978). The entire dashboard is arranged as **Yousfi's detection game** in his honor.

### comma.ai & George Hotz — the free physical prior
openpilot is the unified free physical prior for both scored axes (lane geometry → d_seg, ego-motion screw → d_pose); comma10k is the palette and the scene.

### The council — the lenses we think through
**Shannon** (LEAD; R(D)/entropy/sufficiency — the floor S_floor≈0.118 is his bound), **Dykstra / Rudin / Daubechies** (co-leads; feasibility / interpretability / wavelets), **Ballé** (the neural codec of task-aware compression), and **Schmidhuber** (compression-as-intelligence / POWERPLAY — the campaign-scale front), among the full roster.

---

## The paper to cite (verified, NO-FAKE)

- **MD-Decoupling** — *"Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors,"* **A. Hägele, A. Hernández-Cano, A. Kosson, M. Jaggi** (EPFL / Jaggi lab), **arXiv:2606.25971**, 2026-06-24. Our `--optimizer md` (#175): factorizes each weight matrix into a fixed-norm **direction** + learnable **magnitude** gains at separate LRs, which makes **curriculum stage transitions stable by construction** (the "different stages need different treatment" rule, made structural). Super-recent (June 2026) — a live example of the campaign compounding on the current frontier.

---

## How to render it (guidance for the build)

- On **TRIALITY**: a warm **"Credits & Tribute"** section beneath the campaign-journey — Aaron Leslie and Quantizr first (they're the human origin), then Yousfi/Fridrich, comma/Hotz, the council; and add MD-Decoupling to the lineage/recent-frontier note.
- On **WHY/HOW** (the About plate, design §7): the same names, but framed as *the ideas and the people* — with the honest closing line: *the past few months, given bloom.*
- Tone: genuine, specific, not flattering-generic. Name what each person actually gave us. This is a thank-you, not marketing.
- The "frozen and complete contest information space" line is load-bearing — it *is* the thesis (a closed, fully-observed world → measure it exhaustively → theorize precisely → build the recursively-fractally-optimal stack). Keep it.
