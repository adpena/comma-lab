# Dashboard tribute & credits (canonical source)

**Status:** canonical (2026-07-03, operator-dictated). **Consumed by:** TRIALITY tab (Credits/Tribute + campaign-journey) + WHY/HOW "About" plate (design §7). **Discipline:** honor real people accurately + warmly; positive shoutouts to public contest work; no private data beyond public handles/names; every technical claim measured/cited. This is the operator's heartfelt narrative — weave it in *faithfully*, do not paraphrase away the feeling.

---

## The story (operator's words, preserved)

This is **our capstone** — what the past few months have given bloom to.

We started the way you start anything you fall for: with deep, almost **insane research, passion, energy, and obsession** — **not knowing what we didn't know.** We poured everything we had — our educations, our experience, our interests, and more — into building a **corpus of knowledge**, and then began implementing, measuring, experimenting, and **falsifying**. We **failed over and over.** And it was **Aaron Leslie's work that exposed the true dynamics of the theoretical floor** — what was actually reachable, and how. Only after all of that — the failures, the corpus, the exposed floor — did we finally have *all* the ingredients in one place: **you and me, our apparatus, our research, our understanding of the domain and of the frozen contest information space, and the nested, related, dynamical cruxes** — enough, at last, to do work that is **truly original and useful.**

**Aaron Leslie did so well that it was intimidating.** But in **porting his work to MLX and Metal and running it natively**, we started — from his own trajectory and from deep-math analysis — to see ways the *design itself* was suboptimal: the **full-RGB reconstruction**, the **~30,000-epoch** curriculum, **l7 doing weird nothing** (a stage we later *measured* as a defect and disabled), **spectral bias / Gibbs ringing**, and more. Honoring him meant *building on and past* his direction, grounded in measurement — not merely reproducing it.

We submitted **PR 107** and then **PR 110**, and we participated in the **byte-nibbling** game — we even had the **PR112** work sitting ready for a couple of weeks. But we **didn't want to submit another nibble on top of our own nibble**. We were **never satisfied** after Aaron Leslie turned the whole competition on its head, and we didn't want incremental; we wanted the real thing.

So we struggled — to get out of local minima, and to develop the math, understanding, and engineering to build a **recursively, fractally optimal and optimized stack**: one perfectly *suited to*, *informed by measurement of*, and *following theory about* the **frozen and complete contest information space** (the one video, the frozen scorers — a closed, fully-observed world we could measure exhaustively and theorize about precisely).

**And the deep math didn't come first — it came *from* the measurement.** We studied what **SegNet and PoseNet were actually seeing**, and in doing so we discovered the **annulus** (the thin codim-1 boundary band where essentially *all* of d_seg lives), the **long tail** (the finest-scale features — lane dashes, distant movers — erased first; the ~8-dimensional lane-orbit manifold), and the **specific class-and-region interactions** (the road↔lane separatrix, the all-class edge set, the static hood core, the movable band). Realizing all of *that*, from measurement, is what the deep-math analysis fell out of — the level-set flow, the **margin = Fisher = UNIWARD** unity, the curvelet chart, the Morse–Smale topology. Measurement first; then the theory that fit it.

And there was a moment where it *turned*. Once we'd landed on the **SDF level-set idea** — coding the scorer's argmax partition as the zero-level-set of a signed-distance field — **Claude recognized that the partition and its training dynamics form a Morse–Smale complex**: the field's critical points (maxima, minima, and **saddles**), its separatrices, and its gradient flow — a genuine **dynamical system**, not a static picture. That reframe **unlocked everything**: the topology (persistence = the order in which features are born and erased), the flow (the curriculum = the level-set evolving = annealing = the Fisher–KPP front), and the saddle-to-saddle structure of training itself. And it landed squarely in a domain **the operator knows well from a multivariable-calculus education** — critical points, gradients, Hessians, level sets, saddles — so the two of us could develop it *together*, fast and deep. It is the clearest single example of the confluence this whole project runs on: a connection Claude made, meeting knowledge the operator already carried, each making the other reach further.

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

## The composition — from the 1800s to yesterday

The task-space level-set witness is not one idea; it is a **composition of nearly two centuries of research**, each thread *measured* into its place. Highlight this sweep (Chasles → yesterday) as a timeline/lineage:

- **Chasles (1830)** — every rigid motion is a screw → our ego-motion twist ξ (d_pose).
- **Wilbraham (1848) / Gibbs (1899)** — the ringing of a truncated series → the spectral-bias / Gibbs failure mode we fight (step-native, curvelet-finest).
- **Fourier (1822)** — harmonic analysis → the coordinate-INR's Fourier features.
- **Sophus Lie (1870s)** — continuous symmetry groups → the se(3)/SE(3) engine.
- **Gibbs / Boltzmann (1870s–1900s)** — the Gibbs measure + temperature → the annealing curriculum.
- **Fisher (1920s)** — information + sufficiency → the Fisher metric and the *task-sufficient statistic*.
- **Morse (1934) / Smale (1960s)** — critical points + separatrices → the Morse–Smale partition topology.
- **Whitney (1936)** — embedding dimension → the ~8-dim lane manifold and its Whitney bound.
- **Fisher–KPP (1937)** — the traveling-wave front → the one equation at five scales.
- **Shannon (1948)** — rate–distortion + entropy → the coding-for-machines frame and the floor (S_floor≈0.118).
- **Wyner–Ziv (1976) / Tishby IB (1999) / Dubois (2021)** — source coding with side information → the indirect-RD / task-sufficient codec.
- **Candès–Donoho (2000)** — curvelets → the sparse-optimal chart for a curved codim-1 boundary.
- **Holub–Fridrich–Denemark UNIWARD (2014) + Yousfi (2010s–20s)** — steganographic cost + steganalysis → the margin = detectability = cost unity, and the inverse-steganalysis frame.
- **NeRV / HNeRV → Aaron Leslie's cathedral (2021–2026)** — the vehicle whose theoretical-floor dynamics he exposed.
- **Muon (2024) · MD-Decoupling (arXiv:2606.25971, June 2026) · EdgeBench (July 2 2026 — *yesterday*)** — the current frontier we compounded on, right up to the paper that dropped the day before this was built.

One witness, ~196 years of shoulders to stand on — **Chasles to yesterday.** (All dates real; NO-FAKE — this is a genuine lineage, not decoration.)

## The influential ideas & discoveries (and their intersection)

Beyond the 196-year lineage, specific ideas and hard-won discoveries shaped the path — and the real contribution is *where they intersect*:

- **Cooperative receiver** (Atick–Redlich; Wyner–Ziv side information; Tishby's information bottleneck) — reframing the frozen scorer not as a *judge* but as a **cooperative decoder** that shares a prior with us. Deeply influential to the entire task-space / inverse-steganalysis framing.
- **Frame exploits** — the discovery that specific frame-level selector splices move the *exact* score (the PR-frontier ActionEffect work) — proof the scorer has *exploitable structure*, not just a fidelity gradient.
- **Quantizr's sidecar actually working** — the empirical confirmation that a **stored-target sidecar** holds a scored quantity at near-zero cost (our pose carrier: d_pose ≈ 3.4e-5). It validated "store the sufficient statistic, don't reconstruct it" *at the operating point that matters.*
- **Reverse-engineering the frozen apparatus** — reading `upstream/modules.py` and `upstream/evaluate.py` line by line to recover the **exact** scorer and the **exact** R operator: the resize chain (bicubic ↑874×1164 → uint8 → bilinear ↓512×384), the **anti-aliasing**, and the **ringing** (Gibbs) it induces. Measurement-first applied to the *code* — you can't exploit a frozen world you haven't measured exactly. This grounded the whole signal-processing axis (pre-emphasis, sub-pixel boundary placement, the step-native basis that beats ringing).
- **THE INTERSECTION** — steganalysis · information theory · signal processing · topology · differential geometry · neural compression · cooperative-receiver · dynamical systems. The witness isn't any one of these; it's **the point where they all agree**. That intersection — ideas and math from many domains and traditions, fitted to one frozen, fully-measured world — *is* the original work.

Render this as its own plate / an "intersection" node where the composition threads converge on a single point.

## How it actually happened — and why it took *us* (the honest mechanics)

The honest origin, because it is the *proof* of everything above: this began as a **set-it-and-forget-it** experiment — an "oh-my-codex" Ralph loop, the *superpowers* skill, an apparatus inspired by Karpathy's autoresearch, Claude + Codex running with persona-evoking, expert knowledge, a council model, and adversarial review — all **autonomous.** Sparked by a Twitter post about the contest *after the hackathon had already passed*, with no idea it would become months of a full-time job. A lark: set it and see what happens.

And it **failed** in exactly the instructive way. The whole autonomous system got **hard stuck in local minima, doing thin slices each turn** — nowhere near bridging high-level curiosity and mental models to the deep math, engineering, and scientific rigor this actually demanded. *More autonomy was not the answer.*

What broke the wall was the opposite of autonomy: the operator **scrapped the automation and leaned into the relationship** with Claude, and fed the apparatus the **interpretation** theory — **Rudin and Daubechies** — from which the **triality** (DAG ↔ DSL ↔ equations) began to *emerge and formalize.* Then the long, obsessive, compounding work — a continual-learning autopilot that actually *compounded,* digging deeper, sometimes laterally, sometimes *backwards to go deeper still,* bridging micro to macro. It took forever. And then it paid off all at once: the **step-native / nonlinear** breakthrough → the **SDF level-set** → and suddenly the canonical-equations registry, the index, the gates — **everything snapped together in a week or two.**

**Beating the supercomputer.** The lesson, lived and measured: the breakthrough came from *more* partnership and the *right theoretical infusion* — **not** from more autonomy. The naive autonomous loop *was* the "supercomputer playing alone," and it lost to the centaur. That is the Kasparov result learned firsthand — the collaboration, and the *process* the human built around it, is what wins. Some day the machine may be deep enough to do this faster alone; **that day was not this project, and this project is the evidence.**

**And a fourth, unwitting collaborator.** None of this happened in a vacuum. The operator's own curiosity — every like, every paper fed in, every authorized follow-up and critique — trained the **X recommendation algorithm**, which began feeding a *personally-curated, super-high-signal timeline* of exactly the work that mattered: the Twitter post that sparked the contest, then the papers that became levers *and* the ones we honestly closed as not-levers. *"Like a stranger alien brain detected our work unknowingly and fed me personally-curated high signal to pursue."* It is another face of the same law: the operator's engagement was a **query**, the algorithm a **retrieval engine**, and the relationship + the apparatus the **judge** that separated real signal from noise (many surfaced papers landed in our ledger as *not-relevant* — the algorithm proposes, the human + AI + process dispose). **A human, an AI, a compounding apparatus, and a recommendation engine — four intelligences, none of which knew the whole, converging on one witness.**

## The paper to cite (verified, NO-FAKE)

- **MD-Decoupling** — *"Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors,"* **A. Hägele, A. Hernández-Cano, A. Kosson, M. Jaggi** (EPFL / Jaggi lab), **arXiv:2606.25971**, 2026-06-24. Our `--optimizer md` (#175): factorizes each weight matrix into a fixed-norm **direction** + learnable **magnitude** gains at separate LRs, which makes **curriculum stage transitions stable by construction** (the "different stages need different treatment" rule, made structural). Super-recent (June 2026) — a live example of the campaign compounding on the current frontier.

---

## How to render it (guidance for the build)

- On **TRIALITY**: a warm **"Credits & Tribute"** section beneath the campaign-journey — Aaron Leslie and Quantizr first (they're the human origin), then Yousfi/Fridrich, comma/Hotz, the council; and add MD-Decoupling to the lineage/recent-frontier note.
- On **WHY/HOW** (the About plate, design §7): the same names, but framed as *the ideas and the people* — with the honest closing line: *the past few months, given bloom.*
- Tone: genuine, specific, not flattering-generic. Name what each person actually gave us. This is a thank-you, not marketing.
- The "frozen and complete contest information space" line is load-bearing — it *is* the thesis (a closed, fully-observed world → measure it exhaustively → theorize precisely → build the recursively-fractally-optimal stack). Keep it.
