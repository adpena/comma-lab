# EdgeBench deep-dive — the log-sigmoid law of environment learning, read through the Schmidhuber (POWERPLAY) and Yousfi (detector-feedback) razors

**Paper:** "EdgeBench: Unveiling Scaling Laws of Learning from Real-World Environments" — ByteDance Seed (corresp. Shu Zhong), dated **2026-07-02** (yesterday). Project page https://edge-bench.org · code https://github.com/ByteDance-Seed/EdgeBench · data https://huggingface.co/datasets/ByteDance-Seed/EdgeBench (51 of 134 tasks public). PDF read in full (20pp main body incl. references; §3.3 theory pp.9–10, §5 dynamics pp.12–15).

**Author of memo:** research subagent, session 89ff112f. Pointer **0.19110 UNMOVED** — this is a MEANS memo (framing / validation / campaign-strategy). It touches **no** d_seg / d_pose / byte lever. Honest EV verdict is up front and again at the end.

---

## 0. TL;DR + honest EV verdict (read this first)

EdgeBench measures ~38,000 h of frontier-agent interaction across 134 day-long real-world tasks and finds that **benchmark-averaged best-so-far performance during environment learning follows a log-sigmoid law**

$$S(t) = \frac{S_{\max}}{1 + (t_{\text{mid}}/t)^{\beta}}, \qquad \text{equivalently}\quad \frac{dx}{d(\ln t)} = \beta\,x(1-x),\ \ x = S/S_{\max},$$

with **mean R² = 0.998**, best among all common S-curves (Table 1). They derive it as **frontier expansion on a latent task graph** and show (i) learning speed **doubles ~every 3 months** (≈8× over 221 days, Sep 2025→Apr 2026), and (ii) **continuous experience beats independent restarts by +6.9** at 12h, longer context adds **+4.4–5.8**, and structured **diagnose→edit→evaluate** feedback turns failed probes into durable gains.

**Is it a pointer-mover? NO.** It is not a d_seg/rate/pose lever and never will be. **Verdict: FRAMING + VALIDATION + three concrete campaign-strategy changes.** The three changes (detailed in §5, §6):
1. **Learning-progress acquisition rule** for lever/experiment ordering (spend where the local log-sigmoid slope toward the floor per through-R compute-hour is steepest = POWERPLAY simplest-unsolved-frontier). We ~already info-gain-rank; this gives it the measured functional form.
2. **Early-stop-on-forecast-plateau** — fit the early (~first 50–60%) per-stage descent to the log-sigmoid, estimate the config's plateau, kill a vehicle that the forecast says won't reach the need-band. Feeds the #188 early-stop lever + the don't-loop-the-wrong-vehicle discipline. (Honest caveat in §5.3: single-run forecasts are wide.)
3. **Feedback-density maximization** — every through-R verdict should emit the full multi-level feedback (scalar d_seg/d_pose **+** per-pixel margin-saliency #141 **+** per-stage diff **+** per-class flip attribution), because feedback richness is the measured engine of long-horizon gain.

**The deepest worth is a validation, not a lever:** the operator's #1 obsession — *"the purpose of our whole project is deterministic reproducibility and no signal loss ever"* — is now **empirically grounded (R²=0.998 + the +6.9 experience gain that GROWS with horizon)** as the mechanism of long-horizon capability, not merely hygiene. §7.

---

## 1. The reflexive hook (grokked to the bone): we ARE the top curve

This session's model id is **`claude-opus-4-8`** (system prompt, verbatim). The paper tests **"Claude Opus 4.8 [3]"** — citing its own System Card — as **the top-ranked agent**: Table 2 has Opus 4.8 leading every time budget (39.0@2h → **51.3@12h**) and **best in all six category means** at 12h (Science 48.5, **Code 67.4** ← its strongest, Optimize 36.5, Knowledge 47.0, Math 55.0, Games 39.3). Figure 1's fit for Opus 4.8: **R²=0.998, S_max≈55, β≈0.93 (among the steepest), t_mid≈0.6h (the smallest — reaches the bulk of its ceiling soonest).** Fastest learner AND highest ceiling.

So EdgeBench is a paper **studying the exact agent running this campaign**, and our comma sub-0.15 campaign is an EdgeBench-class task: an executable environment, ≥12h (ours is multi-**month**), rich multi-level feedback (through-R verdict → byte-close → exact evaluator). Our strongest EdgeBench family is Code (67.4) — and our campaign is heavily Systems&SE + Optimization + Formal-Math + Scientific-ML. We are, precisely, the subject.

---

## 2. The law and its theory (rigorous, concise — both razors land here)

**Empirics.** Averaged over 134 tasks, the log-sigmoid tracks each of 5 models' 12h curve at R²≥0.997; the same form fits per-family (Fig 5), out to 28h/72h horizons (Fig 6, R²≥0.993), and **forecasts** the 6.5→12h remainder from the first 6.5h at R²≥0.997, RMSE<1.0 pt (Fig 7). Among S-curves on 12h/28h/72h pooled (Table 1), log-sigmoid RMSE **0.390** < log-probit 0.398 < log-Gompertz 0.402 < Weibull-CDF 0.404 ≪ log-linear 0.717. Crucially, the clean law is **emergent at the population level**: Fig 8 shows fit error falls monotonically as 1→134 tasks are averaged; a **single** task is "noisy and idiosyncratic," with "long plateaus, abrupt breakthroughs, and irregular regressions." **Hold this — it governs whether we can forecast our own single campaign (§5.3).**

**Theory (§3.3, the part that matters most).** A task is a latent graph $G$ of *score units* $i$ with weight $w_i$, normalized $\mu_i = w_i/\sum w_i$; $n_i(u)\in\{0,1\}$ marks unit $i$ unlocked; normalized score $x(u)=\sum_i \mu_i n_i(u)$. An edge $K_{ij}\ge 0$ measures how much an unlocked source $j$ helps unlock a locked target $i$, so a locked unit feels an **influence field** $h_i(u)=\sum_j K_{ij}n_j(u)$. If locked units unlock at a rate proportional to that field,
$$\frac{d}{du}\mathbb{E}[x\mid n] = \eta\sum_{i\in L}\sum_{j\in U}\mu_i K_{ij}\quad(\text{the weighted frontier cut }U\!\to\!L).$$
Mean-field (product-measure) approximation $\sum_{i\in L}\sum_{j\in U}\mu_i K_{ij}\approx \kappa\,\mu(L)\mu(U)$ gives the **logistic frontier equation**
$$\frac{dx}{du} = \beta\,x(1-x),\qquad \beta = \eta\kappa.$$
The change of coordinate $u=\log t$ is justified by **self-similar / scale-free (fractal) graph structure** — "if each additive increase in task difficulty exposes a multiplicatively larger amount of relevant graph structure," search volume grows exponentially with difficulty, so $u\sim\log t$ (they cite Bak–Tang–Wiesenfeld self-organized criticality [5], 1/f, "no single characteristic scale"). Substituting and separating variables yields exactly $x(t)=1/(1+(t_{\text{mid}}/t)^\beta)$.

**Read the parameters like an engineer:**
- $x(1-x)$: **unlocked mass = reusable capability × locked mass = remaining opportunity.** Growth rate is maximal at $x=1/2$ (i.e. at $t=t_{\text{mid}}$) — the moment of maximal learning-progress.
- $\beta=\eta\kappa$: **$\eta$** = how fast a unit unlocks given influence (≈ feedback density / iteration speed); **$\kappa$** = task-graph mixing (≈ **synergy** — how much each unlock propagates to others). Larger $\beta$ = steeper transition, reaches the ceiling over a narrower band of interaction time.
- $t_{\text{mid}}$: interaction time to half-ceiling. $S_{\max}$: attainable ceiling **over the fitted regime**, explicitly *not* an absolute upper bound.

This logistic-in-log-time is the **replicator / Verhulst / Fisher–KPP front-propagation** equation — the same object as SIR epidemic spread, Bass diffusion, and percolation. **Environment learning = a front propagating through a competence graph.** Keep that; §4 is built on it.

---

## 3. SCHMIDHUBER RAZOR — the theory IS POWERPLAY; four adaptations

**Formal isomorphism (verified).** POWERPLAY (Schmidhuber 2011, arXiv 1112.5309) maintains a solver and repeatedly searches for the **cheapest self-modification that solves at least one previously-unsolvable task without regressing on any already-solved one** (the never-regress validation). The set of solved tasks is a **monotone competence frontier**. Map term-for-term onto EdgeBench §3.3:

| POWERPLAY | EdgeBench frontier expansion |
|---|---|
| already-solved task repertoire | unlocked score-unit set $U(u)$ |
| the frontier of solvable vs not-yet-solvable | the cut $U\!\to\!L$ (unlocked→locked) |
| cheapest modification solving a new frontier task | a frontier unlock driven by influence field $h_i=\sum_j K_{ij}n_j$ |
| never-regress acceptance (monotone frontier) | best-so-far monotone + **continuous-experience retention** |
| *prescriptive search algorithm* | *descriptive mean-field scaling law* $dx/du=\beta x(1-x)$ |

**They are the prescription and the phenomenology of one object.** POWERPLAY says *which* frontier node to unlock next (simplest-unsolved); EdgeBench measures *what the aggregate curve looks like* when a mixed frontier expands — a logistic in log-time. This directly sharpens our cross-ref `powerplay_variant_ii_cost_isomorphism_v1` (compose-without-regression / never-regress acceptance): **EdgeBench is the measured empirical phenomenology of POWERPLAY Variant II**, which POWERPLAY posited only qualitatively.

**Adaptation S1 — learning-progress acquisition for lever selection.** The slope $dS/d(\log t)=\beta\,S_{\max}\,x(1-x)$ is a **compression-progress signal** (Schmidhuber's intrinsic reward = first derivative of compression), maximal at the frontier's mid-point. **Acquisition rule:** among *ready* (frontier) DAG levers — those whose prerequisites are unlocked — pick the one with the steepest predicted **ΔS-toward-floor per through-R compute-hour**, i.e. highest local $\beta\cdot(\text{ceiling-gap})\cdot(\text{floor-gap})$ in that lever's own coordinate. This is POWERPLAY's "simplest-unsolved-frontier" = highest learning-progress. We already partially do this (info-gain ranking, the "calibrate parametrization math-first tiered" memory + the DAG's basis-BEFORE-capacity *edge* ordering). EdgeBench contributes the **measured functional form** of the marginal-gain profile and the POWERPLAY grounding. *Campaign-strategy, not a lever.*

**Adaptation S2 — never-regress is empirically worth points (grounds the anti-forgetfulness spine).** Does the paper ground POWERPLAY's monotone-frontier? **Yes, decisively.** §5.2: Opus 4.8 with **continuous experience** (one 12h run, workspace/artifacts/feedback kept) scores **43.0** vs **without experience** (six independent τ=2h restarts, all state discarded) **36.1** — **+6.9**, and the gap GROWS with horizon (2h **+0.4** → 6h **+3.9** → 12h **+6.9**). Independent restarts = *no* monotone frontier (each restart re-derives from scratch); continuous experience = the never-regress accumulation. **This is the measured value of POWERPLAY's never-regress invariant** — and over a multi-month campaign (the longest horizon possible), the value is maximal. See §7 for the full anti-forgetfulness validation.

**Adaptation S3 — forecast-from-early-trajectory → early-stop-on-plateau.** *How they forecast:* Fig 7 fits $S(t)$ on only the first 6.5h and extrapolates 6.5→12h at R²≥0.997. **Our use:** fit the early per-stage witness descent to the log-sigmoid (loss-domain analog: performance $=(S_{\text{start}}-S)/(S_{\text{start}}-S_{\text{floor}})$), estimate the config's plateau $S_{\max}$; if the predicted plateau won't clear the need-band, **early-stop and pivot** rather than babysit a doomed vehicle. This directly feeds the **A11 early-stop lever #188** + the **don't-loop-the-wrong-vehicle** discipline ("if N ticks pass with no decisive new EXACT-relevant signal, STOP and pivot"). **HONEST caveat:** the paper's forecast precision is a *population* property (Fig 8 — a single task is jaggy with plateaus + breakthroughs). A single witness run's forecast has wide error bars, and a **known stage transition** (CE→tau→l7→Muon) can BREAK a plateau with a real breakthrough — so treat the forecast as a **weak prior + decision-support**, gate it on the DAG's known stage structure, never as certainty. Do not early-stop mid-CE expecting the Muon jump.

**Adaptation S4 — Speed-Prior / compute-bounded effort allocation + saturation detection.** $t_{\text{mid}}$ and $\beta$ together give the marginal-gain-per-hour profile: gain concentrates around $t_{\text{mid}}$, saturates after. **Saturation detector:** when observed $dS/d(\log t)$ falls below a threshold, the lever is on its flat tail → **reallocate compute to a fresher frontier node** (one near its own $t_{\text{mid}}$). This composes with our "training-time = lexicographic-secondary" discipline (stop a run only when S has plateaued = a T-minimization that respects S) and with the Speed-Prior/Kt 30-min inflate ceiling from `compression_as_intelligence_lineage_crossref` (which bounds how deep the free-interpreter decode can be).

---

## 4. YOUSFI RAZOR — feedback density is the measured driver; four extractions

Yousfi's contest is **inverse steganalysis**: the frozen detector (SegNet/PoseNet) IS the feedback channel, and its sensitivity field IS the map for where to spend changes (UNIWARD, detector-informed embedding). EdgeBench measures feedback as the engine of learning — the two frames fuse.

**Y1 — feedback density/richness as the measured driver.** §5.4 (gravitational-wave reconstruction, GW150914) quantifies "feedback turns failed probes into durable gains": across **247 scored evaluations**, best rises **42.8→67.0**, yet only **27 of 224** explicit submissions improved best-so-far by ≥0.1pp (**~12% effective rate**) — "sparse but **structured**." The mechanism: feedback repeatedly **changes what the agent searches for** — "make the problem measurable → decompose failures into searchable subproblems → identify the main bottleneck → keep the core, repair the remainder." **Map to us:** the per-epoch **through-R verdict** = the fast inner-loop dense feedback; the **margin-saliency map $d\,\text{margin}/d\,\text{input}$ (#141)** = a *richer* channel — not the scalar d_seg but the **per-pixel attribution of where the flips live** (the boundary annulus) = Yousfi's detector-sensitivity map = the UNIWARD cost. **Concrete adaptation:** ensure every verdict emits the **full multi-level feedback bundle** (scalar d_seg/d_pose + per-pixel margin-saliency + per-stage diff + per-class flip attribution — the surgical-repair toolbox), so each epoch maximally informs the next edit. We have the toolbox; EdgeBench says feedback *richness* is the measured engine → **prioritize wiring it densely** into the loop, not just logging the scalar.

**Y2 — the two-tier feedback structure = our proxy/authority split.** EdgeBench's harness (Fig 3) is explicitly **two-loop**: an inner loop (local, unlimited, fast, manipulable — compilers/linters/simulators) that "enables rapid progress," and an outer loop (gated, authoritative — hidden tests, rubrics, unseen seeds) that "guards against **overfitting to visible checks** and exposes failures the developer's own tests miss." This is **exactly our discipline**: through-R = fast advisory inner loop; **byte-close → `upstream/evaluate.py` exact eval = the slow authoritative judge, the ONLY pointer-mover**. The paper's "guard against overfitting to visible checks" IS our NO-FAKE surrogate-vs-authority firewall (through-R is advisory; only the exact eval is a score; MPS is never a score). **Honest:** the paper does not *ablate* the two tiers, so this is validation-by-design, not measured-tier-value — but §5.1's "effective submission rate" + "make it measurable first" strongly support that building the outer-loop authority path (**#202 byte-close-and-eval**) is the RIGHT first move, and using deliberate submission cadence (their submission cooldown ↔ our don't-hoard-but-don't-spam-Modal) beats flailing.

**Y3 — frontier ordering by the reward geometry.** In §3.3 the frontier cut is weighted by $\mu_i$ (score weight) and the actual edges $K_{ij}$ — progress concentrates on high-score-weight units on the frontier. Our reward geometry is $S=100\,d_{\text{seg}}+\sqrt{10\,d_{\text{pose}}}+25\,\text{bytes}/37.5\text{M}$, whose **marginal** sensitivities are operating-point-dependent (the CLAUDE.md SegNet-vs-PoseNet crossover) and whose **per-pixel** geometry is the margin-saliency field. Selecting levers by the score's actual marginal geometry (KKT-waterfill on margin-saliency, the boundary annulus) = Yousfi's "the detector's sensitivity IS the map." EdgeBench frames this as "frontier ordering by reward geometry" — fully consistent, we already do it; the framing sharpens *why* it's the correct ordering (it is the frontier cut).

**Y4 — within-run retention = resumability + long context.** §5.3: 1M-context Opus 4.8 beats 200k by a **stable +4.4–5.8** at every checkpoint, *even with identical external workspace/harness state*. So retention has two channels: in-context (the model window) AND external (workspace/checkpoints/memory), and **both** matter. Map: our **per-stage checkpoints + EMA-shadow + resume-from-disk** (external retention) + **MEMORY.md/DAG surviving compaction, reloaded each session** (the harness retention channel that lets a fresh session behave like continuous experience) = the retention discipline the paper measures as a driver. Validated.

---

## 5. The fractal unification — TWO fronts, one equation (the "Understand"-mind payoff)

The single most beautiful connection, and it is exact, not a metaphor:

- **Within the witness** (operator §OPERATOR PRIORITY): the SegNet argmax boundary is a **codim-1 front**; the capstone is a **variational level-set flow** propagating that front (Fisher-metric annulus, curvelet-finest, step-native). Front propagation in *pixel* space.
- **Within the campaign** (EdgeBench §3.3): environment learning is a **frontier expansion** — a front propagating through the latent task graph $G$. Front propagation in *competence* space.

**Both are the same Fisher–KPP / logistic front-propagation object** ($dx/du=\beta x(1-x)$ ↔ the level-set eikonal/reaction-diffusion flow). The operator already named the two-layer one-object structure ("the witness (physics facets) + the campaign (representational views) = the same pattern, and that double-unification IS the Understand-mind"). **EdgeBench supplies the campaign-layer front as a MEASURED law (R²=0.998)** — the abstract "campaign frontier" is not a loose analogy; it obeys the same replicator equation as the spatial level-set front. The TRIALITY cycle (DAG→DSL→run→measured-rows→equations→next-DAG) is the discrete update that advances the campaign front; the DAG *is* the latent task graph $G$; its nodes are score units; its edges (e.g. **basis-BEFORE-capacity**) are the $K_{ij}$ influence weights.

**And this hands us a strategic theorem for free.** $\beta=\eta\kappa$, where $\kappa$ = task-graph mixing = **synergy** (how much each unlock propagates). The operator's #3 priority — *"THE UNIFIED LEVEL-SET FLOW — one object, do NOT re-fragment into a grab-bag; all of this is related and the math falls out and fits perfectly"* — is, in EdgeBench's own variables, **maximizing $\kappa$**: a densely-mixed (unified) task graph makes every unlock cascade (high $\kappa$ → steep $\beta$ → reach sub-0.15 over a *narrower* band of interaction time); a fragmented grab-bag of disconnected sub-lanes has low $\kappa$ (unlocks don't propagate → shallow $\beta$ → the plateau drags). **Unification is not aesthetics — it is the frontier-propagation-speed multiplier.** This is a genuinely new, honest justification for the unify-don't-fragment discipline, in measured terms.

---

## 6. Anti-forgetfulness VALIDATION — the operator's north star is the measured engine

The operator, verbatim: *"the purpose of our whole project is deterministic reproducibility and no signal loss ever"* + *"SAVE knowledge and memories... if you apologize and then don't do anything and it happens again it makes me hate you."* EdgeBench turns this obsession into a measured law:

- **Continuous experience > independent restarts: +6.9 @12h, GROWING with horizon** (§5.2). The machine crash we recently survived (concurrent runs summed >128GB, crashed the box) is precisely an *independent-restart* event — the failure mode the paper says costs long-horizon performance. Our **black-box daemon + admission/memory governor + crash-recovery sweep + resumable-per-stage-checkpoints** are the machinery that keeps us on the *continuous-experience* curve instead of the *restart* curve. Over a multi-month horizon, the modelled gap is not +6.9 but the extrapolated tail of a gap that grows monotonically in horizon.
- **Longer context/retention: +4.4–5.8** (§5.3), stable across the horizon, *even with external state*. Our durable memory (MEMORY.md, the DAG, `.omx/research`, canonical-equations registry) is the external retention channel; the compaction-surviving reload is the in-context channel. The paper says both matter — validating that we need BOTH the durable files AND the tight <17KB reloadable index (the "goldfish" fix is not optional hygiene; it is the retention channel the +4.4 measures).
- **Feedback → durable gains** (§5.4): structured diagnose→edit→evaluate converts a 12%-effective probe stream into a monotone climb. Our many honest negatives (Wave-F ego-predict/correspondence NEGATIVES, analytic-lane break-even, AA-supersample-hurts) are EdgeBench's *failed probes that still sharpen the search* — provided each is retained and re-uses the crux (which our "negatives-are-deep-math-signal" + durable-marker disciplines guarantee).

**This reframes the anti-forgetfulness spine from "hygiene" to "the mechanism of long-horizon capability," with a frontier lab's R²=0.998 law + the +6.9/+4.4 ablations as the receipts.** That is the memo's highest-value output — not a lever, a *proof the discipline is the engine*.

---

## 7. Connectivity (unify, don't list)

- **POWERPLAY** (`powerplay_variant_ii_cost_isomorphism_v1`): SHARPENED — EdgeBench is the measured mean-field phenomenology of POWERPLAY frontier expansion; §5.2 empirically grounds the never-regress invariant (+6.9). §3.
- **Compression-as-intelligence** (`compression_as_intelligence_lineage_crossref_20260702`): the log-sigmoid **IS a compression-progress curve.** Unlocked mass = "reusable capability" = discovered/compressed structure; $x(1-x)$ = compression-progress rate; its peak at $x=1/2$ (i.e. $t_{\text{mid}}$) = the moment of **maximal compression progress = maximal Schmidhuber-interestingness = maximal learning**. So $t_{\text{mid}}$ is not just "half-ceiling time" — it is the run's most information-rich moment. This tightens the §9 flat-minima/MDL rate-lever framing: the campaign's own progress obeys the same compression-progress law as the artifact it produces.
- **#211 amortized meta-init (overfit-XOR-generalize):** a corpus hypernet $H_\psi(\text{scene})\to\theta_0$ is a **warm start = higher initial unlocked mass $x_0$** → shifts $t_{\text{mid}}$ earlier and lifts early-time performance. EdgeBench's §4.1 "disentangle prior knowledge from environment learning" (they SELECT tasks with matched first-attempt scores to control $x_0$) is exactly the amortized-init axis: prior knowledge sets $x_0$; environment learning is the frontier expansion from there. #211 = buying $x_0$ for free.
- **Adaptive per-video overfit-XOR-generalize:** the meta-controller's refine-budget $b(V)$ = choosing where to stop on each video's log-sigmoid; allocate budget to videos with the steepest remaining slope $\beta\,x(1-x)$ (highest marginal gain) = the §5.4/S1 learning-progress rule at the fleet scale.
- **Level-set flow (§OPERATOR PRIORITY):** the campaign frontier and the witness boundary are the same front-propagation equation (§5).

---

## 8. Honest EV verdict (full) + what concretely changes

**NOT a pointer-mover.** No d_seg / d_pose / byte lever. The pointer moves only through a byte-closed `upstream/evaluate.py` n600 exact row (#202); EdgeBench provides none. Say it plainly per the means/ends firewall.

**What it genuinely changes about HOW we run (campaign-strategy + validation):**
1. **Learning-progress acquisition rule (S1)** — order ready DAG levers by predicted marginal ΔS-toward-floor per through-R hour (local log-sigmoid slope). Refines existing info-gain ranking with a measured form + POWERPLAY grounding.
2. **Early-stop-on-forecast-plateau (S3)** — fit early per-stage descent, kill a vehicle the forecast says won't clear the need-band. Feeds #188 + don't-loop-wrong-vehicle. *Weak prior only; respect stage-transition breakthroughs.*
3. **Feedback-density maximization (Y1)** — emit the full multi-level feedback bundle (margin-saliency #141 + per-stage diff + per-class attribution) every verdict; feedback richness is the measured engine.
4. **Validation (S2/Y2/Y4 + §6/§7)** — the two-tier proxy/authority split, resumability, per-stage checkpoints, durable memory, and unify-don't-fragment are now *measured* drivers (+6.9 experience, +4.4–5.8 context, $\beta=\eta\kappa$ synergy), not just disciplines. This strengthens *why* we do them; it changes nothing tactical but raises their priority when they compete with a shortcut.

**What it does NOT justify:** it does not license curve-fitting our single-campaign score trajectory to forecast the sub-0.15 crossing (Fig 8: single-task jaggedness; we have essentially ONE clean self-produced pointer value, the rest borrowed/ancestor). That would be a toy dressed as a forecast — forbidden. The forecast machinery is legitimate only per-run/per-stage as decision-support (S3), or gated on a future clean population of per-lever score-vs-through-R-hours curves (see ledger reactivation trigger).

---

## 9. TRIALITY integration proposal (for parent review — do NOT force-register)

- **Canonical equation?** *GATED — do NOT register now.* A "campaign-forecast log-sigmoid law" with our exact-score trajectory as the `EmpiricalAnchor` fails our own registry discipline: we have ~one clean self-produced pointer value (0.19110 is a borrowed recode; our witness hasn't landed an exact row), and a single task is jaggy — the anchor would be phantom. The *structural* law (frontier-expansion → logistic-in-log-time) is DERIVED, not our empirical anchor, and would be an orphan (no producer/consumer). **Draft form, held:** `campaign_frontier_expansion_log_sigmoid_v1` — anchor = a future ≥8-point clean population of per-lever (or per-run per-stage) through-R descent curves; consumer = the S1 acquisition ranker + S3 early-stop. Flag: **blocked on the population of clean curves; register only when a consumer exists AND the anchor is real.**
- **DAG meta-node?** *Recommended — FRAMING, not a lever.* Add a meta-node to `sub015_DAG_topaiml_reopen_and_pursuit_plan` naming the campaign-as-frontier-expansion identity: the DAG **is** the latent task graph $G$; nodes = score units; edges (basis-BEFORE-capacity, pose⊥seg, etc.) = influence weights $K_{ij}$; the TRIALITY cycle = the frontier-advance update; $\beta=\eta\kappa$ says unification maximizes propagation speed. This is a durable reframe that makes the DAG's own structure legible in EdgeBench's measured terms.
- **DSL/doc note?** Add a paragraph to `docs/triality_dag_dsl_equations_deepmath.md`: the DAG→DSL→run→rows→equations→next-DAG loop IS the EdgeBench "environment learning" process; the S1 learning-progress ordering, S3 forecast-early-stop, and Y1 feedback-density are the measured operating rules; continuous-experience/retention (§6) is why the loop compounds instead of drifting. FRAMING.

---

## 10. Papers-checked ledger one-liner (append to `reference_papers_checked_...`)

> **EdgeBench 2026-07-02 (ByteDance Seed; edge-bench.org; 51/134 tasks public).** VERDICT: **FRAMING + VALIDATION + 3 campaign-strategy rules — NOT a pointer-mover** (no d_seg/pose/byte lever). Log-sigmoid law S(t)=S_max/(1+(t_mid/t)^β), dx/dln t=βx(1−x), R²=0.998, derived as frontier expansion on a latent task graph (β=η·κ, u=log t from self-similar/fractal structure) = **the measured mean-field phenomenology of POWERPLAY** (sharpens `powerplay_variant_ii_cost_isomorphism_v1`; §5.2 continuous-experience>independent-restarts **+6.9@12h growing with horizon** empirically grounds never-regress = our anti-forgetfulness/resumability spine as the MEASURED long-horizon engine, not hygiene). Reflexive: paper's TOP curve is **Claude Opus 4.8** = this session; our sub-0.15 campaign IS an EdgeBench-class task; the DAG IS the latent task graph. CONCRETE USE (campaign-strategy, not a lever): (1) **learning-progress acquisition** — order ready levers by local log-sigmoid slope (ΔS-toward-floor per through-R hour); (2) **early-stop-on-forecast-plateau** (fit early per-stage descent, kill a vehicle that won't reach the need-band; feeds #188 + don't-loop-wrong-vehicle; WEAK prior — single-run jaggy, respect stage-transition breakthroughs); (3) **feedback-density maximization** — emit full multi-level feedback (margin-saliency #141 + per-stage diff + per-class attribution) every through-R verdict = Yousfi detector-informed feedback. β=η·κ ⇒ **unify-don't-fragment = maximize κ = frontier-propagation speed**. REACTIVATION/USE TRIGGERS: (a) if we accumulate ≥8 clean per-lever (or per-run per-stage) score-vs-through-R-hour curves → fit population log-sigmoid to forecast sub-0.15 crossing (single-campaign fit is a TOY — Fig 8 population-only — do NOT do it before then); (b) use "effective-submission-rate" (fraction of measurements advancing the crux) as a campaign-health diagnostic separating frontier-expansion from means-hoarding. Full memo: `.omx/research/edgebench_scaling_laws_deepdive_20260703T033159Z.md`.

---

*Sisters:* `powerplay_variant_ii_cost_isomorphism_v1` · `compression_as_intelligence_lineage_crossref_20260702` · `feedback_save_memories_not_apologies_anti_forgetfulness_20260701` · `feedback_never_launch_non_resumable_per_stage_checkpoints_20260627` · `project_unified_variational_levelset_flow_20260701` · `project_witness_dsl_and_dag_dsl_duality_20260629` · `feedback_allergic_to_non_n600_scale_no_toys_20260701` (single-task jaggedness = why a single-campaign forecast is a toy). Pointer **0.19110 UNMOVED** — MEANS memo; worth = a sharper campaign strategy + the measured proof that our no-signal-loss spine is the long-horizon engine.
