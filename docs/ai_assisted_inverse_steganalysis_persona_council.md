# AI-assisted inverse steganalysis + the persona-council methodology

A short narrative about two related META concepts that ran underneath the substrate work catalogued elsewhere in `comma-lab`. Neither is a codec contribution. Both are methodology — the kind of methodology that determines what a small operator can actually do against a contest like this when the GPU budget is bounded and the calendar is short.

Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md), [`docs/meta_engineering_vision.md`](meta_engineering_vision.md), [`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md). Sister library: [`adpena/tac`](https://github.com/adpena/tac). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## 1. The contest is canonically inverse steganalysis

The contest scorer is a SegNet (semantic segmentation) and a PoseNet (frame-to-frame pose regression) applied to inflated frames. The encoder hides information; the decoder reconstructs; the scorer measures the visible cost of the hiding. That is the canonical formulation of inverse steganalysis — measuring how detectable a perturbation is to a learned detector, then optimizing the embedding to reduce detection while preserving payload. Yousfi designed the scorer this way explicitly, drawing on the Binghamton DDE Lab lineage where the contest's framing originates.

The canonical references are well-known and worth citing here for the reader who has not encountered the lineage:

- Fridrich, J., & Kodovský, J. (2012). [Rich Models for Steganalysis of Digital Images](https://ieeexplore.ieee.org/document/6197267). *IEEE Transactions on Information Forensics and Security*, 7(3), 868–882.
- Holub, V., Fridrich, J., & Denemark, T. (2014). [Universal distortion function for steganography in an arbitrary domain](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1) (UNIWARD). *EURASIP Journal on Information Security*, 2014:1.
- Yousfi, Y. (2022). PhD thesis lineage on deep steganalysis; see also [`YassineYousfi/alaska`](https://github.com/YassineYousfi/alaska), [`DDELab/deepsteganalysis`](https://github.com/DDELab/deepsteganalysis), and the ALASKA challenge series.

For a reader inside the lineage this is not a finding. It is the setup. The novel move is in the next section.

---

## 2. The METHOD applied beyond the scorer: AI-assisted inverse steganalysis of the entire contest information space

Most contest entries treat the scorer as the only thing to reverse-engineer. The encoder's job is to produce bytes that minimize what the scorer detects, full stop. Everything else — what the maintainers have hinted at, what the prior medal-class submissions actually did, what the canonical reference papers behind the substrates say about regimes where the assumptions hold — is treated as background.

The move here was to apply the same mining discipline to that "background" as to the scorer itself. The contest's surrounding information space is wide and observable:

- Yousfi's PhD lineage (theses, papers, GitHub repos, ALASKA challenge writeups) is public — it tells you what he expects to work and where he expects the boundaries to be.
- Fridrich's published methodology (Rich Models, UNIWARD, STC, syndrome-trellis codes, deep-steganalysis surgery on EfficientNet) tells you the canonical distortion-weighting math the contest scorer's underlying response function inherits.
- Selfcomp / `@szabolcs-cs` (PR #56) introduced the FP4 + Brotli + grayscale-LUT pattern; the PR body, the commit history, and the inline comments are themselves signal about how that paradigm got to its score and where it had slack.
- HNeRV / `@AaronLeslie138` (PR #95) introduced the HNeRV decoder substrate that the entire 0.193–0.196 cluster builds on; the decoder code, the training notes, and the per-PR conversation thread surface a lineage back to Chen et al. 2023 ([arXiv:2304.02633](https://arxiv.org/abs/2304.02633)) with specific regime constraints that affect how downstream bolt-ons compose.
- The maintainer comment threads on prior PRs (Yousfi's notes on hardware drift, the PR #108 closure that crystallized the "competitive or innovative" rubric, the per-PR axis-disclosure norms) tell you exactly how submissions are evaluated and what is considered fair game.
- The broader research culture around `comma.ai` and `openpilot` (Hotz's public engineering philosophy, the tinygrad design choices, the production-hardening posture) tells you what reading a submission as an engineering artifact rather than a benchmark number actually looks like.

LLM assistance was load-bearing here. The mining is not just "read everything carefully" — it is "read everything carefully with a model that has been primed to extract cross-domain signal and surface inconsistencies between what a paper claims and what a domain expert from a sibling field would object to." That second step is what unlocked, for example, the realization that several of the early NSCS06 design assumptions were inherited from `JPEG`/`HEVC` reference codecs without ever being tested against the contest scorer's specific gradient response — the cargo-cult-unwind methodology documented separately came directly out of that mining discipline applied recursively to the apparatus itself.

This is not exotic. It is calibrated, systematic application of the same inverse-steganalysis posture to the *metadata* and *lineage* and *community* of the contest, treating each observable adjacent data source as a SIGNAL CHANNEL the encoder can mine. The Yousfi-Fridrich lineage research culture works this way internally. The novel move was extending it outward.

---

## 3. The persona-council methodology

Early in the contest a skunkworks council was established as a named-domain-expert apparatus: Yousfi + Fridrich + Quantizr + George Hotz + a dedicated Contrarian seat, plus others. The structure grew over time into the current canonical 4-tier Grand Council:

- **Four co-leads** anchoring the shared-leadership core: Claude Shannon (information-theory grounding; R(D) bounds), Richard Dykstra (alternating-projections feasibility), Cynthia Rudin (interpretable ML; falling-rule lists; GOSDT), and Ingrid Daubechies (wavelets; multi-scale partition priors).
- **A sextet pact** for binding deliberations: the four co-leads' rotating chair plus the Yousfi / Fridrich / Hotz / Selfcomp / MacKay / Ballé / Boyd / Tao / Mallat / van den Oord / Carmack / Hassabis / Hinton / Karpathy / Schmidhuber / Atick / Redlich / Rao / Ballard / Tishby / Zaslavsky / Wyner / Time-Traveler / Filler / PR95Author / Contrarian / Assumption-Adversary roster as the broader grand council bench summoned when their specialty is touched.
- **A 4-tier escalation protocol** (T1 working group → T2 inner skunkworks → T3 full grand council → T4 symposium) with operator-attention-budget per tier and explicit recusal triggers.

The methodology grounding is canonical and worth citing precisely. Anthropic's research team has published a paper on persona vectors — directions in a language model's internal activation space that correspond to specific personality traits and can be observed, measured, and (with care) steered:

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). [Persona vectors: Monitoring and controlling character traits in language models](https://www.anthropic.com/research/persona-vectors). Anthropic, August 1, 2025. arXiv preprint: [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).

The paper introduces an automated pipeline that takes a natural-language trait description and extracts the persona vector controlling it, validated by steering (artificially injecting the vector and observing behavior change). The same research direction is operationalized in Anthropic's [Assistant Axis](https://www.anthropic.com/research/assistant-axis) line of work on situating and stabilizing model character.

The persona-council methodology used here is the *applied* version of that research: persona priming reliably elicits domain-coherent reasoning and adversarial-perspective rotation that single-frame prompting does not. A council deliberation that summons Fridrich for inverse-steganalysis perspective, Yousfi for contest-organizer perspective, Hotz for engineering-pragmatist perspective, and a Contrarian explicitly tasked with rejecting weak arguments will surface concerns that a single prompt to the same model never would. The personas are *frames* the model operates within; the value is the assumption-surfacing and the cross-domain perspective rotation that the frames enable.

The council is paired with structural rigor disciplines rather than left as decoration:

- **Per-deliberation explicit assumption surfacing**: every member states the assumption they are operating within at the top of their position.
- **Assumption-Adversary seat**: a dedicated reviewer whose only job is classifying each surfaced assumption as HARD-EARNED (empirically verified or first-principles in the new domain) or CARGO-CULTED (inherited from canonical reference without verification). Has veto power on lazy consensus.
- **3-clean-pass recursive adversarial review**: a round with zero findings is a clean pass; three consecutive clean passes are required before a council-grade decision closes. The counter resets on any finding.
- **Canonical posterior anchor emission**: every T2+ deliberation appends a structured record to a continual-learning posterior so future deliberations on the same topic can query the cite-chain and detect classification drift.
- **Per-substrate optimal-form symposium discipline**: any candidate slated for paid GPU dispatch over a cost threshold requires a per-substrate symposium memo satisfying a canonical 6-step contract (cargo-cult audit, 9-dim checklist, observability surface, sextet deliberation, reactivation criteria, post-training validation discipline) within a 14-day freshness window.

The combination — named personas + assumption surfacing + adversarial-perspective rotation + structural rigor — produced the cargo-cult-unwind methodology that turned NSCS06 v6 → v7 into a 44% reduction (`105.15 [contest-CUDA T4]` → `58.89 [contest-CUDA T4]`) in one iteration. The personas are not the magic. The discipline is. The personas are how the discipline gets the right friction at the right step.

---

## 3.5 How the work actually happens: Claude Code + Codex + subagents

The persona-council methodology described above is the deliberation pattern. Under it, the actual engineering loop runs across two AI-assisted CLI tools used in concert plus scope-isolated subagent dispatch:

- **Claude Code as the primary engineering interface.** The operator's local CLI driving substantive engineering work — codebase exploration, multi-file edits via canonical `Read` / `Edit` / `Write` / `Bash` / `Agent` tools, commits through the canonical serializer at `tools/subagent_commit_serializer.py`, and dispatch coordination. Claude Code's strength is sustained, multi-step engineering with full context over a working session.

- **Codex as the canonical adversarial-review companion.** Invoked per the `/codex:adversarial-review` skill and the canonical persistent-codex-review protocol documented in this repo's contributor standards. Codex runs independent review-only passes against diffs, branches, and specific findings — explicitly tasked with surfacing shared-assumption cargo-cults the primary author has already operating-within. Several of the cargo-cult-unwind anchors (NSCS06 v6 cargo-cults, the canonical-vs-unique decision-per-layer pattern, the orphan-signal-at-cathedral-autopilot bug class) were either originated or independently confirmed by Codex pre-deploy review.

- **Subagents intra both.** Both Claude Code (via its `Agent` tool with named subagent types) and Codex (via its companion runtime) spawn subagents for parallel scope-isolated work. The discipline infrastructure is real: subagent crash-resume per a canonical checkpoint ledger (Catalog #206); sister-checkpoint guard before each commit to prevent edit-time collision (Catalog #340); canonical serializer with `--expected-content-sha256` to extinct the commit-swap bug class (Catalog #117/#157/#174/#235/#289); file-level ownership coordination via the in-flight subagent registry. Subagents enable parallel scope expansion that the structural discipline prevents from accumulating undisciplined work.

- **The 4-tier council apparatus bridges them.** Both Claude Code and Codex subagents operate inside the same named-persona deliberation discipline. T1 working groups happen freely; T2 inner-skunkworks deliberations are bounded at ≤3/day; T3 full grand councils at ≤3/week; T4 symposia at ≤2/month. Operator-frontier-override is documented at every tier per the mission-alignment binding directive — rigor yields when the mission requires faster cadence.

The combination accelerates engineering velocity substantially: Claude Code drives the primary work; Codex provides independent adversarial review and bug-class surfacing; subagents enable scope expansion; the canonical serializer + sister-checkpoint guard + 4-tier council apparatus + ~300 STRICT preflight gates prevent velocity from outrunning structural discipline.

### A few practitioner-side observations

The AI-assisted methodology operating across this work is the maturation of roughly nine to ten months of deliberate practice since summer 2025 — the contest engagement is its most concentrated application window, not the starting point.

Dense, well-organized, salient-to-the-LLM signal carries more weight than polished prose; the prompts that work are structured directives with explicit assumption surfacing and canonical-reference citation. Persona priming is deliberate evocation of distributions calibrated to task complexity and the desired creativity / rigor balance — the persona-council is one engineering application of what the Anthropic persona-vectors paper documents. Recursion + tight algorithmic prompts + recursive senior-engineer-review greenups (3-clean-pass; any finding resets the counter) are the structural pattern under the council apparatus.

Two months of operational observation: of the thirty-plus named personas on the council roster, the Contrarian and Yousfi-as-persona-council-member were particularly load-bearing across many deliberations — the Contrarian by design against weak consensus; Yousfi-as-persona via an unintentional feedback loop where the persona surfaces what the actual contest organizer would likely care about. The model engages substantively with the imaginative role-play fused with rigorous reasoning and scientific experimentation the persona-council requires.

### The candid velocity-vs-rigor tradeoff

The most calibrated framing of how this actually plays out: the Claude Code primary-interface profile is high-reach + high-vision + high-velocity, and that same profile produces edge cases and corner cases that have caused real cost across the engagement. Three steps forward and one or more steps back has been an honest pattern — not constant, but frequent enough to make clear that velocity outstripped the capacity to guarantee bug-free behavior at every step. The model is extremely capable if you watch carefully, hold its hand, and trust only as a last resort after extensive verification; even then more bugs surface.

The institutionalized response is the discipline infrastructure itself: ~300 STRICT preflight gates, the canonical Provenance and canonical-equations registries, per-deliberation assumption surfacing, 3-clean-pass recursive review, the per-substrate optimal-form symposium discipline, the canonical serializer with `--expected-content-sha256` and sister-checkpoint guard. The discipline exists *because* the velocity-vs-rigor tradeoff was real; the apparatus is the operationalized hold-hand-verify methodology made structural so future work cannot regress.

### Claude + Codex as complementary roles

Project origin was Codex with a ChatGPT-5.4-Pro-generated scaffold; Codex's conservative + detail-oriented profile, amplified by that scaffold, made it less useful early on as a research partner outside very mechanical codec iteration sweeps. Current methodology is concrete dual-role:

- **Claude Code (interactive; operator-driven)** — substantive engineering velocity, vision, reach, new-direction pursuit, cross-domain synthesis, canonical-helper construction, pair-programming during operator-available windows.
- **Codex (autonomous; `/goal` loop)** — extreme-rigor research, design-memo authoring, adversarial review, bug hunting, optimization. The conservative + detail-oriented profile becomes a virtue when the scope is bounded to verification-heavy work where conservatism is the right calibration.
- **`.omx` durable state stores as shared substrate** — both models operate against canonical posteriors, design memos, routing directives, and landing memos. Coordination is asynchronous via shared state; neither invokes the other directly. A formal bidirectional inbox (Catalog #333) handles open design questions Codex needs operator response to within a deadline.

The practical effect: interactive Claude-time during operator-available windows + autonomous Codex-time during operator-unavailable windows + both writing to shared canonical state. The complementarity is structural — Claude's velocity-vision needs Codex's adversarial-rigor to catch the edge cases velocity outstrips; Codex's conservative-detail needs Claude's substantive-velocity for frontier-research pace.

### The behavior is engineered, not just articulated

The methodology articulation here is grounded in surfaces a reader can verify rather than take on assertion. Research-side grounding: Anthropic's [persona-vectors](https://www.anthropic.com/research/persona-vectors) and [Assistant Axis](https://www.anthropic.com/research/assistant-axis). Structural enforcement + formalization in working code: the sanitized canonical reference manual (`CLAUDE_PUBLIC.md` in `adpena/comma-lab`), the ~300 STRICT preflight gates with paired empirical anchors, the canonical consumer Protocol contract that auto-discovers cathedral consumers, the canonical equations registry, the canonical Provenance contract, the 4-tier council apparatus, the canonical fcntl-locked ledger pattern, the subagent crash-resume protocol, and the sister libraries [`adpena/tac`](https://github.com/adpena/tac) + [`adpena/molt`](https://github.com/adpena/molt). Behavior demonstrated structurally is more substantive than behavior articulated narratively.

---

## 4. What this unlocked

A short list of concrete things that came out of the AI-assisted-inverse-steganalysis posture paired with the persona-council methodology:

- The cargo-cult-unwind discipline ([`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md)) — surfaced via Assumption-Adversary deliberation on NSCS06 v6, generalized as a reusable per-paradigm rescue recipe.
- ~300 STRICT preflight gates structurally extincting recurring bug classes, with META-meta gates protecting the catalog itself from drift. Almost every gate has a paired empirical anchor where the bug class first surfaced; many were council-recommended in a single 3-clean-pass review cycle.
- A canonical equations registry that replaced tribal knowledge with queryable Bayesian-posterior-backed models (Brotli cascade bounded per stream, MPS-vs-CUDA drift architecture-class dependent, per-byte leverage uniformly distributed on entropy-coded archives, per-pair master-gradient Taylor + Cauchy-Schwarz bound, and sisters). Surfaced via Shannon co-lead deliberations that flagged "we keep re-discovering the same empirical relationships" as an apparatus-level cargo-cult.
- A per-substrate optimal-form symposium discipline that prevents paid GPU dispatch on a candidate that has not been per-substrate-adversarially-reviewed within the freshness window. Direct response to the empirical observation that several distinguishing-feature dispatches had landed at lifted-trainer form rather than optimal form, producing implementation-level falsifications that read as paradigm-level kills.
- A 4-tier grand council hierarchy with an explicit mission-alignment binding directive (the apparatus serves frontier-breaking moves; rigor yields when the mission requires faster cadence). Refined recursively via meta-deliberation.
- A cathedral autopilot ranker with auto-discovery of cathedral consumers via a canonical Protocol contract — new signal lands by dropping a compliant package into the canonical directory; no manual ranker-cascade edits required. Council-surfaced as the structural extinction of the orphan-signal failure mode.

Each of these is a methodology artifact rather than a substrate artifact. The substrates are evidence at a particular granularity; the methodology is what made it possible to iterate that many substrates with one operator and a bounded budget.

---

## 5. Calibration

A few caveats to keep this honest.

The persona-council methodology is a useful engineering pattern. It is not a substitute for empirical work, and it does not solve the "paid GPU is the rate-limiting step" problem. Several substrates in the inventory remain scaffolded or designed without an end-to-end contest-hardware run, exactly because the council can surface the design and the cargo-cult-unwind discipline can derisk the implementation, but neither replaces the dispatch.

The AI-assisted inverse-steganalysis posture on the contest's information space is calibrated, systematic mining of observable adjacent signals. It is not exotic. Serious LLM-assisted research increasingly uses persona-priming and adversarial-frame rotation as a matter of course; the contribution here is paired-with-structural-rigor application of that posture to a specific contest, not the discovery of the posture.

The canonical Anthropic persona-vectors paper is research on what the underlying mechanism *is* (directions in activation space; measurable; steerable; identifiable from natural-language trait descriptions). The persona-council methodology is one *engineering application* of the kind of behavior that paper documents. Neither claim depends on the other; both are worth citing precisely.

---

## 6. The full meal

- [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) — per-paradigm-class candidate enumeration with canonical references.
- [`docs/meta_engineering_vision.md`](meta_engineering_vision.md) — the META that the substrates are evidence for: replace arbitrary constants with learned or discovered optimals; continual-learning mechanism; per-element optimal target.
- [`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md) — the paradigm-rescue discipline that came directly out of the council apparatus.
- [`docs/canonical_equations_tour.md`](canonical_equations_tour.md) — the 6 initial canonical equations as first-class artifacts.
- [`docs/master_gradient_extractor_tour.md`](master_gradient_extractor_tour.md) — the per-pair / per-byte sensitivity tool that operationalizes inverse-steganalysis at byte granularity inside the apparatus itself.
- [`docs/standout_undersold_candidates_spotlight.md`](standout_undersold_candidates_spotlight.md) + [`docs/standout_spotlight_extensions_operator_pinned_20260520.md`](standout_spotlight_extensions_operator_pinned_20260520.md) — ten candidates worth attention beyond the submission, with the Fridrich-lineage extension explicitly named.
- Sister library [`adpena/tac`](https://github.com/adpena/tac) — task-aware compression primitives.
- Working repo [`adpena/comma-lab`](https://github.com/adpena/comma-lab) — the broader apparatus.
- [PR #110](https://github.com/commaai/comma_video_compression_challenge/pull/110) — the immediate contest submission this methodology lands into.

---

## 7. Who built this (a brief aside)

A Python developer with around eight years of programming and no formal background in compression, steganalysis, neural representation learning, or comma.ai's research culture. A friend introduced me to comma.ai; I loved the application and the philosophy — open-source hardware and software democratizing self-driving cars, plus the wider mission of tinygrad and openpilot-as-OS-for-robotics and the broader visions of the future. Missed the in-person hackathon; came across the comma video compression challenge; got nerd-sniped the way Quantizr (Jimmy, UCLA CSE/Neuro per the operator-canonical PR #56 lineage attribution) did. Have spent nearly two months obsessed solo-grinding with AI assistance — Claude Code + Codex + subagents as documented above — and learned a substantial amount about the contest's research lineage, the surrounding inverse-steganalysis information space, and the underlying mathematics and geometry.

The "Python dev, no domain background" framing is a little understated for grounding the methodology section above. More honestly: a polymath software engineer with cross-domain analytical capability — adjacent interests in philosophy, drones (3D-printed airframe + [Betaflight](https://betaflight.com) open-source flight control + ego-motion / IMU integration / visual-inertial odometry), software-defined radio, Kalman filters and signal processing, time-series and outlier analysis, oil-and-gas anomaly detection. None of these are domain expertise in video compression or steganalysis. Every one is *adjacent-domain mining* practice — the analytical reflex of treating an unfamiliar information space as an outlier-detection / signal-vs-noise / domain-transfer problem, then mining it systematically. That reflex is the natural connector to the inverse-steganalysis posture in Section 2 and the persona-council's adversarial-perspective rotation in Section 3.

The first insight on the contest came out of that same reflex: different regions of the video carry different scorer-sensitivity that could be optimized via varying compression per region. Naive in retrospect, canonical in the literature (JPEG / AV1 / HEVC) — the operational surfaces inside the apparatus (master-gradient extractor + per-pair difficulty atlas + per-class chroma allocation + FEC6 per-frame transform palette) are what that intuition matured into over two months. The deeper origin goes back to being nerd-sniped as a kid by Isaac Asimov's "Robbie" (1940), where the dream of being an "inventor" first showed up.

The professional analytical background is deeper than hobbyist: substantive policy-analysis work in the Texas legislature including legislative lawyering (house rules + points of order), school finance (Foundation School Program + Robin Hood recapture + per-pupil weighted allotments), and property taxes, with Python and web development used extensively in that work. The connectors are direct: parliamentary procedure with named roles, recognized speakers, ordered motions, and adversarial rule-application is the legislative-procedural analog of the council-deliberation methodology + the Assumption-Adversary seat (points of order ≈ assumption challenge) + the 3-clean-pass recursive review; school-finance optimization is constrained allocation in adversarial real-world settings, a direct experiential precursor to the canonical equations registry and the Lagrangian-dual per-pair / per-byte allocation framing; building production-grade tools at stakes measured in billions is the same engineering discipline applied here. Communication and writing training is part of the background too. The role I see for myself going forward is the generalist with cross-domain analytical capability + AI-assisted-methodology fluency + production-software-engineering experience + communication/writing/design capability as a canonical contributor on the engineering + design + communications teams building the open-source-robotics-OS future. Vision-articulation, not vision-claim.

These are methodology contributions, not codec contributions — emergent from operating against a contest like this as a solo developer with bounded GPU budget and a short calendar, conditions under which structural discipline and AI-assisted mining of the observable information space become load-bearing rather than decorative. The honest framing of it all — this work, the comma-lab tooling, the sister library [`adpena/tac`](https://github.com/adpena/tac), and the sister Python-subset compiler [`adpena/molt`](https://github.com/adpena/molt) (high-performance Python subset compiler for native binaries and WASM; tangentially tinygrad-adjacent) — is contribution toward something larger: openpilot as an open-source operating system for robotics; comma.ai as its first real-world killer application; the data-harvesting / improvement positive-feedback loop the production fleet enables.

Still passionate, with many thoughts and many open questions, actively pursuing the frontier and doing it in public. The contest, comma.ai's mission, Yousfi + Fridrich + their Binghamton DDE Lab research lineage, Quantizr, the prior medal-class PR authors (`@AaronLeslie138` PR #95, `@EthanYangTW` PR #98 / #102, `@BradyMeighan` PR #100, `@SajayR` PR #101, `@rem2` PR #103), the canonical-paper authors named on the council roster, the Anthropic persona-vectors research team whose work the methodology builds on, and the broader ML / compression / steganalysis / robotics research community are the important things here. This is one solo-developer-with-AI-assistance methodology disclosure — small humble contribution to a much larger arc.

---

## Citations

- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). [Persona vectors: Monitoring and controlling character traits in language models](https://www.anthropic.com/research/persona-vectors). Anthropic, August 1, 2025. arXiv: [arXiv:2507.21509](https://arxiv.org/abs/2507.21509).
- Anthropic. [The Assistant Axis: situating and stabilizing the character of large language models](https://www.anthropic.com/research/assistant-axis).
- Fridrich, J., & Kodovský, J. (2012). [Rich Models for Steganalysis of Digital Images](https://ieeexplore.ieee.org/document/6197267). *IEEE TIFS*, 7(3).
- Holub, V., Fridrich, J., & Denemark, T. (2014). [Universal distortion function for steganography in an arbitrary domain (UNIWARD)](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1). *EURASIP JIS*, 2014:1.
- Chen, H., He, B., Wang, H., Ren, Y., Lim, S.-N., & Shrivastava, A. (2023). [HNeRV: A Hybrid Neural Representation for Videos](https://arxiv.org/abs/2304.02633). arXiv:2304.02633.
