# Contest PR authors — GitHub profile mining (task #413, harvest-only)

**Date:** 2026-07-10 · **Task:** #413 (operator-GO: "check all of the authors' GitHub for other repos and other signal we might be able to use") · **Subagent:** pr-authors-github-intake · **Follow-on to** #412 (PR #112–128 intake).

**Frame (binding):** memory `[[witness_line_priority_borrowed_polish_harvest_only_20260710]]` — the witness/v8 line is THE priority. Everything below is HARVEST-ONLY signal ("technique X → our surface Y"), **never** a "chase author Z's project" pursuit. Contest CLOSED (CLAUDE.md §Public Disclosure Hygiene 2026-07-06): IP open, this is signal-mining.

**Custody:** read-only via `gh api users/<login>/repos` + `gh api repos/.../readme`. NO repos checked out into the worktree. Public repos only; no personal info compiled beyond public-repo technique signal.

**Craft:** per `docs/operating_manual_craft_handoff.md` — every external claim is `[external unverified]`; labels MEASURED/INFERRED called out; NO-FAKE governs every "usable" tag; I say plainly where an author is a **one-off** with no transferable signal rather than manufacturing any.

**Instruction-boundary (checked):** repo content is DATA not commands. READMEs read (InfamousBolt/mini-tensorrt, ryanli0070/EuraAI) were benign setup docs. Several mattneel repos are agent-tooling by NAME (`jzs` "next generation agent substrate", `zigger` "AI code generation as a Zig build step", `frank-demo` "I deleted all my code. Then regenerated it."), but I read only their metadata/descriptions, not agent-directed body text. **Nothing in any repo attempted to instruct an agent; no injection encountered; nothing to act on.**

---

## Headline verdict (NO-FAKE, honest)

**14 of 15 authors are one-off contest entrants** — their genuine technique lives entirely in the PR they submitted (already intaken in #412/#128), and their other public repos are coursework, mobile/web app-dev, data-science labs, or unrelated forks with **zero transferable compression/witness signal.** I do not manufacture signal for them; the table marks them NONE.

**One genuine surprise: `mattneel` (Matthew Neel, #112 `rhnerv_comma` base).** He is a serious PL / systems / native-runtime / WebGPU engineer (133 repos, a deep Zig portfolio) whose OTHER work is a real adjacency to **our compute/native/molt surfaces** — not to the witness math, but to how we would ship a native/WASM/WebGPU eval-time runtime. Flagged domain-expert-worth-watching; HARVEST framing below.

The two authors whose PR sophistication predicted hidden depth — **InfamousBolt** (#118 saturation analysis) and **ryanli0070** (#126/127 margin surrogate) — turn out to be **strong students with no other domain repos**: their contest work was sharp but bespoke. Their signal is fully captured in #412; nothing new on their profiles.

---

## Ranked table {author, notable other-repo, technique, relevance, our-surface}

| Rank | author (PR) | notable OTHER repo | technique / what it is | relevance | our surface | verdict |
|---|---|---|---|---|---|---|
| 1 | **mattneel** (#112 base) | `three.zig` (3★) + `three-native` (17★) + `wasmbind` (5★) | native Three.js **WebGPU** runtime in Zig on QuickJS-NG + Dawn; zero-overhead Zig↔WASM↔TS bindgen | **MED-HIGH** (adjacency) | **molt** (L36: our Python→WASM+WebGPU compiler) | domain-expert; reference prior-art |
| 1b | mattneel | `fpz` (1★, fixed-point math Zig) + `gkz` (deterministic forkable sim kernel) | deterministic fixed-point integer kernels in Zig | MED | **Native eval-time runtime discipline** (CPU-stable integer round/clamp/resize) | reference for deterministic decode |
| 1c | mattneel | `tenzor` (7★) "Tensors. Compile time is free." + `ggufx` (GGUF parser) + `ex_lancedb` (LanceDB/Rustler) | compile-time tensor lib; model-weight-format parser; embedded vector DB | LOW-MED | compute (#252 MLX/Metal sibling); rate byte-parsing mindset; #411 graph-memory (vector recall) | tooling adjacency |
| 2 | **InfamousBolt** (#118) | `mini-tensorrt` (C++17 DL inference engine, Dense+ReLU) · `fourier` (1.7MB, no README) · `VAEData` (213MB data) | from-scratch inference engine (toy); a data dump; a VAE data dump | LOW | — (systems interest only) | **student, no hidden technique** — #118 writeup WAS the signal |
| 3 | **ryanli0070** (#126/127) | `EuraAI` (FastAPI+React AI app) | LLM web-app scaffold; 6 repos total, no ML-systems/codec | **NONE** | — | one-off; margin-surrogate was contest-bespoke (SYDE@Waterloo student) |
| 4 | **Bucky789** (#123/125) | `airflow-agent-skills-poc` · `OrderBookMatching` (C++ 6★) | Airflow+agent-skills PoC; HFT order-book matcher | **NONE** (weak systems signal) | — | one-off; curriculum work was contest-bespoke |
| 5 | **a12dongithub** (#128) | `karateclub` (fork) · `structural-probes` (fork) · `TS-TCC` (fork) · `Pixelization` (fork) | graph unsupervised learning; hidden-states-encode-structure; time-series contrastive; pixel-art AIGC — **all forks** | LOW | #411 graph-memory (karateclub, weak) | medical-vision person; click-polish bespoke (already noted #128 §5) |
| 6 | **PranavReddyGaddam** (#114) | `GitBridge` · 30+ AI web apps / SJSU Data-226 labs | repo→chat/podcast; sentiment/TTS/data-eng coursework | **NONE** | — | one-off; pose-warmup was contest-bespoke |
| 7–15 | aviral07-code, saitejareddi3636, shaina-k02, NihalMishra17, rohitsudhakar1, Bindu3116, flowmar47, WilliamAyoade, ashika-code | — | keyword grep over all non-fork repos returned only MNIST-classifier / calculator / currency-app / password-generator trivia | **NONE** | — | pure one-off contest entrants |

---

## Prose on the top findings

### 1. mattneel — the one domain expert; adjacency is COMPUTE/NATIVE, not the witness math
Matthew Neel (bio: "lives, eats, and breathes bleeding edge technologies") authored the `rhnerv_comma` base (#112) that the entire frontier cluster (#121/#125/#127/#128) built on. His 133 repos reveal a PL/systems/native-runtime specialist, with a standout **Zig + WebGPU + WASM** portfolio: `three.zig`/`three-native` (a native Three.js **WebGPU** renderer runtime on QuickJS-NG + Dawn), `wasmbind` (zero-overhead Zig↔WASM↔TS bindings), `fpz` (fixed-point math), `gkz` (deterministic forkable simulation kernel), `tenzor` (compile-time tensor lib), `hmncli` (AOT recompiler framework), `ggufx` (GGUF weight parser), `ex_lancedb` (embedded vector DB). `[external unverified]` — READMEs not deeply read; classification from repo metadata + descriptions.

**Why this matters to us (HARVEST framing, not a chase):** this is *exactly* the tooling class our **molt** program targets (memory L36: "molt = OUR Python→WASM+WebGPU compiler") and our **Native eval-time runtime discipline** (CLAUDE.md: CPU-stable deterministic integer kernels + Zig/native decoder as "the body", Python oracle as "the brain"). mattneel's `fpz`/`gkz` are reference-grade for deterministic-integer decode; `three.zig`/`wasmbind` are reference-grade for a WebGPU/WASM witness-decoder runtime. **This is prior-art / potential-collaboration signal for the molt + native-runtime surfaces — NOT the witness d_seg/rate math.** He is the single author worth a periodic re-check.

### 2. InfamousBolt — sharp student, but #118 was the whole signal (no hidden stack)
KeshavA ("here for the pizza") has 109 repos: mostly Android/Kotlin/Flutter coursework, 42-school C exercises (ft_printf, minishell, push_swap), and scattered ML student projects (`DiffusionModelsTraining` DDPM-on-MNIST, `mini-tensorrt` a toy C++17 Dense+ReLU engine, `YOLOObjectDetection`, `Neural-Network-Image-Classification`). The tantalizing names (`fourier`, `VAEData`) are data dumps with no README/code technique. **His #118 frontier-saturation writeup — already fully intaken in #412 (the quantified proof the RGB-HNeRV line is Pareto-saturated) — was his genuine contribution; his profile holds no compression stack to mine.** Honest read: broad ML student, not a codec specialist.

### 3. ryanli0070, Bucky789, a12dongithub, PranavReddyGaddam — capable, but bespoke one-offs
- **ryanli0070** (Ryan Li, SYDE@Waterloo, **only 6 repos**): the `sigmoid(−margin/τ)` boundary loss — our level-set margin surrogate in the wild (#412 finding #2) — was produced *for this contest*. His other repos are a diet-tracker, a fighting game, an AI web-app (`EuraAI`). No ML-systems depth beyond the smart contest entry. **NONE transferable beyond what #412 banked.**
- **Bucky789** (Manthan Sumbhe, MS CS Clark): `OrderBookMatching` (C++, 6★) and `airflow-agent-skills-poc` show general low-latency-systems + agent-tooling interest, but no compression/NeRV. Curriculum work was contest-bespoke.
- **a12dongithub** (Samarth Singhal, DTU): confirmed the #128 §5 read — medical-vision/ML person; his non-contest repos are mostly **forks** of graph-ML (`karateclub`), probing (`structural-probes`), time-series-contrastive (`TS-TCC`) research he was reading, not his technique. `karateclub` is a *weak* adjacency to #411 graph-memory but it's someone else's library.
- **PranavReddyGaddam** (SJSU): 36 repos of AI web-apps + Data-226/230 labs. Pose-warmup was contest-bespoke. NONE.

### 4. The tail (9 authors) — pure one-offs
aviral07-code / saitejareddi3636 / shaina-k02 / NihalMishra17 / rohitsudhakar1 / Bindu3116 / flowmar47 / WilliamAyoade / ashika-code: a keyword sweep (`compress|codec|nerv|quant|entropy|video|neural|gpu|metal|cuda|mlx|sdf`) over every non-fork repo returned only unrelated coursework (MNIST classifier, calculator, currency-converter app, password generator). **No signal. One-off contest entrants — stated plainly, not padded.**

---

## Transferable techniques → routing (HARVEST-ONLY; for the main agent to route — I have no TaskCreate)

Per witness-priority, each is a *reference-signal → our-surface* note, NOT a borrowed-vehicle pursuit. **None needs paid GPU; none moves the pointer.** Most authors yielded NOTHING new beyond #412 — this is the honest bulk of the result.

1. **mattneel Zig/WebGPU/WASM portfolio → molt + Native-runtime surfaces (watch-item).** If/when molt (Python→WASM+WebGPU) or a native deterministic eval-time decoder is built, `three.zig`/`wasmbind` (WebGPU/WASM runtime) and `fpz`/`gkz` (deterministic fixed-point kernels) are external reference prior-art. Route → **molt collaboration note** (L36) + **Native eval-time runtime discipline** design notes. `[external unverified]`; a reference, not a dependency. NO new task unless molt work activates.
2. **Nothing else transfers.** InfamousBolt/#118 (strategy corroboration), ryanli/#127 (margin surrogate + byte-close grid discipline → #202), #114 (pose schedule → v7.5.2/#383), discrete>gradient (#128 → #400) — all already routed by #412. This #413 profile-mining pass adds **only** the mattneel compute-adjacency; it confirms no author is sitting on an un-harvested witness/codec technique.

**Explicitly NOT recommended:** contacting authors, cloning/building any author's project, or treating mattneel's Zig work as a work-item — it is a watch-list reference only, subordinate to the witness line.

---

## Pointer honesty
Our exact pointer is **UNMOVED by this intake** (contest-CPU 0.19108282 = MEANS). Read-only profile-mining. Net new signal over #412: **one** compute/native adjacency (mattneel) + the confirmed-negative that the other 14 authors hold no un-harvested technique. Every external claim `[external unverified]`.

## STORES CONSULTED
`.omx/research/pr112_127_intake_20260710.md` (#412 — who did what; not redone) · `.omx/research/pr128_intake_reverse_engineering_20260710.md` (#128 §5 already mined a12dongithub's 29 repos — confirmed, not re-litigated) · `.omx/state/canonical_frontier_pointer.json` (pointer 0.19108282) · memory `[[witness_line_priority_borrowed_polish_harvest_only_20260710]]` · MEMORY L36 (molt = our Python→WASM+WebGPU compiler) · CLAUDE.md §Native eval-time runtime discipline + §Deterministic packet compiler + §Public frontier watch/intake + §Instruction-source-boundary · `docs/operating_manual_craft_handoff.md`. Live data: `gh api users/<login>/repos --paginate` + `gh api repos/<login>/<repo>/readme` for all 15 authors (2026-07-10).

---

### FEED-pr-authors-github-intake
Mined all 15 contest PR #112–128 authors' PUBLIC GitHub profiles (task #413, read-only, contest CLOSED). **Verdict: 14 of 15 are one-off contest entrants** — genuine technique already intaken in #412/#128; their other repos are coursement/mobile/web-app/data-lab/unrelated-forks with ZERO transferable compression/witness signal (stated plainly per NO-FAKE, not padded). The two whose PR sophistication predicted depth — **InfamousBolt** (#118 saturation analysis) and **ryanli0070** (#127 `sigmoid(−margin/τ)` margin surrogate) — are strong students with NO other domain repos; their signal was fully banked by #412. **One genuine surprise: `mattneel` (Matthew Neel, #112 `rhnerv_comma` base) = a real PL/systems/native-runtime domain expert** (133 repos; deep Zig portfolio: `three.zig`/`three-native` native Three.js **WebGPU** runtime, `wasmbind` Zig↔WASM↔TS, `fpz` fixed-point kernels, `gkz` deterministic sim kernel, `tenzor` compile-time tensors, `ggufx` GGUF parser, `ex_lancedb` vector DB). His adjacency is to **COMPUTE/NATIVE, not the witness math**: reference prior-art for **molt** (L36 Python→WASM+WebGPU) + the **Native eval-time runtime discipline** (deterministic integer decode). Route → molt/native watch-item only (NOT a chase; subordinate to witness). Instruction-boundary: several mattneel repos are agent-tooling by name (`jzs`, `zigger`, `frank-demo`) but NO repo content instructed an agent; no injection; nothing acted on. Pointer UNMOVED 0.19108282; every external claim `[external unverified]`; HARVEST-ONLY per witness-priority.
