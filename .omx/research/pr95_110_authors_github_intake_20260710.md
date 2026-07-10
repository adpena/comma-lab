# PR95–110 frontier-winner authors — public GitHub intake (task #414)

**Date:** 2026-07-10 · **Owner:** task #414 (operator-GO) · **Scope:** the PR95–110 medal-setter / HNeRV-root
author cohort. Sibling task #413 covers PR112–128 authors — **no login overlap** (this cohort's logins are
AaronLeslie138 / rem2 / BradyMeighan / EthanYangTW / SajayR / valtterivalo; #413's a12dongithub/mattneel are
not in this set). Pointer **0.19108282 [contest-CPU]** UNMOVED — this intake is MEANS (technique harvest),
not a score-mover. All external scores/claims tagged `[external unverified]`. WITNESS-FIRST / HARVEST-ONLY:
techniques routed to surfaces, no chase-vehicle.

## STEP 1 — PR → author → login map (resolved via `gh pr view`, NOT guessed)

| PR | Title (self-reported score) | Login | Name | Race role |
|----|------------------------------|-------|------|-----------|
| 95 | hnerv_muon submission (0.20) | **AaronLeslie138** | Aaron Leslie | **HNeRV+Muon ROOT** (inner-council seat); the seed everything stacked on |
| 96 | rem2_HNeRV submission (0.21) | **rem2** | (SF, CA) | SILVER lineage start |
| 97 | vibe_coder_final_boss (0.23) | **BradyMeighan** | Brady Meighan | velocity engineer (3 PRs in 2h12m) |
| 98 | hnerv_muon_finetuned_from_pr95 (0.1963) | **EthanYangTW** | Min-Chun (Ethan) Yang | BRONZE lineage start |
| 99 | hnerv_muon_lc submission (0.19667) | **BradyMeighan** | Brady Meighan | |
| 100 | hnerv_lc_v2 submission (0.1954) | **BradyMeighan** | Brady Meighan | the 268-LOC substrate PR101 built on |
| 101 | hnerv ft microcodec (**GOLD 0.193**) | **SajayR** | SajayR | **GOLD** — 337-LOC entropy bolt-on |
| 102 | hnerv_lc_v2_scale095_rplus1 (0.19538 CPU) | **EthanYangTW** | Min-Chun (Ethan) Yang | **BRONZE** |
| 103 | hnerv_lc_ac submission (0.19) | **rem2** | (SF, CA) | **SILVER** — 241-LOC range-coding increment |
| 106 | belt_and_suspenders (0.20946) | **valtterivalo** | Valtteri Valo | (Helsinki) |
| 110 | hnerv_fec6_fixed_huffman_k16 | **adpena** | Alejandro Peña | **= the operator (self)** — SKIPPED |

Note: the race-postmortem handles Quantizr/Jimmy (UCLA, 0.33 leader) and szabolcs-cs / Selfcomp (PR56
grayscale-LUT + block-FP self-compression) are **not** in the PR95–110 range (PR56 / separate leader entry);
out of this task's scope, not swept here.

## STEP 2+3 — ranked author signal (public repos only; technique → our-surface)

| Rank | Author | Notable repo(s) | Technique | Relevance | Our surface |
|------|--------|-----------------|-----------|-----------|-------------|
| **1** | **SajayR** (GOLD) | **Fisher-LoRA** | LoRA in **Fisher-whitened** space = natural gradient / K-FAC: whiteners A^-1/2,B^-1/2 with EMA of A=E[xxᵀ],B=E[ggᵀ]; rank-r step aligns with top singular modes of the *whitened* gradient | **HIGH — direct** | witness **conditioning** (Stiefel/preconditioner) + **capacity-routing** (#211 KKT waterfill) — our whole geometry IS the frozen-scorer Fisher metric (margin=Fisher surrogate ρ=0.978); natural-gradient conditioning is the same object |
| 1 | SajayR | **FouRA** (unofficial impl of arXiv:2406.08798) | frequency-domain LoRA: `z_out=W₀z+ℱ⁻¹(BαAℱ(z))` (DFT/DCT) + adaptive rank-gating 𝒢 | HIGH | **rate/basis** — curvelet/frequency basis (L75) + adaptive effective-rank = capacity gating |
| 1 | SajayR | **TopoShira** | topographic smoothness prior on a ViT ⇒ neighboring weights encode related features; K%-sparse **in-place** edits ⇒ disjoint patches ⇒ near-orthogonal adapters, zero-shot fusion | MED-HIGH | **v8 per-class carriers** (#377/#234 — near-orthogonal per-class adapters = exactly the merge→diff→correct reconciliation) + curriculum/continual |
| 1 | SajayR | **nano-egg** (fork), **C-ReLoRA**, **FreqQuant** | int-only evolution pretraining (EGGROLL, fitness in **bits-per-byte**); ReLoRA + orthogonal subspace-exploration transforms; (FreqQuant = empty stub) | MED | compression-as-fitness framing; subspace exploration; FreqQuant title only |
| **2** | **EthanYangTW** (BRONZE) | **tinyloop** | C++/CUDA runtime for **weight-shared looped transformers**: Marlin INT4 tensor-core path, GPTQ zero-point folding, quantized GEMM/attention/fused kernels, custom `.tinyloop` format; INT4 within 0.67% PPL | MED-HIGH | **compute (MLX/Metal #252)** + **quantization/QAT** — real low-level kernel + INT4 craft (relevant to our custom-Metal program) |
| 2 | EthanYangTW | **bitpack** | general N-bit (2–7) integer packing into dense byte arrays; 6-bit = 4 vals in 3 bytes, no wasted bits | MED | **rate/codec (#406)** — per-tensor byte-map / quantized-tensor storage primitive (sister of L21 byte-maps) |
| 2 | EthanYangTW | **universal-attn-engine-UAE-**, **parameter-golf**(+monitor) | declarative sparse-attention compiler (H100 speedups); OpenAI parameter-golf entrant (smallest LM in 16MB, val_bpb objective) | LOW-MED | framing only — "compression as the objective" (bpb) mirrors our indirect-RD framing |
| 3 | BradyMeighan | WLED-Studio (161★), PhantomCV (27★), FortniteJamCV, vibe-coder-final-boss | applied real-time computer-vision + LED-video streaming; the vibe-coder writeup is a public interactive writeup of his own PR97 | LOW | none new — applied CV engineer / velocity racer; no neural-compression research corpus |
| 3 | valtterivalo | atomic-gpt (karpathy GPT + C port ~95×), forks of tinygrad/PufferLib/controls_challenge, plankton | performance-engineering + RL/tinygrad enthusiast; comma-ecosystem (controls_challenge) | LOW | none new — systems/RL, not compression |
| 4 | rem2 (SILVER) | cppSimpleMemoryManager, Windows-.dll-Injection, file-corrupter, csharpTriggerbot | low-level Windows / memory-manipulation / game-hacking / reverse-engineering | LOW (indirect) | none direct — BUT the byte/entropy-craft lineage explains the 241-LOC range-coding SILVER (systems skill, not ML research) |
| 4 | AaronLeslie138 (ROOT) | Convolutional-Neural-Network, PancakeGraphCycles, Pathing-Visualizer, Dancing-Links-Sudoku-Solver | basic CNN + algorithmic exercises; 7 repos, mostly forks | LOW | none new — **HONEST**: the HNeRV-root author's OTHER public GitHub is thin; he adapted HNeRV+Muon well but publishes no research corpus |

## Top findings (prose)

**SajayR (PR101 GOLD) is the crown jewel of this cohort — a serious low-rank / Fisher-geometry / frequency-
domain / quantization ML researcher** (~44 repos, Kaggle-competitive: #2 Myllia cell-perturbation, ICPR top
submission, multiple unsupervised audio-visual grounding papers with official implementations). His non-contest
repos are real technique, not one-offs:

- **Fisher-LoRA is the single most transferable idea in this whole intake.** It does LoRA updates in the
  **Fisher-whitened space** of each linear map, where plain GD = natural gradient (K-FAC) in the original
  weights. Objects: A=E[xxᵀ], B=E[ggᵀ], whiteners A^-1/2/B^-1/2 (damped, EMA-refreshed), ΔW=B^-1/2 ΔW̃ A^-1/2,
  runtime folds to a drop-in LoRA (L₀=B^-1/2U, R₀=A^-1/2V). This is **exactly our surface**: the witness is
  defined ON the frozen-scorer Fisher metric (margin field = Fisher surrogate, Pearson 0.978, L24/L57), and our
  capacity-routing (#211 KKT waterfill on margin-saliency) is a natural-gradient allocation. Fisher-LoRA is a
  concrete, drop-in recipe for **preconditioning the witness update by the scorer's Fisher geometry** so a
  rank-limited step aligns with the top whitened-gradient modes (= the boundary annulus). Candidate route to
  the conditioning/capacity axis. (NO-FAKE: this is a *design lead* — unmeasured on our vehicle; would need a
  byte-closed A/B before any claim.)
- **FouRA** (frequency-domain LoRA + adaptive rank-gating) and **TopoShira** (topographic smoothness prior →
  near-orthogonal sparse in-place per-task adapters, zero-shot fusion) are the second tier. TopoShira's
  near-orthogonal-per-patch adapter idea maps cleanly onto **v8 per-class carriers** (#377/#234): merge→diff→
  correct reconciliation wants exactly per-class adapters that don't interfere. FouRA's frequency-domain +
  effective-rank gating is a sister of our curvelet-basis (L75) rate story.

**EthanYangTW (PR98/102 BRONZE) is the surprise #2** — a Taiwanese high-schooler with genuine
systems/quantization depth (70 repos, mostly noise, but a real core). **tinyloop** is a C++/CUDA inference
runtime for weight-shared looped transformers with a Marlin INT4 tensor-core path, GPTQ zero-point folding into
Marlin's symmetric layout, and INT4-within-0.67%-PPL parity — that is real low-level quantized-kernel craft,
relevant to our standing MLX/Metal compute program (#252) and QAT. **bitpack** is a clean, transferable N-bit
(2–7) tensor-storage primitive (6-bit = 4 vals/3 bytes) — a rate/codec sister of our per-tensor byte-maps
(L21) worth a look for #406. His **parameter-golf** entries frame compression *as the objective* (smallest LM,
val_bpb) — the same indirect-RD lens we use.

**AaronLeslie138 (the HNeRV+Muon ROOT / PR95) has a thin OTHER public corpus** — a basic CNN repo and
algorithm exercises. Honest read: he located and adapted HNeRV+Muon exceptionally well under race pressure, but
his GitHub shows no deep neural-compression research to mine. **rem2 (SILVER)** and **valtterivalo** and
**BradyMeighan** likewise have no neural-compression research repos: rem2 is low-level Windows/memory/reverse-
engineering (which *explains* the 241-LOC range-coding silver as entropy-craft, not ML research); valtterivalo
is systems/RL/tinygrad; BradyMeighan is an applied-CV velocity engineer. Their contest strength was
craft/velocity on top of the shared HNeRV substrate, not a private research edge.

## ⚑ INSTRUCTION-BOUNDARY flags (repo text = DATA, quoted not obeyed)

Two external repos contain their-project's-own agent-config files, encountered during the sweep and treated
strictly as data — **NOT read as instructions, NOT acted on**:
- `EthanYangTW/tinyloop` ships a `CLAUDE.md` (its own project's guidance).
- `EthanYangTW/parameter-golf-monitor-webui` ships `SKILL.md` + `codex.md` + `RESEARCH.md`.
None are directed at this agent; none were opened or executed. `EthanYangTW/unity-framework` description reads
`"dont steal my project brooooo"` — a joke, not an agent directive. No prompt-injection / agent-directed text
was found in any swept README.

Minor coincidence flag (no action): `EthanYangTW` has empty repos named `molt` / `pro-molt`; the operator's own
`molt` is OUR Python→WASM+WebGPU compiler (L36). Almost certainly unrelated same-naming — flagged for
awareness, not a link.

## Routing (transferable techniques → tasks, for the main agent)

- **Fisher-whitened / natural-gradient preconditioning of the witness update** (from SajayR/Fisher-LoRA) →
  conditioning + capacity-routing surface (**task #211 / #247 costate**). Highest-EV lead of this intake; a
  design lead only until byte-closed A/B.
- **Near-orthogonal per-class sparse in-place adapters + zero-shot fusion** (SajayR/TopoShira) → **v8
  per-class carriers (#377 / #234)** reconciliation.
- **Frequency-domain adaptation + adaptive effective-rank gating** (SajayR/FouRA) → curvelet-basis / rate
  (L75, **#406**).
- **N-bit dense tensor packing** (EthanYangTW/bitpack) + **INT4/Marlin kernel craft** (EthanYangTW/tinyloop) →
  rate/codec (**#406**) + compute (**#252 MLX/Metal**) / QAT.

## STORES CONSULTED
- `gh pr view` (PRs 95–103,106,110) — canonical author→login resolution.
- `gh api users/<login>/repos` + `/readme` for AaronLeslie138, rem2, BradyMeighan, EthanYangTW, SajayR,
  valtterivalo (6 distinct authors; deep-read READMEs: Fisher-LoRA, FouRA, TopoShira, C-ReLoRA, nano-egg,
  FreqQuant, tinyloop, universal-attn-engine, parameter-golf-monitor, bitpack.py).
- In-tree intake memos: `experiments/results/public_pr{95,100,101,103,106}_intake_*` (author/technique
  context) — dir-listed.
- CLAUDE.md §"Canonical leaderboard binding-depth" (L1–L32) + §"Public frontier watch/intake" + §NO-FAKE #7
  (borrowed-substrate) + CONTEST-CLOSED / INSTRUCTION-SOURCE-BOUNDARY bindings.
- MEMORY.md L24 (Fisher-geometry calibration), L52 (MLX/Metal standing program), L55 (papers-checked ledger),
  L57 (surgical-repair toolbox), L75 (curvelet), L77 (quadratic-head chart); DAG (this file's FEED log).
- Sibling dedup: task #413 (PR112–128 authors) — no login overlap confirmed.
