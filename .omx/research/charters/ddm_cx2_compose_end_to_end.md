# ddm_cx2 — Dig deeper, then COMPOSE ALL END-TO-END into one archive that beats 0.1721417

**Owner:** codex SOL-ULTRA arm · **Base:** PR130 CPR1 · full research authority · scorer-gated
**Mandate (operator 2026-08-09):** "Have a sol ultra agent dig deeper and compose all end to end."

Two halves, and the second is the deliverable. **Dig deeper** — find mechanism nobody has found on
this base. **Then compose ALL of it end-to-end** — one receiver-closed archive, byte-closed, with an
honest score. Not a survey. Not a ranked table. **One archive.**

## THE BAR AND THE EXACT ARITHMETIC

```
S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489
```

**BASE — PR130 CPR1, reproduced BYTE-IDENTICAL here:**
`S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive **191,052 B**,
sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
= seg **0.028609** (16.62%) + pose **0.014025** (8.15%) + rate **0.127214** (73.90%).

**Archive anatomy (MEASURED, exact leave-one-out marginals — the gap is −20 B, so they are ADDITIVE
and budgetable):**

| section | bytes | share | marginal S |
|---|---:|---:|---:|
| HPAC tokens | 116,980 | 61.23% | 0.0778922 |
| semantic | 36,580 | 19.15% | 0.0243571 |
| pose carrier | 23,384 | 12.24% | 0.0155704 |
| hpac weights | 15,092 | 7.90% | 0.0100491 |
| ZIP overhead | 104 | — | — |

Structure: `archive.zip = ZIP(100) + member "p"(190,952)`;
`p = [u32 models_bytes][LZMA(models_raw) 73,968][HPAC tokens 116,980]`;
`models_raw = 83,493 = 8 + semantic 40,252 + pose 23,054 + hpac 20,179`.
Joint LZMA of the three model sections is **224 B WORSE** than three separate streams — a free win
already taken.

**Sub-0.15 by rate alone needs −33,252 B (−17.4%). Beating the bar at all needs any negative ΔS.**

## WHAT IS ALREADY MEASURED — compose from these, do not re-run them

**WINS, banked (arithmetic only; NOT yet one built archive):**
- split-stream brotli pack: **−903 B**, parse-back exact.
- ANS-for-range on the token stream: **−2,120 B** at n600. ⚠ This is a LENGTH, not a payload —
  the encoder computed `len(...)` and discarded the words. The re-encode retaining words is OWED.
- Composed arithmetic to date: 191,052 → **188,029 B**, derived S **0.170128405876608123**.
  That is **−0.002013 S**. Sub-0.15 needs −0.0221. **We are 9% of the way, on arithmetic, unbuilt.**

**LANDED AFTER THIS CHARTER WAS WRITTEN — `ddm_sd1_semantic` (600af8ef7d), the first semantic win:**
PR130's uniform q4 is optimal among UNIFORM post-hoc q3/q4/q5, but NOT within mixed-bit.

| allocation | archive B | d_seg n600 | ΔS_sem |
|---|---:|---:|---:|
| uniform q4 (shipped) | 191,052 | 0.000286161635 | 0 |
| uniform q3 | 184,828 | 0.016552073161 | +1.622446846491 |
| uniform q5 | 202,324 | 0.001315884060 | +0.110477804687 |
| **4 tensors q3 + 12 q4** | **190,204** | 0.000287568834 | **−0.000423928449** |

The four are `frame_embed.weight` + `blocks.{1,2,3}.film.weight`. Real counted archive at
`/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip`,
sha256 `010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67`. Independent double-build
equality, unchanged carrier/HPAC/tokens, exact parse-back of all 38 tensors, all 32 single-tensor
cells on seeded stratified n120.

**TWO BLOCKERS you must respect, not route around:**
1. Its `SD1M` format is NOT public-receiver-readable — current public decode is int4-only. Composing
   it requires a counted semantic-allocation schema in the receiver plus a proven legacy-q4 identity.
   The allocation header costs 14 B; that is in the budget, not free.
2. **Pose was NOT measured.** sd1 explicitly closed "assume pose invariance" as a dead end, because
   PoseNet directly consumes the changed semantic frame. Sparse Seg changes do not prove continuous
   PoseNet stability. Any composition carrying this MUST measure pose, or carry it as unpriced.

Also from sd1, and binding on your composition: **summing marginal tensor deltas is CLOSED** — the
strongest pair carried a measured −0.0000731437 interaction cross-term. Joint replay was essential.
And analytical raw-parameter-byte estimates are CLOSED: actual archive deltas were −6,224 and
+11,272 B against them.

**UPDATED composed arithmetic** (still arithmetic, still unbuilt): 191,052 − 903 − 2,120 − 848 =
**187,181 B**, ΔS ≈ **−0.002437**. Sub-0.15 needs −0.0221. Roughly 11% of the way.

**AXES MEASURED SHUT — do not re-open without new mechanism:**
- **Coder axis, ALL FOUR sections** (`SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`): semantic is 1,772 B
  BELOW its memoryless bound under brotli, hpac 1,605 B below, pose 0.28% above its own Huffman
  entropy and a re-code measured +4 B WORSE. Only tokens had slack, and ANS took it. **The
  mechanism: a coder swap wins only where an explicit conditional model exists that the coder codes
  worse than.** Where the coder IS the model, you need a better MODEL.
- **Pose carrier REPRESENTATION** (`ddm_pk2`, cfddfc503a): 135 real receiver-parseback candidates,
  49 scored at seeded stratified n120, unchanged CPR1 best on every row. Exact predictors
  (first/second/AR/spline) all reconstruct and all ENLARGE by 1,804–2,232 B. Low-rank +4,316 B.
  Every dimension drop exceeds break-even. Two corrections it made: coefficients are **600×12 at
  10.98 bits/value** (not 1,200×12 at 6.59), and the basis is **already signed int5** (int8 is only
  the decoded container). One reopening survives with a fire trigger: learned rate-aware gauge + QAT.

**AXES OPEN, with live arms — coordinate, do not duplicate:**
- `ddm_dt1` (LIVE): decode wall-clock. Structure ANSWERED at source
  (`codec_hpac_integer.py:96-124`): the frame loop is SERIAL (frame *f*'s context is frame *f−1*'s
  output), the group loop is SERIAL (masked-context refinement), positions within a group are
  ALREADY vectorized. The per-group cost is a NEURAL FORWARD, identical in both coder arms.
- `ddm_dv1` (QUEUED): the device axis. PR130 hard-requires CUDA (`inflate.py:665`), so its row is
  contest-CUDA and the CPU axis was never bought on these bytes. Bounded above by 0.0153 S
  (the whole pose+seg terms), direction unknown.
- `ddm_cl1_capacity` (fe8fa4f35e): capacity-ladder apparatus BUILT and verified, Metal-blocked
  inside the arm sandbox, MAIN-fire-only. **Codex arms cannot reach Metal — do not charter around it.**
- hb1/hb2 line: PR130 HPAC trained on OUR labels. tk1: the semantic-vs-latent token-family question.

## WHERE THE DIG MUST GO — the mechanism follows from the arithmetic

Rate is **73.90%** of the bar and tokens are **61.23%** of the archive. The coder on that stream is
now at +0.0071% over its own model's cross-entropy. **So every remaining token byte must come from a
better MODEL or a better REPRESENTATION, not a better coder.** That is the deep question, and it is
where the depth mandate points:

- The HPAC prior is autoregressive with a masked-group context. What does it NOT condition on?
  Cross-frame structure beyond one step? Ego-motion? Class structure? Each un-modelled dependency
  is uncoded entropy, and it is measurable as a conditional-entropy delta before anything is built.
- The semantic and hpac sections beat their memoryless bound via brotli's LZ. What model would beat
  brotli's LZ? Per-tensor conditioning, positional context inside a k×k kernel, cross-tensor
  permutation or shared codebook. Race against **35,033 B and 14,962 B**, never against the floors.
- The 224-B joint-vs-split result says the three model sections are near-independent under LZMA.
  Is that also true under a learned model? If not, that is cross-section structure nobody has coded.
- Our own corpus holds months of this. Consult it before deriving from scratch, and say what you
  consulted.

Follow the math wherever it goes, including away from these. They are the arithmetic's suggestions,
not a fence.

## THE COMPOSITION — this is the deliverable

Build **ONE archive**. Compose every measured win plus whatever the dig produces, in a dependency-
correct order, and take it all the way:

1. Compose → 2. byte-close through the REAL receiver → 3. exact parse-back proving every section is
consumed → 4. decode within budget → 5. score it → 6. report S against 0.172141297491896447.

Composition is not addition. The marginals above are additive at −20 B on the CURRENT sections; a
changed section can break that. **Measure the composed row, do not sum the parts.** If two wins
interact, say so with the measured interaction.

Report the composed S honestly at whatever axis you can reach. A local run is
`[macOS-CPU advisory]`, `score_claim=false` — it is NOT comparable to the contest-CUDA 0.172141 row
(different hardware AND different GT decode; that confound is exactly what dv1 is decomposing). So
the composed number's honest form is a **byte-closed archive plus a delta decomposition per axis**,
with the rate delta EXACT (bytes are bytes) and the distortion deltas axis-labelled.

## OPTIMAL FORM

Reference form: a full-stack composed codec candidate, byte-closed through the real receiver, with
per-stage measured contributions and a named interaction table. Declared reductions: SCOPE only —
n≥120 seeded STRATIFIED-RANDOM for sweeps (NEVER a prefix; pose prefix bias is 2.5–4.2× harder, so a
prefix pose negative is exactly the false-negative shape, m96/`ddm_na2`); the composed winner
re-measured at full n600. MECHANISM reductions are TOY-BRACKET and cannot carry a family verdict —
in particular, an arithmetic sum of banked deltas is NOT a composed row, and a length without a
retained payload is NOT an archive.

Provenance pins (verify each; a pin that does not reproduce is a STOP):
- archive sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B, at
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
- reproduction `12031094d9` · coder-axis closure `f0a7ebf750` + `bf1830d50f` · pk2 `cfddfc503a` +
  receipt binding `c21d39b48d` · receiver `5de03569ad` (`src/tac/pr130_runtime/fx1_runtime_tree/receiver.py`)
- constriction **0.5.0** pinned — record the version you run
- `upstream/` is IMMUTABLE. The intake clone at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY: never edit, never
  `git add` inside. Copy out to work. PR130 code is off-the-shelf-authorized for OUR line; keep the
  borrowed_substrate_accounting and the attribution honest.

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`.
- Metal is UNREACHABLE from the arm sandbox. If a rung needs it, build the apparatus, emit a
  MAIN-fire order with its trigger conditions, and move on. Do not substitute CPU silently.
- No Modal dispatch. It is SINGLE-FLIGHT under a ≤$20 envelope and needs operator GO.
- Every number carries its axis. `score_claim=false` on everything local.
- RESUMABLE from disk with per-stage checkpoints. This is a long arm; a crash must not cost the run.

## DELIVERABLE

The composed archive: path, sha256, exact byte count, per-section byte table, parse-back proof, and
the measured score with its axis. The per-stage contribution table with measured interactions where
they exist. The dig's findings, each labelled MEASURED / DERIVED / CONJECTURE. And an explicit list
of what you did NOT run and why.

If the composition does not beat the bar, say so in the first line and give the honest residual: how
many bytes short, which section holds them, and the cheapest named next measurement. A composed row
that loses, measured and byte-closed, is worth more than a ranked table that wins on arithmetic.
