# ddm_tr2 — operator drop arXiv 1609.04495: TROT (Tsallis Regularized Optimal Transport + Ecological Inference) crosswalk

**Drop (2026-08-08):** https://arxiv.org/abs/1609.04495 — Muzellec, Nock, Patrini, NIELSEN,
"Tsallis Regularized Optimal Transport and Ecological Inference" (2016). Tsallis q-entropy
regularized OT interpolating exact Monge-Kantorovich ↔ Sinkhorn/KL (q=1), covering Pearson/
Neyman/Hellinger divergences; metric properties generalized; convergent Sinkhorn-like solvers;
ecological-inference application = reconstruct a JOINT distribution from MARGINALS + side
information (validated on 2012 election data). Rigor-triage is about APPLICABILITY (math is
settled). Deep-read the paper (fetch the PDF/full text), then crosswalk vs live surfaces.

**Recall-first — the banked OT/Nielsen corpus (cite, don't re-derive; NEW rows only):**
- #288 damped-Newton semi-discrete OT head-offset (memory-held) · #616 Brenier polar
  factorization crosswalk · #617 discrete-OT/Bellman-Ford leg · #539 power-diagram/Laguerre
  witness parametrization (semi-discrete OT ≡ power diagrams; Aurenhammer LP comparator BUILT
  in #583) · #550 Nielsen info-geometry corpus dig · #504 Bregman framework application ·
  m65 dual-metric doctrine (Euclid-vs-Fisher sign-flip; never one metric alone) · rd1
  λ-continuation R(D) frontier · ms2r tolerance-capped solve · cosine≠optimal / Fisher=DERIVED
  memory.

**Seeded crosswalk surfaces (adjudicate these AND sweep beyond):**
1. **q-ladder ≅ our regularization/tolerance ladders.** TROT's q interpolates exact↔entropic
   OT. Our rd1 λ-continuation and ms2r tolerance homotopy ladder the same species of knob.
   Question: does the Tsallis family give a PRINCIPLED parametrization (with convergence
   guarantees + closed-form q-exponential projections) for the solve-tolerance ladder where
   we currently grid λ ad hoc? Consumer: rd1 frontier re-parametrization · ms2r rerun design.
2. **Sparse transport plans (q<1) as a CODING property.** Entropic OT (q=1) yields dense
   plans; q<1 yields sparser, heavier-tailed plans. Sparse plan = sparse assignment = cheaper
   to CODE. Crosswalk vs: cr1's edge-graph conditional carrier (the −110,538 B / −19.22% WIN,
   0.917 B/flip) — can a TROT plan at tuned q serve as the conditional PRIOR/model for
   flip-pair assignment coding, beating the current context model? Consumer: #984 rate axis ·
   cr1 successor race (real payloads only, same-coder discipline per #940 races-not-reputation).
3. **Ecological inference = joint-from-marginals = the m91 per-EDGE object.** The seg gap is
   ONE graph with one hub (Road 87.8% of flips; Road↔Lane = 49.2%); per-class marginals are
   cheap, the EDGE joint is the expensive coupling. TROT reconstructs joints from marginals +
   side info (our side info: g4 spatial stationarity maps, margin fields, class adjacency).
   Question: can the correction/flip JOINT be carried as {cheap marginals + tiny side-info
   + TROT solve at decode} with the solve FREE in inflate.py (rule-118 generic algorithm),
   paying only marginals+side-info bytes? This is a describe-line rate lever shape. Consumer:
   #984 composition · per-edge carrier design (m91 doctrine: decompose per EDGE, never class).
4. **Divergence-family selection for the quotient solve metric.** TROT spans Pearson/Neyman/
   Hellinger — vs our margin-Fisher/rank-4 metric custody (ms3/ms4) and the m65 dual-metric
   law. Is there a measured reason to pick a non-KL member for the seg quotient solve, or is
   this LESSON-ONLY given Fisher is DERIVED from the frozen scorer? Honesty: default skepticism
   — our metric is derived, not chosen; adjudicate whether TROT adds anything beyond naming.
5. **Sinkhorn-family solver reuse for assignment steps** (ms5/ms6 actuator→bucket assignment,
   menu ordering) — likely ALREADY-EMBODIED or N-A (our assignments are measured/causal, not
   inferred); adjudicate honestly.

**Deliverable:** ranked crosswalk table {ADOPT / ADOPT-CLASS / LESSON-ONLY / ALREADY-EMBODIED /
N-A} with named consumers per row, honesty labels (MEASURED/DERIVED/CONJECTURE), and for any
ADOPT: the smallest $0 falsifiable probe PRE-REGISTERED (e.g. seed-3: TROT-decode joint vs
cr1's measured 0.917 B/flip on the SAME payload, same-coder discipline). Full research
authority (online + OSS — check POT/ot.unbalanced/Tsallis implementations) per standing arm
contract + the #984 composition-agent authority clause (adapt-or-DERIVE-ORIGINAL variants
from full-stack deep math). De-dupe against the banked corpus above — cite, never re-derive.

## OPTIMAL FORM

Crosswalk/adjudication arm at reference form — the paper read in FULL (not abstract-only),
probes pre-registered but NOT executed in-arm unless $0 and <30 min CPU. Any probe executed
must run on REAL payloads (cr1's committed flip streams / real n600 marginals), never
synthetic fixtures. No mechanism reduction: a TROT prior raced against cr1 must use cr1's
exact coder harness and payload (provenance pins: cr1 findings + CR1_ROWS.jsonl in
.omx/research/ddm_cr1_20260808/).

**Boundaries:** CPU-only, NO Metal, NO scorer slot. Findings:
`.omx/research/ddm_tr2_20260808/TR2_CROSSWALK.md` + typed JSONL rows.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer. If serializer hits sandbox git-perms, write artifacts + say so.
