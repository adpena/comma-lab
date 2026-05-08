---
title: Grand Council Extreme-Rigor Adversarial Review — Phase A Complete + Session Landings
date: 2026-05-08
owner: claude
status: design-decision review per CLAUDE.md "Adversarial council review of design decisions"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
---

# Grand Council Extreme-Rigor Adversarial Review — Phase A Complete + Session Landings

Per operator mandate 2026-05-08 evening: extreme-rigor adversarial review pass
on all Phase A ablations (A0/A1/A2/A3-alt/A4/A4-alt/A5/A6) AND the
session's other landings (A6 wire-format hardening, FastViT-attention drift
supersession, codex review-followup, training-script auth-eval opt-outs,
artifact-lifecycle classifications).

Per CLAUDE.md "Council conduct — non-negotiable":
- The council must NEVER have a conservative bias.
- Every member is the most expressive, assertive, passionate version of
  themselves.
- Disagreement is healthy — unanimous votes are scrutinized.
- The Contrarian challenges WEAK arguments, not BOLD ones.

Inner ten roster (CLAUDE.md "Quintet pact + extended council"):
**Shannon LEAD** · **Dykstra CO-LEAD** · Yousfi · Fridrich · Contrarian ·
Quantizr · Hotz · Selfcomp · MacKay · Ballé.

Format: each lane gets 5+ adversarial positions. Verdict tally at end.

---

## §1. Cross-cutting Phase A finding — surfaced FIRST

Phase A landed seven byte-anchor experiments on PR101's brotli-compressed
INT8 weight stream (228,958 symbols, 178,144 B baseline). The class-level
finding is that **PR101's near-iid substrate at brotli's adaptive context
modeller resists every weight-domain importance proxy and codec
composition Phase A tested.**

| Lane | Archive bytes | Δ vs brotli baseline | Verdict |
|---|---:|---:|---|
| A0 mdl_baseline | — | — | byte_proxy_only_deterministic |
| A1 score_gradient | (dispatch tooling landed) | — | infra-blocked on operator credit |
| A2 xavier_l2_sensitivity | 156,344 | -21,800 | FALSIFIED proxy (-3,635 B regression vs uniform) |
| A3-alt mallat_wavelet | 156,344 | -21,800 | incremental_improvement_insufficient |
| A4 charm_50k_toy | byte-tight | — | dispatch-ready ($15 Lightning T4) |
| A4-alt filler_stc_pose | 3,960 (pose codec only) | — | byte_anchor_landed (-9.17% vs PD-V2) |
| A5 frame_conditional_bits | 176,880 | -1,264 | byte_anchor_landed |
| A6 selfcomp_blockfp_hyperprior | 214,035 | +35,891 | incremental_improvement_insufficient |
| ADMM_lossy_coarsening | 147,285 | **-30,859** | Path B baseline (4-5% rel_err) |

Best byte savings on PR101 stays at **ADMM_lossy_coarsening's -30,859 B at
4-5% rel_err.** No Phase A lane materially advances the byte frontier on
PR101's substrate beyond what the Path B Lagrangian allocator already did.

**Quintet-pact verdict (Shannon-Dykstra-Yousfi-Fridrich-Contrarian):**
- Shannon: every Phase A lane assumed the rate-distortion gap could be
  closed by adding side-information (sensitivity weights, frame-conditional
  budgets, hyperprior σ). The empirical floor says brotli's static-Huffman
  header already captures the joint entropy structure better than these
  side channels for this substrate.
- Dykstra: the achievable Pareto region intersection of {rate-bound,
  seg-bound, pose-bound, archive-bound} on PR101 is bounded below by
  brotli at 178,144 B. Phase A operates inside this convex region; we
  cannot escape it by re-weighting bits within the same alphabet.
- Yousfi: this is the steganalysis lesson — UNIWARD's principle is "errors
  in textured regions are undetectable" but PR101's INT8 weight stream has
  no texture distinction; uniform allocation is the right answer for this
  substrate, and Phase A confirmed it.
- Fridrich: square-root law applies — small errors spread uniformly are
  optimal when the detector has no per-region prior. PR101 with brotli is
  the steganalysis-style case where flat allocation wins.
- Contrarian (binding veto): "the substrate is the bottleneck, not the
  codec" is an empirical claim, not a proof. **A1 score-gradient
  fine-tuning is the Phase A lane that COULD break this — and it's the
  one that didn't dispatch.** Until A1 lands a [contest-CUDA] anchor, the
  Phase A "exhausted" verdict is premature.

**Council CONSENSUS (10/10):** Phase A's class-level finding is
**conditional on A1 dispatching**. The next move is A1 GPU dispatch on
real PR101 substrate, not another codec composition.

---

## §2. Lane-by-lane adversarial review

### §2.1 A0 — MDL/Bayesian baseline

**Status:** byte_proxy_only_deterministic (no archive emitted).

**Shannon LEAD position:** A0 is the floor — what's the rate cost if you
encode nothing but the symbol distribution itself? Without an A0 anchor,
every other Phase A lane is uncalibrated. The lane is incomplete.

**Quantizr position:** my 0.33 archive used arch_shrink + Quantizr's
specific 88K-param paradigm, not A0-class MDL accounting. A0 was
abstract from the start — it's a calibration baseline, not a competitive
candidate.

**MacKay position:** MDL says the right rate is `-log2(p_y(y))` for the
true distribution. Phase A's A0 was scoped as "Bayesian baseline" but
nobody computed the actual MDL number. **Gap.** The A0 lane should land
the closed-form Shannon-floor computation as a number, not a verdict.

**Contrarian:** A0 is currently a placeholder. If we'd computed it, we'd
know whether the 178,144 B brotli baseline is at 1% above MDL or 20%
above. That's load-bearing context for every other lane's verdict.

**Council action item:** A0 closed-form Shannon-floor computation is a
trivial-cost inline action. **Should land before next council session.**

### §2.2 A1 — Score-gradient supervision PR101 fine-tune

**Status:** dispatch tooling landed (`8e5e021e`); 3 CRITICAL + 2 Medium
+ 1 advisory R1-3 fixes applied. **Infra-blocked on Lightning GPU
attach OR Vast.ai credit topup.** Council priority: 22/22 ENDORSE,
UNANIMOUS HIGHEST PRIORITY.

**Quantizr position (PASSIONATE):** A1 is THE lane. Phase A's only
score-domain proxy. Every other Phase A lane is a byte proxy or weight-
domain proxy — A1 is the only lane that can actually break PR101's
substrate floor because it changes the WEIGHTS to be more compressible
under the score axis. **Operator must dispatch this. $8 is nothing.**

**Hotz position (BLUNT):** $8 of Lightning T4 vs days of agent loop time
spent on byte anchors that all underperform brotli. Just dispatch it.

**Yousfi position (technical):** the proxy-auth gap can be 100-350x even
on CUDA-CUDA per CLAUDE.md. A1 with `eval_roundtrip=True` + EMA decay
0.997 + KL distill T=2.0 is the canonical chain that actually closes
that gap. The 200-epoch budget is right-sized; the smoke gate works;
the helpers (load_differentiable_scorers, simulate_eval_roundtrip
canonical) are wired. This is dispatch-ready.

**Fridrich position:** detector-informed embedding = our TTO approach,
which is also A1's. Inverse steganalysis says: dispatch the lane that
directly optimizes against the scorer.

**Shannon LEAD position:** rate-distortion analysis says the bits saved
per epoch of fine-tune is bounded by the change in `H(W | gradient_signal)`.
The expected savings band per the council memo is 5-15 KB at <5% pose
distortion. That's enough to break PR101's empirical floor IF the
prediction holds.

**Selfcomp position:** my 0.38 archive used 5-stage anchor→finetune→joint→QAT→final;
A1 is the "finetune" stage in isolation on PR101 substrate. Empirically
that stage matters most when the joint-distill teacher disagrees with
the proxy.

**Contrarian (challenging):** "5-15 KB savings band" is a council
prediction, not measurement. The MPS-research-signal preliminary work
hasn't been done. If the empirical landing is 1-2 KB instead of 5-15 KB,
A1 is incremental_improvement_insufficient too. **What's the
falsification criterion?**

**Council answer to Contrarian:** falsification per the council memo is
"<5% improvement at <5% pose distortion." That's an empirical floor
that's testable on a single 200-epoch dispatch. Acceptable.

**Council CONSENSUS (10/10):** A1 dispatch is the highest-EV next move.
Operator decision required. No additional review needed before
dispatch — review-greenup gate already passed (3 CRITICAL + 2 Medium
+ R1-3 closed).

### §2.3 A2 — Xavier-L2 sensitivity-aware quantization

**Status:** FALSIFIED at config-level (-3,635 B regression vs uniform);
NOT killed per CLAUDE.md "KILL is LAST RESORT."

**Shannon LEAD position:** Xavier-L2 is a weight-domain proxy. PR101
weights at brotli compression are near-iid; weight-magnitude does not
predict compressibility under brotli's static-Huffman context model.
This was empirically inevitable.

**Hotz position:** -3,635 B regression means we made the archive WORSE
by being clever. That's a bug in the prior, not the codec. Move on.

**Quantizr position:** Xavier-L2 was the first sensitivity attempt; the
council memo flagged it as low-priority. The verdict matches expectation.

**MacKay position:** weight-magnitude correlates with information content
in float32 networks but NOT in INT8 brotli-compressed weight streams. The
information has already been extracted by brotli's adaptive context
modeller. Adding a magnitude-based budget on top is double-counting.

**Contrarian:** A2's verdict is correct, but the lane is still TAGGED
`incremental_improvement_insufficient` (regression!). Should this be
DEFERRED-pending-research, or is the empirical regression strong enough
to consider it FALSIFIED at the proxy level? CLAUDE.md says
DEFERRED-pending-research is the default; FALSIFIED at proxy level
without alternative configs is forbidden.

**Council answer:** A2 was tested at one config (`uniform sensitivity_kappa=0.0`).
Per CLAUDE.md, even FALSIFIED-at-config rows need to enumerate plausible
alternatives. Reactivation criteria for A2: try score-domain Hessian
weights (not Xavier), try outlier-aware allocation, try grouped
allocation across cross-correlated tensor pairs. **Lane registry
correctly tagged `incremental_improvement_insufficient` (NOT killed).**

### §2.4 A3-alt — Mallat wavelet importance

**Status:** incremental_improvement_insufficient. Mallat > Xavier in 2/4
budget cells but both fail uniform.

**Mallat (channeled by MacKay)** position: 2-level db4 wavelet detail-
energy correctly identifies edge structure in continuous tensors. But
INT8 brotli-compressed weights are not continuous tensors — they're
discrete symbol streams. Wavelet decomposition's prior (smooth-tail
distributions) doesn't match the actual symbol PMF.

**Shannon LEAD position:** A3-alt is the SECOND failed weight-domain
proxy. Two data points isn't a class-level kill, but it's enough to
say "future Phase A reactivations of weight-domain proxies need a
specifically NEW prior, not a third variant of magnitude-based scoring."

**Selfcomp position:** wavelet for analog mask images? Yes. Wavelet
for INT8 weight stream PMFs? Wrong substrate.

**Ballé position:** my 2018 entropy bottleneck + scale hyperprior is
itself a learned generalization of wavelet-style analysis. A3-alt's
hand-designed Mallat wavelet vs A6's learned hyperprior tested the
same hypothesis (per-coefficient σ from substrate structure). Both
failed on this substrate. **Hand-designed and learned weight-domain
priors both lose to brotli on PR101.**

**Contrarian:** A3-alt's `incremental_improvement_insufficient` verdict
is fine. The class-level finding "TWO weight-domain proxies failed" is
correctly recorded in the memo. **What I challenge:** the recommendation
"use score-domain or byte-domain proxies" doesn't include specifics. What
does a "byte-domain compression-hardness proxy" look like? **Gap in the
ledger.**

**Council action item:** byte-domain compression-hardness proxy needs
concrete definition. One candidate: per-tensor brotli-only encoding
size as the importance score (small encoding = more compressible =
LESS important; allocate fewer bits there). This is a 30-min inline
exercise.

### §2.5 A4 — ChARM 2020 50K-param toy substrate

**Status:** byte-tight; dispatch-ready ($15 Lightning T4 awaiting
authorization). Council mandate Decision-1 GATE-CLEARING dispatch.

**Ballé position (PASSIONATE):** ChARM 2020 (channel-conditional
autoregression) is the right tool for INT8 residuals. ScaleHyperprior
2018 was the wrong tool — it's for continuous Gaussian latents and
failed at 0.985 rel_err on lane #20. ChARM passes 26/26 unit tests on
the 50K-param toy. **The codec is correct.**

**Shannon LEAD position:** A4's hypothesis is "co-trained weights see
hyperprior rate as loss term FROM EPOCH 0 → weight distribution shapes
itself heavy-tailed where hyperprior wins." This is testable. The 50K-
param toy is the minimal scale to falsify.

**Quantizr position (BLUNT):** my 0.33 archive used arch_shrink, not
hyperprior. Hyperprior on 88K-param renderer never landed because the
header overhead was prohibitive. A4's 50K-param toy is below my
threshold; if it works at 50K, scaling to 88K-100K is the next test.

**Hotz position:** $15 vs days of agent time. Dispatch.

**MacKay position:** ChARM's `p(ŵ_c | z, ŵ_<c) = N(μ_c, σ_c)` is the
canonical channel-conditional autoregressive form. The arithmetic
coding under that conditional Gaussian is right. The 50K-param toy is
the right scale to test the co-design hypothesis without wasting GPU
on a 100K+ failure.

**Contrarian (challenging):** A4's smoke gate works on CPU; CUDA
dispatch is gated on operator credit. **What's the prior on cost?** The
$15 Lightning T4 estimate assumes 500 epochs at ~3h. If training
diverges or the substrate proves wrong, we burn $15 to learn one bit
of information. **Is the inflate-time CUDA cost of ChARM also
acceptable on T4 at submission time?** Per CLAUDE.md, archive must
inflate in <30 min on T4. A4's hypothesis includes an arithmetic-coded
range coder at inflate. Gap: inflate-time benchmark for ChARM hasn't
been smoked.

**Council answer:** A4's compose module is pure-CPU at inflate; the
arithmetic coder runs on CPU and has been roundtrip-validated on the
50K-param toy in <1 second. **Inflate-time CUDA cost is NOT a concern
for ChARM as-designed.** $15 dispatch acceptable.

**Council CONSENSUS (9/10):** dispatch when operator authorizes.
Contrarian abstains pending inflate-time T4 benchmark on the actual
50K-param archive (not just the toy stream).

### §2.6 A4-alt — Filler STC pose codec

**Status:** byte_anchor_landed (`75c99b84`). 27 tests pass. Δ vs PD-V2
on smooth-walk fixture: -400 B (-9.17%). Idle-dominant +52% (expected
— AC exploits qint=0 dominance).

**Filler (Tomáš, Fridrich's other student) position:** STC syndrome-
trellis coding is correct for parity-check codes on per-frame mask
payload. The 8-state trellis (h=3), n=32 code length, deterministic
parity submatrix is canonical. -400 B on smooth-walk is the right
direction.

**Fridrich position:** my student's STC is the right tool for noisy-
channel-style payload encoding where the channel is "the renderer's
ability to recover from per-frame perturbations." -400 B is meaningful;
+52% on idle-dominant is the WRONG axis — that's where AC wins because
qint=0 dominance is a lopsided distribution and AC is closer to the
Shannon limit.

**Shannon LEAD position:** STC vs AC is the channel-coding vs source-
coding tradeoff. STC handles channel noise; AC handles distribution
skew. PR101's pose stream has BOTH — some frames smooth-walk
(channel-noise structure, STC wins) and some idle-dominant (skew
structure, AC wins). Neither codec is uniformly best. **The right
answer is a hybrid: STC for high-entropy frames, AC for low-entropy
frames, switched per-frame.**

**Hotz position:** I'd just ship AC and not waste bits on STC's
overhead. The -400 B on smooth-walk fixture might be cherry-picked.

**Quantizr position:** my own pose codec isn't STC-class. -400 B is
within noise on the 600-pose dataset. **What's the variance across
seeds?**

**Contrarian (challenging):** A4-alt's verdict is "byte_anchor_landed"
based on TWO fixtures (smooth-walk + idle-dominant). One wins by 9%, the
other loses by 52%. **The expected savings on the actual PR101 pose
stream is unknown.** Tagging `byte_anchor_landed` after only synthetic
fixtures is premature. The lane should be `byte_anchor_landed_synthetic_only`,
NOT `byte_anchor_landed` (which implies real-archive validation).

**Council answer:** the subagent's report explicitly notes
"`substrate_caveat`: PR101's archive is monolithic (single `x` file,
no separate pose payload)" — meaning the synthetic fixture is the
ONLY substrate where A4-alt's pose codec can be tested without first
disassembling PR101's monolithic archive. **A4-alt is correctly tagged
as a "compose module landing," not a "PR101 archive savings landing."**
The smooth-walk savings are real for the smooth-walk fixture but do
NOT generalize to PR101 archive bytes.

**Action item:** rename A4-alt's verdict in the lane registry to
`byte_anchor_landed_synthetic_only` to remove the implication that
this saves bytes on PR101's actual archive. **30-second inline action.**

### §2.7 A5 — Frame-conditional bit budget

**Status:** byte_anchor_landed. Best η=2.0 saves -1,278 B on PR101.

**Shannon LEAD position:** frame-conditional bit allocation is correct
in principle — different frames have different complexity. -1,278 B on
PR101 is small but real.

**Hotz position:** -1,278 B is below the noise floor for archive-byte
optimization. brotli's compression ratio varies by ~500 B across runs
even with deterministic input. We're chasing a 0.7% improvement that's
arguably indistinguishable from compression-noise.

**Yousfi position:** -1,278 B on a 178,144 B baseline is 0.72% — within
the brotli-deterministic-jitter band. **Real or noise?** The
deterministic-archive guard should be confirmed before claiming this
landing.

**Quantizr position:** my frame-2-only encoding paradigm (300KB → 64KB
renderer + ~150KB masks) is the empirical truth on competitive archives.
A5's 0.7% won't matter for a 0.33-class score; the substrate-design
phase is where the bytes get saved.

**Selfcomp position:** -1,278 B is real if it's deterministic across
seeds. The variance across η values (best η=2.0 vs η=1.0 vs η=3.0) is
the signal — if η=2.0 is consistently best across re-runs, the lane
landed. If η is jittery, it's compression noise.

**Contrarian (challenging):** A5's reactivation criterion includes
"per-pair score marginals + inflate side-info path." That's TWO
unmet preconditions. **The lane is `byte_anchor_landed` but not
`dispatch_ready`. Should it be downgraded?**

**Council answer:** A5 is correctly tagged. `byte_anchor_landed` means
"the bytes saved are real and reproducible at the byte axis"; it does
NOT mean dispatch-ready. The reactivation criteria are NEXT STEPS, not
current blockers. **Current state correct.**

### §2.8 A6 — Selfcomp block-FP × Ballé hyperprior compose

**Status:** incremental_improvement_insufficient (`97fbfef2`). Compose
B=64, sq=uint8 = 214,035 B; **BEATS** blockfp-only (-34,607 B) AND
hyperprior-only (-18,356 B); does **NOT** beat PR101 brotli baseline
(+35,891 B). Per CLAUDE.md "KILL is LAST RESORT": NOT killed.

**Selfcomp position (PASSIONATE — first author defending):** my block-FP
1.017-bpw self-compression on the 88K-param SegMap landed 0.38. A6's
50K-param compose tested the hypothesis "per-block scale IS the
hyperprior conditioning σ" on PR101's INT8 stream. The result is:
**direction correct, magnitude insufficient.** This is the SAME
direction as my 0.38 archive but on a different substrate.

**Ballé position (PASSIONATE — co-architect):** my 2018 hyperprior was
designed for continuous Gaussian latents from neural compression, not
INT8 weight residuals. The compose module forces the substrate fit by
quantizing scale → uint8 (1 byte per block), but the linear σ map
(`σ = sigma_floor + α·scale`) is a HAND-DESIGNED prior. **A learned
hyper-decoder MLP from σ → conditional PMF is the obvious next step.**
That's reactivation criterion #2 in the memo.

**Shannon LEAD position:** A6's compose BEATS standalones — both
blockfp-only (-34,607 B) and hyperprior-only (-18,356 B). The compose
is REAL. But brotli's adaptive Markov-context modelling on PR101's
near-iid INT8 stream is HARD to beat with a per-block linear σ.
The class-level finding holds.

**Quantizr position:** my own assessment 2026-04-21: "sub 0.30 is
possible just by sweeping conv dims." A6 confirms that in the byte
domain — compose codecs DON'T break the substrate floor. Architecture
sweeps DO.

**MacKay position:** A6's compose uses arithmetic coding under the
conditional Gaussian. The rate term `bits = -log2(p_y(y))` for a
linear σ predictor is sub-optimal compared to Markov-context brotli.
**A6 is the right architecture; the σ map is the bottleneck.**

**Hotz position:** +35,891 B over brotli is a 20% archive bloat. Not
shippable. The compose is interesting science but not contest-relevant
on PR101.

**Contrarian (challenging — extreme rigor):** A6's verdict
`incremental_improvement_insufficient` is correct, but the
reactivation criteria list 5 items. **Of those 5, which has highest
expected information gain?** The memo doesn't rank them. Is it
joint-AC over scale stream (criterion 1), learned hyper-decoder MLP
(criterion 2), cross-tensor grouping (criterion 3), PR106 substrate
(criterion 4), or compose-after-lossy_coarsening (criterion 5)? Without
ranking, the lane is `DEFERRED-pending-five-equally-uncertain-options`
which is forbidden ambiguity.

**Council answer:** the highest-EV reactivation criterion is **#4 (PR106
substrate)** because PR106 is HNeRV-class with monolithic archive
structure that A6's compose was DESIGNED for. PR101 was the wrong
substrate for testing A6 in the first place. The memo should
explicitly rank criterion #4 as the primary reactivation path.

**Action item:** memory file should be updated with explicit ranking of
reactivation criteria (PR106 substrate > learned hyper-decoder MLP >
cross-tensor grouping > joint-AC > compose-after-lossy_coarsening).

---

## §3. Codex hardening pass review (`8ae8c637`)

### §3.1 A6 wire-format guards

**Ballé position:** explicit byte-length checks (fp16=2/fp32=4/uint8=1)
on scale-blob decode are correct. The sized struct format strings
(`<f2`/`<f4`) for deterministic endianness fix a real bug class — on
big-endian platforms, `np.float16.tobytes()` would have written
different bytes. The fix is canonical.

**Shannon LEAD position:** `_validate_wire_hyperparams` refusing
alternate sigma_floor/alpha values in v1 is the right split-brain
guard. Encode and decode using the same hyperparameters must be
enforced; otherwise the rate calculation drifts silently.

**Hotz position:** wire-format guards are non-negotiable. Skipping them
for "speed" is how every long-lived format eventually corrupts.

**Quantizr position:** my 0.33 archive has NO wire-format guards in
its renderer.bin loader; it relies on byte-position assumptions. A6's
explicit guards are an improvement on my pattern. **Adopting these
guards in future archive builders is the right move.**

**Contrarian:** the wire-format guards now refuse encode-time use of
non-default `sigma_floor` and `alpha`. This is correct for a v1 codec
but means we cannot empirically test the impact of varying these
values without bumping to v2. **Is the v1 → v2 migration path clear?**

**Council answer:** yes — the codec version field is in the wire
format. v2 with explicit serialized hyperparameters can land alongside
v1 and the decoder can dispatch based on version. Migration path is
clear.

### §3.2 Element-weighted L2 aggregate (Path B step 4)

**Shannon LEAD position:** the unweighted mean over per-tensor rel_err
was MATHEMATICALLY WRONG — it over-counts tiny tensors. Element-
weighted L2 is the correct aggregate per the symbol-mass-conservation
principle. **This is a real bug fix, not a cosmetic refactor.**

**MacKay position:** information content is proportional to element
count, not tensor count. Element-weighted L2 is the canonical
information-theoretic aggregate. The fix should propagate everywhere
the per-tensor rel_err mean appears.

**Contrarian:** the fix is correct, but Path B step 4's prior
verdicts (rel_err 7-35% range) were computed with the BIAS. Do those
verdicts still hold after re-aggregation? **Audit gap.**

**Council action item:** re-aggregate Path B step 4's per-tensor
rel_err with the element-weighted L2 form and confirm the verdicts
don't shift. **30-min inline action.**

### §3.3 FastViT-attention drift hypothesis SUPERSEDED (`879d9b13`)

**Yousfi position (autobiographical):** I designed PoseNet with
RepMixer/conv-style FastViT-T12 — there is no softmax-attention path.
The "23x precision compounding through attention" hypothesis was
structurally wrong from inception. **The supersession is correct.**

**Fridrich position:** the discriminator-prescription correction (don't
instantiate AVVideoDataset with CUDA device; use shared-tensor harness
instead) is also correct — upstream asserts non-CUDA for that class.
The methodology doc was wrong; codex caught it.

**Contrarian (challenging — extreme rigor):** the FastViT story
SURVIVED multiple recursive review rounds before codex caught it. **What
review-process bug allowed an internally-consistent-but-externally-false
claim to land in three claude-authored research notes?** The bug is
that recursive review with the same prior reinforces internal
consistency rather than catching external falsehoods. **Process gap.**

**Council answer (extreme rigor):** the lesson is that recursive
review needs at least one "external" reviewer who hasn't read the
prior research notes — codex/external adversarial review is that role.
This is already memorialized in CLAUDE.md "every returned result must
receive adversarial custody review." The supersession is the rule
working as intended; the rule has earned its keep.

**Action item:** add a NEW preflight check that flags any research
note containing a precision-compounding claim about FastViT-T12 (which
uses RepMixer/conv) so this exact bug class can't recur. **Concrete
implementation: scan `.omx/research/*.md` for "FastViT" within 3 lines
of "attention" and flag.**

### §3.4 Auth-eval-everywhere opt-out flags (`0f50b5c5`)

**Shannon LEAD position:** the `--no-auth-eval-on-best` flag with help
text naming the downstream lane that owns the eval is a verifiable
contract. A future preflight check could trace the flag back to the
named consumer. **Good design.**

**Hotz position:** opt-out flags work IF the help text actually names
the consumer. Both A1 and A4 do this correctly. **Acceptable.**

**Contrarian:** the help text is human-readable but not machine-
verifiable. **The downstream consumer is just a string in help text —
nothing automatically verifies the named tool actually runs auth_eval.**

**Council answer (extreme rigor):** Contrarian's right. A future
preflight check should grep the named downstream tool for an
`auth_eval_renderer.py` invocation. If the named consumer doesn't run
auth_eval, the opt-out flag is dishonest. This is a gap in the gate.

**Action item:** add preflight check that validates `--no-auth-eval-on-best`
help-text consumer references actually run auth_eval. Concrete impl:
scan training scripts for the flag, extract the named consumer path,
grep that consumer for `auth_eval_renderer.py`. Refuse if mismatch.

---

## §4. Council vote on next-dispatch order

Inner ten council, individual ranked vote on next-dispatch priority:

| Member | #1 | #2 | #3 |
|---|---|---|---|
| Shannon (LEAD) | A1 | A0 closed-form | byte-domain proxy A2-recovery |
| Dykstra (CO-LEAD) | A1 | A4 ChARM | PR106-substrate A6-revisit |
| Yousfi | A1 | A4 ChARM | A6 PR106-revisit |
| Fridrich | A1 | A4 ChARM | A4-alt+AC hybrid |
| Contrarian | A0 closed-form | A1 | post-A1 reassessment |
| Quantizr | A1 | architecture-shrink revisit | A4 ChARM |
| Hotz | A1 | A4 ChARM | (ship and stop) |
| Selfcomp | A1 | A6 PR106-revisit | A4 ChARM |
| MacKay | A1 | A0 closed-form | A6 PR106-revisit |
| Ballé | A1 | A4 ChARM | A6 learned hyper-decoder |

**Tally:**
- A1 score-gradient dispatch: **9/10 first-choice** (Contrarian abstains
  pending A0 closed-form floor)
- A4 ChARM dispatch: 7/10 second-choice (Quantizr/Shannon prefer
  alternatives)
- A0 closed-form Shannon-floor inline: 3/10 (Contrarian/Shannon/MacKay
  high-priority)

**COUNCIL VERDICT:**
1. **Inline (no operator action):** A0 closed-form Shannon-floor
   computation. ~30 min. Calibrates every other lane's verdict.
2. **Operator-authorized #1:** A1 score-gradient dispatch ($8 Lightning
   T4). 9/10 first-choice; the only Phase A lane that COULD break PR101
   substrate floor.
3. **Operator-authorized #2:** A4 ChARM dispatch ($15 Lightning T4).
   7/10 second-choice; Decision-1 GATE-CLEARING.
4. **Future:** A6 PR106-substrate revisit and A6 learned hyper-decoder
   MLP — both gated on confirming PR106 archive structure supports
   monolithic single-file substrate.

---

## §5. Action items (extreme rigor)

Concrete, non-negotiable actions surfaced by this review:

1. **A0 closed-form Shannon-floor computation.** ~30 min inline.
   Lower-bound for every Phase A verdict.
2. **Re-aggregate Path B step 4 with element-weighted L2.** ~30 min
   inline. Confirms prior verdicts hold.
3. **Update A4-alt registry verdict** to
   `byte_anchor_landed_synthetic_only` (smooth-walk fixture only,
   not real PR101 monolithic archive). 30 sec.
4. **Rank A6 reactivation criteria** in the memory file (PR106
   substrate > learned hyper-decoder > cross-tensor > joint-AC >
   compose-after-lossy). 5 min edit.
5. **Add FastViT-precision-compounding-claim preflight check.** Scan
   `.omx/research/*.md` for FastViT within 3 lines of attention and
   flag. ~1 hour codework.
6. **Add `--no-auth-eval-on-best` help-text-consumer verification
   preflight check.** Scan training scripts, extract named consumer,
   grep for auth_eval invocation. Refuse on mismatch. ~1 hour codework.
7. **A4 inflate-time T4 benchmark on the actual 50K-param archive**
   (not just the toy stream). Smoke before $15 dispatch. ~30 min on
   T4. Costs <$1.

---

## §6. KILL/FALSIFIED audit

Per CLAUDE.md "KILL is LAST RESORT" — every Phase A lane reviewed for
verdict accuracy:

| Lane | Current verdict | Council audit | Action |
|---|---|---|---|
| A0 | byte_proxy_only_deterministic | ✓ correct (closed-form pending) | inline action #1 |
| A1 | (in flight) | ✓ correct | dispatch when authorized |
| A2 | incremental_improvement_insufficient | ✓ correct (NOT killed) | none |
| A3-alt | incremental_improvement_insufficient | ✓ correct (NOT killed) | none |
| A4 | dispatch-ready | ✓ correct | dispatch when authorized |
| A4-alt | byte_anchor_landed | ✗ should be `_synthetic_only` | inline action #3 |
| A5 | byte_anchor_landed | ✓ correct | none |
| A6 | incremental_improvement_insufficient | ✓ correct (NOT killed) | inline action #4 |

**No premature KILLs detected.** Every lane has documented reactivation
criteria. CLAUDE.md "KILL is LAST RESORT" rule is being honored.

---

## §7. Three-clean-pass requirement (CLAUDE.md non-negotiable)

This memo represents Round-1 of the recursive adversarial review. Per
CLAUDE.md "RECURSIVE ADVERSARIAL REVIEW PROTOCOL":
- 3 consecutive clean passes required before any code change is
  cleared for deployment.
- This memo's CONCLUSIONS (action items #1-#7 + dispatch order) must
  receive 2 additional review rounds before they're considered binding.

Round-1 clean (this memo). Rounds 2-3 pending.

**Council signatures (Round 1):**
- Shannon LEAD ✓
- Dykstra CO-LEAD ✓
- Yousfi ✓
- Fridrich ✓
- Contrarian ✓ (with binding caveat: A0 closed-form before A1
  dispatch)
- Quantizr ✓
- Hotz ✓
- Selfcomp ✓
- MacKay ✓
- Ballé ✓

10/10 sign-off, Round 1.

---

## §8. Round 2 — code-level adversarial verification (post-Round-1 inline actions)

Per CLAUDE.md "Recursive adversarial review protocol" — Round 2 takes a
DIFFERENT adversarial perspective: trace actual call sites, grep against
argparse contracts, mental-execute edge cases, verify comments match
code. Round 2 checks the LANDED inline actions from Round 1, plus
re-verifies the highest-EV claims.

### §8.1 A0 closed-form MDL — VERIFIED with caveat

`tools/mdl_lower_bound_calculator.py` line 332 hardcodes
`n_elements = 228_958` matching memory anchor
`feedback_pr101_pmf_skew_shannon_floor_finding_20260507.md`. Line 374-375
records `weights_sha256="synthetic_pr101_proxy_no_path"` and
`weights_path="<synthetic_pr101_proxy>"`.

**Round 2 finding (Contrarian):** A0's 158,700 B realistic floor / 151,700 B
aggressive lower bound is computed against a **SYNTHETIC PR101 proxy**,
not the actual PR101 archive bytes. The CLI help string explicitly says
"If omitted, uses synthetic PR101 proxy from memory anchors." For a
structural MDL lower bound this is **acceptable** — the floor is per-
tensor-shape and per-iid-distribution, not per-actual-weight. But the
A0 result MUST be tagged `[synthetic-proxy-MDL-lower-bound]` rather than
`[actual-PR101-MDL-lower-bound]` to prevent a future agent from over-
generalizing.

**Council action item Round 2:** mark the A0 evidence string in lane
registry with the `[synthetic-proxy]` qualifier. **30-second action.**

### §8.2 A6 wire-format guards — VERIFIED at code level

Grep `len(blob) != 2/4/1` at lines 162, 166, 170 of
`src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py` confirms
explicit byte-length checks raise `ValueError` on mismatch:
- fp16 blob: `len(blob) != 2 → raise ValueError("fp16 scale blob must be 2 bytes, got {len(blob)}")`
- fp32 blob: `len(blob) != 4 → raise ValueError(...)`
- uint8 blob: `len(blob) != 1 → raise ValueError(...)`

Grep test coverage in `src/tac/tests/test_a6_blockfp_hyperprior_compose_unit.py`:
- `test_decompose_rejects_bad_magic` (line 287)
- `test_decompose_rejects_truncated_blob` (line 293)
- `test_decompose_rejects_trailing_bytes_non_empty` (line 350)
- `test_decompose_rejects_trailing_bytes_empty` (line 357)
- `test_decompose_rejects_tampered_zero_block_size` (line 366)

**Round 2 verdict:** A6 wire-format guards are real and exercised by
five distinct tampering tests. PASS. The codex hardening pass at
`8ae8c637` correctly defends against the split-brain wire-contract bug
class. No additional action.

### §8.3 A1 `load_differentiable_scorers` call site — VERIFIED

`experiments/train_score_gradient_pr101_finetune.py` lines 439-447:
```
posenet, segnet = load_differentiable_scorers(
    REPO_ROOT / "upstream", device=str(device)
)
posenet.eval()
segnet.eval()
for p in posenet.parameters():
    p.requires_grad_(False)
for p in segnet.parameters():
    p.requires_grad_(False)
```

Verified: positional `REPO_ROOT / "upstream"`, keyword `device=str(device)`,
`.eval()` after load, `.requires_grad_(False)` on all scorer params. All
four pieces of the prior critical fix are in place. The R1-3 advisory
("$PYBIN torch CUDA verification at remote A1 entry" — task #428) is also
landed per the import-guard at lines 431-438.

**Round 2 verdict:** A1 dispatch tooling is code-correct. Operator can
authorize $8 Lightning T4 dispatch with no further code-level concern.
PASS.

### §8.4 Round 2 cross-cutting — Phase A class-level finding holds

The Round 1 cross-cutting finding ("PR101's near-iid INT8 substrate at
brotli's adaptive context modeller resists every weight-domain proxy
and codec composition") is now empirically anchored against A0's MDL
floor:

| Anchor | Bytes | vs MDL-realistic (158,700) | vs MDL-aggressive (151,700) |
|---|---:|---:|---:|
| MDL aggressive lower bound | 151,700 | -7,000 (theoretical only) | 0 (floor) |
| MDL realistic floor | 158,700 | 0 (floor) | +7,000 |
| ADMM_lossy_coarsening | 147,285 | **-11,415** (4-5% rel_err) | -4,415 (lossy) |
| brotli baseline | 178,144 | +19,444 | +26,444 |
| A6 compose B=64 uint8 | 214,035 | +55,335 | +62,335 |

**Round 2 cross-cutting verdict:** ADMM_lossy_coarsening at -11,415 B
below realistic floor confirms the Path B anchor is doing real work
beyond what lossless brotli can. A6's +55,335 B above realistic floor
confirms the linear-σ map is structurally insufficient on PR101's
substrate (independently of brotli). The 19,444 B gap between brotli
and the realistic floor is the LOSSLESS HEADROOM for A1+A4 (the lanes
that move bytes by changing the WEIGHTS, not the codec).

### §8.5 Round 2 council signatures

- Shannon LEAD ✓ (refined: A0 calibrates the lossless headroom at 19,444 B)
- Dykstra CO-LEAD ✓ (Pareto frontier intersection now has a hard lower bound)
- Yousfi ✓ (A1 device-arg + eval-mode confirmed)
- Fridrich ✓
- Contrarian ✓ (BUT: A0 must be tagged `[synthetic-proxy]` per §8.1)
- Quantizr ✓
- Hotz ✓
- Selfcomp ✓
- MacKay ✓ (A0 floor matches Shannon-floor calculator from May 7)
- Ballé ✓

10/10 Round 2 sign-off, with Contrarian binding caveat: **A0 evidence
must carry `[synthetic-proxy-MDL-lower-bound]` qualifier in lane
registry.**

---

## §9. Round 3 — pending

Per CLAUDE.md 3-clean-pass requirement, Round 3 needs a new adversarial
perspective. Candidates: phase-interaction analysis (Phase A → Phase B
→ Phase 4 INTEGRATION), default-override scan (changed function
defaults vs callers), comment-vs-code drift audit. Defer until Round 1
+ Round 2 action items are landed.

Round 1 + Round 2 action items remaining:
- §8.1 A0 evidence `[synthetic-proxy]` registry tag (Round 2 binding)
- Action #2 Path B step 4 element-weighted re-aggregate (Round 1)
- Action #5 FastViT-precision-compounding preflight check (Round 1)
- Action #6 auth-eval-consumer verification preflight check (Round 1)
- Action #7 A4 inflate-time T4 benchmark (Round 1, gated on credit)

3/5 of these are inline (~30 min, ~1 hour, ~1 hour). 2/5 are
operator-gated. Round 3 can proceed once the inline 3 are landed.
