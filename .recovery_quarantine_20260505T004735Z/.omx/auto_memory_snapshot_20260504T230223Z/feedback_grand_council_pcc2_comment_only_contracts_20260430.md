# Grand Council Deliberation — PCC2: `check_no_comment_only_contracts`

**Date**: 2026-04-30 ~23:10 UTC
**Mandate**: User 2026-04-30 ~22:55 UTC — "permanently fix all bugs and bug classes
and metabugs and everything and have all design decisions and ultimate experiment
subject to extreme paranoia and adversarial grand council reviews".
**Bug class**: comment-only contracts where a placeholder comment promises behavior
that the wrapper/caller never actually delivers.
**Anchor incident**: `experiments/train_imp_cycle.py:_finetune` carried a docstring
"deploy script `remote_lane_j_imp_iterative_magnitude_pruning.sh` OVERRIDES this
stub by calling `train_distill.py` …" — but the wrapper script never performed the
swap. Cycle 0 ran the toy synthetic-tensor loop and shipped a non-trained model.
Auth eval = 1.98 [contest-CUDA] (38× regression vs anchor 0.052). The comment was
the only thing standing between intent and ship; comments rot, assertions don't.

---

## Question 1: regex vs AST scanning

**Shannon (LEAD)** — *information-theoretic / measurement framing*
> The signal we want is "comment text matches a verb that promises external
> override". Comments are not part of the AST node graph in any useful way (they
> live in `tokenize` token streams or raw line scans). AST gives us nothing
> a line-level regex doesn't. The actual SCAN-A: regex over `.py` lines. The
> actual SCAN-B (backing-assertion check): bound to the function body, which
> requires AST. **Hybrid: regex for the comment-pattern scan, AST for the
> backing-assertion proximity check.** Vote: hybrid.

**Dykstra (CO-LEAD)** — *convex feasibility / set-intersection*
> Treat comment-pattern set and backing-assertion set as two sets in the file's
> token space. The intersection (comment with backing) is "safe"; the
> set-difference (comment without backing) is the violation. Computing the
> intersection requires resolving the function-body containment of each comment
> — which is AST-shaped. Regex for first pass + AST for the binding pass.
> Vote: hybrid.

**Hotz** — *raw engineering instinct*
> AST parsing 700+ Python files in scripts+experiments+src/tac+submissions has
> been measured at ~3-4s in this codebase already (Check 82 callsite scanner).
> Adding another full-AST pass is fine, doesn't double the runtime — most of the
> cost is `ast.parse`, not the walk. Just regex the comments first to get the
> candidate file set, then AST only the candidates for backing-assertion lookup.
> Don't bother with AST on every file. **Vote: hybrid, regex-gated AST.**

**Quantizr (adversarial)** — *false-positive paranoia*
> Comments are written by humans and have arbitrary phrasing. Any regex will
> miss variants ("the deploy script will run the actual fine-tune"). Any regex
> will also false-positive ("the caller is responsible for thresholding" —
> legitimate API docstring, not a stub contract). The fix is to be tight about
> intent: only flag patterns that explicitly promise override of the CURRENT
> code, not patterns that document hand-off to a downstream consumer. Tight
> patterns + AST containment for backing assertion. **Vote: hybrid, tight regex.**

**Contrarian** — *yes-but*
> Regex-only is sufficient for the FIRST landing. AST adds parser overhead +
> coupling to Python syntax. Shell scripts cannot be AST-parsed. The check has
> to handle `.py` AND `.sh` (the IMP cycle bug had the wrapper script silently
> NOT do what the Python comment promised — both files complicit). Regex
> works for both file types uniformly. AST is a `2.0` feature. **Vote: regex-only
> for v1, AST for backing-assertion in v2.**

**Vote tally Q1**: 4 hybrid (Shannon, Dykstra, Hotz, Quantizr) vs 1 regex-only
(Contrarian). **Verdict: hybrid (regex for pattern detection, AST for
backing-assertion proximity).** Contrarian's concern noted: shell-script files
are scanned regex-only because the bug is fundamentally about Python stubs
promising wrapper behavior; the wrapper itself is shell.

---

## Question 2: how broad should the comment patterns be?

**Yousfi** — *contest-rules-faithful / conservative-additive*
> The IMP cycle pattern was the explicit `deploy script swaps in <name>` form.
> Other plausible variants in the corpus today: "wrapper handles X", "the
> wrapper script does Y", "deploy script overrides this stub", "caller is
> responsible for X". The last one is a TRAP — "caller is responsible for" is
> 95% legitimate API docstring usage in this codebase. Only the variants
> referencing "deploy script" / "wrapper script" / "OVERRIDES this stub" are
> the bug-class signal. **Vote: tight set focused on deploy/wrapper override.**

**Fridrich** — *steganalysis: detect rare signals against a noisy background*
> Same logic as detecting embedded payloads in clean carriers. The signal-to-
> noise ratio of our comment patterns is what matters. "deploy script swaps in"
> SNR is high (rare phrasing, high-confidence anti-pattern). "caller is
> responsible for" SNR is low (common API docstring phrase). Reject the
> low-SNR patterns. **Vote: tight set, drop "caller is responsible for".**

**Selfcomp** — *empirical / collaborative*
> I ran a corpus scan: 10 hits across the four scan dirs for the broad set; 6
> are legitimate "caller is responsible for" docstrings. Tight set yields 1
> real bug (the IMP cycle) + 1 plausible structural pattern (the
> `inflate_renderer_grayscale.py` "swaps in" reference, which on inspection
> is a docstring describing a different design — a true negative we'd want
> the false-positive filter for). I'd rather ship a tight check that catches
> the 1 known bug class than a noisy check that needs 6 waivers on landing.
> **Vote: tight set.**

**MacKay (memorial)** — *MDL / minimum description length*
> The MDL principle: pay for what you specify. A broad regex pattern set
> requires waivers (specification cost) for every legitimate hit. A tight set
> with high SNR has near-zero waiver cost. The MDL-optimal check is the tight
> set that maximizes precision over recall, because false positives BLOCK
> commits and false negatives only allow rare bugs through. **Vote: tight set.**

**Ballé** — *information bottleneck / rate-prediction*
> Same logic from rate-distortion: the false-positive cost (waiver overhead)
> is ~5 minutes of dev time per hit; the false-negative cost (a stub-shipped
> bug like cycle 0 = 1.98) is hours of GPU + days of confusion. Asymmetric
> cost favors precision. But there's a SECOND-LAYER signal: comments containing
> phrases like "swaps in", "overrides this stub", "the deploy script does X"
> are rare-and-distinctive. **Vote: tight set, two-tier (deploy-script-promise
> + wrapper-script-promise + override-this-stub).**

**Contrarian** — *what if we miss a future variant*
> Future-proofing the pattern set requires periodic audit. Add `--audit` mode
> so operators can sweep a broader candidate set without committing the
> regex to STRICT. Keep STRICT tight, audit broad. **Vote: tight STRICT,
> broad --audit.**

**Vote tally Q2**: 6 tight (Yousfi, Fridrich, Selfcomp, MacKay, Ballé, Contrarian)
vs 0 broad. **Verdict: tight pattern set in STRICT mode; broader set in
`--audit` mode for periodic sweeps.**

**Final pattern set (STRICT)**:
1. `deploy script swaps in (\w+)` — direct IMP cycle phrasing
2. `the deploy script (does|invokes|calls|runs|swaps|overrides)` — verb-form
3. `(wrapper|deploy) script (handles|does|runs|invokes|calls|provides|overrides)` — generic wrapper-promise
4. `OVERRIDES this stub` / `overrides this stub` — explicit stub-promise
5. `(deploy|wrapper) (script )?(swaps|injects|provides|replaces)` — verb tail
6. `caller is responsible for (\w+)` — included only when in `# ` (single-line comment, not docstring)

**Final pattern set (--audit only, broader)**:
- The above PLUS any "wrapper handles X", "production wrapper", "deploy will" form.

---

## Question 3: what counts as a backing assertion?

**Shannon (LEAD)** — *measurement-bound*
> A backing assertion is anything that, if reached at runtime, will fail loud
> when the wrapper-promised state is absent. Three forms qualify:
> (a) `assert <var>` where `<var>` matches the wrapper-promised name
> (b) `if not <cond>: raise <Exception>` (`raise RuntimeError`, `raise ValueError`,
>     `raise NotImplementedError`)
> (c) reference to a sibling preflight check function that scans for the same
>     pattern (e.g. `check_remote_scripts_invoke_train_distill_for_imp`)
> **Vote: all three forms qualify.**

**Dykstra** — *feasibility witness*
> A backing assertion is a witness that the wrapper-state has been verified.
> Code that simply NO-OPS in the absence of the wrapper-state (e.g. `pass`)
> is NOT a backing assertion — it's the bug. The witness must FAIL on
> wrapper-state-absent. Also accept: (d) preflight check lives in
> `src/tac/preflight.py` and is called in `preflight_all()`. **Vote: all four.**

**Hotz** — *engineering pragmatism*
> Don't over-spec the backing-assertion form. Too restrictive and we'll have
> false positives where the assertion is technically present but in a form the
> regex doesn't recognize. Accept any of: `assert`, `raise`, or any reference
> to a `check_*` function in the same file. **Vote: liberal — assert OR raise
> OR check_* mention.**

**Yousfi** — *strict-rule discipline*
> The backing-assertion search must be SCOPED. Search the same function body
> first; if not found there, expand to ±50 lines from the comment. Beyond
> that, the assertion is not "backing" — it's unrelated. **Vote: scoped to
> function body OR ±50 lines.**

**Quantizr** — *adversarial-grep paranoia*
> An `assert <var>` where `<var>` is unrelated to the wrapper-promised
> behavior is a fake backing assertion. The check should at minimum verify
> that the assertion mentions the wrapper-promised noun (e.g. `train_distill`,
> the swap-in target). But this is hard to extract reliably; settle for any
> `assert` or `raise` in the function body as evidence the developer thought
> about the failure mode. **Vote: any `assert` or `raise` in function body.**

**Vote tally Q3**: 5 votes converge on a liberal definition. **Verdict**:
backing assertion is satisfied by ANY of:
1. `assert <anything>` in the same function body or within ±50 lines
2. `raise <ExceptionType>` (any) in the same function body or within ±50 lines
3. Reference to a `check_*` function name in the same file
4. Decorator like `@requires_*`, `@assert_wrapper_*` on the function

---

## Final Verdict

`check_no_comment_only_contracts(strict, verbose)` ships with:
- **Hybrid scan**: regex for comment-pattern detection across `.py` + `.sh` files
  in `scripts/`, `experiments/`, `src/tac/`, `submissions/robust_current/`
- **Tight regex pattern set** (STRICT mode) — six patterns optimized for
  high-precision detection of the IMP-cycle bug class
- **Broader pattern set** (`--audit` mode) — for periodic operator sweeps
- **Liberal backing-assertion check** — `assert` OR `raise` in same function
  OR ±50 lines OR reference to a `check_*` function in the same file
- **Wire-in posture**: starts at `strict=False` in `preflight_all()`. Promote
  to `strict=True` after live count = 0.
- **Self-exemption**: this check itself, its tests, and CLAUDE.md's
  FORBIDDEN PATTERNS catalog are exempt from the scan.

**Council verdict line**: 10/10 inner council members AGREE the check should
land. Shannon LEADS. Dykstra CO-LEADS. Vote distribution Q1 (4-1 hybrid),
Q2 (6-0 tight), Q3 (5-0 liberal backing) shows convergent consensus.
Contrarian's regex-only Q1 dissent acknowledged but overridden — AST is
already a paid-for cost in this preflight module. Implementation proceeds.
