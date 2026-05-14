# Journal/lab-grade documentation standard — directive 2026-05-14

**Operator directive verbatim 2026-05-14**: *"these plans are documented with detail and citations and provenance and rigor and journal and lab grade and math shown and such and links to other docs for deterministic reproducibility and so other agents can help or pick up in case we are interrupted"*

**Tag**: `research_only=true`; canonical persistence handoff per CLAUDE.md "Subagent coherence-by-default" mandatory pre-read of `.omx/research/*_directive_*` files dated within the last 24 hours. **EVERY future subagent + nested spawn MUST honor this standard.**

## Scope

This directive applies to:
1. Every landing memo (`feedback_*_landed_<YYYYMMDD>.md` in `~/.claude/projects/-Users-adpena-Projects-pact/memory/`)
2. Every campaign ledger (`.omx/research/campaign_<lane_id>_<YYYYMMDD>.md`)
3. Every dispatch report / Modal harvest record
4. Every research synthesis (`.omx/research/*.md`)
5. Every operator-routable decision block (in any artifact)
6. Every CLAUDE.md catalog row + non-negotiable section
7. Every probe-disambiguator design + verdict

## The 11 mandatory journal-grade elements

Every plan/landing/synthesis MUST include ALL eleven:

### 1. Hypothesis statement
Single sentence — what is the testable claim?

### 2. Math derivation
Show the equations. Examples:
- Rate-distortion: `R(D) = inf_{p(Y|X): E[d(X,Y)] ≤ D} I(X; Y)` (Shannon 1959)
- Cooperative-receiver: `H(X | W_scorer + A_scorer + P_scorer)` (Atick-Redlich 1990)
- Score formula: `S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489` (contest spec)
- Predicted ΔS = derive from first principles, NOT hand-wave

### 3. Citations (named + cross-ref)
Every theoretical claim cites the original work:
- **Shannon 1959** — *Coding Theorems for a Discrete Source With a Fidelity Criterion* (IRE Conv Rec §16 vector-valued distortion)
- **Atick & Redlich 1990** — *Towards a theory of early visual processing* (Neural Computation 2:308-320)
- **Rao & Ballard 1999** — *Predictive coding in the visual cortex* (Nature Neuroscience 2:79-87)
- **Ballé et al. 2018** — *Variational image compression with a scale hyperprior* (ICLR)
- **Ha & Schmidhuber 2018** — *World Models* (NeurIPS)
- **Tishby & Zaslavsky 2015** — *Deep learning and the information bottleneck principle* (ITW)
- **Rissanen 1978** — *Modeling by shortest data description* (Automatica 14)
- **MacKay 2003** — *Information Theory, Inference, and Learning Algorithms*
- **Fridrich 2009** — *Steganography in Digital Media* (UNIWARD)
- **Yousfi 2022** — *Detector-informed embedding* (alaska2)
- **Wyner-Ziv 1976** — *The rate-distortion function for source coding with side information at the decoder* (IT-22)

Cross-refs to internal docs use `[[memory-name]]` or `.omx/research/<file>.md` paths.

### 4. Provenance chain
- Commit shas for every code change
- Archive sha256 + bytes for every artifact
- Modal call_ids / Vast.ai instance IDs for every dispatch
- Inputs: video sha256, scorer weight sha256, runtime tree sha
- Axis tag per CLAUDE.md "Forbidden score claims": `[contest-CUDA T4]` / `[contest-CPU GHA Linux x86_64]` / `[macOS-CPU advisory only]` / `[predicted]` / `[MPS-research-signal]`

### 5. Empirical evidence tag
Per CLAUDE.md FORBIDDEN_PATTERN: every claim either carries `[empirical:<artifact path>]` OR `[derived:<formula>]` OR `[predicted; uncertainty±X]`. No bare "improves N%" / "verified" / "beats baseline" without a tag.

### 6. Reproducibility recipe
Exact command sequence to reproduce:
```bash
git checkout <commit_sha>
.venv/bin/python <script> --arg1 <val1> --arg2 <val2>
# expected output: ...
# expected sha256: ...
```

Include:
- Python version + key dependency pins (`pyproject.toml` excerpt)
- Hardware required (T4 / 4090 / A100 / CPU)
- Time budget (wall-clock + GPU-hours)
- Cost (provider rate × time)

### 7. Sister-substrate / sister-lane references
- Lane registry: which lanes does this affect?
- Catalog #s touched (#XXX)
- Substrate composition matrix entries (`tac.composition.registry`)

### 8. 6-hook wire-in declaration (Catalog #125)
Per CLAUDE.md "Subagent coherence-by-default":
1. Sensitivity-map contribution
2. Pareto constraint
3. Bit-allocator hook
4. Cathedral autopilot dispatch hook
5. Continual-learning posterior update
6. Probe-disambiguator (if 2+ defensible interpretations)

Each hook either explicitly named OR `N/A — <rationale>`.

### 9. Stop/continue thresholds
Per CLAUDE.md "Long-burn score-lowering campaign default":
- SMOKE threshold: `|score_delta| ≥ X` OR abort
- MID-STAGE threshold
- EXPORT threshold
- EXACT EVAL threshold (refuse if not in [low, high])

### 10. Reactivation criteria
If lane retires / defers / yields negative: explicit conditions that would reopen it. NEVER bare KILL per CLAUDE.md "KILL is last resort" non-negotiable.

### 11. Operator-routable decisions (≥3, typically 5)
Numbered list with:
- Decision name
- Cost (USD GPU + wall-clock)
- Risk assessment
- Recommended action (with rationale)
- Conditional dependencies (e.g., "if smoke green: ...")

## Crash-resume protocol mandatory inclusion

Per CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206:

Every subagent landing MUST include in the memo:
- `parent_id_or_session: <id>` — checkpoint chain
- `inherited_directives: [<files>]` — directive chain
- Final checkpoint status: `step: complete | in_progress | blocked`
- Resume instructions if interrupted (next-action field)

## Deterministic reproducibility table template

Every landing should include a table like:

| Element | Value | Verification |
|---|---|---|
| HEAD commit | `<sha>` | `git rev-parse HEAD` |
| Archive sha256 | `<sha256>` | `sha256sum <archive.zip>` |
| Archive bytes | `<N>` | `stat -c%s <archive.zip>` |
| Score (axis-tagged) | `<S> [contest-CUDA T4]` | `experiments/contest_auth_eval.py ...` |
| Inflate.sh sha | `<sha>` | `sha256sum <inflate.sh>` |
| Runtime tree sha | `<sha>` | `sha256sum -c <runtime-manifest>` |
| Modal call_id (if dispatched) | `<fc-XX>` | `modal app logs <call_id>` |

## Math-shown template

Every score-affecting claim must trace through math like:

> **Hypothesis**: Ballé hyperprior reduces archive bytes by Δb without distortion change → ΔS = +25·(B-Δb)/N - 25·B/N = -25·Δb/N.
>
> For PR101 anchor (B=296KB, N=37,545,489): ΔS for Δb=15KB → -25·15360/37545489 = -0.01023.
>
> Empirical bound: Ballé 2018 §IV.A shows 5-15% byte savings on natural image entropy bottleneck. PR101 latent_blob is 8528 bytes (Catalog #226 archive_sha256 87ec7ca5...492b5); 5-15% of 8528 = 426-1280 bytes savings → ΔS predicted [-0.0003, -0.0009] [predicted; uncertainty ±50%].

Show the formula. Show the derivation. Show the assumptions. Show the uncertainty bound.

## Why this matters

Per the operator: *"so other agents can help or pick up in case we are interrupted"*. The directive ensures:

1. **A new agent reading the memo in 6 months** has all context to reproduce + extend the work
2. **A sister subagent at a different recursion depth** can compose without re-deriving
3. **The operator** can audit any decision back to first principles
4. **Public OSS publication** (per the comma.ai/openpilot-style MIT release) carries journal-grade rigor
5. **Crashed subagent recovery** preserves not just bytes but reasoning

## Anti-patterns (forbidden)

Per CLAUDE.md FORBIDDEN_PATTERNS section, this directive adds:

- **Forbidden hand-wave math** — "this should help score" without R(D) derivation OR empirical anchor with axis tag
- **Forbidden uncited theory** — "cooperative-receiver" without Atick-Redlich 1990 citation
- **Forbidden missing provenance** — archive without sha256 + bytes + axis tag
- **Forbidden missing reactivation criteria** — DEFERRED without explicit re-open conditions
- **Forbidden orphan operator-routable** — surfaced without cost + risk + recommendation
- **Forbidden missing 6-hook wire-in** — every landing names all 6 hooks (with N/A acceptable)

## Cross-refs

- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE" (this directive is its applied instance)
- CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE" (7 mandatory fields aligned with this 11)
- CLAUDE.md "Mandatory crash-resume protocol" (Catalog #206; element 11 of this standard)
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" (the docstring-overstatement trap; element 5)
- CLAUDE.md "Apples-to-apples evidence discipline" (axis-tag rules; element 4)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" (element 4)
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (original 7-rule)
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md` (recursive R1-R4 extension)

## Effective immediately

All in-flight subagents (HARVEST-AND-Z1, D4-UNBLOCK, IBPS1-CANONICAL, CATALOG-226-REFACTOR, plus the 4 remaining about to spawn) MUST honor this on next checkpoint cycle. Their landings MUST include the 11 mandatory elements. Memory file template at the top of each landing should explicitly tag `journal_grade_v1=true` so future audits can confirm compliance.

Tagged `research_only=true`. NO score claims. NO GPU spend by this directive. Effective for all subagents from this directive's commit forward.
