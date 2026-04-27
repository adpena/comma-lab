# AGENTS

You are operating inside a dual-track lab for the comma video compression challenge.

Read `PROGRAM.md` before making changes.

## Score target — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Any auth score above 1.0 is UNACCEPTABLE.** Do the math during training. If projected auth > 1.0, something is wrong — stop and fix it before burning more GPU hours. Every training run, TTO, postfilter, and optimization must be evaluated against this target BEFORE launch, not after.

## Auth eval EVERYWHERE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY chained experiment MUST end with a CUDA auth eval against its best checkpoint.** Tracking only proxy `fp4_scorer` / `pose_loss` / training-loss is a WASTED run unless an authoritative score lands at the end. The proxy-auth gap can be 100-350x even on CUDA-CUDA (LANE-B 2026-04-26: proxy 0.0007 → auth 0.246, 350x). The proxy is a TRAINING SIGNAL, not a measurement.

This applies to:
- `experiments/pipeline.py compress` (HAS step_eval at end ✓)
- `scripts/remote_train_bootstrap.sh` (HAS Stage 5 auth eval ✓)
- `scripts/remote_pose_tto_bootstrap.sh` (HAS Stage 4 auth eval ✓)
- `scripts/remote_pose_tto_only_bootstrap.sh` (HAS Stage 4 auth eval ✓ as of 2026-04-26)
- `src/tac/experiments/train_renderer.py` — **GAP: NO auth eval on best.** Must be added: when a `*BEST*` checkpoint is saved, run a background CUDA auth eval and log the result alongside the proxy.
- ANY new training script, TTO loop, postfilter, or experiment runner.

**Pre-launch checklist (mandatory):**
1. Does the experiment end with `auth_eval_renderer.py` on the best checkpoint?
2. Is the auth eval result captured (RESULT_JSON or .json file) and surfaced to the operator?
3. If a chain has multiple "best" candidates (e.g., proxy-best, kl-best, hinge-best), does each get an auth eval?

**Pose TTO specifically:** the TTO loop MUST run a smoke auth eval at step 100 (and every 200 steps after) so the proxy-auth gap is detected within $0.50 of GPU spend, not at $5+ end-of-run.

**The authoritative measurement loop is:** contest-CUDA `inflate.sh` → `upstream/evaluate.py` on the EXACT archive bytes. Nothing else counts. Memory: `feedback_proxy_auth_math_useless`.

## eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST use eval_roundtrip.** There are ZERO exceptions. This includes:
- train_distill.py (has it)
- training.py Trainer (NOW has it, eval_roundtrip=True by default)
- constrained_gen.py (has it)
- optimize_poses.py (has it)
- qat_finetune.py (has it)
- ANY new training script or optimization

Without eval_roundtrip, proxy-auth gap is 2-6x on PoseNet. Every training run without it is a WASTED run. This mistake has been made on EVERY component in this project. It stops now.

## MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**LOCAL MPS IS NEVER TO BE USED FOR STRATEGY, PLANNING, OR ANALYSIS.** Verified 2026-04-25 with side-by-side gating measurement on the same pinned archive:

| Metric | Local MPS | CUDA A100 (contest scorer) | Drift |
|---|---|---|---|
| PoseNet distortion | 0.245 | **0.0107** | **23x WORSE on MPS** |
| SegNet distortion | 0.0024 | 0.00116 | 2x WORSE on MPS |
| **Final score** | **2.26** | **0.90** | **2.5x WORSE on MPS** |

PoseNet specifically drifts 23×. Likely cause: FastViT-T12 attention softmax + YUV6 chroma plane numerics differ between MPS and CUDA float16 implementations.

**Rules:**
1. ALL auth eval must run on CUDA (Vast.ai 4090, A100, T4). Never MPS, never CPU.
2. MPS is acceptable ONLY for proxy scoring during training (continuous monitoring), smoke tests (architecture validation), and code-correctness checks. NEVER for strategy decisions, ranking, or shipping.
3. Score numbers measured on MPS may NOT be reported as "auth" or "contest-compliant" anywhere — in commits, run_log, BATTLE_PLAN, or summaries. Tag them `[MPS-PROXY]` and treat as advisory only.
4. Before any major decision (kill/promote/ship), the score MUST come from a CUDA `inflate.sh` + `upstream/evaluate.py` run on the EXACT archive bytes.
5. preflight should reject auth eval invocations with `--device mps` and warn loudly.
6. The historical "2.01" / "2.26" / "2.91" numbers in memory and BATTLE_PLAN may all be MPS artifacts. The first verified CUDA contest-compliant baseline is 0.90 (2026-04-25 21:00).

This is the 5th catastrophic measurement bug class. Every score above this line in the run_log was potentially wrong by a factor of 2-3. Sub-Quantizr-0.33 is genuinely reachable from the true 0.90 baseline; do not give up real GPU dollars on the wrong baseline ever again.

## Remote code parity — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Before any remote eval or training run, verify the deployed code matches local HEAD.** Stale code on remote killed SHIRAZ today (16h training successful, then auth eval crashed silently because the deployed version had a NameError I had fixed locally that morning).

Rules:
1. `deploy_vastai.py launch()` MUST run `git pull --ff-only` on the remote BEFORE starting any work. If git pull fails (uncommitted changes, conflict, missing repo), abort the launch.
2. preflight should add a "remote_code_parity" check: SSH in, get `cd /workspace/pact && git rev-parse HEAD`, compare to local HEAD; block launch on mismatch unless `--allow-stale-remote` is passed (with warning).
3. The script process inside tmux MUST write a heartbeat to `/tmp/heartbeat_<session>.log` every N minutes. A separate watchdog reads heartbeats; alerts if stale > 30 min. Tmux session existence is NOT a heartbeat.
4. Any auth eval failure on remote that has been running > 1 hour is a CRITICAL incident — investigate immediately, do not let the instance keep accruing cost while broken.

This is the 6th catastrophic operational pattern. The cost: $3-10 per occurrence in idle GPU time + multi-day delays in measurement. Build the protocol so it never happens again.

## Primary duties

1. Keep `submissions/exact_current` runnable under the current published workflow.
2. Keep `submissions/robust_current` improving under a stricter, rule-faithful interpretation.
3. Leave durable state so a fresh agent iteration can resume work without relying on chat memory.

## Mutation frontier

You may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `jax/**`
- `mojo/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

You must not edit without explicit human approval:

- the pinned upstream snapshot
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `start.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Non-Negotiable Upstream Rule

- The pinned upstream snapshot is the source of truth for official scorer behavior and contest mechanics.
- Never edit, patch, monkeypatch, hotfix, or "temporarily" modify anything inside the pinned upstream snapshot unless the human explicitly approves that exact action.
- Never hack around upstream behavior by altering upstream files to make local experiments or scores look better.
- If upstream behavior appears wrong, inconvenient, or blocking, work around it only from the allowed mutation frontier and record the issue in repo state instead of changing upstream.
- If any experiment, proxy, or tooling change depends on upstream edits, stop treating it as compliant until the human has explicitly authorized that upstream modification.

## Strategic Secrecy Rule

- Protect competitive details for as long as that is strategically useful.
- Do not assume the right time to disclose is "now". Delay irreversible public disclosure until the human explicitly decides it is time to submit or publish.
- Treat the official public PR to the challenge repo as a disclosure moment. Until then, prefer private/local execution, private artifacts, and controlled summaries.
- Do not volunteer exact secret-sauce implementation details, hidden operational levers, or step-by-step reproduction recipes on public-facing surfaces unless the human explicitly wants that level of disclosure.
- Do not publish or surface unpublished private artifacts, credentials, private host details, or anything the human has not approved for disclosure.
- If there is a tradeoff between public writeup richness and preserving competitive edge, bias toward preserving edge unless the human says otherwise.
- **Explicit current exception:** the Cloudflare site may remain specific and detailed for now because the human explicitly approved that. Even there, still avoid exposing credentials, private infrastructure details, or anything the human has not approved for disclosure.
- **Explicit current restriction:** do not proactively publicize or advertise the Cloudflare site URL. Keep that link confined to private repo documentation and the eventual official submission until the human explicitly says the link itself can be shared broadly.

## Operating rules

- Prefer at most 3 experiments per cycle.
- Prefer small, reversible changes.
- Never claim a win without a measured score.
- Do not confuse `current_workflow` accounting with `rule_faithful` accounting.
- Keep both tracks healthy even if one looks dominant.
- Use JAX, Mojo, CUDA, or Rust only when they clearly reduce wall-clock cost or artifact size.
- Treat speculative ideas as side lanes unless evidence forces promotion.
- Keep public-facing detail intentional: specific enough to be credible, not automatically exhaustive.

## Git discipline

We need a fine-grained history of every file touched. Git is our lab notebook's version control.

- **Commit early and often.** After writing or updating any document, log, report, config, or experiment file, `git add` and `git commit` immediately with a descriptive message. Do not batch up changes across unrelated work.
- **One logical change per commit.** A run-log update is one commit. A new experiment script is another. A writeup edit is another. Do not combine them.
- **Always commit durable state files.** Every time you update `.ralph/run_log.md`, `.omx/state/*`, `.omx/research/*`, `reports/**`, or `docs/**`, commit right away. These are the research record.
- **Commit experiment artifacts.** New training scripts, config files, analysis outputs — commit on creation.
- **Never leave docs uncommitted overnight.** If a cycle touches documentation or state files, those changes must be committed before the cycle ends.
- **Commit message format:** `<what changed>: <why>` — e.g., `run_log: record h=64 breakthrough at 1.727` or `writeup: update hero tagline and nav links`.

This is critical for the doc evolution viewer and the competition writeup. Our git history IS our research timeline. Every uncommitted change is invisible history.

## Review gate — non-negotiable

- **NEVER use `REVIEW_GATE_OVERRIDE=1` when committing `.py` files.** The review tracker exists to catch bugs before they ship. Bypassing it on code files is how bugs ship. Work with the review gate, not around it.
- **For `.py` files:** run `python tools/review_tracker.py mark-file <file> --status reviewed` after each review pass, then commit normally. Let the gate pass naturally.
- **For non-code files** (`.md`, `.json`, `.env`, `.sh`, config, docs, reports): `REVIEW_GATE_OVERRIDE=1` is acceptable since the review tracker is designed for code review.
- If the gate blocks a `.py` commit, that means the code needs review first. That is the gate **working**, not the gate being broken.

## Tailscale fleet — non-negotiable

All lab machines are on Tailscale. **Always use Tailscale IPs** for SSH, rsync, and any remote operations. Never use raw LAN IPs or hostnames.

| Machine | Tailscale IP | OS | GPU | Notes |
|---------|-------------|-----|-----|-------|
| primary (M5 Max) | 100.81.85.28 | macOS | MPS 128GB | This machine |
| alejandros-mac-mini | 100.125.140.94 | macOS | Intel | Build server, Python 3.13 + uv |
| bat00 | 100.120.99.124 | Windows + WSL2 Ubuntu 24.04 | RTX 2070S (→3090) | Port 22=PowerShell, port 2222=WSL2. Scripts: `C:\Users\adpena\Desktop\commalab\` |
| molt | 100.114.131.54 | Linux | n/a | |
| tertiary | 100.65.24.39 | macOS | MPS | M1 MacBook Pro |

- `ssh adpena@100.120.99.124` connects to bat00 (Windows OpenSSH → PowerShell)
- bat00 has WSL2 Ubuntu 24.04 running (accessible via `wsl` commands inside PowerShell)
- bat00's NVIDIA driver supports WSL2 GPU passthrough
- Run `tailscale status` to verify all machines are online
- For bat00 Linux commands: use `python scripts/bat00.py wsl "command"` (port 2222, direct WSL2 sshd)
- For bat00 PowerShell: use `python scripts/bat00.py ps "command"` (port 22, Windows OpenSSH — rate-limited, avoid rapid successive calls)
- For bat00 status: `python scripts/bat00.py status`
- Windows OpenSSH has aggressive rate limiting (MaxStartups). Never send more than 2-3 SSH connections in quick succession to port 22. Use WSL2 port 2222 instead.
- **Never waste time debugging LAN connectivity. Tailscale is always the answer.**
- **Always use `scripts/bat00.py` for bat00 interaction — it handles quoting and port selection correctly.**

## Kaggle API/CLI — non-negotiable

- **`kaggle kernels push`** can only UPDATE existing kernels. To CREATE a new kernel, the slug must not already exist AND the slug must be short enough (long slugs like `comma-lab-asym-warp-supervised` fail with "Notebook not found").
- **Working pattern for new kernels**: use a shorter slug (e.g., `comma-lab-supervised-train`), push once to create, then subsequent pushes update.
- **`kaggle kernels status`** returns the LATEST version's status. After pushing a new version, the old version's error status persists until the new version starts running.
- **GPU assignment is random** — Kaggle may assign P100 (sm_60, unsupported by PyTorch >= 2.5) instead of T4. Our P100 check exits with FATAL. Just re-push until T4 is assigned.
- **2 concurrent GPU sessions max** on free tier. Push at most 2 kernels at a time.
- **Dataset mount path**: `/kaggle/input/datasets/<owner>/<slug>/` (NOT `/kaggle/input/<slug>/`).
- **`/kaggle/src/` is read-only** — results must go to `/kaggle/working/`.
- **All kernel code is in the code_file** — Kaggle script kernels only upload the single file. The tac wheel provides runtime deps.

## Canonical pipeline standard — non-negotiable

ALL experiments MUST run through `experiments/pipeline.py` with a profile name. No ad-hoc shell scripts. No hand-crafted SSH commands. One command, one standard, deterministic reproducibility everywhere.

```
python experiments/pipeline.py --profile shiraz --device cuda --output-dir results/shiraz
```

Requirements:
1. **Profile from `profiles.py` is the ONLY config source.** No CLI flag overrides for architecture params. The profile IS the experiment definition.
2. **Seeds pinned.** `torch.manual_seed`, `numpy.random.seed`, `random.seed` — all from `profile.seed`. Deterministic CUDA (`torch.use_deterministic_algorithms(True)` where possible).
3. **Full provenance.** Git hash, GPU info, PyTorch version, profile dict, timestamps per stage — saved as JSON alongside results.
4. **Validate at every boundary.** Checkpoint exists, shapes match, loss is finite, archive size reasonable. Hard errors, not warnings.
5. **Full chain.** train → QAT → pose TTO → build archive → contest_eval. Every stage runs automatically. No manual intervention between stages.
6. **Bundle all artifacts.** Checkpoints, logs, provenance JSON, auth eval results — packaged as tarball for download.
7. **Platform-agnostic.** Works on cuda, mps, cpu. Same pipeline locally and on Vast.ai/Modal/Kaggle.

This is the openpilot standard: deterministic, reproducible, no runtime format negotiation, schema-first data contracts, fail-fast validation at every boundary. We are professional engineers contributing to production infrastructure. The ad-hoc approach is over.

## Deployment version checklist — non-negotiable

Before deploying ANY code to Modal, Kaggle, Lightning, or any remote platform:

1. **Bump `pyproject.toml` version** if any `src/tac/` code has changed since the last wheel.
2. **Update `deploy_config.py` BASE_FLAGS** to match any changed defaults in the training script. The "default override" antipattern has caused 4 bugs: never change a default without grepping for callers that pass it explicitly.
3. **Rebuild the wheel** (`uv build --wheel`) AFTER all code changes are committed.
4. **For Kaggle**: upload the new wheel to the dataset, run `wait_for_dataset_ready()`, then push kernels. The old wheel in the dataset will silently use old code.
5. **For Modal**: `add_local_dir` mounts source at startup — Modal always gets the latest committed code. But `deploy_config.py` CLI flags still override script defaults. Verify the flags match.
6. **Verify the REQUIRED_DATASET_ASSETS dict** in `build_kaggle_kernels.py` includes the new wheel filename (update version string when bumping).
7. **Never push Kaggle kernels without verifying** that every required asset exists in the dataset at the expected size. The preflight disk check inside kernels is a last resort — it should never fire.

The consequence of skipping this checklist: experiments run with stale code, produce misleading results, and waste GPU hours. This has happened repeatedly (tac 1.0.4 deployed with old Lagrangian caps, raft_flow.pt missing from dataset, R1 OOM fix bypassed).

## Recursive adversarial review protocol — non-negotiable

Before deploying any change to training code (`train_renderer_fridrich.py`, training configs, loss functions, Lagrangian parameters), run the recursive skunkworks council review:

1. **Each round**: Every council member (Yousfi, Fridrich, Contrarian, Quantizr, Hotz) takes a different adversarial perspective. Each reviews ALL changed code. Findings are categorized as CRITICAL / Medium / Low.
2. **Fix immediately**: All issues found in a round are fixed and committed before the next round begins.
3. **Clean pass counter**: A round with zero issues is a "clean pass." The counter resets to 0 whenever a round finds any issue.
4. **Gate**: 3 consecutive clean passes required before the code is cleared for deployment (wheel build, Modal launch, Kaggle push).
5. **Adversarial perspectives** (rotate each round): trace actual call sites (not just function signatures), check phase interactions, verify resume scenarios, mental-execute edge cases (`--batch-size 1`, `--rho-max 0`), check default arguments that callers might override, verify comments match code.
6. **The "default override" antipattern**: When changing a function default, ALWAYS grep for callers that pass the argument explicitly. A changed default that no caller uses is dead code. This caught the R1 OOM fix being completely bypassed (Round 3).
7. **Phase-gate all phase-sensitive thresholds**: Any threshold compared against a metric that varies by training phase (e.g., PoseNet distortion starts ~180 in Phase 1, converges to ~0.05 in Phase 2) MUST be phase-gated or set conservatively enough for all phases.

This protocol caught 2 CRITICAL bugs (auto-kill at epoch 200, OOM fix bypassed) and 3 medium issues in the Lagrangian R1-R4 patch. Without it, v5 training would have failed within the first 200 epochs.

## Design decisions — non-negotiable

- **NEVER make design decisions unilaterally.** Always consult the skunkworks council (Yousfi + Fridrich + Hotz + Quantizr + Contrarian) before implementing any change that affects training behavior, loss functions, architecture configuration, interpolation methods, boundary values, optimization strategy, or any other design tradeoff.
- **Clear bugs** (crashes, wrong formulas, missing imports, dead code) can be fixed immediately without council approval.
- **Design tradeoffs** (bicubic vs bilinear, loss function choice, constraint boundaries, rho growth strategy, what to include in archive, etc.) MUST be council-approved before implementation.
- **If unsure** whether something is a bug fix or a design decision, it's a design decision. Ask the council.
- Present the issue, list the options with pros/cons, and let the council make a binding decision.

## Council conduct — non-negotiable

- **The council must NEVER have a conservative bias.** "Don't change working code" is NOT a valid argument. "Ship what we have" is NOT a valid argument. The only valid arguments are mathematical, scientific, geometric, or empirical.
- **Every council member must be the most expressive, assertive, passionate version of themselves.** They bring their full life's work, career, domain expertise, cross-disciplinary insights, and everything they care about to every deliberation. No holding back. No false consensus.
- **The council exists to find the OPTIMAL solution, not the safe solution.** If a 5-line change could improve the score by 0.01, it MUST be debated on its merits — not dismissed as "overengineering" or "not worth the risk."
- **Disagreement is healthy.** Unanimous votes should be scrutinized. If all five members agree instantly, someone isn't thinking hard enough.
- **The Contrarian's role is to challenge, not to conserve.** The Contrarian challenges WEAK arguments, not BOLD ones. A bold, well-reasoned proposal should survive the Contrarian. A lazy consensus should not.

## Experiment design — non-negotiable

Every experiment MUST follow this process before touching any GPU:

1. **Pre-registered hypothesis** with success/kill/concern criteria
2. **Council design review**: Yousfi + Fridrich sign off on config, resolution, step count, conditioning
3. **Faithful to the actual design**: no toy configs, representative resolution, enough steps for signal
4. **No janky smoke tests**: a test at 1/4 resolution for 500 steps cannot kill a technique. Bias toward keeping lanes open.
5. **Resource estimate**: GPU hours, VRAM, expected runtime
6. **Replicability record**: all params saved before running, full results after
7. **No premature kills**: a negative result on an underspecified test means the test was wrong, not the technique
8. **Multiple contenders → multiple paths**: When there are two or more plausible contenders for a design decision (e.g., "supervised" vs "RAFT-only", "architecture A" vs "architecture B"), do NOT pick one and discard the others. Run them in parallel. The score is the only valid arbiter. Never collapse multiple viable hypotheses into one without empirical evidence.

This last rule is non-negotiable. Premature convergence on a single path is how labs fall behind. If you're uncertain which variant is better, the answer is always: run both.

Yousfi, Fridrich, and the Contrarian are the **tripartite pact** — the three voices that must reach consensus before any major decision. Yousfi and Fridrich have domain expertise as the world's foremost steganalysis experts and contest designers. The Contrarian has veto power on any experiment that lacks rigor, wastes resources, or is built on unvalidated assumptions. All three must sign off on experiment design and kill/promote decisions.

Together with **Quantizr** (adversarial member, reverse-engineers competitor approaches, keeps us honest on what the leaderboard actually rewards) and **George Hotz** (raw engineering instinct, builds fast, breaks conventional wisdom, champions analytical shortcuts over learned complexity), these five form the **non-conservative skunkworks inner council**. All five voices are permanently active. No member may be silenced or deferred in any deliberation. The council is non-conservative by charter: the burden of proof is always on *not* trying something, never on trying it.

## Required durable state

After each serious cycle, update and **commit** at least:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`

## Promotion rules

A candidate may be promoted only after:

1. packaging succeeds
2. inflation succeeds
3. shape/frame-count checks pass
4. proxy evaluation looks promising
5. full evaluation confirms the gain or records the failure

## Track-specific guidance

### Track A: `exact_current`

- Preserve transparency.
- Use it as a live test of the currently published workflow.
- If upstream changes invalidate the exploit assumptions, demote it immediately to a research note and keep the repo useful.

### Track B: `robust_current`

- Start with safer codec improvements and task-aware pre/post processing.
- Add sparse residuals before adding heavier learned components.
- Only promote a neural side-model if its bytes and runtime clearly justify themselves.

## GPU budget and compute resources — non-negotiable

### Optimal GPU: RTX 4090 on Vast.ai
- **RTX 4090 at $0.25/hr on Vast.ai** is the optimal price/performance for our workload (287K param model, ~800MB VRAM, dominated by scorer forward/backward passes).
- 4-5x faster than T4 at roughly the same cost. A 2-hour T4 run finishes in ~25 min on 4090.
- Filter: `gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30`
- Budget: $25 credits available. Hard cap at $24. Track all spend.

### Platform hierarchy (price/performance order)
| Platform | GPU | $/hr | Speed vs T4 | $/experiment | Use For |
|----------|-----|------|-------------|--------------|---------|
| Vast.ai | RTX 4090 | $0.25 | 4-5x | $0.20 | New experiments (primary) |
| AWS spot | T4 (g4dn.xlarge) | $0.22 | 1x | $0.60 | Scale-out, auth eval fleet |
| Modal | T4 | $0.59 | 1x | $0.60 | Existing infra, quick deploys |
| Local M5 Max | MPS | Free | ~0.5x | Free | Development, smoke tests |
| Kaggle | T4/P100 | Free | 1x | Free | Bonus parallelism (unreliable) |

### Budget caps (DO NOT OVERSPEND)
- Vast.ai: $25 total ($24 hard cap in deploy script)
- AWS: $100 total (free credits)
- Azure: $200 total (free credits, need `az login`)
- Modal: $30/mo free credits

### Deployment rules
- **Always use `modal run --detach`** for long-running experiments (prevents disconnect kill).
- **Always use unique `--tto-subdir`** per experiment to prevent checkpoint contamination.
- Vast.ai deployment goes through `src/tac/deploy/vastai/` (canonical module, not ad-hoc scripts).
- All platforms must use `load_differentiable_scorers()` for any gradient-based optimization.

## Tooling — non-negotiable

- **Always use `uv`** for Python package management. Never use raw `pip`, `pip3`, or `pip install`.
  - Install packages: `uv pip install <pkg>`
  - Create venvs: `uv venv`
  - Run scripts: `.venv/bin/python` (the uv-managed venv)
  - On remote machines: install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv && uv pip install ...`
- **Always use the tac library** for new training experiments. The canonical entry point is `experiments/train_tac.py`.
  - Do NOT duplicate training code in new experiment scripts.
  - All loss functions, architectures, data loading, and training loops live in `src/tac/`.
  - **Use named profiles** for new training runs: `--profile proven_baseline` is recommended (produced the 1.33 authoritative score).
  - Available profiles: `proven_baseline` (1.33 settings), `psd_standard_adaptive` (PSD arch + frontier), `council_v1` (static, legacy), `segnet_attack` (aggressive), `h96_council`, `smoke` (quick test).
  - Profiles live in `src/tac/profiles.py`. CLI args override profile values.
  - **Use precomputed data** when available: `--precomputed experiments/precomputed_local` (skips 5-min video decode).
  - **Adaptive weight formula was retired** (`src/tac/adaptive.py`): T² cancels in the derivation, making the formula vacuous. Use standard loss with static weights instead.
- **Always commit after every change.** Git history is the research timeline.
- **Use `scripts/modal_check.py`** to check Modal TTO progress. Shows batch progress, ETA, recent PoseNet snapshots, and running apps. Run with `.venv/bin/python scripts/modal_check.py`.
- **Use `scripts/kaggle_check.py`** to check Kaggle kernel status. Run with `.venv/bin/python scripts/kaggle_check.py`.
- **Use `scripts/bat00.py`** for bat00 interaction. Handles quoting and port selection (port 22=PowerShell, port 2222=WSL2).
- **"Multipane matplotlib data viz"** or **"canonical comma.ai data viz"** means the 6-panel analysis GIF/MP4:
  - Row 1: GT Original | Our Reconstruction | Pixel Error (hot colormap)
  - Row 2: GT SegNet masks | Our SegNet masks | SegNet Disagreement (red)
  - Generated inline with pyav + SegNet + matplotlib colormaps, output to `~/Downloads/`
  - Requires TTO frames (`tto_frames.pt` from Modal volume) and GT video (`upstream/videos/0.mkv`)
  - SegNet needs `(B, T, C, H, W)` input format with `T=1` for the sequence dimension

## Critical lessons — DO NOT repeat these mistakes

### CATASTROPHIC FAILURES (2026-04-21) — never again

These failures cost weeks of wasted work and produced months of invalid measurements:

- **MASKS.MKV AT 48x64 DESTROYED THE SCORE.** The mask video was at 1/8 resolution (48x64), but the renderer was trained on 384x512. The renderer outputs at the same resolution as input masks — so it produced 48x64 frames upscaled 18x to camera resolution. PoseNet distortion was 94.63 (catastrophic) vs 0.015 with correct masks. Score was 103.27 vs projected ~0.71. **ALWAYS verify mask resolution matches renderer training resolution. ALWAYS run the full inflate.sh → evaluate.py pipeline before claiming any score.**
- **ARCHIVE MEASUREMENT DISASTER.** All auth evals for weeks used a renderer-only archive (119-180KB) instead of the full submission archive (338KB+). Rate term was wrong by 0.108 points. Every score reported was optimistic. **ALWAYS use `submission_archive.require_valid_archive()` before any eval.**
- **1199 OVERLAPPING PAIRS vs 600 NON-OVERLAPPING.** auth_eval.py used `range(N-1)` (1199 overlapping pairs) but upstream evaluate.py uses `seq_len=2` non-overlapping batching (600 pairs). Every `eval_checkpoint()` score was computed with wrong pair construction. **ALWAYS diff new scoring code against upstream evaluate.py line by line.**
- **eval_roundtrip DEFAULTED FALSE.** All TTO runs optimized against a proxy that didn't simulate the contest eval roundtrip (384→874→uint8→384). Combined with noise_std=0 (Hotz fix dead code), this caused proxy-auth PoseNet drift up to 11x. **eval_roundtrip MUST default True. noise_std MUST be threaded.**
- **AUTO-BUNDLE BY FILE EXISTENCE.** compress.sh auto-included any .pt/.bin file sitting next to the submission. Stale experiment artifacts silently inflated archive size. **ALL archive contents must require explicit flags. No implicit bundling.**

### Root cause pattern

Every failure above is the same pattern: **a component quietly produced wrong output, and no downstream check caught it.** The fix is the same every time: hard errors, not warnings. Validation gates, not hopes. Full e2e pipeline tests, not component-level checks.

### Non-negotiable protocol after every change

1. Run `inflate_renderer.py` on the archive
2. Run upstream `evaluate.py` on the inflated output
3. Compare the score to the last known-good score
4. If any component was changed, verify the full e2e score moved in the expected direction

If you skip this protocol, you WILL produce invalid scores. This has happened 4 times. There is no excuse for a 5th.

### Previously known failures (still valid)

- **KL distill caused PoseNet collapse as primary loss.** BUT Quantizr uses kl_on_logits(T=2.0) for SegNet during specific training phases alongside standard loss. Revisit with staged approach — KL distill for SegNet only, not as sole loss.
- **Adaptive weights are DEAD.** Hinton T² double-correction.
- **Neural artifacts must be inside archive.zip** per contest rules (affects rate calculation).
- **Do NOT use PoseNet gradient caps/clamps.** Caused 26x PoseNet regression.
- **Do NOT use segnet_loss_weight > 100 with any loss mode.** Overwhelms PoseNet signal.
- **Standard loss is the ONLY proven technique.** All other loss modes (KL distill, SegNet attack) failed authoritative eval.

## Current frontier experiments

- **PSD architecture** (PixelShuffle-Downscale): promising for SegNet but untested with standard loss on authoritative scorer
- **5 adaptive frontier items**: boundary dispatch for standard loss, sin² ramp, replay gate, 3-phase eval, plateau LR scheduler
- These are implemented but unvalidated. Do not promote without authoritative eval.

## Strict scorer rule — non-negotiable (canonical, binding)

- **NO loading PoseNet or SegNet at inflate time.** If our inflate script loads scorer weights for ANY purpose (TTO optimization, mask extraction, embedding computation, gradient descent), those weights must be in archive.zip per Yousfi's PR #35 rule. Including them (~73MB) destroys the rate term. Therefore: no scorers at inflate time, period.
- **TTO is a compress-time tool ONLY.** TTO frames are training data for the renderer, not submission artifacts. Unlimited compute at compress time, single forward pass at inflate time.
- **Any inflate-time feature that loads scorers** must be labeled "non-compliant, requires compliance ruling" and disabled by default (`INFLATE_TTO=0`).
- **NEVER claim a contest-compliant score** that depends on inflate-time scorer access.

## Lane separation — non-negotiable

There are TWO score lanes. They MUST NEVER be conflated.

- **Lane 1: Contest-Compliant (PRIORITY).** Goes through inflate.sh → inflate_renderer.py → evaluate.py within 30 min on T4. No scorers at inflate time. Previous "0.87" was INVALID (48x64 masks + wrong pairs + wrong archive). True baseline with full-res masks: pending full e2e eval (projected ~2.2 from 10-pair sample).
- **Lane 2: Unlimited Compute (Paper).** TTO optimization at compress time, unlimited steps. Previous "0.41" was INVALID (same measurement bugs). For the arXiv paper scalability section ONLY.
- **Every score must be labeled** `[contest-compliant]` or `[unlimited-compute]`. No exceptions.
- **NEVER say "our score is X"** without specifying which lane.

## Auth eval measurement — non-negotiable

- **EVERY auth eval must use the EXACT archive that will be submitted.** Never create a temporary archive with different contents. The rate term depends on archive.zip file size — wrong archive = wrong score.
- **EVERY auth eval report must print the archive size used.** If it doesn't match the submission archive, the score is INVALID.
- **Auto-auth-eval in training must construct archives with ALL submission artifacts** (renderer.bin, masks.mkv, poses.pt, any other bundled files). Not just renderer.bin.
- **NEVER celebrate a score without verifying the measurement apparatus.** Check: archive size, inflate pipeline, eval pipeline. A wrong measurement is worse than no measurement.
- **Proxy scores are APPROXIMATIONS, not truth.** The proxy-auth gap can be 2-11x for PoseNet. Always label proxy vs auth. Always run auth eval before claiming any result.

This rule exists because we celebrated auth 0.36 that was actually ~0.41 due to using a renderer-only archive (119KB) instead of the full submission archive (183KB). Every auth eval in the session was wrong by 0.04-0.05 points.

## Submission PR gate — non-negotiable

- **NEVER submit a PR** until the score has undergone a 5-turn consecutive clean-pass adversarial skunkworks council review with extreme paranoia. This is stricter than the standard 3-pass greenup. All 15 council members review. ANY issue resets the counter to 0.
- **The score used for submission** must come from the contest-compliant auth eval (through inflate.sh), not proxy or bypassed eval.

## Quantizr intelligence — verified competitive data (2026-04-21)

Quantizr (Jimmy, UCLA CSE/Neuro) leads at 0.33. **Archive is 299,970 bytes (293KB), NOT 15KB.**

- **Architecture**: FiLM-conditioned depthwise-separable CNN, 88K params, ~64KB FP4
- **Archive contents**: renderer.bin (FP4+Brotli) + masks.mkv (AV1, ONLY frame2 masks, higher CRF) + poses.pt
- **Training**: 5-stage pipeline (anchor→finetune→joint→QAT→final), EMA, diff_round(), diff_rgb_to_yuv6()
- **SegNet**: kl_on_logits() with T=2.0 for distillation during training
- **Key trick**: Encodes only 600 odd-frame masks (frame1 is warped from frame2)
- **His own assessment**: "sub 0.30 is possible just by sweeping conv dims" — he stopped optimizing
- **Rate**: 25 * 299970 / 37545489 = 0.200. Their distortion is ~0.13.

Yousfi (challenge creator) was Fridrich's PhD student at Binghamton DDE Lab. EfficientNet steganalysis surgery → informed SegNet scorer design. The challenge IS inverse steganalysis.

## Exact scorer architectures — VERIFIED from upstream modules.py

**SegNet**: `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)`
- EfficientNet-B2 (NOT B4), vanilla stride-2 stem (no Yousfi surgery)
- Input: LAST frame only `x[:, -1, ...]`, bilinear resize to (512, 384)
- Output: 5-class logits, distortion = argmax disagreement rate
- **Blind spot**: stride-2 stem loses half resolution immediately → artifacts below (256,192) invisible
- **Key**: only argmax matters — tiny logit perturbations at class boundaries are the ENTIRE signal

**PoseNet**: FastViT-T12 backbone (NOT EfficientNet)
- 12-channel input: 2 frames × YUV6 (4 luma + 2 chroma subsampled)
- rgb_to_yuv6 → resize to (512,384) → normalize (mean=127.5, std=63.75)
- Hydra head: vision(2048) → summary(512) → ResBlock → 12-dim pose → first 6 used
- Distortion = MSE on first 6 pose dimensions

**Yousfi's repos (competitive intelligence)**:
- `github.com/DDELab/deepsteganalysis` — surgery code for EfficientNet steganalysis
- `github.com/YassineYousfi/alaska` — JPEG steganalysis challenge code
- `github.com/YassineYousfi/OneHotConv` — CNN vs classical features paper
- `github.com/YassineYousfi/comma10k-baseline` — comma segmentation baseline
- `github.com/YassineYousfi/autostego` — adversarial steganography framework

## Fridrich inverse steganalysis — how to beat the scorer

1. **UNIWARD**: errors in textured regions are undetectable. Weight loss by inverse local variance.
2. **Detector-informed embedding** = our TTO approach. Fridrich-approved (Yousfi 2022).
3. **Square root law**: spread small errors (L∞ penalty), don't concentrate large ones.
4. **CNN blind spots**: EfficientNet misses DCT statistics, has texture-region blind spots.

## QAT pipeline — non-negotiable for FP4 deployment

For our ~80-100K param renderer:
1. **Train float first** with all techniques (eval_roundtrip, noise, EMA, hinge loss)
2. **Freeze BatchNorm stats** (eval mode on BN layers)
3. **Insert per-channel FP4 fake-quant** on weights + per-tensor on activations
4. **Fine-tune 20% of original epochs** at 0.1× LR (LSQ step size lr = 0.01 × base_lr)
5. **Export**: 4 bits/param → ~40-50KB for 80K params

We HAVE FakeQuantSTE, Uint8STE, FakeQuantFP4 in `src/tac/quantization.py`. We HAVE LSQ support in training.py. These are wired but have never been used in a complete training pipeline for the renderer.

## Mask encoding — verified data (2026-04-21)

- **Renderer REQUIRES 384x512 masks.** Lower resolution catastrophically degrades: 192x256 → 2.9x worse, 96x128 → 34x worse, 48x64 → 108x worse.
- **Entropy coder** (mask_entropy_coder.py): 990KB for 1200 frames at 384x512, lossless. ~495KB for 600 frames.
- **AV1 monochrome** (mask_codec.py): has int8_t overflow bug at 384x512. Must fix.
- **Quantizr paradigm**: Store ONLY 600 odd-frame masks (frame2). Frame1 is warped.
- **inflate_renderer.py has mask upsample fix** (added 2026-04-21) for sub-native resolution masks.

## TRUE score data (2026-04-21) — verified via upstream evaluate.py

| Config | Seg | Pose | Rate | TOTAL | Notes |
|--------|-----|------|------|-------|-------|
| 384x512 masks + ASYM + poses | 0.116 | 0.374 | 1.528 | **2.01** | Full-res masks, rate-limited |
| 48x64 masks (old, NOT upsampled) | 72.3 | 30.8 | 0.23 | **103.27** | Catastrophic mask bug |
| 48x64 masks (old, upsampled) | 28.3 | 25.0 | 0.23 | **53.61** | Old AV1 artifacts in masks |

## Vast.ai deployment — non-negotiable

- **API key** at `~/.config/vastai/vast_api_key`. SSH key must be registered at account level BEFORE creating instances.
- **Always use `python3 -u`** (unbuffered) for background jobs on Vast.ai. Python stdout buffering eats logs otherwise.
- **Always include repo root in PYTHONPATH**: `PYTHONPATH=src:upstream:$PWD`.
- **Search pattern**: `vastai search offers 'gpu_name=RTX_4090 reliability>0.95 inet_down>200 disk_space>30 num_gpus=1' -o 'dph'`
- **Budget**: $25 total ($24 hard cap). Track all spend. Destroy instances immediately when done.
- **Modal credits exhausted** as of 2026-04-15. Use Vast.ai for all new GPU work.

## SegNet paradigm shift — non-negotiable knowledge

At our operating point, **SegNet is 77x more important than PoseNet** in the scoring formula:
- SegNet: 100 × seg_dist → dominates the score
- PoseNet: √(10 × pose_dist) → essentially solved at 100 TTO steps
- **ALL optimization effort must prioritize SegNet reduction.**
- PoseNet improvement has negligible score impact. SegNet improvement has massive score impact.
- TTO step curve (empirical, Vast.ai 4090): phase transition at 80-100 steps for PoseNet. SegNet only moves at 300-500 steps.
- The renderer has hit its SegNet architectural ceiling. Breaking through requires architectural changes, not more TTO.

## Ralph-style execution model

Treat files and git as memory.
Each iteration should be resumable from disk.
Do not rely on long chat context for continuity.
Commit after every meaningful file change — git history is the research timeline.
