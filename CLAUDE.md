# AGENTS

You are operating inside a dual-track lab for the comma video compression challenge.

Read `PROGRAM.md` before making changes.

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

## Critical lessons — DO NOT repeat these mistakes

- **KL distill is DEAD.** Two authoritative evals confirmed PoseNet collapse: 1.85 and 2.05. NEVER use KL distill loss_mode.
- **Neural artifacts must be inside archive.zip** per contest rules (affects rate calculation).
- **Do NOT use PoseNet gradient caps/clamps.** Caused 26x PoseNet regression.
- **Do NOT use KL distill loss_mode.** Two authoritative evals confirmed PoseNet collapse (1.85 and 2.05).
- **Do NOT use adaptive_rebalance=True.** The formula was vacuous (T² cancels).
- **Do NOT use segnet_loss_weight > 100 with any loss mode.** Overwhelms PoseNet signal.

## Ralph-style execution model

Treat files and git as memory.
Each iteration should be resumable from disk.
Do not rely on long chat context for continuity.
Commit after every meaningful file change — git history is the research timeline.
