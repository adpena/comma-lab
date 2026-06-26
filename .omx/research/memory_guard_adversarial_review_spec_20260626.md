# Memory Guard — Adversarial Review + Hardening Test Matrix (2026-06-26)

**Status:** BLOCKING GATE for all training relaunch (Task #172). No `train_witness*` /
`witness_capstone*` / paid n600 / any GPU arm may relaunch until this matrix passes
**3 consecutive clean passes** under an **independent** reviewer (reviewer ≠ the build
subagent `af3caf8f2a2d2596d` that authored the guard; reviewer ≠ me, who authored the
original stopgap). Per CLAUDE.md "Recursive adversarial review protocol" + reviewer-vs-author
separation.

## Why this exists (the two binding incidents)

1. **The OOM (operator 2026-06-25):** a naive/blind/confident fleet-launch to "saturate"
   128 GB OOM'd and killed the machine. Binding rule: **ALWAYS protect ≥30 GB free unified
   memory; OOM must NEVER happen again.** Scaling IS wanted — but MEASURED + SAFEGUARDED +
   INCREMENTAL. See `[[scale-measured-safeguarded-never-blind-confident-30gb-floor]]`.
2. **The molt control-plane-kill bug class (operator 2026-06-25/26):** *"Some fixes were made
   to bugs in the molt memory guard that were killing control plane surfaces over and over
   again."* The guard MUST NEVER kill control-plane (claude / codex app / codex cli / node /
   ssh / tmux / host-control). Vendored guard was from molt PRE-FIX `353784a1a`; molt is now
   `3b1e49b18` with the custody/identity-gating fixes. **Re-vendor from the fixed HEAD.**

The fix PRINCIPLE (must be vendored, not pattern/RSS-only): kill ONLY processes the guard has
explicit **CUSTODY** of (it launched / owns them) AND that pass **IDENTITY-GATING**; NEVER
terminate Codex lineage, host-control processes, or anything not under explicit custody — even
if it is the largest RSS and coincidentally matches a training pattern. Single-PID termination
must be identity-gated. **Better to ALERT than kill the control plane.**

## Provenance gate (P)

- **P1.** Final vendored kill-selector derives from molt `3b1e49b18` (custody/identity-gating),
  NOT `353784a1a`. The docstring/borrowed-substrate-accounting cites `3b1e49b18`.
- **P2.** The custody model (`memory_guard_core/process_model.py`-equivalent),
  `harness_memory_guard.py`, `process_sentinel.py` logic are present (or faithfully ported)
  in the pact vendor. List exactly which molt files/functions were vendored.

## Control-plane safety matrix (CP) — THE molt bug class

Each row: construct/simulate the scenario, run the kill-selector in `--select-victim-dry-run`,
assert the selected victim. **A single CP failure fails the whole pass.**

- **CP1.** `claude` process is the largest RSS → selector MUST NOT select it.
- **CP2.** `codex` CLI process present (any RSS) → MUST NOT select.
- **CP3.** Codex *app* / Code Helper / `@anthropic-ai/claude-code` / node host → MUST NOT select.
- **CP4.** `ssh` / `tmux` / login shell / `-zsh` / `-bash` → MUST NOT select.
- **CP5.** The guard's own PID + every ancestor PID (`_ancestor_pids`) → MUST NOT select.
- **CP6. [THE EXACT MOLT BUG]** a control-plane process whose **cmdline coincidentally
  contains a training token** (e.g. a Codex/claude session that has
  `train_witness_realized_through_R_mlx.py` in its argv because it is *editing* the file) and
  is the largest RSS → selector MUST NOT select it. **Custody/identity-gating MUST override the
  positive training pattern-match.** This is precisely what killed Codex/host repeatedly.
- **CP7.** When memory is critical AND no process under explicit custody matches → selector
  returns NO victim → guard **ALERTs and kills NOTHING**. (Fail toward preserving control plane.)
- **CP8.** A process matching the training allowlist but NOT under the guard's custody (e.g. an
  unrelated user-launched `train_*` not spawned by the guard) → MUST NOT be killed by RSS/pattern
  alone; custody is required. (Or: documented explicit exception with operator-visible rationale.)
- **CP9. Fail-safe on error:** any exception/ambiguity in victim selection → return NO victim
  (never default to killing the largest process).

## Available-memory metric matrix (M) — the macOS undercount finding

Finding (2026-06-26): naive `free + inactive` from `vm_stat` reported **25 GB** while
`memory_pressure` reported **96% free (~123 GB)** and the guard's own `--free` reported
**109.25 GB available / 83.03 GB strict-free**. macOS holds most RAM in compressed/purgeable
reclaimable states counted as neither free nor inactive. A guard flooring on `free+inactive`
would **falsely refuse launches** and **falsely kill training arms** when 90+ GB is free.

- **M1.** `--free` `available_gb` tracks `memory_pressure` reality (within a small margin),
  i.e. it includes purgeable/reclaimable — NOT naive free+inactive. Document the exact source
  (psutil `virtual_memory().available`? vm_stat with purgeable? host_statistics64?).
- **M2.** `strict_free_gb` is the conservative floor metric and is internally consistent
  (`strict_free_gb ≤ available_gb`).
- **M3.** Decide + document which metric drives (a) launch-preflight REFUSE and (b) the watchdog
  shed trigger. Recommendation: use the conservative `strict_free_gb` for the REFUSE side
  (fail-closed on launch) but ensure the watchdog SHED side does not false-positive on the
  reclaimable-memory undercount (else it kills arms when memory is actually fine).

## Floor enforcement matrix (F)

- **F1.** `--check --projected-gb N` → rc=3 (REFUSE) iff `(metric − N) < 30`; rc=0 (OK) otherwise.
  Spot-check: `--projected-gb 90` REFUSE, `--projected-gb 25` OK (at ~109 GB available).
- **F2.** Watchdog `--watch`: when the floor metric approaches 30 GB, sheds the largest arm
  **under custody** (CP-gated), re-measures, repeats; ALERTs (CP7) if nothing custody-killable.
- **F3.** Watchdog poll interval is sane (not a busy-loop); each poll re-reads live memory.

## Defense-in-depth presence matrix (D)

- **D1. Launch-preflight:** `spawn_durable_daemon.py` (or the launcher) calls `--check` and
  REFUSES any spawn that would breach the 30 GB floor.
- **D2. Whole-machine watchdog:** present, custody-gated (F2/CP).
- **D3. Per-arm RSS cap:** each arm wrapped in a `safe_run`-style self-RSS+walltime cap that
  process-group-kills the arm (not the machine) before OOM (exit 137=OOM / 124=TIMEOUT).
- **D4. MLX lazy-graph bound:** the through-R MLX trainer calls `mx.eval` frequently enough that
  a single arm cannot balloon past its steady RSS (the measured-not-assumed OOM root cause).
  Verify by profiling one arm's peak RSS over ≥N steps.

## No-orphan daemon discipline (O)

- **O1.** The watchdog daemon launches via `start_new_session`/`killpg` (no orphan) per
  `[[durable-detached-daemons-not-session-watchers]]`; canonical start/stop/status; STRICT
  preflight (Catalog #389) not regressed.

## Pass/clean-counter rules

- A **pass** = run the full P+CP+M+F+D+O matrix. A pass is **clean** iff zero failures.
- **3 consecutive clean passes** required to SEAL. Any failure resets the counter to 0; fix +
  re-run from a clean slate.
- Every pass records: molt HEAD vendored, victim-selection dry-run outputs for CP1–CP9, the
  metric numbers for M, and rc codes for F.

## On SEAL

1. Append a DAG FEED (`tools/subagent_commit_serializer.py`, `--expected-content-sha256`):
   guard SEALED, control-plane-safe (molt 3b1e49b18 custody/identity-gating), 30 GB floor,
   macOS-correct metric — training relaunch UNBLOCKED.
2. Relaunch the **n600 realized-axis MLX witness** as **ONE** bounded, watchdog-protected arm
   (the key measurement lost in the OOM): `experiments/train_witness_realized_through_R_mlx.py`,
   custom Metal backward (~59 s/ep), launch-preflight + per-arm RSS cap + watchdog all on.
   MEASURE peak RSS of the single arm BEFORE considering any second arm. NEVER N arms blind.
3. Pointer remains UNMOVED at contest-CPU **0.19110** until a byte-closed exact row beats it.

Cross: `[[scale-measured-safeguarded-never-blind-confident-30gb-floor]]` ·
`[[durable-detached-daemons-not-session-watchers]]` ·
`[[dag-survives-compaction-deterministic-repro-crux-convergence-standing]]`.
