# BUILD #298 — tier-scaled DYNAMICAL safety floor + continuous band wiring (+ GB-F1 units fix + #246 full-tree pause)

**2026-07-04 · #294 follow-up (operator: "all must be sophisticated and dynamical and scalable and
tested and hardened") + coordinator scope addition ("Fix all regardless of severity": GB-F1 + F1 from
`.omx/research/throughput_review_findings_20260704.md`). $0 verifications only; #205 (pid 29129)
untouched; the RUNNING blackbox daemons (primary pid 19895, tertiary) NOT restarted. Pointer
contest-CPU 0.19110 UNMOVED — everything here is MEANS (apparatus/machine-protection).**

## 1. The formula (replaces `DEFAULT_SAFETY_MARGIN_FLOOR_GIB = 8.0` as the default)

```
floor = clamp( max( ABS_MIN,                                # 2.0 GiB   jetsam-avoidance minimum
                    measured_cp_rss + cp_headroom(T),       # DYNAMICAL leg (follows the live control plane)
                    0.08 * T ),                             # STATIC physics leg (measurement-free)
               ABS_MIN, 0.5 * T )                           # cap: a floor may NEVER eat the box
cp_headroom(T) = max(1.0, 0.05*T)
```

Every term traceable (docstring + constants block in `tools/system_memory_governor.py`):
- **ABS_MIN 2.0** — the M1 ran its 3.69 GiB smoke without swap-thrash at ~1.9–2.5 GiB conservative
  available (tertiary spec §2/§3); below ~2 GiB reclaimable macOS pressure escalates.
- **cp_headroom = max(1, 0.05T)** — control-plane burst headroom: 6.4 GiB @128 (2–3 concurrent
  build/review agents at 1–3 GiB each — the 07-02 crash's unaccounted contributors); 1.0 @8
  (single-agent burst; same order as the tier's measured +1.26 GiB verdict step).
- **measured leg (cp_rss + headroom)** — the floor FOLLOWS what the control plane actually uses
  (`measured_cp_rss = used − Σ tracked RSS`, both true GiB post GB-F1); recomputed every tick.
- **0.08T static leg** — reproduces the operator-policy ≥10 GiB @128 (10.24); protects when
  measurement is unavailable.
- **cap 0.5T** — the pre-fix pathology (8.0 GiB floor = the ENTIRE 8 GiB box → adaptive_ceiling 0.0,
  training_budget −5.99, admission refuses everything) is now STRUCTURALLY impossible in every
  mode, overrides included. On the M1 the cap binds at 4.0 GiB = the tier's MEASURED safe envelope
  (~4.0–4.4 GiB).

**Overrides:** `--safety-floor-gib` CLI > `TAC_GOV_SAFETY_FLOOR_GIB` env > derived
(`resolve_floor_override_gib`); both clamped to [ABS_MIN, 0.5T] with a loud
`SAFETY-FLOOR CLAMP` log (verified live: env=200 @128 → clamped 64.0, log emitted).
`--safety-floor-mode {derived,fixed}` (fixed = legacy max(8, 0.08T), still cap-clamped).
**Hysteresis:** `SafetyFloorSmoother` — floor rises INSTANTLY, decays ≤0.25 GiB/tick (one persistent
instance per daemon loop; flap test proves raw verdicts flap / smoothed verdicts stable).
**Observability:** every tick's full decomposition (which leg won + all leg values + cap/clamp +
applied smoothed floor) goes into the blackbox sample row (`safety_floor`) and the band ledger row.

## 2. Tier table (exact values; asserted literally in `test_tier_scaled_safety_floor.py`)

| T (GiB) | unavailable | idle (cp=2) | loaded (cp=6) | usable envelope T−floor (loaded) |
|---|---|---|---|---|
| 8   | 2.00 | 3.00 | **4.00** (cap-clamped from 7.00) | **4.0** (>2 ✓; == tertiary measured ~4.0–4.4) |
| 16  | 2.00 | 3.00 | 7.00 | 9.0 |
| 32  | 2.56 | 3.60 | 7.60 | 24.4 |
| 64  | 5.12 | 5.20 | 9.20 | 54.8 |
| 128 | 10.24 | 10.24 | 12.40 | 115.6 |
| 192 | 15.36 | 15.36 | 15.60 | 176.4 |

Invariants tested: monotone in RAM per scenario; ≤0.5T; ≥2.0; **128 GB ⇒ ≥10.24 in EVERY scenario
(≥ the operator-policy 10 — backward-compatible-by-default, verified LIVE below)**; 8 GB loaded ⇒
usable envelope 4.0 > 2. ("Usable budget" here = T − floor, the sizing envelope; the live
`training_budget` additionally subtracts the measured baseline — on the M1 at idle that is honestly
still negative until the OS yields under load, which the tertiary smoke measured it doing.)

**Tier-scaled pressure thresholds** (same headroom physics): `critical = ABS_MIN + cp_headroom(T)`,
`warn = critical + cp_headroom(T)` → @128 (8.4, 14.8) vs legacy (8, 15): critical strictly MORE
protective, warn within 1.4%; @8 (3.0, 4.0) — the legacy 15/8 were above anything an 8 GiB box can
free (permanent-warn bug). Wired into `_govern_tick`, `band_tick`, and the `--governor-tick` CLI
defaults (explicit flags still win).

## 3. Live 128 GB decomposition rows (this box, dry, $0 — #205 untouched)

`--ceiling` (with #205 running): floor **10.24** (winning_leg `static_frac`, clamped false, cap 64.0,
cp_headroom 6.4, measured_cp 0.0*, static_leg 10.24) → ceiling 117.76. **Derived floor ≥ 10 GiB live ✓.**
`--band-tick` (dry): band green, `actual_rss_gib 54.05` true GiB, envelope 108.8,
`pause_scope_rss_gib 54.05`, `efficacy_gap null`, row carries `safety_floor` + `tracked_units:true_gib`.
`--sample-once`: `safety_floor.applied_floor_gib 10.24`, `tracked_units true_gib`, ceiling 117.76.

*Honest note on `measured_cp 0.0`: the known blackbox double-count (mine §7 F2 — the safe_run
custody group AND the trainer pid are both tracked, so Σtracked 108.1 > used 63.3) drives
`baseline→0` while #205 runs, suppressing the measured leg. The floor degrades GRACEFULLY to the
static leg (never below policy). The double-count fix remains a separate owned item (#246 wave);
post preserve-and-stop (single tracked row) the measured leg reads correctly.*

## 4. GB-F1 — admission-path units mix FIXED (units-at-read, single boundary)

`list_tracked_jobs` now converts `memory_guard.group_rss_gb` units → TRUE GiB (×0.95367431640625)
EXACTLY ONCE at the read boundary; `band_tick`'s local conversion removed (double-conversion
guarded by a structural test: the constant appears once in the read path, zero times in band_tick,
and `_mg.group_rss_gb(` has exactly one governor callsite). Downstream — `growth_headroom_gib`,
admission baseline, band comparison, blackbox rows — is all true GiB.
**Verified LIVE: `active_growth_headroom_gib 13.55`** = 67.61 − 54.06 (the review's true-GiB
number; pre-fix 10.9 = a 2.63 GiB ANTI-CONSERVATIVE under-count).
Consumer audit (grep `current_rss_gib|tracked_sum_gib`, non-test): exactly 3 files —
`system_memory_governor.py` (fixed at read), `memory_blackbox.py` (rows now true GiB + a
`tracked_units:"true_gib"` marker), `witness_memory_preflight.py::actual_peak_from_blackbox`
(now MIXED-ERA aware: per-row ×0.9537 only for legacy unmarked rows). No half-converted path left.

## 5. #246 (F1) — pause/resume now cover the FULL process tree

`pause_job`/`resume_job` route through `job_tree_pids`: recursive ppid-descendants of the registered
root (crosses the `start_new_session` boundary the old `killpg` missed — measured pre-fix scope
**0.02 GiB of a 54 GiB run**) plus the root's own pgid members; SIGSTOP deepest-first, SIGCONT in
reverse; idempotent; per-pid race-tolerant; legacy pgid fallback when the sampler is unavailable.
**Untargetable stays absolute:** the set is CONSTRUCTED from the job's own tree (an unregistered
bystander is unreachable by construction) and every member is re-gated (control-plane apps, shell
denylist, the guard + its ancestors) — tested with a mock tree including an in-tree `claude`
process and a 99 GB bystander. Kill-class signals are structurally impossible
(`_signal_job_tree` asserts STOP/CONT only; source-level tests). Band rows now report
`pause_scope = job_tree_rss_gib` — the standing VERIFICATION: **live dry read-only estimate
54.05 ≈ actual 54.05** (gap field null; pre-fix 0.02). The mock bash intermediary is excluded by
the shell denylist (defense-in-depth; a parent bash blocks in wait() — zero memory consequence).

## 6. Continuous band wiring (part 2)

`memory_blackbox.py --daemon` now evaluates `gov.band_tick` IN-PROCESS every `--band-interval`
(default 30 s; `--no-band` opt-out; `--band-envelope-frac`, `--band-run-dir` pass-through).
Semantics are EXACTLY #294: never-kill (closed action vocabulary), defer_to_throttle under system
pressure, red = reversible clean-pause via canonical `pause_job` only sole-workload; actuation
enabled iff the daemon governs (`--no-govern` → dry band rows). Band thresholds auto-scale because
they are fractions of the frac×RAM envelope: 128@0.85 → yellow 92.48 / red 97.92; 8@0.55 → 3.74 /
3.96 (reproduced in tests). **The RUNNING daemons (primary pid 19895 + tertiary) are NOT restarted
by this landing — the wiring benefits the NEXT daemon start**; until then the standalone
`--band-tick` cron/loop remains the live band path.

## 7. Consumers-of-old-constant audit

`DEFAULT_SAFETY_MARGIN_FLOOR_GIB` grep: governor internal (`compute_safety_margin_gib` — now the
legacy/fixed-mode leg only) + `src/tac/canonical_equations/adaptive_ceiling_admission_control_20260703.py`
(a MIRROR copy `SAFETY_MARGIN_FLOOR_GIB=8.0`; no active drift-guard test found). The constant is
KEPT (legacy fixed-mode + mirror compatibility). All ceiling/admission consumers
(`live_admission_decision` ← spawn_durable_daemon `_system_admission_gate` + launch_witness_run +
`witness_memory_preflight.system_aware_admission`; blackbox `sample_once`; CLI `--ceiling/--admit`)
flow through `compute_adaptive_ceiling`, whose default is now the derived floor — one code path,
coherent everywhere. NOTE: the admission-enforce flag is ARMED on this box; the derived floor only
raises protection (10.24 → up to cp-following), never lowers it.

## 8. Tests + verification

- `src/tac/tests/test_tier_scaled_safety_floor.py` — **36 tests**: exact tier matrix ×3 scenarios,
  monotonicity, cap/abs-min bounds, 128⇒≥10, 8GB-loaded usable envelope, winning-leg labels, fixed
  mode, env/CLI override precedence + clamp + loud log, smoother + admission flap (raw flaps /
  smoothed stable), tier pressure thresholds, band auto-scale numbers, defer-to-throttle at edge
  tier, GB-F1 true-GiB admission arithmetic + band/admission agreement + single-boundary structural
  guards, #246 mock-tree walk (session-crossing scope, deepest-first, idempotency, reverse resume,
  bystander/control-plane exclusion, kill-refusal, race tolerance, no-sampler fallback), daemon band
  wiring (cadence, interval, opt-out, apply-follows-govern, error-nonfatal, no-kill source, persistent
  smoother).
- Updated: `test_system_memory_governor.py` (derived-margin + derived-ceiling tests + hermetic env
  fixture), `test_memory_blackbox.py` (new sample fields + derived numbers 14.4/113.6/105.6).
- Suites green: governor+bands+blackbox+waterfill **134 passed / 1 failed** — the failure is the
  KNOWN pre-existing machine-state `test_admission_enforcing_defaults_advisory` (armed
  `admission_enforce.flag` on this box; unchanged). memory_guard + spawn_durable_daemon_memguard +
  waterfill: 86 passed. witness_memory_preflight + projection ledger + launch_witness_run: 71 passed.
- ruff on touched files: back to the HEAD baseline of 8 pre-existing findings (0 new); the known
  RUF046 at HEAD left as-is.

## 9. Honest blockers / follow-ups

1. **Blackbox tracked double-count (mine §7 F2)** suppresses the measured floor leg while a wrapped
   run is live (floor falls back to static ≥ policy — safe, but the dynamical leg is blind on this
   box until the dedup lands). Owned by the #246 wave.
2. **Canonical equation `adaptive_ceiling_admission_control_v1`** still codifies the legacy
   margin law; a v2 registration (derived-floor law) is the APPEND-ONLY follow-up — the mirror
   module was deliberately NOT mutated.
3. **Running daemons carry pre-#298 code until their next restart** (no restarts performed per hard
   rules); the tertiary sweep's operative guards remain safe_run + the standalone band-tick loop
   until its daemon is restarted with `--band-envelope-frac 0.55`.
4. `memory_guard.group_rss_gb`'s misleading `_gb` unit remains upstream (REPORTED; the governor now
   converts at its own read boundary; memory_guard's INTERNAL comparisons are self-consistent).
5. Warn threshold @128 moved 15.0 → 14.8 (derived; −1.4%, warn-pause fires ~0.2 GiB later) while
   critical moved 8.0 → 8.4 (strictly earlier). Documented tradeoff; explicit CLI flags restore any
   fixed values.
