# Instruction And State Anti-Conservatism Audit

Date: 2026-05-13  
Author: Codex  
Repo: `/Users/adpena/Projects/pact`  
Branch: `main`  
Score claim: false  
Promotion eligible: false  

## Operator Correction

The operator correctly identified a process failure: the system was still
behaving too conservatively after explicit direction for aggressive
score-lowering, no holds barred, no local-minimum trapping, funded reproduction,
and no meaningful budget ceiling. The failure mode was not lack of ideas. It
was allowing review, guardrails, and stale cost caps to become reasons to defer
timing smokes and long-burn representation campaigns.

## Current State Checked

- Branch is `main`.
- `.omx/state/RACE_MODE_ACTIVE.flag` exists.
- `.gitignore` already contains `__pycache__/`, `*.pyc`, `.pytest_cache/`,
  `.mypy_cache/`, and `.ruff_cache/`; pycache files are already ignored.
- `AGENTS.md` already had `Frontier Velocity And Anti-Conservatism`, but it did
  not force a long-burn campaign/timing-smoke decision.
- `CLAUDE.md` already had race-mode inversion and HNeRV parity discipline, but
  old `$25/$24` budget language could be read as a stale hard stop after a newer
  funded-campaign directive.
- Recent HYBRID workers produced promising non-score advisory signals:
  SABOR boundary stability and S2SBS blindspot capacity are `GO-FOR-PROTOTYPE`,
  while PR95 curriculum recovery is `READY-TO-WIRE, not ready-to-dispatch`.

## Harmful Biases Found

1. **Budget caps were over-generalized.** Old default GPU caps protected against
   accidental spend, but could override a newer operator statement that funding
   is available. This biases agents toward cheap meta-work.

2. **Archive-grammar gates were too easy to misread.** Missing byte-closed
   archive grammar must block promotion and score claims, but it should not
   block a non-promotional timing smoke or source-faithful reproduction probe.

3. **`research_only=true` became a parking lot.** A frontier-relevant
   research-only artifact must name the next byte-closed prototype, timing
   smoke, or blocker. Otherwise it silently becomes orphan signal.

4. **Lanes were not campaigns.** NeRV, RAFT/ego-motion, SIREN, foveation,
   Ballé/Cool-Chic/C3, SABOR, and S2SBS were often treated as lane artifacts
   instead of managed campaigns with cost telemetry, checkpoints, harvest, and
   stop/continue thresholds.

5. **Meta-review could outrank actuation.** With race mode active, additional
   council text should be subordinate to launchable commands, actuator
   hardening, or concrete blockers.

## Durable Protocol Changes Landed

Patched `AGENTS.md`:

- Added `Long-Burn Campaign Default -- NON-NEGOTIABLE`.
- Defined a campaign as lane id, evidence, timing smoke, full-run command,
  cost model, byte-closed plan, and stop gates.
- Made budget uncertainty a timing-smoke trigger, not a blocker.
- Clarified stale caps/no-GPU memos are superseded by newer explicit funded
  campaign directives while claim/custody/compliance remain mandatory.
- Required high-EV directions to become a same-session campaign ledger plus
  timing-smoke/launch decision or a written blocker.

Patched `CLAUDE.md`:

- Added the same long-burn score-lowering campaign default at the highest
  emphasis tier.
- Added a legacy-budget supersession note under GPU budget caps.
- Preserved claim lifecycle, provider probes, artifact custody, contest
  compliance, and CPU/CUDA axis separation.

## PR95 Curriculum Clarification

Curriculum means the staged optimization protocol, not just epoch count. For
PR95, the recovered eight-stage schedule includes:

- Stage 1 CE segmentation warmup;
- Stage 2 tau-softplus segmentation;
- Stage 3 smooth-disagreement segmentation;
- Stage 4 QAT join;
- Stages 5-7 C1a entropy regularization and lambda/sigma sweeps;
- Stage 8 Muon finetune over hidden 2D+ weights while AdamW keeps stem/RGB,
  bias/1D parameters, and latents.

The source totals `29650` epochs. Because the dataset is one fixed video with
about 600 frame pairs, this is many small optimizer steps rather than a normal
large-dataset epoch regime.

## How To Beat PR95 Instead Of Merely Reproduce It

The next campaign should recover PR95 faithfully, then improve it in ways that
preserve byte-closed contest authority:

1. Run a PR95 timing smoke to measure seconds/epoch on the chosen provider.
2. Port PR95 architecture, losses, QAT, C1a, differentiable YUV6, eval
   roundtrip, EMA archive selection, Muon partition, and archive parse/build
   into canonical `tac`/`experiments` surfaces.
3. Add PR101-grade microcodec export over improved weights.
4. Run Stage-8-only engineering probes only as wiring smoke unless an f32 Stage
   7 handoff is recovered.
5. Search curriculum mutations: earlier Muon, dual EMA, lambda/sigma grids,
   hard-pair sampling, score-domain early stopping on the exported archive,
   quantization-native training, and pair/category water-filling.
6. In parallel, build byte-closed residual prototypes for SABOR, S2SBS,
   SIREN/FINER/WIRE, wavelets, LA-pose/telescope foveation, and scorer-inverse
   perturbations.

## Immediate Action Standard

For any frontier-relevant direction with plausible `>0.01` score movement, the
next turn should produce one of:

- timing-smoke result;
- claimed dispatch;
- byte-closed prototype;
- exact-eval candidate;
- hardened actuator;
- or a dated blocker with launch criteria.

More research is useful only when it changes one of those artifacts.
