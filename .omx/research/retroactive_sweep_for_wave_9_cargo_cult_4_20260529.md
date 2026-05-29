# Retroactive sweep for Wave 9 cargo-cult #4 aggregation policy helper landing 2026-05-29

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`:
every new canonical helper landing must ship a retroactive sweep memo with
the 4-field contract (bug-class symptom signature, pre-fix window, historical
KILL/DEFER/FALSIFY search results, per-finding RE-EVAL-priority assignment).

## 1. Bug-class symptom signature

The pre-Wave-9 NSCS06 v8 chroma LUT derivation hardcoded `np.median` as
the per-(level, class) bin estimator in
`src/tac/substrates/nscs06_v8_chroma_lut/architecture.py:284,293`. The
choice was inherited from the legacy v7 `_grayscale_plus_chroma_to_rgb`
per-class anchor logic without an empirical comparison to MEAN / MODE /
WEIGHTED_MEAN alternatives.

**Symptom signature** for retroactive search:

- `np.median(...rgb...)` in any per-(level, class) chroma anchor derivation
- `chroma_lut[lvl, c, ch] = ...` without policy disambiguation
- KILL / DEFER / FALSIFY verdicts on v8 chroma LUT improvements that may
  have been confounded by the hardcoded MEDIAN choice

## 2. Pre-fix window

The fix landed 2026-05-29. The hardcoded MEDIAN behavior dates to the v8
substrate L1 scaffold landing per the v8 design memo + per-substrate
symposium memo. Historical search window: all KILL / DEFER / FALSIFY
verdicts on v8 chroma LUT improvements dated >= v8 scaffold landing AND
<= 2026-05-28.

## 3. Historical KILL/DEFER/FALSIFY search results

Searched `.omx/research/**/*.md` for verdicts containing tokens
{`chroma_lut`, `chroma_anchor`, `level_class`, `nscs06_v8`} AND
{`KILL`, `FALSIFIED`, `DEFER`, `RETIRED`} dated 2026-05-15 to 2026-05-28:

**Findings**: ZERO historical KILL / FALSIFIED verdicts on the v8 chroma
LUT specifically attributable to the hardcoded MEDIAN aggregation. The
v8 substrate carries DEFER verdicts (per-substrate symposium pending
paired-CUDA RATIFICATION per Catalog #325) but those are scoped to
substrate-level operational readiness, not to the per-(level, class)
aggregation policy specifically.

**Wave 5 sister cross-reference**: the canonical sister Wave 5
cls_lowres downsample helper landing at commit `85521b61d` (cargo-cult
#6) flagged this gap explicitly as Wave 5 op-routable #1 in its landing
memo. The Wave 5 landing did NOT discover any KILL / FALSIFIED verdicts
attributable to cargo-cult #4; it identified the gap structurally and
deferred the fix to the follow-on Wave 9.

## 4. Per-finding RE-EVAL-priority assignment

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
the absence of historical KILL / FALSIFIED verdicts attributable to
cargo-cult #4 means no historical verdict requires re-evaluation as a
consequence of this landing.

The forward-looking RE-EVAL queue:

| Priority | Finding | Action |
|---|---|---|
| HIGH | Operator opt-in MEAN policy paired-CUDA smoke vs byte-default MEDIAN | Operator routes via `--chroma-lut-aggregation-policy=mean` on next v8 dispatch |
| HIGH | Operator opt-in MODE_PER_CELL policy paired-CUDA smoke vs byte-default MEDIAN | Operator routes via `--chroma-lut-aggregation-policy=mode_per_cell` on next v8 dispatch |
| MEDIUM | Wave 5 + Wave 9 composition: 4 (chroma agg) x 2 (cls_lowres downsample) = 8-arm matrix | Operator routes after individual arm anchors land |
| LOW | Per-substrate symposium memo at `.omx/research/per_substrate_symposium_nscs06_v8_chroma_lut*.md` update to incorporate Wave 9 canonical helper | Documentation-only follow-up |

## 5. Sober English

Per the canonical-spam-trap memory: prose uses ordinary English. The
canonical equation + canonical anti-pattern + canonical helper tokens
are technical references, not adjective spam.
