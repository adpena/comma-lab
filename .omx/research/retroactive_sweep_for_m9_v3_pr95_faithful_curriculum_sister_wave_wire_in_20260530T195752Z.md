# Catalog #348 retroactive sweep — M9-v3 PR95-faithful curriculum sister wave wire-in

**Sweep timestamp**: 2026-05-30T19:57:52Z
**Lane**: `lane_m9_v3_pr95_faithful_curriculum_wire_in_substrate_cascade_20260530`
**Wave landing memo**: `feedback_m9_v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530.md`
**Trigger**: canonical 2-landing pattern per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable; this wave lands the canonical wire-in cascade so the m9-v3 helper at commit `c91481212` is reachable from substrate trainers via the canonical opt-in flag.

---

## 1. Bug-class symptom signature

The bug class this wave structurally extincts at the wire-in surface is the **orphan-canonical-helper class**: the m9-v3 `PR95FaithfulCurriculumFactory` + extended `MlxScoreAwareAdapter` landed at commit `c91481212` were complete + tested (26 dedicated factory tests + 67 baseline = 93/93 PASS), but the canonical wire-in into actual substrate trainer dispatch was DEFERRED to a sister wave per the landing memo op-routable #1. Without the wire-in:

- The 3 MLX-first priority substrate trainers (z6_v2 / z8 / dreamer_v3_rssm) cannot opt into the canonical PR95 8-stage curriculum at dispatch time.
- The canonical helper sits unconsumed (orphan-signal-at-canonical-helper class per CLAUDE.md "Results must become system intelligence" non-negotiable).
- The Tier-2 paired-CUDA RATIFICATION dispatch per m9-v3 op-routable #2 cannot fire (no opt-in path from CLI through harness through adapter).

The bug-class signature is: a canonical helper module is checked in + tested + documented, but no production caller in the dispatch path imports + invokes it with the canonical opt-in. Per CLAUDE.md "Subagent coherence-by-default" (anti-fragmentation primitive: a result that does not make the system smarter is incomplete) + "Catalog #336 sister cathedral consumer discovery invoker", this is structurally identical to the cathedral autopilot orphan-signal class.

---

## 2. Pre-fix window

The pre-fix window opens at the m9-v3 canonical helper landing commit `c91481212` (2026-05-30 14:27 -0500) and closes at this wire-in cascade landing (2026-05-30 14:57 -0500). The wire-in cascade is a single-session sister-wave landing per the m9-v3 landing memo's explicit op-routable #1 enumeration ("SISTER WAVE wire pr95_faithful_curriculum_enabled=True into substrate trainers (z6/atw_v2/hnerv-family) per Catalog #270"). The window is ~30 minutes wide; no historical KILL/DEFER/FALSIFY verdicts predate or fall inside this window in any of: `.omx/research/`, `.omx/state/probe_outcomes.jsonl`, `~/.claude/projects/-Users-adpena-Projects-pact/memory/`.

---

## 3. Historical KILL/DEFER/FALSIFY search

Sweep targets:

| Source | Search pattern | Results |
| --- | --- | --- |
| `.omx/research/*.md` | `pr95.*faithful.*curriculum.*KILL\|DEFER\|FALSIFY` | 0 hits |
| `.omx/state/probe_outcomes.jsonl` | `pr95_faithful_curriculum` substring with verdict in `{KILL, DEFER, FALSIFY}` | 0 hits |
| `~/.claude/projects/.../memory/feedback_*.md` | `pr95_faithful_curriculum.*FALSIFIED\|pr95_faithful_curriculum.*KILLED` | 0 hits |
| `.omx/state/canonical_equations_registry.jsonl` | `pr95_faithful_curriculum.*EVENT_DEPRECATED` | 0 hits |

The canonical helper landing itself (`feedback_m9_v3_pr95_faithful_curriculum_scaffold_landed_20260530.md`) is the only relevant historical artifact and it carries a `PROCEED` verdict, not a KILL/DEFER/FALSIFY.

---

## 4. Per-finding RE-EVAL priority assignment

Zero historical findings invalidated. The wave structurally extends the canonical helper landing forward without invalidating any prior verdict; the m9-v3 wire-in cascade is the canonical 2-landing pattern's natural continuation per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.

The OPERATOR-FACING re-eval candidates surfaced by THIS wave (not historical findings, but forward-looking work that this wave enables):

| Candidate | Priority | Notes |
| --- | --- | --- |
| **Tier-2 paired-CUDA RATIFICATION** on z6_v2 + atw_v2 + z8 Modal A100 at 29,650-epoch budget | **OPERATOR_DECIDE** | First PR111-candidate empirical anchor for the curriculum × class-shift compound. Estimated cost: ~$2-5 per substrate. Operator-routable per m9-v3 op-routable #2; this wave's wire-in is the prerequisite. |
| Sister wire-in wave for CUDA/PyTorch substrate trainers (`atw_codec_v2` / `sane_hnerv` / `wyner_ziv_cooperative_receiver`) | **DEFER** | Out of scope for MLX-first wave. Separate Tier-2 lane required because those trainers route through CUDA/PyTorch (no `MlxScoreAwareAdapter`); the curriculum logic would need a sister `PR95FaithfulCurriculumPyTorchAdapter` per UNIQUE-AND-COMPLETE-PER-METHOD operating mode. |
| Auto-recalibration of `pr95_faithful_curriculum_cross_substrate_compounding_savings_v1` canonical equation per Catalog #371 once 3+ Tier-2 empirical anchors land | **AUTO** | Per the canonical recalibration trigger `when_3+_new_empirical_anchors_in_domain`; fires structurally without operator intervention once the Tier-2 anchors land. |

---

## 5. Sister gate impact

This wave is a **wire-in landing**, not a new STRICT preflight gate landing. Catalog #348 sweep is performed because Catalog #348 requires every new gate landing to ship a retroactive sweep; the canonical 2-landing pattern's wire-in surface is structurally equivalent for the purpose of "did any historical finding rely on the absence of this wire-in?". The answer is no: the m9-v3 helper landed today; no historical finding could have predicted its existence + relied on its absence simultaneously.

---

## 6. Cross-references

- m9-v3 canonical helper landing memo: `feedback_m9_v3_pr95_faithful_curriculum_scaffold_landed_20260530.md`
- m9-v3 wire-in cascade landing memo (THIS wave): `feedback_m9_v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530.md`
- Canonical equation: `pr95_faithful_curriculum_cross_substrate_compounding_savings_v1` (registered THIS wave with 1 structural EmpiricalAnchor)
- Probe outcome ledger entry: `m9v3_pr95_faithful_curriculum_sister_wave_wire_in_z6v2_z8_dreamer_20260530` (verdict PROCEED advisory 14-day staleness)
- Council anchor: `council_t1_m9v3_pr95_faithful_curriculum_sister_wave_wire_in_landed_20260530` (T1 PROCEED 6-voice)
- CLAUDE.md anchors: "HNeRV / leaderboard-implementation parity discipline" L14 + L15; "Bugs must be permanently fixed AND self-protected against"; "Results must become system intelligence"; "Subagent coherence-by-default" (canonical 2-landing pattern); "Optimize + iterate as we go" 5-invariant standing directive.

---

## 7. Verification

| Check | Status |
| --- | --- |
| Sweep targets enumerated (`.omx/research/`, `.omx/state/probe_outcomes.jsonl`, Claude memory, canonical_equations_registry) | DONE |
| Historical KILL/DEFER/FALSIFY hits inventoried | 0 hits across all sources |
| Per-finding RE-EVAL priority assigned | 0 historical findings; 3 forward-looking operator-routables surfaced |
| Sister gate impact documented | N/A (this is a wire-in landing, not a STRICT gate landing); Catalog #348 sweep emitted for completeness per canonical 2-landing pattern |
| Memo dated, lane-referenced, Catalog #348-conformant | YES |
