# 01_dseg_aware_taper — revision_claude_20260719

Verdict: `CANNOT-RESOLVE(ep725-fork) / RE-SCOPED-TO-FRESH-RUN (fresh pair full-compile BLOCKED by pre-existing V9-432 LawRef defect)`

Measured at HEAD ``. Containment: compile/static evidence only — NO launch occurred; pointer 0.1910828242 [contest-CPU] UNMOVED (MEANS).

```json
{
  "ticket": "01_dseg_aware_taper",
  "verdict": "CANNOT-RESOLVE(ep725-fork) / RE-SCOPED-TO-FRESH-RUN (fresh pair full-compile BLOCKED by pre-existing V9-432 LawRef defect)",
  "adjudication": {
    "question": "is removing a structural epoch-0 lever after an ep725 warm start a valid contrast?",
    "answer": "NO. (a) DsegAwareTaper is STRUCTURAL (its own factory docstring: 'active from ep0 by construction \u2014 it changes the input feats the in_proj is trained on'; the trainer's F2 resume-divergence guard REFUSES adding/changing it on resume as a basis change); (b) the ep725 trunk (mod32cap->v9c2 lineage) was trained WITHOUT the taper \u2014 the canonical ON control (taper-trained trunk) does not exist at ep725; adding the taper at a fork would measure the add-shock of reweighting features the in_proj never trained under, NOT the lever's value; (c) --warm-start-weights-only auto-allows lever drift, so the trainer would not even refuse \u2014 the contrast would run and be silently confounded. Charter path taken: 'emit the honest verdict that this lever NEEDS a fresh-run arm'.",
    "re_scope": "the canonical contrast IS the existing fresh mod19/3000-ep ISO pair: v9_cgauge_432 base (taper ON control) vs compile_v9_cgauge_432_taper_off_launch_config (whole-Lever removal), duty 78.9%, one-lever delta verified below."
  },
  "iso_contract": {
    "one_lever_delta": true,
    "argv_diff": {
      "--dseg-aware-taper": [
        null,
        "<ABSENT>"
      ],
      "--dseg-aware-taper-floor": [
        "0.05",
        "<ABSENT>"
      ],
      "--dseg-aware-taper-scale": [
        "0.0",
        "<ABSENT>"
      ],
      "--dseg-aware-taper-strength": [
        "1.0",
        "<ABSENT>"
      ]
    },
    "control_config": "v9_cgauge_ideal_mod19",
    "config_id": "v9_cgauge_432_taper_off",
    "duty_to_measure_percent": 78.9
  },
  "arms": {
    "control_on": {
      "program": "v9_cgauge_432",
      "typed_config_hash": "bfaa639a6af1e8a29b5c2ecb42d1cc17d4c7682a187c614436ef35385984e26c",
      "full_dsl_compile_hash": null,
      "full_compile_blocker": "V9ProvenanceGateError: self-recompile of DSL compile binding failed: LawRef compiled value for 'hosc_beta_end' differs from WitnessProgram flag --hosc-beta-end: 10.0 != 3.177"
    },
    "treatment_off": {
      "program": "v9_cgauge_432_taper_off",
      "typed_config_hash": "0ea55dfa63f9f713064a7f3f1dd13c0fce8a052386d6ec010c1ad008942fdcc5",
      "full_dsl_compile_hash": null,
      "full_compile_blocker": "V9ProvenanceGateError: self-recompile of DSL compile binding failed: LawRef equation recompute differs for 'hosc_beta_end': 10.0 != 3.177"
    }
  },
  "named_blocker": {
    "id": "V9_432_HOSC_BETA_END_LAWREF_RECOMPUTE_DEFECT",
    "detail": "build_dsl_compile_provenance_document self-recompile refuses BOTH 432-family arms: LawRef equation recompute yields hosc_beta_end 10.0 while the config emits 3.177 (the CLAUDE.md 2026-07-15 reconciliation's OWED LawRef/compiler-record debt, surfaced live here). Pre-existing, NOT introduced by this revision; the fresh pair cannot carry a full_dsl_compile_hash until that custody row is repaired at its own surface.",
    "measured_at_head": "679e78ab0352d7a1efbcf2368d49c86f7360d000"
  },
  "confound_note": "fresh from-scratch arms: seeds/data-order identical by config; #518 fork machinery N/A (no resume); thresholds for a 3000-ep fresh pair must be re-derived from ITS lineage verdict noise at fire time (the ep725-fork thresholds do not transfer)."
}
```

