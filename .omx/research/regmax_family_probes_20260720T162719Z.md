# Regularized-max family probes — 2026-07-20

**Lane:** `regmax_family_probes` · `research_only=true` · local `$0` · no training · no paid dispatch · no evaluator score

**Authority:** delegated probe execution against the three formulations preregistered in
`09cc0026cb:.omx/research/erm_2607_10128_crosswalk_20260720T154953Z.md`; no redesign and no live-config wiring.

**Axis:** `[macOS-CPU advisory]` for target-cache arithmetic. Frozen CPU Torch is reserved as terminal
`HARD_ACCEPT` authority after decoded uint8 parse-back; it was not invoked because no receiver candidate existed.

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**. No score, promotion, or rank claim.

## Verdict

**`N-A / N-A / N-A — MATCHED RECEIVER SURFACES ABSENT`.** The real full-n600 target-side sparsemax A/B
was measured, but the current branch has no typed logits-to-uint8 preimage adapter. It also has neither a
frozen rank-4 valid-cell prototype artifact nor an executable Aurenhammer min-generator same-coder
comparator/principal-cell fixture. Inventing any of those inside this arm would change the as-written
treatments. Consequently none of the three preregistered falsifiers fired, and no family negative follows.

## Input custody and reproducer

The tool read the same real n600 logits and hard targets as #542:

| input | shape / bytes | SHA-256 |
|---|---:|---|
| `gt_segnet_logits.f16` | `(600,5,384,512)` / `1,179,648,000` | `41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52` |
| `gt_segnet_argmax.u8` | `(600,384,512)` / `117,964,800` | `36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68` |

Exact command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
tools/probe_regmax_family.py \
  --output-dir .omx/research/regmax_family_probes_20260720T162719Z \
  --timestamp 2026-07-20T16:27:19Z
```

Runtime was `13.0798502499 s` under Python `3.13.12`, NumPy `1.26.4`, macOS arm64. Manifest SHA-256:
`6fd8218c2a501fb94e8ca07a64bada85e054565c39f478f989c7966aa2f0e733`.

## Probe 1 — sparsemax margin-band preimage A/B

**Standalone verdict: `N-A_MISSING_MATCHED_PREIMAGE_ADAPTER`.**

The target-side comparison is **MEASURED** on all `117,964,800` pixels at the preregistered unit scale.
Target debt is reported transparently as `mean(1-p[target_hard_label])`; it is a prediction-map diagnostic,
not a score or an acceptance rule.

| quantity | entropy / Cole–Hopf | sparsemax |
|---|---:|---:|
| mean support size | `5.0` | `1.02680106269` |
| exact-one-hot fraction | `0` | **`0.973330917358`** |
| mean target debt | `0.0223443410092` | `0.00684220670265` |
| high-margin interior debt (`margin>=1`) | `0.0123438090130` | `0.0` |
| annulus/tie debt (`margin<1`) | `0.387329785492` | `0.256559507299` |

The exact-one-hot fraction differs from the preregistered `0.9733` prediction by only
`3.09173584e-5`. The independent self-review re-counted `114,818,787` pixels with fp16
`top1-top2>=1`, exactly matching the sparsemax one-hot count. The fp16 logits disagree with the hard-label
cache at `2,629` pixels, preserved as a custody diagnostic rather than coerced away.

Per-hard-target-class sparsemax exact-one-hot fractions are Road `0.9437847823`, Lane
`0.2576483517`, Undrivable `0.9915382086`, Movable `0.8232277062`, and MyCar `0.9886576463`.
This sharpens the preregistered Lane exception: aggregate 97.33% exactness is not a per-class guarantee.

**Preregistered falsifier (not evaluated):** falsified if the matched sparsemax arm fails to improve hard
accepts, exact-oracle calls, or bytes versus entropy/Cole–Hopf on identical cells and budget.
`hard_accepts=null`, `exact_oracle_calls=null`, and `candidate_bytes_same_coder=null` are therefore the only
honest terminal fields. The target-side debt advantage is not an adoption result.

Receipt SHA-256: `fbe2210af7086a2b52413255d65159600c8eb75b2fb898604a175166c6ad92ae`.

## Probe 2 — tropical principal-cell representative

**Standalone verdict: `N-A_NO_MATCHED_PRINCIPAL_CELL_AND_AURENHAMMER_COMPARATOR`.**

The exact rank-4 head law exists, but the branch does not contain the as-written `A,b` principal-cell
fixture or an executable Aurenhammer min-generator comparator under the same coder. “Aurenhammer adopted”
in the survey/crosswalk is a design disposition, not a landed callable or a byte receipt. Constructing a new
LP/fixture would be redesign, so no principal vector, cell identity, gauge bytes, or comparator bytes were
manufactured.

**Preregistered falsifier (not evaluated):** falsified if the gauge-fixed principal representative changes
the hard cell, is longer after the same coder, or requires uncounted state.

Receipt SHA-256: `4f9c6cef578c9cb589879ad2b3aa4974537f1d1468fb7dffafd72401edd91ab2`.

## Probe 3 — entropy/Hopfield pre-prox before uint8

**Standalone verdict: `N-A_NO_FROZEN_PROTOTYPES_OR_MATCHED_PREIMAGE`.**

No frozen rank-4 valid-cell prototype artifact is registered or implemented on this branch, and the same
logits-to-uint8 preimage adapter needed by probe 1 is absent. Deriving prototypes from the test population
would train/select the treatment after seeing the measurement and violate the frozen-prototype A/B.

**Preregistered falsifier (not evaluated):** falsified if one frozen-prototype memory-prox step does not
improve hard-accept count or exact-call cost versus no-prox on identical cells and budget.

Receipt SHA-256: `c0bf63effc60dd2094208760707abc7c42526d6a6c04359e8ad04c4f95a65330`.

## One bounded self-review

The single post-measurement pass was **CLEAN**. It independently counted unit-margin pixels without calling
the probe implementation, canonical-reencoded all receipts, rechecked receipt hashes, asserted all three
verdicts are `N-A`, asserted every falsifier remains unevaluated, and asserted hard accepts/oracle calls/bytes
remain null. Focused tests: `4 passed`; ruff: clean. Receipt:
`.omx/research/regmax_family_probes_20260720T162719Z/self_review.json`.

## Triality, system intelligence, and routing

- **DAG:** `.omx/research/regmax_family_probes_DAG_FEED_20260720T162719Z.md` records the three fail-closed
  edges and exact reopen requirements.
- **DSL:** N-A with rationale. No probe adopted, so no loss/config lever is authorized or wired.
- **Equations:** N-A with rationale. Sparsemax's known simplex law gained a measured target-population anchor,
  but no receiver-closed terminal law was measured; no canonical equation or posterior row was registered.
- **Lane:** `regmax_family_probes` remains L0 `research_only=true`; target-cache characterization is not a
  real-archive empirical gate.
- **Future consumer names are preserved, not actuated:** a receiver-positive sparsemax result would route to
  the band-slack/annulus loss config; tropical to the PDW/quotient representative coder; entropy prox to the
  R1 `d_B` pre-step. MAIN must first supply and independently review the missing matched surfaces.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; current project-memory top entries; latest sister Codex/Claude memos; `reports/latest.md`;
`.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/canonical_equations_registry.jsonl`;
the preregistration at commit `09cc0026cb`; `alternative_forms_conv_wall_20260718.md`; the exact rank-4 head law;
the bounded uint8 lattice solver/tool/receipt; the generator-description survey/crux synthesis; source searches
under `src/`, `tools/`, and `tests/`; the per-arm and broadcast inboxes. The 2026-07-19 EV/Fisher directives were
consumed as custody constraints; this arm did not introduce a residual basis or a naive realization proxy.

## MAIN landing requirement

This isolated branch is not authority. MAIN must review base-to-head, confirm that no newer implementation
supplies the three missing matched surfaces, verify the raw-input hashes and target-debt definition, and keep
the terminal verdicts `N-A` unless it reruns the exact preregistered receiver comparisons. The pointer remains
unchanged.
