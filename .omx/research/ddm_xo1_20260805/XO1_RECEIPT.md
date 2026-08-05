# XO1 Receipt - EU2-X1-10K Context Orderer

## Answer First

XO1 is **NO-GO on the executed control leg**. The 201 B int8 named-feature head parsed cleanly and had strong held-out flip discrimination, but it did **not** buy token/context bytes under the current token coder. Reordering the current IX2 token lattice by the control head made the token frame **11,561 B larger**:

| row | packet B | coder | token frame B | delta vs current | rank/order result | verdict |
|---|---:|---|---:|---:|---|---|
| current order | 0 | IX2 token frame L16 | 341,295 | 0 | baseline | keep |
| XO1 control head order | 201 | IX2 token frame L16 | 352,856 | **+11,561 B** | pair Spearman vs GP1 flips 0.6068 | **NO-GO control** |
| oracle flip pair order | 0 | IX2 token frame L16 | 353,133 | +11,838 B | upper control comparator | also worse |
| public comma10k 10K student | not built | not run | n/a | n/a | GitHub DNS failed in sandbox | DATA-BLOCKED |

Pre-registered bar result: packet <=15,000 B passed, but recovered **0.0%** of GP1's 106,954 B ordering gap and saved **-11,561 token/context bytes**. That is below WEAK-GO. No 50K escalation is authorized.

Axis: `[macOS-CPU scorer-free cached-ordering byte-only]`. Score claim: false. Scorer forwards: 0. `upstream/evaluate.py`: not run. Contest pointer: borrowed/unmoved.

## Measurement

Command:

```sh
PYTHONPATH=src .venv/bin/python experiments/ddm_xo1_context_orderer.py \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_xo1_20260805 \
  --n-pairs 600
```

Artifacts:

| artifact | bytes | sha256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_xo1_20260805/xo1_control_head.xo1pkt` | 201 | `e2a48e785e05615ddad913eb2d8673452c1b748eccf7c2be66f04ea0b0f5a450` |
| `/Volumes/VertigoDataTier/pact/ddm_xo1_20260805/xo1_control_measurement.json` | 6,693 | `df8ff4f3d40c58e26153f5bd967cae64548e1eba333a8c47bbf66cdcbfa8861c` |
| `/Volumes/VertigoDataTier/pact/ddm_xo1_20260805/public_data_attempt/comma10k_fetch_absence.json` | 788 | `24d9a16c47771876ad9ad0a101aec263545f26c3fd8011529b28205c49d821b7` |

Input custody recorded in the JSON includes `gt_argmax_n600.npy`, `cx1_argmax_n600.npy`, `cx1_tokens.npy`, `gp1_per_pair_flips.npy`, and the QA75 margin-field directory.

## Control Details

The control is an int8 additive named-feature logistic/Rudin-style head with receiver-available feature names:

- distance-to-boundary bucket
- receiver class
- nearest differing class or interior
- row bucket
- column bucket
- token activity bucket

Training used even pairs; held-out reporting used odd pairs. Denominators:

| split | samples | positive rate |
|---|---:|---:|
| train, even pairs | 731,682 | 0.3450 |
| held-out, odd pairs | 480,000 | 0.4754 |

Held-out rank/agreement:

| metric | value |
|---|---:|
| flip AUC | 0.9807988148549549 |
| Spearman vs negative frozen-margin field | 0.762053333826635 |
| top-2% overlap vs margin-priority sample | 349 / 9,600 = 0.036354 |
| top-5% overlap vs margin-priority sample | 2,161 / 24,000 = 0.090042 |

Interpretation: the receiver-feature head separates flips from ordinary pixels, but it does not recover the low-margin oracle ordering that GP1 priced as the 106,954 B R5 gap. On the current token stream, the order it induces is actively anti-rate for IX2.

## Parse-Back Proof

Packed section: `XO1CTL1` version 1, 6 feature blocks, 52 int8 weights plus quantized bias. Parse-back result:

| field | value |
|---|---:|
| packet bytes | 201 |
| final cursor | 201 |
| consumed exactly once | true |
| weights sha256 | `aa7f7f2a061f9d8e219235ae3cd85e75a95542c4d3ab0bc44b41092a0181d2ad` |

Token coder parse-back also closed: current order, control order, and oracle-flip order all round-tripped through `src.tac.optimization.ddm_ix2_archive_container.encode_token_frame(levels=16)` / `decode_token_frame`, and inverse-order restoration matched the original token lattice.

## Public Data Attempt

Bounded command:

```sh
GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 \
  ls-remote https://github.com/commaai/comma10k.git
```

Result: exit 128, `Could not resolve host: github.com`. This is a sandbox network/DNS blocker, not evidence about comma10k data quality. No public-data student was trained, no torch training smoke ran, and no external dataset was fetched.

## RECALL EVIDENCE

| source searched | finding beyond charter seeds | plan impact |
|---|---|---|
| `.omx/tmp/codex_runs/xo1_prompt.md`, `_common_contract.md` | XO1 forbids scorer/eval/launch work and requires packed bytes, rank-agreement JSON, same-coder byte delta, parse-back proof, and serializer landing. | Kept the run scorer-free and made the packet/JSON/receipt the durable outputs. |
| `.omx/research/ddm_eu2_20260805/EU2_RECEIPT.md` | The first experiment is exactly EU2-X1-10K-context-orderer; GO/WEAK/NO-GO bars are pre-registered. | Used the 15,000 B, 30% gap, and 30,000 B saved thresholds without redesign. |
| `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md`, `.omx/state/main_hot_state.md` | LC1 demoted PE3 to conditioning-only and crowned TR1 learned carrier primary, firing XO1 rank-1. | Treated XO1 as the current rate lever but not a scorer-slot owner. |
| `.omx/research/ddm_gp1_selective_gt_student_pricing_20260803.md`, `/Volumes/VertigoDataTier/pact/ddm_gp1_20260803/gp1_pass*.json` | GP1's student value is the 106,954 B free-to-oracle ordering gap; margin fields are frozen-scorer oracle surfaces. | Evaluated rank agreement against QA75 margins and computed recovered fraction against 106,954 B. |
| `.omx/research/ddm_pa2_20260805/PA2_RECEIPT.md` | PA2's shared-context win is 3,975 B on the n32 persisted stream, modest and scorer-free. | Used PA2 as the floor XO1 needed to beat; XO1's -11,561 B result is below that floor. |
| `experiments/ddm_ob1_ordering_ceiling_n600.py` | Receiver-legal distance/class/edge features can be priced as conditional ordering features without scorers. | Chose the control's named feature set from the existing free-feature law rather than inventing a dense field. |
| `src/tac/optimization/ddm_ix2_archive_container.py`, `.omx/research/ddm_qo1_repair_stream_optimal_form_20260804.json` | The current `sub_auto_pairbit` token bulk is IX2 `encode_token_frame` at 341,295 B, not the older 346,478 B SMEVR section. | Measured byte delta under the current IX2 coder surface. |
| Canonical equations registry via `tools/list_canonical_equations.py --json` filtered mentally for token/context/orderer/student surfaces | Found adjacent entropy/coder laws, but no direct measured comma10k/openpilot micro-student law overriding EU2. | Kept the public-student leg empirical and blocked on fetch rather than importing a law from another surface. |

I did not find, in the searched scopes above, a measured current-vehicle comma10k/openpilot 10K public-data student result.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| XO1 control head as token/context orderer | FOLDED | Do not route to ship or 50K; it worsens the current IX2 token stream by 11,561 B. |
| 50K escalation | FOLDED | EU2 allows 50K only after GO-10K plus slope >=1.2 saved B per added counted B; XO1 measured neither. |
| public comma10k 10K student | QUEUED-WITH-FIRE-ORDER only in a network-enabled environment | Fetch <=2,000 public comma10k images/masks to SSD with SHAs, train <=10K CPU-torch int8 smoke under 30 min, then rerun the same packed-byte, rank-agreement, same-coder byte-delta, and parse-back gates. No scorer slot. |

## NEXT_IF_RESUMED

```json
{
  "run_id": "ddm_xo1_20260805",
  "status": "NO_GO_CONTROL_ONLY_DATA_BLOCKED_STUDENT",
  "axis": "[macOS-CPU scorer-free cached-ordering byte-only]",
  "scorer_forwards": 0,
  "evaluate_py": false,
  "control_packet_bytes": 201,
  "control_packet_sha256": "e2a48e785e05615ddad913eb2d8673452c1b748eccf7c2be66f04ea0b0f5a450",
  "heldout_flip_auc": 0.9807988148549549,
  "heldout_spearman_vs_negative_margin": 0.762053333826635,
  "token_coder": "IX2 encode_token_frame(levels=16)",
  "baseline_token_bytes": 341295,
  "control_order_token_bytes": 352856,
  "control_token_context_saved_bytes": -11561,
  "gp1_ordering_gap_recovered_fraction": 0.0,
  "go_nogo": "NO_GO_CONTROL_ONLY",
  "public_data_student": {
    "status": "NOT_RUN_DATA_UNAVAILABLE",
    "fetch_exit_code": 128,
    "blocker": "github.com DNS resolution failed in sandbox"
  },
  "do_not_do": [
    "Do not escalate XO1 to 50K from this result.",
    "Do not claim a score, scorer result, or frontier move.",
    "Do not rerun the control as a scorer job; the measured blocker is byte economics."
  ]
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
