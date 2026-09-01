---
schema: ddm_qx3_receiver_closure_memo.v1
date: 2026-08-31
arm: ddm_qx3_receiver_closure
status: BLOCKED
axis: "[scorer-free exact receiver/rate measurement]"
score_claim: false
pointer_moved: false
selection_mode: full_n600
custody: /Volumes/APDataStore/pact/ddm_qx3
---

# QX3 receiver closure — exact receiver works, but the decoder baseline misses 1,669,798 sites and the cheapest measured closure is 486,311 B over cap

## Conclusion

**Verdict: `BLOCKED`.** QX1's decoded core does not reproduce the baseline on
which QX2's 22,661-byte event section is conditioned. The exact QX1/QBT
receiver differs from QX2's retained C1 baseline at **1,669,798 / 117,964,800
sites (1.4155053%)** and at **9,619 / 17,926 QX2 event sites (53.659489%)**.
Pure decoder derivation at zero bytes is therefore falsified for this exact
QX1 core and QX2 baseline.

I built two counted correction forms, joined each with QX2's exact event object
inside one QXE section, raced Brotli q11, LZMA-9-extreme, and zlib-9 with
deterministic repeats, and decoded the best complete archive twice. The best
section is dense four-way delta + QX2 events under LZMA at **510,404 B**, versus
the exact **24,093 B** section cap. Its complete archive is **624,296 B**, or
**486,311 B over** the strict 137,985-byte maximum. Both receiver passes
reconstructed all 17,926 events and the exact 117,964,800-byte target, but the
rate gate failed.

Named blocker:
`QX1_QBT_BASELINE_DIFF_REQUIRES_OVER_CAP_COUNTED_CORRECTION` — the missing
input is a decoder-available statistic that reproduces QX2's GT-derived C1
conditioning baseline. QX1 carries an approximate QBT partition field, not
that exact baseline.

## Stage 0 — exact source boundary

QX2 constructs its conditioning field in
`experiments/ddm_qx2_events_section_redesign.py:build_baseline_and_stats` as
follows for every pair:

1. read `lstars` from the pinned 5,078,017,610-byte GT cache;
2. copy that 384x512 label raster;
3. at each of the 17,926 retained S2 event sites, replace the target class with
   the event's `baseline_class`.

The resulting encoder-side field is
`/Volumes/APDataStore/pact/ddm_qx2/retained/baseline/c1_baseline_labels.u8`,
117,964,800 B, SHA-256
`02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34`.
Both inputs are encoder-only relative to QX1's seven-section core: the core
contains QBT config, 42 quantized model tensors split across three QXT role
groups, latent metadata, 600 joint latent records, and a pose stream. It does
not contain the GT `lstars` raster or the S2 event section.

QX3 reassembled the three role groups by preserving each quantized tensor
record and restoring QBT's global tensor-name order. The result is byte-exact
to the retained ancestor model: 87,854 B, SHA-256
`2280c2d3c54d1781559ec130123a05ec664dbdf347b04f379805bfbe67f59085`.
The fresh all-n600 decode is also exact to QBZ1's retained quantized-packet
native field: **0 / 117,964,800 mismatches**. Thus the Stage-1 mismatch is not a
QX3 receiver implementation error.

Input custody and complete trace are retained in
`/Volumes/APDataStore/pact/ddm_qx3/checkpoints/STAGE0_INPUT_TRACE.json`.

## Stage 1 — bit comparison

| comparison | denominator | mismatches | mismatch fraction | exact |
|---|---:|---:|---:|---|
| fresh QX1 decoder vs retained QBZ1 exact-packet native field | 117,964,800 sites | 0 | 0 | yes |
| QX1 decoder baseline vs QX2 conditioning baseline | 117,964,800 sites | 1,669,798 | 0.0141550530 | no |
| same comparison restricted to QX2 event sites | 17,926 events | 9,619 | 0.5365948901 | no |

Every one of the 600 pairs has at least 1,442 mismatches; the mean is
2,782.997 and the maximum is 9,587. The mismatch is therefore not a small
event-site exception table. The largest transition counts are `0->1` 672,737,
`3->0` 168,252, `2->0` 162,453, `0->2` 135,872, and `3->2` 132,803.

An ideal exchangeable multinomial assignment length, conditional on the
decoded source labels and the observed five-way row counts, is 12,747,362.74
bits or **1,593,421 B**. This is a reference, not a universal information lower
bound: it does not rule out a structurally different procedural model. It does
show why the 1,432-byte slack is not a plausible ordinary correction stream.

Receipt:
`/Volumes/APDataStore/pact/ddm_qx3/checkpoints/STAGE1_BIT_COMPARE.json`.

## Stage 2 — counted closure race

The two exact correction ABIs are:

- `dense_delta`: one decoded-source-relative code per site (`0=unchanged`,
  `1..4=target rank with the source class excluded`), 117,964,875 raw bytes;
- `sparse_u32_delta`: sorted global uint32 site plus the same four-way code for
  each of the 1,669,798 mismatches, 8,349,073 raw bytes.

Each correction was joined with QX2's 26,666-byte raw QXC1 event stream before
compression. This is the fair rate test: correction and events share one coder
and one 48-byte QXE section header, so no second-header tax creates the failure.

| exact closure form | Brotli q11 | LZMA-9e | zlib-9 | winner | delta vs 24,093 B cap |
|---|---:|---:|---:|---:|---:|
| dense delta + QXC1 events | 528,554 | **510,404** | 686,846 | LZMA-9e | **+486,311** |
| sparse uint32 delta + QXC1 events | 1,723,137 | **934,612** | 4,304,455 | LZMA-9e | +910,519 |

All six primary payloads, all six deterministic repeats, both raw corrections,
and both joined raw closure sections are retained. The dense winner is
`/Volumes/APDataStore/pact/ddm_qx3/retained/closure_candidates/dense_delta/candidate.lzma9e.bin`,
510,404 B, SHA-256
`6795e0470c65fbf33ccd50c3e859ec2df00342605e156e424f3dd50118d6da60`.
It is 487,743 B larger than QX2's 22,661-byte event-only payload.

## End-to-end receiver proof

The complete QX3 packet retains QX1 sections 1-7 byte-for-byte and replaces
section 8 with the selected joined closure. The receiver:

1. parses and integrity-checks the eight QXE sections;
2. reconstructs and evaluates QX1's QBT field from counted model/latent state;
3. applies the counted correction to recover baseline SHA-256 `02a2a3f5...ccf34`;
4. decodes QX2's address-free QXC1 stream against that baseline;
5. applies all 17,926 decoded transitions.

Primary and repeat archives are byte-identical. Both receiver passes produced:

- corrected baseline: 117,964,800 B, SHA-256
  `02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34`;
- 17,926 / 17,926 exact source event identities;
- reconstructed target: 117,964,800 B, SHA-256
  `36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68`.

The exact archive is
`/Volumes/APDataStore/pact/ddm_qx3/retained/complete/archive.zip`, **624,296 B**,
SHA-256
`5be6693516348f2a25c87fcea65f205477f339d6090c64636ef1c4b98531901c`.
The receiver is closed; the byte-feasible receiver is not.

## Prior negatives — named or folded

- **Pincer 352,525 B exact-address side information:** folded. QX3 transmits no
  explicit `(pair,row,col)` event tuple in QXC1, but the exact baseline repair
  still costs 510,404 B when jointly coded with events.
- **QBW2 188,860 B serialize-anything:** folded. QX3 used only the frozen QX1
  core plus an exact source-relative correction; the measured closure is even
  larger, so generic serialization is not reopened.
- **#1219 prevention-not-repair:** honored. Admission required exact baseline,
  exact 17,926-event identity, full target SHA equality, and a repeat; an
  approximate QBT field was never substituted for the specified object.
- **GF1 form-and-fit 5.09x / foreign baseline:** honored. The native QBT field
  was tested only as QX1's actual decoded state. QX2 was not declared closed on
  that foreign baseline; its baseline-bound QXC1 payload was repriced through
  an exact correction.
- **BR2 born route:** folded as terminal. No scorer, born-object detour, or
  distortion claim was introduced.

`verdict_scope=INSTANCE for pure zero-byte derivation from the pinned QX1 core;
FORMULATION for the two measured exact correction ABIs and three-coder race.
This is not a family theorem against a new core representation or a new direct
baseline-conditioned generator.`

## RECALL EVIDENCE

I searched beyond the charter seeds before treating the receiver premise or
the correction price as load-bearing:

- Full research corpus and arm receipts, content queries:
  `conditioning baseline|decoder-deriv|receiver closure|baseline field|QBT native|event/exception`,
  plus `Pincer|qbw2|gf1|#1219|e5a|E4 receiver`, across
  `.omx/research/` with Markdown/JSON/JSONL files bounded to 4 MiB.
- Canonical equations:
  `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  `argmax|cell identity|residual|rate marginal|archive|receiver`. Consumed
  `argmax_cell_identity_ideal_bytes_v1`,
  `procedural_predictor_plus_residual_correction_savings_v1`, and
  `canonical_frontier_pointer_v1`.
- Research index and graph memory: content search for
  `qx1|qx2|receiver closure|conditioning baseline|QBT native` across
  `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` FEED surfaces.
  Nothing QX1/QX2-specific was found beyond the same-day seed memos in those
  bounded graph surfaces.
- Design/SPEC corpus: searched `docs/`, `.omx/research/SPEC*`, and
  `.omx/research/charters/` for `conditioning baseline|decoder-deriv|receiver
  closure|QBT native|QBF1|QXT1`. The frozen QBF1 SPEC established that only
  receiver-derived coarse Road/tangent computation is free; no GT, masks, or
  video-derived constants may enter free code.
- Task and handoff ledgers: searched `codex_arm_queue*.jsonl` and bounded task
  ledger/status surfaces for QX1/QX2/QX3. QX1 and QX2 are both recorded as
  `BLOCKED_PENDING_SERIALIZER`; QX2's harvested fire order demands the exact
  baseline SHA without GT/S2 tables.

Findings beyond the seeds changed the plan in three ways. First,
`ddm_qbz1_descent_rate_configuration_20260829.md` supplied the retained exact
quantized-packet native field, so QX3 added a 117,964,800-site parity control
before blaming the baseline. Second, the QBF1 SPEC required raw tensor-record
reassembly and forbade hiding the GT baseline in generic code. Third,
`argmax_cell_identity_ideal_bytes_v1` explicitly excludes site grammar and
receiver cost, so its 2,724.8733-byte known-site result was not misused as a
baseline-patch price; QX3 retained and raced real complete payloads instead.

## Custody and reproducibility

- Result:
  `/Volumes/APDataStore/pact/ddm_qx3/RESULT.json`, 33,964 B, SHA-256
  `f9a71967ec01aa8905aeb31806f29ebb40a8c9729d0db9c58d36a02a540d7867`.
- Run manifest:
  `/Volumes/APDataStore/pact/ddm_qx3/RUN_MANIFEST.json`, SHA-256
  `bc09ee0c3fce6996680bd8ce374e7092fdd4a47411f3274c4eac0ab2cf38bb95`.
- Fresh decoded QX1 baseline: 117,964,800 B, SHA-256
  `afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd`.
- Dense raw correction: 117,964,875 B, SHA-256
  `85fb011b411530f2ed3897b5455da6ba039a6b7963cd0afd94990c1cf528e872`.
- Sparse raw correction: 8,349,073 B, SHA-256
  `a23577638ca054c1d74211a463877d1e31c6d6c904355be3b06c38f5fd6cf179`.
- Complete QXE: 624,172 B, SHA-256
  `d0382d650acd1050ad53b4cc02f2cc4eec89d8ff20f9da1dfec53765f4c14efe`.
- Command:
  `.venv/bin/python experiments/ddm_qx3_receiver_closure.py --resume-from /Volumes/APDataStore/pact/ddm_qx3`.
- Payload policy: every derived field, raw correction, joined raw section,
  coder candidate/repeat, packet/archive repeat, corrected baseline, and target
  output is retained under AP custody. No cleanup fired.

## Authority boundaries

- **Measured:** exact QX1/QBT decoder output; exact full-field and event-site
  label mismatch; two complete correction representations; six real coder
  outcomes plus repeats; complete archive bytes; exact receiver parse-back and
  target equality; deterministic archive/output repeat.
- **Not measured:** SegNet or PoseNet distortion, score components, contest
  score, CPU/CUDA parity, runtime under the contest harness, or a distortion
  row for QX1. No scorer, MPS, Metal, Modal, remote launch, or contest evaluator
  ran.
- The 1,593,421-byte conditional multinomial figure is an ideal reference, not
  a universal lower bound. The terminal negative is scoped to the pinned core
  and the two implemented correction forms.
- QX1/QX2's currently unlanded same-day source custody remains separate; QX3
  did not absorb or edit their files.

**Own-vehicle/effective frontier remains:** afr1 — S =
**0.14797617125559104 @ 180,002 B `[contest-CUDA T4, n600]`**, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`;
QX3 made no score measurement and did not move the pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned QX representation owner; consumer store: `/Volumes/APDataStore/pact/ddm_qx3/RESULT.json`; fire trigger: a structurally new receiver-available baseline statistic or a QBT-baseline-direct event grammar is specified that contains no GT/S2/scorer table in free code and can plausibly reduce the complete jointly coded section from 510,404 B to at most 24,093 B; then run an exact full-n600 payload-retaining race, not either closed dense/sparse patch again.

## LIVE-HYPOTHESES

- A direct event grammar from the decoded QBT field to the final S2 target may
  exploit boundary continuity better than "repair C1, then apply QXC1." It is
  plausible because dense LZMA compresses 117.99 MB to 510 KB, revealing strong
  structure, but it must still demonstrate a further 21.2x reduction and exact
  receiver closure.
- Re-valuing or retraining the counted QBT packet against the exact C1 field may
  make the required baseline decoder-native. It is plausible because the
  present packet already misses only 1.4155% of sites, but it is a new core
  representation and must reprice all sections and remeasure any distortion;
  no number transfers from QX3.

## DEAD-ENDS

- Pure zero-byte derivation from this exact QX1 core is closed: the fresh
  decoder is exact to its retained packet field yet misses 1,669,798 C1 sites.
- Treating agreement only at the 17,926 event addresses as sufficient is
  closed: 9,619 event sites already disagree, and QXC1's candidate field/ranks
  depend on the complete baseline raster.
- Dense four-way and sparse uint32 exact patches with Brotli q11, LZMA-9e, or
  zlib-9 are closed at formulation scope: their best joined sections are
  510,404 B and 934,612 B, versus 24,093 B.
- Substituting the foreign QBT baseline without repricing QX2 is closed: QXC1
  binds and verifies the exact C1 baseline SHA.
- Promoting the exact 624,296-byte decode as goal progress is closed: it has no
  scorer row, exceeds the archive gate by 486,311 B, and leaves afr1 unchanged.
