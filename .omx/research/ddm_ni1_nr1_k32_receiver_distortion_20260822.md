# OUTCOME — QUEUED, NOT SCORED: NR1 K32 is byte-closed in the shipping DX2 full-RGB receiver at 122,250 B, but d_seg, d_pose, Lane retention, and S are NOT MEASURED because NI1 does not own the sole n600 scorer slot and its charter says do not fire.

`[byte-closed shipping receiver build; scorer-free]` The exact archive is
`/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/runtime/archive.zip`,
122,250 B, SHA-256 `fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e`.
This is an executable candidate and a queued measurement, not a score row and not a sub-0.12 claim.

## Result boundary

| Object | Result | Authority |
|---|---:|---|
| Exact `archive.zip` | 122,250 B | MEASURED, byte-closed build |
| Counted NR1 packet | 69,004 B, SHA `a68765dc…dab28` | MEASURED, inherited and revalidated |
| Decoded K32 field | 117,964,800 B, SHA `d416895a…b8d8`, identical repeat | MEASURED, shipping adapter parse-back |
| Paid-section consumption | QPARAM/QCTX/QPAIR/QEVENT = 1/1/1/1 | MEASURED, strict receiver trace |
| n600 d_seg | NOT MEASURED | MAIN-fire-only |
| per-class d_seg, including Lane | NOT MEASURED | MAIN-fire-only |
| n600 d_pose | NOT MEASURED | MAIN-fire-only |
| NI1 score S | NOT MEASURED | no component substitution allowed |

The prior-law prediction is therefore **INCONCLUSIVE** with
`verdict_scope=INSTANCE:NR1_K32_ON_DX2_PENDING_N600_SHIPPING_RECEIVER_SCORE`.
There is no pass/fail result against `d_seg = 0.00021731`; writing one from token agreement would be
the exact fake this charter forbids.

## Arithmetic check and realized byte price

The DX2 control receipt at
`/Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1/MODAL_REMOTE_RESULT.json` records
`d_seg=0.00020139`, `d_pose=0.00000637`, and 180,368 B. Recomputing from those report-precision
components gives:

- pose contribution `sqrt(10*d_pose) = 0.007981227975693965`;
- distortion `= 0.020139 + 0.007981227975693965 = 0.028120227975693968`;
- DX2 rate contribution `= 25*180368/37545489 = 0.1200996476567398`;
- DX2 `S = 0.14821987563243377 [contest-CUDA T4 n600]`;
- continuous fixed-distortion cap `= 137986.83879444358 B`, so STRICT integer max is 137,986 B.

The charter's prose rounds the continuous cap to 137,986.88 B; the recomputation differs by 0.041 B
because its displayed components were rounded, but the binding integer cap, required 42,382 B cut,
and every disposition agree exactly. For its conservative 135,595 B projection, the recomputed slack
is 2,391 B = 0.0015920687569151116 S, and the fixed-pose ceiling is
`d_seg=0.0002173162727570521`, 1.0790817456529724 times DX2. This operationally agrees with the
charter's `0.00021731` / 7.9% bar.

The shipping integration is smaller than that conservative projection:

| Byte row | Bytes | Delta vs strict cap | Status |
|---|---:|---:|---|
| DX2 control | 180,368 | +42,382 | measured control |
| Charter K32 projection | 135,595 | -2,391 | projection, assumed all non-token bytes kept |
| NI1 exact archive | **122,250** | **-15,736** | measured byte-closed archive |

The 13,345 B improvement over the projection is real archive arithmetic: NI1 removes the 13,515 B
learned HPAC stream because QCTX and QPAIR replace its context/temporal job, while the authenticated
NI1 header costs 170 B more than RX1. Keeping HPAC as a paid but output-inert section would violate the
exact-consumer contract. At 122,250 B, rate is `0.08140125701918545`; holding the rounded DX2 pose
fixed, the realized-byte d_seg ceiling would be `0.00030617515005120584`. This wider ceiling is a
budget calculation only. The queued harvest must still report the measured result against both the
charter's conservative `0.00021731` bar and the realized-pose ceiling.

## Shipping receiver integration

The source adapter is `experiments/ddm_ni1_runtime_receiver.py`; the reproducible builder is
`experiments/ddm_ni1_nr1_k32_receiver_distortion.py`. The builder copies the sealed DX2 runtime to a
fresh tree and changes only the archive parser/terminal token decoder boundary:

- copied unchanged: `cpr1/inflate.py` SHA `ff446edd…a736b`, semantic weights, carrier, frame-0
  selector, compensation, `SemanticTokenRenderer`, R/uint8 operations, and `render_video`;
- added: `runtime/nr1_taskcell_quotient.py` SHA `66500b81…cbb6` and the NI1 shipping adapter;
- replaced: the RC64+HPAC terminal token decode with strict NR1 K32 decode;
- refused: every decoder other than `F26_TOKEN_DECODER=ni1-nr1` and every prefix other than n600.

The adapter invokes the copied shipping `runtime.residual_archive._decode_rx1_models` to restore the
semantic and carrier objects, validates the residual table through the copied shipping inverse, then
passes the independently decoded uint8 K32 tensor to the unchanged renderer. The generic `IHS1`
format marker is fixed, video-independent runtime code; it contains zero learned bytes. Candidate
runtime digest: 41 files, 663,578 B, SHA `0913c430ed2a05ed3eaf72ca915377aacd2e2a99e1fd95f5a52021c1cf5a72d7`.

The retained producer runner SHA (`44e8ac10…7847c`) differs from the current workspace runner SHA
(`25271724…c93b`) only by exactly two uses of the symbol rename
`DX2_TOKEN_SHA256 -> DX2_TASK_FIELD_SHA256`. The actual packet module is byte-identical at
`66500b81…cbb6`. The builder refuses any drift beyond that exact reviewed rename.

## Retention and controls

The durable root is
`/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/`.

- `RESULT.json` and seven fsynced stage checkpoints make the build crash-auditable and resumable.
- `retained/receiver/tokens_full.u8` and `tokens_full.repeat.u8` are each 117,964,800 B with identical
  SHA `d416895a…b8d8`.
- `retained/archive.repeat.zip` is byte-identical to the candidate.
- `retained/inherited_coder_race/` copies and revalidates all 36 K32 real-coder candidate/repeat
  payloads, including every loser and both packet ZIPs.
- Seven retained mutation archives cover semantic, carrier, residual, QPARAM, QCTX, QPAIR, and
  QEVENT. The shipping receiver refuses all seven; the inner NR1 parser separately refuses each
  mutated Q surface. These are integrity/refusal controls only, not distortion evidence.
- The targeted ALWAYS-KEEP detector examined both NI1 Python files and found 0 measure-and-discard
  sites.

No retained payload was deleted. Earlier build_r1-r3 receipts remain under the same root as preserved
development provenance; build_r4 is the selected exact candidate.

## Scorer disposition

NI1 did not launch a scorer. The common contract permits only one full-n600 scorer job fleet-wide,
the charter does not assign NI1 that slot, and it explicitly requires a sealed MAIN fire order rather
than a fire. RI1's different-payload advisory occupied the slot when NI1 sealed the order.

`/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/build_r4/SEALED_FIRE_ORDER.json`
(5,325 B, SHA `98ab92044bfff448f99b8e25a0d1eabc3d752097421c441bfb3a40a16785c86e`)
queues, in order:

1. canonical local advisory n600 repeat 1 via `tools/fire_local_advisory.py`;
2. the RI1-landed per-class frozen-SegNet method, batch size 16, with Lane class 1 on its own row;
3. canonical local advisory n600 repeat 2.

Each row names MAIN as owner, its durable consumer store, and an explicit sole-slot fire trigger.
Even if the advisory falsifies the prior law, promotion still requires a separately authorized exact
contest-axis evaluation; local CPU is advisory only.

## RECALL EVIDENCE

I searched the full corpus, not only the charter seeds:

- content queries `task-cell|taskcell|decision-quotient|quotient|K32|QPARAM|QCTX|QPAIR|QEVENT|token agreement|receiver`
  over `.omx/research/` memos and arm receipts;
- the JSON output of `.venv/bin/python tools/list_canonical_equations.py --json`;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC files, and
  `.omx/state/canonical_task_status.jsonl` / task-ledger surfaces;
- live and retained K32/K64/K128 payload/result stores under
  `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/`.

Beyond the charter seeds, the search recovered task-ledger row `deferral_ledger::D42` for a whole-
teacher decision-quotient student at K32/64/128, FEED-603/MS2/DC1 quotient contracts, and the later ES1
task-cell program design. None is a current-DX2 shipping receiver substitute: D42 is an old deferred
student, the July quotient surfaces include the already-closed synthetic/no-op ABI, and ES1 is design
only. No executable canonical equation adds a competing current-DX2 packet. This changed the plan by
keeping the integration on the copied DX2 receiver, treating HPAC replacement as an exact-consumer
requirement, and refusing borrowed distortion credit. It also strengthens K64 as the next rung if K32
fails: with the same honest HPAC replacement and header overhead, its measured 79,876 B packet would
project to 133,122 B, below the strict fixed-distortion cap; that archive and distortion remain
unbuilt/unmeasured.

## Verdict and handoff

The rate hypothesis is not closed: K32 is substantially more byte-feasible than the charter's
conservative projection. The distortion hypothesis is not answered: 1,558,833 tokens changed, no
qualifying token-sensitivity corpus exists, and the sister RC1 rare-class collapse remains only a
prior. The next valid evidence is the queued same-archive n600 shipping-receiver row, not more token
agreement analysis.

### LIVE-HYPOTHESES

- K32 may pass the realized-byte ceiling because removing superseded HPAC widens fixed-pose d_seg
  headroom to `0.00030617515`; this is plausible from exact archive arithmetic but entirely untested.
- The charter's rarer-class-collapse prior may still make K32 fail even the wider ceiling; 1,558,833
  changed tokens and RC1's class-1 collapse make this plausible, but cross-payload transfer is not a
  measurement.
- K64 may be the better rate-distortion rung if K32 fails: its retained packet changes 80,417 fewer
  tokens and projects to 133,122 B under the same honest receiver shape, but it needs its own exact
  archive and n600 row.

### DEAD-ENDS

- Token agreement as evaluator evidence is closed: 98.6786% does not determine d_seg, d_pose, or Lane.
- The July/synthetic quotient ABI is closed as a shortcut: it lacks genuine current QCTX and the
  shipping full-RGB receiver path.
- Keeping the old learned HPAC beside NR1 is closed: it would be paid, output-inert duplicate context.
- Mutation refusal as receiver-effect evidence is closed: mutations prove integrity only; the shipped
  output and frozen scorer decide distortion.
- K128 in this current packet family remains rate-dead at fixed DX2 distortion.

OWN-VEHICLE FRONTIER: UNMOVED — NI1 S NOT MEASURED; DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600].
