# DDS1 — decoder-derivable born-statistics verdict

**Typed verdict: `PARTIAL` (`SCREEN`, seeded random n=120).** The original GF1 tuple is not
decoder-derived. The one surviving zero-payload substitute is the HPAC model's own current argmax plus
strictly prior predicted-boundary state. It captures **10.8575495545%** of GF1's cross-fitted
wrong-half predictive gain, barely above the charter's 10% partial-admission floor, but agrees with the
exact demanded GF1 tuple on only **6.2934782609%** of wrong sites. This is a weakened M-only
formulation for task #1374 to price jointly, not a derivation of the 47,603-byte GF1 packet and not a
physical-byte win.

No scorer, real coder, archive build, receiver mutation, Modal job, Metal run, or authority evaluation
ran. RXC1 and `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` were not touched. The
canonical pointer did not move.

## Result first

The measurement used the real retained AFR1 field, DF1 coding argmax/pmax, and GF1 generated field.
Every row used the same 250-cell categorical log-odds family and the same pair-level two-fold split.
`SCREEN bits` are conditional-codelength estimates, not coder bytes.

| charter route | zero-payload formulation | wrong-half GF1-value overlap | exact tuple agreement on wrong sites | typed outcome |
|---|---|---:|---:|---|
| (a) causal boundary transform | latest legal current-frame prefix class + boundary born before the current group | **2.5529375246%** | 6.3326086957% | `CLOSED` |
| (b) causal GF1 refit | exact HG1 fit over all complete prior frames, then one-pair last-parameter extrapolation | **0.0000000000%** | 37.3565217391% | `CLOSED` |
| (c) M-derived | current HPAC argmax + predicted boundary born before the current group | **10.8575495545%** | 6.2934782609% | **`PARTIAL`** |
| (d1) missed state | previous fully decoded class + boundary bucket | **0.8822765640%** | 25.4586956522% | `CLOSED` |
| (d2) missed state | per-site modal class over all complete prior frames + boundary bucket | **4.1327843413%** | 21.0934782609% | `CLOSED` |

The wrong-half screen contains **46,000 wrong sites** among **9,991,193 live sites** and
**23,592,960 sampled sites**. Baseline wrong-half codelength is **124,048.366485450 SCREEN bits**.
GF1's matched-context gain is **6,219.375657132 SCREEN bits** on the wrong half and
**981.498160630 SCREEN bits** over all live sites. The M-derived surrogate gains
**675.271793953 SCREEN bits** on the wrong half but only **3.322195548 SCREEN bits** over all live
sites. Its wrong-half overlap is therefore `675.271793953 / 6,219.375657132 = 0.108575495545`.
No one may divide these figures by eight and cite the result as physical bytes.

## Demand type

For decode position `s = (pair:u16, y:u16, x:u16)` in the shipped 64x64, delta=2 causal group order:

```text
BornContext(s) = (
    generated_class:u3,
    hpac_model_class:u3,
    boundary_distance:u3,  # clipped L1 distance 0..4
    agreement:u1,          # generated_class == hpac_model_class
)

PeelContext(s, k) = (rung:u3, BornContext(s))

Required zero-payload derivation:
F(M, CPR1/dxi, X_<s, s) -> (generated_class, boundary_distance, agreement)
```

Candidate 1 consumes `BornContext`; candidate 2 consumes the same tuple as chain-rule conditioners.
The M-derived survivor substitutes `generated_class := hpac_model_class`. It is deterministic and
zero-payload, but it collapses the agreement bit to true and does not reproduce GF1's generated class.

## Already-counted decoder state

The payload inventory below is copied byte-for-byte from the pinned AFR1 archive into durable DDS1
custody. The five sections plus the 14-byte RX1 header close the 179,902-byte archive member; ZIP
framing closes the complete 180,002-byte archive.

| state available to the receiver | durable path | bytes | SHA-256 |
|---|---|---:|---|
| complete AFR1 archive | `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/runtime_candidate_native/archive.zip` | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| exact decoded `X` used as causal prefix | `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/out/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8` | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| counted HPAC model `M` | `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/retained/counted_decoder_state/hpac_model_counted.bin` | 13,515 | `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` |
| counted semantic section | `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/retained/counted_decoder_state/semantic_counted.bin` | 30,856 | `39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f` |
| counted CPR1/dxi carrier | `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/retained/counted_decoder_state/carrier_dxi_counted.bin` | 22,010 | `932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12` |
| counted compact mixer table | `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/retained/counted_decoder_state/mixer_table_counted.bin` | 96 | `8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1` |
| counted RC64 token stream | `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/retained/counted_decoder_state/token_stream_counted.bin` | 113,411 | `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` |

The bounded receiver-code census found the following generic deterministic functions over that state.
Rule 118 makes the algorithms free; none contains a video-derived replacement for GF1.

| receiver surface under `…/runtime_candidate_native/` | bytes | SHA-256 |
|---|---:|---|
| `runtime/residual_archive.py` | 30,622 | `aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530` |
| `runtime/rr4_free_corrector.py` | 15,421 | `96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0` |
| `runtime/fx2_model_axis_corrector.py` | 31,470 | `6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064` |
| `runtime/free_corrector.py` | 14,709 | `dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5` |
| `runtime/f26_hpac_native.py` | 26,781 | `f13fe8beee4268cbc1df4f20016a6db6635bebebb53afc85e5025ada3748ecba` |
| `runtime/f26_hpac_native.c` | 40,911 | `1326fd9dd2c85c9e78da64b1a4986536f21eb9a416ad6fbfaf1bb88698d70c00` |
| `runtime/entropy/rc64.py` | 2,904 | `dad52aa013aafcc64666630724523431597f4da5997aebcaec91c63b515ade36` |
| `cpr1/inflate.py` | 13,792 | `ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b` |
| `cpr1/hpac_integer.py` | 17,471 | `cea40a9bf2fe6db36e7269de0d25711eb77dffdc58955d3eb9228767d448db66` |
| `cpr1/integer_model_io.py` | 5,881 | `dee0bfa4c8e46a47d10216a94f5f43f9b27eb41e781ab99f740099e089c63be1` |
| `cpr1/carrier_codec.py` | 7,353 | `9824bd3f8913e756b1b2d76d50e7b439187e2ce16b5014b6418bda059b2eb4b3` |
| top-level `inflate.py` | 2,282 | `a499942a9993737755f771a95a81b8a12fc4a3b2e1b5ba5cd9d9cbfc738ea958` |

The receiver already sees predicted class, confidence, temporal agreement, run length, prior-frame
boundary, group/patch state, CPR1, dxi, and corrector state. Position is generic, but no searched code
computes GF1's video-derived horizon/lane/movable/MyCar parameters. The surviving M route is assembled
only from current predicted class and predicted boundaries whose endpoint groups are both earlier than
the current group.

## Measurement contract and causality

- Selection: `numpy.default_rng(20260901).choice(600, 120, replace=False)`, sorted only after selection;
  sampled pair range 2..599. This is a seeded random sample, never a contiguous prefix.
- Decode order: 64x64 patches, local group `x mod 64 + 2*(y mod 64)`, 190 groups. Current-group tokens
  never enter any current-frame surrogate.
- Cross-fit: pair-level two-fold split from seed 20260902. Every GF1/surrogate context has the same 250
  possible cells and nests the shipped pmax logit with a zero offset.
- Wrong half: `X != hpac_model_class` among finite `pmax < 1` positions. The prior full-population
  physical-model decomposition names 227,671 flips and about 76,601.54 B of wrong-branch mass; DDS1
  does not relabel its screen codelength as that physical byte figure.
- Overlap: `clip(surrogate wrong-half cross-fitted gain / GF1 wrong-half cross-fitted gain, 0, 1)`.
  A 2,500-cell route-by-GF1 joint table was rejected during review because its changed complexity made
  held-out residual gains negative and could manufacture overlap above 100%.
- Causality controls: synthetic future-group mutation left prefix class and distance unchanged at seven
  tested thresholds; DDS1 boundary buckets matched the retained receiver on 983,040 tested sites.
- Retention: all 15 sampled raw fields, refit diagnostics, counted archive sections, checkpoints,
  receipts, and manifest remain under `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/`.

## Per-route conclusions

### Route (a): causal current-prefix boundary — `CLOSED`

Class agreement looks high over all live sites (97.2825%) because the easy mass dominates, but on the
wrong half class agreement is 46.6826%, boundary agreement is 11.3109%, and exact tuple agreement is
6.3326%. It captures only 2.5529% of GF1's wrong-half value and loses 750.5766 SCREEN bits over all
live sites. Do not retry the same latest-neighbour plus L1-boundary transform.

### Route (b): causal GF1 refit — `CLOSED`

The final v2 route runs HG1's real horizon, coherent-lane, movable-box, and running-MyCar fits over every
fully decoded prior frame, then causally extrapolates the last fitted parameters one pair. Its wrong-half
class/boundary/tuple agreements are 68.8913% / 51.2196% / 37.3565%, but its wrong-half predictive gain
is **-11.466812714 SCREEN bits**, so overlap clips to zero. The charter's 0.37% target-independence premise
does not rescue this route: GF1 measured similar mismatch counts against two targets; it did not prove
packet-parameter stability under temporal-prefix refitting.

### Route (c): M-derived — `PARTIAL`

Current HPAC argmax plus strictly causal predicted-boundary distance captures 10.8575% of GF1's
wrong-half gain. It is legal and zero-payload, but its exact tuple agreement is only 6.2935% on wrong
sites and its all-live gain is only 3.3222 SCREEN bits. OC2 already closed the simpler predicted-class ×
causal-neighbour chart on the LB1 body and found only a 2 B separate rank rider; DDS1's different
AFR1/GF1 boundary-distance formulation therefore requires one joint #1374 price and gets no transferred
byte credit.

### Route (d): other retained decoder state — `CLOSED` for tested formulations

Previous decoded class/boundary captures 0.8823%; temporal mode/boundary captures 4.1328%. Both lose
SCREEN bits over all live sites. These two formulations add no admissible escape.

## Replacement bar and candidate disposition

| formulation | counted GF1 packet | maximum passing G+M pool | current G+M pool | replacement required | fraction of current pool |
|---|---:|---:|---:|---:|---:|
| original candidates 1+2 with exact GF1 tuple | 47,603 B | 37,306 B after packet | 126,926 B | **89,620 B** | **70.6081%** |
| weakened M-only partial variant | 0 B new payload | 84,909 B | 126,926 B | **42,017 B** | **33.1035%** |

The original candidates are not derived and remain dead at the 89,620 B bar. The partial variant may be
handed to #1374 because it needs no GF1 packet, but it retains only 10.8575% of GF1's screened wrong-half
predictive value and overlaps state the shipped mixer/correctors already consume. The 42,017 B bar is
the pool arithmetic after removing the packet; it is not a claim that the partial context can replace
42,017 B. Only a real joint encode on the #1374 object can decide that.

MI1's paid-context 47.4x break-even miss remains binding. JT21/JT22 forbid adding banked marginal
credits. The #1199 agreement exponent concerns d_seg transfer, while DDS1 measures rate-side surprise;
neither quantity is substituted for the other.

## RECALL EVIDENCE

DDS1 consumed XOV1's complete 84-cell matrix and parent custody rather than repeating its receiver
search. XOV1's retained result is
`/Volumes/APDataStore/pact/ddm_xov1_crossover_pass/RESULT.json`, 69,658 B, SHA-256
`59003d28f2399cd9f6e4a7431d8107a2b90c232062976b71a95eae423ef8094a`; its memo commit is
`78f570edca`.

The bounded recall also read the GF1 form/capacity and gap decomposition, MI1, HC1, WH1, OC2,
JT21/JT22, SFP1, the canonical equation registry, the rate research index, the main DAG, and live hot
state. It did not find a prior derivation of GF1's class/boundary tuple in those scopes. The additional
facts that changed execution were:

- GF1 target-independence is a mismatch-count comparison, not a parameter-stability receipt.
- HC1 retains the exact prior-frame boundary bucket; MI1 enumerates current predicted class/confidence,
  temporal agreement, run length, and prior boundary as receiver contexts.
- OC2 already drained simpler free temporal/causal charts on LB1, so the DDS1 partial is a
  current-object boundary-distance screen, not licence to rebuild those charts.
- SFP1 has already prepared distinct changed-field candidates for #1374; DDS1 did not duplicate them.
- `main_hot_state.md` makes #1374 the sole SCMDL coder/model owner. DDS1 made no parallel instrument.

## Artifacts and verification

| artifact | bytes | SHA-256 |
|---|---:|---|
| runner `experiments/ddm_dds1_decoder_derivable_born_stats.py` | 38,886 | `4a38c01d21ace94c704cbc3225c9482fcf2b59de330e28382ec67c24a2a01769` |
| v2 `RUN_CONFIG.json` | 3,770 | `127983918cd2e10a826ddd559d85524d88013ad533f45458380e21f223a1bb31` |
| v2 materialize receipt | 7,461 | `bbb39ebb4601144db01767641b16f13bea3aa15b8eaf341f97b740bb5885f123` |
| v2 `RESULT.json` | 15,534 | `057b073eb874bb74e35915f5ac5939551c442b1d8bad255c5f4c5d5f20aa54d8` |
| v2 analyze receipt | 391 | `3fd06da4982b5b57afdcfd4c994d23b01e625bcc00c60eb80ed18adb7fa85640` |
| v2 refit diagnostics | 61,979 | `38a2e086fc67b36ca2c48663d1ce3ea1efa2b042d39759c986ef85653c11ec45` |
| v2 manifest | 6,506 | `6b0590485c56e04b069d8b4f729268f8cf9641a055c248126525a47e63387414` |

The manifest covers **27 files / 424,945,477 B** excluding itself. The earlier root-level v1 screen is
preserved as superseded evidence; v2 did not overwrite or delete it. Verification passed: Python
compile, Ruff, `git diff --check`, the P0 measure-and-discard gate (0 findings), source hash/size checks,
120/120 materialization checkpoints, 15/15 finalized surrogate/source fields, causal future-mutation
control, receiver boundary-twin control, and two independent manual source-review passes.

## NEXT_IF_RESUMED

- **`FOLDED-INTO-ACTIVE-OWNER`** — owner: task #1374 SCMDL `X,G,M`; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: when #1374 next ingests its candidate roster after `GATE-1-PASSED`, keep original GF1-dependent candidates 1+2 closed, and admit only the weakened M-argmax/predicted-boundary variant to one real joint price if it is not already identical to shipped mixer/corrector state.
- **`DEFERRED-FULL-POPULATION-CONFIRMATION`** — owner: #1374 successor selected by MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_dds1/measurement_v2/`; fire trigger: only if the weakened M-only variant survives deduplication and a joint-pricing slot depends on resolving the near-threshold 10.8575% n120 screen, rerun the same frozen 250-cell estimator on seeded/full n600 before any byte claim.

## LIVE-HYPOTHESES

- The M-argmax plus predicted-boundary route may retain a small independent marginal on the AFR1 body
  because it crossed the frozen 10% screen while the current-prefix and temporal routes did not. It is
  plausible only as a joint-priced rider; OC2's drained LB1 charts make a substantial byte win unlikely.
- A full-n600 repeat may move the 10.8575% point estimate across the 10% boundary because the legal n120
  screen is close to the threshold. The complete retained fields and frozen estimator make that question
  reproducible if #1374 actually needs it.
- XOV1 candidate 3 and SFP1's changed-field proposals remain live because they alter the jointly coded
  object rather than requiring GF1's tuple. DDS1 neither prices nor closes those families.

## DEAD-ENDS

- The exact HG1 temporal-prefix refit is closed in this screen: its wrong-half gain is negative and its
  overlap is 0%. Do not retry the lag-one approximation or cite GF1's target-independence as parameter
  stability.
- The latest-causal-neighbour plus L1 boundary transform is closed at 2.5529% wrong-half overlap; the
  previous-frame and temporal-mode variants are closed at 0.8823% and 4.1328%.
- Exact decoder derivation of the original GF1 tuple was not found across 5/5 tested zero-payload
  formulations, 120/120 random pairs, 23,592,960 sampled sites, 9,991,193 live sites, and 46,000 wrong
  sites. The M partial's 6.2935% wrong-site tuple agreement is not a derivation.
- The initial 50-cell analysis and its 2,500-cell conditional joint table are superseded: the former
  omitted HPAC model class from the declared type, while the latter changed model complexity and produced
  impossible overlap above 100%.
- Converting SCREEN bits to bytes, scaling the 47,603-byte packet by overlap, or adding JT21/JT22/OC2
  credits is closed. A physical rate conclusion requires one real joint encode on the exact object.

**OWN-VEHICLE FRONTIER: AFR1 S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`; UNMOVED.**
