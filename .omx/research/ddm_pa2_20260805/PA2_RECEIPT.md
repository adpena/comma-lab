# PA2 Receipt - Persisted-Stream Pricing Probes - 2026-08-05

## Answer First

PA2 moved no score pointer and ran no scorer. It priced two scorer-free coder races on od9's verified SSD-persisted OD2 n32 stream.

Probe 1, AM1 acceleration residual: **flat wins** on the available pose/carriage stream (`stage2_qcoeffs`). Brotli q11 is `flat 1,917 B`, `delta1 2,014 B`, `delta2 2,172 B`; LZMA1 raw is `flat 2,161 B`, `delta1 2,303 B`, `delta2 2,477 B`. The AM1 falsifier fires: delta2 loses to delta1 by `+158 B` under Brotli and `+174 B` under LZMA1. Verdict scope: **FORMULATION**, OD9 stage2 cheapdct4 qcoeff carriage on OD2 n32. The acceleration prior is dead on this stream.

Probe 2, IG1-F2 shared context: **plain shared context wins modestly** on the full per-pair persisted stream. Brotli q11: `shared_context 66,497 B`, `xi_conditioned_shared 66,541 B`, `independent_per_pair 70,472 B`; shared saves `3,975 B` (`5.64%`) over independent. LZMA1 raw: `shared_context 67,946 B`, `xi_conditioned_shared 68,047 B`, `independent_per_pair 69,013 B`; shared saves `1,067 B` (`1.55%`). The xi-conditioned ordering does not beat plain shared context (`+44 B` Brotli, `+101 B` LZMA1).

Routing consequence: the shared-context win is real evidence for cross-pair coupling/shared parametrization, but it is modest and still projects to `1,246,819 B` n600 under the best Brotli row. It supports the TR1/shared-carrier direction weakly-to-moderately; it does not rescue flat solved-paint shipping. pk1 still needs a receiver-closed boundary grammar under the 45-90 KB corridor.

## Denominators And Boundaries

- Axis: `[macOS-CPU byte-only persisted-native pricing]`.
- Selection: OD2 n32 seed20260805 stratified pair set inherited by OD3/OD9.
- Pair ids: `8, 32, 46, 57, 70, 107, 112, 119, 148, 154, 168, 198, 225, 234, 244, 251, 284, 328, 336, 349, 383, 399, 411, 423, 445, 465, 481, 516, 536, 561, 582, 583`.
- Projection: `ceil(n32 coded bytes * 600 / 32)`, byte-routing only.
- Scorer forwards: `0`. `upstream/evaluate.py`: not run. Paid launches: `0`. n600 archive: not built.
- Subset caveat: OD3/OD9 label this set pose-easy `0.42628664334579025x` population and seg-matched `1.0099888594483923x`; PA2 measures bytes only.
- xi/dxi bounded absence: no `xi`/`dxi`-named sections or NPZ keys were found in `od9_ssd_payload_manifest.json` or the 32 pair-payload NPZ files. PA2 used `stage2_qcoeffs` as the persisted pose/carriage stream.

## Input Custody

OD9 manifest verification passed: 42/42 declared SSD entries matched both byte count and SHA-256; total checked SSD payload bytes `981,167`; recorded tree SHA `7a127223953ab330f5539d48ceefd4173972d5728d0d71f6803fb0567c63c4a8`; `no_tmp_persisted_evidence=true`.

Full JSON artifact: `.omx/research/ddm_pa2_20260805/PA2_RECEIPT.json`, `27,616` bytes, SHA-256 `d5f674fab4773187de2a87212069ddd5a0bbb19d7df740eca10953c77c6e3e5e`.

## Probe 1 - AM1 Acceleration Residual

Surface: `stage2_qcoeffs`, shape `32 x 48`, one 3x16 int16 cheapdct4 carriage block per pair. Packet family: signed zigzag-varint residual packet plus outer Brotli q11 / LZMA1 raw coder.

| variant | transform | raw packet B | Brotli q11 B | LZMA1 raw B | decode equality |
|---|---|---:|---:|---:|---|
| flat | qcoeff values in pair-id order | 2,636 | 1,917 | 2,161 | true |
| delta1 | first pair values, then pair-axis first differences per coefficient | 2,770 | 2,014 | 2,303 | true |
| delta2 | first pair values, first delta, then pair-axis second differences per coefficient | 2,923 | 2,172 | 2,477 | true |

Residual concentration moved the wrong way:

| variant | mean abs | median abs | zero frac | +/-4 frac | +/-16 frac |
|---|---:|---:|---:|---:|---:|
| flat | 133.875 | 107.0 | 0.001953 | 0.027995 | 0.098307 |
| delta1 | 189.460938 | 163.0 | 0.001302 | 0.011068 | 0.063151 |
| delta2 | 326.970703 | 268.5 | 0.000651 | 0.009766 | 0.038411 |

Conclusion: the stream is not acceleration-smooth in the AM1 sense. delta2 loses to delta1 under both coders, and delta1 loses to flat. m38's prediction fired.

## Probe 2 - IG1-F2 Shared Context

Surface: full persisted per-pair packet: `stage1_flat` support, `stage1_rgb`, and `stage2_qcoeffs`.

Variant definitions:

- `independent_per_pair`: each pair encoded as a standalone PA2P1 member and compressed independently; coded bytes are summed across 32 members.
- `shared_context`: one PA2IG1 stream in ascending pair-id order; global pair-id list once; per-pair stage1 support/RGB plus stage2 qcoeffs.
- `xi_conditioned_shared`: one PA2IG1 stream ordered by stage2 absolute qcoeff energy then pair id; global pair-id list once; stage2 qcoeffs stored first as delta1 residuals and stage1 support/RGB follows in that conditioned order.

| variant | Brotli q11 B | projected n600 B | LZMA1 raw B | projected n600 B | decode equality |
|---|---:|---:|---:|---:|---|
| shared_context | 66,497 | 1,246,819 | 67,946 | 1,273,988 | true |
| xi_conditioned_shared | 66,541 | 1,247,644 | 68,047 | 1,275,882 | true |
| independent_per_pair | 70,472 | 1,321,350 | 69,013 | 1,293,994 | true |

Matched-coder deltas:

| coder | shared vs independent | fraction saved | xi-conditioned vs shared |
|---|---:|---:|---:|
| Brotli q11 | -3,975 B | 5.6405% | +44 B |
| LZMA1 raw | -1,067 B | 1.5461% | +101 B |

Conclusion: shared context wins, but the win is near the low end of m38's predicted 5-15% band and is coder-sensitive. The xi-conditioned ordering does not add value on this persisted surface.

## Recall Evidence

| query / source | finding beyond charter seeds | plan impact |
|---|---|---|
| `pa2_prompt`, `_common_contract`, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, operating manual, `main_hot_state` | PA2 is scorer-free; latest hot state names pa2 as persisted-stream coder races and pk1 as the receiver-closed boundary-grammar successor. | No scorer/eval/launch; route result to pk1/TR1 decision only. |
| `AM1_CROSSWALK_RECEIPT`, `acceleration residual`, `OD8 native DOF delta entropy` | AM1's only open item was a flat/delta1/delta2 byte-closed residual race after persistence existed. | Ran the exact residual race on the persisted stage2 carriage stream; closed AM1 for this stream. |
| `IG1_CROSSWALK_RECEIPT`, `IG1-F2`, `shared-context`, `vanishing condition` | IG1-F2 required independent vs shared-context coding on the same persisted stream with exact decode equality. | Ran independent, shared, and xi-conditioned shared variants under both coders. |
| `OD9_RECEIPT`, `od9_ssd_payload_manifest`, `pair_payloads` | OD9 had already killed flat/base-delta solved-paint shipping, but explicitly left AM1/IG1 structural pricing as follow-ons. | Did not re-claim OD9's old result; measured the two new structural questions. |
| `TC1_RECEIPT`, `trajectory coupling`, `record-censored` | True cross-step interaction fields were absent before OD9; TC1 folded future instrumentation into OD9 or next persistence touch. | Treated PA2 as a coder race over completed persisted payloads, not a Fisher/trajectory law registration. |
| `SD1_CROSSWALK_RECEIPT`, `CPWL`, `TR1`, `pk1` | The live route is receiver-closed boundary grammar/task description; generic spline or dense ReLU detours stay folded. | Shared-context evidence affects pk1/TR1 routing, not a new scorer slot. |
| `tools/list_canonical_equations.py --json` filtered for Fisher, trajectory, Bregman, Brotli, codec | Existing rows cover Fisher geometry, trajectory stopping, and entropy-code locality (`trajectory_derived_stopping_law_v1`, `master_gradient_locality_violation_by_codec_v1`, `brotli_cascade_bounded_per_stream_v1`). | No duplicate canonical equation registered. |
| `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `OD9/AM1/IG1/TR1/pk1/shared context` | Found routing context and prior shared-context references, but no completed PA2 receipt. | Scoped negatives to these persisted OD9 streams and kept shared carrier families open. |

## Follow-On Disposition

| follow-on | disposition | fire order |
|---|---|---|
| AM1-smooth-residual-packet on OD9 stage2 carriage | FIRED_AND_CLOSED for this stream | Reopen only for a different persisted stream whose residuals are actually smoother, or for a true xi/dxi packet not present in OD9's manifest. |
| IG1-F2 vanishing-condition A/B | FIRED | Plain shared context beats independent; xi-conditioned ordering loses to plain shared. Use as modest evidence for shared parametrization, not as a shipping packet. |
| pk1 boundary grammar | QUEUED/ALREADY-FIRED in hot state | PA2 does not alter pk1's gate: receiver-closed packet first, scorer only if representation projects at or below the corridor gate. |
| TR1 learned carrier | ROUTED-STRONGER-IF-PK1-FAILS | Shared-context win supports TR1 direction, but the flat stream remains rate-dead; TR1 becomes primary if pk1 cannot beat pe3/corridor at representation. |

## Next If Resumed

1. Do not rerun AM1 delta2 on OD9 stage2 qcoeffs; the falsifier fired under both coders.
2. Do not promote the shared-context packet to a score candidate; best projected bytes are still over 1.2 MB n600.
3. Feed the modest shared-context win into pk1/TR1 routing only. If pk1 lands under the corridor, PA2 is supportive context; if pk1 fails representation, TR1 learned/shared carrier becomes cleaner.
4. If future persistence emits true xi/dxi or a lower-entropy shared/task-description stream, rerun the same flat/delta1/delta2 and independent/shared/xi-conditioned races on that new stream with exact decode equality.

```json
{
  "schema": "ddm_pa2_summary.v1",
  "axis": "[macOS-CPU byte-only persisted-native pricing]",
  "score_claim": false,
  "scorer_forwards_run": 0,
  "probe1": {
    "winner": "flat",
    "brotli_q11_bytes": {"flat": 1917, "delta1": 2014, "delta2": 2172},
    "lzma1_raw_bytes": {"flat": 2161, "delta1": 2303, "delta2": 2477},
    "verdict": "delta2 loses to delta1; acceleration prior dead on OD9 stage2_qcoeffs"
  },
  "probe2": {
    "winner": "shared_context",
    "brotli_q11_bytes": {"shared_context": 66497, "xi_conditioned_shared": 66541, "independent_per_pair": 70472},
    "lzma1_raw_bytes": {"shared_context": 67946, "xi_conditioned_shared": 68047, "independent_per_pair": 69013},
    "verdict": "shared context wins modestly; xi-conditioned ordering loses to plain shared"
  },
  "json_artifact": ".omx/research/ddm_pa2_20260805/PA2_RECEIPT.json"
}
```

## Boundary

Measured in PA2: scorer-free coder bytes, exact decode equality for every row, and OD9 SSD manifest SHA/byte verification.

Not measured: d_seg, d_pose, n600 archive bytes, receiver-closed public-wire survival, `upstream/evaluate.py`, contest-CPU, contest-CUDA, or any new exact score.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
