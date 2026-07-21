# DAG FEED — M1 byte-close first n600 row

`feed_id=FEED-M1-BYTECLOSE-FIRST-ROW-20260721` · `lane_id=m1_byteclose_closer` · `research_only=true` · `[macOS-CPU advisory]` · pointer unchanged

## Typed nodes

| node | evidence type | state |
|---|---|---|
| `T_M1` | completed M1 training | MEASURED 900/900 steps; three one-epoch stages; EMA; windowed-curvelet carrier |
| `A_base` | counted 83,838-byte base archive/materialization | MEASURED raw SHA `8565df10...`; no hard-oracle row inferred |
| `A_ctl` | settled counted control archive | MEASURED 94,344 B; SHA-bound |
| `A_M1` | exact counted composed candidate | MEASURED 90,566 B; SHA `a386a854...` |
| `R_1`, `R_2` | independent receiver decodes | MEASURED 753.163059 s and 738.52 s; identical 3,662,409,600-byte raw |
| `I_R` | control/candidate receiver identity | MEASURED exact SHA equality `dbfcdcfa...` |
| `M_M1` | unchanged hard CPU-Torch aggregate | MEASURED n600: `d_seg=.003515794640406966`, `d_pose=127.36588287353516`, advisory `S=36.10024069328613` |
| `D_M1` | canonical class/tail decomposition | MEASURED candidate; integer-closed 414,740/117,964,800 mismatches; 38 resumable batch rows |
| `D_ctl` | control class/tails | DERIVED EXACT from `I_R` plus deterministic scorer; aggregate independently settled |
| `G_rate` | counted rate and runtime gates | PASS |
| `G_seg` | `d_seg<=3.39e-4` | FAIL 10.3711× limit |
| `Q_578c` | Modal exact-eval candidate status | ELIGIBLE FOR MAIN REVIEW; not claimed or dispatched |
| `V_form` | first-firing efficacy verdict | FORMULATION-NEGATIVE vs scored control; family remains open |

## Authority-preserving edges

```text
T_M1 --EMA export + counted compile--> A_M1
A_base --materialization receipt--> raw_base SHA=8565df10...
A_M1 --candidate decode--> raw_M1 SHA=dbfcdcfa...
raw_base != raw_M1; candidate is +6,728 B vs base; no base score equality inferred

A_ctl --settled exact decode--> raw_ctl --SHA--> I_R
A_M1 --decode workers=8 run 1--> R_1 --SHA-->
A_M1 --decode workers=8 run 2--> R_2 --SHA--> I_R

A_M1 --unchanged control harness
     {seed=1234,batch=16,CPU threads=8,same GT/scorers}
     --> M_M1

M_M1 --official DistortionNet hook, no scorer mutation
     --> D_M1 --integer/class/pair closure--> G_seg

I_R + D_M1 + deterministic scorer --> D_ctl [DERIVED EXACT]

archive delta=-3,778 B + distortion deltas=0
     --> advisory delta_S=-0.0025156151249
     --> G_rate=PASS, G_seg=FAIL
     --> V_form=FORMULATION_NEGATIVE_EFFICACY
     --> Q_578c=MAIN_REVIEW_ELIGIBLE_ONLY
```

No edge permits the local advisory action to become a contest score, promotion, operator-GO, or pointer mutation. The inherited harness’s “R1b candidate absent” wording is scoped to its embedded boundary-coordinate schema; it does not erase the measured whole-archive M1 candidate row.

## Empirical update to the solver stack

The candidate changes all 4,800 fp16 code elements, zeros half of them, changes all 12 quotient-head values, and saves 3,778 archive bytes versus the scored control, while changing zero receiver bytes. It is 6,728 B larger than the distinct 83,838-byte base materialization. This is a measured realized-decode dead-zone and firm efficacy-negative result for the first three-epoch positive-anisotropic formulation.

- **Sensitivity map:** prioritize Lane first only through the registered Fisher/margin and realization-necessity field. Lane contributes `0.0012633260091145834` overall and has `0.21578277508220647` conditional error. Preserve the measured Seg tail indices `522,515,572,517,518,510,74,566` and Pose tail indices `523,21,90,1,7,24,49,41` as exact diagnostic targets.
- **Pareto constraint:** candidate passes `150.9433<=477.8 B/pair`, `90,566<=216,222 B`, and `753.163059<=1,800 s`, but fails `d_seg<=3.39e-4`; joint admission remains false. The harness’s legacy 216,223-byte field is not used for the mission verdict.
- **Bit allocator:** do not blanket-widen. Require an actual receiver-byte change, rank on Fisher/top1-top2 margin with the corrected realized inner Jacobian, and stop at marginal `Delta S/byte < 25/37,545,489`. The present compressibility improvement is retained; efficacy spending must escape the measured decode dead zone.
- **Cathedral/autopilot:** emit a composed-candidate review item for #578(c), `dispatch_authority=MAIN_ONLY`, with exact archive/raw/receipt hashes. Do not auto-promote or move the pointer.
- **Continual learning:** ingest the repository JSON receipt plus external exact/decomposition receipts. The typed lesson is `payload_changed=true`, `receiver_changed=false`, `rate_improved=true`, `d_seg_improved=false`, `d_pose_delta=0`.
- **Probe disambiguator:** the next bounded A/B is `{longer band_fit, band-width tuning, receiver-closed eval_roundtrip STE, r1b7 fixed-positive magnitude, hotter schedule}` under the same counted receiver, with measured reverse-waterfill as allocator. The existing training loss already consumes `torch_uint8` through `Uint8STE`; the owed STE is the full receiver-closed roundtrip, not another claim that all uint8 STE is absent. Interpretations remain live until receiver actuation and rate marginal decide.

## Triality

- **DSL:** consumes the existing typed positive-anisotropic M1 config and sealed `warmup → band_fit → rate_polish` stages. No new control flag or loss schedule is introduced.
- **DAG:** closes the exact archive/decode/scorer/decomposition/gate chain and exposes `Q_578c` without granting dispatch authority.
- **Equations:** consumes the existing task action and registered Fisher/margin, corrected inner-Jacobian, curvelet/shearlet, xi-factorization, and reverse-waterfill laws. No new equation is registered from one instance.

## Reactivation predicate

MAIN may route #578(c) because the exact local advisory action is lower. The M1 efficacy lane itself should reactivate only with a counted reformulation whose independently decoded receiver output changes, whose full n600 hard row improves the relevant debt, and whose marginal bytes remain above the registered rate waterline. A longer run that merely changes latent codes while remaining raw-identical does not close `G_seg`.

`verdict_scope=FORMULATION-NEGATIVE efficacy: first three-epoch positive-anisotropic M1 firing is receiver-identical to the scored control; broader banded-generator and windowed-curvelet families remain open.`

## MAIN landing requirement

MAIN must review and merge this branch before the feed is canonical, verify the external SHA-bound receipts, and separately claim any dispatch lane. This feed authorizes no paid dispatch, contest-axis claim, promotion, or pointer mutation.
