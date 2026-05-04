# PR65 QPost Atom Worker - 2026-05-03

Evidence grade: `byte_trace_planning_only_until_exact_cuda`.
Score claim: `false`.
Remote dispatch: `false`.

## Scope

Local-only PR65/Henosis qpost atom extraction and C-089 archive builds.
The worker treats PR65 postprocess streams as charged atom dictionaries, not as
a wholesale transplant.  Any score truth still requires exact CUDA auth eval on
the exact archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Inputs

- C-089 A++ source archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- C-089 SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- C-089 score: `0.3154707273953505`
- PR65 Henosis archive:
  `experiments/results/top_submission_delta_reverse_engineering_20260503/sources/pr65_henosis_archive.zip`
- PR65 SHA-256:
  `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- PR65 head SHA:
  `a8b53b5280ee8f05db65740cd48cf7c321a55497`
- Pair ranking:
  C-089 component trace minus PR65 compatibility component trace, filtered to
  PR65 qpost-active pairs with positive public-trace opportunity.

## Code Landed

- `experiments/build_pr65_qpost_atom_candidates.py`
- `src/tac/tests/test_build_pr65_qpost_atom_candidates.py`

The builder:

- verifies C-089 and PR65 source SHA custody;
- verifies `inflate.sh` has the counted `qpost.bin` runtime hook;
- decodes PR65 qpost streams into pair-local atoms;
- omits `randmulti` because no reviewed sparse pair-filter encoder exists;
- fails closed on no-op selected subsets, PR65 source SHA drift, missing runtime
  hook, unsupported streams, and inherited full/global qpost shapes;
- records qpost stream SHA, filtered stream SHA, selected pairs, active atom
  counts, charged bytes, and exact-eval command templates;
- emits no remote dispatch.

## Prior Exact Negative Context

Global qpost streams are not safe standalone transplants:

| shape | score | bytes | pose | seg | SHA |
|---|---:|---:|---:|---:|---|
| global bias | `0.31571526640225916` | `276776` | `0.00049575` | `0.00061012` | `fef5ec9eafcd5844df169f2f36076eb2f9e11dd2cffabd9fb75ccb68b370008b` |
| global region | `0.31618813549225566` | `276826` | `0.00050196` | `0.00061012` | `7591ccdebc775d14584425f1493b7ec1041578a5e1ce7829934b622e3641ab4a` |
| global post | `0.32611385442467516` | `277953` | `0.0006546` | `0.00060129` | `b3e1ec5a60d16c374c4930572b997acdc2e5d3cb643ba3eb58c35af8db745179` |
| full PR65 qpost sidecar | `0.32247639916142556` | `282818` | `0.00054805` | `0.00060129` | `ce65a1d22276a3c13334682f065bd6595b11c36f54d95233e813754d232c7957` |

This is why the worker only emits tiny pair-filtered atom subsets and marks
post/region/motion streams as component-risky unless exact evidence says
otherwise.

## Real C-089 Builds

Default screen:

| candidate | bytes | delta | qpost bytes | public-trace opportunity | rate delta | SHA-256 | decision |
|---|---:|---:|---:|---:|---:|---|---|
| `pr65_qpost_bias_poseadv_top008` | `276521` | `+179` | `85` | `0.0006731328932252625` | `0.00011918875260886867` | `e6b018ead2b14c37dd7f13b8acab7644746140bee1650a2655c7854cf7b8f8cf` | `do_not_dispatch` |
| `pr65_qpost_bias_region_poseadv_top008` | `276565` | `+223` | `129` | `0.0006731328932252625` | `0.00014848654654624422` | `f448b3bda601c07a6fb203dceeb437325d7157661ef84b64af6a5b7090bbbf44` | `do_not_dispatch` |
| `pr65_qpost_post_bias_poseadv_top004` | `276563` | `+221` | `127` | `0.00036688274820657766` | `0.00014715482863999986` | `abca159885bae9a30bd261cb71d1d54ccc9b710e28edf703e80e660bf1744296` | `do_not_dispatch` |
| `pr65_qpost_post_region_bias_poseadv_top004` | `276594` | `+252` | `158` | `0.00036688274820657766` | `0.0001677964561867872` | `7d3c639678399b3713c36d5cfcee0225056454f1741fc22159a2c8be5eaeafc5` | `do_not_dispatch` |
| `pr65_qpost_shift_frac_poseadv_top004` | `276597` | `+255` | `161` | `0.0003649106715766208` | `0.0001697940330461537` | `bcfab367c98da99a8fdd1ede34056e05ea637572675d2fcf2b703984c28a065a` | `do_not_dispatch` |

Expanded bias screen:

| candidate | bytes | delta | qpost bytes | public-trace opportunity | rate delta | SHA-256 | decision |
|---|---:|---:|---:|---:|---:|---|---|
| `pr65_qpost_bias_poseadv_top016` | `276533` | `+191` | `97` | `0.001141575829399205` | `0.00012717906004633474` | `aac4fdebc06354e360132cceb586071fe179ddfd430d4c943021663a7a9efc1e` | `do_not_dispatch` |
| `pr65_qpost_bias_poseadv_top032` | `276551` | `+209` | `115` | `0.0017422633563209306` | `0.00013916452120253382` | `91cd1690fdcca47eedbe12dc8bdaba8191210199169449c474730ad30831efbd` | `exact_cuda_eval_candidate_after_lane_claim` |
| `pr65_qpost_bias_poseadv_top064` | `276590` | `+248` | `154` | `0.002247988534515191` | `0.0001651330203742985` | `656d438c8723482e3500df33d4b7a02d3168b1f42004dfdec4d97ae37281029c` | `exact_cuda_eval_candidate_after_lane_claim` |
| `pr65_qpost_bias_region_poseadv_top016` | `276587` | `+245` | `151` | `0.001141575829399205` | `0.000163135443514932` | `75467167719f6301880606ad7fb88b6a927e82fe54716ee996c4a06ef6c4a115` | `do_not_dispatch` |
| `pr65_qpost_bias_region_poseadv_top032` | `276624` | `+282` | `188` | `0.0018161651762220522` | `0.0001877722247804523` | `4a00a549425cab15e51eb903bd5d85479091dd2e207fc76fa9f92692a087c6d1` | `do_not_dispatch` |
| `pr65_qpost_post_bias_poseadv_top016` | `276680` | `+338` | `244` | `0.00114578263065797` | `0.00022506032615529392` | `146e038fb89934a667727cae11e35ef3c9f27f6e3907e3434d190bbd28f7dd38` | `do_not_dispatch` |

## Recommended Exact Candidate

The cleanest qpost candidate is:

```text
candidate: pr65_qpost_bias_poseadv_top032
archive: experiments/results/pr65_qpost_atom_worker_20260503/expanded/pr65_qpost_bias_poseadv_top032/archive.zip
bytes: 276551
sha256: 91cd1690fdcca47eedbe12dc8bdaba8191210199169449c474730ad30831efbd
qpost container bytes: 115
filtered bias stream bytes: 79
selected active atoms: 32
public-trace opportunity bound: 0.0017422633563209306
break-even component gain for sub-0.314: 0.0016098919165530134
```

This is still planning evidence.  It is worth at most one fast exact CUDA
screen if the operator wants to spend a T4/L40S diagnostic slot, but it must
first claim a lane and preserve exact custody.  The top64 sibling has a larger
trace bound but also more atoms and more risk.

## Artifacts

- `experiments/results/pr65_qpost_atom_worker_20260503/candidate_summary.json`
- `experiments/results/pr65_qpost_atom_worker_20260503/build_stdout.json`
- `experiments/results/pr65_qpost_atom_worker_20260503/expanded/candidate_summary.json`
- `experiments/results/pr65_qpost_atom_worker_20260503/expanded/build_stdout.json`
- Per-candidate `manifest.json` files under the same result directories.

## Verification

```text
.venv/bin/python -m py_compile experiments/build_pr65_qpost_atom_candidates.py src/tac/tests/test_build_pr65_qpost_atom_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr65_qpost_atom_candidates.py -q
4 passed in 0.10s
```

No remote GPU, training, or eval job was dispatched.
