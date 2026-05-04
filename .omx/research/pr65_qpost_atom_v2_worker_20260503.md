# PR65 QPost Atom V2 Worker - 2026-05-03

Evidence grade: `byte_trace_planning_only_until_exact_cuda`.
Score claim: `false`.
Remote dispatch: `false`.

## Scope

Local-only expansion of PR65/Henosis qpost atom sidecar candidates on the
C-089/QP1 frontier archive.  The worker generated deterministic byte-closed
archives under the robust runtime qpost hook, but did not run training, exact
eval, or remote GPU dispatch.  Any future exact eval must first claim a lane
with `tools/claim_lane_dispatch.py claim ...` and then run the canonical CUDA
auth path on the exact archive bytes.

## Code And Artifacts

- Tool: `experiments/build_pr65_qpost_atom_candidates_v2.py`
- Tests: `src/tac/tests/test_build_pr65_qpost_atom_candidates_v2.py`
- Result summary:
  `experiments/results/pr65_qpost_atom_v2_worker_20260503/candidate_summary.json`
- Core builder summary:
  `experiments/results/pr65_qpost_atom_v2_worker_20260503/candidates/candidate_summary.json`
- Per-candidate archives and `v2_manifest.json` overlays:
  `experiments/results/pr65_qpost_atom_v2_worker_20260503/candidates/*/`

## Inputs

- C-089 source archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- C-089 SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- C-089 bytes: `276342`
- C-089 exact score used only as break-even baseline: `0.3154707273953505`
- PR65/Henosis archive:
  `experiments/results/top_submission_delta_reverse_engineering_20260503/sources/pr65_henosis_archive.zip`
- PR65 SHA-256:
  `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- PR65 head SHA:
  `a8b53b5280ee8f05db65740cd48cf7c321a55497`
- Pair ranking inputs:
  C-089 component trace minus PR65 compatibility component trace, restricted to
  positive public-trace opportunity and PR65 qpost-active pairs.

## V2 Expansion

Generated `31` deterministic archives across:

- bias-only qpost atoms;
- bias+region atoms;
- post+bias atoms;
- post+region+bias atoms;
- shift/frac motion atoms;
- bias+shift atoms.

Only bias-only candidates were allowed into the primary screen.  Post, region,
and motion/fraction streams remain diagnostic or blocked because prior global
qpost exact ablations were negative and the current evidence is only trace
planning.

## Primary Candidate

Recommended single exact-screen candidate, if an operator chooses to spend a
lane after claiming it:

```text
candidate: v2_pr65_qpost_bias_poseadv_top040
archive: experiments/results/pr65_qpost_atom_v2_worker_20260503/candidates/v2_pr65_qpost_bias_poseadv_top040/archive.zip
bytes: 276561
sha256: cc44cb245e7772076fd136d457e4b4f549c726db9985a826de2d64b5e29db51e
qpost container bytes: 125
selected active atoms: 40
public-trace opportunity bound: 0.001933836193009743
break-even component gain for sub-0.314: 0.001616550506084235
trace slack versus target: 0.00031728568692550784
```

Rationale: top040 is the first bias-only candidate that clears the configured
minimum planning slack (`0.00020`) while staying under the primary pair cap
(`64`).  The smaller top028 and top032 siblings clear the raw break-even but
have thinner trace slack; larger top048/top056/top064 have more slack but
change more pairs.

Aggressive follow-up, only after the primary screen:

```text
candidate: v2_pr65_qpost_bias_poseadv_top080
archive: experiments/results/pr65_qpost_atom_v2_worker_20260503/candidates/v2_pr65_qpost_bias_poseadv_top080/archive.zip
bytes: 276600
sha256: 5bb174735af9db4f32963933b4c6279777da2b0d90a16332ca339475eabf4da7
qpost container bytes: 164
trace slack versus target: 0.0006789241649128143
```

## Bias-Only Candidate Screen

| candidate | bytes | qpost bytes | public-trace opportunity | trace slack vs 0.314 | SHA-256 | class |
|---|---:|---:|---:|---:|---|---|
| `v2_pr65_qpost_bias_poseadv_top024` | `276547` | `111` | `0.0014933407043021267` | `-0.00011388777643839797` | `10037f92c2cb5b486702c8a8655511fd36dfb03ab7bb475786d1e0b2319e5a23` | `do_not_dispatch` |
| `v2_pr65_qpost_bias_poseadv_top028` | `276546` | `110` | `0.0016262789238011164` | `0.00001971630201371387` | `02a384deca55590d3b359d4643adcc333618b76baea99257c852e254686425eb` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top032` | `276551` | `115` | `0.0017422633563209306` | `0.00013237143976791723` | `91cd1690fdcca47eedbe12dc8bdaba8191210199169449c474730ad30831efbd` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top040` | `276561` | `125` | `0.001933836193009743` | `0.00031728568692550784` | `cc44cb245e7772076fd136d457e4b4f549c726db9985a826de2d64b5e29db51e` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top048` | `276574` | `138` | `0.0020706116447299725` | `0.00044540497225514916` | `a0fec5615a29cf04db6f8422c9c33cc6e4db36851deb4a58684370c5095f49b4` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top056` | `276583` | `147` | `0.0021706635987927443` | `0.0005394641957398214` | `1ee4eaf62e9c3d3b46d54de86ca3bb89d89e3b93a9320842aec19a17604ea854` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top064` | `276590` | `154` | `0.002247988534515191` | `0.000612128118790413` | `656d438c8723482e3500df33d4b7a02d3168b1f42004dfdec4d97ae37281029c` | `primary_bias_only_exact_screen_candidate` |
| `v2_pr65_qpost_bias_poseadv_top080` | `276600` | `164` | `0.0023214431701688142` | `0.0006789241649128143` | `5bb174735af9db4f32963933b4c6279777da2b0d90a16332ca339475eabf4da7` | `aggressive_bias_only_after_primary_screen` |
| `v2_pr65_qpost_bias_poseadv_top096` | `276604` | `168` | `0.0023231006749742248` | `0.0006779182339057364` | `12d810b0ca4efa1c0b907111564fdd3b6bbcf652b2ac049855d4e785c072c9bf` | `aggressive_bias_only_after_primary_screen` |
| `v2_pr65_qpost_bias_poseadv_top128` | `276604` | `168` | `0.0023231006749742248` | `0.0006779182339057364` | `12d810b0ca4efa1c0b907111564fdd3b6bbcf652b2ac049855d4e785c072c9bf` | `aggressive_bias_only_after_primary_screen` |

`top096` and `top128` converge to the same archive because the positive
trace-filtered bias atom set is exhausted at `82` active selected pairs.

## Verification

```text
.venv/bin/python -m py_compile experiments/build_pr65_qpost_atom_candidates_v2.py src/tac/tests/test_build_pr65_qpost_atom_candidates_v2.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr65_qpost_atom_candidates_v2.py -q
3 passed in 0.11s
```

No remote GPU, training, or eval job was dispatched.
