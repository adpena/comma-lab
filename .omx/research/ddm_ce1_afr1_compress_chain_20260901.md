# ddm_ce1 AFR1 compression chain — exact end-to-end rebuild

**Verdict: PASS.** The single compression entry point now expresses the shipping AFR1
lossless chain from the retained rc2 Stage-A boundary. Two complete chain invocations
produced the exact submitted archive: **180,002 B**, SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
The two retained finals are byte-identical. Axis:
`[macOS-CPU advisory / scorer-free EXACT byte measurement]`; `score_claim=false`;
`promotable=false`. No scorer, Modal job, submission, or frontier update ran.

The authority is the machine receipt at
`/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/RESULT_pq2_e2e.json`
(SHA-256 `5814eb93969d86cf234d95852e617546b2b18fa66fb5c5310f8fcfbe948c9564`).
It records runner SHA-256
`5fe3a8ca94c4b16163057c1c6774ebaffac151b8f63b6f6ee93649f7a6ea4add`,
base Git commit `bb42e6eba8bad03c172a6539e53fd9ed5378cf69`, seed 1234, all 15/15
content-pinned inputs, the command ledger, all stage outputs, and both final archives.

## RECALL EVIDENCE

I searched the full original-work corpus rather than accepting the charter's linearized
seed. Queries included `afr1|tile48_groupbin8|lb1|jt21|jt22|jt23|gb1`, every archive
SHA prefix, `compress|reproduction|NOT_EXPRESSIBLE`, `rc64.*role`, and `rate corner`
across `.omx/research/`, the canonical research indexes/DAG FEED blocks, directives, and
the task ledger. I also ran `tools/list_canonical_equations.py --json` and inspected the
actual tools and argparse surfaces named below.

The source set included the five stage receipts in the lineage table; the JT22/JT23
verdicts; `ddm_jt21_joint_21family_reencode_verdict_20260825.md`;
`ddm_directive_provenance_three_way_20260901.md`;
`ddm_directive_synthesis_path_20260901.md`; the live hot state; the packet directory;
and `reverse_engineering/rc64_backend_role_registry.json`.

What changed beyond the charter seeds:

- GB1 is a fork, not a single additive pointer step. The pointer branch is
  `groupbin8_surprise` at 180,215 B, while LB1 consumes the separately retained JT21
  21-family joint bank at 180,192 B. The registry and runner now type both outputs.
- The admitted JT21 runtime carries both `groupbin8_surprise` and `cls_groupbin8`.
  Treating `cls_groupbin8` as a singleton stage was falsified during this arm and is not
  the shipping mechanism.
- JT22's best exact result was only -1 B and was refused at its fire bar; JT23 found 0 B
  collectible. Neither is a separate admitted stage and neither is inside LB1 under a
  hidden name. The JT21 bank is the consumed base; LB1 adds `patch192_only`.
- The September 1 directives change packet wording/accounting and operator gates, but
  add no archive transform to this chain. No canonical equation changed the exact
  byte recipe.

## Traced lineage

Every SHA and byte count below was checked against the named receipt and then reproduced
from disk. The rc2 boundary is a declared scope reduction: content-deciding training,
semantic edit solve/admission, and pose re-solve remain documented Stage-A provenance.

| stage | real tool | input archive | output archive | receipt |
|---|---|---|---|---|
| rc2 boundary | retained Stage-A custody | — | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`, 180,456 B | `.omx/research/ddm_rc2_t4_row_sixteenth_move_20260820.md` |
| FX5 widened causal corrector | `experiments/ddm_fx5_build_e1_runtime.py` + physical `ddm_jg2_tail_reencode.py` | `df7fd266…`, 180,456 B | `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`, 180,386 B | `.omx/research/ddm_fx5_composed_rate_candidate_20260821.md` |
| DX2 CABAC coefficient fold | `experiments/ddm_dx2_cabac_receiver_fold.py` | `4b54fccc…`, 180,386 B | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B | `.omx/research/ddm_dx2_cabac_receiver_fold_20260821.md` |
| GB1 collection, pointer branch | `experiments/ddm_gb1_groupbin8_conditioning.py` + physical `ddm_jg2_tail_reencode.py` | `976f706d…`, 180,368 B | `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`, 180,215 B | `.omx/research/ddm_gb1_groupbin8_conditioning_20260824.md`; `ddm_gb1_groupbin8_verdict_20260824.md` |
| GB1 collection, JT21 bank consumed by LB1 | same GB1 collection; admitted 21-family runtime + physical `ddm_jg2_tail_reencode.py` | `976f706d…`, 180,368 B | `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`, 180,192 B | `.omx/research/ddm_lb1_banked_lossless_joint_collect_20260829.md` lines 22, 41, 60-69 |
| LB1 banked joint `patch192_only` | `experiments/ddm_lb1_banked_lossless_joint_collect.py` + physical `ddm_jg2_tail_reencode.py` | `ec0dd68f…`, 180,192 B | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9`, 180,083 B | `.omx/research/ddm_lb1_banked_lossless_joint_collect_20260829.md` |
| AFR1 `tile48_groupbin8` | `experiments/ddm_afr1_tile48_receiver_identity.py` + physical `ddm_jg2_tail_reencode.py` | `5b856e66…`, 180,083 B | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 B | `.omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md` |

## Chain wiring and test receipt

`experiments/ddm_pq2_compress_e2e.py` now contains one ordered `AFR1_CHAIN` registry,
one input-role registry, and one fail-closed executor behind `--stage chain`/`--chain
afr1`. It hashes all 15 inputs before work. The rc64 encoder source remains role-pinned
to `5c75e2c7…`; the shipped decoder-only member remains separately role-pinned to
`05839d14…`. No private custody path is compiled into the new registry; paths enter via
the existing `--inputs-json`/environment contract.

Each transform runs its landed apparatus and then hashes the newly generated archive.
AFR1's encoding runtime is not a retained output copy: each run copies its freshly
reproduced LB1 runtime, installs the exact `6462ba51…` Python corrector supplied and
revalidated by the AFR1 identity tool, checks the LB1 archive handoff, and performs a
full n600 encode. Stage-tool failures and archive divergences persist typed refusals.
Resume propagates to the physical encoder and restores the pre-FX5 pointer body after
the post-stage repin.

CI-safe test: `.venv/bin/pytest -q experiments/tests/test_ddm_pq2_compress_e2e.py` →
**3 passed**. The tests cover ordered/pin-consistent registry shape, tool and receipt
existence, the GB1 fork, rc64 roles, the AFR1 source-runtime bridge, removal of AFR1 from
`NOT_EXPRESSIBLE`, and resume propagation. Ruff and `py_compile` also pass.

Two genuine code reviews are recorded through `tools/review_tracker.py` for all 31
entities in the runner and test module. Structural pass 1 found and cured missing resume
propagation, untyped child failure, post-repin FX5 resume, opaque AFR1 runtime staging,
and missing runner provenance. No-fake/custody pass 2 independently checked that
reference archives are diagnostics only, that all retained candidates originate at real
tool outputs, that the GB1 fork is not flattened, and that `cmp` finds the two final
archives identical. **Assumption challenge:** the initial shared assumption was that the
post-DX2 lineage was a linear succession of singleton correctors. Violating it exposed
the actual GB1 pointer/JT21-bank fork and was necessary to reproduce the submitted bytes;
retaining the linear assumption would have institutionalized the 180,268-byte false row.

The bounded global developer preflight ran 24 reported checks. Twenty-three were green;
the sole failure was `check_lane_pre_registered_before_work_starts` on two pre-existing
`lane_lines` references in unrelated
`experiments/ddm_dds1_decoder_derivable_born_stats.py:298,341`. That file was outside
this arm and was not edited. This is a scoped repository blocker, not a CE1 green claim;
the serializer's changed-file hooks remain the commit authority for this landing.

## End-to-end run receipt

Command form:

```text
.venv/bin/python experiments/ddm_pq2_compress_e2e.py --stage chain --resume \
  --store /Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain \
  --inputs-json .omx/tmp/codex_runs/ddm_ce1_afr1_compress_chain_inputs.json \
  --expected-archive-sha256 cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25 \
  --expected-archive-bytes 180002
```

The final invocation resumed exact encoder checkpoints produced earlier in this same
run store. Run 1's predecessor stages resumed near their terminal boundary, then its new
AFR1 bridge encoded cold. Run 2 resumed FX5/GB1, completed LB1 from its saved frame-50
control, and encoded LB1 plus both AFR1 control/candidate streams cold. Thus the table's
wall seconds describe the admitted invocation; zero means an exact, already-retained
stage output was pin-verified at its stage boundary, not that a substitute archive was
copied into the next stage.

| run | FX5 s | DX2 s | GB1 s | LB1 s | AFR1 s | total s | final |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 54.912 | 0.000 resumed | 89.699 | 61.610 | 755.456 | 962.153 | 180,002 B, `cbb8d928…` |
| 2 | 55.675 | 0.000 resumed | 89.084 | 1,352.877 | 1,437.277 | 2,935.382 | 180,002 B, `cbb8d928…` |

Both runs reproduced these stage identities: FX5 `4b54fccc…`/180,386 B; DX2
`976f706d…`/180,368 B; GB1 pointer `ba1f3830…`/180,215 B; consumed JT21 bank
`ec0dd68f…`/180,192 B; LB1 `5b856e66…`/180,083 B; AFR1 `cbb8d928…`/180,002 B.
The determinism check reports `byte_identical=true` and `first_differing_offset=null`.

The AFR1 tool also revalidated native/Python context parity at 196,608/196,608
positions and full receiver identity at 600 frames, 117,964,800 tokens, and
3,662,409,600 raw bytes. Those are identity checks, not a scorer run.

### Bearing falsifiers

- The first honest wiring used the standalone `cls_groupbin8` runtime as JT21. It
  refused at GB1: observed 180,268 B / `bd1c663f05951f6fb4399428310d3527dc2986841f0f5355afdb2063f9b53692`,
  expected 180,192 B / `ec0dd68f…`, first differing offset 14. The intact refusal is at
  `/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain_attempt1_refused/RESULT_pq2_e2e.json`
  (SHA-256 `1f811a9df666dcf5cbf7430623b016702481267d207e5c1b81b3cb8b61c39032`).
  This exposed the 21-family joint-runtime fork and changed the recipe.
- Review then found that outer `--resume` was not reaching the checkpointed encoder,
  existing builders refused valid stage directories, and FX5 resume saw the post-stage
  repin rather than rc2. The final runner propagates resume, records verified stage
  resumes, and restores the hash-pinned rc2 handoff before continuing FX5. The admitted
  invocation exercised those cures.
- Review also rejected an opaque pre-staged AFR1 encoder runtime. The admitted path
  creates the runtime inside each run from the reproduced LB1 body and the identity
  tool's exact corrector source.

## Retained custody

Manifest:
`/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/RETENTION_MANIFEST.json`,
4,448 B, SHA-256
`55b448f89ab14b9862d9a2fd6c5ec41c4dccb7c9fe24b022f0263a11e36f2d15`.
It enumerates 26/26 files and 2,219,836 B: the 12 intended archives below plus 14
APDataStore AppleDouble metadata sidecars of 4,096 B each. The sidecars are recorded
honestly, are not archive inputs, and do not alter any candidate SHA.

| retained relative path (under each `run_N/`) | bytes | SHA-256 |
|---|---:|---|
| `01_fx5.zip` | 180,386 | `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` |
| `02_dx2.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| `03a_gb1_pointer.zip` | 180,215 | `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` |
| `03b_jt21_bank.zip` | 180,192 | `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3` |
| `04_lb1.zip` | 180,083 | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` |
| `05_afr1.zip` | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |

Every row exists under both `retained/run_1/` and `retained/run_2/` with the same SHA.
The work tree additionally preserves the per-frame bit ledgers, real streams, RC64
builds, immutable encoder checkpoints, and named-tool receipts used to resume.

## Proposed packet text for MAIN

Replace the compression-script answer with:

> Yes. `experiments/ddm_pq2_compress_e2e.py` deterministically rebuilds the exact
> submitted 180,002-byte archive, SHA-asserts every admitted lossless stage and a second
> complete final build, and includes the content-deciding solve stages as their own
> scripts with receipts.

This is a proposal only. The sealed generation-7 packet, runtime tree, public draft,
and `upstream/` were not edited.

| packet surface | proposed tight cut/update for MAIN |
|---|---|
| `PR_BODY_FINAL_DRAFT_TIGHT.md:31-33,81-94` | Keep unqualified “Yes to both”; add the one verification sentence above; remove the now-discharged internal CE1 contingency block before publication. |
| `PR_BODY_FINAL_DRAFT.md:48-54` | Replace the five-stage limitation paragraph with the verified answer; do not carry two competing PR answers. |
| `README_PUBLIC.md:73-80,105-111` | Replace the generation-3-only refusal boundary and “expected to refuse” command note with the CE1 exact command/result. |
| `REPORT_PUBLIC.txt:63-67` | Replace “not re-run for AFR1” and exact-SHA refusal with the CE1 result and scope boundary at retained rc2. |
| `PACKET_TARGET.json:53-57` | Set AFR1 end-to-end verification true and point at the CE1 result/runner hashes. Preserve the score and custody blocks. |
| `BORROWED_SUBSTRATE_ACCOUNTING.md:678` | Remove the exact-SHA refusal sentence; keep the borrowed Stage-A versus ours-original lossless accounting. |
| `GAP_REPORT.md:52` | Retire the stale generation-0 hash/“answer no” row under the live overlay; keep publication and policy holds separate. |
| prose hash literals across the packet | Keep the full AFR1 SHA in machine authorities (`PACKET_TARGET.json`, archive manifest, CE1 receipt); use `cbb8d928…` in explanatory prose to remove duplicate drift surfaces. |

## Honest residuals and dispositions

AFR1 is now expressible. These non-shipping candidates remain honestly refused because
their content-deciding builders are outside the lossless chain grammar:

- `rc2` (`df7fd266…`): inherits the JG5 semantic/pose solve chain and additionally needs
  the RR5 lossless arithmetic-basis builder named in `ddm_rc2_t4_row_sixteenth_move_20260820.md`.
- `jg5` (`f3bce5d2…`): requires `experiments/ddm_jg3_joint_solve.py`, the JG4 splice,
  `experiments/ddm_jg5_pose_resolve_on_edited_renders.py`, and
  `experiments/ddm_up3_carrier_splice.py::build_archive` plus the damped pose solve.
- `ck1` (`35c318d5…`): requires `experiments/ddm_ck1_build_composed_archive.py` and
  `experiments/ddm_ck1_pose_resolve_kneeA.py`.

Follow-ons are not left “noted”:

| disposition | owner | consumer store | fire trigger |
|---|---|---|---|
| `QUEUED-WITH-A-FIRE-ORDER` — apply the proposed tight compression text and retire stale refusal claims | MAIN | `.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_FINAL_DRAFT_TIGHT.md` and sibling packet docs | CE1 serializer commit is present on MAIN and its scoped checks remain green |
| `QUEUED-WITH-A-FIRE-ORDER` — publication decision remains operator-gated | operator/MAIN | task rows 1111 and 1363; owning packet directory | operator clears both submission-hold and contest-policy gates |

**Own-vehicle frontier: AFR1 remains S 0.14797617125559104 @ 180,002 B
`[contest-CUDA T4 n600]`; unchanged by this scorer-free reconstruction.**
