# ddm_jo6 receiver/container compatibility receipt — 2026-08-22

## Outcome

**Fork B.** `RX1M` is the intended shipping container. The defect was in the
JO2-generated F26 receiver tree: it tested the wrapped semantic body for an
inner `WANS1` / `SD1M` / `SM3R` tag before calling `split_semantic_blob`.
The archive did not lose its semantic weights and does not need conversion.

The fix moves the JO2 unwrap before the F26 semantic-family guard and loads
the inner semantic body. Receiver closure now also runs the exact staged
shipping `inflate.sh` and requires a retained `0.raw` of exactly
3,662,409,600 bytes before it can write `RECEIVER_CLOSE_RESULT.json`; because
the entrypoint writes `RECEIVER_CLOSE_POINTER.json` only after that function
returns, the missing two-landing gate is now fail-closed.

The fix is implemented and independently serializer-committed as
`0f018e8216ee9fc7d7049656d7c2d6303a6856ae`. The canonical shared checkout
could not accept the commit because its Git object store rejected `git add`
with `Operation not permitted`; its index remains untouched. Durable transfer
artifacts are:

- `.omx/research/ddm_jo6_receiver_container_compat_20260822/SOURCE_FIX_0f018e821.bundle`
  — 3,734 bytes, SHA-256
  `8e8edf0a9b50e19157703f15dda1c3d081d23b8af5303dd5f6b8be420a5152a5`.
- `.omx/research/ddm_jo6_receiver_container_compat_20260822/0001-JO6-validate-staged-shipping-receiver-before-closure.patch`
  — 11,847 bytes, SHA-256
  `6b2519b7f150a4316e2d766dd37f04ce7c3b92039aa84664a460c6e400ed7fc9`.

## Design-intent receipt

- The exact FX5 base archive
  `/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5/archive.zip`
  is 180,386 bytes, SHA-256
  `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`.
  Its single stored member `p` begins `RX1M`; the shipping
  `runtime.residual_archive.read_residual_archive` parses it as schema
  `fixed_boundary_int6`, token codec `rc64`, inner semantic tag `SM3R`, and
  carrier bytes. This proves RX1M and the F26 shipping family coexist in the
  base object.
- The PQ1 shipping packet manifest also records `RX1M`, and the RC2 clean
  decode memo records that the same exact header was decoded through the full
  shipping receiver.
- The JO1 design memo says learned content is folded into the RC2 RX1 `p` and
  that the exact receiver remains required. The JO2 memo says the JO2 residual
  state is stored inside the existing semantic body and the package remains a
  single-member RX1 archive.
- In the failed staged receiver, the JO2 unwrap was present later in
  `runtime/f26_inflate.py`, after the pre-load guard that rejected the wrapper.
  This local ordering defect, not the archive container, explains the observed
  `InflationError`.

## Real receiver control

The exact failed r8 candidate archive was restaged from the fixed builder; no
custody-frozen attempt bytes were edited. The exact staged shipping receiver,
with `tools/host_shims/python` on `PATH`, completed in 722.4754065410234 seconds
on `[macOS-CPU real staged shipped receiver; no score authority]`:

- archive: 181,484 bytes, SHA-256
  `9c6ab0e3812e6ca998e3c01d51fca8ed646e2eeb885b673c47424d731937e623`;
- extracted `p`: 181,384 bytes, SHA-256
  `f3bf1454fab63a192598ff92885fb2577eb776f97a91f4f911280f331f100267`;
- retained `0.raw`: 3,662,409,600 bytes, SHA-256
  `98eb8cb2d78eaafe9db177a3a4cd4ef59eb09f90e207797ddc87f2949573006e`;
- identity census: 0 mismatches in 1,831,204,800 frame-0 elements and 0
  mismatches in 1,831,204,800 master elements, n=600.

Receipts:

- `/Volumes/VertigoDataTier/pact/ddm_jo6_receiver_container_compat/real_receiver_control_0000/CONTROL_RESULT.json`
  — SHA-256
  `b9f7bbe40dda5be818ea7fa22016d215bb022f1a88d66fe37feda71a9d1672b8`;
- `/Volumes/VertigoDataTier/pact/ddm_jo6_receiver_container_compat/real_receiver_control_0000/IDENTITY_CONTROL_RESULT.json`
  — SHA-256
  `d986005c5e6e0c83f1507f5236d53f2ed5f0c962352c9bffa6a5594bf414d8cb`.

All generated payloads, the full raw output, checkpoints, logs, and both
candidate representations are retained. No scorer was run and no score is
claimed.

## Source, tests, and reviews

- Entry point start/end SHA-256:
  `766d3494751b27343df8904db2b74fd21e3d7804274a7e3931316ae11736bcdd`.
  It was not edited.
- Fixed receiver source SHA-256:
  `b4653bf62e6cec7c11e3108f33d860eeac79f6a3213b77f47fc06ca1610b32b4`.
- Test source SHA-256:
  `d8138fa6ded5725f7de4f4b7a90964a340f833c50be90ad5214391a7094beba5`.
- `experiments/tests/test_ddm_jo1_joint_objective.py`: 37 passed, two existing
  Pydantic field-shadow warnings.
- Ruff, `py_compile`, and `git diff --check`: pass.
- Both changed Python files received two genuine review-tracker passes in the
  shared checkout and again in the clean serializer clone.

The added negative control retains a short raw and its log but refuses to
write `RECEIVER_EXECUTION.json`; the positive control writes the receipt only
after the exact byte-count check.

## Resume and two-landing status

The charter's statement that imported modules were not SHA-pinned was false
for the live r8 seal: `load_config` verifies `inputs.receiver_close_source`.
The fixed builder therefore cannot lawfully resume under the old config SHA
`38d2f96d...`. I generated r9 with the fixed source pin, ran a fresh
workload-bound memory preflight, and resealed:

- final compiled config: 16,045 bytes, SHA-256
  `33039c1c9c7ab7bae755e35fd0850c2c9738220e806a6f5ba27b46380dfc3786`;
- workload identity:
  `9991c20e3c2966817caa9839861cedbd548bcf954520337b075cc0f39fe0670a`;
- readiness: `READY_TO_FIRE_UNDER_STANDING_GO`, zero blockers;
- memory preflight: PASS, measured peak RSS 2,855,534,592 bytes and projected
  n600 peak 5,360,154,440 bytes against the 16 GiB cap;
- checkpoint migration receipt:
  `.omx/research/ddm_jo6_receiver_container_compat_20260822/CHECKPOINT_MIGRATION_R9.json`,
  7,202 bytes, SHA-256
  `a90cb88ef099d6d7029d9b7e938ef383f5cfa41f1d48eff5bd59d16515f50e0f`.

The six training-state payloads migrated byte-identically at
`target_birth / step 600 / field 0 / package 0`; training did not restart.
The governed launch used lane `ddm_jo2_joint_objective_fx5`, with 16 GiB RSS
and 259,200-second wall caps. It completed r9 n600 candidate materialization
and retained ten fresh-Schur pair results, then stopped without a safe-run
terminal receipt while below its memory cap. After preserving
`STORAGE_POLICY.json` aside exactly as required, a second exact charter
relaunch resumed pair 10 and is active under PID 37877; the memory governor
continues to sample it and pair receipts continue to advance. An interim
`kill -0` inference that the second daemon had vanished was wrong under the
managed process-isolation boundary and is explicitly retracted in
`DAEMON_LOSS_R9_0001.json`. The r8 and r9 residual payloads and
candidate-master arrays are byte-identical, but their receipts have different
workload identities, so I did not transplant the old pointer.

The two-landing source gate is implemented and real-control-proven, but the
canonical source commit is still blocked by Git write permissions. The r9
chain has not yet written its own `RECEIVER_EXECUTION_POINTER.json`; its
runtime terminal condition remains active under the governed daemon.

## RECALL EVIDENCE

Sources searched:

- content queries `RX1M`, `receiver_close`, `run_receiver`, `RX1`, `RX2`,
  `F26`, and `fx5_e1` across `.omx/research/`, arm final messages, current
  state ledgers, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG/FEED surfaces,
  and the JO1/JO2/JO3 code;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  receiver, container, RX1, and F26;
- exact archive/member inspection and the current shipping parser under the
  FX5 runtime tree.

Beyond the charter seeds, PQ1's exact shipping manifest and the FX5 base
archive independently proved RX1M is already the shipping F26 container, and
the JO1/JO2 memos explicitly placed the residual inside the semantic body.
That changed the plan from archive conversion to fixing the generated receiver
builder. The equations registry had generic stream/container laws but no
JO6-specific cure.

## GESTALT-DELTA

Before JO6, receiver closure meant parse-back identity inside Python, while
shipping execution happened only afterward; that gap allowed a valid-looking
pointer to hide an impossible decode. After JO6, receiver closure is a
two-landing contract: the archive must parse back and the exact shipping tree
must materialize the complete n600 raw object before closure exists. The
container question is also closed at the right boundary: RX1M is the outer
archive grammar, while WANS1/SD1M/SM3R classify the unwrapped semantic body.

## Live hypotheses

- The r9 replay should produce the same candidate codes and archive bytes as
  r8 because the migrated EMA/state, residual payload, and candidate master
  are byte-identical; this remains untested until r9 reaches packaging.
- Once r9 reaches receiver closure, its newly embedded validation should make
  the later `run_receiver` execution deterministic and redundant as a defect
  detector, though it is still required as a separate custody receipt.
- The active r9 daemon should finish the remaining Schur pairs and reach the
  new receiver gate because its retained cursor resumed correctly and the
  second launch continues to advance under the memory governor.

## Dead ends

- Rewrapping RX1M into WANS1/SD1M/SM3R: closed because those are inner semantic
  tags, not alternative outer containers, and the exact base shipping archive
  already uses RX1M.
- Swapping in a non-shipping receiver: closed by the real-control result using
  the exact generated shipping tree.
- Editing the custody-frozen failed attempt: closed; the repair is in the
  builder and every validation used a fresh staged tree.
- Resuming r8 directly under its old compiled config: closed because the live
  inventory verifies the receiver-source SHA despite the charter's contrary
  assumption.
- Transplanting r8 stage pointers into r9: closed because their embedded
  workload identities differ even though the large candidate arrays match.
- Treating `kill -0` failure as proof that a managed detached process is dead:
  closed because the governor and advancing pair receipts directly refuted
  that inference for PID 37877.

Own-vehicle frontier remains dx2: S 0.14821987563243377 @ 180,368 B
`[contest-CUDA T4 n600]`, archive SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
