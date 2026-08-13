# DDM VD1b worker adapted-decode fix

**Status: `READY_TO_FIRE`, not fired.** The unchanged K=200 validator command is locally unblocked at
the receiver seam. No Modal job, scorer, candidate evaluation, archive composition, or exact score ran
in this arm. The exact frontier pointer is therefore unchanged.

## Conclusion

The archive was not corrupt. The worker bypassed the adapted runtime's `inflate.sh` dependency
bootstrap and invoked its parser inside the upstream-locked Python, where Brotli was absent. The split
model decoder consequently returned no result and the compatibility path attempted LZMA on the F24S
model section, producing the remote `invalid F24S model section` failure recorded by call
`fc-01KZWHW948TQD3EAW2WY7WR2Z8`.

There was a second latent seam error: the worker passed `adapted_runtime/runtime/` as the HPAC code
directory, while the adapted runtime's canonical `f26_inflate.inflate_archive` passes
`adapted_runtime/cpr1/`. Supplying Brotli alone exposed that mismatch as a missing
`hpac_integer_sparse` import.

The worker now follows the adapted runtime's own receiver path:

- reads the exact `Brotli==1.2.0` pin from the staged `inflate.sh`;
- reuses an exact installed version or installs it into retained SSD custody with the same worker
  Python and records the dependency tree, cache, command, hashes, and bootstrap receipt;
- imports and verifies the staged `runtime.f26_inflate` module;
- uses that module's archive reader, carrier materializer, semantic decoder, selector decoder, and
  production token decoder rather than reconstructing a parser;
- passes the canonical `cpr1/` code directory; and
- fails closed unless the decoded n600 token plane matches the retained CP135 receiver golden
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.

The dispatcher's already-present lazy mounts, top-level module mount, dual-path import, and
`base_archive_sha256` ledger fix are carried forward unchanged in meaning.

## Measured proof

| Gate | Result | Durable receipt |
|---|---|---|
| Pre-fix worker in a Brotli-absent locked-worker environment | Reproduced the exact parser-to-LZMA failure on the real CP135 archive | `/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/repro_before_no_brotli/WORKER_LOAD_RECEIVER_STATE_FAILURE.json`, 1,650 B, SHA-256 `c240600f620e5067a9eb83e565c9125f97644b7c5dc6e1550c507a71c0546f48` |
| Adapted reader control | Parsed the same archive successfully as `fixed_boundary_int6` / `rc64` | `/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/adapted_reader/ADAPTED_READER_SUCCESS.json`, 1,144 B, SHA-256 `4d44cbfa8980ddff15eb239f9f80a7d511223102a6f11627ec2cdae6a2e1fb93` |
| Fixed worker, full n600 CPU decode | **PASS**, 600x384x512 = 117,964,800 token bytes; worker digest equals retained canonical digest; `np.array_equal` true | `/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/fix_no_brotli_detached/FIXED_WORKER_LOAD_SUCCESS.json`, 8,247 B, SHA-256 `dd5e6194e14a5ea0451d2b39c7a615cb4c7c80401ef37997fed45e879f068097` |

The decisive gate used the exact immutable archive:

- path: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip`
- bytes: `186252`
- SHA-256: `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`
- axis: `[macOS-CPU scorer-free full-n600 adapted-runtime decode]`
- decoded token digest: `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`
- retained worker token payload:
  `/Volumes/VertigoDataTier/pact/ddm_vd1b_20260812/fix_no_brotli_detached/retained/decoded/base_tokens.uint8.npy`,
  117,964,928 B including the NPY header, SHA-256
  `03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4`
- retained canonical raw plane:
  `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/hp3_step2/decoded_spatial_tokens.fresh_rc64.bin`,
  117,964,800 B, SHA-256
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`
- decode runtime: 625.096 s; complete gate: 631.306 s
- scorer ran: false; Modal dispatched: false

The full receipt also retains and hashes the semantic weights, carrier basis, carrier coefficients,
selector indices, RC64 library, exact adapted module, exact Brotli installation tree, and dependency
bootstrap receipt. No materialized payload was reduced to scalars and discarded.

## RECALL EVIDENCE

Before changing the seam, `tools/corpus_query.py --json --top 12` searched all seven durable stores:
research 8,463, equations 886, memory 2,114, DAG 915, council 297, tasks 531, and docs 96. Exact
queries were:

- `VD1 adapted runtime archive loader`
- `CP135 split Brotli locked venv`
- `canonical token decode cpr1 code_dir`
- `JO1 token byte identity receiver parseback`

`tools/list_canonical_equations.py --json` was also searched for archive, decode, receiver, Brotli,
RC64, token, and runtime terms. Direct bounded searches covered `CANONICAL_RESEARCH_INDEX*`, the
sub-0.15 DAG, `main_hot_state.md`, and the ledgers.

Findings and how they changed the implementation:

- CP135's custody memo established that its adapted entrypoint self-pins Brotli 1.2 and that
  `c5c7671d...` is the retained golden token plane. The fix therefore derives the dependency from the
  staged entrypoint and enforces that digest instead of inventing a new package default or oracle.
- T1R1 and JO1 established the canonical CPR1 receiver seam and exact token-plane identity as the
  required parse-back proof. The fix imports the shipped F26 module and tests the same-object token
  plane.
- Inspecting the shipped F26 entrypoint established `cpr1/`, not `runtime/`, as the decoder code
  directory. That closed the second latent import failure.
- The canonical-equation search found no equation that displaced exact byte identity for this loader
  repair.
- The bounded corpus search did not find an existing VD1b loader fix, so no prior implementation was
  duplicated or superseded.

## Validation and boundaries

- Focused tests: 11 passed.
- Ruff: worker, dispatcher, and focused test passed.
- Python compilation and `git diff --check`: passed.
- P0 retention audit: worker, dispatcher, and focused test produced zero measure-and-discard findings.
- Review tracker: two post-fix reviewed passes for all three Python files.
- Archive/runtime source mutation: none; transient `__pycache__` files produced by the local import
  were removed as rebuildable cache, and the source archive SHA-256 remains
  `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- Authority boundary: the full token gate is macOS CPU and scorer-free. It proves receiver decode
  equivalence only. It is not a contest score, CUDA result, runtime-wall-time prediction, event-delta
  measurement, selection result, or composition result.
- Remote boundary: the actual locked Modal T4 bootstrap and downstream K=200 scorer loop remain
  unmeasured because this arm was forbidden to dispatch Modal.

Unified-Lagrangian wire-in (Catalog #125): sensitivity-map N/A, because this repair does not produce
or consume a new sensitivity map; Pareto N/A, because no row is selected or promoted; bit-allocator
N/A, because no allocation changes; cathedral-autopilot ACTIVE through the unchanged scorer-free
fire gate; continual-learning N/A, because no score row or posterior update occurred;
probe-disambiguator ACTIVE, because exact pre-fix failure versus fixed full-n600 token identity is the
registered seam discriminator.

## Exact MAIN command

Fire only after the live ps135 scorer claim is terminal, claim reconciliation shows no other Modal
work, and the release preflight is adjudicated. The parent command is unchanged:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_vd1_modal_batch_event_validator.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --event-store /Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200 \
  --jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812 \
  --k 200 --run-id ddm_vd1_20260812 --resume-from ddm_vd1_20260812 \
  --lane-id ddm_vd1_modal_batch_event_validator \
  --instance-job-id modal:ddm_vd1_20260812 --claim-agent codex:ddm_vd1 \
  --detach --provider-detach-ack
```

## Disposition

- **QUEUED-WITH-A-FIRE-ORDER:** one K=200 validator dispatch. Owner: MAIN scorer-lane router.
  Consumer store: Modal volume `comma-ddm-vd1-event-validator-retained/ddm_vd1_20260812/` plus local
  harvest `.omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812/`. Fire trigger: ps135 and
  every other Modal/scorer claim are terminal, single-flight passes, and release-preflight findings are
  adjudicated.

Effective frontier remains **CP135 S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.
Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.
