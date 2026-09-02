# ddm_g8c — generation-8 compression vendoring

Date: 2026-09-02

Owner: `ddm_g8c`

Verdict: **READY-FOR-MAIN-BUNDLE-PREPARED-NOT-LANDED-NOT-PUBLISHED**

Authority: `[macOS-CPU advisory / scorer-free exact byte measurement]`

## Conclusion

The public-minimal generation-8 tree is `submissions/semantic_joint_ctxmix/`. It has 39 repo-side code/README files plus the required `MANIFEST.sha256` receipt: 40 physical files, 688,488 bytes, and exactly one Markdown file. The 55-file frozen generation-7 handoff was copied without editing its sealed source, the internal `FX5_BUILD_MANIFEST.json` and every AppleDouble sidecar were excluded, and the live operator-reviewed Drive `SUBMISSION_PACKAGE.md` section 2 is the tree's sole README.

`compress.py` now contains the complete encode-side dependency closure. It restores eleven located source members byte-for-byte into the retained work store, checks every restored SHA-256, decodes and retains the full 600-plane token field from the pinned generation-6 archive, executes the real FX5 RC64 encode, the real DX2 CAP1 Rice-to-CABAC carrier fold, both GB1 RC64 branches, LB1, and AFR1, retains every stage payload and checkpoint, and refuses unless two full runs both produce the exact AFR1 archive. The sources are compressed as code payloads only to preserve the 39-file public shape; they are not replacements, estimates, or stubs.

The compressor proof and the edited-runtime full raw identity proof are recorded below. This packaging work does not change `archive.zip`, does not run a scorer, does not dispatch Modal, does not publish, and does not move the frontier. The managed sandbox refused writes to the checkout's Git object store; the required serializer therefore authored the intended commit in its governed Vertigo bundle fallback. The source tree and memo remain uncommitted in this checkout until MAIN applies that bundle.

## RECALL EVIDENCE

The original recall searched content—not just filenames—for `compress.py`, `AFR1`, `FX5`, `DX2`, `GB1`, `LB1`, `RC64`, `bare checkout`, `Brotli`, `AppleDouble`, `runtime identity`, and the exact archive/token hashes across `.omx/research/`, arm receipts on both SSD tiers, `experiments/`, the canonical research index, the `sub015_DAG_*` FEED surfaces, and the task ledger. It also ran `.venv/bin/python tools/list_canonical_equations.py --json` and read the live pointer board.

Findings beyond the charter seeds changed the build in five ways:

1. DX2 changes `runtime/residual_archive.py`, not only the corrector configuration, so the pre-DX2 and post-DX2 sources are separately embedded and pinned.
2. GB1 is a fork: `ba1f3830…` is the pointer candidate, while LB1 consumes the independently admitted JT21 joint bank `ec0dd68f…`. Both branches therefore run and both payloads are retained.
3. Each rate stage needs its receipt-exact `fx2_model_axis_corrector.py`; carrying only the final AFR1 source would not replay the real chain.
4. The ce1 receipts' short displayed wall times were resume tails from frame 575, not fresh-run prices. A fresh smoke exposed the true per-stage checkpoint cadence, so the full proof was launched under the charter's detached contract.
5. The live Drive note's 39-file estimate equals the 37 receiver files plus `compress.py` and README. The initially located eleven-file helper directory was therefore folded into `compress.py` as exact compressed source members rather than left as a 50-file almost-consolidation.

No closed-form score operator was needed: this arm changes no scored object and tests exact byte equalities. Fitted or sampled substitutes were not used.

## Vendored encode modules

Every destination below means `compress.py::embedded/<member>`. At execution, the member is restored under `<store>/work/embedded_encoder_sources/compress_vendor/`, hash-checked, and executed from there.

| located source | embedded member | restored bytes | restored SHA-256 | disposition |
|---|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/runtime/fx2_model_axis_corrector.py` | `fx2_rc2.py` | 29,094 | `3cbddcf85e82d7a17e3f19e649a8af1901ea62fd5d91a7ca0d13f1f7edbcec79` | exact source retained |
| `/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/work/run_1/fx5/runtime_fx5/runtime/fx2_model_axis_corrector.py` | `fx2_fx5.py` | 29,515 | `77e81ac827d6d1f820229c7d21b1c749caf18acc23c6635fb327884a0da04be1` | exact source retained |
| `/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/work/run_1/gb1/runtime_groupbin8_surprise/runtime/fx2_model_axis_corrector.py` | `fx2_gb1_pointer.py` | 30,839 | `db39d22b9d4b5590d864d9c3676accaa5e45e23b6f3649698b7353b159b2a593` | exact pointer branch retained |
| `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_joint21/runtime/fx2_model_axis_corrector.py` | `fx2_gb1_joint.py` | 30,864 | `06cc74279e485e2d73558b2ea5ec9a5c68606e685231c10bbf5ef1bac5c2f296` | exact LB1 input branch retained |
| `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/runtime_joint22_patch192/runtime/fx2_model_axis_corrector.py` | `fx2_lb1.py` | 31,142 | `460490e427e54d89f0a074d785cb8bd7678df509215be1d2a37a6f5a6f617a75` | exact source retained |
| `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/runtime_candidate_native/runtime/fx2_model_axis_corrector.py` | `fx2_afr1.py` | 31,470 | `6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064` | exact source retained |
| `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/runtime/residual_archive.py` | `residual_pre_dx2.py` | 30,058 | `e62489099c6d6d236bbb946ccd5fc9f55e75696dd74c0a1e0ebeece093bede5e` | exact pre-DX2 source retained |
| `/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/work/run_1/dx2/candidate_runtime_dx2/runtime/residual_archive.py` | `residual_dx2.py` | 30,622 | `aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530` | exact post-DX2 source retained |
| commit `2c3a2153e4`: `experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py` | `route_b_rc64.py` | 14,138 | `c2d9759a77e793d643ca1d4a557934cdb66f39473b244f382dd9f0b8faaf89e5` | exact source retained |
| `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c` | `rc64_backend_encoder.c` | 12,222 | `5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6` | exact encoder-bearing C retained |
| commit `2c3a2153e4`: `experiments/ddm_jg2_tail_reencode.py` | `jg2_tail_reencode.py` | 46,331 | `3a89c2b2e64e5ec018e134179d8da57c00557a043c1a9d27c114bbcc67e5f9b8` | mechanism preserved; only local route/source resolution, guarded compiler discovery, and portable `.so` output name changed |

The materialization test restored all **11/11 members** and matched all **11/11 byte counts and SHA-256 values**. A recursive AST census over the 34 shipped Python files plus the ten embedded Python members found exactly three external dependency roots—`numpy`, `torch`, and `brotli`—and zero unresolved roots after accounting for stdlib and tree-local modules.

## Bare-checkout two-run rebuild

The proof snapshot is local commit `b8567433cf1ccb5cab4932367ad8e45ecc8e2571`, cloned at `/private/tmp/ddm_g8c_gen8_clone_v3_ooaw5F`. Its tracked porcelain was empty before the venv was created and its manifest passed 39/39 rows. A fresh CPython 3.13.12 venv received an offline byte copy of the declared dependency environment because sandbox policy denied access to the host package-manager cache; the repo-editable `__editable__.tac-0.2.0rc2.pth` was excluded, `import tac` was absent, and NumPy/Torch/Brotli resolved inside that fresh venv.

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .proof_venv313/bin/python compress.py \
  --base-archive /Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed/archive.zip \
  --store /Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/bare_proof_v3 \
  --resume
```

| run | environment fingerprint | stage denominator | final bytes | final SHA-256 | wall time | receipt | verdict |
|---|---|---:|---:|---|---:|---|---|
| 1 | CPython 3.13.12; NumPy 1.26.4; Torch 2.12.1; Brotli 1.2.0; repo `tac` absent | 6/6 pinned outputs | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` | 3,495.107 s | `.../bare_proof_v3/retained/run_1/RESULT.json` | PASS |
| 2 | same isolated environment | 6/6 pinned outputs | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` | 3,499.376 s | `.../bare_proof_v3/retained/run_2/RESULT.json` | PASS |

The initial token decode retained 117,964,800 bytes at `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` after 595.879 seconds. Every RC64 encoder checkpointed every 25 frames with its complete corrector and coder state, and each stage archive was retained before the next stage began. DX2 separately retained its incumbent repeat, candidate repeat, original and transformed carrier bodies, Rice and CABAC payloads, coefficient array, parameters, and corrupted negative. The two final outputs were also compared byte-for-byte. Overall status: PASS in 7,591.067 seconds; terminal receipt SHA-256 `27048158372606ef4b6bde743adb0a283e47a85a37922ef16d371eb6acf0ce9f`.

Detached custody: launcher `.omx/tmp/codex_runs/ddm_g8c_run_bare_proof_v3_20260902.sh`, pidfile `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/bare_proof_v3/PID`, log `.../run.log`, resumable store `.../bare_proof_v3/`, result `.../bare_proof_v3/RESULT.json` (SHA-256 `12bc265d…e438`), and terminal receipt `.../bare_proof_v3/DONE.json`. The sandbox refused daemon-socket creation; the `nohup` process remained live and was monitored through durable receipts, checkpoints, and open file descriptors.

## Runtime edit ledger

Fresh generation-8 source-tree digest: `477f2a3dbf6299ed4cbcc7ffe7ca13becfcaba6c116a5a1dbe8ba8c803ae789e` over the canonical JSON list of all 40 relative paths, byte counts, and SHA-256 values. Receipt: `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/TREE_DIGEST.json` (SHA-256 `63cc8cb3802cb11f5f3fc6bfb3de2a66d4576ee934b06a0c2eaf8d5e64ed3c4b`). Manifest: `0612e9114d1050ef296a3e7be177a6e4e747f36f3b2c933e1aae5a7dbaf93bf5`, 39/39 rows passed.

| edit | fail-close probe | full all-600 identity | disposition |
|---|---|---|---|
| `inflate.sh`: removed runtime network install and requires preinstalled `Brotli==1.2.0` | missing dependency returned 69 with exact error; retained `runtime_preflight/missing_brotli.*.log` | composed edited tree: 600/600 pairs; 3,662,409,600 B; `7246a4ff…f5f2de7` | PASS-COMPOSED |
| `inflate.sh`: resolve and guard `CC` once before any compile | missing compiler returned 69 with exact error; retained `runtime_preflight/missing_cc.*.log` | composed edited tree: 600/600 pairs; 3,662,409,600 B; `7246a4ff…f5f2de7` | PASS-COMPOSED |
| `inflate.py`: removed silent CPU fallback and requires CUDA | no-CUDA invocation returned 1 with explicit measured-budget error and created no raw output; retained `runtime_preflight/no_cuda.*.log` | lower-level CPU receiver from composed edited tree: 600/600 pairs; 3,662,409,600 B; `7246a4ff…f5f2de7` | PASS-COMPOSED |

The local identity call intentionally entered the lower-level device-parameterized receiver with `device_name="cpu"`; the public `inflate.py` refuses that axis. This tested the exact receiver math while preserving the new public CUDA-only policy. It completed 600/600 pairs in 819.616 seconds and retained all 3,662,409,600 raw bytes at SHA-256 `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`, byte-identical to the ux1-verified local identity object. It is not the contest-CUDA raw hash and makes no cross-axis inference.

The terminal identity receipt is `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/runtime_identity_v3/RESULT.json` (SHA-256 `5a7d99a8562695ab9e8de2e476e0ca879cb3cbf604a6017f891589dac2b46c97`); its `DONE.json` is `bf852c8b…fc507`. The retained raw is `runtime_identity_v3/retained/0.raw`. `RETENTION_MANIFEST.json` (SHA-256 `fa30dc15ee747301aa2f41884b988383e04bad46e7e40ee6d447b20499686abf`) binds the tree receipt, commands, two compressor runs, all payload indexes, raw identity, compiled receiver libraries, and retained refusals. The charter-required AppleDouble cleanup removed 1,462 metadata sidecars totaling 5,988,352 bytes from this arm's exact custody root; the post-cleanup count is zero, and no proof payload was removed.

## Ship-set ledger

Disposition is per frozen generation-7 file. `KEEP-EXACT` means hash-identical to the frozen manifest; `KEEP-EDITED` means the two charter-required fail-close changes; `REPLACE` means a new generation-8 artifact occupies the role; `DROP` means absent from the public tree.

| generation-7 file | disposition | generation-8 reason |
|---|---|---|
| `cpr1/carrier_codec.py` | KEEP-EXACT | receiver dependency |
| `cpr1/ddm_mp2_semantic_receiver.py` | KEEP-EXACT | receiver dependency |
| `cpr1/hpac_integer.py` | KEEP-EXACT | receiver dependency |
| `cpr1/hpac_integer_sparse.py` | KEEP-EXACT | receiver dependency |
| `cpr1/inflate.py` | KEEP-EXACT | receiver dependency |
| `cpr1/integer_model_io.py` | KEEP-EXACT | receiver dependency |
| `inflate.py` | KEEP-EDITED | explicit CUDA-required refusal replaces CPU fallback |
| `inflate.sh` | KEEP-EDITED | fail-closed Brotli and compiler preflights |
| `runtime/__init__.py` | KEEP-EXACT | receiver dependency |
| `runtime/baseline.py` | KEEP-EXACT | receiver dependency |
| `runtime/bits.py` | KEEP-EXACT | receiver dependency |
| `runtime/carrier_repack.py` | KEEP-EXACT | receiver dependency |
| `runtime/compensation_overlay.py` | KEEP-EXACT | receiver dependency |
| `runtime/ddm_wc1_advisory_runtime.py` | KEEP-EXACT | receiver dependency |
| `runtime/dx2_cabac_coefficients.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/__init__.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/adaptive_ans.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/coefficient_ar1_codec.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/coefficient_predictor.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/rc64.py` | KEEP-EXACT | receiver dependency |
| `runtime/entropy/rc64_backend.c` | KEEP-EXACT | decoder-bearing receiver source |
| `runtime/entropy/renderer_weight_codec.py` | KEEP-EXACT | receiver dependency |
| `runtime/f26_corrector_native.c` | KEEP-EXACT | receiver dependency |
| `runtime/f26_hpac_native.c` | KEEP-EXACT | receiver dependency |
| `runtime/f26_hpac_native.py` | KEEP-EXACT | receiver dependency |
| `runtime/f26_inflate.py` | KEEP-EXACT | device-parameterized exact receiver |
| `runtime/frame0_selector.py` | KEEP-EXACT | receiver dependency |
| `runtime/free_corrector.py` | KEEP-EXACT | receiver dependency and encoder mirror |
| `runtime/fx1_logistic_mixer_corrector.py` | KEEP-EXACT | receiver dependency and encoder mirror |
| `runtime/fx2_model_axis_corrector.py` | KEEP-EXACT | AFR1 receiver authority |
| `runtime/hpac_inference.py` | KEEP-EXACT | receiver dependency |
| `runtime/ihs2.py` | KEEP-EXACT | receiver dependency |
| `runtime/ihs2_gate_a.py` | KEEP-EXACT | receiver dependency |
| `runtime/native_free_corrector.py` | KEEP-EXACT | receiver dependency |
| `runtime/residual_archive.py` | KEEP-EXACT | AFR1 parser/decoder authority |
| `runtime/rr4_free_corrector.py` | KEEP-EXACT | receiver dependency and encoder mirror |
| `runtime/rr5_arith_basis.py` | KEEP-EXACT | receiver dependency |
| `README.md` | REPLACE | exact live Drive section 2; only Markdown in tree |
| `compress.py` | REPLACE | real self-contained five-stage encode closure |
| `MANIFEST.sha256` | REPLACE | generation-8 39-row manifest |
| `FX5_BUILD_MANIFEST.json` | DROP | internal absolute-path leak; explicitly forbidden |
| `.compress_py_pre_ce1_superseded_20260902` | DROP | superseded lab residue |
| `BORROWED_SUBSTRATE_ACCOUNTING.md` | DROP | internal accounting; lineage is in the one README |
| `COMPRESS.md` | DROP | contradicted bare-checkout answer and duplicates README |
| `LICENSE` | DROP | not in measured receiver/compressor dependency closure; MAIN may attach only if operator requires it |
| `THIRD_PARTY_NOTICES.md` | DROP | not in measured receiver/compressor dependency closure; MAIN may attach only if operator requires it |
| `archive.zip` | DROP-FROM-REPO-TREE | exact 180,002-byte authority archive remains a separately pinned presentation/publish attachment |
| `archive_manifest.json` | DROP | superseded by clean manifest and in-script archive pins |
| `report.txt` | DROP | internal lab handoff report |
| `._.compress_py_pre_ce1_superseded_20260902` | DROP | AppleDouble sidecar |
| `._BORROWED_SUBSTRATE_ACCOUNTING.md` | DROP | AppleDouble sidecar |
| `._COMPRESS.md` | DROP | AppleDouble sidecar |
| `._README.md` | DROP | AppleDouble sidecar |
| `._compress.py` | DROP | AppleDouble sidecar |
| `._report.txt` | DROP | AppleDouble sidecar |

Generation-8 adds no helper files beside the manifest: the eleven encode sources live inside `compress.py`. The public-tree census found 0 AppleDouble files, 0 `__pycache__` directories, 0 `.pyc` files, and exactly one `.md` file.

## One-page verification note

| gate | denominator / fact | receipt | status |
|---|---|---|---|
| frozen copy | 35/35 unchanged receiver files exact; 2/2 wrappers intentionally edited; 1/1 FX5 manifest dropped | frozen `MANIFEST.sha256` vs generation-8 | PASS |
| public shape | 39 code/README rows + one manifest; 40 files; 688,488 B; one Markdown | `TREE_DIGEST.json` | PASS |
| manifest | 39/39 rows | `submissions/semantic_joint_ctxmix/MANIFEST.sha256` | PASS |
| embedded encode closure | 11/11 restored sources match byte count and SHA; zero unresolved import roots | materialization + AST census | PASS |
| fail-closed entrypoints | missing compiler 69; missing Brotli 69; no CUDA 1/no output | `runtime_preflight/*.log` | PASS |
| review | 426 entities × 2 initial passes; changed `compress.py` 20 entities × 2 post-DX2-fix passes | review passes `ddm_g8c_review_pass1` through `ddm_g8c_review_pass4` | PASS |
| clean-clone compression | two runs × six pinned outputs; finals byte-identical | `bare_proof_v3/RESULT.json` | PASS |
| local edited-tree identity | 600/600 pairs and 3,662,409,600/3,662,409,600 raw bytes; SHA-256 `7246a4ff…f5f2de7` | `runtime_identity_v3/RESULT.json` | PASS |
| custody | compressor payloads, raw output, native libraries, terminal receipts, refusal receipts; AppleDouble count 0 | `RETENTION_MANIFEST.json` | PASS |
| Git custody | serializer attempted the exact 41-file set with per-file post-edit and `new` base hashes; checkout object write was denied; governed bundle fallback is under `/Volumes/VertigoDataTier/pact/ddm_g8c_gen8_compress_vendoring/commit_fallback/` | serializer `receipts.jsonl`, bundle, format-patch | PASS-BUNDLE-FALLBACK / NOT-LANDED |

## READY-FOR-MAIN

| disposition | owner | consumer store | fire trigger | action |
|---|---|---|---|---|
| READY-BUNDLE-PREPARED | MAIN | `/Volumes/VertigoDataTier/pact/ddm_g8c_gen8_compress_vendoring/commit_fallback/`, `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/`, and `submissions/semantic_joint_ctxmix/` | both bare rebuilds and edited-tree raw identity are terminal PASS, MAIN applies and verifies the serializer fallback commit in a writable checkout, the tree manifest revalidates, and operator confirms under #1111 | MAIN lands the exact fallback commit, verifies the READY row, presents the tree plus separately pinned authority archive to the operator, and publishes only the operator-approved packet |

No Modal re-buy is requested: archive bytes and receiver raw bytes are required to remain exact, so this preparation creates no new score object. The local scorer lane remained untouched.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; UNMOVED by packaging.`

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_g8c_gen8_compress_vendoring/commit_fallback/`, `/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/`, and `submissions/semantic_joint_ctxmix/`; fire trigger: MAIN applies and verifies the serializer fallback commit in a writable checkout, this READY row remains PASS, the tree manifest revalidates, operator confirms under #1111, and MAIN's presentation checklist passes; action: land the exact bundle commit, attach the exact authority archive, and publish only the operator-approved packet.

## LIVE-HYPOTHESES

- **INSTANCE:** on 1:1 contest-CUDA hardware with `Brotli==1.2.0` and a working C compiler already present, the strict public entrypoint will admit the unchanged receiver and reproduce the separately pinned contest-CUDA raw object. This remains untested here because this arm was forbidden to dispatch Modal or infer CUDA identity from the macOS-CPU PASS. Falsifier: entrypoint refusal or any byte/hash difference on the same archive and frozen CUDA axis.

## DEAD-ENDS

- **FORMULATION closed:** shipping a helper directory is not the requested consolidation. The exact eleven-source closure works, but 50 code/README files contradict the live 39-file public shape; the sources are now embedded and restored exactly.
- **INSTANCE closed:** a fresh venv that reads the main repo's site-packages through `PYTHONPATH` is weaker than an isolated bare-checkout proof. It was stopped before encoding and replaced by an offline self-contained dependency copy with the repo-editable `.pth` excluded.
- **INSTANCE closed:** this sandbox cannot create a tmux daemon socket. The retained `nohup` process remained live without a daemon socket and was monitored through open file descriptors, checkpoints, and its terminal receipt; tmux is not required for this proof.
- **INSTANCE closed:** treating DX2 as another RC64 token encode reproduced FX5 unchanged and correctly failed the `976f…` pin. The real DX2 mechanism is the receipt-defined CAP1 Rice-to-CABAC carrier fold; that mechanism now reproduces the exact 180,368-byte archive in both runs.
- **INSTANCE closed:** the first raw-identity launch omitted `CPR1_RC64_LIBRARY` and correctly refused before materializing raw output. The passing launch compiled and retained the exact receiver backend before decode.
- **INSTANCE closed:** the second raw-identity launch used a non-PTY detached host that reaped the process before any terminal receipt or payload. Its RUNNING receipt is retained as abandoned evidence; the passing launch used a persistent detached shell, per-stage checkpoint, PID, log, and atomic terminal receipt.
- **FAMILY inherited closed:** do not replace the actual RC64 encoders with a byte-copy shortcut or reconstructed proxy. Exact source restoration plus execution is the minimum admissible mechanism.
