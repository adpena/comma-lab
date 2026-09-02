Prepared and verified the generation-8 public package at [semantic_joint_ctxmix](/Users/adpena/Projects/pact/submissions/semantic_joint_ctxmix):

- 40 files total: 39 manifest rows plus `MANIFEST.sha256`, 688,488 bytes, one Markdown file, no AppleDouble or Python cache residue.
- Tree digest: `477f2a3d…789e`; manifest: 39/39 PASS.
- Two isolated compressor runs reproduced the exact 180,002-byte AFR1 archive, SHA-256 `cbb8d928…05bf25`.
- Full local receiver identity: 600/600 pairs, 3,662,409,600 bytes, SHA-256 `7246a4ff…f5f2de7`, PASS.
- Missing compiler/Brotli and unavailable CUDA all fail closed.
- Payloads, raw output, libraries, commands, and refusals are preserved in the [retention manifest](/Volumes/APDataStore/pact/ddm_g8c_gen8_compress_vendoring/RETENTION_MANIFEST.json).
- No scorer, Modal dispatch, publication, or archive mutation occurred.

The evidence memo is [ddm_g8c_gen8_compress_vendoring_20260902.md](/Users/adpena/Projects/pact/.omx/research/ddm_g8c_gen8_compress_vendoring_20260902.md).

The checkout’s Git object store rejected writes with `Operation not permitted`. The serializer therefore produced and independently verified fallback commit `da3d06881c8a6a27acaf141d15193a726e73d5af`: exactly 41 intended files, 41/41 content hashes matching, with bundle SHA-256 `6f189510…d477`. The [bundle](/Volumes/VertigoDataTier/pact/ddm_g8c_gen8_compress_vendoring/commit_fallback/20260902T235005.401901Z-60238/intended-commit.bundle) is durable but not landed in this checkout.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; UNMOVED by packaging.`

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Vertigo `commit_fallback/`, APDataStore `ddm_g8c_gen8_compress_vendoring/`, and `submissions/semantic_joint_ctxmix/`; fire trigger: MAIN applies and verifies commit `da3d0688…d5af` in a writable checkout, the manifest remains PASS, and the operator confirms under #1111; action: land the bundle commit, attach the separately pinned authority archive, and publish only the approved packet.

## LIVE-HYPOTHESES

- On 1:1 contest-CUDA hardware with Brotli 1.2.0 and a working compiler already installed, the strict public entrypoint should reproduce the frozen contest-CUDA raw object. This remains untested because this arm was forbidden from dispatching Modal or inferring CUDA identity from the macOS-CPU PASS.

## DEAD-ENDS

- A separate eleven-file helper directory violated the requested 39-file public shape; the exact sources are now embedded and restored by `compress.py`.
- Modeling DX2 as another RC64 encode reproduced FX5 unchanged. The real CAP1 Rice-to-CABAC fold reproduced the pinned DX2 archive.
- The first identity launch lacked the RC64 decoder binding and correctly refused before raw materialization.
- A non-PTY detached launch was reaped before producing a terminal receipt; the persistent launch completed with checkpoints and retained output.
- Direct landing in this checkout is blocked by sandboxed Git writes. The verified serializer bundle is the recoverable landing path.