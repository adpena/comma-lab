# v10 power-diagram generator byte-close build specification and containment amendment (2026-07-18)

Final containment amendment: the original executable measurement design below
was exercised once and stopped fail-closed at frame 195. Fresh-eyes review then
found its success cleanup certificate unsafe. The exact source is retained only
inside deterministic non-source gzip checkpoint-lineage evidence; the live path
is a fail-closed tombstone.
All post-blocker work uses separate read-only harvester/diagnostic tools. No
resume, certification, or cleanup surface from the historical tool remains
authorizing.

## Authority and verdict boundary

- Delegated lane: `v10_power_diagram_byteclose_20260718`; v10 completeness factor 6; Task #543.
- Frozen authority is the contest CPU-Torch SegNet at
  `/Users/adpena/Projects/pact/upstream/models/segnet.safetensors`, evaluated in
  canonical frame order against the real n600 `lstars` cache.
- This build measures the byte cost and classifier-cell fidelity of a fitted
  `PDW1` generator payload. It does **not** manufacture a legal RGB inverse.
  Generator-only parity is therefore a feature-space pullback measurement, not
  receiver closure and not an upstream contest score.
- A positive result may close the generator-rate subquestion. It may not mark
  factor 6 complete, move the score pointer, or register
  `affine_head_power_diagram_generator_duality_v1` in the shared equation
  registry unless generator-to-cell parity and a real byte-closed RGB receiver
  are both proven.

## Implementation packet

The original bounded measurement tool plus unit tests reused
`tac.boundary_math.power_diagram_witness` for the frozen-head quotient,
power-diagram conversion, and strict `PDW1` parse-back.

The tool must:

1. Accept explicit `--upstream-root`, `--gt-cache`, `--feature-cache-dir`, and
   `--output` paths. It must never infer or mutate the sacred live run.
2. Fail closed through `tac.admission_guard.assert_governed_admission` before
   loading the scorer. The governing shell wrapper is `tools/safe_run.py`.
3. Load the frozen CPU-Torch SegNet, install a hook on the input to the final
   segmentation-head convolution, and process frames in canonical n600 order
   with batch size one. Batch size is fixed because alternate Torch batch
   geometry has already produced authoritative argmax drift.
4. Compute the canonical rank-4 quotient field without materializing a giant
   unfold: convolve the hooked 16-channel feature map with the four quotient
   basis filters using the exact final-head stride, padding, dilation, and
   groups. Verify that the frozen-head power-diagram target reproduces the
   cached `lstars`; any mismatch is a hard blocker.
5. Write quotient features as rebuildable scratch on the selected SSD tier,
   while accumulating float64 ridge sufficient statistics. Fit class-affine
   scores from the real `(quotient feature, lstar)` pairs without materializing
   an n600 one-hot design matrix.
   The extraction is resumable: keep a distinct partial cache plus an atomic
   progress checkpoint containing the next canonical frame index and the
   sufficient statistics. Preserve the completed extraction-stage checkpoint;
   `--resume` must verify its input/config hashes before continuing and may lose
   at most one frame. A fresh run must refuse to absorb stale partial state.
6. On a second chunked pass, measure the fitted power-diagram assignments
   against canonical `lstars`. Report mismatch count and
   `d_seg = mismatches / (600*384*512)` as
   `MEASURED_FEATURE_PULLBACK`, never as through-R receiver dseg.
7. Encode the fitted generator in strict `PDW1`, decode it, re-encode it, and
   require byte identity. Report actual raw bytes, actual Brotli quality-11
   bytes, and a clearly labeled optimistic rounded-up ideal order-0 entropy
   estimate under free-PMF and zero-overhead assumptions. It is neither a
   realizable ceiling nor a lower bound on a legal archive.
   Compare actual Brotli bytes with 228,764 B (optimistic shared-edge MDL
   contour), 235,974 B (optimistic contour+xi), and 225,272 B (strict sub-0.15
   threshold). Do not call a tiny generator payload an equivalent rate win when
   the spatial quotient field and RGB receiver are absent.
8. Atomically write a machine-readable JSON receipt containing exact argv,
   paths, byte counts, SHA-256 hashes, source/model/cache custody, Torch/runtime
   custody, frame order, sample count, measured/derived labels, positive-control
   result, fitted result, rate comparison, and the narrow verdict.
9. Preserve every partial or blocked feature cache. Any future cleanup design
   must prove deterministic rebuild from fresh output, validate the newly
   generated receipt/command rather than replaying an existing path, and receive
   separate review before it can authorize deletion. This landing performs no
   cleanup; the historical cleanup functions are tombstoned.

## Unit acceptance

- Streaming sufficient statistics and fitted assignments agree with a dense
  small-fixture reference.
- Quotient convolution agrees with an explicit unfold/matrix calculation on a
  small deterministic tensor.
- `PDW1` parse-back is byte-identical and compression accounting is exact.
- Every historical run/resume/certify/cleanup entrypoint now refuses. The
  deterministic gzip container is decompressed only in memory for hash/size
  validation; direct interpreter execution fails without mutation. Read-only
  evidence tools expose no cleanup/resume API.
- The harvester rejects container/tombstone masquerade, verifies checkpoint
  lineage against the manifest, confines output to an existing resolved
  `REPO_ROOT/.omx/research/*.json` path, refuses overwrite/parent creation, and
  proves checkpoint/cache hashes and metadata unchanged after use.
- The governed one-frame diagnostic is deterministic and proves its inputs
  unchanged with a post-inference full-input rehash while independently
  reproducing the frame-195 arithmetic split.
- Receipt validation rejects any claim of receiver or contest-score authority.

## Measurement and landing

Before the n600 pass, create a storage-waterfall plan and launch only through
`tools/safe_run.py`; a governor refusal is a measured blocker. Keep all bulk on
the SSD tier. After measurement, land the JSON receipt, a dated findings memo,
an append-only DAG FEED for Task #543, and a factor-6 completeness update.
Keep the equation as a temporary candidate if receiver closure is absent.
Round 1 must include an author confound hunt, followed by an independent
fresh-eyes reviewer and three clean review passes. Final commits require the
serializer and MAIN landing review.
