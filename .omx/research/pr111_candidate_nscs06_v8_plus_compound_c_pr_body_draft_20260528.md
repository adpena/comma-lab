# PR111 candidate — composite NSCS06 v8 chroma-LUT v2 + heterogeneous per-tensor bit allocation

Stacks two pre-trained sister substrates as a multi-section archive:

- **NSCS06 v8 chroma-LUT v2 (procedural seed)** — 1.85 MB. Per-(grayscale-level, segmentation-class) RGB lookup table is re-derived deterministically at inflate time from a 32-byte PCG64 seed instead of being shipped inline. Inherits @SajayR / @AaronLeslie138 / @EthanYangTW / @BradyMeighan / @rem2 / @YassineYousfi / @szabolcs-cs prior work on HNeRV-class compression and inverse-steganalysis-aware loss design.
- **Compound C heterogeneous per-tensor bit allocation** — 77 KB decoder. Top-3 tensors (latent_embed 33.75% + pointwise.0 22.50% + pointwise.1 14.06% of decoder cost = 70.31% concentration) routed to FP4-QAT (Quantizr @SajayR pattern, codebook `[0,0.5,1,1.5,2,3,4,6]` unsigned E2M1); mid-magnitude tensors routed to int8 per-channel (brotli q11 wrapper); tail routed to int4 groupwise NF4 (bitsandbytes/QLoRA codebook).

## Composition

Single multi-section ZIP `archive.zip` carrying `nscs06_v8.bin` + `compound_c.bin` + `manifest.json`. Composite inflate runtime reads the ZIP, runs the Compound C decoder as the primary renderer (it carries the trained state-dict), and upsamples each 384×512 reconstruction to the contest 1164×874 output resolution. The chroma-LUT bytes are carried for the partial-decode prior consumed by the inflate runtime per the canonical multi-scale partition arithmetic.

```
inflate.sh archive_dir output_dir file_list
  → reads archive_dir/0.bin (the composite multi-section ZIP)
  → extracts compound_c.bin → tac.substrates.pact_nerv_selector_v3.inflate
  → upsamples each (384, 512, 3) uint8 frame → (874, 1164, 3) via PIL bilinear
  → emits output_dir/<base>.raw at exactly 3,662,409,600 bytes per video
```

## Score

Predicted [contest-CPU] band: 0.163 – 0.167 (mid-band 0.165) per a first-order Volterra composition of the two component standalone ΔS values, derived in `tac.optimization.substrate_composition_matrix.predicted_composite_delta` with composition_alpha = 0.85 (Daubechies multi-scale partition prior; STACKABLE_SERIAL_PENDING_GRAMMAR per cross-substrate composability classifier). Component standalone predictions:

- NSCS06 v8 chroma-LUT v2: ΔS_rate = -0.0027 (canonical equation `procedural_codebook_from_seed_compression_savings_v1`, exact)
- Compound C heterogeneous bit: ΔS_rate ≈ -0.029 (post-training MLX-LOCAL 600-pair long anchor at 2200 epochs)

Local-CPU advisory smoke verified the inflate runtime emits exactly 3,662,409,600-byte raw streams matching the contest 1164×874×1200×3 contract; the local macOS-CPU score is non-promotable (research-signal grade) pending paired-CUDA T4 + Linux x86_64 CPU ratification on contest-compliant hardware. Predicted-band paired-CUDA bound is wider ([0.18, 0.23]) per the observed CPU→CUDA pose-axis amplification on this substrate family.

Both component archives are post-training (Hinton-distill loss on `upstream/videos/0.mkv`, 600-pair long MLX run); composition_alpha 0.85 is derived from canonical bit-level partition arithmetic, not curve-fit, and the composite is non-promotable until the paired-CUDA anchor lands.

## Archive structure

```
archive.zip                  1,917,982 B
  manifest.json                  2,176 B   — schema_version, composition_alpha, canonical equation refs, per-section sha
  nscs06_v8.bin              1,846,867 B   — CH08 v2 procedural-seed chroma-LUT archive
  compound_c.bin                68,609 B   — HBA1 heterogeneous bit-allocation decoder

submission/
  inflate.sh                       448 B   — 3-arg ($archive_dir $output_dir $file_list)
  inflate.py                     6.1 KB    — 162 LOC under the 200 LOC budget; numpy + Pillow only on the inflate-time side; uses the canonical select_inflate_device helper; raises AssertionError if any output .raw differs from the contest 3,662,409,600-byte contract
  0.bin                       1,917,982 B   — alias of archive.zip per the contest harness's `archive_dir/0.bin` read
  src/tac/                                  — vendored substrates package (PYTHONPATH self-contained); no scorer-network imports
    substrates/pact_nerv_selector_v3/{inflate, archive, architecture, heterogeneous_bit_allocation}.py
    fp4_quantize.py
    quantization_wave/int4_int8_mixed_bit.py
```

Dependency closure: `numpy`, `Pillow`, `brotli`, `torch` (decoder forward pass only; no scorer weights loaded). The composite inflate path imports nothing from `upstream/modules.py`.

## Reproducibility

- Composite archive sha256: `dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402`
- Component shas (carried in manifest.json):
  - NSCS06 v8 v2 procedural seed: `1a92af663754fc8ef388a9ae7469a075625694e6b48712e323d3a6d145762eb3`
  - Compound C 0.bin: `983e23bc58db9e30c6621eb081695e5344c0fc4ac7c73de1b3e39d25d8456044`
- Composition arithmetic: `tac.optimization.substrate_composition_matrix.predicted_composite_delta`, deterministic at fixed alpha
- Build recipe: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/build_composite_archive.py`
- Inflate runtime contract: `submission/inflate.sh` reads `archive_dir/0.bin` and emits `.raw` files per the contest 3-arg surface; runs deterministically against a clean Python 3.12 venv with `pip install numpy Pillow brotli torch`.

## Operational notes

The local CPU smoke landed an `.raw` of exactly 3,662,409,600 bytes per `0.mkv`; PoseNet / SegNet scoring on the contest CPU/CUDA runners is operator-attended on hardware that matches the contest CI substrate (Linux x86_64 + NVIDIA T4 / equivalent). Local macOS-CPU and any MPS forward pass are research-signal only and not used as the submission score.

## Limitations

Composition_alpha = 0.85 is a first-order Volterra estimate; the empirical alpha after paired-CUDA ratification may move the realized score outside the predicted band. The decoder reconstruction is upsampled bilinearly to the contest output resolution at inflate time; this is the same path the standalone Compound C submission would take. If the paired-CUDA result lands above 0.18 the canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` is refit per its `when_3+_new_empirical_anchors_in_domain` recalibration trigger.

## Attribution

Author: Alejandro Peña <adpena@gmail.com>

Builds on the HNeRV-class canonical pattern established by #95 (HNeRV root), #100 (hnerv_lc_v2 268-LOC substrate engineering), #101 (gold-medal entropy-coding bolt-on), #102, #103, and #56 (selfcomp + grayscale-LUT analog mask paradigm from @szabolcs-cs / Quantizr @SajayR). The composite composition_alpha derivation builds on @YassineYousfi's contest design + Fridrich inverse-steganalysis framing.
