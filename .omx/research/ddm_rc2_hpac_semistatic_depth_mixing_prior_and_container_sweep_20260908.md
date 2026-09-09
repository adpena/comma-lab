# ddm_rc2 — semi-static and shared-depth HPAC priors on the live sj1 object

Date: 2026-09-09  
Tokens: `[no-triality] [p0-ledger-ok]`  
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`  
Score claim: `false`  
Verdict: **ADMIT the 8-byte shared logistic mixer to a sealed contest-CUDA replay; do not admit the semi-static table family.**

The measured candidate is 181,414 B, 231 B below the 181,645 B live archive. Its projected score at unchanged distortion is 0.13885056455024844, but that is arithmetic, not a contest score. The canonical pointer is unmoved until MAIN fires the validated seal.

## RECALL EVIDENCE

I searched the full local evidence corpus before building: `.omx/research/` and receipts for `IHS1|HPAC|model section|semi-static|conditional|first-order|logistic mix`; `experiments/ddm_rc1_*.py`; the canonical equations registry through `tools/list_canonical_equations.py --json`; `CANONICAL_RESEARCH_INDEX*`; the `sub015_DAG_*` FEED blocks; design/SPEC files; and the task ledger. The charter seeds were read in full, including the whole rc1 memo and its five conditioned negatives.

Beyond those seeds, four items changed the plan:

- `ddm_mz1_model_section_rate_race_20260815.md` supplied the framed-byte and retained parse-back precedent, but its measurements were on an older object and were not transferred numerically.
- `experiments/ddm_fx1_logistic_mixer_corrector.py` supplied the deterministic integer/dyadic log-odds construction. Only the construction transferred; no token-side saving did.
- `coder_strength_substitutes_for_capacity_v1` and the cl3 receipts established that capacity prices must be re-read under each model coder. This arm therefore changed no model capacity.
- The live RX1 header has reserved byte `0x7a`; every bit through `0x40` is already assigned and the current HPAC convention uses `0x40` to imply whole-rider ck2. Consequently `ck2=false` grid cells are measurement-only: there is no unambiguous free reserved bit with which to ship them. Selection was restricted to the nine `ck2=true` cells per object.

The DAG/index searches did not find another current-object semi-static IHS1 table or shared-depth HPAC mixer. Searches did find token-stream logistic mixtures; those were kept out of this model-section result because the objects and receiver surfaces differ.

The required implementation dispatch was attempted twice from `.omx/tmp/codex_runs/ddm_rc2_impl_SPEC.md`. Both attempts failed before reading the spec with `failed to initialize in-process app-server client: Operation not permitted`; the arbitrage two-strike escape hatch then authorized local implementation. Dispatch logs are retained under `.omx/tmp/codex_runs/ddm_rc2_impl_dispatch/`.

## Live object and custody

The source was re-read from `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime/archive.zip`: 181,645 B, SHA-256 `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`. The header is `(RX1M, v1, Brotli, table_mode=0, reserved=0x7a, hpac=12,343 B, semantic=30,246 B, carrier=18,621 B)`.

Brotli decode plus ck2 inverse produced the 16,267 B RC1H rider. The live receiver restored a 17,770 B IHS1 body, SHA-256 `817281908d993d89f62349fa466ad53f204b7e83d36e29ba0edc30cf2cb8f085`, byte-identical to rc1's retained body. Geometry is 517 rows and 20,416 signed values. Every materialized table, fitted weight set, arithmetic payload, rider, Brotli stream, archive, and smoke artifact is retained with byte count and SHA-256 under `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/`; the machine-readable census is `RESULT.json`.

Storage preflight found 18.9 GB free on the first-choice Vertigo tier. No payload was discarded.

## Closed-form ceiling, re-derived on this tree

For each bit depth (d), rows were concatenated in stored order. Let (n_x) count values, (n_{xy}) adjacent values, (K_0) the occupied marginal alphabet, (K_{xy}) the occupied transition alphabet, and (K_x) the occupied predecessor alphabet. The measured sums are:

\[
L_0=-\sum_x n_x\log_2(n_x/N),\qquad
L_{0,MM}=L_0+\frac{K_0-1}{2\ln 2},
\]

\[
L_1=-\sum_{xy}n_{xy}\log_2\frac{n_{xy}}{n_x}+L_{\text{first}},\qquad
L_{1,MM}=L_1+\frac{K_{xy}-K_x}{2\ln 2}.
\]

The first symbol at each depth is priced under that depth's order-0 distribution. Summing depths 0–8 gives:

| quantity | bits | bytes |
|---|---:|---:|
| order-0 plug-in | 76,862.2532 | 9,607.7816 |
| order-0 Miller–Madow | 77,122.6596 | 9,640.3325 |
| order-1 plug-in + first symbols | 64,712.1037 | 8,089.0130 |
| order-1 Miller–Madow + first symbols | 67,754.7475 | 8,469.3434 |
| current RC1 ideal adaptive-tree length | 77,115.7280 | 9,639.4660 |
| current RC1 realized range payload | — | 9,643 |

Thus the chased gap is `9,643 - 8,469.3434 = 1,173.6566 B` against realized bytes, or `9,639.4660 - 8,469.3434 = 1,170.1226 B` against RC1's ideal length. This re-derivation supersedes the charter's rounded `≈1,139 B` for this receipt; the qualitative conclusion is unchanged.

## Design A — counted two-pass frozen tables

Pass 1 counted bit-tree outcomes conditional on the previous value at the same depth across intervening rows; pass 2 used the frozen table. The first value at each depth remained uniform. Four context families raced with three counted sparse serializers: ULEB zero/one counts, quantized uint8 probability, and uint12 probability. Every row decoded to the exact IHS1 body.

`J = counted table bytes + arithmetic payload bytes`; RC1's comparison J is its 9,643 B arithmetic payload.

| context | table form | table B | coded B | J B | J saving vs RC1 B | q11/ck2 container B |
|---|---|---:|---:|---:|---:|---:|
| previous zero | ULEB counts | 2,113 | 9,553 | 11,666 | -2,023 | 13,522 |
| previous zero | uint8 p | 1,300 | 9,553 | **10,853** | **-1,210** | **12,927** |
| previous zero | uint12 p | 1,949 | 9,553 | 11,502 | -1,859 | 13,406 |
| previous sign/zero | ULEB counts | 3,087 | 9,373 | 12,460 | -2,817 | 13,891 |
| previous sign/zero | uint8 p | 1,938 | 9,373 | 11,311 | -1,668 | 13,088 |
| previous sign/zero | uint12 p | 2,906 | 9,373 | 12,279 | -2,636 | 13,673 |
| previous bit length | ULEB counts | 9,868 | 8,879 | 18,747 | -9,104 | 15,987 |
| previous bit length | uint8 p | 6,487 | 8,880 | 15,367 | -5,724 | 14,491 |
| previous bit length | uint12 p | 9,729 | 8,879 | 18,608 | -8,965 | 16,302 |
| exact previous value | ULEB counts | 29,625 | 8,094 | 37,719 | -28,076 | 21,850 |
| exact previous value | uint8 p | 19,727 | 8,097 | 27,824 | -18,181 | 18,956 |
| exact previous value | uint12 p | 29,570 | 8,094 | 37,664 | -28,021 | 23,678 |

The table-cost wall is decisive: the best table saves 90 B in arithmetic payload but costs 1,300 B. The prior-law prediction of a 300–600 B net semi-static conversion is falsified on this formulation.

## Design B — counted shared-depth logistic mixing

Eight causal adaptive predictors were mixed in fixed-point log-odds space: depth/node base, previous-zero, previous-sign, previous-bit-length, exact-previous, shared bit-position/node, bit-position, and global. Float64 L-BFGS-B fit the weights offline; int8 quantization at scale 32 plus one exact integer coordinate pass produced the transmitted weights. The receiver's coding decision uses only deterministic integer stretch, weighted sum, monotone integer squash, and already-decoded state. The weight bytes are inside RC2H and counted.

| counted weights | groups | coded B | J B | J saving vs RC1 B | q11/ck2 container B | container delta B |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 9,419 | 9,427 | +216 | **12,112** | **-231** |
| 16 | 2 | 9,406 | **9,422** | **+221** | 12,113 | -230 |
| 32 | 4 | 9,410 | 9,442 | +201 | 12,146 | -197 |
| 64 | 8 | 9,407 | 9,471 | +172 | 12,168 | -175 |

All four streams decoded exactly. The best raw J is the 16 B / two-group row, but the archive objective selects the 8 B / one-group row by 1 container byte. The predicted 100–300 B logistic conversion is confirmed: 216–221 B raw J and 231 B at the selected container. The charter falsifier does **not** fire because Design B nets at least 150 B.

Design A+B was pruned by its precondition: Design A does not beat the current 9,643 B J. Combining its paid table with the mixer would not test the claimed complementary-win mechanism.

## Complete container sweep

Each tuple below is `lgwin22 / lgwin23 / lgwin24` bytes. All 36 streams are retained and Brotli-round-tripped.

| object | ck2 | q9 | q10 | q11 | disposition |
|---|---:|---:|---:|---:|---|
| RC1 base | false | 13,093 / 13,093 / 13,093 | 12,401 / 12,401 / 12,401 | 12,372 / 12,372 / 12,372 | measurement-only; no free RX1 flag |
| RC1 base | true | 13,211 / 13,211 / 13,211 | 12,368 / 12,368 / 12,368 | **12,343 / 12,343 / 12,343** | shippable |
| RC2 8 B mixer | false | 12,823 / 12,823 / 12,823 | 12,198 / 12,198 / 12,198 | 12,164 / 12,164 / 12,164 | measurement-only; no free RX1 flag |
| RC2 8 B mixer | true | 12,962 / 12,962 / 12,962 | 12,154 / 12,154 / 12,154 | **12,112 / 12,112 / 12,112** | shippable |

The shippable argmin is a three-way lgwin tie at `ck2=true, q11`; lgwin 22 was selected deterministically. The base reproduces the shipped q11/lgwin24 stream byte-for-byte. Container retuning itself moves 0 B, within the predicted ≤60 B; the 231 B credit is the mixer, not a window-selection artifact.

## Candidate proof and seal

The live tree was copied without mutation, then only `runtime/rc2_hpac_semistatic_mixing.py`, the HPAC magic dispatch in `runtime/residual_archive.py`, the HPAC materializer dispatch in `runtime/ihs2.py`, archive pins, and the manifest were changed. The candidate and repeat archives are byte-identical:

- archive: 181,414 B, SHA-256 `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`;
- HPAC: 12,112 B, down 231 B;
- semantic: 30,246 B, SHA-256 `f1f0f85730981f2639edea2fdb36bf190c6c065a0c91b15afe4f6ffd0f99b186`, identical;
- carrier: 18,621 B, SHA-256 `fa18c86fe9158a4bb9a22df47f1ee2a68948c25c8cbbb865877cd40ac438064d`, identical;
- token tail: 120,321 B, SHA-256 `af6b0997bf26c446f2b235d8cad7a83b927c17a0bb53b40e18ef4e05ff4832dc`, identical;
- staged public receiver restores IHS1 to the exact 17,770 B / `81728190…` body;
- source 600-frame decoded token field: 117,964,800 B, SHA-256 `a73289e0a30dd765215fbb615f7802804b298ebdc816aeb9e921f32b2971f4d4`.

Per-pair receipts are not applicable: no decoded field value changes, proven by exact model restoration plus identical semantic, carrier, and token-tail sections.

The candidate and frontier each reached token decode through `runtime.f26_inflate.inflate_archive` at the 240 s bound and each `bash inflate.sh` reached the intentional `linux-nvidia-t4` CUDA gate. Both probes used one process; `num_threads=4` is required by the public receiver. A two-thread diagnostic was retained as a typed negative because the receiver explicitly refused it.

`SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json` validates `SEAL_VALID`, seal SHA-256 `56378bf8da946b102583b92264020f173cd3f42bbd89937e93208964cdb749d2`, against the live 181,645 B / 0.13900437796841966 contest-CUDA row. Its admission rule is net `dS < -0.0001`; rate-only arithmetic predicts:

\[
\Delta S=-231\cdot\frac{25}{37{,}545{,}489}=-0.0001538134181712216,
\qquad S_{\text{projected}}=0.13885056455024844.
\]

The seal also binds the base row's computed report-8dp bound. This is still `score_claim=false`; MAIN owns the T4 custody replay.

SEAL READY: `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json` — 181,414 B, SHA-256 `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`.

## Canonical equations leg

This result is formalized through `tac.canonical_equations.update_equation_with_empirical_anchor` on:

- `model_section_adaptive_recode_ceiling_v1`, anchor `rc2_semistatic_and_shared_logistic_model_prior_20260909`: the H1 ceiling is real, paid semi-static tables lose, and an 8–16 B shared mixer converts 216–221 B of raw J;
- `coder_strength_substitutes_for_capacity_v1`, anchor `rc2_shared_mixer_third_model_container_20260909`: a third current-object model container improves RC1 by 231 B while preserving the field, so future capacity prices must be re-run under RC2 rather than transferred from Brotli or RC1.

Catalog #344 is checked before handoff.

## Boundaries and dispositions

Measured here: exact bytes, closed-form entropy bounds, all 16 design rows, all 36 container cells, lossless receiver identity, section census, twin determinism, and the public smoke pair. Not done: no scorer, no MPS/Metal, no Modal, no model-capacity change, no permutation rerun, and no write to `upstream/`, sj1, or pc2.

- `QUEUED-WITH-A-FIRE-ORDER`: MAIN owns one contest-CUDA custody replay. Consumer store: the validated seal above. Fire trigger: a unique T4 lane is claimed and the canonical pointer still names archive SHA `06c44dc4…` at 181,645 B / 0.13900437796841966. On success, the measured components must preserve distortion and report archive SHA `c810c2…`; otherwise withdraw and diagnose.
- `FOLDED`: the semi-static table family is closed at FORMULATION scope for these four previous-value contexts and three serializers because every counted J loses at least 1,210 B to RC1.
- `FOLDED`: a combined semi-static+mixer arm is not opened because the charter's “both help” precondition is false.
- `FOLDED`: `ck2=false` cells are not staged because the live RX1 reserved-byte surface has no free unambiguous flag.
- `FOLDED`: model-capacity and permutation work remain closed by their inherited evidence and were not re-run.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
