# Rate-attack legal-receiver-path audit + Codex F1 finding relay
# Date: 2026-05-18
# Codex's adversarial finding (operator-relayed 2026-05-18 verbatim): "dims 7-12 are genuinely unscored inside PoseNet, but they are not a contest-visible byte channel without a legal receiver path."
# Per CLAUDE.md "Strict scorer rule — non-negotiable (canonical, binding)" Catalog #6 + "Deterministic packet compiler" contest_one_video_replay target + HNeRV parity discipline L4 inflate.py ≤200 LOC + 2 dep budget
# Sister to: PRIMARY rate-attack memo (`rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` commit 2cae89a87 estimated) + ADVERSARIAL paradigm-challenger memo (commit 4c6e46bfa) + supplement (commit d43ecddb0) + in-flight SYNTHESIS-V2 subagent (`a18c228872a761bdb`)

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Strict scorer rule — non-negotiable" Catalog #6 + "HNeRV / leaderboard-implementation parity discipline" L4 inflate.py LOC budget + Catalog #229 premise verification + Catalog #287 evidence tags)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `upstream/modules.py` (line 26: `HEADS = [Head('pose', 32, 12)]`; line 84: `compute_distortion` uses `[..., : h.out // 2]` — first 6 of 12 dims; SOURCE OF TRUTH for F1 claim)
4. `submissions/exact_current/inflate.py` (canonical inflate template; lines 11-28 read upstream video bytes)
5. `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` (PRIMARY master memo; 43 vectors; TOP-5 includes F1)
6. `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` (ADVERSARIAL memo; 5 binding critiques include Hotz "apparatus_maintenance" critique structurally similar to this finding)
7. `.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md` (supplement; per-axis matrix)
8. `src/tac/packet_compiler/deterministic_compiler.py` (canonical packet compiler; contest_one_video_replay profile)
9. `src/tac/packet_compiler/pr98_decode_side_nudge.py` (fixed-video generated-code precedent)
10. `runtime-rs/crates/tac-packet-compiler/README.md` (native packet-compiler parity surface)
11. Sister in-flight: `a18c228872a761bdb` SYNTHESIS-V2 reconciliation subagent (will incorporate this finding when it discovers this memo via `.omx/research/` glob)

## SECTION 0 — OPERATOR CORRECTION: A1-SPECIALIZED IS LIVE

This memo originally over-collapsed A1 into `STRICT_SCORER_RULE_VIOLATION`.
That was too conservative. The corrected classification is two separate
claims:

- **A1-CANONICAL / generic scorer receiver**: loading full PoseNet, SegNet, or a
  generic scorer-feature inverter at inflate time remains rejected. It either
  ships a catastrophic full-weight payload or violates the strict scorer rule.
- **A1-SPECIALIZED / deterministic packet compiler**: a tiny, self-contained
  per-pattern byte transducer, fixed table, generated code path, symbolic
  closed form, or distilled sparse/quantized native binary is a live canonical
  path under CLAUDE.md `contest_one_video_replay`.

The legality test is not "does it approximate a scorer concept?" The legality
test is:

1. all runtime code/data needed for replay is inside the archive or fixed
   contest runtime;
2. all shipped bytes are charged to the archive, with no hidden sidecars,
   network state, scorer modification, or uncharged dependency;
3. the scored `inflate.sh archive_dir output_dir file_list` path consumes the
   packet and emits contest RGB frames deterministically;
4. exact CUDA auth eval validates the archive/runtime packet;
5. the packet compiler records a typed runtime-consumption proof and payload
   change proof.

Therefore every `STRICT_SCORER_RULE_VIOLATION` label below should be read as
applying to the naive full-scorer/full-generic-inverter variant only unless the
row explicitly says otherwise. Specialized packet-compiler variants are
`RECLAIMABLE_VIA_PACKET_COMPILER` pending a byte-size proof and exact-eval
artifact.

## SECTION 1 — CODEX'S FINDING (operator-relayed 2026-05-18)

**Verbatim**: *"dims 7-12 are genuinely unscored inside PoseNet, but they are not a contest-visible byte channel without a legal receiver path."*

**Source verification per Catalog #229**:
- `upstream/modules.py:26`: `HEADS = [Head('pose', 32, 12)]` → PoseNet outputs 12 dims (HARD-EARNED-VERIFIED)
- `upstream/modules.py:84`: `compute_distortion` uses `[..., : h.out // 2]` where `h.out = 12` → only first 6 dims contribute to d_pose (HARD-EARNED-VERIFIED)
- ✅ Codex's premise is structurally correct: dims 7-12 ARE unscored in `compute_distortion`

**Where my F1 framing was wrong**: I conflated two different concepts:
- **TRUE**: PoseNet dim 7-12 are unscored (no contribution to d_pose)
- **FALSE (what I implied)**: Dims 7-12 are a "free byte channel in the archive"

The TRUE statement is a property of `compute_distortion`'s indexing. The FALSE statement implies an exploit shape that doesn't exist: dims 7-12 are NOT in the archive bytes — they're COMPUTED by PoseNet at inflate time from RGB inputs. There's no "dim 7-12 territory" in archive.zip for us to populate with information.

## SECTION 2 — STRUCTURAL DIAGNOSIS

### The pipeline (canonical):
```
Archive bytes → inflate.py → RGB frames → SCORER (T4) → PoseNet → 12-dim pose tensor
                                                              ↓
                                                     compute_distortion uses [:6]
                                                              ↓
                                                          d_pose → score
```

### F1 as I originally framed it (BROKEN):
"Encode information in dim 7-12 → free bytes from scorer's perspective"

**Why it's broken**: there's no encoder path FROM archive bytes TO "dim 7-12 territory". Dims 7-12 are OUTPUTS of PoseNet's forward pass on RGB inputs. The only way to influence them is to perturb the RGB inputs (which are themselves OUTPUTS of inflate.py reading archive bytes).

### F1 corrected probe framing (canonical):

The genuine F1 hypothesis is: **find RGB perturbations whose corresponding PoseNet first-6-dim output is INVARIANT** (i.e., perturbations that change dim 7-12 but NOT dim 1-6). Those perturbations would be SCORE-INVARIANT but pixel-level-distinct. The legal receiver path exists for a probe because standard inflate emits RGB and the scorer reads RGB; the unproven part is capacity and net archive-byte savings.

This is STRUCTURALLY IDENTICAL to vector A2 (Adversarial steganography on scorer blind-spots — Fridrich + Yousfi PhD territory) applied specifically to the PoseNet first-6-dim invariance manifold.

```
                  ENCODER PATH                       RECEIVER PATH
                  ─────────────────────────────────  ─────────────────────────────────
Original framing  Information → "dim 7-12 channel"   (impossible — no path exists)
(BROKEN)          → archive bytes

Canonical         Information → RGB perturbation     Standard inflate.py reads archive
probe framing     in PoseNet-1to6-invariant          → produces RGB frames (perturbations
(UNPROVEN)        manifold → archive bytes (entropy  embedded) → scorer reads RGB →
                  coded; smaller because we have     PoseNet → 12-dim → compute_distortion
                  more degrees of freedom)           ignores dim 7-12 → score unchanged
```

The exploit value is REAL but the mechanism is different than I described. The compression ratio comes from:
- Larger entropy of allowed RGB perturbations (more bits available per pixel)
- Scorer-blindness to those perturbations

## SECTION 3 — 43-VECTOR LEGAL-RECEIVER-PATH AUDIT

For each of the 43 vectors enumerated in PRIMARY rate-attack memo + supplement, classify:
- **NO_RECEIVER_NEEDED**: standard inflate.py + scorer is sufficient; exploit is scorer-blind input
- **LEGAL_RECEIVER_IN_BUDGET**: requires inflate.py code addition WITHIN ≤200 LOC + ≤2 dep budget per HNeRV parity L4
- **LEGAL_RECEIVER_OVER_BUDGET**: requires inflate.py changes EXCEEDING budget; structural risk
- **STRICT_SCORER_RULE_VIOLATION**: requires loading scorer component at inflate; FORBIDDEN per Catalog #6 (~73MB rate hit + non-compliance per CLAUDE.md non-negotiable)
- **RECLAIMABLE_VIA_PACKET_COMPILER**: naive scorer-load framing is forbidden,
  but a tiny self-contained deterministic transducer/generated-code/fixed-table
  variant is admissible if exact CUDA auth eval validates it

### Category A — SCORER-AWARE BYTE-LEVEL (3 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| A1-CANONICAL scorer-feature-space encoding (generic/full scorer) | YES — full scorer or generic scorer-feature inverter | **STRICT_SCORER_RULE_VIOLATION** ⚠️ |
| A1-SPECIALIZED deterministic packet compiler | YES — tiny per-pattern transducer/fixed table/generated code; no full scorer | **RECLAIMABLE_VIA_PACKET_COMPILER** ✓ pending byte-size proof + exact CUDA auth eval |
| A2 Adversarial steganography on scorer blind-spots | NO — scorer IS the receiver | **NO_RECEIVER_NEEDED** ✓ |
| A3 MIN-CARDINALITY adversarial pruning | NO — encode-time pruning; standard inflate | **NO_RECEIVER_NEEDED** ✓ |

### Category B — DECODER-SIDE INFORMATION (4 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| B1 Contest-video-as-codebook | YES — need byte-index→video-byte lookup; ~5-30 LOC | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| B2 Distributional encoding (encode dist + seed) | YES — need sampler at inflate; depends on distribution complexity | **LEGAL_RECEIVER_IN_BUDGET** ✓ (if simple distribution) |
| B3 Decoder-driven byte rejection (encode tests) | YES — need test executor at inflate; depends on test complexity | **LEGAL_RECEIVER_IN_BUDGET** ✓ to **LEGAL_RECEIVER_OVER_BUDGET** ⚠️ |
| B4 Inflate.py code-as-bytes (Turing-complete) | YES — by definition the code IS the receiver | **LEGAL_RECEIVER_IN_BUDGET** ✓ (counted in LOC budget) |

### Category C — CROSS-ARCHIVE / TEMPORAL (3 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| C1 Cross-archive bytes-as-libraries | NO — bytes from sibling files OUTSIDE archive per maintainer precedent | **NO_RECEIVER_NEEDED** ✓ |
| C2 Non-obvious time-domain patterns | YES — need temporal pattern decoder at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| C3 Negative-cost bytes via error correction (Filler STC) | YES — need EC decoder at inflate; ~50-100 LOC | **LEGAL_RECEIVER_IN_BUDGET** ✓ |

### Category META — ZIP/STRUCTURAL OVERHEAD (3 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| M1 ZIP STORED method | NO — ZIP unzip handles it | **NO_RECEIVER_NEEDED** ✓ |
| M2 Minimum-overhead ZIP headers | NO — ZIP unzip handles it | **NO_RECEIVER_NEEDED** ✓ |
| M3 Zero-byte / dead-byte audit | NO — encode-time audit; nothing to decode | **NO_RECEIVER_NEEDED** ✓ |

### Category D — YUV-NATIVE (7 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| Y1 YUV-native encoding (skip RGB conversion) | YES — need YUV→RGB at inflate (or YUV→scorer if scorer takes YUV) | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| Y2 Chroma-only encoding | YES — need luma-from-prediction at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| Y3 Luma-only encoding (Quantizr proven) | YES — chroma reconstruction at inflate; Quantizr precedent | **LEGAL_RECEIVER_IN_BUDGET** ✓ (proven) |
| Y4 YUV 4:2:0 chroma subsampling | YES — chroma upsampling at inflate; standard codec | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| Y5 DCT-domain encoding in YUV blocks | YES — inverse DCT at inflate; standard | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| Y6 JPEG quantization tables as steganographic carriers | YES — JPEG decoder; could use NVJPEG | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| Y7 YUV adaptive bit-depth per channel | YES — per-channel bit-depth unpacker | **LEGAL_RECEIVER_IN_BUDGET** ✓ |

### Category E — HARDWARE-CODEC (9 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| H1 NVDEC hardware video decode | YES — NVDEC API at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H2 NVENC hardware video encode | (compress-time only; no inflate receiver) | **NO_RECEIVER_NEEDED** ✓ |
| H3 GPU tensor-core native formats (fp4/fp8) | YES — fp4/fp8 dequantizer at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H4 NVJPEG hardware JPEG decode | YES — NVJPEG API at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H5 CPU SIMD bit-packing | YES — bit-unpacker at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H6 CUDA sparse tensor formats | YES — sparse dequantizer at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H7 VVC / H.266 codec | YES — VVC decoder | **LEGAL_RECEIVER_OVER_BUDGET** ⚠️ (VVC decoder is large; may bust ≤200 LOC + ≤2 dep budget) |
| H8 AV1 mode/profile optimization | YES — AV1 decoder; NVDEC has native | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| H9 NVIDIA DALI pipeline | YES — DALI consumes bytes; could be heavy | **LEGAL_RECEIVER_OVER_BUDGET** ⚠️ (DALI ≠ standard inflate.py environment) |

### Category F — HYDRA / DUAL-HEAD (7 vectors) — **CODEX'S FINDING APPLIES HERE**

| Vector | Receiver path needed? | Status |
|---|---|---|
| F1 PoseNet Hydra dims 7-12 (originally framed) | YES — would need scorer at inflate to "read" dim 7-12 | **STRICT_SCORER_RULE_VIOLATION** ⚠️ |
| **F1-CORRECTED (scorer-blind RGB perturbation)** | **NO — scorer IS the receiver via RGB invariance manifold** | **LEGAL_RECEIVER_PATH_EXISTS_FOR_PROBE; CAPACITY_AND_NET_SAVINGS_UNPROVEN** (collapses to A2) |
| F2 SegNet non-argmax logits | YES if framed as "encode in logits"; NO if framed as scorer-blind | **Same as F1** ⚠️ → ✓ |
| F3 PoseNet vision(2048) feature-space encoding | YES — generic vision-2048 inverter is forbidden; specialized byte transducer may be tiny | **RECLAIMABLE_VIA_PACKET_COMPILER** ⚠️ |
| F4 PoseNet summary(512) bottleneck | YES — generic summary-512 inverter is forbidden; specialized bottleneck table may be tiny | **RECLAIMABLE_VIA_PACKET_COMPILER** ⚠️ |
| F5 PoseNet ResBlock output deterministic | YES — generic ResBlock inverter is forbidden; specialized generated code may be tiny | **RECLAIMABLE_VIA_PACKET_COMPILER** ⚠️ |
| F6 Hydra trunk-vs-head split | YES — generic trunk/head inverter is forbidden; specialized split transducer may be tiny | **RECLAIMABLE_VIA_PACKET_COMPILER** ⚠️ |
| F7 PR95 Phase 2-4 dual-RGB-head | YES — dual-head architecture | **Depends on framing** ⚠️ |

**Verdict for F-category**: The direct "dims/logits/intermediate tensors are free
archive channels" framing is broken. Reframed as scorer-blind RGB
perturbations, F1/F2 become **legal-receiver-path A2 probe instances** whose
capacity and net byte savings remain unproven. Reframed as specialized
deterministic packet-compiler transducers, F3-F6 remain live
research/implementation candidates pending byte-size proof, runtime-consumption
proof, and exact CUDA auth eval.

### Category G — CPU-vs-GPU STRUCTURAL ASYMMETRY (7 vectors)

| Vector | Receiver path needed? | Status |
|---|---|---|
| G1 CPU-axis-specific optimization (re-rank existing) | NO — scoring choice between existing archives | **NO_RECEIVER_NEEDED** ✓ |
| G2 AVX-512 / NEON SIMD-aligned bit-packing | YES — bit-unpacker at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| G3 PyTorch CPU-MKL kernel-specific numerics | (compress-time only; no inflate receiver) | **NO_RECEIVER_NEEDED** ✓ |
| G4 fp32 vs fp64 vs CPU 80-bit extended | (compress-time numerics) | **NO_RECEIVER_NEEDED** ✓ |
| G5 CPU cache-line vs GPU SM access pattern | YES — aligned reads at inflate | **LEGAL_RECEIVER_IN_BUDGET** ✓ |
| G6 CPU-CUDA score-gap structural exploit | NO — exploit the gap via existing scorer | **NO_RECEIVER_NEEDED** ✓ |
| G7 Inflate device selection per Catalog #205 | NO — already canonical helper | **NO_RECEIVER_NEEDED** ✓ |

## SECTION 4 — VECTORS FLAGGED BY THIS AUDIT

### Vectors that need RE-FRAMING (canonical-collapse to scorer-blind input perturbation)

- F1, F2 (when framed as "dim 7-12 channel" / "non-argmax logits channel") → collapse to A2
- F3, F4, F5, F6 (encode in scorer intermediate features) → generic scorer-load
  variants fail closed; specialized packet-compiler variants remain live

### Vectors that need budget audit (LEGAL_RECEIVER_OVER_BUDGET candidates)

- H7 VVC / H.266 codec — decoder size may bust ≤200 LOC budget; needs empirical measurement
- H9 NVIDIA DALI pipeline — DALI ≠ standard inflate.py environment; needs structural-fit audit
- B3 Decoder-driven byte rejection — depends on test complexity; needs per-test audit

### Vectors that need deterministic-packet-compiler re-classification

- A1-SPECIALIZED scorer-feature-space encoding — viable when reduced to a
  small self-contained byte transducer/fixed table/generated code packet
- F3, F4, F5, F6 — viable only as specialized per-pattern transducers; generic
  scorer-intermediate components remain rejected
- These should be routed through the deterministic packet compiler feasibility
  path, not silently killed and not silently promoted

## SECTION 5 — IMPLICATIONS FOR PRIMARY + ADVERSARIAL + SYNTHESIS-V2

### For PRIMARY rate-attack memo (already landed)

PRIMARY's TOP-5 likely included F1 as TOP-1. Per this audit, F1's canonical instance is A2 — not a separate vector. PRIMARY's TOP-5 needs RE-RANKING after Codex's finding:
- Remove F1 as a distinct vector (collapse to A2)
- Promote A2 as a TOP-5 candidate (Fridrich + Yousfi PhD territory; canonical steganalysis)
- Re-check whether other TOP-5 vectors have similar issues (likely G1 and B1 are clean per audit above)

### For ADVERSARIAL paradigm-challenger memo (already landed)

ADVERSARIAL's "5 binding critiques" were structurally correct in spirit. Codex's specific F1 finding empirically validates Boyd's composition-infeasibility critique (sub-additive default) AND Assumption-Adversary's empirically-pending classification. The Hotz "apparatus_maintenance NOT frontier-breaking" critique is partly validated (F-category vectors WERE apparatus-maintenance if they required scorer-load).

### For SYNTHESIS-V2 in-flight subagent (`a18c228872a761bdb`)

SYNTHESIS-V2 should incorporate this audit + Codex's finding into its reconciliation:
- F1/F2 direct-channel framing collapses to A2; F3-F6 require deterministic
  packet-compiler reclaimability audit
- A1-SPECIALIZED should be reopened as a high-priority feasibility path
- G1 + G6 + G7 (SATURATION-INDEPENDENT + NO_RECEIVER_NEEDED) become highest-EV candidates
- A2 + B1 + C1 become structurally strongest (NO_RECEIVER_NEEDED canonical)

If SYNTHESIS-V2 doesn't pick this up via `.omx/research/` glob, write a follow-on memo to explicitly cite.

## SECTION 6 — RECOMMENDED RECONCILED TOP-5 (strict no-full-scorer subset)

Per this audit's NO_RECEIVER_NEEDED + SATURATION-INDEPENDENT + HARD-EARNED criteria:

| Rank | Vector | Cost | Receiver path | Saturation-dep | Empirical anchor |
|---|---|---|---|---|---|
| 1 | **G1** CPU-axis re-rank | $0 | NONE | NO | PR102 +0.033 gap (HARD-EARNED) |
| 2 | **G6** CPU-CUDA gap exploit | $0 | NONE | NO | Same as G1 |
| 3 | **A2** Adversarial steganography on scorer blind-spots (subsumes F1) | $0-$5 | NONE (scorer IS receiver) | NO (input-side exploit) | Fridrich PhD canonical |
| 4 | **B1** Contest-video-as-codebook | $0-$5 | LEGAL_RECEIVER_IN_BUDGET (~30 LOC) | YES (depends on A-2 N-7) | Wyner-Ziv 1976 canonical |
| 5 | **C1** Cross-archive bytes-as-libraries | $0 | NONE (sibling files OUTSIDE rate) | NO | Maintainer precedent per CLAUDE.md |

This RECONCILED TOP-5 is the strict no-full-scorer/no-specialized-binary subset.
It does not supersede A1-SPECIALIZED; instead, A1-SPECIALIZED becomes a
parallel highest-EV feasibility track under the deterministic packet compiler.

## DISCIPLINE

- Catalog #229 premise verification: `upstream/modules.py:26` + `:84` source-verified
- Catalog #287 evidence tags throughout
- Catalog #6 strict-scorer-rule: naive full-scorer variants flagged explicitly
- Deterministic packet compiler: specialized per-pattern transducer variants
  preserved as live `contest_one_video_replay` candidates
- HNeRV parity discipline L4 inflate.py ≤200 LOC + ≤2 dep budget: vectors at risk flagged LEGAL_RECEIVER_OVER_BUDGET
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha
- Catalog #314 absorption avoidance: this memo's scope is `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_*.md` ONLY

## 6-HOOK WIRE-IN DECLARATION per Catalog #125

1. Sensitivity-map contribution: N/A (audit-only memo)
2. Pareto constraint: N/A
3. Bit-allocator hook: N/A
4. Cathedral autopilot dispatch hook: **ACTIVE** — autopilot ranker should weight vectors by NO_RECEIVER_NEEDED + HARD-EARNED criteria from this audit
5. Continual-learning posterior update: **ACTIVE** via council deliberation anchor when SYNTHESIS-V2 cites this audit
6. Probe-disambiguator: **ACTIVE** — this audit IS the canonical disambiguator between "exploits unscored dim 7-12 territory" (broken framing) vs "exploits scorer-blind input perturbations" (canonical framing)

## CROSS-REFERENCES

- PRIMARY rate-attack master memo (sister; F1 TOP-1 ranking should be re-ranked per this audit)
- ADVERSARIAL paradigm challenger (sister; 5 critiques empirically validated for F-category)
- Supplement (sister; per-axis matrix unchanged by this audit)
- SYNTHESIS-V2 in-flight (will incorporate this audit when discovered)
- Codex routing directives (G1 + A-2 N-7 are unaffected; both already align with this audit's recommendations)
- Cross-stack synthesis 9×9 matrix (sub-additive default validated by F-category collapse)
- CLAUDE.md "Strict scorer rule — non-negotiable" Catalog #6 (canonical reference)
- HNeRV parity discipline L4 inflate.py LOC budget (canonical reference)

— Main-Claude 2026-05-18 (Codex F1 finding relay + 43-vector legal-receiver-path audit per operator question "do we need to build or add a legal receiver path or something")
