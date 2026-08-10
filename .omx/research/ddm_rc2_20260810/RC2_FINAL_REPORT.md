# ddm_rc2 — corrected reopening complete; both reference families lose bytes

**UTC:** 2026-08-10T04:50:42Z  
**Owner:** codex `ddm_rc2`  
**Axis:** `[macOS-CPU advisory]`, scorer-free, `score_claim=false`  
**State:** `COMPLETE-CLOSED-NO-BYTE-WIN`  
**Base:** PR130 CPR1, `archive.zip` 191,052 B, SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`

## Conclusion

The rc2 reopening was not stale: task #996 had not built either chartered reference family. rc2
has now done so on retained real streams. Higher-order PPMd adaptive arithmetic loses on all four
unchanged serialized PR130 sections. Real LDPC syndrome coding with deterministic min-sum belief
propagation loses on the one applicable residual object, the HPAC top-class hit field. Both families
are closed at the chartered reference form on this unchanged base.

This is a coder-axis negative, not a global nonexistence claim. It says to stop retrying PPMd-style
recoding of these four serialized objects and LDPC/BP coding of this causal HPAC hit formulation.
A future arm must bring a different model, representation, or legitimately stronger receiver-owned
side-information object. No candidate was integrated into `archive.zip`, no scorer ran, and the
frontier did not move.

## #996 scope correction

| Surface | In task #996? | rc2 disposition |
|---|---:|---|
| Existing HPAC probability model: Range versus ANS | Yes | #996 measured it; ANS won 2,120 B. |
| Generic whole-section recodes and memoryless/order-1 comparisons | Yes | #996 measured or bounded its named set. |
| PPMd higher-order adaptive context model with retained real stream | No | rc2 measured 144 packets; family loses on every section. |
| BP/LDPC syndrome stream with iterative decode | No | rc2 measured six n600 native packets; family loses on the applicable token residual. |

The downstream sentence “coder axis closed” was therefore too broad before rc2. It is now supported
for these newly named reference formulations as well as #996's original candidate set.

## Adaptive-arithmetic race

Implementation: Python 3.13.12 calling `pyppmd` 1.3.1's native PPMd7/PPMd8 arithmetic codecs.
The grid was 2 variants × 6 orders (`2,4,6,8,12,16`) × 3 memory sizes (1, 4, 16 MiB) × 4
sections = 144 real packets. The counted packet includes a 12 B parameter/length header. Every
packet was retained; 136 decoded exactly. Eight library candidates failed decode and retain both
their packet and error receipt, so they are candidate-invalid rather than silently dropped.

| section | raw | incumbent | memoryless/model comparison | best exact PPMd packet | delta vs incumbent | native decode |
|---|---:|---:|---:|---:|---:|---:|
| token wire | 116,980 | ANS 114,860 | HPAC cross-entropy 114,852 | 119,478 | **+4,618** | 0.023059 s |
| semantic raw | 40,252 | Brotli-q11 35,033 | order-0 36,805 | 37,296 | **+2,263** | 0.007589 s |
| pose raw | 23,054 | shipped 23,054 | order-0 22,989 | 23,495 | **+441** | 0.004773 s |
| HPAC raw | 20,179 | Brotli-q11 14,962 | order-0 16,567 | 15,562 | **+600** | 0.002709 s |

Every best row is PPMd7 variant H, order 2, 1 MiB. Larger contexts and memories did not reverse any
byte verdict. The token row is explicitly a recode of the 116,980 B serialized token wire; it is not
misrepresented as a new HPAC model over the 117,964,800 latent class symbols.

Verdict scope: **FAMILY — PPMd-style adaptive arithmetic recoding of the unchanged serialized PR130
sections. LOSES-BYTES.** Decode time is reported only; it did not gate the verdict.

## BP/LDPC-syndrome race

Applicability was established before encoding. The token receiver owns HPAC per-group logits after
causal prefix decode, so those logits are legitimate correlated side information. The semantic,
pose, and HPAC raw byte objects have no decoder-owned correlated source on the unchanged base and are
typed `NOT_APPLICABLE`; inventing one would change the representation or model.

The native Rust reference constructs a deterministic sparse parity graph, sends the syndrome of the
HPAC top-class hit field, runs normalized min-sum BP, selects an exact raw hit-vector fallback only
when BP fails to recover the encoder object, and arithmetic-codes each real miss class under the HPAC
probabilities. All flags, selected syndrome or fallback bits, miss bits, and framing are counted.
The attempted syndrome is retained separately for audit even when fallback wins.

| native variant | n600 packet | delta vs ANS 114,860 | fallbacks / 114,000 groups | native decode sum |
|---|---:|---:|---:|---:|
| alpha 4.0, degree 4, 30 iters | **655,769** | **+540,909** | 1,886 | 74.601 s |
| alpha 5.0, degree 4, 30 iters | 674,242 | +559,382 | 980 | 54.699 s |
| alpha 6.0, degree 4, 30 iters | 748,495 | +633,635 | 593 | 43.637 s |
| alpha 8.0, degree 4, 30 iters | 952,770 | +837,910 | 291 | 32.524 s |
| alpha 5.0, degree 3, 30 iters | 702,059 | +587,199 | 1,081 | 32.646 s |
| alpha 8.0, degree 3, 30 iters | 960,323 | +845,463 | 323 | 22.562 s |

All six assembled 117,964,800-symbol decoded outputs match source SHA-256
`8eb51ab7a2884c9d7b6e73ee60f78ded38c691d6b82e639b75dddec6e0ac1366`. Sending more parity reduces
fallback count and decode time but increases total bytes; the low-parity end is already 5.71× the ANS
incumbent. Degree-3 controls also lose. This is not a decoder-speed rejection.

Verdict scope: **FAMILY — LDPC/BP syndrome coding of the unchanged PR130 causal HPAC top-class
hit/residual formulation. LOSES-BYTES.** Other sections are `NOT_APPLICABLE`, not negative rows.

## Native parity and negative control

The authoritative native grid uses the PR130 NumPy/Torch first-equal-maximum argmax rule and records
227,035 miss symbols. An earlier last-on-tie grid created 752 avoidable misses; its complete payload
tree remains retained but is explicitly superseded. The correction improved the best packet by 193 B
and did not change the verdict.

The Rust packet semantics have an independent Python oracle. On retained frames 0:2, Python, Rust,
and the source produce the same 393,216 B object with SHA-256
`ef719b437b4436215b3186626a323c4a997c9d3ae4d8c26a4cefa0deac8557db`. The golden contract states
the first-equal-maximum tie rule explicitly. A retained one-bit mutation at packet byte 64 causes both
Python and Rust BP decoders to fail on a non-fallback group and does not reproduce the source. The
Rust CLI reads the source-symbol NPY only to assert the research decode afterward; `decode_chunk`
reconstructs solely from the packet and receiver-owned HPAC logits.

No contest-runtime Rust lowering plan fired because neither family won bytes. The native LDPC
research reference and its parity surface remain reproducible evidence, not a promoted receiver.

## Payload custody and receipts

All materialized payloads are under the chartered APDataStore root. Vertigo was read-only input;
there are no temporary evidence paths and no candidate payload was discarded.

| receipt | bytes | SHA-256 |
|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_rc2_20260810/ppmd_reference/ppmd_reference_manifest.json` | 146,421 | `ea80b914974dbda887fb0109431641426b344ebefed87bf804234aa0f777a5e0` |
| `/Volumes/APDataStore/pact/ddm_rc2_20260810/ldpc_reference_tiefirst/ldpc_reference_manifest.json` | 959,021 | `cddca3a5a27d1876f5295474bdcdfeee6d5b730aee74e4c91346a40ea0cef016` |
| `/Volumes/APDataStore/pact/ddm_rc2_20260810/ldpc_reference_tiefirst/golden_vector_alpha4000_d4/python_parity_receipt.json` | 1,974 | `bbfc6e29ba0de24038eaec1d32b6da6b216d559ab498bb2c981799c158cb66bd` |
| `/Volumes/APDataStore/pact/ddm_rc2_20260810/ldpc_reference_tiefirst/golden_vector_alpha4000_d4/rust_negative_control.log` | 633 | `dc662bb19f90ef45ca92b7c975b3331a7d896ca51cbc1b1baad499c371d6b8f3` |

The PPMd tree contains 293 non-AppleDouble files (111 MiB allocated); the authoritative LDPC tree contains 1,195
non-AppleDouble files (1.6 GiB allocated), including every stage packet, attempted syndrome, decoded
stage, metrics file, invocation transcript, six assembled packets, six full decoded outputs, retained
native binary, build log, golden output, mutation, and parity receipt. APDataStore created `._*`
metadata sidecars; they were left intact and are excluded only from the file-count denominator above.
The 1,194-file, 1.6 GiB last-on-tie predecessor tree is also retained under `ldpc_reference/` as
superseded evidence; no bytes were deleted when the wire rule was corrected.
The authoritative manifest is schema `ddm_rc2_ldpc_reference_race.v2`; all 168 stage receipts bind
the source payload hash and native binary SHA, and a completed resume pass reused only those matching
stages while reproducing every assembled byte count.

## Rule-118 boundary

Generic PPMd logic, deterministic sparse-graph generation from a non-video seed, min-sum BP,
finite-precision arithmetic operations, and packet parsing are free receiver code. Every coded stream,
syndrome or fallback flag, miss stream, parameter header, and video-derived model, table, reset
schedule, or graph choice is counted. Attempted syndromes and decoded outputs are research evidence
outside a hypothetical archive, and are labeled as such rather than hidden from the rate ledger.

## RECALL EVIDENCE

### Sources and queries searched

- The #996 primary receipts were read in full:
  `.omx/research/ddm_pr130_reproduce_20260809/RATE_AXIS_LOSSLESS_RACE.md` and
  `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`.
- Corpus query:
  `.venv/bin/python tools/corpus_query.py --stores research,equations,memory,dag,council,tasks,docs
  --top 30 --json 'PR130 adaptive arithmetic context model LDPC syndrome BP coder #996'`.
- Canonical equations were queried with `tools/list_canonical_equations.py --json` for arithmetic,
  Markov/context, syndrome, LDPC, residual, and entropy terms.
- The canonical research indices, sub-0.15 DAG, task/harness ledgers, and bridge snapshots were
  searched for #996 scope, coder ownership, receiver ownership, and prior native decode evidence.
- `tools/`, `src/tac/`, and `runtime-rs/` were searched for real encoder/decoder implementations,
  Python oracles, golden vectors, SHA parity, and negative-control patterns.
- Primary references consulted during the scope pass: Witten, Neal, and Cleary's adaptive arithmetic
  coding formulation (DOI `10.1145/214762.214771`); Liveris, Xiong, and Georghiades on compression
  using LDPC codes (DOI `10.1109/LCOMM.2002.804244`); and Filler et al. on syndrome-trellis coding
  (DOI `10.1109/TIFS.2011.2134094`).

### Findings beyond the charter seeds

1. `tools/pr101_markov1_aac_codec.py` is a real constriction-based adaptive range codec, but it is
   prior implementation evidence rather than a PR130 byte result.
2. `src/tac/codec/syndrome_trellis_codec.py` is a pedagogical distortion-minimizing STC and is not
   an LDPC/BP lossless receiver. It was not relabeled or reused as if it were one.
3. Existing Rust crates supplied the required finite-wire, Python-oracle, SHA-parity, negative-control,
   rebuild, and benchmark contract. The rc2 native reference follows that pattern.
4. The retained DT1 n600 chunk manifest supplies the actual HPAC symbols and per-group logits needed
   to make the token LDPC test legitimate and causal.
5. The HPAC source contains 227,035 model-top misses and has a decomposed ideal cost of 114,851.827 B;
   this confirms that exact miss values must be entropy-coded. A toy fixed-width flip stream would not
   be the requested reference form.

### What recall changed

Recall narrowed #996's claim to its measured candidate set, prevented a stale stop, prevented the STC
module from being passed off as LDPC/BP, required receiver-owned side information before syndrome
coding, and reused the repository's established native parity contract. rc2 then supplied the missing
real-stream verdicts instead of projecting them.

## Boundaries

- **MEASURED:** retained packet bytes, exact decoded hashes, named implementation decode times, native
  build identity, Python/Rust golden parity, and the corruption control.
- **RECALLED:** #996 incumbent and bound values and the base's exact contest score.
- **NOT MEASURED:** a changed-model adaptive token coder, a changed-representation section, composed
  archive bytes, evaluator components, and any new exact score.
- **Frontier:** PR130 remains `S = 0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`.
  rc2 did not move it.

## LIVE-HYPOTHESES

- A changed HPAC model may shrink the 114,860 B ANS token stream because the current model bound is
  114,852 B only for the current logits; this is a model-capacity/rate allocation problem, not another
  entropy-engine swap.
- A new residual representation with materially stronger receiver-owned correlation could make
  syndrome coding useful because LDPC source coding pays only when side information makes the virtual
  channel sufficiently clean. The current causal HPAC hit field is not clean enough at competitive
  parity rates.
- Semantic or pose bytes may shrink after a representation change that exposes tensor, geometry, or
  trajectory structure. PPMd's losses show only that higher-order byte contexts on the current wire
  objects do not expose enough of that structure.

## DEAD-ENDS

- Treating #996's original broad sentence as proof these families had already lost: its receipts did
  not contain the required streams.
- PPMd7/PPMd8 recoding of any unchanged serialized PR130 section across orders 2–16 and 1–16 MiB:
  every best exact packet is 441–4,618 B worse than its incumbent.
- LDPC/min-sum syndrome coding of the causal HPAC top-class hit field: the best exact n600 packet is
  540,909 B worse than ANS, and more parity only worsens bytes.
- Applying LDPC syndrome coding to unchanged semantic, pose, or HPAC raw bytes: no legitimate
  decoder-owned correlated source exists, so those cells are not applicable.
- Rejecting either family on wall-clock time: decode time is report-only and was not used as a gate.
- Lowering the losing adaptive decoder into the contest Rust runtime: the charter authorizes runtime
  lowering only for a byte winner, and none exists.
