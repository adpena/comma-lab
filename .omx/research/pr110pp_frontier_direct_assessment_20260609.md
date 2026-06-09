# V5 / PR110++ as the FRONTIER-DIRECT layer — assessment against the 0.19199 archive

**Subagent:** `pr110pp_frontier_direct_research_20260609` · UTC 2026-06-09 · READ-ONLY research+audit.
**Axis discipline:** everything below is `[macOS-CPU advisory]` / mechanism-only. No score, promotion,
rank/kill, or dispatch authority is minted here. No training, no GPU, no carrier edits, no dispatch.
**Provenance discipline** (`src/tac/optimization/audit_provenance.py`): every load-bearing claim cites
`{file, field, observed_value, reproduce_command}`. Git HEAD at audit: `b28a15fe1`.

---

## 0. The frame that changes the question (the central correction)

The prompt framed V5 PR110++ as a layer that "attaches per-pair selector atoms TO the existing frontier
archive." The audit shows a stronger fact that reframes the whole assessment:

> **The 0.19199 frontier archive IS ALREADY a PR110++ selector archive.** It is not a foreign carrier
> that selector atoms bolt onto — it is an HNeRV decoder + per-pair K=16 frame-0 selector packet, with a
> terminal FP11 brotli recode applied on top of the decoder blob only.

Provenance for this correction:

- **Pointer label is the last byte-transform, not the architecture.** `architecture_class =
  "fp11_source_brotli_recode_b7106c9bdbb8_cpu_exact"`
  (`.omx/state/canonical_frontier_pointer.json` → `our_local_frontier_contest_cpu.extra.architecture_class`).
  Reproduce: `python -c "import json;print(json.load(open('.omx/state/canonical_frontier_pointer.json'))['our_local_frontier_contest_cpu']['extra']['architecture_class'])"`
- **The actual inflate runtime is PR110++.** The frontier `inflate.py` imports `HNeRVDecoder`,
  `frame_selector.{PALETTE_MODE_IDS, apply_selector_to_frames, unpack_selector_indices}`, and defines
  `FEC6_FIXED_K16_MODE_IDS` (16 modes), `DQS1` decoder-q selector, `OUTER_MAGIC=b"FP11"`.
  File: `…/submission/fp11_source_brotli_recode_b7106c9bdbb8/inflate.py:1-90` (full path in §1).
  Reproduce: `grep -nE "HNeRVDecoder|PALETTE_MODE_IDS|FEC6_FIXED_K16_MODE_IDS|OUTER_MAGIC" <submission>/inflate.py`
- **The FP11 recode touched only the decoder blob, not the selector.** Runtime consumption proof:
  `decoder_raw_roundtrip_equal=True`, `selector_payload_unchanged=True`, `dqs1_tail_unchanged=True`,
  `source_tail_unchanged=True`, `decoder_saved_bytes=37`, `runtime_decoder_blob_len_before=162164`,
  `runtime_decoder_blob_len_after=162127`, `operation_family="source_brotli_recode"`.
  File: `<submission>/runtime_consumption_proof.json`.
  Reproduce: `python -c "import json;d=json.load(open('<submission>/runtime_consumption_proof.json'));print({k:d[k] for k in ['decoder_raw_roundtrip_equal','selector_payload_unchanged','dqs1_tail_unchanged','decoder_saved_bytes']})"`

So "V5 as the frontier-direct layer" is not a new attachment problem — it is the question of whether the
**existing selector inside the frontier** can be improved (or extended) to move exact `evaluate.py` score,
and whether the PR110++ tooling fleet can produce ONE exact `ActionEffect` row proving it.

---

## 1. The frontier archive — what it structurally IS (byte budget)

- **Archive path (LOCAL, bytes verified):**
  `.omx/research/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/results/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/per_archive/current_contest_cpu_frontier/fp11_source_brotli_recode_v1/fp11_source_brotli_recode/exact_eval_handoff/submission_closure/submission/fp11_source_brotli_recode_b7106c9bdbb8/`
- **archive.zip:** 178,493 bytes, sha256 `b7106c9bdbb8…8997c8c` — **verified on disk**.
  Reproduce: `shasum -a 256 <archive.zip>` → matches pointer.
- **ZIP structure:** ONE stored (method 0) member `x`, 178,393 bytes. (`unzip -l <archive.zip>`)
- **Member `x` internal byte budget** (from `<submission>/src/codec.py:26-27,174-176`):
  | section | bytes | share | content |
  |---|---:|---:|---|
  | `DECODER_BLOB_LEN` | 162,127 | 90.9% | HNeRV decoder weights (FP11-recoded brotli streams) |
  | `LATENT_BLOB_LEN` | 15,387 | 8.6% | per-pair latents |
  | sidecar_blob (remainder) | 879 | 0.5% | **K=16 selector indices + DQS1 tail + corrections** |
  Reproduce: `grep -nE "DECODER_BLOB_LEN|LATENT_BLOB_LEN" <submission>/src/codec.py`;
  sidecar = 178393 − 162127 − 15387 = 879.
- **Provenance chain:** FP11 recode of source FECa packet
  `…/feca_selector_reparameterized_runtime_stable_20260527Tlocal/submission_dir/archive.zip`
  (sha `18e3155f…`, 178,530 bytes, **present on `/Volumes/VertigoDataTier`**), net −37 bytes.
  File: `<submission>/archive_manifest.json` → `source_archive` + `serialized_archive_delta`
  (`archive_delta_bytes=-37`, `status="realized_saving"`).
- **inflate.sh contract:** numpy + torch CPU/CUDA-agnostic, single-member `x` → `.raw` frames; scorer-free
  at inflate. `<submission>/inflate.sh:1-40`.

**The PR110++ attack surface on this archive is the 879-byte sidecar (0.5% of bytes).** The K=16 per-pair
selector picks one of 16 deterministic frame-0 transforms per pair (identity / luma-bias / RGB-bias /
blue-chroma-amp / 1-px roll), entropy-coded. The decoder (91%) and latents (8.6%) are the HNeRV carrier,
not selector territory.

---

## 2. Five-link chain status per PR110++ surface

Five-link chain (per `docs/vehicle_operating_system.md` + strategic memo §"type system"):
`name → mechanism → gradient/effect → authority`. "Measured exact `ActionEffect`" = a
`tac.action_effect.v1` row with real `old/new_d_seg`, `old/new_d_pose`, byte endpoints, parseback +
inflate survival, under a `contest_cpu`/`contest_cuda` authority. All sources below were searched for such
a row.

| Surface (file) | name | mechanism present? | gradient/effect | authority | measured exact `ActionEffect`? |
|---|---|---|---|---|---|
| `src/tac/analysis/action_effect.py` (`ActionEffect` v1 + `EvaluatorActionEffect`) | the **currency** of the chain | YES — full typed schema, `compute_delta_scores` routes the one nonlinear formula via `tac.score_geometry.contest_score`; admission needs receiver+scorer motion + fakequant + parseback + byte-pricing | n/a (it is the ledger row type) | `promotion_eligible` structurally pinned False | n/a — it is the schema, not a producer |
| `src/tac/analysis/pr110_baseline_reproduction.py` | PR110 global-K=16 baseline reproduction PROOF | YES — fail-closed validator; `build_…_from_action_effects` mints the K=16 proof only from a real selector-replay row (600 pairs, exact score endpoints, byte endpoints, one authority) | emits `MENU_ILP_BASELINE_BLOCKER` until proof passes | gated; nothing minted | **NO** — no `pr110_global_k16_baseline_reproduction.v1` artifact exists anywhere (`grep -rl` over `.omx`,`experiments`,`reports` = 0 hits) |
| `tools/convert_pr110_k16_packet_to_action_effect.py` | K=16 packet → ActionEffect converter | YES — requires a packet manifest with `proxy.{baseline,selected}_avg_{segnet,posenet}_dist` + byte endpoints + `selector_pack.{palette_size=16, selector_code_bits_total, selector_payload_sha256}` | would mint the K=16 selector-program row | `[contest-CPU] pr110_fec6_k16_packet_manifest` (proxy-authority unless real proxy block) | **NO row produced** — no PR110/FEC6 K=16 *packet manifest* with a `proxy` block was found to feed it |
| `src/tac/substrates/pr110_opt11_multi_mode_per_pair_composition/` | multi-mode-per-pair (2 frame-0 perturbations / pair) | L0 SCAFFOLD — grammar + builder + inflate + smoke present; `research_only=true` | analytical upper bound ΔS −0.00052 (sub-additive caveat per Catalog #373); smoke applies 2 modes | `[macOS-CPU advisory]`; canonical eq `FORMALIZATION_PENDING` | **NO** — predicted-only, no L1 anchor |
| `src/tac/substrates/pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1/` | UNIWARD inverse-scorer basis (Yousfi-T1) | L1 IMPL_COMPLETE — binds 5 landed primitives; archive grammar + inflate present | declared dispatch-ready; no exact eval landed | `[macOS-CPU advisory]`; "paired-CUDA RATIFICATION pending" | **NO** — L1 = mechanism, not exact-scored |
| `src/tac/substrates/boost_nerv_pr110_residual/` | NeRV residual boosting on PR110 base | architecture + archive + curriculum + sign-bitmap codec present | residual-extraction mechanism; no exact eval | `[macOS-CPU advisory]` | **NO** |
| `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/` (opt1 widened-top10, opt12 posenet-null, opt13 tier-split) | per-region menus / PoseNet-null subsets / tier-split menus | menus computed analytically from MPS per-pair rows | `aggregate_advantage_vs_unified=0.0` (opt13); mean component ΔS ≈ −0.0069 on n=2 "hard tier" pairs | `[macOS-CPU advisory]` | **NO** — every artifact carries the same 5 blockers incl. `no_byte_closed_archive_candidate`, `selector_bytes_not_charged` |
| `…/final_rate_repair_b7106_z8_priority_loop…20260531Tlocal/` (5 stages against the LIVE frontier) | segnet-region-waterfill / posenet-null-bottom-decile / frame0-k16-palette-asymmetry / per-region-selector-codec / entropy-boundary | YES — actually ran byte-transform executor on the frontier archive bytes | **all 5 stages: `charged_bits_changed=False`, candidate sha == frontier sha, terminal `archive_bound_noop_candidates_refused`** | `[macOS-CPU advisory]` | **NO** — no charged-byte change ⇒ no admissible row |

**The fleet has a complete, rigorous `ActionEffect` currency + a fail-closed K=16 baseline gate, but ZERO
measured exact `ActionEffect` rows.** Every PR110++ optimization lever is at link-3 (analytical/MPS effect)
or earlier. The K=16 baseline reproduction proof — the structural precondition for any menu/per-region
escalation per `pr110_baseline_reproduction.py:MENU_ILP_BASELINE_BLOCKER` — **does not exist**.

### The shared root cause (one finding, not eight)

All PR110++ "untried lever" artifacts derive from ONE source:
`experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl`
(600 pairs × 21 modes). That source is **MPS research-signal**:
`…/mps_research_signal_manifest.json → evidence_grade="MPS-research-signal"`, `score_claim=False`.
Per CLAUDE.md "MPS auth eval is NOISE" (PoseNet drift 23×), these rows cannot mint score authority. The
per-pair `posenet_dist` values in the rows are O(1e-5) — exactly the operating point where the √(10·d_pose)
term's marginal sensitivity is largest, so MPS pose noise is maximally corrupting here. **The entire
PR110++ menu cluster is built on a measurement substrate that cannot, by construction, produce an exact
row.** This is the binding gap — not the menu math.

---

## 3. Attachment verdict (the core question)

**VERDICT: ATTACHMENT IS LEGAL AND ALREADY-PRESENT — but the win surface is structurally tiny and
currently no-op-blocked; the binding gap is EXACT MEASUREMENT, not grammar compatibility.**

Three sub-findings, each provenance-cited:

1. **Grammar compatibility: TRUE, trivially — the selector is already in the archive.** The frontier
   `inflate.py` already runs `apply_selector_to_frames(…, unpack_selector_indices(sidecar))` over the
   K=16 `PALETTE_MODE_IDS`. PR110++ atoms (per-region menus, EV-weighted selection, both-frame composites,
   PoseNet-null subsets, multi-mode-per-pair) are all re-parameterizations of *that same 879-byte sidecar*.
   No foreign-archive grammar bridge is needed. (`<submission>/inflate.py:30,151-157`,
   `<submission>/src/frame_selector.py:17,90,182-200`.)

2. **The win surface is ~879 bytes (0.5%) and the selector is at the entropy floor.** The frontier already
   selected K=16 per pair; the FP11 recode left `selector_payload_unchanged=True`. The repair campaign's
   `per_region_selector_codec` stage tried to re-code the selector region and produced a candidate with
   **identical sha** (`charged_bits_changed=False`, `archive_bound_candidate_material_change_not_proven`).
   Rate-axis wins from re-coding the existing selector are exhausted to the byte. ANY further PR110++ rate
   win must come from a **different selection** (better per-pair modes → lower d_seg/d_pose at equal-or-
   fewer selector bits), i.e. a *distortion* win, not a *coding* win. That requires exact d_seg/d_pose per
   candidate mode — which only exact `evaluate.py` provides.
   (`…/stage_004_per_region_selector_codec…/stage_execution_report.json`.)

3. **The FP11 wrapper does NOT block sub-section editing.** Member `x` is a stored (uncompressed) ZIP
   member whose internal layout is `[decoder | latent | sidecar]` at fixed offsets; FP11 only recompressed
   the decoder brotli streams. The 879-byte sidecar is directly addressable at
   `archive_bytes[162127+15387:]`. So a new selector CAN be written into the frontier archive losslessly
   w.r.t. decoder + latents. The blocker is not "can we write it" — it is "we have no exact d_seg/d_pose to
   know WHICH selection is better," because the only per-mode effect data is MPS noise.

**Net:** V5 PR110++ is the correct frontier-direct layer (it already owns the frontier's selector), but it
is starved exactly as the strategic memo's fleet table says V5 is — "needs … exact ActionEffect rows."
The frontier-direct path does NOT need to wait for SNeRV/HiNeRV to cross link-5 (the strategic memo's
framing); it needs **one exact paired-CPU re-eval of one alternative selector on the frontier base**.

---

## 4. The single cheapest $0-local falsifiable next experiment

The cheapest experiment that produces ONE exact `ActionEffect` row **cannot** be fully $0-local, because
the contest law is `upstream/evaluate.py` and the only contest-faithful authority is CPU(Linux-x86_64) /
CUDA — local macOS is advisory per CLAUDE.md. So the honest answer is a **typed blocker on the exact axis**
plus the smallest $0-local artifact that makes the eventual exact row a single deterministic call.

### Ranked next experiment

**EXPERIMENT R1 (rank 1): "Frontier selector-swap exact `ActionEffect`" — build the byte-closed candidate
$0-local, then one exact CPU re-eval.**

- **$0-local half (does NOT need GPU/contest hardware):** Take the LIVE frontier archive bytes. Decode the
  879-byte sidecar selector with `unpack_selector_indices`. Produce ONE alternative selector (the cheapest
  credible improvement: re-pick each pair's mode by argmin of an *exact-decodable* per-mode proxy that is
  NOT MPS — e.g. recompute d_seg as true SegNet-argmax-disagreement on the frame1 the receiver actually
  renders, locally, as a *candidate generator only*). Re-encode the selector, splice it back at offset
  `162127+15387`, repack the single member `x`, and emit `candidate_archive.zip`. Prove
  `charged_bits_changed=True` + receiver consumption (decoder/latent byte-identical; only sidecar changed)
  via the existing no-op detector. This is byte-closed, deterministic, and free.
- **Exact half (the one paid/contest-hardware call, ~minutes CPU):** run `upstream/evaluate.py --device cpu`
  on BOTH the frontier archive and the candidate on Linux-x86_64 (the contest-CPU axis the 0.19199 lives
  on), and mint ONE `tac.action_effect.v1` row via `convert_pr110_k16_packet_to_action_effect.py` (build a
  packet manifest whose `proxy` block carries the EXACT, not MPS, `avg_segnet_dist`/`avg_posenet_dist` from
  the two evals).

- **Falsifiable prediction:** If the alternative selector improves exact d_seg/d_pose at equal selector
  bytes, `delta_score_total < 0` and the K=16 baseline-reproduction → menu-ILP gate unlocks. **Prediction
  with explicit kill criterion:** I predict the FIRST naive selector-swap will land `delta_score_total ≥ 0`
  (no improvement) — because (a) the frontier selector was already chosen by the FECa reparameterization,
  and (b) the only per-mode signal we have is MPS-noisy, so a locally-recomputed pick is unlikely to beat
  the incumbent on the exact axis. A `delta_score_total ≥ +0.0005` result FALSIFIES "the existing K=16
  selector is leaving exact-axis distortion on the table" and re-routes effort to the decoder (91% of bytes)
  or to producing a non-MPS per-mode effect table first. A `delta_score_total < 0` result is the first
  exact PR110++ frontier win and unlocks the entire menu/per-region cluster.

- **Why this over the alternatives:** opt11 multi-mode and opt7 UNIWARD both ADD selector bytes (rate cost)
  on top of an unproven distortion win — strictly dominated as a first measurement. The repair-campaign
  rate stages are exhausted (no-op). R1 is the minimal change (sidecar-only, equal bytes) that isolates
  the one unmeasured variable (does a different exact-axis selection beat the incumbent?) and produces the
  one row the entire fleet is starved for.

**EXPERIMENT R2 (rank 2, the $0-local-ONLY fallback if no contest-CPU hardware is available this cycle):**
Build the **non-MPS per-mode effect table** — recompute, locally, true SegNet-argmax-disagreement d_seg
(deterministic, not drift-prone like PoseNet) for each of the 16 modes on a handful of frame1 receiver
renders from the frontier. This replaces the MPS `pair_component_rows.jsonl` substrate with a partially-
trustworthy local table (d_seg argmax is far less MPS-sensitive than pose; CLAUDE.md: SegNet drift ~2×
vs PoseNet 23×). Output: a typed candidate-generator table for R1 — NOT a score row. Falsifiable: if the
local-d_seg mode ranking disagrees with the MPS ranking, it proves the MPS substrate was mis-ranking modes
(the root cause), independent of any exact eval.

---

## 5. The biggest blocker

**`pr110_k16_baseline_reproduction_missing` → no measured exact `ActionEffect` row exists on the frontier.**
The fail-closed gate `tac.analysis.pr110_baseline_reproduction.MENU_ILP_BASELINE_BLOCKER` correctly refuses
ALL menu/per-region/EV-weighted/multi-mode escalation until a real 600-pair selector-replay row with exact
score+byte endpoints under one authority reproduces the K=16 baseline. That row has never been minted,
because the only per-mode effect substrate the whole PR110++ cluster consumes is **MPS research-signal**
(`frame_exploit_…_mps600_codex/pair_component_rows.jsonl`, `evidence_grade="MPS-research-signal"`), which
by CLAUDE.md cannot produce score authority. Every PR110++ tool, substrate, and artifact is downstream of
this one starvation. The exact axis (`upstream/evaluate.py` on contest-compliant CPU) is the unavoidable
unblocker — it cannot be substituted by any local-macOS computation.

---

## 6. Manifest-style summary (machine-readable verdict)

```
attachment_legal: true
attachment_already_present: true            # frontier IS a PR110 K=16 selector archive
attachment_win_surface_bytes: 879           # sidecar selector; 0.5% of 178393
selector_rate_floor_reached: true           # per_region_selector recode = no-op (sha-identical)
remaining_pr110pp_win_class: distortion     # not coding; needs exact d_seg/d_pose per mode
measured_exact_action_effect_rows: 0
k16_baseline_reproduction_proof_exists: false
binding_gap: exact_measurement_not_grammar
per_mode_effect_substrate_authority: MPS-research-signal   # non-promotable noise
frontier_archive_bytes_locally_available: true
feca_source_archive_locally_available: true   # /Volumes/VertigoDataTier
next_experiment_rank1: frontier_selector_swap_exact_action_effect  # $0 build + 1 contest-CPU eval
next_experiment_rank1_falsifiable_prediction: delta_score_total >= 0 (naive swap fails); <0 unlocks menu-ILP
next_experiment_rank2_zero_dollar_only: local_nonMPS_dseg_per_mode_table
biggest_blocker: no_exact_action_effect_row__pr110pp_cluster_starved_on_MPS_substrate
axis_tag: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
```

---

## Wire-in / orphan-signal note (Catalog #125)

This is a research memo (the strategic memo's "evidence base that POPULATES a manifest"); per
`docs/vehicle_operating_system.md` subagent contract, the manifest-producing follow-up is EXPERIMENT R1's
byte-closed candidate + exact row. Hooks: #6 probe-disambiguator = R1/R2 (the two interpretations of "is
the existing selector optimal on the exact axis"); #2 Pareto = the 879-byte selector is the only PR110++
rate lever and it is at floor; #5 continual-learning = the one R1 exact row reseeds the V3 ΔS-judge and
recalibrates `pr110_opt11_multi_mode_per_pair_composition_savings_v1` (currently FORMALIZATION_PENDING).
Hooks #1/#3/#4 N/A until a real row lands.
