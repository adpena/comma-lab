# SJ1 SkyJEPA Crosswalk Receipt - 2026-08-05

## RECALL EVIDENCE

| Scope | Evidence | Impact |
|---|---|---|
| Run contract | `.omx/tmp/codex_runs/_common_contract.md`, `.omx/tmp/codex_runs/sj1_prompt.md` | Bound this unit to $0, scorer-free, online-authority paper intake; no `evaluate.py`, no n600, no frozen-scorer forward, no launch, no protected-file edits, serializer commit required. |
| Governing state | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Pointer honesty and no-fake boundaries apply. Live own-vehicle line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved. |
| #485 JEPA-latent prior-work gate | `.omx/research/jepa_latent_surrogate_20260713.md`, `.omx/research/p0_recovery_rate_probes_20260715.md`, `.omx/research/whole_teacher_distilled_student_20260713.md`, `.omx/research/surrogate_vjp_fidelity_metric_20260714.md`, `src/tac/canonical_equations/segnet_decision_quotient_surrogate_20260713.py` | Blocks blanket JEPA adoption. The settled Pact target is the centered-logit decision quotient, and any useful surrogate/prober must pass value-plus-Jacobian/VJP or Sobolev gates. Value-only latent prediction is not sufficient. |
| #941 / transport / generator-coordinate lines | `.omx/research/ddm_xi1_carried_xi_inter_race_20260729.md`, `.omx/research/ddm_sg3_counted_gt_granularity_ladder_20260804.md`, `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md`, `.omx/research/ddm_od5_20260805/OD5_GENERATOR_PACKET_RECEIPT.md` | Existing evidence already separates pixel/xi transport instances from the still-open generator/worldsheet coordinate packet family. SkyJEPA's useful transfer is the latent-rollout/no-reconstruction argument for that open family, not a new score. |
| Sampling-control consumers | `.omx/research/mc_finisher_396_design_20260710.md`, `.omx/research/mc_finisher_400_energy_ranked_topk_design_20260720T154953Z.md`, `.omx/research/curriculum_candidate_pool_p0_20260710.md`, `.omx/research/erm_energy_guided_recursive_model_route_to_mc_finisher_DAG_FEED_20260714T150000Z.md` | Pact already has the "generate K candidates, rank/select under exact authority" route. SkyJEPA's MPPI control confirms the shape but does not authorize a duplicate sampler or a scorer run. |
| Automated-data / domain-randomization consumers | `.omx/research/synthetic_data_for_costate_organ_supercharge_20260711.md`, `.omx/research/synthetic_data_nvidia_sota_organ_434_20260711.md`, `.omx/research/transient_forge_434_build_20260711.md` | The closest Pact analog is #434 trajectory/costate data, not image/video data. That engine is built and measured as an honest negative on the plateau trajectory; domain randomization is already embodied and remains gated by real walk-forward skill. |
| CPWL / two-chart adjacency | `.omx/research/ddm_sd1_20260805/SD1_CROSSWALK_RECEIPT.md` plus existing tropical/boundary grammar surfaces named there | Any Balestriero/latent regularity adjacency folds into the existing CPWL/two-chart packet grammar. No new canonical equation is justified here. |
| Corpus search sources | `rg -l` passes over `.omx/research`, `.omx/state`, and `src/tac` for `#485`, `JEPA`, `latent surrogate`, `centered-logit`, `Sobolev`, `VJP`, `#941`, `generator-coordinate`, `worldsheet`, `transport`, `#396`, `#400`, `#319`, `#434`, `synthetic`, `CPWL`, and `SkyJEPA` | Bounded absence: in the searched local scope I did not find a prior SJ1 receipt or a receiver-closed SkyJEPA-specific packet. This is not a global nonexistence claim. |

Beyond-charter finding: #485 is the decisive safety gate. It makes "latent prediction is better" too weak unless the Pact consumer is explicitly a decision-quotient/value-plus-VJP surrogate or a generator-coordinate packet whose receiver bytes can be checked without scorer authority.

Plan impact: SJ1 should route one design-only update to #941/OD5-style generator-coordinate transport, fold the prober and MPPI pieces into already-built consumers, and refuse any claim that SkyJEPA itself moves d_seg, d_pose, bytes, or the contest pointer.

## Answer First

SJ1 yields one narrow `ADOPT`: use SkyJEPA as an argument for generator-coordinate latent transport that avoids full future observation reconstruction and recursive pixel-state drift. The immediate consumer is the #941/OD5/BF1 open packet family, and the first next artifact must be scorer-free: exact bytes, parse-back, double-decode, receiver equality, and a baseline race against existing OD5/BF1/Xi transport packets.

Everything else is already embodied or not applicable. The physics prober maps to #485's centered-logit/Sobolev/VJP-gated surrogate discipline and existing pose/transport state readers. The sampling-based controller maps to #396/#400/#319 candidate selection. The domain-randomized data pipeline maps to #434 Transient Forge, which is already built and honestly negative on the plateau trajectory. No scorer was run, no candidate was built, no archive was evaluated, and no frontier promotion is claimed.

## Paper Custody

- Paper: arXiv:2606.23444v2, `SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors`.
- Online authorities reached this turn: `https://arxiv.org/abs/2606.23444` and `https://arxiv.org/html/2606.23444`.
- arXiv custody facts: submitted 2026-06-22, revised to v2 on 2026-06-23, subjects `cs.RO` and `cs.LG`, authors Pratyaksh Rao, Wancong Zhang, Randall Balestriero, Yann LeCun, Giuseppe Loianno.
- Method facts used: the paper frames autoregressive observation/state reconstruction as a long-horizon drift source; predicts structured future latents instead; maps frozen latent rollouts to physical state with a physics-inspired prober; uses sampling-based control; and generates domain-randomized closed-loop simulation data.
- Custody limit: no local PDF/source hash was established in this turn. This receipt relies on the live arXiv abstract/HTML plus local Pact recall.

## Per-Seed Verdicts

| Seed | Verdict | Label | Named consumer | Falsifier |
|---|---|---|---|---|
| Latent dynamics instead of autoregressive observation reconstruction | `ADOPT` | `INFERRED / DESIGN_ONLY` from arXiv method plus local transport evidence | #941 generator-coordinate transport; OD5 generator/worldsheet packet child; BF1/PE3 receiver-closed packet families | A receiver-closed generator-coordinate packet fails parse-back/double-decode, loses bytes to existing OD5/BF1/Xi baselines at matched denominator, or needs counted dense video-derived tables. |
| Physics-inspired prober from frozen latents to interpretable state | `ALREADY-EMBODIED` | `RECALLED` | #485 centered-logit quotient surrogate; whole-teacher distilled student; `scorer_targets`, `pose_from_embedding`, `taskspace_predictor_state_v2` style transport readers | A Pact prober over frozen internal latents passes the preregistered centered-logit value-plus-VJP/Sobolev gates and beats the existing quotient route at matched real rows. |
| Sampling-based control over learned rollouts | `ALREADY-EMBODIED` | `RECALLED` | #396 MC finisher, #400 energy-ranked top-k selector, #319 K>1 emission, ERM-routed selector refinements | A SkyJEPA/MPPI-derived candidate sampler measurably improves exact-best recall or exact-call efficiency after the base #396/#400 selector benchmark, with full authority separation. |
| Automated domain-randomized data generation | `N-A` for scorer/candidate work; folded as `ALREADY-EMBODIED` for the organ | `RECALLED / MEASURED_NEGATIVE` | #434 Transient Forge and costate-organ synthetic trajectory data | Tier-2 real witness micro-runs or a transient-rich real trajectory show synthetic + real-prefix training beats persistence and incumbent arms under the existing walk-forward gate. |
| Low-dimensional quadrotor physical-state deployment | `N-A` | `INFERRED` | None for current byte-closed video receiver work | The paper's future RGB/RGB-D extension supplies a concrete visual latent packet that can be mapped to Pact receiver bytes without scorer smuggling; absent that, it stays outside the current vehicle. |
| Balestriero/latent regularity adjacency to CPWL/two-chart decoders | `ALREADY-EMBODIED` | `RECALLED / INFERRED` | SD1 CPWL/two-chart packet grammar; tropical boundary grammar; OD5/PE3 child packets | A new SkyJEPA-specific CPWL packet produces receiver-closed bytes below the existing coherent-boundary corridor without duplicating SD1/tropical machinery. |

No `REFUTED-SEED` verdict is needed. The paper's claims are not locally refuted; they are mostly out-of-domain or already routed.

## Design Consequences

1. The only new work item is a scorer-free `sj1_latent_transport_packet` design/build race against current #941 children.
2. The packet should carry generator/worldsheet or task-latent coordinates, not pixel predictions. Its receipt must state exact packet bytes, denominator, selection mode, parse-back equality, double-decode equality, and projection labels.
3. The paper's prober should not become a new canonical equation. It consumes the existing #485 quotient law and must satisfy the same value-plus-VJP gate if implemented.
4. The MPPI/control result should not duplicate #396/#400. If resumed, it becomes an optional sampler-prior benchmark only after the base exact-selector measurement exists.
5. The automated-data result should not reopen generic synthetic-video or random-domain data. The only legitimate Pact route is through #434-style trajectory/costate data and real walk-forward validation.

## Authority Boundaries

- `MEASURED_THIS_TURN`: online paper metadata/method custody from arXiv; local recall facts from existing Pact artifacts; filesystem artifact creation.
- `NOT_MEASURED_THIS_TURN`: d_seg, d_pose, archive bytes for any new candidate, scorer outputs, n600 behavior, runtime decode, public-wire survival.
- `FORBIDDEN_THIS_TURN_AND_NOT_DONE`: frozen-scorer forward, `upstream/evaluate.py`, n600 job, paid dispatch, long launch, lane claim, protected-file edit, duplicate canonical equation.
- `POINTER_DELTA`: none.

## NEXT_IF_RESUMED

`QUEUED-WITH-FIRE-ORDER`: build `sj1_latent_transport_packet` as a scorer-free receiver artifact after the current OD5/BF1 persistence facts are available. Inputs should be existing generator-context/targeter rows, PE3/BF1-style chart sections, and taskspace transport state. First receipt requirements: exact bytes, SHA-256, parse-back equality, double-decode equality, projection denominator, comparison against Xi1/BF1/OD5 packet baselines, and an explicit refusal if any dense video-derived latent table would be hidden in code.

`FOLDED`: prober work folds into #485 and whole-teacher centered-logit quotient policy. Do not implement a value-only JEPA prober.

`FOLDED`: MPPI sampling folds into #396/#400/#319 after the base selector benchmark. Do not open a new scorer lane for it.

`HELD`: automated data stays with #434 until tier-2 real witness micro-runs or a transient-rich real walk-forward test exists.

## Machine-Readable JSON

```json
{
  "schema": "ddm_sj1_crosswalk_receipt.v1",
  "arm": "SJ1",
  "date": "2026-08-05",
  "paper": {
    "arxiv_id": "2606.23444v2",
    "title": "SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors",
    "sources": [
      "https://arxiv.org/abs/2606.23444",
      "https://arxiv.org/html/2606.23444"
    ],
    "submitted": "2026-06-22",
    "revised": "2026-06-23",
    "local_pdf_hash": null
  },
  "answer": {
    "primary_verdict": "ADOPT_NARROW_DESIGN_ONLY",
    "primary_consumer": "#941 generator-coordinate transport via OD5/BF1-style packet child",
    "score_claim": false,
    "pointer_delta": "none"
  },
  "recall_evidence": [
    "#485 centered-logit decision quotient and Sobolev/VJP gate",
    "#941 Xi transport negative instance plus open generator/worldsheet packet family",
    "#396/#400/#319 exact candidate selection surfaces",
    "#434 Transient Forge built with measured honest negative on plateau trajectory",
    "SD1 CPWL/two-chart packet grammar"
  ],
  "seed_verdicts": [
    {
      "seed": "latent dynamics instead of autoregressive observation reconstruction",
      "verdict": "ADOPT",
      "label": "INFERRED_DESIGN_ONLY",
      "consumer": "#941 / OD5 generator-coordinate transport"
    },
    {
      "seed": "physics-inspired prober",
      "verdict": "ALREADY-EMBODIED",
      "label": "RECALLED",
      "consumer": "#485 centered-logit quotient surrogate"
    },
    {
      "seed": "sampling-based control",
      "verdict": "ALREADY-EMBODIED",
      "label": "RECALLED",
      "consumer": "#396/#400/#319 candidate selection"
    },
    {
      "seed": "automated domain-randomized data",
      "verdict": "N-A",
      "label": "RECALLED_MEASURED_NEGATIVE_FOR_CLOSEST_LOCAL_ANALOG",
      "consumer": "#434 Transient Forge"
    },
    {
      "seed": "low-dimensional quadrotor state deployment",
      "verdict": "N-A",
      "label": "INFERRED",
      "consumer": null
    },
    {
      "seed": "CPWL/two-chart latent regularity adjacency",
      "verdict": "ALREADY-EMBODIED",
      "label": "RECALLED_INFERRED",
      "consumer": "SD1 / tropical boundary grammar"
    }
  ],
  "authority_boundaries": {
    "scorer_forward": false,
    "evaluate_py": false,
    "n600": false,
    "launch": false,
    "new_canonical_equation": false,
    "protected_file_edit": false
  },
  "next_if_resumed": [
    "Build scorer-free sj1_latent_transport_packet and race exact receiver bytes against Xi1/BF1/OD5 baselines.",
    "Fold prober into #485 centered-logit value-plus-VJP gate; refuse value-only JEPA prober.",
    "Fold MPPI sampling into #396/#400/#319 after base selector measurement.",
    "Hold automated data until #434 tier-2 or transient-rich real walk-forward evidence exists."
  ],
  "frontier_line": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved."
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
