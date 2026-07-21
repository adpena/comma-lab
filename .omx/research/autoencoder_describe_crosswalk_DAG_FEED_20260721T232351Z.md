---
schema: pact.dag_feed.autoencoder_describe_crosswalk.v1
feed_id: FEED-AUTOENCODER-DESCRIBE-20260721T232351Z
utc: 2026-07-21T23:23:51Z
lane_id: lane_autoencoder_describe_crosswalk_20260721T231126Z
research_only: true
execution_authority: false
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
pointer_delta: 0
axis: "[research-only; exact description-stream byte custody plus source/repository audit]"
verdict: "CONTROL_ROWS_ADOPTED; NO_AUTOENCODER_PROMOTED; FAMILIES_OPEN; U4_ROUTED"
main_landing_review_required: true
---

# DAG FEED — counted stream autoencoders for v10 descriptions

## Decision graph

```text
live real description object, seed=1234, decision requires n600
  |
  +--> S1 PNTG: 6,791 B exact
  +--> S2 literal components: 180,196 B raw -> 152,160 B zlib9 exact
  +--> PPCS B2: 884,872 B raw -> 78,969 B zlib9 exact
  `--> PDW2: 138 B raw -> 111 B zlib9 exact (beats 133 B Brotli-q11)
          |
          v
best same-object direct control (#557/#558 race)
          |
          v
candidate typed stream autoencoder
  counted = decoder weights + latent/residual + codebook/probability state
            + normalization + section/parser/container overhead
          |
          +--> exact stream reconstruction
          |       `--> bit-exact parseback gate
          |
          `--> lossy stream reconstruction
                  `--> real RGB receiver -> frozen scorer -> Seg/Pose decomposition
          |
          v
net_delta = counted candidate bytes - best exact control bytes
          |
          +--> net_delta < 0 and all protected facets green
          |       `--> direct_description_minimizer_builder -> receiver-closed archive build
          |
          `--> otherwise N-A-WHY with named exit measurement

external-corpus shared decoder candidate
          |
          v
#604 U4 strict-vs-generic boundary ruling
  +--> strict: shared decoder weights count
  `--> alternative: only U4 may authorize fixed external prior as free;
                    contest latent/adaptation/state always count
          |
          `--> until ruling: research_only, no archive promotion

#574 xi temporal predictor
  `--> publishes residual receipt -> later AE consumes residual; never re-model predictor
```

## Ranked routed edges

| Priority | Producer | Consumer | Payload | Admission/falsifier |
|---:|---|---|---|---|
| 1 | This receipt | `#557 coder rung` | S2 zlib-9 152,160 B; PDW2 zlib-9 111 B; exact hashes and bit-exact replay | Consumer must reproduce on identical bytes. No archive marginal is claimed. |
| 2 | Future bounded S2 probe | `direct_description_minimizer_builder` | Counted shared decoder plus typed S2 latent/residual | Total below 152,160 B; exact parseback, or real through-R score decomposition if lossy. |
| 3 | This A2 matrix | `#604 U4 / einstein_kolmogorov` | COIN++, VC-INR, external VAE/foundation decoder, pre-fitted entropy prior under two byte readings | U4 ruling is required; absent ruling, free-decoder edge refuses promotion. |
| 4 | `xi_temporal_delta_coder_574` | future stream AE/coder | Custodied residual object | Receipt absent at poll; consumer must wait and must not fit a duplicate predictor. |
| 5 | Future measured AE candidate | `canonical_equations` | `decoder_counted_bytes + stream_bytes_after < best_exact_control_bytes` with evaluator | No registration until a parser/receiver-bound candidate exists. |

## Verdict scopes

| Negative | Scope | What remains open |
|---|---|---|
| Current pose MLP raw-fp16 representation exceeds PNTG. | `INSTANCE x CURRENT-POSE-MLP` | Smaller model, compressed weights, shared decoder, better formulation. |
| No S2 autoencoder candidate receipt exists. | `FORMULATION x CURRENT-S2-LITERAL-PACKET` | Typed shared stream AEs and nonlinear transforms. |
| No PPCS autoencoder receipt exists. | `FORMULATION x CURRENT-PPCS-B2-LOOSE` | Section-aware exact or scorer-measured lossy codec. |
| Standalone PDW2 has a 111 B exact control. | `FORMULATION x STANDALONE-PDW2` | Amortized already-counted shared decoder with RGB receiver. |
| #574 input receipt is absent. | `INSTANCE x INPUT-CUSTODY-ABSENT` | AE over the eventual #574 residual stream. |
| DeepCABAC measured configuration loses 1,436 B. | `INSTANCE x MEASURED-MACOS-CONFIGURATION` | Other transforms/priors with exact receiver custody. |
| Generic pretrained decoder status is unresolved. | `RULE-BOUNDARY x U4-PENDING` | Both technical readings remain open; free-reading work is research only. |

No negative closes an autoencoder family or paradigm.

## Triality

### DSL

No Lever, flag, trainer, or curriculum mutation. A future codec must be a typed parser-consumed
description section, save its full counted state, and preserve additive resume compatibility.

### DAG

This file is the standalone FEED. `autopilot_action=NONE`; no provider dispatch or long job. The
only active edge is ingestion of the two exact lossless control rows. Candidate measurement and U4
adjudication remain explicit owner-routed edges.

### Equations

- `lambda_B = 25 / 37,545,489 score/B`.
- `net_delta_bytes = decoder_counted_bytes + stream_bytes_after - best_exact_control_bytes`.
- Candidate admission requires `net_delta_bytes < 0`; lossy candidates additionally require total
  `Delta S = 100*(d_seg_candidate-d_seg_control) + sqrt(10*d_pose_candidate) - sqrt(10*d_pose_control) + lambda_B*net_delta_bytes < 0`
  on the same receiver/evaluator cells with protected facets green.
- `law_selection=NO_LAW_REGISTERED`; inputs are unmeasured for every AE candidate.

## Six-hook wire-in

1. **Sensitivity map:** reuse current per-class/stratum signals; no new constant or posterior.
2. **Pareto:** exact total counted bytes; if lossy, require the same-object Seg/Pose/rate tuple.
3. **Bit allocator:** use the strict break-even inequality as a preflight refusal; optimistic zero-
   payload parameter ceilings are not allocation targets.
4. **Cathedral/autopilot:** no actuation. Future S2 probe and U4 ruling are named but not dispatched.
5. **Continual learning:** ingest only the exact zlib control rows; no AE efficacy row exists.
6. **Probe disambiguator:** #604 U4 owns the strict-versus-generic pretrained-decoder fork.

## Authority boundary

Research only; $0. No training, scorer run, archive mutation, provider dispatch, equation
registration, score promotion, or pointer movement. MAIN must review the complete branch diff
before landing.
