# ddm_gb2 — gb1's owed object: an UNCONDITIONAL lower bound on the counted bytes of a generated carrier basis on the lattice (not the span), or a proof that no non-vacuous bound exists at this scope — the design gate gb1 blocked on, done closed-form (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, xhigh) · Spawned by MAIN 2026-09-09 (operator: "Codex should be available too"). Sources: gb1 (`.omx/research/ddm_gb1_generated_carrier_basis_lattice_priced_closed_form_20260909.md`, `7e27eb732`: BLOCKED honestly — a span projection is a bound for the affine problem alone and adding the observed rounding penalty does not make it a lattice bound; seed-regenerated SVD atoms carry fitted information; six-pair Jacobians are not n600), pc2 (memory `carrier_bytes_buy_a_lattice_point_rank_cut_free_in_span_costly_on_lattice_20260909`: rank cut free in span, 4–2,733× on the lattice; the shipped carrier codes a `refine_pair` fixed point 17/7,200), pc1 (`pose_carrier_buys_a_positioned_subspace_lattice_coarsening_with_resolve_pays_20260905`: basis edits fail 39,748×), qbw2 (`explicit-boundary-floor-equals-gb1-archive-generate-dont-serialize`), gs3 Addendum 18 §IV item 5 (`ae6a0b837`). Axes: derived `[closed-form, constants sourced]`; `score_claim=false`.

## MANDATE
The carrier section is ~X B of the archive (VERIFY at source from the cmp2 seal's section table) and every basis-level actuator has failed on the lattice. gb1 asked whether a GENERATED basis (seed + few parameters, free in inflate.py per rule 118) could replace the stored basis at lower counted bytes and could not price it because its bound construction was defective. Build the valid bound: for a carrier that must reproduce the shipped per-pair pose coefficients to within the pose budget (≤ 1.25e-4 S, memory m110) on the INTEGER lattice the shipped codes live on, derive a lower bound on counted bytes that is (a) unconditional on the lattice (a closest-vector / covering-radius argument or a counting bound over the lattice's Voronoi cells — not a span projection), (b) built from n600 quantities the shipped tree exposes (the 600 coefficient vectors, the scales, the codes), and (c) explicit about which information a seed-generated atom may carry for free (a seed is ≤ 8 B; anything fitted is counted). Then evaluate it for the three families gb1 listed (K = the same as gb1's table) and state whether ANY family can beat the stored basis by ≥ 300 B at the pose budget.

## PRIOR-LAW PREDICTION (m38)
- Prediction: the unconditional lattice bound for every generated family lands ABOVE the stored basis's counted bytes minus 300 B, because the lattice point the shipped carrier codes is a fixed point of `refine_pair` (17/7,200) and pc2 measured that leaving it costs 4–2,733× — the information in the basis is positional (pc1), not spectral, and a seed cannot carry position.
- **FALSIFIER:** a family whose valid bound is ≥ 300 B under the stored basis at ≤ 1.25e-4 S pose — then gb1's build is un-blocked and this charter writes its BUILD charter (do not build here). If the bound is vacuous for every construction attempted, say so with the reason each construction fails, and close the generated-basis door at formulation scope.

## SCOPE
Closed-form only; every constant read from the cmp2 seal / archive by the tree's own reader; no training, no search, no archive build. ≤ 2 procs, no Metal.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; no writes to any live tree; your dir `/Volumes/VertigoDataTier/pact/ddm_gb2_generated_basis_bound/` (small; `df -h` first — both SSDs are near full, vr4 is reclaiming).
- No fitted-model "lower bound" (gb1's defect); no −log2 p averages (m166); every bound carries its derivation and constants' source lines.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes; tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_gb2`.

## PRIOR NEGATIVE SIGNAL
- gb1's three defects (span ≠ lattice; seeded atoms carry fitted info; 6-pair Jacobians ≠ n600) — each must be named as avoided in the construction.
- pc1 ×39,748 basis edits; pc2 lattice step ×1/2 fails paired; ra3 carrier closed (price the CEILING first, m118) — this charter IS the ceiling price.

## OPTIMAL FORM
- Reference form: tc1's bound memo (`.omx/research/ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md`, `faac73963`) — a bound from real n600 counts with a registered equation; gb1 (`7e27eb732`) as the honest-block reference. SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no subset-pair Jacobians, no span bounds.
- **PRIOR-LAW PREDICTION (falsifiable):** as above. FALSIFIER: any family ≥ 300 B under the stored basis at the pose budget.

## DELIVERABLE
Memo `.omx/research/ddm_gb2_generated_basis_unconditional_lattice_bound_20260909.md` with the derivation, the family table, the verdict; register `generated_carrier_basis_lattice_floor_v1` via `register_canonical_equation` (`tac.canonical_equations`); lane `lane_ddm_gb2_generated_basis_bound_20260909`. Commit via the serializer. End with the live frontier line.
