# Codex findings — DDM M6 close the 22,645-byte gap

UTC: 2026-07-23  
Lane: `ddm_m6_close_22645_byte_gap`  
Authority: delegated bounded build/audit; `$0`; no launch; no scorer run; no config
change; `research_only=true`; `score_claim=false`; pointer unchanged; MAIN landing
review required.

## Decisive result

Pool-aware composed reduction is **Y = 13 bytes**, taking the exact #575 archive from
177,169 B to a deterministically derivable 177,156 B compact-framing form. The exact
settled-C1 sub-0.15 cap remains 154,524 B, so the residual is:

`177,156 - 154,524 = 22,632 bytes`.

`SUB015_NOT_REACHED_Y13_RESIDUAL22632`.

No byte-close submission candidate and no R6 exact-eval request were flagged because
`13 < 22,645`. The 13-byte result is a rate-only structural identity proof, not a new
score row. At settled C1 it changes the counterfactual score from
`0.1650779449085630846...` to `0.1650692887421724964...`, still above 0.15.

## Per-lever attribution

| Lever | Pool | Admitted bytes | Authority and scope |
|---|---|---:|---|
| ker(A) payload hiding | `P_NULL_GAUGE` | **0** | MEASURED ZERO. The 80.6742315223% nullity is geometric freedom; no parser-consumed counted payload was removed. |
| Rule-118 free migration | `P_TEMPORAL_DESCRIPTION` | **13** | MEASURED transitive exact receiver parse-back. Only fixed/derived wrapper facts move to generic receiver code. |
| g2g2 integer-lattice native | `P_REALIZE` | **0** | MEASURED ZERO / premise conflation. g2g2 already closes `factor2_uint8_exact`; the cited -1.4% belongs to a different n16 absolute-write formulation and is not byte savings. |

The final same-artifact archive delta is 13 B. It is not the naive sum of nullity,
scheduled score debt, or singleton description estimates.

## Rule-118 receiver proof

The exact archive is a single ZIP_STORED member:

```text
ZIP overhead                         100 B
FP11 magic                            4 B  -> generic code
source length                         4 B  -> derived from retained lengths
CTXR magic                            4 B  -> generic code
CTXR fixed version                    1 B  -> generic code
three section lengths                 9 B  -> counted/retained boundary data
decoder section                 161,104 B  -> counted video-fit data
latent section                   15,070 B  -> counted video-fit data
sidecar                             607 B  -> counted video-derived correction data
selector length                       2 B  -> counted/retained boundary data
selector                            222 B  -> counted video-derived data
DQS1 tail                            42 B  -> counted video-derived data
```

The compact receiver contract is:

```text
3*u24 lengths | decoder | latents | sidecar | u16 selector length |
selector | DQS1 tail
```

`tac.packet_compiler.ddm_m6_implicit_fp11` reconstructs the legacy member and
canonical ZIP container. On the exact source bytes:

- source archive: 177,169 B,
  SHA-256 `cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f`;
- compact archive: 177,156 B,
  SHA-256 `3fb7fe6d9e37e5545d9b8757514d3b82ff6c2296b474bf29026ac3875128ff3f`;
- reconstructed archive is byte-identical to the source, including ZIP metadata;
- mutation tests independently alter decoder, latent, sidecar, selector, and DQS1
  fields and prove each retained field changes only its corresponding reconstructed
  receiver input.

This transitive identity is why a scorer rerun is unnecessary for the 13-byte
structural fact: the existing receiver receives the exact original archive bytes
after the generic adapter. No submission runtime was staged and no compact archive
was written as a candidate because the gap is not closed.

The remaining 11 framing bytes, all five semantic sections, and the 100-byte ZIP
overhead receive no speculative free credit. In particular, the selector length and
three CTXR section lengths were retained rather than inferring boundaries by sentinel
search or by hardcoding video-specific sizes.

## g2g2 premise falsification

The durable g2g2 receipt is hash-bound at:

`/Volumes/VertigoDataTier/pact/evidence/g2g2_joint_multichart_20260721/measurement_20260721T172244Z/receipt.json`

File SHA-256:
`928d3cd74cc92ef52aa9f821229ada12fbf4c3e9dad772e8a76adffcfcfcb078`.

It contains 13 measured prefixes over six pairs. Every prefix already satisfies
`counted_bytes`, `receiver_RGB`, `factor2_uint8_exact`, and `double_decode`. Yet 0/6
pairs are admitted because semantic exactness, pose tube, and rate gates fail. Its
base counted vehicle is 121,128 B, not the 177,169-byte FP11 vehicle.

Therefore “make g2g2 lattice native to recover the -1.4% uint8 loss” joins two
different facts:

1. g2g2's current exact factor-2 uint8 receiver gate; and
2. M4's n16 source-closest-sign absolute-write result, where scheduled
   `+0.01583 S` realized `-0.00022162 S` (-1.4%).

The second is unmet scheduled score debt (`0.01605162 S`), not a measured byte
quantity, and it cannot be transferred to the first vehicle. This falsifies the
requested byte attribution, not the broader integer-lattice or multichart family.

## Pool-aware law

For each non-additive pool `p`, let `Delta B_p^joint` be its strict joint receiver
credit. Let `B_final` be the one composed artifact after receiver closure.

```text
pool_bound = sum_p Delta B_p^joint
Y = B_baseline - B_final, only if the final artifact is receiver-closed
require 0 <= Y <= pool_bound
residual = max(0, B_baseline - Y - B_cap)
```

Here the pool credits are `0 + 13 + 0 = 13`, the exact final artifact delta is 13,
and the residual is 22,632. The typed law refuses duplicate singleton rows for the
same pool and refuses any final delta larger than its admitted pool bound.

## Re-derivation

```bash
PYTHONPATH=src uv run --frozen --with scipy python \
  tools/derive_ddm_m6_gap_closure.py \
  --verify-receipt \
  .omx/research/ddm_m6_close_22645_byte_gap_20260723_receipt.json
```

Expected:

```json
{"Y_bytes":13,"residual_gap_bytes":22632,"status":"PASS","sub015_reached":false}
```

## Verification

- Deterministic receipt verification: PASS; receipt SHA-256
  `194c6951246cf25bb2fca5a1ec0d429b72161cea90683d7865d3a895c77f5c0e`.
- Focused adapter/equation/authority suite: **8 passed**.
- Ruff, Python compilation, and `git diff --check`: clean.
- Review tracker: three fresh clean passes across all five Python files. The lane is
  honestly L2 with `impl_complete`, `real_archive_empirical`, and
  `three_clean_review` true; contest CPU/CUDA, strict preflight, memory, and deploy
  gates remain false.

## STORES CONSULTED

- Delegated authority file, complete `CLAUDE.md`, complete `AGENTS.md`, `PROGRAM.md`,
  operating handoff, v7.5 operating-contract spec, and v8 decomposition spec.
- M4 findings, deterministic receipt, three-pass review, DAG/FEED, and exact
  re-derivation output.
- Exact #575 archive bytes and ZIP/member anatomy on the SSD evidence tier.
- #580/null-compiler receipt through the hash-bound M4 receipt.
- g2g2 full SSD receipt plus its preserved branch summary.
- #602 MDL-member, #553 gauge-packet, #557 entropy-ceiling, #559 rank-4 head,
  #417 receiver-consumption, #580 ker(A), and #532 uint8-realization laws.
- Both delegated inboxes through each checkpoint.

## HISTORICAL_PROVENANCE

- M4 established the 177,169 B audited relaxed floor, exact-C1 154,524 B cap,
  four-pool non-additivity law, zero ker(A) byte credit, and scoped n16 lattice debt.
- The g2g2 receipt remains `MEASURED_G2G2_RATE_BREAK_EVEN_STOP_FAMILY_OPEN`;
  this lane narrows only the proposed cross-formulation byte transfer.
- No prior score, pointer, promotion, or family verdict is superseded.

## MAIN landing review

MAIN must review the commit and explicitly confirm:

1. the 13 migrated bytes are only fixed magic/version or a derivable length;
2. compact-to-legacy reconstruction is exact at both member and archive level;
3. all video-derived sections and required boundary lengths remain counted;
4. g2g2's exact factor-2 custody makes the n16 -1.4% transfer inadmissible;
5. ker(A) remains zero-byte credit and the three pools were not naively summed; and
6. no candidate, exact eval, score, pointer movement, or promotion was claimed.
