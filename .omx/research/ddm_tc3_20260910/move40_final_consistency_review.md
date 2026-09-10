# TC3 move40 final consistency review

Clean bounded read-only audit of completed public proof, selected seal,
consumer validation, final memo and registered handoff. No concrete
inconsistency found. No raw or label-field rehash, receiver/scorer launch,
seal execution, source change, state/index mutation or commit was performed.

The retained child PROCESS equals the public receipt's process, exits 0 in
1232.4976633330007 seconds and names the literal candidate `inflate.sh`.
Its small log rehashes correctly and contains exactly the report carried in
`PUBLIC_IDENTITY.json`. That report has n600, the full current field SHA
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5`,
and candidate/source raw SHA
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`
at 3,662,409,600 bytes. The recorded resume boundary is frame 2. Both raw files
and the full label field still have their recorded sizes; their content hashes
are verified by the completed producer receipt, not independently repeated
by this review. Same-host identity is not promoted to CUDA parity or a score.

The seal, current candidate archive, member `p`, five explicit receiver pins,
public identity and consumer validation agree. The candidate remains
180,154 bytes, SHA
`299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`,
and runtime SHA
`5ff52ed675be24db04a32b83f2097d48881fc7c2959864eef0356f6c6446a4c8`.
Both seal digests recompute under their distinct definitions:

- Physical file SHA: `34df0c69d0adce50a5bdf373a1c669bdd4dee34ef512e3a965734d09391e1fcd`.
- Canonical semantic digest, excluding its own signature field:
  `fcd43ea3a9db6d533f062a82d6c84ec9d530125905c617775eba1e0ebb846f8d`.

The exact-evaluation fire order pins the physical file and separately names
the semantic digest. Both listed fire-order files rehash to their index facts.
Current canonical task rows agree with the memo: TC2 BETTER_PREDICTOR and
SEAL_INTAKE are completed/green/FIRED; TC3 EXACT_EVAL and LANDING are pending,
owned by MAIN and QUEUED-WITH-A-FIRE-ORDER. The memo's two next-action bullets
match their consumers and triggers. No historical recovery or completed
inventory-edit action remains queued. The exact-eval argv explicitly requires
MAIN's real lane claim and pairing/authorized waiver before it can be used.

| Reviewed artifact | SHA-256 |
| --- | --- |
| Final memo snapshot | `123274bc80fa8723c412af0f7643c4011295d9a5031d14e7bf0c7e5ed21fd0f3` |
| PUBLIC_IDENTITY.json | `f0c0e8a9ecebc135cd164c28b6f3ee2df3bab2ce23285623e3b03ab602fc8a94` |
| SEAL_VALIDATION.json | `7528e88e7a30a4b758f7163e6941d048f3dc24abd0b01d0f9fd7d12cbbe14c04` |
| handoff/FIRE_ORDERS.json | `0c642d650741a77734bdc4dcd660ffb764da282e6a7b3c32132177c0a9a80709` |
| handoff/exact_eval/FIRE_ORDER.json | `3991905e323d026a67524b59d1bcba32218e8a3f244d6bc88c7b9197f1a29336` |
| handoff/landing/FIRE_ORDER.json | `e391582d21691e009e4e915044ad1313a5ca80297e4361a5580731a123bab524` |

All receipt paths above are relative to the current move40 store. The final
serializer/landing artifacts are being finalized by the parent and are outside
this audit's certification. This review does not claim a main landing or a
contest result.

## RECALL EVIDENCE

Continued from `move40_measured_comparison_review.md` and
`move40_memo_seal_review.md`. Read the new completed public, validation,
fire-order and latest four scoped canonical task records; inspected
`canonical_seal_bytes` to verify the physical/semantic distinction. The
previous measured arithmetic, formulation limits and inventory exclusion
remain unchanged. No new mechanism or follow-on is introduced.

LIVE-HYPOTHESES: A may earn its 79-byte rate credit on the exact contest axis;
same-host full public output identity supports that possibility, while MAIN's
queued contest measurement remains the authority.

DEAD-ENDS: a different physical and semantic seal digest is not a custody
failure; each now recomputes correctly. Move39 reuse, repeated unused-inventory
proofs and universal lane-predictor closure remain unsupported paths.
