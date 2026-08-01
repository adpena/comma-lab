# G38 findings — verified continuation certificate hardening

Date: 2026-07-26  
Lane: `lane_g38_verified_continuation_certificate_20260726`  
Scope: G37 P0-3, original Pact research-only implementation  
Authority: structural verification only; no evaluator run, candidate, dispatch, score, promotion, or pointer mutation

## Concrete landing

G38 adds `src/tac/witness_control/verified_continuation_certificate_v1.py` and its focused adversarial test. The verifier refuses the former false-authority shape in which a favorable scalar plus a syntactically valid SHA-256 could become a subtree/cost-to-go bound.

A finite certificate is now derived only after reopening and joining all of the following exact bytes:

1. a canonical `GeneratorDomainManifestV1` identity payload whose SHA, base, epoch, horizon, full family/scale vocabulary, leaf descriptors, leaf identities, descriptor Merkle root, and leaf-ID root recompute;
2. a self-hashed canonical continuation proof envelope;
3. one strict finite measurement receipt per terminal leaf; and
4. one strict passing hard-constraint receipt per terminal leaf.

The joins bind exact base and epoch, manifest, leaf and endpoint, action path, first action, support hyperedge, continuation equivalence, horizon, authority and axis, hard-constraint contract, score implementation, binary64 `d_seg`/`d_pose` bits, archive bytes, public/evaluator receipts, and passing constraint state. Covered manifest leaves must stay inside one first-action/support branch. Proof status and `L/U` are verifier-derived; no caller supplies a certified label or a bound scalar.

`BASE_STOP` is controller-owned and may be admitted only with the strict G36 base receipt role plus strict passing decode-constraint receipt. The public proof-byte builder alone has no authority.

## Adversarial fixes

The independent G38 review initially returned NO-FIRE and reproduced four load-bearing failures. The landing addresses them as follows:

- **Opaque dependency laundering:** exact dependency bytes are now canonical-schema parsed and every semantic foreign key/component/pass bit is rechecked. Opaque bytes with a correct hash fail.
- **Caller-consistent manifest:** actual canonical generator-manifest bytes are mandatory and reopened; SHA strings or caller-authored root/cardinality alone cannot establish the partition.
- **Unsafe primitive lower bound:** primitive interval proof construction remains available as a storage format, but admission always fails closed until a typed interval theorem is independently verified. No primitive `L/U` or `VERIFIED` status can be emitted.
- **Forgeable Python type:** the public `_construct`/module seal were removed. Verifier-local construction registers a full-field fingerprint; mutation through `object.__setattr__` and `object.__new__` forgery are rejected. Durable consumers must reverify proof, manifest, and dependency bytes with `reverify_continuation_certificate`; the Python type alone is explicitly not an authority capability.
- **Authority laundering:** exact authority-to-axis tags are enforced and G38 currently admits only `RESEARCH_ADVISORY`. Production contest-CPU/CUDA admission stays closed until a governed production receipt adapter exists.
- **Numeric ambiguity:** score components require canonical finite binary64 JSON floats, reject integer aliases and negative zero, and measurement receipts preserve exact `float.hex()` bits. Duplicate keys, NaN, and pathological huge integers fail closed.

## Scorer arithmetic eureka

The adversarial review found a one-ULP implementation drift under our nose. Pinned `upstream/evaluate.py` computes:

`rate = archive_nbytes / N` then `score = 100*d_seg + sqrt(d_pose*10) + 25*rate`.

The prior `tac.contest_score.rate_term` computed `(25*archive_nbytes)/N`. At `archive_nbytes=1` these differ by one ULP, so the old supposedly exact finite lower bound could be optimistic. G38 now binds the pinned `upstream/evaluate.py` SHA (`7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`), the exact operation DAG, denominator, and binary64 component bits, and reproduces upstream operation order locally. The root lane separately repaired the shared helper and added the one-byte parity regression.

This is exactly the kind of micro-to-macro coherence failure the controller must prevent: a one-operation representation choice changes strict dominance at equality-scale margins.

## Consumer contract for G33/G36

G36 must emit canonical bytes for the exported schemas and exact receipt roles:

- `tac.continuation_finite_measurement_receipt.v1`, role `G36_VERIFIED_WHOLE_OBJECT_ACTION_ENDPOINT` or `G36_VERIFIED_WHOLE_OBJECT_BASE`;
- `tac.continuation_hard_constraint_receipt.v1`, role `G36_VERIFIED_DECODE_CONSTRAINTS`.

G33 must pass `generator_domain_manifest_payload_bytes` into every verification call and must not use a certificate merely because `type(x) is VerifiedContinuationCertificateV1`. At each durable/authority-consuming boundary it must call `reverify_continuation_certificate(...)`; `assert_registered_verified_certificate(...)` is only the cheaper same-process mutation/forgery guard.

Primitive certificates and production certificates are intentionally unavailable, not silently downgraded. A missing theorem adapter or governed production adapter blocks branch closure.

## Validation

- Focused verifier suite: `30 passed`.
- Ruff lint: clean.
- Ruff format check: clean.
- Adversarial cases include opaque bytes, resealed scalar tamper without expected-proof SHA, missing manifest bytes, authority/axis mismatch, production-label attempt, primitive-bound attempt, forged/mutated object, duplicate JSON keys, NaN, negative zero, integer float alias, huge integer, and the one-byte upstream arithmetic counterexample.

## Pointer honesty

Pointer moved: **no**.  
Exact authoritative score row: **none**.  
Candidate archive: **none**.  
Dispatch/heavy launch: **none**.  

This closes a decision-authority bug; it is infrastructure in service of an imminent exact row, not goal progress by itself.

