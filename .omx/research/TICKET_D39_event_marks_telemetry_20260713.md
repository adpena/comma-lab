# TICKET-D39 — implement marked-event rows in `pact.causal_manifest.v1`

**Status:** BUILD-OWED
**Source spec:** `.omx/research/pact_causal_manifest_v1_event_marks_increment_spec_20260713.md`
**Do not infer launch authority:** implementation and local tests only

## Owned landing

- `src/tac/causal_manifest.py`: additive `event_mark` dataclasses, validation,
  canonical id, parser/writer, idempotent append/conflict behavior.
- Existing causal-manifest test module: the ten acceptance cases in the spec.
- One small fixture JSONL with all three event families; no score fields.
- A typed producer adapter at the event detector boundary only after the schema
  landing is sealed.

## Acceptance gates

1. Existing v1 rows parse unchanged.
2. New rows are marked, not counted: class edge, spacetime location, and
   attachment/incidence are mandatory.
3. The priority partition `topology > chart > receiver_lattice` is deterministic.
4. Restart is idempotent and conflicting ids fail closed.
5. Every row remains `[observability-only] NON-PROMOTABLE`.
6. Three clean review passes under the normal hot-file/serializer protocol.

## Explicit non-goals

- no edit to the causal manifest in the D39 research lane;
- no score, promotion, or Markov-sufficiency claim;
- no count-only compatibility shortcut;
- no training or provider dispatch.
