# Original recall and corrected source mapping

## RECALL EVIDENCE

The parent searched the full `.omx/research/` Markdown corpus by content with
`context.statistic|f26_corrector_native|rlc1_geometry|prior.network|integer.*overflow|__int128`.
Output is retained at `/Volumes/APDataStore/pact/ddm_mrs3/recall_research.txt`.
The query `corrector|hpac|context.statistic|rlc1|mrs[123]` covered
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, all `docs/` including design/SPEC
surfaces, and `.omx/state/canonical_task_status.jsonl`; output is
`recall_graph_docs_tasks.txt` in that store. The canonical command
`.venv/bin/python tools/list_canonical_equations.py --json` returned 490 records;
all are retained in `equations_recall.json`, with 82 content matches in
`equations_selected.json`. These are search matches, not 82 applicable laws.

Independent read-only agent `mrs3_recall_review` separately searched research
content, equations, index/DAG, design/SPECs, task ledger and historical harness
bridge. Its research pass was bounded to Markdown files no larger than 300 KiB;
the parent's content search had no such file-size exclusion. Queries included
`context.statistic|f26_hpac_native|serial.python|1185\.9|1699`,
`corrector.*native|native.*corrector`, and
`corrector.*(tim|profil)|tim.*corrector`.

Findings beyond the charter seeds changed the proposed implementation:

- `ddm_rr8_corrector_only_native_port_20260820.md:63-109` identifies the actual
  native-corrector mechanism: avoiding repeated NumPy allocations while keeping
  ordered float64 products and roots. It also requires persistent state equality,
  because cold cells can emit equal probability rows despite divergent tables.
  Its old prefix measurements are historical evidence, not current measurements.
- `ddm_tc3_20260910/native_trace_state_review.md:20-76` enumerates the current
  23-family corrector's 79 persistent arrays plus `have_prev`, and requires a
  completed frame, not merely a closed group, for a restorable snapshot.
- `ddm_rr8_t4_wallclock_verdict_20260820.md:89-101` explicitly withdraws numerical
  transfer of unmatched local component times to T4. This keeps both mrs2 ratios
  as one whole-process transfer assumption and prevents invented GPU-prior time.
- `ddm_mxo1_free_decode_time_online_context_mixing_20260911.md:64` confirms that
  the 23-family adaptive mixer is existing decoder machinery; no invention or
  originality claim is appropriate for a native lowering.
- Direct source inspection found a stronger correction than the memos:
  the sealed `runtime/f26_inflate.py:455-463` explicitly rejects native HPAC
  because it is unpatched and would decode a different field. The original
  successful move-53 row therefore did not run the C file named in this charter.

`experiments/ddm_mrs2_profile.py:75-92` assigns adaptive-corrector and Lane-mixer
methods to `prior_context_statistics`; it assigns HPAC `prepare_frame_context`
to `prior_network`. The actual native references are the sealed
`f26_corrector_native.c` and `rlc1_geometry.c`, with the remaining shared mixer
in Python. The former uses float64 arithmetic and is 1,248 physical lines;
the latter is 130 lines and uses `__int128`. A plain int64 transcription is
not automatically equivalent. The geometry's actual contribution was then
measured on the pre-registered 48 pairs instead of inferred from its visible loop.

The task ledger's mrs2 successor row names MAIN as owner and the mrs2 fire-order
store as consumer. This arm registers an exact source-scope correction and
custody handoff, rather than leaving an unowned suggestion.

The Codex memory registry was consulted for prior Pact landing conventions.
Only its serializer guidance was relevant: repeat post-edit hash arguments,
perform two review marks after source indexing, and preserve unrelated work.
All current source, timing and custody facts were verified from live files.

## OPTIMAL FORM

This is a source-mapping and timing-attribution result, `research_only=true`,
`score_claim=false`, `promotable=false`. No context cure was implemented.
The source copy is byte-identical to landed mrs2. The declared diagnostic change
is finer external timers and explicit deterministic CPU algorithms. Consequently
the new profile supports within-run attribution, not a controlled speedup claim
against mrs2. No CUDA execution, new cold n600 decode, or bare-venv smoke is claimed.

## Integration boundaries

Sensitivity-map consumer: the measured stage shares and source map in HANDOFF.md.
Pareto consumer: the original reviewer and projection gates, still unpassed.
Bit allocator: N/A because charged bytes are unchanged.
Autopilot dispatch: N/A because this charter forbids dispatch.
Empirical posterior: the 48-pair attribution, bounded to this host and instrument.
Probe-disambiguator: the finer profile distinguishes corrector work from the
visible Lane geometry loop and from HPAC prior computation.
All future work is consumed by MAIN_FIRE_ORDER.json. No new strict gate,
canonical equation, trainer, codec or production helper is claimed.

<!-- # FORMALIZATION_PENDING: scoped source audit and timing attribution, not a new score law -->
