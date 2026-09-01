# ddm_ce1_afr1_compress_chain — consolidate the compression script into ONE clean chain entry point and PROVE it end-to-end (byte-exact afr1 rebuild); packet text/code tightness rides the same landing (owning memo: this charter + .omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_FINAL_DRAFT_TIGHT.md)

## MANDATE

Operator 20260901 (three steers, one deliverable): *"What is the compression script
limitation? We want to submit a tight production ready version"* + *"We want to consolidate
and ensure a clean world class but not over engineered compression script and test end to
end to ensure it works"* + *"Want end to end testing to ensure"* + *"Want to ensure the PR
itself is super tight and clean too all text and code and everything"*.

The shipping candidate is afr1 (archive sha
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 B,
S 0.14797617125559104 [contest-CUDA T4 n600]). The packet's single compression entry point
`experiments/ddm_pq2_compress_e2e.py` currently REFUSES that sha by name: its
`NOT_EXPRESSIBLE` registry records the five admitted post-rc2 lossless coder/container
stages (fx5 widened causal corrector · dx2 CABAC coefficient fold · gb1 groupbin8
conditioning · lb1 joint patch192-by-bank re-encode · afr1 tile48×groupbin8 with native
receiver binding) as NOT wired behind the entry point. Every one of those stages has a
landed tool, a landed receipt, and retained custody. This arm wires them into ONE
fail-closed chain, runs the chain END TO END, and proves the output byte-hashes to the
shipping archive. That converts the PR body's "Yes, with a limitation" into an honest,
verified "Yes."

## SCOPE

1. **Lineage trace at source (no trusting this charter's reconstruction).** From the
   receipts, pin the exact admitted stage ORDER and every intermediate archive identity
   (sha256 + bytes) from the rc2 body to afr1. Starting hypothesis (from the
   `NOT_EXPRESSIBLE['cbb8d928…']` entry authored at afr1 landing): rc2 body
   `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` (180,456 B) →
   fx5 → dx2 → gb1 → lb1 (180,083 B per the afr1 receipt) → afr1 (180,002 B). Receipts:
   `.omx/research/ddm_fx5_composed_rate_candidate_20260821.md` ·
   `ddm_dx2_cabac_receiver_fold_20260821.md` · `ddm_gb1_groupbin8_conditioning_20260824.md`
   + `ddm_gb1_groupbin8_verdict_20260824.md` ·
   `ddm_lb1_banked_lossless_joint_collect_20260829.md` ·
   `ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md`. If jt22/jt23's admitted
   contributions (`ddm_jt22_mixer_context_race_verdict_20260825.md`,
   `ddm_jt23_coder_collection_compose_verdict_20260826.md`) are IN the lb1 joint
   re-encode's inputs rather than separate stages, record that explicitly with the
   receipt line that proves it. Emit the traced lineage as a typed table (stage · tool ·
   input sha · output sha · receipt) — this table IS the chain recipe's provenance.
2. **Wire the chain behind the ONE entry point.** Extend
   `experiments/ddm_pq2_compress_e2e.py` with a chain-recipe grammar: an ordered list of
   stages, each naming its real tool (`experiments/ddm_fx5_build_e1_runtime.py` /
   `ddm_dx2_cabac_receiver_fold.py` / `ddm_gb1_groupbin8_conditioning.py` /
   `ddm_lb1_banked_lossless_joint_collect.py` / `ddm_afr1_tile48_receiver_identity.py` —
   verify each tool's ACTUAL argparse before emitting any flag; never invent flags), its
   pinned input sha, and its expected output sha. Execution = run stage tool → hash
   output → assert == pin → next stage. NOT over-engineered: a data table + a loop +
   fail-closed assertions; no plugin system, no new abstraction layer, no parallel entry
   point. Inputs resolve through the existing `--inputs-json` pattern (NO private paths in
   the file). The rc64 role pins stand: encoder `5c75e2c7…`, shipped member `05839d14…`.
3. **END-TO-END TEST — the falsifier.** Run the full chain from the retained rc2 body
   (custody under `/Volumes/APDataStore/pact/ddm_rc2*/` + Vertigo afr1 identity dir).
   PASS iff: final archive sha256 ==
   `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` AND bytes ==
   180,002 AND a second run from the same inputs is byte-identical (determinism repeat,
   the existing build-twice pattern). On PASS: move afr1 OUT of `NOT_EXPRESSIBLE` into
   the chain-recipe registry and update the module docstring honestly (the rc2/jg5/ck1
   entries STAY refused — they are not the shipping candidate). On any stage FAIL: do
   NOT synthesize substitute bytes; land the typed per-stage blocker naming the exact
   divergence (which stage, which sha, first differing offset) — an honest partial with
   the divergence named is the correct outcome.
4. **Consolidation + cleanliness pass on the compression-script surface.** One readable
   module: current 882 lines + the chain. Kill dead branches, keep the four-role rc64
   table and the honest-limitation docstrings but compress where the chain supersedes
   them. `.py` = 2 genuine review passes. A pytest that validates the chain registry
   shape + tool existence + pin consistency WITHOUT running the heavy chain (CI-safe).
5. **Packet tightness (text + code) — proposal, MAIN finalizes.** On e2e PASS, draft the
   replacement compression-script section for
   `.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_FINAL_DRAFT_TIGHT.md`:
   unqualified "Yes" + one sentence stating what the entry point verifiably rebuilds
   (exact submitted bytes, SHA-asserted, deterministic) and that content-deciding solve
   stages are included as their own scripts with receipts. Also sweep the packet dir for
   remaining looseness (stale limitation language, duplicated hashes) and list proposed
   cuts as rows — MAIN applies the operator-facing text. Do NOT edit the sealed gen-7
   packet custody, the shipped runtime tree (38-file, digest-bound), or `upstream/`.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. This arm's falsifier is BYTE-identity —
  no scorer run is needed (afr1's score is already the pointer row); if any step seems to
  want a scorer, emit a typed fire order and stop.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; retain EVERY intermediate stage archive (not just the final)
  + sha256+bytes in the result JSON, to
  `/Volumes/APDataStore/pact/ddm_ce1_afr1_compress_chain/retained/`.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
  (Per-stage re-encodes measured minutes-scale historically; the chain naturally
  checkpoints at stage boundaries — each stage output IS a resume point.)
- CLOSED-FORM-FIRST (operator 2026-08-31): all five stages are exact integer/rational
  coder transforms — NOTHING in this chain may be fitted or sampled; a stage that cannot
  reproduce its pin exactly is a blocker, never an approximation target.
- SEALED CUSTODY UNTOUCHED: the gen-7 packet swap (pq12) binds afr1 + the 38-file runtime
  by digest. No edit to submission runtime bytes, inflate path, or archive members. The
  compression script is REPO-side; the packet's runtime custody is frozen.
- Publication remains operator-gated: the submission-hold and contest-policy surfaces
  (task rows 1111 and 1363, owning packet dir
  `.omx/research/ddm_pq1_submission_packet_prep_20260815/`) stay open; nothing this arm
  lands publishes, hosts, or pushes anything.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rc64 filename conflation cost two arms a byte-close each (pq2 header, MEASURED
  `ddm_rv14f` 241 copies / 4 distinct bodies, `reverse_engineering/rc64_backend_role_registry.json`)
  — pin by ROLE sha, never by filename; the stale-pin cure (task row 1131) is recorded in
  the `ddm_pq2_compress_e2e.py` module header itself (live runtime member `05839d14…`).
- rr2 device-scoped decode desync (memo `ddm_rc2_t4_row_sixteenth_move_20260820.md`
  lineage, task row 1096): CPU-prob encode vs CUDA-prob decode produced S 27.83 on a
  byte-closed archive — any stage that consumes probability tables must consume the
  RETAINED table custody, never regenerate on a different device.
- jg2 zip-metadata defect (sp2 wave, task row 1145): unpinned zip timestamps/metadata
  broke archive reproducibility at 4 sites — every zip write in the chain uses the
  pinned-metadata pattern already landed there.
- The 08-31 NO-TOY erratum (`ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md`):
  two of MAIN's own numbers were toy-priced the day afr1 landed — every number in the
  lineage table must carry its receipt path, no remembered constants (the one-byte
  arithmetic slip of task row 1195 is the same genus).
- ddm_ma1's "pin unclearable" verdict was a SEARCH failure, not a custody failure (pq2
  header §THE PIN IS CORRECT): the encoder body existed all along at the named Vertigo
  path — when a stage input seems missing, search by ROLE/sha across all three custody
  roots before declaring a blocker.

## OPTIMAL FORM

- Family exemplar: the entry point itself is the reference —
  `experiments/ddm_pq2_compress_e2e.py` at commit edb0bd7ee8 (current main HEAD lineage;
  receipt path `.omx/research/ddm_afr1_pointer_move_and_no_toy_erratum_20260831.md` for
  the target identity). Its existing encode→build→verify→decode stage grammar, run-twice
  determinism assertion, `--inputs-json` no-private-paths contract, and fail-closed typed
  refusals are the family form this chain extension must match — same module, same
  discipline, five more stages.
- SCOPE reductions declared: (a) chain starts at the RETAINED rc2 body, not raw video —
  the content-deciding stages (jg3/jg5 edit solve, up3 pose re-solve, PR135-inherited
  training) remain Stage-A DOCUMENTED provenance exactly as the module already handles
  them; this is a scope reduction (legal), the five wired stages run at FULL mechanism
  through their real tools. (b) Packet text edits are PROPOSED by the arm, APPLIED by
  MAIN (operator-facing surface). MECHANISM reductions FORBIDDEN — no stage may be
  replaced by a copy of its retained output (that would be a fake rebuild; the tool must
  RUN and its output must MATCH).
- **PRIOR-LAW PREDICTION (falsifiable):** all five admitted stages are deterministic
  lossless transforms with byte-close receipts, so the chain reproduces afr1 byte-exactly
  on the first honest wiring. FALSIFIER: any stage output diverging from its receipt pin
  — that would measure an UN-RECORDED input or nondeterminism in an admitted pointer
  move's build chain (a custody gap worth more than the feature); count it plainly,
  name the stage, land the blocker.

## DELIVERABLE

`.omx/research/ddm_ce1_afr1_compress_chain_20260901.md` — typed rows: (1) traced lineage
table (stage · tool · input sha · output sha · receipt); (2) chain wiring diff summary +
pytest receipt; (3) the E2E RUN RECEIPT — per-stage wall seconds, output shas, final
sha/bytes vs the falsifier, determinism-repeat verdict; (4) retained-custody manifest
(APDataStore paths + shas); (5) proposed PR-body compression-script section + packet
tightness rows for MAIN; (6) honest residual list (anything still not expressible, with
its named builder). Commit via the serializer. End with the own-vehicle frontier line.
