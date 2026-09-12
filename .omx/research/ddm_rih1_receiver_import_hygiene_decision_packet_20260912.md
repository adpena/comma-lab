# ddm_rih1 — receiver import-hygiene decision packet

`[no-triality] [p0-ledger-ok]` · `$0` · `research_only=true` · `new_score_claim=false`

## Answer first

Clean the receiver and send the cleaned tree through a fresh FIRST-MEASUREMENT T4 chain before
publishing. The cleanup is only **3 added / 10 removed receiver-code lines** across two files, plus
two same-length manifest hash replacements. It passes
`submission_runtime_imports_within_allowlist`, and a cold local n600 decode is byte-identical to move
48. It nevertheless changes `measure_receiver_behavior_digest.v2` from `9f6e7168…` to `d3295528…`,
so pr19 correctly refuses timing inheritance. The incremental authoritative cost is approximately
**$0.30 and 20 minutes** for the T4 first-measurement fire.

One-line operator recommendation: **clean the three fallback symbols, fire one fresh T4
FIRST-MEASUREMENT chain, then publish only after the hosted-manifest check also closes.**

## Import census

The compliance detail reports two disallowed `file:root` records because its AST checker collapses
all absolute imports to their top-level root (`scripts/pre_submission_compliance_check.py:1214-1260`).
Source inspection resolves those two records into three offending imported symbols:

| Shipped file and line | Offending symbol | Contest decode reachability |
|---|---|---|
| `runtime/rc3_shared_mixer.py:17` | `experiments.ddm_rc2_hpac_semistatic_mixing_codec as base` | **Fallback-only; not taken on the successful contest path.** `inflate.py:16` imports `runtime.f26_inflate`; `f26_inflate.py:37` imports `residual_archive`; `residual_archive.py:40` imports `ihs2`; and `ihs2.py:47-48` imports this module. Its normal packaged import is `rc3_shared_mixer.py:15`, and the fallback is reached only if that relative import raises `ImportError`. The shipped tree has the relative target and does not ship an `experiments` package. |
| `runtime/sm1_semantic_mixer.py:17` | `experiments.ddm_rc1_adaptive_section_codec as rc1` | **Fallback-only; not taken on the successful contest path.** `f26_inflate.py:35-36` imports this module at public-entrypoint import time. Its normal packaged target is imported at `sm1_semantic_mixer.py:14`; the fallback is only the broad `except ImportError` branch. |
| `runtime/sm1_semantic_mixer.py:18` | `experiments.ddm_rc2_hpac_semistatic_mixing_codec as fixed` | **Fallback-only; not taken on the successful contest path.** Same module-load chain; its normal packaged target is `sm1_semantic_mixer.py:15`. |

The direct scratch import smoke resolved all three aliases to the shipped local files:
`runtime/rc2_hpac_semistatic_mixing.py`, `runtime/rc1_adaptive_model_sections.py`, and again
`runtime/rc2_hpac_semistatic_mixing.py`. A measured successful contest decode could not have taken
the fallback: `experiments` is neither a shipped local root nor an allowed external root, so that
branch would fail instead of producing the retained move-48 row. These branches are not globally
unreachable; they are repository-development fallbacks outside the contest package contract.

## Minimal diff, verbatim

Scratch tree: `/Volumes/VertigoDataTier/pact/ddm_rih1/scratch_runtime`. The preserved staging-shaped
copy used to run the compliance check is
`/Volumes/VertigoDataTier/pact/ddm_rih1/scratch_shippable_copy`. Neither source tree was edited.

```diff
--- runtime/rc3_shared_mixer.py
+++ runtime/rc3_shared_mixer.py
@@
-try:
-    from . import rc2_hpac_semistatic_mixing as base
-except ImportError:
-    from experiments import ddm_rc2_hpac_semistatic_mixing_codec as base
+from . import rc2_hpac_semistatic_mixing as base
--- runtime/sm1_semantic_mixer.py
+++ runtime/sm1_semantic_mixer.py
@@
-try:
-    from . import rc1_adaptive_model_sections as rc1
-    from . import rc2_hpac_semistatic_mixing as fixed
-except ImportError:
-    from experiments import ddm_rc1_adaptive_section_codec as rc1
-    from experiments import ddm_rc2_hpac_semistatic_mixing_codec as fixed
+from . import rc1_adaptive_model_sections as rc1
+from . import rc2_hpac_semistatic_mixing as fixed
--- MANIFEST.sha256
+++ MANIFEST.sha256
@@
-c0ac51f359f5bca5ef2e8ab803f8a1961be79dd27aa223e5496a0cfa6564a9fa  runtime/rc3_shared_mixer.py
+b643b0f1998151c5d30593caf334e286db68b70fd4880b28cb404c38a9935012  runtime/rc3_shared_mixer.py
@@
-505a0749124f94ece64f33c91f540b4bd1b2bb382a977479e293336dd80ceb26  runtime/sm1_semantic_mixer.py
+5668b060521e8c1c34e0657dff0591f7d3f8fbf14fedc49614dfdb684b45dd0e  runtime/sm1_semantic_mixer.py
```

Receiver-code size falls by 275 bytes: RC3 `8,010 → 7,908` and SM1 `9,693 → 9,520`.
The manifest remains 4,570 bytes. Total textual diff including the required manifest rebind is
**3 files, 5 insertions, 12 deletions**; the executable change alone is **2 files, 3 insertions,
10 deletions**. `archive.zip` remains exactly 179,111 B, SHA-256 `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`.

## Compliance, digest, and timing verdicts

The move-48 receipt says:

```text
disallowed=['submissions/_staging_move48_pr140_swap/shippable/runtime/rc3_shared_mixer.py:experiments',
'submissions/_staging_move48_pr140_swap/shippable/runtime/sm1_semantic_mixer.py:experiments']
```

The identical checker over the staging-shaped scratch copy returns
`submission_runtime_imports_within_allowlist: PASS`, `disallowed=[]`, with no import parse errors.
Receipt: `/Volumes/VertigoDataTier/pact/ddm_rih1/SCRATCH_COMPLIANCE.json`, SHA-256
`d1c5f3906617275610f71e3dde5f6ab876d558f813b92827d0a403b3bcd4111e`. Its other failures are
expected stale-receipt/runtime-binding failures plus hosting; it is evidence for this one check, not a
claim that the changed tree is release-ready.

The landed canonical function was run directly because a full caller search found no standalone
argparse wrapper for `measure_receiver_behavior_digest`; the implementation is
`tac.decode_wall_clock.measure_receiver_behavior_digest` (`src/tac/decode_wall_clock.py:228-230`).

| Tree | v2 behavior digest | Behavior rows | Legacy digest |
|---|---|---:|---|
| move 48 | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` | 49 | `9c0699015d36e3c8b90738d27152fc7adf9ac61297dd1801059245a464b49bdf` |
| cleaned scratch | `d32955281c10e8207930da20cc8168819ba44c3c84752dfb690475158f8c468e` | 49 | `da37ec112fd10861e25a408a58f58067610a34d7e0360186d17443d7f8b35242` |

The v2 row diff contains exactly the two receiver modules; `MANIFEST.sha256` is independently
validated and excluded from the behavior digest. The raw receiver-row diff has those two modules plus
the manifest rebind.

**Timing identity verdict: DIFFERENT.** Pr19's `measured_t4_identity_class_envelope` requires a
byte-identical receiver behavior digest and zero differing receiver rows. The cleanup violates both
conditions by construction. Therefore clean + inherit is not a legal option; the cleaned tree needs a
FIRST-MEASUREMENT chain and its own T4 fire. The local decode below is behavior evidence, not a timing
inheritance waiver.

## Decoded-output receipt

The real public `inflate.sh` ran cold on the cleaned scratch receiver with
`RLC1_ADVISORY_CPU=1`, `F26_TOKEN_DECODER=python`, token cache disabled, and no checkpoint resume.
The storage waterfall passed at 49.01 GiB free against an 8 GiB artifact budget. The run persisted the
117,964,800-byte token checkpoint before rendering and then persisted the complete raw.

| Fact | Measured result |
|---|---|
| Axis | `[macOS-CPU advisory]`; no score claim |
| Population | n600 pairs / 1,200 frames |
| Decode elapsed | 1,272.251592583023 s; host load was about 16, so this is not a contest timing verdict |
| Cache/resume | `token_cache=DISABLED`; `checkpoint_resume=false` |
| Token payload | 117,964,800 B, SHA-256 `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` |
| Cleaned raw | 3,662,409,600 B, SHA-256 `2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc` |
| Move-48 retained raw | 3,662,409,600 B, the same SHA-256 |
| Seeded subset | seed `20260912`; sorted `numpy.random.default_rng(seed).choice(600,120,replace=False)` |
| Subset comparison | 120 pairs / 732,481,920 B; zero mismatched pairs; both selected SHA-256 `b407935bee09fe7f98ea9a60f42680eade7707480ed1f0045af624a20cec9091` |

Run log: `/Volumes/VertigoDataTier/pact/ddm_rih1/decode_launch2/run.log`, SHA-256
`e7b62346aae5950a4a1c062e131f47d08827555759c69bea5a335e42a06dd36a`. Launch manifest:
`/Volumes/VertigoDataTier/pact/ddm_rih1/decode_launch2/launch_manifest.json`, SHA-256
`0c1c2138cdab0ea32c7c2bbf2e80209ea9a8647c757b9f8e3c7ab6ff6fc91bcb`. Raw and checkpoint remain
retained; nothing was deleted. The compact digest, diff, decode, and exact selected-index receipt is
`/Volumes/VertigoDataTier/pact/ddm_rih1/DECISION_RECEIPT.json`, SHA-256
`3b82227660c749008e9165f8fcef0966d10794830e5348e143ea29612f1cdfb7`.

## Three options, priced

| Option | Incremental money / time | What it buys and what remains |
|---|---|---|
| Publish as-is with import hygiene open | **$0 / no new measurement** | Keeps move 48's measured receiver and timing leg. Strict compliance remains **91/93**, with import hygiene and hosted manifest open. This requires an explicit operator acceptance of the receiver-hygiene exception; it is not a full strict PASS. |
| Clean + inherit move 48 | **$0 nominal, but unavailable** | The source diff passes the import check and local raw is identical, but v2 digest inequality and two behavior-row deltas make pr19 inheritance refuse. Do not attempt to manufacture an inherited leg. |
| Clean + FIRST-MEASUREMENT fire | **about $0.30 / about 20 min T4**, after this completed $0 local proof | Produces the cleaned receiver's own contest-T4 timing and exact row. The archive bytes and full local raw are unchanged, so the expected score is unchanged, but no authority claim transfers until the new fire lands. Hosting still closes separately at publish. |

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus, arm receipts, canonical research indexes and
`sub015_DAG_*` FEED blocks, design/SPEC surfaces, and `.omx/state` ledgers with the content queries
`receiver import`, `import hygiene`, `fallback import`,
`submission_runtime_imports_within_allowlist`, `rc3_shared_mixer`, `sm1_semantic_mixer`,
`measure_receiver_behavior_digest`, and `measured_t4_identity_class_envelope`. I also ran
`tools/list_canonical_equations.py --json` (490 registry entries) and searched its output for the
receiver/import/timing/identity surface.

Beyond the charter seeds, the search found: pr18's rule that the validated derived manifest is
excluded from v2 but real receiver modules remain included; pr19's zero-row-delta timing-class rule;
cpx3's earlier, explicitly untested hypothesis that fallback removal might preserve normal behavior;
and the live P0 ledger's move-48 91/93 state. These changed the plan in two ways: the scratch manifest
was rebound and validated rather than ignored, and digest inequality was treated as an automatic
first-measurement requirement even after full raw identity passed. No canonical equation authorizes
overriding that custody rule, and no cheaper already-measured cleanup was found in the searched scope.

## Boundaries

No staged tree, live PR tree, sealed tree, contract code, upstream file, scorer, Modal lane, hosted
asset, PR, or dispatch ledger was changed. No scorer ran. The failed first launch exited before decode
because system `python` lacked pinned Brotli 1.2.0; it materialized no payload. The successful relaunch
put the repo virtualenv first on `PATH` and retained every payload it created. This packet does not move
or publish the exact frontier.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: operator + MAIN; consumer store: a reviewed successor of
  `submissions/_staging_move48_pr140_swap/shippable` plus its first-measurement custody directory;
  fire trigger: operator chooses the recommended clean option; action: apply this exact two-module
  cleanup, rebind `MANIFEST.sha256`, emit a fresh intent/authorization, fire one T4
  FIRST-MEASUREMENT row, restage the packet, and require the import check to remain PASS.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: operator + MAIN; consumer store:
  `submissions/_staging_move48_pr140_swap/SWAP_COMMANDS.md`; fire trigger: operator instead explicitly
  accepts the import-hygiene exception and gives the one-line publish confirmation; action: retain the
  unchanged receiver, host/fetch-back the archive, supply the hosted manifest, and record that strict
  compliance remains open on import hygiene rather than calling 91/93 a full PASS.

composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48) unchanged (NOT published)

## LIVE-HYPOTHESES

- A cleaned T4 row will reproduce move 48's exact components because the complete local n600 raw is
  byte-identical and the archive is unchanged. This is plausible but remains untested on contest T4;
  only FIRST-MEASUREMENT may promote it.

## DEAD-ENDS

- **INSTANCE:** clean + inherit is closed. The v2 behavior digest changes and two behavior-bearing
  rows differ; pr19 requires zero.
- **INSTANCE:** hiding the absolute imports behind dynamic import machinery is closed. It would evade
  the AST check without cleaning the shipped dependency surface and would be a fake hygiene cure.
- **INSTANCE:** a cached or prefix-only decode as the identity proof is closed. The completed cold n600
  public-entrypoint decode now supplies the stronger proof.
- **INSTANCE:** treating the two compliance records as only two imported symbols is closed. The checker
  reports roots per file; source inspection proves three offending symbols.

<!-- # FORMALIZATION_PENDING: decision packet over an existing receiver and existing custody contracts; no new score law or canonical equation is introduced -->
