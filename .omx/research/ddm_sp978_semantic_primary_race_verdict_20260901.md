# DDM SP978 semantic-primary race verdict

**Verdict: `PRECONDITION-REFUSED / REQUESTED-CONTRAST-COLLAPSES` at FORMULATION scope.**
The pinned AFR1 comparator is already the PR130/135 semantic-primary vehicle: its receiver decodes a
dense `600x384x512` five-class field, and `SemanticTokenRenderer` consumes those class IDs through a
five-entry embedding. AFR1 does not ship the `IX2TOK01` latent lattice named by the charter. Rebuilding
the named semantic-primary treatment on the current AFR1 cell therefore produced the identical
`180,002 B` archive, identical `117,964,800`-byte semantic field, and identical
`3,662,409,600`-byte RGB output. There is no distinct control/treatment race unless a successor changes
the representation contract or chooses an actually latent parent.

This is not a semantic-primary family closure. It is a refusal to call the already-present incumbent
mechanism a new treatment. No score row was created, and the AFR1 exact frontier did not move.

Axis for new work: `[macOS-CPU full receiver identity / scorer-free exact byte measurement]`.
Primary machine receipt:
`/Volumes/VertigoDataTier/pact/ddm_bx1_semantic_primary_full_vehicle_race/RESULT.json`, `6,898 B`,
SHA-256 `e8ad6a5fdf21bd1b65e8d6213a924ca40db658ad1a3d84d27963f772e3ef5f20`.

## WHAT WAS MEASURED

### The requested representation is already the incumbent

The physical AFR1 archive begins with `RX1M`, not `IX2TOK01`. Strict parse-back reports an RC64 token
stream and the current receiver emits a `uint8` field with alphabet exactly `{0,1,2,3,4}`. The field is
consumed by `cpr1/inflate.py::SemanticTokenRenderer`, whose token embedding has `NUM_CLASSES=5`.

The current-cell pins all reverified:

| object | bytes | SHA-256 |
|---|---:|---|
| AFR1 complete archive | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| decoded dense semantic field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| realized RGB | 3,662,409,600 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` |
| fresh runtime source tree | repository/runtime tree | `6579ac6dd84eae6a2b6b6f3cbac9fb6b4c44913b1dce76e71a560ea1879f94da` |
| token-decoder fingerprint | receiver contract | `8cabd39009d70cc617406ee64b484b34da0e92edfb98a3bdaa0ba598ca561109` |

The charter's pinned base-field SHA `cc10a7b...63efb` is therefore not evidence of a latent parent. It
is the exact five-class semantic object that AFR1 already sends through the PR130/135 renderer.

### Exact byte table

Both control and semantic-primary identity treatment use the same exact archive bytes. Every number is
a physical count from the strict `RX1M` parse or the ZIP itself.

| counted section | control bytes | identity-treatment bytes | delta |
|---|---:|---:|---:|
| RX1 header | 14 | 14 | 0 |
| counted HPAC probability model | 13,515 | 13,515 | 0 |
| semantic renderer weights | 30,856 | 30,856 | 0 |
| CPR1 pose carrier | 22,010 | 22,010 | 0 |
| compact residual | 96 | 96 | 0 |
| dense semantic RC64 token stream | 113,411 | 113,411 | 0 |
| ZIP framing | 100 | 100 | 0 |
| **complete archive** | **180,002** | **180,002** | **0** |

The model-plus-token pool is `13,515 + 113,411 = 126,926 B`. The section sum closes exactly:
`14 + 13,515 + 30,856 + 22,010 + 96 + 113,411 + 100 = 180,002 B`.

All four retained archive copies—control, control repeat, identity treatment, and identity-treatment
repeat—are `180,002 B` at the same SHA `cbb8d928...d405bf25`. These are full counted containers, not
TK1 label-stream projections.

### Parse-back and repeat receiver identity

Two independent full-n600 receiver runs now exist for the identical archive:

| pass | source | token result | RGB result | timing |
|---|---|---|---|---:|
| control | AFR1 retained full receiver identity | 117,964,800 B, SHA `cc10a7b...63efb` | 3,662,409,600 B, SHA `7246a4ff...f2de7` | 698.324 s wall |
| identity treatment | fresh SP978 retained decode | 117,964,800 B, SHA `cc10a7b...63efb` | 3,662,409,600 B, SHA `7246a4ff...f2de7` | 718.861 s decode+render |

The fresh run did not use a token cache and retained its own complete token checkpoint and raw output.
Its token stage took `449.602 s`; render plus resize took `220.598 s`; frame-0 selector and I/O took
`47.160 s`. The exact runtime report is retained in
`receipts/treatment_repeat_decode.log`, `2,544 B`, SHA
`4e703b366e6cbfa8cb3ce33beda71095e94b06c0b744e43fc899dd325e9630fc`.

Control versus identity treatment has:

- archive delta: `0 B`;
- decoded-token differing bytes: `0 / 117,964,800` by equal full SHA;
- realized-RGB differing bytes: `0 / 3,662,409,600` by equal independently computed full SHA;
- repeat-archive differences: `0 / 4` retained copies.

### Pose carriage

Pose is not assumed or borrowed from another vehicle. The current archive carries the exact CPR1
frame-0 carrier in the counted `22,010 B` compressed carrier section. Parse-back expands it to
`22,316 B`; the receiver consumes a `14 B` selector payload at SHA
`67d43d9050b1005ef04ef8f0e5657d10bd3cbd3920a9874449c9876a676b9a17`. No compensation overlay is
present in this AFR1 object. Because archive, carrier, selector, token field, and final RGB are all
identical, pose carriage is identical by construction; no new pose number is claimed.

## CONDITIONAL N600 REALIZATION

The charter's scorer condition is an AND gate: the treatment must change realized RGB and exact bytes
must preserve an improvement path. Neither condition holds. Realized RGB is identical and complete
archive bytes are identical. The conditional n600 scorer realization therefore **did not fire**.

This arm did run the required full receiver repeat, but it ran no SegNet, PoseNet,
`upstream/evaluate.py`, Modal job, contest-CPU row, or contest-CUDA row. Scoring the identical archive
would duplicate the already-authoritative AFR1 row and consume MAIN's scorer lane without a new object.
No authority fire order is warranted.

The charter's prior-law prediction—"a new semantic-primary serialization is rate-heavier than AFR1"—is
not tested by the identity row. The named treatment is not new. Equality is evidence that the contrast
collapsed, not evidence that a distinct semantic representation costs zero bytes.

## CROSS-HALF DECLARATION

- **Distortion half:** held exactly, but only because control and identity treatment are the same
  semantic field, renderer, pose carrier, selector, and complete archive. This is identity, not a new
  semantic-primary distortion result.
- **Rate half:** no new mechanism was measured. The current HPAC/RC64 semantic representation is already
  AFR1's `126,926 B` model-plus-token pool. A direct-explicit coder, a multi-token object, or an
  `IX2TOK01` parent would be a different formulation and needs an explicit distinct receiver contract.

The Cross therefore has one held half and one **not-instantiated-as-distinct** half. It has no new
meeting point.

## CCS1 FOLD

CCS1 is now terminal and folds cleanly into this adjudication. Its one declared instance—seed
`20260901`, 512 nonlinear receiver-causal leaves on unchanged AFR1 X—built a receiver-closed,
repeat-identical **664,770 B** archive at SHA
`a56d587659864f97ce56e2a8fd5e9332ce0e36c46b1c9d651052d7976bd75fa0`. It exceeds the `137,986 B`
fixed-distortion gate by **526,784 B** and is `4.8177x` the gate.

CCS1's two full decodes and two full renders are also byte-identical to AFR1. Its negative is
`INSTANCE`-scoped, not a causal-model family closure. The measured pool says that this sparse
joint-table schedule does not hold the rate half; this SP978 result says the proposed semantic-primary
escape is already the incumbent and therefore is not an exterior treatment. Neither supplies a new
Cross meeting point.

Load-bearing CCS1 receipt:
`/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/ccs1_20260901/RESULT.json`, `15,362 B`, SHA
`7ee51b8c8d6dee8fc98aeac308877e6b25df8dcc441a4120c50c2bf9e5705a43`.

## RECALL EVIDENCE

The recall was not limited to the charter seeds. Content searches covered `.omx/research/`, arm
receipts, current source/runtime, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, canonical equations,
lane/task state, and retained custody with query families:

- `semantic-primary|semantic primary|#978|multi-token|dense semantic`;
- `latent tokens|IX2TOK01|tq1c|HP1|SV2`;
- `PR130|PR135|HPAC|SemanticTokenRenderer|token_embed`;
- `AFR1|cc10a7b|117964800|RX1M|RC64`.

Findings beyond the charter's named seeds changed the plan:

1. `ddm_tc1_tr1_lifecycle_spec_20260817.md` already says the live PR130-lift successor is the
   semantic-primary architecture. That warned that a fresh semantic-primary label could duplicate an
   existing vehicle.
2. `ddm_xov1_crossover_pass_20260901.md` pins AFR1's exact object as a **semantic token field X** and
   decomposes the current `126,926 B` HPAC-model-plus-RC64-token pool. That directly contradicts the
   charter's `IX2TOK01` description of the current frontier.
3. Live receiver source and strict archive parse proved the contradiction on the actual bytes:
   `RX1M`, five-class field, five-entry token embedding, and no `IX2TOK01` member.
4. RR3 and the PR130 intake confirm that raw dense semantic labels are the canonical CPR1 target. The
   current runtime is a later derivative, so the charter's historical `hpac_integer.py` SHA is an
   existence-proof pin, not the current runtime-file SHA.
5. The canonical-equation search did not reveal a law that makes two identical archive hashes a vehicle
   race. Nothing beyond the exact parser/receiver evidence displaced the identity refusal.

This changed execution from "invent a direct semantic treatment" to "retain and repeat the exact
semantic incumbent, prove the contrast collapses, and refuse a fake novelty claim." A generic raster or
TK1 label-only stream would have recreated the section-projection defect the charter forbids.

## DENOMINATORS AND BOUNDARIES

- Requested full-vehicle contrast: `1/1` classified; it collapses because both named sides resolve to
  AFR1 semantic-primary.
- Complete archive copies: `4/4` retained, `4/4` at `180,002 B` and SHA `cbb8d928...d405bf25`.
- Full receiver passes: `2/2`; each covers 600 frames, `117,964,800` decoded semantic bytes, and
  `3,662,409,600` realized RGB bytes.
- Section accounting: `7/7` counted components close exactly to the complete archive.
- Field alphabet: five observed values, `{0,1,2,3,4}` over all `117,964,800` sites.
- Conditional scorer gate: `0/1` fired because both required predicates are false.
- CCS1 fold: `1/1` declared CCS1 instance consumed; it is `526,784 B` above its gate.
- New scorer runs: 0. New authority rows: 0. Remote calls: 0. Training burns: 0. Upstream writes: 0.
- Negative scope: **FORMULATION** for the requested AFR1-versus-semantic-primary contrast. Semantic
  representation families, direct-explicit coders, and multi-token changed objects remain unclosed.

Bulk custody is under
`/Volumes/VertigoDataTier/pact/ddm_bx1_semantic_primary_full_vehicle_race/`. Storage preflight observed
`191 GiB` free at start. The retained tree is `7,385,708 KiB`; `201,130,643,456 B` remained free after
the run. Both archives, both repeats, both raw outputs, both token fields, runtime, extracted payload,
and logs are preserved. Nothing was deleted.

## DISPOSITIONS

- Chartered AFR1-current-cell versus semantic-primary race: `PRECONDITION-REFUSED` at FORMULATION scope.
- Identity archive and fresh full receiver repeat: `FIRED-AND-RETAINED`.
- Conditional n600 scorer row: `FOLDED`; the treatment changed neither bytes nor RGB.
- CCS1 v1: `FOLDED-CLOSED-AT-GATE` at INSTANCE scope.
- Authority promotion: `NOT-WARRANTED`; no distinct candidate exists.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`; owner: `MAIN / newly assigned #978 successor`; consumer
  store: `/Volumes/VertigoDataTier/pact/ddm_bx1_semantic_primary_full_vehicle_race/`; fire trigger: MAIN
  pins two actually distinct receiver contracts—either an `IX2TOK01` latent parent versus the AFR1
  semantic vehicle, or a new direct/multi-token semantic schema on AFR1—confirms no duplicate #1374/QX
  or MT1 owner, and requires both complete archives to parse back and repeat before any scorer request.**
  Build that one corrected full-container race; do not relabel the AFR1 identity archive as a treatment.

## LIVE-HYPOTHESES

- A corrected latent-versus-semantic race could still expose a vehicle-level difference because the old
  tq1c `IX2TOK01` object is genuinely distinct from AFR1's dense five-class field. It remains plausible
  only if the cells and renderer obligations are matched rather than importing old TK1 projections.
- A new direct-explicit or multi-token schema on the exact AFR1 field could trade the `13,515 B` HPAC
  model against a larger or smaller token stream. It is plausible because the current rate pool is
  physically decomposed, but no distinct full container was built here and generic coder reraces on the
  same current body are already drained.
- A parameter-sharing causal model different from CCS1's sparse 512-leaf table may still approach the
  shipped HPAC positive control. CCS1 is only one bad instance; its `607,228 B` stream does not prove
  nonlinear causal models cannot reach the current `113,411 B` stream.

## DEAD-ENDS

- Do not rerun the charter literally with AFR1 as control and "PR130/135 semantic-primary" as treatment.
  They are the same representation on the pinned bytes; the repeat produced zero archive, token, and RGB
  delta.
- Do not use TK1's label-stream-only prices as the treatment archive. They omit the full renderer,
  carrier, residual, and container and recreate the exact projection defect this charter was meant to
  cure.
- Do not call the identity copy a new vehicle, score, or semantic-primary win. Equal SHA is an identity
  proof, not novelty.
- Do not fire an n600 scorer on this identity archive. The authoritative AFR1 archive already has its
  contest-CUDA row, and the charter's conditional gate is false.
- Do not rerun CCS1 v1 seed `20260901` with the same 512-leaf schema. Its exact complete archive is
  `664,770 B`, `526,784 B` above the fixed-distortion gate.

Own-vehicle frontier: **S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, AFR1 archive
SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — UNMOVED.**
