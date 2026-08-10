# ddm_sr1 — sd1 found −848 B on the semantic section. Build the receiver half so it can be banked.

**Owner:** codex arm · scorer-free build + identity proof; the pose leg is QUEUED not run · no Modal

## OPTIMAL FORM (read first)

Reference form: a counted semantic-allocation schema in the receiver, plus a proof that legacy
uniform-q4 archives decode BIT-IDENTICALLY through the extended receiver, plus exact tensor parity for
the mixed allocation. Declared reductions: SCOPE only — the pose measurement is deliberately OUT of
scope (it needs the scorer, which `cx2` holds) and is handed on as a named fire-order. MECHANISM
reductions are TOY-BRACKET: a schema that is not counted in the archive, an identity claim not proven
byte-for-byte, or parity checked on a subset of the 38 tensors.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## THE MEASURED WIN, AND WHY IT IS BLOCKED

`ddm_sd1_semantic` (600af8ef7d) measured the semantic section's quantization curve at n600:

| allocation | archive B | d_seg n600 | ΔS_sem |
|---|---:|---:|---:|
| uniform q4 (PR130 shipped) | 191,052 | 0.000286161635 | 0 |
| uniform q3 | 184,828 | 0.016552073161 | +1.622446846491 |
| uniform q5 | 202,324 | 0.001315884060 | +0.110477804687 |
| **4 tensors q3 + 12 q4** | **190,204** | 0.000287568834 | **−0.000423928449** |

Uniform loses in both directions. The mixed point wins **−848 B** at essentially flat d_seg. The four
demoted tensors are `frame_embed.weight` and `blocks.{1,2,3}.film.weight`. Real counted archive at
`/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip`,
sha256 `010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67`, with independent
double-build equality, unchanged carrier/HPAC/tokens, and exact parse-back of all 38 tensors.

**Three blockers, all named by sd1 itself:**
1. Its `SD1M` format is checkpoint-template-bound and **not public-receiver-readable** — current public
   decode is int4-only. A counted allocation schema is required. sd1 measured the header cost at 14 B.
2. **Pose was not measured.** sd1 explicitly closed "assume pose invariance" as a dead end, because
   PoseNet directly consumes the changed semantic frame and sparse Seg changes do not prove continuous
   PoseNet stability.
3. Legacy-q4 identity is unproven through the extended receiver.

## YOUR JOB — blockers 1 and 3 only

1. **Design and land a COUNTED semantic-allocation schema** in the receiver: per-tensor bit assignment,
   read at decode, priced in the archive. sd1's 14 B header is the measured reference, not a target to
   beat by cheating — if your schema costs more, say so and net it against the 848 B.
2. **Prove legacy identity.** An archive with no allocation record, or an all-q4 record, must decode
   BIT-IDENTICALLY to today's shipped path. Prove it byte-for-byte on the reproduced PR130 archive
   (sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B). A regression
   here would silently break the base, so this is the load-bearing proof.
3. **Prove selected-archive tensor parity** — all 38 tensors, exact, through the extended receiver.
4. **Re-price the net.** Schema bytes are counted. Report the true net: −848 B minus your schema cost.
5. **Hand off the pose leg** as a fire-order with the exact measurement named (matched q4-versus-selected
   n600 pose through the real evaluate path). Do NOT run it — `cx2` holds the scorer.

Two closures from sd1 that bind your design: **summing marginal per-tensor deltas is dead** (the
strongest pair carried a measured −0.0000731437 interaction cross-term; joint replay is required), and
**analytical raw-parameter-byte estimates are dead** (actual archive deltas were −6,224 and +11,272 B
against them). Price with real coders on real archives.

## HARD RULES

- `upstream/` IMMUTABLE. Intake clone READ-ONLY — copy out to work, never `git add` inside.
- No scorer run, no Modal, no training launch. The pose leg is handed on, not executed.
- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_sr1_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`. Always `.venv/bin/python`. `score_claim=false` throughout.
- `ddm_cx2` (composition, holds the scorer) and `ddm_tm1` (token model) are LIVE. Your output is a
  receiver CAPABILITY that a future composition consumes — do not compose an archive yourself.

## DELIVERABLE

The landed counted schema, the byte-for-byte legacy-identity proof, 38/38 tensor parity, the honest
net-byte figure after schema cost, and the named pose fire-order. If legacy identity cannot be proven,
that is a STOP and the schema does not land.
