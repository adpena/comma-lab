# ddm_rp1 round 2 — NOTE TO MAIN: the n600 acceptance was verified on the wrong base

Written 2026-09-10 by the respawned ddm_rp1 agent, BEFORE any heavy compute (host-quiet
protocol honoured; no n600 pass, encode or parse-back has been started).

## The finding

All five n600 K=192 shards finished rc=0, and their numbers are internally consistent —
4,503 neutral tokens over 600 pairs, 15,552.3 first-order bits = 1,944.0 B. But they were
**not measured on move 40's field.**

`experiments/ddm_rp1_sizing.py` takes two objects from two different places:

| object | source | what it was in round 2 |
|---|---|---|
| the RANKING | `--rank-dir` (a flag) | built on `field_move40.u8` — correct |
| the BASE FIELD | `rp1.load_live_field()` (a **source constant**) | sj1's pass-4 npz — **move 37's field** |

`sizing.py:65` is `self.base_field, self.live_field, self.field_receipts =
rp1.load_live_field()`, and `load_live_field` is pinned to
`ddm_sj1_multipass_token_predistortion/admission_pass4/field_admitted.npz`, sha
`813bf1e6…`, with a hardcoded `changed != 9_209` refusal. The `sizing` subcommand has no
field flag at all. So every "argmax identical on all 196,608 cells" verdict the shards
emitted was measured against the body the pointer shipped **two moves ago**.

The shard receipts say so in the open and nobody read it that way:
`pricing_field.tokens_changed_vs_pristine` reads 10,099 with
`tokens_changed_vs_live_field` 890 → the live field carried 9,209 tokens vs pristine, and
9,209 is pass-4's count. Move 40's field is pass-4 + 473 (rp1 move 39) + 78 (sj1 pass 5)
tokens over ~295 pairs.

Nothing warned because **a flag and a constant can disagree silently.** Same genus as the
carrier silent-revert and the subset-writer revert: the guard existed on one object and not
on the one beside it.

## What survives, and why the run is recoverable

The rank computes `sj1_edit_mask = target != base` on the **override** field, and every
masked position is excluded from proposals (`order = order[~is_sj1_edit]`). So no accepted
proposal sits on a banked token, and at every proposed position the two bases carry the
**same** symbol. The edit `(pos, sym → best)` is therefore the identical edit on either
body — and that is now CHECKED per edit rather than argued.

What does not survive is the render **context**: the renderer's receptive field is 9 token
cells, so a banked edit near a proposal moves both the base argmax and the edited argmax.
So the joint neutrality has to be re-measured on the shipping base.

## The repair (committed, not yet run)

`experiments/ddm_rp1_rebase.py`, commit `daadbf7a2`. It re-verifies the accepted set on a
named base field with the same cumulative bisection the acceptance used, so a set that
fails is NARROWED rather than dropped whole. Cost is ~2 realizations per pair instead of
192, i.e. minutes, not the 4.6 shard-hours a re-run would take. It is conservative by
construction: it can only lose yield, never invent it (a proposal that was rejected on
pass-4 but would be neutral on move 40 is not recovered).

It carries its own control group: pairs whose two base planes are identical must transfer
at fraction 1.0. If they do not, the instrument moved, not the base — and the receipt
reports both fractions separately.

I did NOT patch `ddm_rp1_sizing.py`. Another arm may be running it, and the binding-drift
law says an edit to a live arm's script re-binds its checkpoints. If you want the class
killed rather than this instance repaired, the structural cure is: **any harness whose
ranking is field-overridable must take its base field from the same flag**, or refuse when
the two disagree.

## What I have NOT done

No n600 pass, no encode, no parse-back, no multi-thread job. Waiting on
`MAIN_GO_HEAVY_COMPUTE`.

`score_claim=false`. Frontier unmoved: `composition S 0.13763861019288715 @ 180,233 B
[contest-CUDA T4 n600]` (move 40).

---

## READY — waiting only on `MAIN_GO_HEAVY_COMPUTE` (appended 2026-09-10 10:26Z)

`quiesced_decode_timing` for rlc1 g3/threads-4 finished at 10:23Z (its receipts dir is
written); no timing process is running and this arm has started nothing. Everything that can
be done without CPU is done and committed:

- `daadbf7a2` — `experiments/ddm_rp1_rebase.py` (the repair; ruff clean, two review passes)
- `7e770638e` — the memo section + this note
- `fcc783547` — the rebase stop rule and eleven seal falsifiers, **pre-registered before the
  rebase runs**

The moment the GO file appears the chain fires in this order, all through
`tools/launch_detached_process.py`: verify-field → rebase ×5 → merge → twin pricing encode →
render candidate overlay → pose base/stale → carrier re-solve → resolved → frame-0 inside the
admission → admit on the resolved pose → twin encode of the selected subset → stage → close →
parse-back + seg identity → seal with move 40's inherited decode-wall-clock leg.

If you are queueing more quiesced windows, say so and I will keep holding; if the GO is simply
owed, `touch /Volumes/VertigoDataTier/pact/ddm_rp1_round2/MAIN_GO_HEAVY_COMPUTE` releases it.

## BLOCKER as of 10:47Z — the hold's premise expired

Your instruction named ONE 25-minute window. Since 10:07Z the host has run a SERIES:
rlc1 g3/threads-4 (10:07→10:23), g4 (10:24→10:40), and at least one more from 10:40Z, each
~16 min with ~30 s gaps. This arm has held the whole time and started nothing — no n600
pass, no encode, no parse-back. Fifty minutes of the chain's ~3-hour budget are gone to the
hold, and I will not start into a live window and cost you a 25-minute measurement.

**Release with:** `touch /Volumes/VertigoDataTier/pact/ddm_rp1_round2/MAIN_GO_HEAVY_COMPUTE`

Everything is armed and resumable from disk. The next agent (or this one, on your reply)
starts at `verify-field` with no re-derivation owed — checkpoint step 27 carries the full
ordered command chain.

The part that should not wait for the GO: **do not let any arm consume
`ddm_rp1_round2/n600/shard*/field_rp1_sizing.npz`.** Those planes are the sj1 pass-4 body
plus this round's edits, so shipping one silently REVERTS the 551 tokens moves 39 and 40
banked. The only field this round may ship is the rebase's output, and it does not exist yet.
