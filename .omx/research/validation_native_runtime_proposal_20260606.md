# Validation — native (Rust/Zig/C/asm) eval-time runtime proposal

UTC: 2026-06-06T19:30:00Z · Agent: claude · Input: partner-agent essay on
native-code policy for the eval-time witness interpreter.

## Verdict

The proposal's policy core is ALREADY repo doctrine and partially scaffolded
code; its citations check out verbatim; its EV self-correction ("low-level
code is a multiplier after the witness/action grammar is right") matches the
current blocker order. Additive value: the action-VM design target, the named
payload-cleanliness audit bundle, and the ScoreProgramRuntime-first
sequencing. No CLAUDE.md amendment landed by agent fiat — the proposed rule is
~subsumed by the existing "Deterministic packet compiler — NON-NEGOTIABLE"
section; a delta amendment is DRAFTED below for operator/T3-council routing
per the council hierarchy (CLAUDE.md non-negotiable changes = T3).

## Claim-by-claim against ground truth

1. Contest split (offline unlimited / eval-time ≤30min / no payload outside
   archive) — VERIFIED verbatim at `upstream/README.md:114` (30-minute limit,
   T4-or-CPU) and `:118` (external tools free UNLESS large artifacts → must
   be in archive, counts toward size; explicitly applies to PoseNet/SegNet).
2. "Python remains the oracle until native passes byte-for-byte" — ALREADY
   NON-NEGOTIABLE: CLAUDE.md "Deterministic packet compiler" (fail-closed on
   hidden sidecars / non-deterministic builds / missing golden vectors /
   missing runtime-tree custody; identity/canonicalize/optimize modes;
   four target profiles) + `runtime-rs/README.md` states the same oracle rule.
3. Native runtime state — `runtime-rs/` workspace EXISTS: `qma-codec`,
   `stbm1br-codec`, `raw-locality-compare`, `python-ast-indexer`,
   `tac-packet-compiler` (SCAFFOLD: every fn `unimplemented!()`, parity
   harness 11/11 asserting scaffold contract, golden vectors at
   `src/tac/packet_compiler/golden_vectors/`, per-function promotion gate =
   flip to `assert_sha256_parity`). `inflate-cli` is a 3-line placeholder
   gated on "once profiling proves the need" — i.e. the repo already encodes
   the partner's "don't start in assembly / profile first" discipline.
4. PR110 selector-as-action-program + CPU/CUDA divergence — consistent with
   repo memory; deterministic CPU integer kernels for inflate are the right
   authority-stability response (matches the apples-to-apples discipline).
5. Mamba/Dreamer offline-teacher-only — matches existing MLX-portable +
   research-signal doctrine; no eval-time heavy models.

## Genuinely additive (adopted into planning)

A. **Action-program VM as the native decoder's design target** — the archive
   grammar section table (MAGIC/version/menus/selector streams/mask grammar/
   pose grammar/residuals/entropy models/crc) is the byte-level twin of the
   ActionEffect IR. Routing: the VM's section grammar must be DERIVED from
   `lane_action_effect_thin_ir_20260606` outputs; the Python
   `ScoreProgramRuntime` reference interface belongs to that lane's scope
   (decode_sections / render_pair / apply_action / write_raw).
B. **Named payload-cleanliness audit bundle** for any native submission:
   `binary_source_audit.md`, `embedded_constants_audit.txt`,
   `archive_payload_manifest.json`, `rebuild_instructions.md`,
   `python_reference_equivalence_test.py`. The packet-compiler non-negotiable
   implies these; naming them as required artifacts is a concrete tightening.
C. **Sequencing**: grammar in Python first → profile inflate wall-clock →
   native only for proven hot paths, per-function sha256-parity promotion.
   This is exactly the `tac-packet-compiler` promotion gate; no new mechanism.

## DRAFT CLAUDE.md amendment (routed to operator/T3 council; NOT landed)

> Native eval-time code is allowed when it expands the legal witness-program
> class or hardens deterministic replay; it is FORBIDDEN as a carrier for any
> learned/video-derived constant (weights, codebooks, masks, trajectories,
> LUTs) outside `archive.zip`. Every native runtime ships with: a Python
> reference oracle, the payload-cleanliness audit bundle (A–E above), and a
> bit-identical or scorer-identical equivalence test against the same archive
> bytes on the target authority.

Delta vs existing doctrine: only the named audit bundle and the explicit
"expand-the-program-class" allowance test are new; the rest restates the
packet-compiler section.

## EV ordering (unchanged by this proposal)

Native work stays BEHIND: pose-teacher threading at the birth-actuator
callsite, fakequant/parse-back survival for accepted births, SNeRV TUB
closure, ActionEffect IR + commutator ledger + menu ILP. A faster interpreter
for the wrong witness emits the wrong witness faster — partner's own words,
and the repo's `inflate-cli` placeholder already says the same thing.

Authority: planning prose; research_only; no score claims.
