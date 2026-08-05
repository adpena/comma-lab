# TP1 packet - TR1 primary launch assembly - 2026-08-05

schema: `ddm_tp1_tr1_primary_launch_packet.v1`

authority: design/config/packet only. No trainer file edits, no launch, no scorer, $0.

Current own-vehicle line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]` from qo1/sb1. Contest pointer is borrowed and unmoved.

## Verdict

TP1 can honestly seal the BI1 birth ON/OFF launch surface today. It cannot honestly claim that PE3 conditioning or cheapdct4 pose carriage are already consumed by the TR1 trainer: the current trainer argparse declares no `pe3`, `cheap`, `dct`, or generic `conditioning` launch flags. Therefore this packet has two parts:

1. A real, DSL-validated BI1 birth A/B packet, runnable by MAIN under the governed launcher after smokes.
2. A fire-ordered build table for the two missing consumers needed before anyone may call the full object `birth ON/OFF x pose-carrying base x PE3-conditioning slot`.

No score progress is claimed. This is a launch/control artifact.

## Recall Evidence

| Input | Fact consumed | TP1 use |
|---|---|---|
| `.omx/state/main_hot_state.md` | LC1 made `PE3 = CONDITIONING-ONLY`; TR1 learned carrier is crowned primary; BI1 birth + cheapdct4 carriage routed as TR1-line assets. | Primary route is TR1 learned carrier, not PE3 target replacement. |
| `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md` | Ideal PE3 labels worsened all 32 pairs; net fixed `-12,884`; generator-pair-bisector caused most introduced errors. | PE3 target/label substitution is FOLDED. PE3 may only be a conditioning input with trust gates. |
| `.omx/research/ddm_bi1_20260805/BI1_RECEIPT.md` | Built default-OFF TR1 birth seed/amplify flags: `--tr1-birth-seed-weight`, `--tr1-birth-seed-classes`, `--tr1-birth-seed-dilate-px`, `--tr1-birth-amplify-weight`, `--tr1-birth-amplify-persist`. Start values: `0.35`, `lane`, `1`, `0.05`, `inverse_thickness`. | BI1 ON/OFF is the only TP1 score-affecting trainer mechanism that is currently wired. |
| `.omx/research/ddm_tp1_v9_telemetry_port_20260731.md` | `--telemetry-v9-port on` is read-only/default-OFF, checkpoint-byte invariant when off, and emits loss/lever/alarm rows. | All TP1 tickets turn telemetry on. |
| `.omx/research/ddm_od9_20260805/OD9_RECEIPT.md` | `stage2_only_cheapdct4_qcoeffs` coded n32 as `2,157 B`; projected n600 as `40,444 B`. | cheapdct4 is a routed pose-carriage asset, not yet a TR1 in-loop consumer. |
| `tools/launch_tr1_run.py` | Governed launcher gates: seal freshness, import custody, memory preflight, scorer/trainer slot, detached receipt. Memory floor = `12.8 GiB x 2.0 = 25.6 GiB`. | Packet commands use the governed launcher, dry-run first. |
| `experiments/train_tr1_partition_renderer_mlx.py` argparse | Current declared flags include BI1, v9 telemetry, composed-S, optimizer persistence, and existence-hinge flags. No PE3 or cheapdct4 consumer flags exist. | Missing PE3/cheapdct4 consumers are build debts, not hidden ticket flags. |

QO1/SB1 score arithmetic checked from the live receipt:

`100*0.00431179 + sqrt(10*0.00071459) + 25*357836/37545489 = 0.7539807296911207`.

## Warm Anchor

Primary measured warm anchor for the BI1 A/B:

`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz`

Reason: b4s window_02 has the confirmed incumbent full-confirm `realized_dseg_mean = 0.004067128499348959`, `tokens_bytes = 272,578`, `total_counted_bytes = 276,078`. Window_03 contains gate-ranked candidates, but those are not n600-confirmed in TP1's no-scorer scope.

The smoke/full tickets resume from this same checkpoint, use identical inherited b4s levers, and differ only by BI1 birth flags. Both add `--persist-optimizer-state on` so future checkpoints preserve optimizer moments. Because the parent checkpoint predates optimizer-state persistence, the first resume cannot restore old moments; the trainer must record its no-moment note and persist moments from the next checkpoint onward.

## Compiled Ticket Matrix

All four tickets compile through `TR1RendererProgramV1.sealed_ticket()` against the current trainer's argparse. They share:

- `num_pairs=600`, `seed=0`, `variant=lotto`, `D16/c4`, `renderer_width=24`, `shared_base`, `solve_project`, `lane_guard`, `margin_weighted_loss=on`, `rate_model=entropy`, `byte_ledger_coder=smevr`.
- `--telemetry-v9-port on`
- `--persist-optimizer-state on`
- `--gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
- `--resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz`

| Ticket | Purpose | Epoch cap | Wall cap | BI1 flags | ticket_hash |
|---|---:|---:|---:|---|---|
| `smoke_birth_off` | MAIN bounded OFF smoke | 821 | 30 min | none | `a0a9af3546568971b8ee4d1f9e72f9ab4dba6b4213febedca2c60b2055f72842` |
| `smoke_birth_lane_on` | MAIN bounded ON smoke | 821 | 30 min | lane seed/amplify | `a2be27e59589f9e6cabed19bace110bc1be0350c556907e81703118513baa15e` |
| `full_birth_off` | clean OFF full A/B, not historical reuse | 946 | 130 min | none | `833d849a3a75e18a6086fa10526e7debedf3fe71b9bd6225eaa831e4722a4ce3` |
| `full_birth_lane_on` | clean ON full A/B | 946 | 130 min | lane seed/amplify | `6478bb749887fc5688f3c4b19d9242e6c3f40167927764045616ead9818d1434` |

ON-only BI1 delta:

```text
--tr1-birth-amplify-persist inverse_thickness
--tr1-birth-amplify-weight 0.05
--tr1-birth-seed-classes lane
--tr1-birth-seed-dilate-px 1
--tr1-birth-seed-weight 0.35
```

## Ticket Materialization

MAIN can materialize the exact tickets into durable research custody with this command. If any hash differs from the table above, stop and re-read the changed DSL/trainer before launch.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from tac.witness_dsl.curriculum_dsl import Lever
from tac.witness_dsl.spec_tr1_renderer_20260728 import TR1RendererProgramV1, lever_window
from tac.witness_dsl.bi1_birth_seed_levers_20260805 import lever_tr1_birth_seed_amplify

repo_ticket_dir = Path(".omx/research/ddm_tp1_20260805/tickets")
repo_ticket_dir.mkdir(parents=True, exist_ok=True)

parent_ticket = Path("/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/tickets/window_03_ticket.json")
parent = json.loads(parent_ticket.read_text())
parent_levers = [
    Lever(name=d["name"], overrides=d["overrides"], notes=d.get("notes", ""))
    for d in parent["levers"]
    if not d["name"].startswith("tr1_window_ep")
]
persist = Lever(
    name="tr1_persist_optimizer_state_on",
    overrides={"--persist-optimizer-state": "on"},
    notes=("ddm_op2 OP2-1 optimizer-state persistence; parent b4s checkpoint has no "
           "moments, so the first boundary records no-opt-state and future checkpoints persist."),
)

base = "/Volumes/VertigoDataTier/pact/ddm_tp1_20260805"
resume = "/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz"
gt_cache = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
expected = {
    "smoke_birth_off": "a0a9af3546568971b8ee4d1f9e72f9ab4dba6b4213febedca2c60b2055f72842",
    "smoke_birth_lane_on": "a2be27e59589f9e6cabed19bace110bc1be0350c556907e81703118513baa15e",
    "full_birth_off": "833d849a3a75e18a6086fa10526e7debedf3fe71b9bd6225eaa831e4722a4ce3",
    "full_birth_lane_on": "6478bb749887fc5688f3c4b19d9242e6c3f40167927764045616ead9818d1434",
}

def ticket(name, epochs, wall, birth):
    levers = list(parent_levers) + [lever_window(epochs, wall, batch_pairs=8, lr=2e-3), persist]
    if birth:
        levers.append(lever_tr1_birth_seed_amplify(
            seed_weight=0.35, amplify_weight=0.05, classes="lane",
            dilate_px=1, persist="inverse_thickness"))
    prog = TR1RendererProgramV1(
        levers=tuple(levers), num_pairs=600, out_dir=f"{base}/{name}", seed=0,
        gt_cache=gt_cache, resume_from=resume, full_confirm=True)
    t = prog.sealed_ticket()
    if t["ticket_hash"] != expected[name]:
        raise SystemExit(f"{name} hash drift: {t['ticket_hash']} != {expected[name]}")
    path = repo_ticket_dir / f"{name}.json"
    path.write_text(json.dumps(t, indent=2, sort_keys=True) + "\n")
    print(name, t["ticket_hash"], path)

ticket("smoke_birth_off", 821, 30.0, False)
ticket("smoke_birth_lane_on", 821, 30.0, True)
ticket("full_birth_off", 946, 130.0, False)
ticket("full_birth_lane_on", 946, 130.0, True)
PY
```

## MAIN Smoke Scripts

Dry-run both smoke tickets first. A dry-run writes only a launcher receipt under the target output directory and does not start training.

```bash
.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/smoke_birth_off.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/smoke_birth_off \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 smoke birth OFF" \
  --dry-run

.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/smoke_birth_lane_on.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/smoke_birth_lane_on \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 smoke birth lane ON" \
  --dry-run
```

Fire the bounded smokes only if both dry-runs pass all gates. These are MAIN-run commands, not TP1-executed commands:

```bash
.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/smoke_birth_off.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/smoke_birth_off \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 smoke birth OFF"

.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/smoke_birth_lane_on.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/smoke_birth_lane_on \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 smoke birth lane ON"
```

Smoke pass criteria:

- Both launcher receipts show `seal_freshness=PASS`, `venv_custody_gate0=PASS`, `scorer_slot=FREE`, and `memory_free_gib >= 25.6`.
- OFF argv contains no `--tr1-birth-*` flags. ON argv contains all five BI1 flags above.
- ON telemetry contains `tr1_birth_seed_init`; OFF telemetry does not.
- Both write resumable checkpoints under distinct run directories. The first resume may state parent optimizer moments are absent; subsequent checkpoints must preserve optimizer state because `--persist-optimizer-state on` is active.
- No adoption verdict is allowed from the smokes. At most they clear implementation and gross-regression gates.

## Full Fire Commands

After smokes pass, after MAIN owns the relevant slot, and after the missing PE3/cheapdct consumer question is either built or explicitly waived for a BI1-only A/B, dry-run then fire:

```bash
.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/full_birth_off.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/full_birth_off \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 full birth OFF"

.venv/bin/python tools/launch_tr1_run.py \
  --ticket .omx/research/ddm_tp1_20260805/tickets/full_birth_lane_on.json \
  --cwd /Users/adpena/Projects/pact \
  --out-dir /Volumes/VertigoDataTier/pact/ddm_tp1_20260805/full_birth_lane_on \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_02/checkpoints/stage_seg_trunk_tau_final.npz \
  --purpose "TP1 full birth lane ON"
```

The historical b4s window_03 OFF run remains evidence, but the clean TP1 full A/B should use the sealed OFF ticket above because it shares optimizer-state persistence and current-code custody with ON.

## Gates And Falsifiers

R8 pose gate: TP1 does not run a pose scorer. The first score-bearing successor must use n32 before n600 and must reject immediately if pose-term erosion exceeds `0.005` or if projected score cannot beat `0.7539807296911207`. This is an admission gate, not a score claim.

BI1 birth falsifier: if ON has no seed telemetry, seeded token support is absent, or seeded support is immediately unanchored, mark BI1 implementation failure before any scorer A/B. If ON survives implementation but fails matched-epoch/matched-byte n32 against OFF, close this seed/amplify formulation at INSTANCE scope only, not the birth-completion family.

PE3 falsifier: any TP1 successor that uses PE3 labels as a replacement target is REFUSED by LC1. PE3 may re-enter only as conditioning, with a learned/trusted gate and explicit overclaim safeguards for Road/Lane side assignment.

cheapdct4 falsifier: any successor claiming `pose-carrying base` while only storing OD9 coefficients outside the TR1 descent loop is REFUSED. Until a trainer consumer exists, cheapdct4 is an archive/receiver asset, not an in-loop pose account.

Memory preflight: the launcher floor is anchor-derived, not guessed: `12.8 GiB measured T2 peak RSS x 2.0 safety = 25.6 GiB`. A REFUSE is a valid result.

## Owed Build Table

| Item | Status | Owner / fire order |
|---|---|---|
| BI1 birth seed/amplify ON/OFF | FIRED as packet scripts only; not run by TP1 | MAIN runs dry-runs, bounded smokes, then full A/B. |
| PE3 conditioning slot | QUEUED-WITH-FIRE-ORDER | Build a trainer args-only consumer such as `--pe3-conditioning-cache` and `--pe3-conditioning-mode conditioning_only`; no label replacement loss. Fire after BI1 smoke if operator wants the full crossed object. |
| cheapdct4 pose-carriage in-loop accounting | QUEUED-WITH-FIRE-ORDER | Build a trainer/receiver consumer for OD9 `stage2_qcoeffs` or a pose-accounting hook; do not call OD9 storage joint descent until consumed during training or admission. |
| #924 existence-hinge A/B | QUEUED-WITH-FIRE-ORDER | Trainer flags exist; run after the BI1 smoke or at the frame_0-repaired base boundary so hinge effects do not confound the first BI1 seed/amplify read. |
| direct PE3 target shipping | FOLDED | LC1 n32 ideal target labels worsened all 32 pairs. |
| OD9 cheapdct4 as standalone shipping row | FOLDED-AS-STANDALONE for TP1 | Retained only as routed pose-carriage asset. |

## NEXT_IF_RESUMED

1. Materialize the four tickets with the command above and verify hashes.
2. Run governed dry-runs for the two smoke tickets.
3. If dry-runs pass, MAIN fires the two bounded smokes.
4. Read `launch_receipt.json`, `tr1_window_receipt.json`, and telemetry for the smoke pass criteria.
5. If smoke clears, decide whether to build PE3/cheapdct consumers before the full A/B or explicitly fire the BI1-only A/B as a narrower row.

```json
{
  "schema": "ddm_tp1_next_if_resumed.v1",
  "score_claim": false,
  "scorer_runs_by_tp1": 0,
  "launches_by_tp1": 0,
  "primary_next_action": "MAIN materializes tickets, runs governed dry-runs, then bounded BI1 ON/OFF smokes if gates pass.",
  "blocked_full_claim": [
    "PE3 conditioning slot has no trainer consumer flag",
    "cheapdct4 pose carriage has no TR1 in-loop consumer flag"
  ],
  "current_own_vehicle": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]",
  "contest_pointer": "borrowed/unmoved"
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
