"""Retained current-pointer semantic composition; scorer-free, MAIN owns evaluation.

Each small stage resumes from immutable payloads. No live or predecessor tree is
written. B is a pricing candidate until MAIN supplies its new scorer receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

import brotli
import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from experiments import ddm_sm1_semantic_bound as base
from experiments import ddm_sm1_semantic_mixer_codec as codec
from experiments import ddm_sm1_stage as public
from tac.candidate_seal import measure_runtime_digest

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cmp2_compose")
SM1 = Path("/Volumes/VertigoDataTier/pact/ddm_sm1_semantic_shared_mixer")
MOVE = Path("/Volumes/VertigoDataTier/pact/ddm_fe1_frame_embedding_predistortion/handoff/pair331_move.json")
SOURCE = ROOT / "source_runtime"
LIVE = Path("/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/candidate_runtime")
AXIS = base.AXIS


def guard():
    inputs = json.loads((ROOT / "INPUTS.json").read_text())
    pointer = json.loads(base.POINTER.read_text())
    for key in ("effective_frontier", "our_local_frontier_contest_cuda"):
        if pointer[key]["archive_sha256"] != inputs["source_archive"]["sha256"]:
            raise RuntimeError("pointer moved: rebase into a new retained generation before staging/sealing")
    for fact in inputs["source_files"]:
        public.checked(fact)
    if shutil.disk_usage(ROOT).free < 256 * 1024**2:
        raise RuntimeError("storage preflight: 256 MiB SSD free required")
    return inputs, pointer["our_local_frontier_contest_cuda"]


def configure_public():
    public.ROOT, public.SOURCE = ROOT, SOURCE
    public.CANDIDATE = ROOT / "candidate_runtime"
    public.inputs = guard
    public.stage = staged


def init():
    inputs = base.initialize(ROOT, LIVE)
    guard()
    configure_public()
    parts, _ = base.sections(SOURCE / "archive.zip")
    member = b"".join(parts.values())
    rebuilt = public.archive_bytes(member)
    base.retain(ROOT / "retained/null/archive.zip", rebuilt)
    if rebuilt != (SOURCE / "archive.zip").read_bytes():
        raise RuntimeError("null build is not byte-identical")
    _, rc1, template = base.load_source(ROOT)
    body = rc1.restore_semantic((ROOT / "retained/source/rider.rc1s").read_bytes(), template)
    base.retain(ROOT / "retained/source/body.sm3r", body)
    if body != (SM1 / "retained/source/body.sm3r").read_bytes():
        raise RuntimeError("semantic source moved from SM1; do not transfer fitted weights")
    race = json.loads((SM1 / "RACE.json").read_text())
    winner = race["winner"]
    old_container = public.checked(winner["semantic_container"])
    if len(parts["semantic"]) - len(old_container) != 263:
        raise RuntimeError("263-byte source control failed")
    handoff = json.loads(MOVE.read_text())
    old_archive = Path(
        "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass4/candidate_runtime/archive.zip"
    )
    # The handoff SHA, not an equal-length candidate, selects the source receipt.
    matches = [
        p
        for p in MOVE.parent.parent.rglob("archive.zip")
        if p.stat().st_size == handoff["PRICE_ON_MOVE_35_FOR_REFERENCE_ONLY"]["archive_bytes"]
        and base.fact(p)["sha256"] == handoff["PRICE_ON_MOVE_35_FOR_REFERENCE_ONLY"]["archive_sha256"]
    ]
    if not matches or old_archive.stat().st_size - matches[0].stat().st_size != 61:
        raise RuntimeError("61-byte historical source control failed")
    layout = base.layout_tools.parse_sm3r_mixed(body, template)
    field = next(f for f in layout.fields if f.name == "frame_embed.weight" and f.kind == "codes")
    values = base.layout_tools.unpack_signed_codes(body[field.start : field.stop], field.count, field.bits)
    move = handoff["THE_MOVE"]
    if (move["pair"], move["dim"], move["old_code"], move["new_code"], field.bits) != (331, 6, -1, 0, 3):
        raise RuntimeError("unexpected move geometry")
    index = 331 * 8 + 6
    if values[index] != -1:
        raise RuntimeError("old code differs on this source")
    values[index] = 0
    if values.reshape(600, 8)[331].tolist() != move["full_row_after"]:
        raise RuntimeError("intended row differs")
    changed = body[: field.start] + base.layout_tools.pack_signed_codes(values, field.bits) + body[field.stop :]
    base.retain(ROOT / "retained/B/body.sm3r", changed)
    base.save(
        ROOT / "PLAN.json",
        dict(
            axis=AXIS,
            score_claim=False,
            seed=base.SEED,
            expected_A_bytes=inputs["source_archive"]["bytes"] - 263,
            expected_A_saving_interval=[233, 293],
            expected_B_additional_saving_interval=[0, 80],
            null_identity=True,
            source_263=dict(base_bytes=len(parts["semantic"]), candidate=winner["semantic_container"]),
            source_61=dict(base=base.fact(old_archive), candidate=base.fact(matches[0])),
            move=base.fact(MOVE),
            sm1_race=base.fact(SM1 / "RACE.json"),
            weights=winner["weights"],
            code=base.fact(Path(__file__)),
            codec=base.fact(Path(codec.__file__)),
            B_admission="OWED: new parse-back n600 equal flipped cells and own-instrument pose base/re-solve; MAIN lane",
            retention="all candidates and checkpoints retained atomically on owned SSD; no deletion",
        ),
    )
    return json.loads((ROOT / "PLAN.json").read_text())


def weights_for(sample):
    w = np.asarray(json.loads((ROOT / "PLAN.json").read_text())["weights"], dtype=np.int8)
    if sample:
        rng = np.random.default_rng(base.SEED + sample)
        positions = rng.choice(24, size=4, replace=False)
        w[positions] += rng.choice(np.asarray([-1, 1], dtype=np.int8), size=4)
    return w


def encode(build, sample, trial):
    guard()
    plan = json.loads((ROOT / "PLAN.json").read_text())
    public.checked(plan["codec"])
    work = ROOT / "retained" / build / f"sample_{sample}" / f"trial_{trial}"
    done = work / "ENCODE.json"
    if done.exists():
        row = json.loads(done.read_text())
        for item in row["artifacts"]:
            public.checked(item)
        return row
    _, _, template = base.load_source(ROOT)
    source_path = ROOT / ("retained/source/body.sm3r" if build == "A" else "retained/B/body.sm3r")
    body = source_path.read_bytes()
    weights = weights_for(sample)
    # Observe actual integer frequency decisions at the encoder boundary. These
    # are empirical code lengths, not an order-zero histogram model.
    events = []
    original = codec.rc1._RangeEncoder

    class Observed(original):
        def encode(self, low, freq, total):
            events.append(freq)
            return super().encode(low, freq, total)

    codec.rc1._RangeEncoder = Observed
    try:
        rider, payload, metadata = codec.encode(body, template, weights)
    finally:
        codec.rc1._RangeEncoder = original
    artifacts = [
        base.retain(work / name, value)
        for name, value in (
            ("rider.sm1s", rider),
            ("range.bin", payload),
            ("metadata.bin", metadata),
            ("weights.int8", weights.tobytes()),
        )
    ]
    stream = io.BytesIO()
    np.save(stream, np.asarray(events, dtype=np.uint16), allow_pickle=False)
    artifacts.append(base.retain(work / "actual_frequencies.npy", stream.getvalue()))
    decoded = codec.restore_semantic(rider, template)
    artifacts.append(base.retain(work / "decoded.sm3r", decoded))
    if decoded != body:
        raise RuntimeError("exact semantic decode differs")
    if trial == 2:
        twin = work.parent / "trial_1/rider.sm1s"
        if twin.read_bytes() != rider:
            raise RuntimeError("twin encoding differs")
    row = dict(
        axis=AXIS,
        score_claim=False,
        build=build,
        sample=sample,
        trial=trial,
        source=base.fact(source_path),
        weights=weights.tolist(),
        artifacts=artifacts,
        payload_bytes=len(payload),
        body_identical=True,
        twins_identical=trial == 2,
    )
    base.save(done, row)
    return row


def sweep(build, sample):
    guard()
    configure_public()
    work = ROOT / "retained" / build / f"sample_{sample}"
    target = work / "SWEEP.json"
    if target.exists():
        row = json.loads(target.read_text())
        for variant in row["variants"]:
            public.checked(variant["archive"])
            public.checked(variant["semantic"])
        return row
    a, b = [json.loads((work / f"trial_{trial}/ENCODE.json").read_text()) for trial in (1, 2)]
    if not b["twins_identical"]:
        raise RuntimeError("twins required before sweep")
    rider = (work / "trial_1/rider.sm1s").read_bytes()
    if rider != (work / "trial_2/rider.sm1s").read_bytes():
        raise RuntimeError("twin stream drift")
    parts, header = base.sections(SOURCE / "archive.zip")
    variants = []
    for ck2 in (True, False):
        outer = base.layout_tools.ck2_interleave(rider) if ck2 else rider
        base.retain(work / f"container_input_{ck2}.bin", outer)
        for q in (9, 10, 11):
            for win in (16, 18, 20, 22, 24):
                dest = work / f"container_{ck2}_{q}_{win}"
                blob = brotli.compress(outer, quality=q, lgwin=win)
                bf = base.retain(dest / "semantic.br", blob)
                if brotli.decompress(blob) != outer:
                    raise RuntimeError("container roundtrip failed")
                h = list(header)
                h[6] = len(blob)
                h[4] = (h[4] | 2) if ck2 else (h[4] & ~2)
                member = public.HEADER.pack(*h) + parts["hpac"] + blob + parts["carrier"] + parts["tail"]
                archive = public.archive_bytes(member)
                af = base.retain(dest / "archive.zip", archive)
                check, _ = base.sections(Path(af["path"]))
                if any(check[n] != parts[n] for n in ("hpac", "carrier", "tail")):
                    raise RuntimeError("nonsemantic section changed")
                variants.append(dict(build=build, sample=sample, ck2=ck2, q=q, lgwin=win, archive=af, semantic=bf))
    # Byte identity pins the chosen stream even when lengths tie.
    winner = min(variants, key=lambda r: (r["archive"]["bytes"], r["archive"]["sha256"]))
    row = dict(
        axis=AXIS,
        score_claim=False,
        variants=variants,
        winner=winner,
        twin_encodes=[base.fact(work / f"trial_{trial}/ENCODE.json") for trial in (1, 2)],
    )
    base.save(target, row)
    return row


def stage_a():
    """Stage the smallest admissible lossless A; B cannot bypass scorer debt."""
    inputs, _ = guard()
    configure_public()
    target = ROOT / "STAGED.json"
    if target.exists():
        return staged()
    rows = [sweep("A", sample) for sample in range(5)]
    b = sweep("B", 0)
    winner = min((r["winner"] for r in rows), key=lambda r: (r["archive"]["bytes"], r["archive"]["sha256"]))
    if winner["archive"]["bytes"] >= inputs["source_archive"]["bytes"]:
        raise RuntimeError("no smaller A candidate")
    dest = ROOT / "candidate_runtime"
    patches = public.patched_files()
    archive = public.checked(winner["archive"])
    for src in sorted(SOURCE.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(SOURCE).as_posix()
        if rel in ("archive.zip", "MANIFEST.sha256"):
            continue
        value = patches.get(rel, src.read_bytes())
        if rel == "inflate.py":
            text = value.decode()
            text, n = re.subn(
                r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
                'ARCHIVE_SHA256 = "' + hashlib.sha256(archive).hexdigest() + '"',
                text,
                flags=re.M,
            )
            text, m = re.subn(r"^ARCHIVE_BYTES = [0-9_]+$", f"ARCHIVE_BYTES = {len(archive)}", text, flags=re.M)
            if (n, m) != (1, 1):
                raise RuntimeError("archive pin seam moved")
            value = text.encode()
        base.retain(dest / rel, value)
    base.retain(dest / "runtime/sm1_semantic_mixer.py", Path(codec.__file__).read_bytes())
    base.retain(dest / "archive.zip", archive)
    manifest = "".join(
        f"{base.fact(p)['sha256']}  {p.relative_to(dest)}\n"
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "archive.zip"
    )
    base.retain(dest / "MANIFEST.sha256", manifest.encode())
    changed = [
        p.relative_to(dest).as_posix()
        for p in sorted(dest.rglob("*"))
        if p.is_file()
        and (
            not (SOURCE / p.relative_to(dest)).exists() or p.read_bytes() != (SOURCE / p.relative_to(dest)).read_bytes()
        )
    ]
    if set(changed) != set(patches) | {"archive.zip", "inflate.py", "MANIFEST.sha256", "runtime/sm1_semantic_mixer.py"}:
        raise RuntimeError("unexpected runtime census")
    old, _ = base.sections(SOURCE / "archive.zip")
    new, _ = base.sections(dest / "archive.zip")
    census = {
        n: dict(bytes=len(new[n]), identical=new[n] == old[n], sha256=hashlib.sha256(new[n]).hexdigest()) for n in new
    }
    # Actual per-symbol decision cost includes the changed symbol and its causal
    # adaptation downstream; archive-level price still includes Brotli effects.
    fa, fb = [
        np.load(ROOT / f"retained/{tag}/sample_0/trial_1/actual_frequencies.npy", allow_pickle=False).astype(np.float64)
        for tag in ("A", "B")
    ]
    costs_a, costs_b = -np.log2(fa / 4096), -np.log2(fb / 4096)
    start = 480 * 4 + (331 * 8 + 6) * 3
    price = dict(
        symbol_bit_start=start,
        symbol_bits=3,
        changed_symbol_delta_bits=float(np.sum(costs_b[start : start + 3] - costs_a[start : start + 3])),
        whole_stream_delta_bits=float(np.sum(costs_b - costs_a)),
        actual_range_delta_bytes=int(
            (ROOT / "retained/B/sample_0/trial_1/range.bin").stat().st_size
            - (ROOT / "retained/A/sample_0/trial_1/range.bin").stat().st_size
        ),
        matched_shape_delta_bytes=next(
            x["archive"]["bytes"] for x in b["variants"] if x["ck2"] and x["q"] == 11 and x["lgwin"] == 24
        )
        - next(x["archive"]["bytes"] for x in rows[0]["variants"] if x["ck2"] and x["q"] == 11 and x["lgwin"] == 24),
        searched_B_minus_A_sample0_bytes=b["winner"]["archive"]["bytes"] - rows[0]["winner"]["archive"]["bytes"],
    )
    guard()
    result = dict(
        axis=AXIS,
        score_claim=False,
        binding={"inputs": base.fact(ROOT / "INPUTS.json"), "plan": base.fact(ROOT / "PLAN.json")},
        archive=base.fact(dest / "archive.zip"),
        runtime_sha256=measure_runtime_digest(dest).sha256,
        winner=winner,
        A_samples=[r["winner"] for r in rows],
        B_pricing=b["winner"],
        symbol_price=price,
        section_census=census,
        runtime_changed_files=changed,
        B_disposition="QUEUED-WITH-A-FIRE-ORDER: MAIN scorer proofs and live pair-331 pose re-solve required; never admitted by carried neutrality",
        net_archive_bytes=len(archive) - inputs["source_archive"]["bytes"],
    )
    base.save(target, result)
    return result


def staged():
    guard()
    row = json.loads((ROOT / "STAGED.json").read_text())
    public.checked(row["archive"])
    if measure_runtime_digest(ROOT / "candidate_runtime").sha256 != row["runtime_sha256"]:
        raise RuntimeError("staged runtime drift")
    return row


def stage_b():
    """Retain B's public reader with the live carrier explicitly unresolved."""
    guard()
    a = staged()
    target = ROOT / "STAGED_B.json"
    if target.exists():
        row = json.loads(target.read_text())
        public.checked(row["archive"])
        if measure_runtime_digest(ROOT / "candidate_B_runtime").sha256 != row["runtime_sha256"]:
            raise RuntimeError("B runtime drift")
        return row
    rows = [sweep("B", sample) for sample in range(5)]
    winner = min((r["winner"] for r in rows), key=lambda r: (r["archive"]["bytes"], r["archive"]["sha256"]))
    archive = public.checked(winner["archive"])
    dest = ROOT / "candidate_B_runtime"
    for src in sorted((ROOT / "candidate_runtime").rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT / "candidate_runtime").as_posix()
        if rel in ("archive.zip", "MANIFEST.sha256"):
            continue
        value = src.read_bytes()
        if rel == "inflate.py":
            text, n = re.subn(
                r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
                'ARCHIVE_SHA256 = "' + hashlib.sha256(archive).hexdigest() + '"',
                value.decode(),
                flags=re.M,
            )
            text, m = re.subn(r"^ARCHIVE_BYTES = [0-9_]+$", f"ARCHIVE_BYTES = {len(archive)}", text, flags=re.M)
            if (n, m) != (1, 1):
                raise RuntimeError("B archive pin seam moved")
            value = text.encode()
        base.retain(dest / rel, value)
    base.retain(dest / "archive.zip", archive)
    manifest = "".join(
        f"{base.fact(p)['sha256']}  {p.relative_to(dest)}\n"
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "archive.zip"
    )
    base.retain(dest / "MANIFEST.sha256", manifest.encode())
    guard()
    row = dict(
        axis=AXIS,
        score_claim=False,
        binding=a["binding"],
        winner=winner,
        B_samples=[r["winner"] for r in rows],
        archive=base.fact(dest / "archive.zip"),
        runtime_sha256=measure_runtime_digest(dest).sha256,
        carrier_status="UNCHANGED LIVE COEFFICIENTS; re-solve and fresh seg/pose proofs owed to MAIN; NOT ADMITTED",
        delta_bytes_vs_best_A=len(archive) - a["archive"]["bytes"],
    )
    base.save(target, row)
    return row


def proof_b():
    """Verify the intended one-scalar difference through the actual public path."""
    row = stage_b()
    configure_public()
    public.ROOT = ROOT / "B_public"
    public.CANDIDATE = ROOT / "candidate_B_runtime"
    public.stage = lambda: row
    base.retain(public.ROOT / "STAGED.json", (ROOT / "STAGED_B.json").read_bytes())
    base.retain(public.ROOT / "retained/source/body.sm3r", (ROOT / "retained/B/body.sm3r").read_bytes())
    result = public.public_probe("candidate", "proof")
    source = json.loads((ROOT / "PROOF_frontier.json").read_text())["tensors"]
    differences = []
    for a, b in zip(source, result["tensors"], strict=True):
        if any(a[k] != b[k] for k in ("name", "shape", "dtype")):
            raise RuntimeError("B public tensor geometry changed")
        av = np.frombuffer(public.checked(a["artifact"]), dtype=a["dtype"]).reshape(a["shape"])
        bv = np.frombuffer(public.checked(b["artifact"]), dtype=b["dtype"]).reshape(b["shape"])
        where = np.argwhere(av != bv)
        if len(where):
            differences.append(
                dict(
                    name=a["name"],
                    coordinates=where.tolist(),
                    old=av[tuple(where[0])].item(),
                    new=bv[tuple(where[0])].item(),
                )
            )
    if (
        len(differences) != 1
        or differences[0]["name"] != "frame_embed.weight"
        or differences[0]["coordinates"] != [[331, 6]]
        or differences[0]["new"] != 0
    ):
        raise RuntimeError("B public tensor differences exceed intended scalar")
    verdict = dict(
        axis=AXIS,
        score_claim=False,
        tensors_compared=38,
        identical_tensors=37,
        differences=differences,
        decoded_body_exact=True,
        public_proof=base.fact(public.ROOT / "PROOF_candidate.json"),
        seg_neutrality="NOT MEASURED: model identity except the intended scalar does not prove flipped-cell neutrality",
        pose="NOT MEASURED: unchanged carrier requires fresh live-base re-solve",
    )
    base.save(ROOT / "PUBLIC_B_TENSOR_DIFFERENCE.json", verdict)
    return verdict


def staged_twins(build):
    """Re-encode using the actual staged reader, retaining twin full archives."""
    guard()
    configure_public()
    row = staged() if build == "A" else stage_b()
    runtime = ROOT / ("candidate_runtime" if build == "A" else "candidate_B_runtime")
    sys.path.insert(0, str(runtime))
    from runtime import sm1_semantic_mixer as shipped

    if Path(shipped.__file__).resolve() != runtime / "runtime/sm1_semantic_mixer.py":
        raise RuntimeError("staged codec import escaped runtime")
    _, _, template = base.load_source(ROOT)
    body = (ROOT / ("retained/source/body.sm3r" if build == "A" else "retained/B/body.sm3r")).read_bytes()
    winner = row["winner"]
    parts, header = base.sections(SOURCE / "archive.zip")
    facts = []
    for trial in (1, 2):
        dest = ROOT / "retained/staged_twins" / build / str(trial)
        rider, payload, metadata = shipped.encode(body, template, weights_for(winner["sample"]))
        for name, data in (("rider.sm1s", rider), ("range.bin", payload), ("metadata.bin", metadata)):
            base.retain(dest / name, data)
        decoded = shipped.restore_semantic(rider, template)
        base.retain(dest / "decoded.sm3r", decoded)
        if decoded != body:
            raise RuntimeError("staged decoder differs")
        outer = base.layout_tools.ck2_interleave(rider) if winner["ck2"] else rider
        base.retain(dest / "container_input.bin", outer)
        blob = brotli.compress(outer, quality=winner["q"], lgwin=winner["lgwin"])
        base.retain(dest / "semantic.br", blob)
        h = list(header)
        h[6] = len(blob)
        h[4] = (h[4] | 2) if winner["ck2"] else (h[4] & ~2)
        member = public.HEADER.pack(*h) + parts["hpac"] + blob + parts["carrier"] + parts["tail"]
        archive = public.archive_bytes(member)
        facts.append(base.retain(dest / "archive.zip", archive))
        if archive != public.checked(row["archive"]):
            raise RuntimeError("staged twin archive differs")
    result = dict(
        axis=AXIS,
        score_claim=False,
        build=build,
        twins=facts,
        byte_identical=True,
        staged_codec=base.fact(runtime / "runtime/sm1_semantic_mixer.py"),
    )
    base.save(ROOT / f"STAGED_TWINS_{build}.json", result)
    return result


def seal_a():
    """Seal only A's proved zero-distortion body; B is expressly excluded."""
    guard()
    configure_public()
    public.combine()
    row = staged()
    for name in ("PUBLIC_TENSOR_IDENTITY.json", "STAGED_TWINS_A.json"):
        proof = json.loads((ROOT / name).read_text())
        if not proof.get("all_tensors_bit_identical", proof.get("byte_identical", False)):
            raise RuntimeError("A identity proof missing")
    _, pointer = guard()
    mirror_path = Path(pointer["source_path"])
    if not mirror_path.is_absolute():
        mirror_path = REPO / mirror_path
    mirror = json.loads(mirror_path.read_text())
    receipt = Path(mirror["source_receipt"])
    if (
        base.fact(receipt)["sha256"] != mirror["source_receipt_sha256"]
        or mirror["archive_sha256"] != pointer["archive_sha256"]
    ):
        raise RuntimeError("source auth receipt does not bind the live base")
    out = ROOT / "SEAL_ddm_cmp2_sm1_fe1_composed.json"
    argv = [
        sys.executable,
        "-B",
        str(REPO / "tools/make_candidate_seal.py"),
        "--candidate-id",
        "ddm_cmp2_sm1_fe1_composed",
        "--runtime-dir",
        str(ROOT / "candidate_runtime"),
        "--axis",
        "contest_cuda",
        "--out",
        str(out),
        "--archive-member",
        "p",
        "--public-entrypoint-smoke",
        str(ROOT / "PUBLIC_ENTRYPOINT_SMOKE.json"),
        "--bound-base-receipt",
        str(receipt),
        "--admit-bar-net-ds",
        "0",
        "--pointer-axis",
        "contest_cuda",
        "--sealed-by",
        "ddm_cmp2",
        "--verify-archive-sha",
        row["archive"]["sha256"],
        "--retained-path",
        str(ROOT / "retained"),
        "--falsifier",
        "Exact S fails to improve on the bound base",
        "--falsifier",
        "Public semantic tensors or hpac/carrier/tail content differs from the source",
        "--notes",
        "Selected A only: SM1 coder with sampled counted weights. FE1 pair-331 code is NOT included; B scorer proofs queued to MAIN. No scorer or Modal call.",
    ]
    for rel in (
        "inflate.py",
        "inflate.sh",
        "runtime/f26_inflate.py",
        "runtime/residual_archive.py",
        "runtime/sm1_semantic_mixer.py",
        "runtime/rc3_shared_mixer.py",
        "runtime/tc1_shared_mixer.py",
        "runtime/tc1_receiver_checkpoint.py",
        "cpr1/inflate.py",
        "cpr1/ddm_mp2_semantic_receiver.py",
    ):
        argv += ["--receiver", rel]
    if not out.exists():
        work, env = public.attempt("seal_A")
        base.save(work / "COMMAND.json", argv)
        process = public.bounded(argv, work, env, "seal")
        base.save(work / "PROCESS.json", process)
        if process["returncode"] != 0:
            raise RuntimeError((work / "seal.stderr").read_text())
    from tac.candidate_seal import validate_seal

    validation = validate_seal(out)
    guard()
    if not validation.ok:
        raise RuntimeError(validation.summary())
    result = dict(
        status="SEAL READY — A ONLY; B VERIFICATION OWED",
        seal=base.fact(out),
        selected_build="A",
        fe1_code_included=False,
        score_claim=False,
        validation=validation.summary(),
    )
    base.save(ROOT / "SEAL_RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "init",
            "encode",
            "sweep",
            "stage",
            "stage-b",
            "proof",
            "proof-b",
            "staged-twins",
            "seal",
            "smoke-candidate",
            "smoke-frontier",
            "combine",
        ),
    )
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--build", choices=("A", "B"), default="A")
    parser.add_argument("--sample", type=int, choices=range(5), default=0)
    parser.add_argument("--trial", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.resume_from.resolve() != ROOT.resolve():
        raise ValueError("resume-from must name owned retained store")
    configure_public()
    if args.stage == "init":
        row = init()
    elif args.stage == "encode":
        row = encode(args.build, args.sample, args.trial)
    elif args.stage == "sweep":
        row = sweep(args.build, args.sample)
    elif args.stage == "stage":
        row = stage_a()
    elif args.stage == "stage-b":
        row = stage_b()
    elif args.stage == "proof":
        row = public.proof()
    elif args.stage == "proof-b":
        row = proof_b()
    elif args.stage == "staged-twins":
        row = staged_twins(args.build)
    elif args.stage == "seal":
        row = seal_a()
    elif args.stage.startswith("smoke-"):
        row = public.public_probe(args.stage.removeprefix("smoke-"), "smoke")
    else:
        row = public.combine()
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
