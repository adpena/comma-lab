"""ddm_jg4 -- the re-encoder's resume must be bit-faithful, and the guard must bite.

WHAT BROKE.  ``ddm_jg2_tail_reencode`` checkpointed the arithmetic-coder mirror by
calling ``corrector.state_dict()``.  That method lives ONLY on the ``rr4`` base
class; the shipped ``FreeCorrector`` is three subclasses deeper and neither
subclass overrides it.  So the checkpoint saved 7 arrays and silently dropped 90 --
the 4000x13 logistic-mixer ``weights``, ``sse_weight``, the ``ma1`` within-miss
tables, and all 39 arrays owned by the 13 ``MixerFamily`` members.  A resumed run
restarted the model-mixing half COLD and emitted a stream that was not the one a
straight-through encode produces, with no error anywhere.  Two 600-frame controls
were spent on that: they diverged from the shipped stream at exactly their own
resume frame (75 -> +40 B, ~275 -> +127 B) and nowhere earlier.

WHAT THIS FILE PINS.  Two things, because pinning only the cure would be vacuous:

  * ``test_resume_is_byte_identical_to_straight_through`` -- the CURE.  Encode N
    frames straight through; encode the same N with a checkpoint at N/2 and a
    resume across it; require the two streams to be byte-identical AND the two
    per-frame bit ledgers to be equal.  This is the falsifier that would have
    caught the defect before it cost a control.
  * ``test_detector_fires_on_the_v1_capture`` -- the POSITIVE CONTROL.  Re-create
    v1's key set and require ``uncaptured_divergent_state`` to name the arrays it
    drops.  Without this, a detector that always returned ``[]`` would pass the
    cure test forever.  (Sister discipline: "vacuity == pass" -- report the
    denominator, never let a silent instrument read green.)

The cheap tests here need no runtime tree.  The end-to-end resume test needs the
shipped runtime plus a C compiler, so it is marked ``slow`` and skips when the
body is not mounted, rather than failing on a machine that has no SSD attached.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "experiments" / "ddm_jg2_tail_reencode.py"

#: The body whose decoder the mirror inverts.  Any sibling body works; this is the
#: one the ``ddm_jg4`` control was proved on.
RUNTIME_ROOT = Path(
    os.environ.get(
        "TAC_JG4_RUNTIME_ROOT", "/Volumes/APDataStore/pact/ddm_br1/candidate_runtime_r1"
    )
)
TOKENS = Path(
    os.environ.get(
        "TAC_JG4_TOKENS",
        "/Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated"
        "/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8",
    )
)

#: Frames the resume test encodes.  Small enough to stay a test (~30 s), large
#: enough that the temporal and mixing feedback paths are all live: the corrector
#: only gains a previous frame at frame 1, and the families need several frames of
#: counts before their multipliers leave 1.0.
RESUME_FRAMES = 12


def _load_tool():
    spec = importlib.util.spec_from_file_location("ddm_jg4_tool_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _require_body() -> None:
    if not (RUNTIME_ROOT / "archive.zip").is_file():
        pytest.skip(f"runtime body not mounted at {RUNTIME_ROOT}")
    if not TOKENS.is_file():
        pytest.skip(f"decoded token field not mounted at {TOKENS}")
    if shutil.which("cc") is None:  # pragma: no cover - every dev box has cc
        pytest.skip("no C compiler for the RC64 encoder")


@pytest.fixture(scope="module")
def corrector():
    _require_body()
    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_ROOT))
    try:
        from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - body present but unimportable
        pytest.skip(f"shipped runtime not importable: {exc}")
    return FreeCorrector


# --------------------------------------------------------------------------------------
# The structural walk.
# --------------------------------------------------------------------------------------


def test_state_names_sees_slots_and_dict(tool):
    """``vars()`` alone cannot see the base class -- that is the whole defect."""

    class Base:
        __slots__ = ("slotted",)

    class Child(Base):
        def __init__(self) -> None:
            self.slotted = np.zeros(3)
            self.dicted = np.ones(4)

    child = Child()
    names = tool.state_names(child)
    assert "slotted" in names, "a __slots__ attribute must be visible"
    assert "dicted" in names, "a __dict__ attribute must be visible"
    assert "slotted" not in vars(child), "guard: the walk is not just vars()"


def test_capture_covers_every_corrector_array(corrector, tool):
    """The capture must reach the base slots, the subclass dicts, and the families."""
    live = corrector(384 * 512)
    state = tool.corrector_state(live)

    for name in tool.state_names(live):
        if name == "families":
            continue
        if isinstance(getattr(live, name, None), np.ndarray):
            assert f"self.{name}" in state, f"self.{name} is not captured"
    for index, family in enumerate(live.families):
        for name in tool.state_names(family):
            if isinstance(getattr(family, name, None), np.ndarray):
                assert f"fam.{index:02d}.{name}" in state, f"family {index} {name} lost"

    # The 7-key v1 capture is a strict, and small, subset of the real state.
    assert len(state) > len(live.state_dict()) * 5


def test_detector_fires_on_the_v1_capture(corrector, tool):
    """POSITIVE CONTROL: the detector must name what v1 dropped, or it is vacuous."""
    live = corrector(384 * 512)
    cold = corrector(384 * 512)

    # Nothing has moved yet, so nothing can be lost -- by either capture.
    assert tool.uncaptured_divergent_state(live, cold, set(tool.corrector_state(live))) == []
    v1_keys = {f"self.{key}" for key in live.state_dict()}
    assert tool.uncaptured_divergent_state(live, cold, v1_keys) == []

    # Move one array in each layer v1 forgets: the fx2 mixer, the ma1 miss law, and
    # a MixerFamily count table.
    live.weights[:] += 1
    live._miss_seen[:] += 1
    live.families[0].hits[:] += 1

    lost = tool.uncaptured_divergent_state(live, cold, v1_keys)
    assert "self.weights" in lost
    assert "self._miss_seen" in lost
    assert "fam.00.hits" in lost

    # The v2 capture loses none of it.
    assert tool.uncaptured_divergent_state(live, cold, set(tool.corrector_state(live))) == []


def test_state_round_trips_into_a_fresh_corrector(corrector, tool):
    live = corrector(384 * 512)
    live.weights[:] += 3
    live.counts[:] += 5
    live._miss_counts[:] += 7
    live.families[2].phat_q[:] += 11
    live.have_prev = True

    fresh = corrector(384 * 512)
    tool.load_corrector_state(fresh, tool.corrector_state(live))

    before = tool.corrector_state(live)
    after = tool.corrector_state(fresh)
    assert set(before) == set(after)
    for key in before:
        assert np.array_equal(before[key], after[key]), key
    assert fresh.have_prev is True


def test_legacy_checkpoint_schema_is_refused(tool):
    """A v1 checkpoint is a wrong answer, not a slow one: it must refuse, not resume."""
    assert tool.CHECKPOINT_SCHEMA.endswith(".v2")
    assert set(tool.LEDGER_KEYS) == {
        "schema", "frame", "code_bits", "per_frame", "previous",
    }
    # Namespaced state keys can never collide with the ledger keys.
    assert all("." not in key for key in tool.LEDGER_KEYS)


# --------------------------------------------------------------------------------------
# The end-to-end falsifier.
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_resume_is_byte_identical_to_straight_through(tool, tmp_path):
    """THE falsifier the defect slipped past.

    Encoding N frames in one pass and encoding them across a checkpoint/resume must
    produce the same bytes.  Anything else means the resume restores less state than
    the encode consumes, and every byte delta measured after a resume is fiction.
    """
    _require_body()

    def run(store: Path, checkpoint_every: int, resume: bool) -> tuple[bytes, np.ndarray]:
        argv = [
            "--stage", "control",
            "--store", str(store),
            "--runtime-root", str(RUNTIME_ROOT),
            "--tokens", str(TOKENS),
            "--frames", str(RESUME_FRAMES),
            "--checkpoint-every", str(checkpoint_every),
        ]
        if resume:
            argv.append("--resume")
        assert tool.main(argv) == 0
        work = store / "work"
        stream = (work / f"tail_control_{RESUME_FRAMES}.bin").read_bytes()
        ledger = np.load(work / f"bits_per_frame_control_{RESUME_FRAMES}.npy")
        return stream, ledger

    straight_store = tmp_path / "straight"
    straight, straight_ledger = run(straight_store, checkpoint_every=0, resume=False)

    # Halt at the midpoint, then resume across it.
    split_store = tmp_path / "split"
    run(split_store, checkpoint_every=RESUME_FRAMES // 2, resume=False)
    resumed, resumed_ledger = run(split_store, checkpoint_every=RESUME_FRAMES // 2, resume=True)

    assert resumed == straight, (
        f"resume is not bit-faithful: {len(resumed)} B vs {len(straight)} B straight "
        "through -- the checkpoint restores less state than the encoder consumes"
    )
    assert np.array_equal(resumed_ledger, straight_ledger), (
        "per-frame bit ledgers differ, so the CODING ROWS differ: the corrector's "
        "adaptive state, not just the coder interval, was lost across the resume"
    )
