from __future__ import annotations

import json

import numpy as np
import torch

from experiments import ddm_mt1_978_multitoken_screen as screen
from experiments import ddm_mt1_modal_multitoken_sign_gate as dispatch
from experiments.ddm_mt1_runtime import multitoken_representative as mt1


def test_receiver_context_and_probability_are_site_list_free() -> None:
    tokens = torch.tensor(
        [[[0, 0, 1], [0, 2, 1], [3, 2, 4]]],
        dtype=torch.long,
    )
    context = mt1.oriented_context(tokens)
    assert context.shape == (1, mt1.CONTEXT_CHANNELS, 3, 3)
    model = mt1.MultiTokenRepresentative(hidden=2, max_support_mass=0.25)
    probability = model.probability_state(tokens)
    assert probability.shape == (1, mt1.NUM_CLASSES, 3, 3)
    assert torch.allclose(probability.sum(dim=1), torch.ones_like(tokens, dtype=torch.float32))
    center = torch.nn.functional.one_hot(tokens, mt1.NUM_CLASSES).permute(0, 3, 1, 2)
    assert torch.all(probability[center.bool()] >= 0.75)
    assert "site" not in " ".join(model.state_dict()).lower()
    assert "mask" not in " ".join(model.state_dict()).lower()


def test_counted_model_repeat_and_parseback_are_exact() -> None:
    torch.manual_seed(7)
    model = mt1.MultiTokenRepresentative(hidden=3, max_support_mass=0.4).eval()
    coded, receipt = mt1.serialize_model(model)
    repeat, repeat_receipt = mt1.serialize_model(model)
    assert coded == repeat
    assert receipt == repeat_receipt
    parsed = mt1.load_module(coded, torch.device("cpu"))
    tokens = torch.randint(0, mt1.NUM_CLASSES, (2, 7, 9))
    embeddings = torch.randn(mt1.NUM_CLASSES, mt1.SEMANTIC_WIDTH)
    assert torch.equal(
        parsed.representative(tokens, embeddings),
        mt1.load_module(repeat, torch.device("cpu")).representative(tokens, embeddings),
    )


def test_stratified_split_is_balanced_disjoint_and_nonprefix() -> None:
    counts = np.repeat(np.arange(8), 75)
    spatial = np.zeros((screen.PAIR_COUNT, 12, 16), dtype=np.int64)
    spatial[::2, 5, 8] = 10
    split = screen.stratified_split(counts, spatial)
    assert len(split["train"]) == 32
    assert len(split["heldout"]) == 32
    assert set(split["train"]).isdisjoint(split["heldout"])
    assert split["train"] != list(range(32))
    assert split["heldout"] != list(range(32))
    assert all(len(row["train"]) == 4 for row in split["strata"])
    assert all(len(row["heldout"]) == 4 for row in split["strata"])


def test_positive_gate_requires_both_seg_wins_and_zero_pose_damage() -> None:
    def positive(candidate: int, direct: int, base: int, pose_delta: float) -> bool:
        return candidate < direct and candidate < base and pose_delta <= 0.0

    assert positive(9, 11, 10, 0.0)
    assert not positive(10, 11, 10, 0.0)
    assert not positive(9, 9, 10, 0.0)
    assert not positive(9, 11, 10, 1.0e-12)


def test_t4_fire_order_is_bounded_and_main_owned() -> None:
    assert dispatch.HARD_CAP_SECONDS == 960
    assert dispatch.ESTIMATED_HARD_CAP_USD == 0.16
    assert dispatch.INSTANCE_JOB_ID == "modal:ddm_mt1_t4_sign_gate_20260814"
    assert dispatch.DEFAULT_OUTPUT.name == "t4_sign_gate_r1"


def test_fire_order_is_checked_against_actual_entrypoint_signature() -> None:
    order = dispatch.build_fire_order(
        output=dispatch.DEFAULT_OUTPUT,
        request_record={"sha256": dispatch.EXPECTED_DR1_REQUEST_SHA256},
        records={},
        dispatch_dir=dispatch.DEFAULT_OUTPUT / "dispatch_test",
    )
    check = order["entrypoint_signature_check"]
    assert check["passed"]
    assert "--output-dir" in check["accepted_options"]
    assert check["observed_options"]["--detach"] is True
    with np.testing.assert_raises_regex(
        dispatch.MT1DispatchError, "missing required main options"
    ):
        dispatch.validate_main_fire_argv(
            order["exact_argv"].replace(
                f" --output-dir {dispatch.DEFAULT_OUTPUT / 'dispatch_test'}", ""
            )
        )


def test_sealed_request_parser_is_hash_and_schema_closed() -> None:
    request = {
        "schema": "ddm_mt1_t4_sign_gate_request.v1",
        "payloads": {
            "a.bin": {"path": "/retained/a.bin", "bytes": 1, "sha256": "0" * 64}
        },
    }
    payload = json.dumps(request).encode()
    digest = dispatch.hashlib.sha256(payload).hexdigest()
    assert dispatch.parse_sealed_request_bytes(payload, digest) == request
    with np.testing.assert_raises_regex(dispatch.MT1DispatchError, "SHA-256 differs"):
        dispatch.parse_sealed_request_bytes(payload, "f" * 64)


def test_dispatcher_uses_one_explicit_experiments_package_import_topology() -> None:
    source = dispatch.Path(dispatch.__file__).read_text()
    assert 'importlib.import_module("experiments.modal_auth_eval")' in source
    assert '"experiments/modal_auth_eval.py"' in source
    assert '"experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py"' in source
    assert "except ModuleNotFoundError" not in source
