from __future__ import annotations

from types import SimpleNamespace

from experiments import ddm_up3_carrier_splice as up3


class _ResidualArchive:
    RR5_RESERVED_ARITH_BASIS = 0x08
    DX2_RESERVED_CABAC_COEFFICIENTS = 0x10


def test_entropy_riders_apply_and_restore_in_receiver_inverse_order(monkeypatch) -> None:
    calls: list[str] = []

    def wrap(prefix: bytes, body: bytes) -> bytes:
        return prefix + b"(" + body + b")"

    def unwrap(prefix: bytes, body: bytes) -> bytes:
        assert body.startswith(prefix + b"(") and body.endswith(b")")
        return body[len(prefix) + 1 : -1]

    dx2 = SimpleNamespace(
        apply_cabac_to_carrier_body=lambda body: (
            calls.append("dx2.apply") or {"body": wrap(b"D", body)}
        ),
        restore_carrier_body=lambda body: (
            calls.append("dx2.restore") or unwrap(b"D", body)
        ),
    )
    rr5 = SimpleNamespace(
        apply_rider_to_carrier_body=lambda body: (
            calls.append("rr5.apply") or {"body": wrap(b"R", body)}
        ),
        restore_carrier_body=lambda body: (
            calls.append("rr5.restore") or unwrap(b"R", body)
        ),
    )
    monkeypatch.setattr(
        up3,
        "_import_entropy_riders",
        lambda _runtime, *, require_rr5, require_dx2: (
            rr5 if require_rr5 else None,
            dx2 if require_dx2 else None,
        ),
    )

    encoded = up3._apply_entropy_riders(
        b"carrier",
        reserved=0x18,
        runtime_dir=up3.DEFAULT_RUNTIME,
        residual_archive=_ResidualArchive,
    )

    assert encoded == b"R(D(carrier))"
    assert calls == ["dx2.apply", "rr5.apply", "rr5.restore", "dx2.restore"]
    calls.clear()
    assert (
        up3._restore_entropy_riders(
            encoded,
            reserved=0x18,
            runtime_dir=up3.DEFAULT_RUNTIME,
            residual_archive=_ResidualArchive,
        )
        == b"carrier"
    )
    assert calls == ["rr5.restore", "dx2.restore"]


def test_entropy_rider_helpers_are_identity_when_reserved_bits_are_clear(
    monkeypatch,
) -> None:
    def unexpected_import(*_args, **_kwargs):
        raise AssertionError("a rider module was imported with no rider bit set")

    monkeypatch.setattr(up3, "_import_entropy_riders", unexpected_import)
    for helper in (up3._apply_entropy_riders, up3._restore_entropy_riders):
        assert (
            helper(
                b"carrier",
                reserved=0,
                runtime_dir=up3.DEFAULT_RUNTIME,
                residual_archive=_ResidualArchive,
            )
            == b"carrier"
        )
