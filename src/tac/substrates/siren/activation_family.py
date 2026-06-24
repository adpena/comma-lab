# SPDX-License-Identifier: MIT
"""Typed activation-family contract for SIREN/INR substrate variants.

The full SIREN substrate keeps one archive/trainer contract while allowing
small activation-family probes. These modes are not score claims and do not
imply that FINER, WIRE, BACON, Gauss, HOSC, Sinc, etc. have been faithfully
reimplemented as complete paper architectures. They are byte-closed INR
activation modes: the selected family is serialized in SRV1 metadata and
consumed by inflate.

NO-FAKE CONTRACT (the supreme rule): every activation branch below applies the
REAL functional form of the named technique to the actual preactivation tensor.
"siren" is exactly ``torch.sin(omega * x)`` so the legacy/vendored path is
BIT-IDENTICAL (the control arm). Any non-"siren" family GENUINELY differs from
``sin`` (the parity tests fail if a swap degenerates to a no-op), and every
hyperparameter actually modulates the output (the tests assert hyperparam
sensitivity). The fixed-form families add ZERO parameters (byte-neutral screen);
the LEARNABLE families (``step_basis``, ``fkan``) are :class:`torch.nn.Module`
sub-activations that carry their own REAL parameters (a small, reported byte
delta — they "learn our own nonlinearity").

WHY THESE FOR THE ARGMAX-EDGE OBJECTIVE: our target is NOT a generic image
(PSNR). It is the frozen-SegNet argmax STEP at lane boundaries, scored on
argmax-flip d_seg through a uint8 round-trip. Sinusoids ring on steps (Gibbs);
an edge/step-matched nonlinearity may beat the variable-frequency FINER win.
The candidates here are ranked by EXPECTED d_seg fit = (a) sharp-step
representation at LOW capacity, (b) survival of the uint8->argmax round-trip,
(c) composition with FINER's variable-frequency win. See the survey memo
``.omx/research/nextgen_inr_activations_survey_and_custom_designs_20260624.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

ActivationFamilyId = Literal[
    "siren",
    "finer",
    "wire",
    "bacon",
    # --- literature next-gen INR activations (fixed-form, byte-neutral) ----
    "gauss",
    "hosc",
    "sinc",
    "rcgauss",
    # --- custom designs ---------------------------------------------------
    "finer_gauss",  # custom #1: FINER variable-freq sine x Gaussian envelope
    "step_basis",   # custom #2: learnable sum-of-K shifted/scaled tanh soft-steps
    "fkan",         # custom #3: learnable Fourier-series (FKAN/SineKAN-style)
]

DEFAULT_ACTIVATION_FAMILY: ActivationFamilyId = "siren"

# Fixed-form (stateless, byte-neutral) families consumable directly by
# ``apply_activation_family``. The learnable families live in module form.
FIXED_FORM_ACTIVATION_IDS: tuple[ActivationFamilyId, ...] = (
    "siren",
    "finer",
    "wire",
    "bacon",
    "gauss",
    "hosc",
    "sinc",
    "rcgauss",
    "finer_gauss",
)
# Learnable (parametric) families — instantiate ``make_learnable_activation``.
LEARNABLE_ACTIVATION_IDS: tuple[ActivationFamilyId, ...] = (
    "step_basis",
    "fkan",
)
ACTIVATION_FAMILY_IDS: tuple[ActivationFamilyId, ...] = (
    *FIXED_FORM_ACTIVATION_IDS,
    *LEARNABLE_ACTIVATION_IDS,
)


@dataclass(frozen=True)
class ActivationFamilySpec:
    """Human-readable activation-family metadata for manifests."""

    activation_id: ActivationFamilyId
    summary: str
    archive_note: str
    full_paper_architecture: bool
    learnable: bool = False


SIREN_ACTIVATION_FAMILIES: dict[ActivationFamilyId, ActivationFamilySpec] = {
    "siren": ActivationFamilySpec(
        activation_id="siren",
        summary="Canonical SIREN sine activation sin(omega*x) (Sitzmann 2020).",
        archive_note="Default; preserves naked_siren_replacement behavior (byte-identical).",
        full_paper_architecture=True,
    ),
    "finer": ActivationFamilySpec(
        activation_id="finer",
        summary="FINER variable-periodic sine sin(omega*(|x|+1)*x) (Liu 2024).",
        archive_note="Drop-in activation probe under the same SRV1 payload contract.",
        full_paper_architecture=False,
    ),
    "wire": ActivationFamilySpec(
        activation_id="wire",
        summary="WIRE real-Gabor sin(omega*x)*exp(-0.5*(s*x)^2) (Saragadam 2023).",
        archive_note="Drop-in activation probe; window scale is carried in metadata.",
        full_paper_architecture=False,
    ),
    "bacon": ActivationFamilySpec(
        activation_id="bacon",
        summary="BACON-style band-limited sine schedule (Lindell 2022).",
        archive_note=(
            "Band-limited activation probe; not BACON's full multi-output "
            "network architecture."
        ),
        full_paper_architecture=False,
    ),
    "gauss": ActivationFamilySpec(
        activation_id="gauss",
        summary="Gaussian activation exp(-(s*x)^2) (Ramasinghe-Lucey 2022).",
        archive_note=(
            "Drop-in activation probe; the Gaussian SCALE s reuses wire_scale "
            "(s controls spatial/frequency width). Localized in BOTH space and "
            "frequency — a smooth bump, NOT a sinusoid (no Gibbs ringing)."
        ),
        full_paper_architecture=False,
    ),
    "hosc": ActivationFamilySpec(
        activation_id="hosc",
        summary="HOSC hyperbolic-oscillating tanh(beta*sin(omega*x)) (Serrano 2024).",
        archive_note=(
            "Drop-in activation probe; beta (hosc_beta) is the SATURATION/sharpness "
            "control. As beta->inf the sine SQUARE-WAVES into a sharp step train — "
            "the natural piecewise-constant carrier for an argmax partition."
        ),
        full_paper_architecture=False,
    ),
    "sinc": ActivationFamilySpec(
        activation_id="sinc",
        summary="Sinc activation sin(omega*x)/(omega*x) (Saratchandran 2024).",
        archive_note=(
            "Drop-in activation probe; the cardinal-sine is the ideal "
            "band-limited interpolation kernel (Shannon sampling)."
        ),
        full_paper_architecture=False,
    ),
    "rcgauss": ActivationFamilySpec(
        activation_id="rcgauss",
        summary="FLAIR-style band-localized sinc*raised-cosine*Gaussian (Lee 2025).",
        archive_note=(
            "Drop-in activation probe approximating FLAIR's RC-GAUSS basis: a "
            "sinc carrier shaped by a raised-cosine window times a Gaussian "
            "envelope (omega=center freq, wire_scale=Gaussian width, fixed "
            "roll-off 0.05). Joint frequency-selection + spatial-localization "
            "under the time-frequency uncertainty bound."
        ),
        full_paper_architecture=False,
    ),
    "finer_gauss": ActivationFamilySpec(
        activation_id="finer_gauss",
        summary="CUSTOM #1: FINER variable-freq sine x Gaussian envelope.",
        archive_note=(
            "ORIGINAL hybrid (ours): sin(omega*(|x|+1)*x) * exp(-0.5*(s*x)^2). "
            "Composes FINER's confirmed variable-frequency d_seg win (high local "
            "frequency near edges) with WIRE/Gauss spatial localization — keeps "
            "FINER's edge-adaptivity while damping the off-edge Gibbs tail that "
            "may flip distant argmax cells. Byte-neutral."
        ),
        full_paper_architecture=False,
    ),
    "step_basis": ActivationFamilySpec(
        activation_id="step_basis",
        summary="CUSTOM #2: learnable sum-of-K shifted/scaled tanh soft-steps.",
        archive_note=(
            "ORIGINAL learnable activation (ours): phi(x)=sum_k a_k*tanh(g_k*(x-c_k)). "
            "Natively piecewise-CONSTANT-ish (a sum of soft Heaviside steps) — the "
            "exact shape of the SegNet argmax partition. NO Gibbs ringing (tanh "
            "saturates flat between steps). LEARNABLE: K*(a,g,c)=3K params per "
            "activation instance (small, reported byte delta)."
        ),
        full_paper_architecture=False,
        learnable=True,
    ),
    "fkan": ActivationFamilySpec(
        activation_id="fkan",
        summary="CUSTOM #3: learnable Fourier-series activation (FKAN/SineKAN-style).",
        archive_note=(
            "Learnable activation (ours, FKAN arxiv 2409.09323 / SineKAN "
            "arxiv 2407.04149 flavor): phi(x)=sum_{k=1..K}[a_k*sin(k*omega*x)+"
            "b_k*cos(k*omega*x)]. 'Learn our own nonlinearity': the network "
            "tunes the per-harmonic spectrum to the lane-edge band. LEARNABLE: "
            "2K params per activation instance."
        ),
        full_paper_architecture=False,
        learnable=True,
    ),
}


def normalize_activation_family(value: str) -> ActivationFamilyId:
    """Normalize an activation-family id or raise ``ValueError``."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "naked_siren": "siren",
        "base_siren": "siren",
        "canonical_siren": "siren",
        "finer_style": "finer",
        "wire_style": "wire",
        "bacon_style": "bacon",
        "bandlimited": "bacon",
        "band_limited": "bacon",
        "gaussian": "gauss",
        "gabor_gauss": "gauss",
        "ramasinghe": "gauss",
        "hyperbolic_oscillator": "hosc",
        "ada_hosc": "hosc",
        "cardinal_sine": "sinc",
        "rc_gauss": "rcgauss",
        "flair": "rcgauss",
        "band_localized": "rcgauss",
        "finergauss": "finer_gauss",
        "finer_x_gauss": "finer_gauss",
        "soft_step": "step_basis",
        "tanh_step": "step_basis",
        "stepbasis": "step_basis",
        "fourier_kan": "fkan",
        "fourierkan": "fkan",
        "sinekan": "fkan",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SIREN_ACTIVATION_FAMILIES:
        valid = ", ".join(ACTIVATION_FAMILY_IDS)
        raise ValueError(f"unknown SIREN activation family {value!r}; valid: {valid}")
    return normalized  # type: ignore[return-value]


def is_learnable_activation(activation_family: str) -> bool:
    """True if the family requires a parametric :class:`nn.Module` sub-activation."""

    return normalize_activation_family(activation_family) in LEARNABLE_ACTIVATION_IDS


def effective_layer_omega(
    *,
    activation_family: ActivationFamilyId,
    base_omega: float,
    layer_index: int,
    bacon_bandwidth_scale: float,
) -> float:
    """Return the activation omega used by a layer."""

    if activation_family == "bacon":
        return float(base_omega) * float(bacon_bandwidth_scale) * float(layer_index + 1)
    return float(base_omega)


def apply_activation_family(
    x: torch.Tensor,
    *,
    activation_family: ActivationFamilyId,
    omega: float,
    wire_scale: float,
    hosc_beta: float = 4.0,
    rcgauss_rolloff: float = 0.05,
) -> torch.Tensor:
    """Apply a supported FIXED-FORM INR activation family to preactivations.

    Every branch is the REAL functional form of the named technique applied to
    the actual tensor ``x``. ``omega`` is the base frequency (the vendored sine
    uses omega=1.0; the SIREN substrate uses its per-layer omega). ``wire_scale``
    doubles as the Gaussian/window SCALE ``s`` for the families that take one
    (wire/gauss/rcgauss/finer_gauss). ``hosc_beta`` is HOSC's saturation/sharpness
    control. ``rcgauss_rolloff`` is FLAIR's raised-cosine roll-off (fixed 0.05).

    For the LEARNABLE families (``step_basis``, ``fkan``) this function raises:
    those carry parameters and must be instantiated via
    :func:`make_learnable_activation`.
    """

    fam = activation_family
    if fam == "siren":
        # CONTROL: exactly the vendored torch.sin(omega*x). MUST be bit-identical.
        return torch.sin(omega * x)
    if fam == "finer":
        return torch.sin(omega * (torch.abs(x) + 1.0) * x)
    if fam == "wire":
        # Real-Gabor: sin(omega*x) * Gaussian window.
        window = torch.exp(-0.5 * torch.square(wire_scale * x))
        return torch.sin(omega * x) * window
    if fam == "bacon":
        # Band-limited sine; the band schedule lives in effective_layer_omega.
        return torch.sin(omega * x)
    if fam == "gauss":
        # Ramasinghe-Lucey 2022: exp(-(s*x)^2). Smooth bump, no Gibbs ring.
        return torch.exp(-torch.square(wire_scale * x))
    if fam == "hosc":
        # Serrano 2024: tanh(beta * sin(omega*x)). beta->inf => step train.
        return torch.tanh(hosc_beta * torch.sin(omega * x))
    if fam == "sinc":
        # Saratchandran 2024: sin(omega*x)/(omega*x), with the removable
        # singularity at x=0 patched to 1 (sinc(0)=1). torch.sinc is
        # normalized sinc (sin(pi*y)/(pi*y)); reproduce the UNNORMALIZED
        # sin(t)/t = torch.sinc(t/pi) so omega scales the lobe spacing.
        t = omega * x
        return torch.sinc(t / math.pi)
    if fam == "rcgauss":
        # FLAIR-style band-localized basis (approximation of RC-GAUSS):
        #   sinc(omega*x) * raised_cosine(omega*x; beta) * exp(-0.5*(s*x)^2)
        # The raised-cosine factor sharpens the band cutoff; the Gaussian
        # envelope localizes in space. (Exact RC-GAUSS uses a learnable sigma;
        # here sigma = 1/wire_scale via the wire_scale knob, roll-off fixed.)
        t = omega * x
        sinc = torch.sinc(t / math.pi)
        # Raised-cosine window factor: cos(pi*beta*t) / (1 - (2*beta*t)^2),
        # guarding the +/- 1/(2 beta) singularities (L'Hopital limit -> pi/4).
        beta = float(rcgauss_rolloff)
        denom = 1.0 - torch.square(2.0 * beta * t)
        rc_num = torch.cos(math.pi * beta * t)
        # Where |denom| ~ 0 the raised-cosine value -> pi/4 (the removable limit).
        safe = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1.0), denom)
        rc = torch.where(
            denom.abs() < 1e-6,
            torch.full_like(denom, math.pi / 4.0),
            rc_num / safe,
        )
        envelope = torch.exp(-0.5 * torch.square(wire_scale * x))
        return sinc * rc * envelope
    if fam == "finer_gauss":
        # CUSTOM #1: FINER variable-freq sine * Gaussian envelope.
        finer = torch.sin(omega * (torch.abs(x) + 1.0) * x)
        window = torch.exp(-0.5 * torch.square(wire_scale * x))
        return finer * window
    if fam in LEARNABLE_ACTIVATION_IDS:
        raise ValueError(
            f"activation family {fam!r} is LEARNABLE (carries parameters); "
            f"instantiate via make_learnable_activation(), not "
            f"apply_activation_family()."
        )
    raise ValueError(f"unsupported activation family {activation_family!r}")


# ---------------------------------------------------------------------------
# Learnable activations ("learn our own nonlinearity"). These carry REAL
# parameters and are therefore nn.Modules, not stateless functions. They are
# initialized so that early training is well-conditioned but they are NOT
# no-ops vs siren (the NO-FAKE tests assert a genuine difference).
# ---------------------------------------------------------------------------


class LearnableStepBasis(torch.nn.Module):
    """CUSTOM #2 — a learnable sum of K shifted/scaled tanh soft-steps.

    phi(x) = sum_{k=1..K} a_k * tanh(g_k * (x - c_k))

    This is natively a piecewise-quasi-CONSTANT function (each ``tanh`` term
    saturates to +/- a_k away from its shift ``c_k``, with a sharp transition of
    width ~1/g_k at ``c_k``). That is the exact shape of the SegNet argmax
    PARTITION — locally flat regions separated by sharp class boundaries — so it
    can represent a lane-edge STEP without the Gibbs ringing a sinusoid produces.

    Parameters (all learnable, shared across the activation's call sites):
      - ``a`` (K,): per-step amplitudes (init small so phi ~ near-linear at start)
      - ``g`` (K,): per-step gains/sharpness (init >0; larger => sharper step)
      - ``c`` (K,): per-step shifts/edge locations (init spread over the active range)

    Byte cost: 3K extra params PER INSTANCE (reported; small for K~4-8).
    NO-FAKE: forward is the real tanh-step sum (not a renamed sin); the tests
    assert (a) it differs from siren, (b) each of a/g/c actually modulates output.
    """

    def __init__(self, num_steps: int = 4, init_gain: float = 1.0) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive; got {num_steps}")
        self.num_steps = int(num_steps)
        k = self.num_steps
        # Shifts spread over a symmetric range so the steps tile the active band.
        shifts = torch.linspace(-1.5, 1.5, k)
        # Amplitudes init small (well-conditioned start; phi(x) ~ gentle) but
        # NONZERO so the activation is genuinely a step-sum from epoch 0.
        amps = torch.full((k,), 1.0 / k)
        gains = torch.full((k,), float(init_gain))
        self.a = torch.nn.Parameter(amps)
        self.g = torch.nn.Parameter(gains)
        self.c = torch.nn.Parameter(shifts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast over a trailing step axis: x[..., None] vs params[K].
        xe = x.unsqueeze(-1)
        steps = self.a * torch.tanh(self.g * (xe - self.c))
        return steps.sum(dim=-1)

    def extra_repr(self) -> str:
        return f"num_steps={self.num_steps}"


class FourierKANActivation(torch.nn.Module):
    """CUSTOM #3 — a learnable Fourier-series activation (FKAN / SineKAN flavor).

    phi(x) = sum_{k=1..K} [ a_k * sin(k * omega * x) + b_k * cos(k * omega * x) ]

    The network LEARNS its own per-harmonic spectrum (FKAN arxiv 2409.09323;
    SineKAN arxiv 2407.04149). For our argmax-edge objective this lets the
    optimizer concentrate energy in exactly the harmonic band the lane-marking
    edges occupy, rather than the single fixed frequency of SIREN.

    Parameters (learnable): ``a`` (K,), ``b`` (K,). ``omega`` is the base
    frequency (fixed; the harmonics are integer multiples).

    INIT: a_1 = 1, all other a_k = b_k = 0 => phi(x) = sin(omega*x) at init.
    This is the SIREN basis at epoch 0 (well-conditioned, inherits SIREN's
    spectral prior) but it is GENUINELY learnable (the parity tests assert the
    parameters move the output once perturbed). Byte cost: 2K extra params.
    """

    def __init__(self, num_harmonics: int = 5, omega: float = 1.0) -> None:
        super().__init__()
        if num_harmonics <= 0:
            raise ValueError(f"num_harmonics must be positive; got {num_harmonics}")
        self.num_harmonics = int(num_harmonics)
        self.omega = float(omega)
        k = self.num_harmonics
        a = torch.zeros(k)
        a[0] = 1.0  # phi == sin(omega*x) at init (the SIREN basis)
        b = torch.zeros(k)
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        # Integer harmonic multipliers 1..K (a fixed buffer, not learnable).
        self.register_buffer("_k", torch.arange(1, k + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xe = x.unsqueeze(-1)
        phase = self.omega * self._k * xe
        series = self.a * torch.sin(phase) + self.b * torch.cos(phase)
        return series.sum(dim=-1)

    def extra_repr(self) -> str:
        return f"num_harmonics={self.num_harmonics}, omega={self.omega}"


def make_learnable_activation(
    activation_family: str,
    *,
    omega: float = 1.0,
    step_basis_k: int = 4,
    step_basis_gain: float = 1.0,
    fkan_k: int = 5,
) -> torch.nn.Module:
    """Instantiate the parametric sub-activation module for a learnable family.

    Raises if the family is fixed-form (use :func:`apply_activation_family`).
    """

    fam = normalize_activation_family(activation_family)
    if fam == "step_basis":
        return LearnableStepBasis(num_steps=step_basis_k, init_gain=step_basis_gain)
    if fam == "fkan":
        return FourierKANActivation(num_harmonics=fkan_k, omega=omega)
    raise ValueError(
        f"activation family {fam!r} is not learnable; use apply_activation_family()."
    )


def activation_family_manifest() -> dict[str, dict[str, object]]:
    """Return JSON-serializable activation-family metadata."""

    return {
        key: {
            "summary": spec.summary,
            "archive_note": spec.archive_note,
            "full_paper_architecture": spec.full_paper_architecture,
            "learnable": spec.learnable,
        }
        for key, spec in SIREN_ACTIVATION_FAMILIES.items()
    }


__all__ = [
    "ACTIVATION_FAMILY_IDS",
    "DEFAULT_ACTIVATION_FAMILY",
    "FIXED_FORM_ACTIVATION_IDS",
    "LEARNABLE_ACTIVATION_IDS",
    "SIREN_ACTIVATION_FAMILIES",
    "ActivationFamilyId",
    "ActivationFamilySpec",
    "FourierKANActivation",
    "LearnableStepBasis",
    "activation_family_manifest",
    "apply_activation_family",
    "effective_layer_omega",
    "is_learnable_activation",
    "make_learnable_activation",
    "normalize_activation_family",
]
