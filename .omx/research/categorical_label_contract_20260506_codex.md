# Categorical Label Contract - 2026-05-06

## Finding

The categorical/CLADE/SPADE work had been mixing two different contracts:

- comma10k / contest SegNet semantic class order
- Selfcomp grayscale-LUT wire targets

Those are related but not interchangeable. comma10k documents one-based
semantic IDs:

1. road, `#402020`
2. lane markings, `#ff0000`
3. undrivable, `#808060`
4. movable, `#00ff66`
5. my car, `#cc00ff`
6. movable in my car, `#00ccff`, interior-only and not part of the five-class
   contest SegNet channel contract

Contest tensors use the five non-interior labels in the same order, zero-based:

`0=road, 1=lane_markings, 2=undrivable, 3=movable, 4=my_car`.

The public PR56/Selfcomp inflate path exposes `CLASS_TARGETS =
[0, 255, 64, 192, 128]`. This is a grayscale codebook indexed by contest class
ID. It is not evidence that class 0 should be renamed `background`.

## Source Evidence

- External: `https://raw.githubusercontent.com/commaai/comma10k/master/README.md`
  lines 13-20 list internal SegNet categories and colors.
- External: `https://blog.comma.ai/crowdsourced-segnet-you-can-help/` describes
  five labels in the road-scene grouping: road, lane markings, undrivable,
  movable, and my car.
- Local public-frontier source: `reports/raw/leaderboard_intel_20260501/pr56_inflate.py`
  defines `CLASS_TARGETS = [0, 255, 64, 192, 128]` without semantic renaming.
- Local contest code: `src/tac/mask_codec.py` and lane-mark pose modules already
  treat class 0 as road and class 1 as lane markings.

## Code Contract

`src/tac/semantic_label_contract.py` is now the canonical source for:

- `CONTEST_SEGNET_CLASSES`
- `CONTEST_SEGNET_CLASS_NAMES`
- `SELFCOMP_CLASS_TO_GRAY`
- `SEMANTIC_QUANTIZATION_DEFAULT_BITS`

`src/tac/mask_grayscale_lut.py` consumes `SELFCOMP_CLASS_TO_GRAY` as a wire
codebook over contest class IDs.

`src/tac/semantic_quantization.py` consumes `SEMANTIC_QUANTIZATION_DEFAULT_BITS`
for CLADE/SPADE per-class parameter quantization, preserving high precision for
road and lane-marking channels.

## Follow-Up

Audit remaining older prose that says class 4 is `sky` or `background`. The
likely source of that drift is treating comma10k undrivable-including-sky as a
separate sky/background channel. Do not change runtime behavior from prose alone;
only promote changes with class-channel provenance or exact archive evidence.
