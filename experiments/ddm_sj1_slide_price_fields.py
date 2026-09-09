"""ddm_sj1 -- build the three token fields that PRICE the slide family by real encode.

The slide sizing measured the MECHANISM (cells repaired per slide). It cannot answer the
second clause of the pre-registration -- *"or where the second token is one the coder
charges near zero"* -- because that is a property of the coder, not of the renderer, and
-log2 p is direction-dependent ([[fs2]]) with average != marginal ([[fs3]], 2.24x).

So it is measured, with the shipped encoder, by building three fields on the live row's
admitted field and encoding each against the live control:

  * ``both``     -- every accepted slide applied (advance AND retreat tokens)
  * ``advance``  -- only the advance token of each slide (the GT class pushed forward)
  * ``retreat``  -- only the retreat token (our class written behind the boundary)

``both`` gives the marginal bits per accepted slide, which is the number the rate clause is
decided on.  ``advance`` and ``retreat`` split it: the standing hypothesis is that the
retreat token is CHEAP or even negative because it restores context that earlier
pre-distortion had broken, and that hypothesis is exactly what the split falsifies or
confirms.  The same split is the number the tc1 question needs, since a stronger context
model reprices a context-violating token rather than an average one.

Nothing here scores.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizing", type=Path, required=True, help="SLIDE_SIZING.json")
    ap.add_argument("--field", type=Path, required=True, help="the live row's admitted field")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    sizing = json.loads(args.sizing.read_text())
    blob = np.load(args.field, allow_pickle=False)
    base = {int(k): np.asarray(blob[k], dtype=np.uint8) for k in blob.files}

    variants = {name: {k: v.copy() for k, v in base.items()} for name in
                ("both", "advance", "retreat")}
    slides = 0
    writes = {"both": 0, "advance": 0, "retreat": 0}
    for row in sizing["rows"]:
        pair = int(row["pair"])
        for s in row["accepted"]:
            slides += 1
            for name in variants:
                plane = variants[name][pair]
                if name in ("both", "advance"):
                    if plane[s["ay"], s["ax"]] != s["advance_class"]:
                        writes[name] += 1
                    plane[s["ay"], s["ax"]] = s["advance_class"]
                if name in ("both", "retreat"):
                    if plane[s["by"], s["bx"]] != s["retreat_class"]:
                        writes[name] += 1
                    plane[s["by"], s["bx"]] = s["retreat_class"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ddm_sj1_slide_price_fields.v1",
        "accepted_slides": slides,
        "token_writes_vs_live_field": writes,
        "source_field": str(args.field),
        "sizing": str(args.sizing),
        "score_claim": False,
    }
    for name, planes in variants.items():
        path = args.out_dir / f"field_slide_{name}.npz"
        np.savez_compressed(path, **{str(k): v for k, v in planes.items()})
        changed = sum(int((planes[k] != base[k]).sum()) for k in base)
        report[f"{name}_tokens_changed"] = changed
        report[f"{name}_path"] = str(path)
        print(f"  {name:8s}: {changed:4d} tokens differ from the live field -> {path.name}")
    (args.out_dir / "SLIDE_PRICE_FIELDS.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"  {slides} accepted slides; break-even at 1.08 cells/slide is ~10.6 bits at pass 5's price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
