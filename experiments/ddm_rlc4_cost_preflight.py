"""Produce the first-measurement cost-preflight object for one T4 auth-eval dispatch.

Emits two files under ``--store``: ``modal_price_source.json`` (machine-readable provider
prices transcribed from a retained capture of https://modal.com/pricing, with the capture's
path/bytes/sha256) and ``cost_preflight.json`` in the exact shape
``tac.candidate_seal._pf_cost`` validates: one T4, four CPUs, sixteen GiB, the full
4,800-second function cap, every charged resource enumerated, each bound re-derived, total < 5 USD.
Volumes are not attached by the auth-eval dispatch, so storage is not a charged resource here.
Re-run within 24 h of authorization; the validator refuses a stale fetch time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

REMOTE_SECONDS = 4800
RESOURCES = (("gpu", 1), ("cpu", 4), ("memory", 16))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="durable SSD directory holding the retained page capture")
    parser.add_argument("--page", required=True, help="retained raw capture of https://modal.com/pricing")
    parser.add_argument("--gpu-usd-per-second", type=float, required=True)
    parser.add_argument("--cpu-usd-per-core-second", type=float, required=True)
    parser.add_argument("--memory-usd-per-gib-second", type=float, required=True)
    parser.add_argument("--fetched-at-utc", required=True, help="ISO-8601 time the page was captured")
    args = parser.parse_args(argv)

    from tac.candidate_seal import _pf_cost, prefire_file_reference

    store = Path(args.store)
    page = Path(args.page)
    rates = {"gpu": args.gpu_usd_per_second, "cpu": args.cpu_usd_per_core_second,
             "memory": args.memory_usd_per_gib_second}
    source_doc = {
        "provider": "modal", "url": "https://modal.com/pricing", "gpu": "T4", "currency": "USD",
        "unit": "usd_per_unit_second", "rates": rates,
        "chargeable_resources": ["gpu", "cpu", "memory"],
        "not_charged_for_this_dispatch": {
            "volume": "auth-eval dispatch attaches no modal.Volume; results return through the function-call cache"},
        "page_capture": prefire_file_reference(page), "fetched_at_utc": args.fetched_at_utc,
    }
    source_path = store / "modal_price_source.json"
    source_path.write_text(json.dumps(source_doc, indent=2, sort_keys=True) + "\n")
    resources = []
    for name, qty in RESOURCES:
        price = rates[name]
        resources.append({"resource": name, "quantity": qty, "usd_per_unit_second": price,
                          "price_field": ["rates", name], "upper_bound_usd": qty * price * REMOTE_SECONDS})
    cost = {"gpu": "T4", "currency": "USD", "paid_dispatches": 1, "remote_seconds": REMOTE_SECONDS,
            "provider_price_fetched_at_utc": args.fetched_at_utc,
            "provider_price_source": {**prefire_file_reference(source_path), "url": "https://modal.com/pricing"},
            "resources": resources, "upper_bound_usd": sum(r["upper_bound_usd"] for r in resources)}
    cost_path = store / "cost_preflight.json"
    cost_path.write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n")
    _pf_cost(prefire_file_reference(cost_path))
    print(json.dumps({"cost_preflight": prefire_file_reference(cost_path), "upper_bound_usd": cost["upper_bound_usd"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
