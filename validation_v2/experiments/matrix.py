"""Deterministic, content-addressed experiment matrix enumeration."""

from __future__ import annotations

import hashlib
from itertools import product
from typing import Any, Mapping

from .provenance import canonical_json


REQUIRED_AXES = ("models", "seeds", "topologies", "rates", "protocols")


def _identifier(item: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(item).encode("utf-8")).hexdigest()


def enumerate_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate stable missingness and separate timestamp-irregularity cases."""

    missing = [name for name in REQUIRED_AXES if name not in config]
    if missing:
        raise ValueError("missing matrix axes: " + ", ".join(missing))
    axes = {name: list(config[name]) for name in REQUIRED_AXES}
    if any(not values for values in axes.values()):
        raise ValueError("matrix axes must not be empty")

    payloads: list[dict[str, Any]] = []
    for model, seed, topology, rate, protocol in product(
        axes["models"], axes["seeds"], axes["topologies"], axes["rates"], axes["protocols"]
    ):
        if isinstance(rate, Mapping):
            requested = rate.get("requested_fraction")
            realized = rate.get("realized_fraction")
        else:
            requested = rate
            realized = None
        payloads.append(
            {
                "case_type": "missingness",
                "model": model,
                "seed": seed,
                "protocol": protocol,
                "topology": topology,
                "requested_fraction": requested,
                "realized_fraction": realized,
            }
        )

    for model, seed, irregular, protocol in product(
        axes["models"], axes["seeds"], list(config.get("irregular_cases", ())), axes["protocols"]
    ):
        if not isinstance(irregular, Mapping):
            raise ValueError("irregular cases must be mappings")
        payloads.append(
            {
                "case_type": "irregular",
                "model": model,
                "seed": seed,
                "protocol": protocol,
                "topology": None,
                "requested_fraction": None,
                "realized_fraction": None,
                "irregular_method": irregular.get("method"),
                "requested_irregularity": irregular.get("requested_irregularity"),
                "realized_irregularity": irregular.get("realized_irregularity"),
            }
        )

    canonical = [canonical_json(item) for item in payloads]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate combination")
    return [
        {"combination_id": _identifier(item), **item}
        for item in sorted(payloads, key=canonical_json)
    ]
