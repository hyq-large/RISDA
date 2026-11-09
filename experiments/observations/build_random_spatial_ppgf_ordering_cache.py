#!/usr/bin/env python
"""Build a fixed random spatial ordering cache for PPGF or PCGrad."""

from __future__ import annotations

import argparse
import json
import torch

from pathlib import Path
from typing import Mapping

from appa.save import safe_load, safe_save
from experiments.observations.build_spatial_ordering_cache import (
    SPATIAL_PPGF_MATRIX_KIND,
    file_identity,
)

RANDOM_ORDERING_METRIC = "uniform_random_source_permutation"
RANDOM_ORDERING_GRANULARITIES = ("spatial_channel", "spatial_vector")
RANDOMIZATION_GRANULARITY_CONTRACT = {
    "spatial_channel": "independent_per_H_lat_W_lat_C_lat",
    "spatial_vector": "independent_per_H_lat_W_lat_shared_C_lat",
}


def build_random_spatial_ordering_payload(
    reference_payload: Mapping,
    *,
    reference_cache_path: Path,
    seed: int,
    granularity: str = "spatial_channel",
) -> dict:
    """Encode fixed uniform source permutations at the requested granularity."""
    if reference_payload.get("kind") != SPATIAL_PPGF_MATRIX_KIND:
        raise RuntimeError(
            "reference cache kind mismatch: "
            f"{reference_payload.get('kind')!r} != {SPATIAL_PPGF_MATRIX_KIND!r}"
        )
    sources = tuple(str(value).lower() for value in reference_payload.get("sources", ()))
    latent_shape = tuple(int(value) for value in reference_payload.get("latent_shape", ()))
    if len(sources) < 2:
        raise RuntimeError("random ordering requires at least two sources")
    if len(latent_shape) != 3 or any(value <= 0 for value in latent_shape):
        raise RuntimeError(f"invalid reference latent shape: {latent_shape}")
    granularity = str(granularity).lower()
    if granularity not in RANDOM_ORDERING_GRANULARITIES:
        raise ValueError(
            "random ordering granularity must be one of "
            f"{RANDOM_ORDERING_GRANULARITIES}"
        )

    reference_values = torch.as_tensor(reference_payload.get("variance"))
    expected_shape = (len(sources), *latent_shape)
    if tuple(reference_values.shape) != expected_shape:
        raise RuntimeError(
            f"reference ordering shape {tuple(reference_values.shape)} != {expected_shape}"
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_shape = (
        expected_shape
        if granularity == "spatial_channel"
        else (len(sources), *latent_shape[:2], 1)
    )
    random_keys = torch.rand(random_shape, generator=generator, dtype=torch.float64)
    priority_order = torch.argsort(
        random_keys,
        dim=0,
        descending=True,
        stable=True,
    )
    rank_values = torch.arange(
        len(sources),
        0,
        -1,
        dtype=torch.float32,
    ).reshape(len(sources), 1, 1, 1)
    ordering_values = torch.empty(random_shape, dtype=torch.float32)
    ordering_values.scatter_(
        0,
        priority_order,
        rank_values.expand(random_shape),
    )
    if granularity == "spatial_vector":
        ordering_values = ordering_values.expand(expected_shape).clone()

    random_unit = (
        "spatial latent channel"
        if granularity == "spatial_channel"
        else "spatial latent vector, shared by all latent channels"
    )

    payload = {
        "kind": SPATIAL_PPGF_MATRIX_KIND,
        "version": 1,
        "description": (
            "Fixed independent uniform-random source permutation for every "
            f"{random_unit}"
        ),
        "ordering_ready": True,
        "ordering_only": True,
        "ordering_metric": RANDOM_ORDERING_METRIC,
        "random_seed": int(seed),
        "period": reference_payload.get("period"),
        "time_scope": reference_payload.get("time_scope"),
        "expected_hours": reference_payload.get("expected_hours"),
        "mask_frac": reference_payload.get("mask_frac"),
        "sources": sources,
        "source_axis": 0,
        "layout": "(source, H_lat, W_lat, C_lat)",
        "latent_shape": latent_shape,
        "randomization_granularity": RANDOMIZATION_GRANULARITY_CONTRACT[
            granularity
        ],
        "priority_semantics": (
            "larger value enters earlier; value 1 is the final/highest-priority source"
        ),
        "variance": ordering_values.contiguous(),
        "reference_cache": file_identity(reference_cache_path, include_sha256=True),
    }
    counts = reference_payload.get("counts")
    if counts is not None:
        counts = torch.as_tensor(counts)
        expected_count_shape = (len(sources), *latent_shape[:2])
        if tuple(counts.shape) != expected_count_shape:
            raise RuntimeError(
                f"reference count shape {tuple(counts.shape)} != {expected_count_shape}"
            )
        payload["counts"] = counts.contiguous()
    return payload


def cache_matches(
    payload: Mapping,
    *,
    reference_identity: Mapping,
    seed: int,
    granularity: str = "spatial_channel",
) -> bool:
    return (
        payload.get("kind") == SPATIAL_PPGF_MATRIX_KIND
        and payload.get("ordering_metric") == RANDOM_ORDERING_METRIC
        and int(payload.get("random_seed", -1)) == int(seed)
        and payload.get("randomization_granularity")
        == RANDOMIZATION_GRANULARITY_CONTRACT[granularity]
        and payload.get("reference_cache") == reference_identity
    )


def write_summary(payload: Mapping, output_path: Path) -> Path:
    summary_path = output_path.with_suffix(".json")
    summary = {
        key: payload.get(key)
        for key in (
            "kind",
            "version",
            "description",
            "ordering_ready",
            "ordering_only",
            "ordering_metric",
            "random_seed",
            "sources",
            "layout",
            "latent_shape",
            "randomization_granularity",
            "priority_semantics",
            "reference_cache",
        )
    }
    summary["variance_shape"] = list(torch.as_tensor(payload["variance"]).shape)
    summary["output"] = str(output_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--granularity",
        choices=RANDOM_ORDERING_GRANULARITIES,
        default="spatial_channel",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.reference_cache.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reference_identity = file_identity(reference_path, include_sha256=True)

    if output_path.is_file() and not args.force:
        cached = safe_load(output_path, map_location="cpu")
        if cache_matches(
            cached,
            reference_identity=reference_identity,
            seed=args.seed,
            granularity=args.granularity,
        ):
            print(f"Reuse {output_path}", flush=True)
            return
        raise RuntimeError(
            f"random ordering cache already has a different contract: {output_path}"
        )

    reference_payload = safe_load(reference_path, map_location="cpu")
    payload = build_random_spatial_ordering_payload(
        reference_payload,
        reference_cache_path=reference_path,
        seed=args.seed,
        granularity=args.granularity,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(payload, output_path)
    summary_path = write_summary(payload, output_path)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
