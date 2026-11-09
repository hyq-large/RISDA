#!/usr/bin/env python
"""Build an ordering-only spatial Channel-PPGF cache from raw statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import torch

from pathlib import Path
from typing import Mapping

from appa.save import safe_load, safe_save

RAW_SPATIAL_KIND = "mask_frac_obs_encoder_spatial_uncertainty_v1"
SPATIAL_PPGF_MATRIX_KIND = "mask_frac_obs_encoder_spatial_channel_ordering_v1"
DEFAULT_MINIMUM_COUNT = 2
DEFAULT_FILL_VALUE = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool = False) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    result = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = _sha256(path)
    return result


def build_spatial_ordering_payload(
    raw_payload: Mapping,
    *,
    raw_cache_path: Path,
    minimum_count: int = DEFAULT_MINIMUM_COUNT,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> dict:
    """Replace unsupported spatial variances and return an ordering-only cache."""
    minimum_count = int(minimum_count)
    fill_value = float(fill_value)
    if minimum_count < 2:
        raise ValueError("minimum_count must be at least 2 for centered variance")
    if not math.isfinite(fill_value) or fill_value <= 0.0:
        raise ValueError("fill_value must be finite and positive")
    if raw_payload.get("kind") != RAW_SPATIAL_KIND:
        raise RuntimeError(
            "raw spatial cache kind mismatch: "
            f"{raw_payload.get('kind')!r} != {RAW_SPATIAL_KIND!r}"
        )

    sources = tuple(str(value).lower() for value in raw_payload.get("sources", ()))
    latent_shape = tuple(int(value) for value in raw_payload.get("latent_shape", ()))
    if not sources:
        raise RuntimeError("raw spatial cache has no source order")
    if len(latent_shape) != 3 or any(value <= 0 for value in latent_shape):
        raise RuntimeError(f"invalid raw spatial latent shape: {latent_shape}")

    variance = torch.as_tensor(raw_payload.get("variance"), dtype=torch.float32)
    counts = torch.as_tensor(raw_payload.get("counts"), dtype=torch.int64)
    expected_variance_shape = (len(sources), *latent_shape)
    expected_count_shape = (len(sources), *latent_shape[:2])
    if tuple(variance.shape) != expected_variance_shape:
        raise RuntimeError(
            f"raw variance shape {tuple(variance.shape)} "
            f"does not match {expected_variance_shape}"
        )
    if tuple(counts.shape) != expected_count_shape:
        raise RuntimeError(
            f"raw count shape {tuple(counts.shape)} " f"does not match {expected_count_shape}"
        )
    if bool((counts < 0).any()):
        raise RuntimeError("raw spatial counts contain negative values")

    supported_positions = counts >= minimum_count
    supported_scalars = supported_positions.unsqueeze(-1).expand_as(variance)
    supported_values = variance[supported_scalars]
    if supported_values.numel() == 0:
        raise RuntimeError("raw spatial cache has no supported variance values")
    if not torch.isfinite(supported_values).all() or bool((supported_values <= 0.0).any()):
        raise RuntimeError("supported raw spatial variances must be finite and positive")

    ordering_variance = variance.clone()
    ordering_variance[~supported_scalars] = fill_value
    if not torch.isfinite(ordering_variance).all() or bool((ordering_variance <= 0.0).any()):
        raise RuntimeError("filled spatial ordering variance is invalid")

    filled_positions = (~supported_positions).sum(dim=(1, 2))
    zero_count_positions = (counts == 0).sum(dim=(1, 2))
    single_count_positions = (counts == 1).sum(dim=(1, 2))
    filled_by_source = {
        source: int(value)
        for source, value in zip(
            sources,
            filled_positions.tolist(),
            strict=True,
        )
    }
    zero_count_by_source = {
        source: int(value)
        for source, value in zip(
            sources,
            zero_count_positions.tolist(),
            strict=True,
        )
    }
    single_count_by_source = {
        source: int(value)
        for source, value in zip(
            sources,
            single_count_positions.tolist(),
            strict=True,
        )
    }
    raw_identity = file_identity(raw_cache_path, include_sha256=True)
    return {
        "kind": SPATIAL_PPGF_MATRIX_KIND,
        "version": 1,
        "description": (
            "Ordering-only hourly Direct Encoder residual variance at every "
            "fixed latent position and channel"
        ),
        "ordering_ready": True,
        "ordering_only": True,
        "ordering_metric": "centered_standardized_residual_variance",
        "calibration_kind": RAW_SPATIAL_KIND,
        "period": raw_payload.get("period"),
        "time_scope": raw_payload.get("time_scope"),
        "expected_hours": raw_payload.get("expected_hours"),
        "mask_frac": float(raw_payload.get("mask_frac")),
        "sources": sources,
        "source_axis": 0,
        "layout": "(source, H_lat, W_lat, C_lat)",
        "latent_shape": latent_shape,
        "minimum_count": minimum_count,
        "fill_value": fill_value,
        "missing_value_policy": (
            f"positions with count < {minimum_count} are filled with " f"variance {fill_value:g}"
        ),
        "filled_position_count": int(filled_positions.sum()),
        "filled_scalar_count": int(filled_positions.sum()) * latent_shape[-1],
        "filled_position_count_by_source": filled_by_source,
        "zero_count_position_count": int(zero_count_positions.sum()),
        "single_count_position_count": int(single_count_positions.sum()),
        "zero_count_position_count_by_source": zero_count_by_source,
        "single_count_position_count_by_source": single_count_by_source,
        "variance": ordering_variance.contiguous(),
        "counts": counts.contiguous(),
        "raw_cache": raw_identity,
    }


def _cache_matches(
    payload: Mapping,
    *,
    raw_identity: Mapping,
    minimum_count: int,
    fill_value: float,
) -> bool:
    return (
        payload.get("kind") == SPATIAL_PPGF_MATRIX_KIND
        and payload.get("raw_cache") == raw_identity
        and int(payload.get("minimum_count", -1)) == int(minimum_count)
        and float(payload.get("fill_value", float("nan"))) == float(fill_value)
    )


def write_summary(payload: Mapping, output_path: Path) -> Path:
    summary_path = output_path.with_suffix(".json")
    summary = {
        key: payload[key]
        for key in (
            "kind",
            "version",
            "ordering_ready",
            "ordering_only",
            "ordering_metric",
            "calibration_kind",
            "period",
            "time_scope",
            "expected_hours",
            "mask_frac",
            "sources",
            "layout",
            "latent_shape",
            "minimum_count",
            "fill_value",
            "missing_value_policy",
            "filled_position_count",
            "filled_scalar_count",
            "filled_position_count_by_source",
            "zero_count_position_count",
            "single_count_position_count",
            "zero_count_position_count_by_source",
            "single_count_position_count_by_source",
            "raw_cache",
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
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-count",
        type=int,
        default=DEFAULT_MINIMUM_COUNT,
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=DEFAULT_FILL_VALUE,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cache_path = args.raw_cache.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    raw_identity = file_identity(raw_cache_path, include_sha256=True)

    if output_path.is_file() and not args.force:
        cached = safe_load(output_path, map_location="cpu")
        if _cache_matches(
            cached,
            raw_identity=raw_identity,
            minimum_count=args.minimum_count,
            fill_value=args.fill_value,
        ):
            print(f"Reuse {output_path}", flush=True)
            return
        raise RuntimeError(
            f"ordering cache already exists with a different contract: {output_path}"
        )

    raw_payload = safe_load(raw_cache_path, map_location="cpu")
    payload = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache_path,
        minimum_count=args.minimum_count,
        fill_value=args.fill_value,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(payload, output_path)
    summary_path = write_summary(payload, output_path)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
