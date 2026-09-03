#!/usr/bin/env python3
"""
Generate portable HORIZONS_CACHE CSV files for the LIneA asteroid pipeline.

Expected repository layout
--------------------------
choose_lsm.txt
parquet/
    109312.csv
    2006_GQ59.csv
    ...
HORIZONS_CACHE/

The output filename and columns intentionally match the cache format used by
Period_search_Multiband_Lomb_Scargle.py / Period_search_High_Order_Fourier.py.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

MJDREF = 2400000.5


def safe_slug(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._")
    return value if value else "target"


def read_choose(path: Path) -> list[str]:
    objects: list[str] = []
    seen: set[str] = set()

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Current convention: one asteroid identifier per line.
        obj = line
        if obj not in seen:
            seen.add(obj)
            objects.append(obj)

    return objects


def object_csv_path(input_dir: Path, obj: str) -> Path:
    # Current batch convention: NO "_ALL".
    # Example: "2006 GQ59" -> "2006_GQ59.csv"
    return input_dir / f"{safe_slug(obj)}.csv"


def read_mjd_range(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path, usecols=lambda c: c in {
        "midpointMjdTai", "mjd", "midPointMjdTai"
    })

    if "midpointMjdTai" in df.columns:
        col = "midpointMjdTai"
    elif "mjd" in df.columns:
        col = "mjd"
    elif "midPointMjdTai" in df.columns:
        # Legacy spelling fallback only.
        col = "midPointMjdTai"
    else:
        raise ValueError(
            f"{csv_path} has no supported MJD column. "
            "Expected 'midpointMjdTai' (current convention)."
        )

    mjd = pd.to_numeric(df[col], errors="coerce").dropna()
    if mjd.empty:
        raise ValueError(f"No valid MJD values in {csv_path}")

    return float(mjd.min()), float(mjd.max())


def mjd_to_iso_utc(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot


def cache_path(
    output_dir: Path,
    target_id: str,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    start_mjd: float,
    stop_mjd: float,
) -> Path:
    return output_dir / (
        f"{safe_slug(target_id)}__loc_{safe_slug(location)}"
        f"__step_{step_minutes}m"
        f"__pad_{pad_minutes}m"
        f"__{start_mjd:.6f}_{stop_mjd:.6f}.csv"
    )


def cache_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, nrows=3)
    except Exception:
        return False
    required = {"jd", "mjd", "pred_V", "r_au", "delta_au", "alpha_deg"}
    return len(df) >= 2 and required.issubset(df.columns)


def query_and_save(
    target_id: str,
    csv_path: Path,
    output_dir: Path,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    retries: int,
    sleep_seconds: float,
) -> Path:
    mjd_min, mjd_max = read_mjd_range(csv_path)
    start_mjd = mjd_min - pad_minutes / 1440.0
    stop_mjd = mjd_max + pad_minutes / 1440.0

    out_path = cache_path(
        output_dir=output_dir,
        target_id=target_id,
        location=location,
        step_minutes=step_minutes,
        pad_minutes=pad_minutes,
        start_mjd=start_mjd,
        stop_mjd=stop_mjd,
    )

    if cache_is_valid(out_path):
        print(f"[SKIP cached] {target_id} step={step_minutes}m -> {out_path.name}", flush=True)
        return out_path

    epochs = {
        "start": mjd_to_iso_utc(start_mjd),
        "stop": mjd_to_iso_utc(stop_mjd),
        "step": f"{int(step_minutes)}m",
    }

    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[QUERY] {target_id} | {csv_path.name} | "
                f"step={step_minutes}m | attempt={attempt}/{retries}",
                flush=True,
            )

            obj = Horizons(
                id=target_id,
                location=location,
                epochs=epochs,
            )
            eph = obj.ephemerides()
            eph_df = eph.to_pandas()

            out = pd.DataFrame({
                "jd": pd.to_numeric(eph_df.get("datetime_jd", np.nan), errors="coerce"),
                "pred_V": pd.to_numeric(eph_df.get("V", np.nan), errors="coerce"),
                "r_au": pd.to_numeric(eph_df.get("r", np.nan), errors="coerce"),
                "delta_au": pd.to_numeric(eph_df.get("delta", np.nan), errors="coerce"),
                "alpha_deg": pd.to_numeric(eph_df.get("alpha", np.nan), errors="coerce"),
                "RA_deg": pd.to_numeric(eph_df.get("RA", np.nan), errors="coerce"),
                "DEC_deg": pd.to_numeric(eph_df.get("DEC", np.nan), errors="coerce"),
            })

            out = out.dropna(subset=["jd"]).copy()
            out["mjd"] = out["jd"] - MJDREF
            out = out.sort_values("mjd").reset_index(drop=True)

            if len(out) < 2:
                raise RuntimeError(
                    f"Horizons returned only {len(out)} usable rows for {target_id}"
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp.csv")
            out.to_csv(tmp, index=False)
            tmp.replace(out_path)

            print(
                f"[OK] {target_id} step={step_minutes}m "
                f"rows={len(out)} -> {out_path.name}",
                flush=True,
            )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return out_path

        except Exception as exc:
            last_exc = exc
            print(
                f"[ERROR] {target_id} step={step_minutes}m "
                f"attempt={attempt}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                wait = sleep_seconds * (2 ** (attempt - 1))
                wait = max(wait, 2.0)
                print(f"  retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"Failed Horizons query for {target_id}, step={step_minutes}m "
        f"after {retries} attempts: {last_exc}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choose-file", type=Path, default=Path("choose_lsm.txt"))
    parser.add_argument("--input-dir", type=Path, default=Path("parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("HORIZONS_CACHE"))
    parser.add_argument("--location", default="X05")
    parser.add_argument("--step-minutes", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--pad-minutes", type=int, default=10)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Pause after successful queries; retries use increasing waits.",
    )
    args = parser.parse_args()

    if args.num_batches < 1:
        raise SystemExit("--num-batches must be >= 1")
    if not 0 <= args.batch_index < args.num_batches:
        raise SystemExit("--batch-index must be in [0, num-batches)")

    objects = read_choose(args.choose_file)

    # Stable round-robin partitioning.
    batch_objects = [
        obj for i, obj in enumerate(objects)
        if i % args.num_batches == args.batch_index
    ]

    print(
        f"Total objects: {len(objects)} | "
        f"batch {args.batch_index + 1}/{args.num_batches}: {len(batch_objects)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    successes = 0

    for n, obj in enumerate(batch_objects, start=1):
        print(
            f"\n=== batch {args.batch_index + 1}/{args.num_batches} "
            f"object {n}/{len(batch_objects)}: {obj} ===",
            flush=True,
        )

        csv_path = object_csv_path(args.input_dir, obj)
        if not csv_path.exists():
            msg = f"{obj}: missing input CSV {csv_path}"
            print(f"[MISSING] {msg}", file=sys.stderr, flush=True)
            failures.append(msg)
            continue

        object_ok = True
        for step in args.step_minutes:
            try:
                query_and_save(
                    target_id=obj,
                    csv_path=csv_path,
                    output_dir=args.output_dir,
                    location=args.location,
                    step_minutes=step,
                    pad_minutes=args.pad_minutes,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                )
            except Exception as exc:
                object_ok = False
                failures.append(f"{obj} step={step}m: {exc}")
                print(f"[FAILED] {obj} step={step}m: {exc}", file=sys.stderr, flush=True)

        if object_ok:
            successes += 1

    print("\n================ SUMMARY ================", flush=True)
    print(f"Batch: {args.batch_index + 1}/{args.num_batches}", flush=True)
    print(f"Objects assigned: {len(batch_objects)}", flush=True)
    print(f"Objects fully successful: {successes}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)

    if failures:
        print("\nFailure list:", file=sys.stderr, flush=True)
        for item in failures:
            print(f"  - {item}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
