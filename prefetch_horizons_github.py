#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

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
    if not path.exists():
        raise FileNotFoundError(f"Choose file not found: {path}")

    objects = []
    seen = set()

    for raw in path.read_text(encoding="utf-8").splitlines():
        obj = raw.strip()
        if not obj or obj.startswith("#"):
            continue
        if obj not in seen:
            seen.add(obj)
            objects.append(obj)

    return objects


def object_csv_path(input_dir: Path, obj: str):
    """
    Find the object's CSV recursively anywhere under input_dir.

    Current convention:
        2006 GQ59 -> 2006_GQ59.csv

    No _ALL suffix is used.
    """
    filename = f"{safe_slug(obj)}.csv"

    direct = input_dir / filename
    if direct.is_file():
        return direct

    matches = sorted(p for p in input_dir.rglob(filename) if p.is_file())

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"[WARNING] Multiple CSVs found for {obj}; using {matches[0]}",
            file=sys.stderr,
            flush=True,
        )
        for extra in matches[1:]:
            print(f"          also found: {extra}", file=sys.stderr, flush=True)

    return matches[0]


def read_mjd_range(csv_path: Path) -> tuple[float, float]:
    wanted = {"midpointMjdTai", "mjd", "midPointMjdTai"}

    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)

    if "midpointMjdTai" in df.columns:
        col = "midpointMjdTai"
    elif "mjd" in df.columns:
        col = "mjd"
    elif "midPointMjdTai" in df.columns:
        col = "midPointMjdTai"
    else:
        raise ValueError(
            f"{csv_path} has no supported time column. "
            "Expected midpointMjdTai."
        )

    mjd = pd.to_numeric(df[col], errors="coerce").dropna()

    if mjd.empty:
        raise ValueError(f"No valid MJD values found in {csv_path}")

    return float(mjd.min()), float(mjd.max())


def mjd_to_iso_utc(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot


def build_cache_path(
    output_dir: Path,
    target_id: str,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    start_mjd: float,
    stop_mjd: float,
) -> Path:
    return output_dir / (
        f"{safe_slug(target_id)}"
        f"__loc_{safe_slug(location)}"
        f"__step_{step_minutes}m"
        f"__pad_{pad_minutes}m"
        f"__{start_mjd:.6f}_{stop_mjd:.6f}.csv"
    )


def cache_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        df = pd.read_csv(path, nrows=3)
    except Exception:
        return False

    required = {
        "jd",
        "mjd",
        "pred_V",
        "r_au",
        "delta_au",
        "alpha_deg",
    }

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

    out_path = build_cache_path(
        output_dir,
        target_id,
        location,
        step_minutes,
        pad_minutes,
        start_mjd,
        stop_mjd,
    )

    if cache_is_valid(out_path):
        print(
            f"[SKIP cached] {target_id} step={step_minutes}m -> {out_path.name}",
            flush=True,
        )
        return out_path

    epochs = {
        "start": mjd_to_iso_utc(start_mjd),
        "stop": mjd_to_iso_utc(stop_mjd),
        "step": f"{step_minutes}m",
    }

    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[QUERY] {target_id} | {csv_path} | "
                f"step={step_minutes}m | attempt={attempt}/{retries}",
                flush=True,
            )

            table = Horizons(
                id=target_id,
                location=location,
                epochs=epochs,
            ).ephemerides()

            eph = table.to_pandas()

            out = pd.DataFrame(
                {
                    "jd": pd.to_numeric(
                        eph.get("datetime_jd", np.nan), errors="coerce"
                    ),
                    "pred_V": pd.to_numeric(
                        eph.get("V", np.nan), errors="coerce"
                    ),
                    "r_au": pd.to_numeric(
                        eph.get("r", np.nan), errors="coerce"
                    ),
                    "delta_au": pd.to_numeric(
                        eph.get("delta", np.nan), errors="coerce"
                    ),
                    "alpha_deg": pd.to_numeric(
                        eph.get("alpha", np.nan), errors="coerce"
                    ),
                    "RA_deg": pd.to_numeric(
                        eph.get("RA", np.nan), errors="coerce"
                    ),
                    "DEC_deg": pd.to_numeric(
                        eph.get("DEC", np.nan), errors="coerce"
                    ),
                }
            )

            out = out.dropna(subset=["jd"]).copy()
            out["mjd"] = out["jd"] - MJDREF
            out = out.sort_values("mjd").reset_index(drop=True)

            if len(out) < 2:
                raise RuntimeError(
                    f"Horizons returned only {len(out)} usable rows "
                    f"for {target_id}"
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            tmp_path = out_path.with_suffix(".tmp.csv")
            out.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)

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
                wait = max(sleep_seconds * (2 ** (attempt - 1)), 2.0)
                print(f"[WAIT] retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)

    raise RuntimeError(
        f"Failed Horizons query for {target_id}, "
        f"step={step_minutes}m after {retries} attempts: {last_exc}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--choose-file",
        type=Path,
        default=Path("choose_horizons.txt"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("HORIZONS_CACHE"),
    )
    parser.add_argument("--location", default="X05")
    parser.add_argument("--step-minutes", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--pad-minutes", type=int, default=10)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=1.5)

    args = parser.parse_args()

    if args.num_batches < 1:
        raise SystemExit("--num-batches must be >= 1")

    if not 0 <= args.batch_index < args.num_batches:
        raise SystemExit(
            "--batch-index must be between 0 and num-batches-1"
        )

    if not args.input_dir.is_dir():
        raise SystemExit(
            f"Input directory does not exist: {args.input_dir}"
        )

    objects = read_choose(args.choose_file)

    batch_objects = [
        obj
        for index, obj in enumerate(objects)
        if index % args.num_batches == args.batch_index
    ]

    print(f"Total objects: {len(objects)}", flush=True)
    print(
        f"Batch {args.batch_index + 1}/{args.num_batches}: "
        f"{len(batch_objects)} objects",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    successful_objects = 0

    for number, obj in enumerate(batch_objects, start=1):
        print(
            f"\n=== Batch {args.batch_index + 1}/{args.num_batches} | "
            f"Object {number}/{len(batch_objects)}: {obj} ===",
            flush=True,
        )

        csv_path = object_csv_path(args.input_dir, obj)

        if csv_path is None:
            msg = (
                f"{obj}: {safe_slug(obj)}.csv not found "
                f"anywhere under {args.input_dir}"
            )
            print(f"[MISSING] {msg}", file=sys.stderr, flush=True)
            failures.append(msg)
            continue

        print(f"[INPUT] {obj} -> {csv_path}", flush=True)

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
                msg = f"{obj} step={step}m: {exc}"
                failures.append(msg)
                print(f"[FAILED] {msg}", file=sys.stderr, flush=True)

        if object_ok:
            successful_objects += 1

    print("\n================ SUMMARY ================", flush=True)
    print(
        f"Batch: {args.batch_index + 1}/{args.num_batches}",
        flush=True,
    )
    print(f"Objects assigned: {len(batch_objects)}", flush=True)
    print(
        f"Objects fully successful: {successful_objects}",
        flush=True,
    )
    print(f"Failures: {len(failures)}", flush=True)

    if failures:
        print("\nFailure list:", file=sys.stderr, flush=True)
        for item in failures:
            print(f"  - {item}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
