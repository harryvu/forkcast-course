"""Download one month of NYC TLC yellow-taxi trips and save a cleaned, sampled CSV.

Used from the Lesson 6 notebook. Needs pandas and pyarrow (both present in Colab).

Usage:  python tools/get_taxi.py --month 2024-01 --rows 200000 --out data/taxi_2024-01.csv

The TLC publishes monthly parquet files; the URL pattern below has been stable for
years but check https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page if it 404s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "passenger_count", "PULocationID", "DOLocationID"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2024-01")
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"data/taxi_{args.month}.csv")
    print(f"downloading {URL.format(month=args.month)} ...")
    frame = pd.read_parquet(URL.format(month=args.month), columns=COLUMNS)
    print(f"{len(frame):,} trips in the month; sampling {args.rows:,}")
    frame = frame.sample(n=min(args.rows, len(frame)), random_state=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB). Cleaning happens in forkcast.trips.load_trips.")


if __name__ == "__main__":
    main()
