"""This module downloads data from Official Binance platform at minutely precision
    For simiplicity we are now focusing on ETHUSDT Spot Data"""

import csv
import json
import time
import datetime as dt
import argparse
from typing import List, Any
import requests


API_ENDPOINT = (
    "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1m&limit=1000"
)

HEADERS = [
    "open_ts",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "close_ts",
    "quote_asset_vol",
    "num_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "reserved",
]


def query_api(start_time: int, end_time: int) -> List[List[Any]]:
    """Queries BINANCE Official API to fetch SPOT KLine/CandleStick Data

    Args:
        start_time (int): Start Epoch time
        end_time (int): End epoch time

    Returns:
        List[List[Any]]: Data Response in JSON
    """
    api_url = API_ENDPOINT + f"&startTime={start_time}&endTime={end_time}"

    data = requests.get(api_url, timeout=5)

    data = json.loads(data.content.decode("utf-8"))

    return data


def fetch_minutely_data(date: dt.datetime) -> None:
    """Fetches minutely data for a particular date

    Args:
        date (dt.datetime): Date
    """
    master_data = []
    date_str = date.strftime("%Y%m%d")
    start_epoch = int(time.mktime(date.timetuple())) * 1000
    end_epoch = start_epoch + 86399999

    while start_epoch <= end_epoch:
        batch_data = query_api(start_epoch, start_epoch + 43199999)
        master_data += batch_data
        start_epoch += 43200000

    if len(master_data) == 0:
        print(f"No data found for - {date_str}")
        return

    transformed_data = [dict(zip(HEADERS, row)) for row in master_data]

    output_path = f"data/klines/ETHUSDT_{date_str}.csv"
    with open(output_path, "w", encoding="utf-8") as fin:
        writer = csv.DictWriter(fin, fieldnames=HEADERS, lineterminator="\n")

        writer.writeheader()

        for row in transformed_data:
            writer.writerow(row)

    print(f"Data dumped at - {output_path}")

    print(f"Successfully generated KLine data for {date_str}")


def extract_data(start_date: str, end_date: str) -> None:
    """Extracts and saves data in dated CSV files

    Args:
        start_date (str): Start Date in YYYYMMDD format
        end_date (str): End Date in YYYYMMDD format
    """
    start = dt.datetime(
        int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]), 0, 0, 0, 0
    )
    end = dt.datetime(
        int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]), 0, 0, 0, 0
    )

    while start <= end:
        fetch_minutely_data(start)
        start += dt.timedelta(days=1)


def define_parser() -> argparse.ArgumentParser:
    """Parser for the script

    Returns:
        argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser("KLine Data Extractor")
    parser.add_argument(
        "-s",
        "--start",
        type=str,
        required=True,
        help="Start Date , shoudl be in YYYYMMDD format",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        required=True,
        help="End Date , shoudl be in YYYYMMDD format",
    )

    return parser


if __name__ == "__main__":

    parser = define_parser()
    args = parser.parse_args()
    extract_data(args.start, args.end)
