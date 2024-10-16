import csv
import datetime as dt
from dateutil.relativedelta import relativedelta

from bqr.utils import decoder


def get_daily_data(ticker, date):
    fp = f"data/klines/{ticker}_{date}.csv"

    data = []

    with open(fp, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            data.append(decoder.decode_values(row))

    return data


def get_monthly_data(ticker, year, month):
    start_date = dt.date(year, month, 1)
    end_date = start_date + relativedelta(day=31)

    data = []
    while start_date <= end_date:
        data += get_daily_data(ticker, start_date.strftime("%Y%m%d"))
        start_date += dt.timedelta(days=1)

    return data


def get_yearly_data(ticker, year):
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)

    data = []
    while start_date <= end_date:
        data += get_daily_data(ticker, start_date.strftime("%Y%m%d"))
        start_date += dt.timedelta(days=1)

    return data


def get_custom_data(ticker, start_date_str, end_date_str):
    start_date = dt.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = dt.datetime.strptime(end_date_str, "%Y%m%d")

    assert start_date <= end_date, f"{start_date_str=} is after {end_date_str=}"

    data = []
    while start_date <= end_date:
        data += get_daily_data(ticker, start_date.strftime("%Y%m%d"))
        start_date += dt.timedelta(days=1)

    return data
