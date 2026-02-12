import pandas as pd
import numpy as np
import requests
import io
import os


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
PARQUET_URL = "https://github.com/philippdubach/options-dataset-hist/tree/main/data/parquet_spy"
TOP_N = 25
FILTER_MODE = "weeklies"   # options: None, "weeklies", "0dte"


# ---------------------------------------------------
# Load Parquet from GitHub
# ---------------------------------------------------
def load_parquet_from_github(url):
    print("Downloading parquet from GitHub...")
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    df = pd.read_parquet(buffer)
    print("Download complete.")
    return df


# ---------------------------------------------------
# Preprocess
# ---------------------------------------------------
def preprocess(df):
    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df.sort_values(["contract_id", "date"])

    df["prev_mid"] = df.groupby("contract_id")["mid"].shift(1)

    df["pct_change"] = (
        (df["mid"] - df["prev_mid"]) / df["prev_mid"] * 100
    )

    df["dte"] = (df["expiration"] - df["date"]).dt.days

    return df


# ---------------------------------------------------
# Filters
# ---------------------------------------------------
def apply_filters(df, mode):
    if mode == "weeklies":
        return df[df["dte"] <= 5]
    elif mode == "0dte":
        return df[df["dte"] == 0]
    return df


# ---------------------------------------------------
# Print Results
# ---------------------------------------------------
def print_movers(df):
    movers = df.dropna(subset=["pct_change"])

    print("\n===== TOP GAINERS =====")
    print(
        movers.sort_values("pct_change", ascending=False)
        .head(TOP_N)[
            ["date", "expiration", "strike", "type", "mid", "pct_change"]
        ]
    )

    print("\n===== TOP LOSERS =====")
    print(
        movers.sort_values("pct_change", ascending=True)
        .head(TOP_N)[
            ["date", "expiration", "strike", "type", "mid", "pct_change"]
        ]
    )

    os.makedirs("output", exist_ok=True)
    movers.to_csv("output/daily_option_moves.csv", index=False)


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    df = load_parquet_from_github(PARQUET_URL)
    df = preprocess(df)
    df = apply_filters(df, FILTER_MODE)
    print_movers(df)


if __name__ == "__main__":
    main()
