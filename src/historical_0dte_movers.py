import pandas as pd
import numpy as np
import requests
import io
import heapq
import os

# ====================================
# CONFIG
# ====================================
GITHUB_API_URL = "https://api.github.com/repos/philippdubach/options-dataset-hist/contents/data/parquet_spy"
TOP_N = 100


# ====================================
# Get Parquet File List from GitHub
# ====================================
def get_parquet_files():
    print("Fetching file list from GitHub API...")
    r = requests.get(GITHUB_API_URL)
    r.raise_for_status()
    files = r.json()

    parquet_files = [
        f["download_url"]
        for f in files
        if f["name"].endswith(".parquet")
    ]

    print(f"Found {len(parquet_files)} parquet files.")
    return parquet_files


# ====================================
# Process Single Parquet File
# ====================================
def process_parquet(url):

    r = requests.get(url)
    r.raise_for_status()

    df = pd.read_parquet(io.BytesIO(r.content))

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["dte"] = (df["expiration"] - df["date"]).dt.days

    # 0DTE only
    df = df[df["dte"] == 0]

    df["mid"] = (df["bid"] + df["ask"]) / 2

    df = df.sort_values(["contract_id", "date"])

    df["prev_mid"] = df.groupby("contract_id")["mid"].shift(1)

    df["pct_change"] = (
        (df["mid"] - df["prev_mid"]) / df["prev_mid"] * 100
    )

    df = df.dropna(subset=["pct_change"])

    return df


# ====================================
# Maintain Top Movers Efficiently
# ====================================
def update_leaderboard(df, top_gainers, top_losers):

    for _, row in df.iterrows():
        pct = row["pct_change"]

        entry = (
            pct,
            row["date"],
            row["expiration"],
            row["strike"],
            row["type"],
            row["prev_mid"],
            row["mid"],
        )

        # Gainers
        if len(top_gainers) < TOP_N:
            heapq.heappush(top_gainers, entry)
        else:
            heapq.heappushpop(top_gainers, entry)

        # Losers (store negative for heap logic)
        neg_entry = (-pct,) + entry[1:]
        if len(top_losers) < TOP_N:
            heapq.heappush(top_losers, neg_entry)
        else:
            heapq.heappushpop(top_losers, neg_entry)


# ====================================
# Main
# ====================================
def main():

    parquet_files = get_parquet_files()

    top_gainers = []
    top_losers = []

    for url in parquet_files:
        print(f"Processing {url}...")
        df = process_parquet(url)
        update_leaderboard(df, top_gainers, top_losers)

    # Convert heaps to DataFrames
    gainers_df = pd.DataFrame(sorted(top_gainers, reverse=True),
        columns=["pct_change","date","expiration","strike","type","prev_mid","mid"]
    )

    losers_df = pd.DataFrame(sorted(top_losers),
        columns=["pct_change","date","expiration","strike","type","prev_mid","mid"]
    )

    print("\n=== BIGGEST 0DTE GAINERS ===")
    print(gainers_df)

    print("\n=== BIGGEST 0DTE LOSERS ===")
    print(losers_df)

    os.makedirs("output", exist_ok=True)

    combined = pd.concat([gainers_df, losers_df])
    combined.to_csv("output/historical_0dte_movers.csv", index=False)

    print("\nSaved to output/historical_0dte_movers.csv")


if __name__ == "__main__":
    main()
