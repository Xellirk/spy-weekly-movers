import pandas as pd
import numpy as np
import requests
import io
import os

# =============================
# CONFIG
# =============================
BASE_API_URL = "https://api.github.com/repos/philippdubach/options-dataset-hist/contents/data/parquet_spy"
TOP_N = 100


# =============================
# Recursively Get All Parquet URLs
# =============================
def get_all_parquet_urls(api_url):
    urls = []

    r = requests.get(api_url)
    r.raise_for_status()

    items = r.json()

    for item in items:
        if item["type"] == "file" and item["name"].endswith(".parquet"):
            urls.append(item["download_url"])

        elif item["type"] == "dir":
            urls.extend(get_all_parquet_urls(item["url"]))

    return urls


# =============================
# Process One Parquet File
# =============================
def process_parquet(url):

    print(f"Downloading: {url}")

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


# =============================
# Main
# =============================
def main():

    print("Fetching parquet file list...")
    parquet_urls = get_all_parquet_urls(BASE_API_URL)

    print(f"Found {len(parquet_urls)} parquet files.")

    all_results = []

    for url in parquet_urls:
        df = process_parquet(url)
        all_results.append(df)

    full_df = pd.concat(all_results)

    print("\n=== TOP 0DTE GAINERS (ALL TIME) ===")
    print(
        full_df.sort_values("pct_change", ascending=False)
        .head(TOP_N)[
            ["date","expiration","strike","type","prev_mid","mid","pct_change"]
        ]
    )

    print("\n=== TOP 0DTE LOSERS (ALL TIME) ===")
    print(
        full_df.sort_values("pct_change", ascending=True)
        .head(TOP_N)[
            ["date","expiration","strike","type","prev_mid","mid","pct_change"]
        ]
    )

    os.makedirs("output", exist_ok=True)
    full_df.to_csv("output/historical_0dte_movers.csv", index=False)

    print("\nSaved to output/historical_0dte_movers.csv")


if __name__ == "__main__":
    main()
