import pandas as pd
import os

TOP_N = 100

def main():
    print("Loading SPY parquet dataset...")
    
    df = pd.read_parquet(
        "options-dataset-hist/data/parquet_spy"
    )

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["dte"] = (df["expiration"] - df["date"]).dt.days

    # Focus on 0DTE + weeklies (<=5 days)
    df = df[df["dte"] <= 5]

    df["mid"] = (df["bid"] + df["ask"]) / 2

    df = df.sort_values(["contract_id", "date"])

    df["prev_mid"] = df.groupby("contract_id")["mid"].shift(1)

    df["pct_change"] = (
        (df["mid"] - df["prev_mid"]) / df["prev_mid"] * 100
    )

    df = df.dropna(subset=["pct_change"])

    os.makedirs("output", exist_ok=True)

    gainers = df.sort_values("pct_change", ascending=False).head(TOP_N)
    losers = df.sort_values("pct_change", ascending=True).head(TOP_N)

    gainers.to_csv("output/top_gainers.csv", index=False)
    losers.to_csv("output/top_losers.csv", index=False)

    print("Done. Results saved.")

if __name__ == "__main__":
    main()
