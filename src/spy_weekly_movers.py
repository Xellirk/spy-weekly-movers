import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def compute_return(open_mid, intrinsic):
    if open_mid == 0:
        return np.nan
    return (intrinsic - open_mid) / open_mid * 100


def get_weekly_expirations(ticker, days_ahead=14):
    expirations = ticker.options
    today = datetime.utcnow().date()

    weekly = []
    for exp in expirations:
        d = datetime.strptime(exp, "%Y-%m-%d").date()
        if 0 <= (d - today).days <= days_ahead:
            weekly.append(exp)

    return weekly


def main():
    print("Starting SPY Weekly Options Analysis...")

    spy = yf.Ticker("SPY")
    weekly_exps = get_weekly_expirations(spy)

    if not weekly_exps:
        print("No weekly expirations found.")
        return

    print(f"Weekly expirations found: {weekly_exps}")

    all_opts = []

    for exp in weekly_exps:
        chain = spy.option_chain(exp)

        for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
            temp = df.copy()
            temp["expiration"] = exp
            temp["type"] = opt_type
            all_opts.append(temp)

    df_opts = pd.concat(all_opts, ignore_index=True)

    df_opts["mid"] = (df_opts["bid"] + df_opts["ask"]) / 2

    # Get price history for expiration dates
    min_exp = min(weekly_exps)
    max_exp = max(weekly_exps)

    price_history = spy.history(
        start=(datetime.strptime(min_exp, "%Y-%m-%d") - timedelta(days=3)),
        end=(datetime.strptime(max_exp, "%Y-%m-%d") + timedelta(days=3))
    )

    results = []

    for _, row in df_opts.iterrows():
        exp = row["expiration"]
        strike = row["strike"]
        opt_type = row["type"]
        open_mid = row["mid"]

        try:
            expiry_spot = price_history.loc[exp]["Close"]
        except KeyError:
            expiry_spot = price_history["Close"].loc[:exp].iloc[-1]

        if opt_type == "call":
            intrinsic = max(0, expiry_spot - strike)
        else:
            intrinsic = max(0, strike - expiry_spot)

        ret_pct = compute_return(open_mid, intrinsic)

        results.append({
            "expiration": exp,
            "type": opt_type,
            "strike": strike,
            "open_mid": open_mid,
            "expiry_spot": expiry_spot,
            "intrinsic": intrinsic,
            "return_pct": ret_pct
        })

    df_res = pd.DataFrame(results)

    df_res = df_res.sort_values("return_pct", ascending=False)

    os.makedirs("output", exist_ok=True)
    df_res.to_csv("output/weekly_option_movers.csv", index=False)

    print("Top 10 Winners:")
    print(df_res.head(10))

    print("\nTop 10 Losers:")
    print(df_res.tail(10))

    print("Analysis complete.")


if __name__ == "__main__":
    main()
