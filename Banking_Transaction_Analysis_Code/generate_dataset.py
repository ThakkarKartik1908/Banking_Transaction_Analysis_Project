"""
generate_dataset.py
-------------------
Generates a synthetic retail-banking transaction dataset (banking_transactions.csv)
that follows the schema given in the PBL case study:

    TxnID, AccountID, TxnDate, Type, Amount, MerchantCategory, BalanceAfter

One extra column, IsFraud (0/1), is added as a ground-truth label so that the
fraud-prediction models can be trained and evaluated.

Realistic "dirt" (exact duplicate rows, missing merchant categories and
inconsistent text in the Type column) is injected on purpose so that the
cleaning steps of the analysis have something real to do.

Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_ACCOUNTS = 300
MONTHS = pd.period_range("2026-01", "2026-09", freq="M")
MONTH_FACTOR = [1.00, 0.95, 1.05, 1.00, 1.10, 1.00, 0.95, 1.10, 1.05]

DEBIT_CATS = ["Grocery", "Shopping", "Utilities", "Travel", "Dining", "Fuel",
              "Healthcare", "Entertainment", "Education", "ATM Withdrawal",
              "Online Transfer"]
DEBIT_P = [0.22, 0.14, 0.11, 0.07, 0.12, 0.09, 0.05, 0.07, 0.03, 0.07, 0.03]
BASE = {"Grocery": 1100, "Shopping": 2600, "Utilities": 1700, "Travel": 4200,
        "Dining": 650, "Fuel": 1400, "Healthcare": 2000, "Entertainment": 800,
        "Education": 4500, "ATM Withdrawal": 4000, "Online Transfer": 3000}
SIGMA = {"Shopping": 0.80, "Travel": 0.90, "Education": 0.70, "Healthcare": 0.75,
         "ATM Withdrawal": 0.55, "Online Transfer": 0.70}

FRAUD_CATS = ["Online Transfer", "Shopping", "Travel", "ATM Withdrawal"]
FRAUD_P = [0.40, 0.25, 0.20, 0.15]


def days_in(period):
    return period.days_in_month


rows = []
seq = 0
for a in range(N_ACCOUNTS):
    acc = f"ACC-{1001 + a}"
    salary = float(np.clip(rng.lognormal(np.log(42000), 0.55), 14000, 300000))
    salary = round(salary / 100) * 100
    scale = (salary / 42000) ** 0.6
    opening = salary * rng.uniform(0.3, 3.0)
    has_rent = rng.random() < 0.70
    rent = round(salary * rng.uniform(0.18, 0.32) / 500) * 500
    lam = rng.uniform(5, 9)
    pay_day = int(rng.integers(1, 5))
    typical = 1500 * scale

    events = []  # (date, type, amount, category, fraud)
    for m, f in zip(MONTHS, MONTH_FACTOR):
        nd = days_in(m)
        d0 = pd.Timestamp(m.start_time)
        # salary credit
        events.append((d0 + pd.Timedelta(days=min(nd - 1, pay_day - 1 + int(rng.integers(0, 2)))),
                       "Credit", salary, "Salary", 0))
        # other credits
        if rng.random() < 0.25:
            events.append((d0 + pd.Timedelta(days=int(rng.integers(0, nd))), "Credit",
                           round(salary * rng.uniform(0.05, 0.5), 2), "Transfer", 0))
        if rng.random() < 0.12:
            events.append((d0 + pd.Timedelta(days=int(rng.integers(0, nd))), "Credit",
                           round(rng.uniform(300, 3000), 2), "Refund", 0))
        if m.month in (3, 6, 9) and rng.random() < 0.5:
            events.append((d0 + pd.Timedelta(days=nd - 1), "Credit",
                           round(rng.uniform(100, 900), 2), "Interest", 0))
        # rent
        if has_rent:
            events.append((d0 + pd.Timedelta(days=int(rng.integers(4, 9))), "Debit",
                           float(rent), "Rent", 0))
        # ordinary debits
        n_deb = rng.poisson(lam * f)
        for _ in range(n_deb):
            cat = rng.choice(DEBIT_CATS, p=DEBIT_P)
            amt = BASE[cat] * scale * rng.lognormal(0, SIGMA.get(cat, 0.55))
            # rare, legitimate large purchases (create realistic false positives)
            if cat in ("Shopping", "Travel", "Healthcare") and rng.random() < 0.05:
                amt = BASE[cat] * scale * rng.uniform(5, 10)
            amt = round(max(amt, 50), 2)
            events.append((d0 + pd.Timedelta(days=int(rng.integers(0, nd))), "Debit", amt, cat, 0))

    # fraud events (bursts of 1-3 transactions on the same / next day)
    for _ in range(rng.poisson(0.8)):
        day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 273)))
        for _ in range(int(rng.integers(1, 4))):
            cat = rng.choice(FRAUD_CATS, p=FRAUD_P)
            mult = rng.uniform(4, 14) if rng.random() < 0.75 else rng.uniform(1.5, 4)
            events.append((day + pd.Timedelta(days=int(rng.integers(0, 2))), "Debit",
                           round(typical * mult, 2), cat, 1))

    # chronological replay -> BalanceAfter (ordinary debits cannot breach the overdraft limit)
    events.sort(key=lambda e: (e[0], e[1] == "Debit"))
    bal = opening
    overdraft = -0.25 * salary
    for d, t, amt, cat, fr in events:
        if t == "Debit":
            if not fr and bal - amt < overdraft:
                continue
            bal -= amt
        else:
            bal += amt
        seq += 1
        rows.append((d, acc, seq, t, amt, cat, round(bal, 2), fr))

df = pd.DataFrame(rows, columns=["TxnDate", "AccountID", "seq", "Type", "Amount",
                                 "MerchantCategory", "BalanceAfter", "IsFraud"])
df = df.sort_values(["TxnDate", "AccountID", "seq"]).reset_index(drop=True)
df.insert(0, "TxnID", [f"TXN-{i:06d}" for i in range(1, len(df) + 1)])
df = df.drop(columns="seq")
df["TxnDate"] = df["TxnDate"].dt.strftime("%Y-%m-%d")
df = df[["TxnID", "AccountID", "TxnDate", "Type", "Amount", "MerchantCategory",
         "BalanceAfter", "IsFraud"]]

# ---------- inject realistic data-quality problems ----------
miss_idx = rng.choice(len(df), 70, replace=False)
df.loc[miss_idx, "MerchantCategory"] = np.nan

typo_idx = rng.choice(len(df), 30, replace=False)
variants = ["debit", "credit", " Debit", "Credit ", "DEBIT", "CREDIT"]
for i in typo_idx:
    base_type = df.at[i, "Type"]
    df.at[i, "Type"] = rng.choice([v for v in variants if v.strip().lower() == base_type.lower()])

dup_idx = rng.choice(len(df), 45, replace=False)
df = pd.concat([df, df.iloc[dup_idx]]).sort_index(kind="stable").reset_index(drop=True)

df.to_csv("banking_transactions.csv", index=False)
print("Saved banking_transactions.csv ->", df.shape)
print(df.head(8).to_string(index=False))
print("Fraud rows:", int(df["IsFraud"].sum()))
