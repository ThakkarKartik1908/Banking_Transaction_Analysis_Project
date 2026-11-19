# %% [markdown]
# # Banking Transaction Analysis - Real World Case Study
# **Subject:** Python for Data Science (BE05000231) | **SVIT, Vasad (GTU)**
# **Student:** Kartik Thakkar (240410107057)
#
# ## Introduction
# Banks need analytical reports on customer transactions to understand spending
# behaviour and account activity. This notebook processes a banking-transaction
# dataset (`banking_transactions.csv`), cleans it, explores it, segments customers
# with K-Means, flags anomalies with Isolation Forest and predicts fraud with a
# Random Forest. All numbers used in the project report are produced by this file.

# %% [markdown]
# ## 1. Import Libraries

# %%
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve,
                             silhouette_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42

DATA_PATH = "banking_transactions.csv"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

R = {}            # every number used in the report is stored here -> results.json


def savefig(name):
    """Save the current figure into outputs/figures and show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def r2(x, n=2):
    return round(float(x), n)


print("Libraries imported successfully")

# %% [markdown]
# ## 2. Upload Dataset
# In Google Colab a file-upload dialog is shown. When running locally the CSV
# placed next to this script is used.

# %%
try:
    from google.colab import files          # only exists inside Google Colab
    if not os.path.exists(DATA_PATH):
        uploaded = files.upload()           # choose banking_transactions.csv
        DATA_PATH = list(uploaded.keys())[0]
except ImportError:
    print("Not running in Colab - using local file:", DATA_PATH)

# %% [markdown]
# ## 3. Load Dataset

# %%
df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH, "->", df.shape)
df.head()

# %% [markdown]
# ## 4. Dataset Information

# %%
df.info()
raw_dates = pd.to_datetime(df["TxnDate"], errors="coerce")
R["raw_rows"], R["raw_cols"] = int(df.shape[0]), int(df.shape[1])
R["columns"] = list(df.columns)
R["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
R["raw_accounts"] = int(df["AccountID"].nunique())
R["raw_categories"] = int(df["MerchantCategory"].nunique())
R["date_min"] = str(raw_dates.min().date())
R["date_max"] = str(raw_dates.max().date())
R["raw_type_values"] = {str(k): int(v) for k, v in df["Type"].value_counts().items()}
print(R["raw_type_values"])

# %% [markdown]
# ## 5. Data Cleaning
# Trim whitespace, standardise the `Type` text, convert data types and validate values.

# %%
df_raw = df.copy()
for col in ["TxnID", "AccountID", "Type", "MerchantCategory"]:
    df[col] = df[col].str.strip()
df["Type"] = df["Type"].str.title()
df["TxnDate"] = pd.to_datetime(df["TxnDate"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["BalanceAfter"] = pd.to_numeric(df["BalanceAfter"], errors="coerce")

R["type_rows_fixed"] = int((df_raw["Type"] != df["Type"]).sum())
R["type_values_clean"] = {str(k): int(v) for k, v in df["Type"].value_counts().items()}
R["validation"] = {
    "invalid_dates": int(df["TxnDate"].isna().sum()),
    "non_positive_amounts": int((df["Amount"] <= 0).sum()),
    "invalid_type_values": int((~df["Type"].isin(["Credit", "Debit"])).sum()),
    "negative_balance_rows": int((df["BalanceAfter"] < 0).sum()),
}
print(R["validation"])
df["Type"].value_counts()

# %% [markdown]
# ## 6. Missing Values

# %%
miss = df.isna().sum()
miss_pct = (miss / len(df) * 100).round(2)
missing_tbl = pd.DataFrame({"Missing values": miss, "Missing %": miss_pct})
R["missing"] = {c: [int(missing_tbl.at[c, "Missing values"]), float(missing_tbl.at[c, "Missing %"])]
                for c in missing_tbl.index}
print(missing_tbl)

# Missing merchant categories are labelled explicitly instead of being dropped,
# because the Amount / Balance information of those rows is still valid.
df["MerchantCategory"] = df["MerchantCategory"].fillna("Uncategorized")
print("Remaining missing values:", int(df.isna().sum().sum()))

# %% [markdown]
# ## 7. Duplicate Values

# %%
R["full_duplicates"] = int(df.duplicated().sum())
R["repeated_txnids"] = int(df["TxnID"].duplicated().sum())
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
R["rows_after_dedup"] = int(len(df))
R["repeated_txnids_after"] = int(df["TxnID"].duplicated().sum())
print("Duplicate rows removed:", rows_before - len(df), "| rows now:", len(df))

# %% [markdown]
# ## 7b. Feature Engineering
# Derived variables used for analysis, clustering and fraud prediction.

# %%
df = df.sort_values(["AccountID", "TxnDate", "TxnID"]).reset_index(drop=True)
df["Month"] = df["TxnDate"].dt.to_period("M").astype(str)
df["MonthNum"] = df["TxnDate"].dt.month
df["DayOfWeek"] = df["TxnDate"].dt.dayofweek
df["DayName"] = df["TxnDate"].dt.day_name()
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["IsMonthStart"] = (df["TxnDate"].dt.day <= 5).astype(int)
df["SignedAmount"] = np.where(df["Type"] == "Credit", df["Amount"], -df["Amount"])
df["BalanceBefore"] = df["BalanceAfter"] - df["SignedAmount"]
df["NegBalanceFlag"] = (df["BalanceAfter"] < 0).astype(int)

is_debit = df["Type"] == "Debit"
acct_stats = df[is_debit].groupby("AccountID")["Amount"].agg(["mean", "std"]).rename(
    columns={"mean": "acct_mean", "std": "acct_std"})
df = df.join(acct_stats, on="AccountID")
df["AmountZ"] = np.where(is_debit, (df["Amount"] - df["acct_mean"]) / df["acct_std"].replace(0, np.nan), 0)
df["AmountZ"] = df["AmountZ"].fillna(0)
df["AmountToBalance"] = np.where(is_debit, df["Amount"] / df["BalanceBefore"].clip(lower=1), 0)
df["AmountToBalance"] = df["AmountToBalance"].clip(upper=20)

if "IsFraud" not in df.columns:               # fallback when a dataset has no fraud label
    print("WARNING: no IsFraud column found - creating a rule-based proxy label.")
    df["IsFraud"] = (is_debit & ((df["AmountZ"] > 4) | (df["AmountToBalance"] > 1))).astype(int)

R["clean_rows"] = int(len(df))
R["engineered_features"] = ["Month", "MonthNum", "DayOfWeek", "DayName", "IsWeekend", "IsMonthStart",
                            "SignedAmount", "BalanceBefore", "NegBalanceFlag", "AmountZ", "AmountToBalance"]
df.to_csv(os.path.join(OUT_DIR, "cleaned_transactions.csv"), index=False)
df.head()

# %% [markdown]
# ## 8. Descriptive Statistics

# %%
desc = df[["Amount", "BalanceAfter"]].describe().T
desc["skew"] = df[["Amount", "BalanceAfter"]].skew()
print(desc.round(2))
R["describe"] = {idx: {k: r2(v) for k, v in row.items()} for idx, row in desc.iterrows()}
by_type = df.groupby("Type")["Amount"].describe().round(2)
print(by_type)
R["describe_by_type"] = {idx: {k: r2(v) for k, v in row.items()} for idx, row in by_type.iterrows()}
R["accounts"] = int(df["AccountID"].nunique())
R["categories"] = int(df["MerchantCategory"].nunique())
R["fraud_rows"] = int(df["IsFraud"].sum())
R["fraud_pct"] = r2(df["IsFraud"].mean() * 100)
R["txns_per_account_mean"] = r2(df.groupby("AccountID").size().mean(), 1)

# %% [markdown]
# ## 9. Credit vs Debit Analysis

# %%
cd = df.groupby("Type")["Amount"].agg(Transactions="count", Total="sum", Average="mean",
                                      Median="median", Maximum="max")
cd["Share of txns %"] = cd["Transactions"] / cd["Transactions"].sum() * 100
print(cd.round(2))
R["credit_debit"] = {t: {k: r2(v) for k, v in row.items()} for t, row in cd.iterrows()}
net = cd.at["Credit", "Total"] - cd.at["Debit", "Total"]
R["net_flow"] = r2(net)
R["spend_to_income_pct"] = r2(cd.at["Debit", "Total"] / cd.at["Credit", "Total"] * 100)

# total spending per account (step 1 of the PBL guideline)
debits = df[df["Type"] == "Debit"]
acct_debit = debits.groupby("AccountID")["Amount"].sum().sort_values(ascending=False)
R["top_accounts"] = [[k, r2(v)] for k, v in acct_debit.head(5).items()]
R["acct_debit_mean"] = r2(acct_debit.mean())
R["acct_debit_median"] = r2(acct_debit.median())

# negative-balance flags
R["neg_balance_rows"] = int(df["NegBalanceFlag"].sum())
R["neg_balance_accounts"] = int(df.loc[df["NegBalanceFlag"] == 1, "AccountID"].nunique())
R["neg_balance_pct"] = r2(df["NegBalanceFlag"].mean() * 100)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#2a9d8f", "#e76f51"]
ax[0].bar(cd.index, cd["Total"] / 1e6, color=colors)
ax[0].set_title("Total value (INR million)"); ax[0].set_ylabel("INR million")
ax[1].bar(cd.index, cd["Transactions"], color=colors)
ax[1].set_title("Number of transactions"); ax[1].set_ylabel("Transactions")
savefig("fig01_credit_vs_debit.png")

# %% [markdown]
# ## 10. Monthly Spending

# %%
monthly = df.pivot_table(index="Month", columns="Type", values="Amount", aggfunc="sum").fillna(0)
monthly["Net"] = monthly["Credit"] - monthly["Debit"]
monthly["Debit growth %"] = monthly["Debit"].pct_change() * 100
print(monthly.round(2))
R["monthly"] = [[m, r2(r["Credit"]), r2(r["Debit"]), r2(r["Net"])] for m, r in monthly.iterrows()]
R["peak_spend_month"] = [monthly["Debit"].idxmax(), r2(monthly["Debit"].max())]
R["low_spend_month"] = [monthly["Debit"].idxmin(), r2(monthly["Debit"].min())]
R["avg_monthly_debit"] = r2(monthly["Debit"].mean())

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(monthly.index, monthly["Credit"] / 1e6, marker="o", color="#2a9d8f", label="Credit")
ax.plot(monthly.index, monthly["Debit"] / 1e6, marker="s", color="#e76f51", label="Debit")
ax.set_title("Monthly credit vs debit value"); ax.set_ylabel("INR million"); ax.set_xlabel("Month")
ax.legend()
savefig("fig02_monthly_trend.png")

# %% [markdown]
# ## 11. Category Spending

# %%
cat_spend = debits.groupby("MerchantCategory")["Amount"].agg(Total="sum", Transactions="count", Average="mean")
cat_spend["Share %"] = cat_spend["Total"] / cat_spend["Total"].sum() * 100
cat_spend = cat_spend.sort_values("Total", ascending=False)
print(cat_spend.round(2))
R["category_spend"] = [[c, r2(r["Total"]), int(r["Transactions"]), r2(r["Average"]), r2(r["Share %"])]
                       for c, r in cat_spend.iterrows()]

fig, ax = plt.subplots(figsize=(8, 4.8))
cs = cat_spend.sort_values("Total")
ax.barh(cs.index, cs["Total"] / 1e6, color=sns.color_palette("viridis", len(cs)))
ax.set_xlabel("Total debit (INR million)"); ax.set_title("Spending by merchant category")
savefig("fig03_category_spend_bar.png")

# %% [markdown]
# ## 12. High-Value Transactions

# %%
# Rent is a fixed, predictable monthly payment, so "high-value" is judged on the
# remaining (discretionary) debits only.
disc = debits[debits["MerchantCategory"] != "Rent"]
q1, q3 = disc["Amount"].quantile([0.25, 0.75])
iqr = q3 - q1
fence = q3 + 1.5 * iqr
p95 = disc["Amount"].quantile(0.95)
p99 = disc["Amount"].quantile(0.99)
df["HighValue"] = ((df["Type"] == "Debit") & (df["MerchantCategory"] != "Rent") & (df["Amount"] >= p95)).astype(int)
hv = df[df["HighValue"] == 1]
R["hv"] = {"q1": r2(q1), "q3": r2(q3), "iqr_fence": r2(fence), "p95": r2(p95), "p99": r2(p99),
           "disc_debits": int(len(disc)),
           "count": int(len(hv)), "pct_of_debits": r2(len(hv) / len(disc) * 100),
           "value_share_pct": r2(hv["Amount"].sum() / disc["Amount"].sum() * 100),
           "iqr_outliers": int((disc["Amount"] > fence).sum()),
           "fraud_in_hv": int(hv["IsFraud"].sum()),
           "fraud_captured_pct": r2(hv["IsFraud"].sum() / max(df["IsFraud"].sum(), 1) * 100)}
hv_cat = hv.groupby("MerchantCategory")["Amount"].agg(["count", "sum"]).sort_values("count", ascending=False)
R["hv_by_category"] = [[c, int(r["count"]), r2(r["sum"])] for c, r in hv_cat.head(6).iterrows()]
top10 = hv.sort_values("Amount", ascending=False).head(10)
R["top10_hv"] = [[r.TxnID, r.AccountID, str(r.TxnDate.date()), r.MerchantCategory, r2(r.Amount)]
                 for r in top10.itertuples()]
hv.sort_values("Amount", ascending=False)[["TxnID", "AccountID", "TxnDate", "Amount", "MerchantCategory",
                                           "BalanceAfter"]].to_csv(os.path.join(OUT_DIR, "high_value_transactions.csv"), index=False)
print(R["hv"])
top10[["TxnID", "AccountID", "TxnDate", "Amount", "MerchantCategory"]]

# %% [markdown]
# ## 13. Visualization using Matplotlib

# %%
# (a) pie chart - category spending share (small categories grouped as "Other")
share = cat_spend["Share %"]
major = share[share >= 4]
pie_vals = pd.concat([major, pd.Series({"Other categories": share[share < 4].sum()})])
fig, ax = plt.subplots(figsize=(7, 6))
ax.pie(pie_vals.values, labels=pie_vals.index, autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("Set3", len(pie_vals)), wedgeprops={"edgecolor": "white"})
ax.set_title("Share of total debit spending by category")
savefig("fig04_category_pie.png")

# (b) line graph - daily balance trend of the highest-spending account
top_acc = acct_debit.index[0]
bal = df[df["AccountID"] == top_acc].groupby("TxnDate")["BalanceAfter"].last()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(bal.index, bal.values / 1e3, color="#264653")
ax.axhline(0, color="red", ls="--", lw=1)
ax.set_title(f"Daily closing balance trend - {top_acc}")
ax.set_ylabel("Balance (INR thousand)"); ax.set_xlabel("Date")
fig.autofmt_xdate()
savefig("fig05_daily_balance_line.png")

# (c) histogram of debit amounts
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.log10(debits["Amount"]), bins=40, color="#457b9d", edgecolor="white")
ax.axvline(np.log10(p95), color="red", ls="--", label=f"95th percentile (excl. rent) = INR {p95:,.0f}")
ax.set_xlabel("log10(Debit amount in INR)"); ax.set_ylabel("Transactions")
ax.set_title("Distribution of debit amounts"); ax.legend()
savefig("fig06_amount_histogram.png")

# (d) high-value transactions by category
fig, ax = plt.subplots(figsize=(8, 4))
hv_cat["count"].sort_values().plot(kind="barh", color="#e9c46a", ax=ax)
ax.set_xlabel("High-value transactions"); ax.set_title("High-value (top 5%, excl. rent) debits by category")
savefig("fig07_high_value_category.png")

# %% [markdown]
# ## 14. Visualization using Seaborn

# %%
# (a) box plot of debit amount per category (log scale)
order = cat_spend.index.tolist()
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=debits, x="MerchantCategory", y="Amount", order=order, ax=ax, palette="Set2", fliersize=2)
ax.set_yscale("log"); ax.tick_params(axis="x", rotation=40)
ax.set_title("Debit amount distribution by merchant category (log scale)")
savefig("fig08_box_category.png")

# (b) heat map - category x month spending
heat = debits.pivot_table(index="MerchantCategory", columns="Month", values="Amount", aggfunc="sum").loc[order] / 1e3
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "INR thousand"})
ax.set_title("Spending heat map: category x month (INR thousand)"); ax.set_ylabel(""); ax.set_xlabel("")
savefig("fig09_heatmap_category_month.png")

# (c) correlation heat map
num_cols = ["Amount", "BalanceBefore", "BalanceAfter", "AmountZ", "AmountToBalance", "DayOfWeek",
            "MonthNum", "IsWeekend", "IsFraud"]
corr = df.loc[is_debit.values, num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6.2))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (debit transactions)")
savefig("fig10_correlation.png")
R["corr_amount_fraud"] = r2(corr.at["Amount", "IsFraud"], 3)
R["corr_ratio_fraud"] = r2(corr.at["AmountToBalance", "IsFraud"], 3)
R["corr_z_fraud"] = r2(corr.at["AmountZ", "IsFraud"], 3)

# (d) average debit by weekday
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = debits.groupby("DayName")["Amount"].agg(["mean", "count"]).loc[dow_order]
R["weekday"] = [[d, r2(r["mean"]), int(r["count"])] for d, r in dow.iterrows()]
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=dow.index, y=dow["mean"], ax=ax, palette="crest")
ax.set_ylabel("Average debit (INR)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
ax.set_title("Average debit amount by weekday")
savefig("fig11_weekday_average.png")

# (e) KDE of amounts by transaction type
fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=df, x="Amount", hue="Type", log_scale=True, fill=True, common_norm=False,
            palette={"Credit": "#2a9d8f", "Debit": "#e76f51"}, ax=ax)
ax.set_title("Amount density: credit vs debit (log scale)")
savefig("fig12_kde_type.png")

# %% [markdown]
# ## 15. K-Means Customer Segmentation

# %%
cust = df.groupby("AccountID").agg(
    total_credit=("SignedAmount", lambda s: s[s > 0].sum()),
    total_debit=("SignedAmount", lambda s: -s[s < 0].sum()),
    txn_count=("TxnID", "count"),
    avg_balance=("BalanceAfter", "mean"),
)
cust["avg_txn_amount"] = df.groupby("AccountID")["Amount"].mean()
cust["spend_ratio"] = cust["total_debit"] / cust["total_credit"]

feat_cols = ["total_credit", "total_debit", "avg_txn_amount", "txn_count", "avg_balance", "spend_ratio"]
X = cust[feat_cols].copy()
for c in ["total_credit", "total_debit", "avg_txn_amount", "avg_balance"]:
    X[c] = np.log1p(X[c].clip(lower=0))
Xs = StandardScaler().fit_transform(X)

ks = list(range(2, 9))
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))
R["kmeans_k"] = {"k": ks, "inertia": [r2(v, 1) for v in inertia], "silhouette": [r2(v, 3) for v in sil]}

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, marker="o", color="#264653"); ax[0].set_title("Elbow method")
ax[0].set_xlabel("Number of clusters (k)"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, marker="s", color="#e76f51"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("Number of clusters (k)"); ax[1].set_ylabel("Score")
savefig("fig13_kmeans_elbow.png")

# final model: k = 3 -> Low / Medium / High value tiers
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs)
cust["cluster"] = km.labels_
rank = cust.groupby("cluster")["total_credit"].mean().sort_values().index.tolist()
tier_map = {rank[0]: "Low-value", rank[1]: "Medium-value", rank[2]: "High-value"}
cust["Segment"] = cust["cluster"].map(tier_map)
R["kmeans_final"] = {"k": K, "silhouette": r2(silhouette_score(Xs, km.labels_), 3), "inertia": r2(km.inertia_, 1)}

seg = cust.groupby("Segment").agg(Accounts=("txn_count", "size"), Avg_credit=("total_credit", "mean"),
                                  Avg_debit=("total_debit", "mean"), Avg_txn=("avg_txn_amount", "mean"),
                                  Avg_balance=("avg_balance", "mean"), Spend_ratio=("spend_ratio", "mean"),
                                  Txns=("txn_count", "mean")).loc[["Low-value", "Medium-value", "High-value"]]
seg["Share %"] = seg["Accounts"] / seg["Accounts"].sum() * 100
print(seg.round(2))
R["segments"] = {s: {k: r2(v) for k, v in row.items()} for s, row in seg.iterrows()}
# share of total spending coming from each tier
spend_share = cust.groupby("Segment")["total_debit"].sum() / cust["total_debit"].sum() * 100
R["segment_spend_share"] = {s: r2(spend_share[s]) for s in seg.index}
cust.reset_index().to_csv(os.path.join(OUT_DIR, "customer_segments.csv"), index=False)

pcs = PCA(n_components=2, random_state=RANDOM_STATE)
P = pcs.fit_transform(Xs)
R["pca_var"] = r2(pcs.explained_variance_ratio_.sum() * 100, 1)
fig, ax = plt.subplots(figsize=(7.5, 5))
pal = {"Low-value": "#e9c46a", "Medium-value": "#2a9d8f", "High-value": "#264653"}
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=cust["Segment"].values, palette=pal, s=40, ax=ax,
                hue_order=["Low-value", "Medium-value", "High-value"])
ax.set_xlabel("Principal component 1"); ax.set_ylabel("Principal component 2")
ax.set_title(f"K-Means customer segments (k=3, PCA view, {R['pca_var']}% variance)")
savefig("fig14_kmeans_clusters.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x=cust["Segment"], order=["Low-value", "Medium-value", "High-value"], palette=pal, ax=ax[0])
ax[0].set_title("Accounts per segment"); ax[0].set_xlabel("")
ax[1].bar(spend_share.loc[["Low-value", "Medium-value", "High-value"]].index,
          spend_share.loc[["Low-value", "Medium-value", "High-value"]].values,
          color=[pal[s] for s in ["Low-value", "Medium-value", "High-value"]])
ax[1].set_title("Share of total spending (%)")
savefig("fig15_segment_profile.png")

# %% [markdown]
# ## 16. Isolation Forest Anomaly Detection
# Unsupervised: the fraud label is **not** used for training - only to check the result.

# %%
if_feats = ["Amount", "AmountZ", "AmountToBalance", "BalanceBefore"]
dmask = df["Type"] == "Debit"
D = df[dmask].copy()
CONTAM = 0.03
iso = IsolationForest(n_estimators=300, contamination=CONTAM, random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(D[if_feats])
D["iso_pred"] = (iso.predict(D[if_feats]) == -1).astype(int)
D["iso_score"] = -iso.score_samples(D[if_feats])           # higher = more anomalous

R["iso"] = {
    "contamination": CONTAM, "debit_rows": int(len(D)), "flagged": int(D["iso_pred"].sum()),
    "fraud_in_flagged": int(D.loc[D["iso_pred"] == 1, "IsFraud"].sum()),
    "precision": r2(precision_score(D["IsFraud"], D["iso_pred"]), 3),
    "recall": r2(recall_score(D["IsFraud"], D["iso_pred"]), 3),
    "f1": r2(f1_score(D["IsFraud"], D["iso_pred"]), 3),
    "roc_auc": r2(roc_auc_score(D["IsFraud"], D["iso_score"]), 3),
}
print(R["iso"])
flag_cat = D[D["iso_pred"] == 1].groupby("MerchantCategory").size().sort_values(ascending=False)
R["iso_flag_by_category"] = [[c, int(v)] for c, v in flag_cat.head(5).items()]
D[D["iso_pred"] == 1].sort_values("iso_score", ascending=False)[
    ["TxnID", "AccountID", "TxnDate", "Amount", "MerchantCategory", "BalanceBefore", "iso_score", "IsFraud"]
].to_csv(os.path.join(OUT_DIR, "flagged_anomalies.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(D.loc[D["iso_pred"] == 0, "Amount"], D.loc[D["iso_pred"] == 0, "AmountToBalance"],
           s=6, alpha=0.35, color="#8d99ae", label="Normal")
ax.scatter(D.loc[D["iso_pred"] == 1, "Amount"], D.loc[D["iso_pred"] == 1, "AmountToBalance"],
           s=14, alpha=0.9, color="#d62828", label="Anomaly")
ax.set_xscale("log"); ax.set_xlabel("Debit amount (INR, log scale)"); ax.set_ylabel("Amount / balance before")
ax.set_title("Isolation Forest - detected anomalies"); ax.legend()
savefig("fig16_isoforest_scatter.png")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=D, x="iso_score", hue="IsFraud", bins=50, stat="density", common_norm=False,
             palette={0: "#8d99ae", 1: "#d62828"}, ax=ax)
ax.set_title("Anomaly score: legitimate (0) vs fraud (1)"); ax.set_xlabel("Anomaly score (higher = more unusual)")
savefig("fig17_isoforest_scores.png")

# %% [markdown]
# ## 17. Random Forest Fraud Prediction
# Fraud can only occur on debit transactions, so the classifier is trained on debit rows.
# 80:20 stratified train/test split, random_state = 42.

# %%
cat_dummies = pd.get_dummies(D["MerchantCategory"], prefix="cat")
num_feats = ["Amount", "AmountZ", "AmountToBalance", "BalanceBefore", "NegBalanceFlag",
             "MonthNum", "DayOfWeek", "IsWeekend", "IsMonthStart"]
Xf = pd.concat([D[num_feats], cat_dummies], axis=1).astype(float)
yf = D["IsFraud"].values

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    Xf, yf, D.index, test_size=0.20, stratify=yf, random_state=RANDOM_STATE)
R["split"] = {"train": int(len(X_tr)), "test": int(len(X_te)), "features": int(Xf.shape[1]),
              "fraud_train": int(y_tr.sum()), "fraud_test": int(y_te.sum()),
              "fraud_pct_debit": r2(yf.mean() * 100)}

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_prob = rf.predict_proba(X_te)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                         class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
                         Xf, yf, cv=cv, scoring="roc_auc")
R["rf_cv_auc"] = {"mean": r2(cv_auc.mean(), 3), "std": r2(cv_auc.std(), 3), "folds": [r2(v, 3) for v in cv_auc]}

imp = pd.Series(rf.feature_importances_, index=Xf.columns).sort_values(ascending=False)
R["rf_importance"] = [[k, r2(v, 4)] for k, v in imp.head(10).items()]
fig, ax = plt.subplots(figsize=(8, 4.5))
imp.head(10).sort_values().plot(kind="barh", color="#2a9d8f", ax=ax)
ax.set_xlabel("Importance"); ax.set_title("Random Forest - top 10 feature importances")
savefig("fig18_rf_importance.png")

# %% [markdown]
# ## 18. Model Evaluation

# %%
def metrics(y, pred, score):
    return {"accuracy": r2(accuracy_score(y, pred), 3), "precision": r2(precision_score(y, pred, zero_division=0), 3),
            "recall": r2(recall_score(y, pred), 3), "f1": r2(f1_score(y, pred), 3),
            "roc_auc": r2(roc_auc_score(y, score), 3), "pr_auc": r2(average_precision_score(y, score), 3)}


# threshold sensitivity (shown for information - the reported model uses the default 0.5)
R["rf_threshold"] = []
for th in [0.2, 0.3, 0.4, 0.5]:
    p_ = (rf_prob >= th).astype(int)
    R["rf_threshold"].append([th, r2(precision_score(y_te, p_, zero_division=0), 3),
                              r2(recall_score(y_te, p_), 3), r2(f1_score(y_te, p_), 3), int(p_.sum())])
print(R["rf_threshold"])

cm = confusion_matrix(y_te, rf_pred)
tn, fp, fn, tp = [int(v) for v in cm.ravel()]
R["rf_confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
R["rf_metrics"] = metrics(y_te, rf_pred, rf_prob)
print(classification_report(y_te, rf_pred, target_names=["Legitimate", "Fraud"]))
rep = classification_report(y_te, rf_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
R["rf_report"] = {k: {m: r2(v, 3) for m, v in rep[k].items()} for k in ["Legitimate", "Fraud", "macro avg", "weighted avg"]}

fig, ax = plt.subplots(figsize=(5.2, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Random Forest confusion matrix")
savefig("fig19_rf_confusion.png")

# %% [markdown]
# ## 19. Comparison of Algorithms
# Logistic Regression and Decision Tree are trained as supervised baselines. Isolation Forest
# (unsupervised) is scored on the same test rows.

# %%
sc = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE).fit(sc.transform(X_tr), y_tr)
lr_prob = lr.predict_proba(sc.transform(X_te))[:, 1]
dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced",
                            random_state=RANDOM_STATE).fit(X_tr, y_tr)
dt_prob = dt.predict_proba(X_te)[:, 1]
iso_te_pred = D.loc[idx_te, "iso_pred"].values
iso_te_score = D.loc[idx_te, "iso_score"].values

scores = {"Logistic Regression": (lr_prob >= 0.5).astype(int), "Decision Tree": (dt_prob >= 0.5).astype(int),
          "Random Forest": rf_pred, "Isolation Forest": iso_te_pred}
probs = {"Logistic Regression": lr_prob, "Decision Tree": dt_prob, "Random Forest": rf_prob,
         "Isolation Forest": iso_te_score}
comp = {name: metrics(y_te, scores[name], probs[name]) for name in scores}
R["comparison"] = comp
comp_df = pd.DataFrame(comp).T
print(comp_df)
comp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
R["best_model_f1"] = comp_df["f1"].idxmax()
R["best_model_auc"] = comp_df["roc_auc"].idxmax()

fig, ax = plt.subplots(figsize=(9, 4.5))
comp_df[["precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.tick_params(axis="x", rotation=15)
ax.set_title("Algorithm comparison on the test set"); ax.legend(loc="lower right")
savefig("fig20_algorithm_comparison.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={comp[name]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[1].plot(rc, pr, label=f"{name} (AP={comp[name]['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curves"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall curves"); ax[1].legend(fontsize=8)
savefig("fig21_roc_pr_curves.png")

# %% [markdown]
# ## 20. Business Insights

# %%
top_cat = R["category_spend"][0]
top3_share = sum(c[4] for c in R["category_spend"][:3])
best = R["best_model_f1"]
insights = [
    f"Customers spend about {R['spend_to_income_pct']}% of the money they receive (debits INR {R['credit_debit']['Debit']['Total']/1e6:,.1f} M vs credits INR {R['credit_debit']['Credit']['Total']/1e6:,.1f} M), i.e. a net inflow of INR {R['net_flow']/1e6:,.1f} M across the nine months.",
    f"'{top_cat[0]}' is the largest spending category ({top_cat[4]}% of debit value) and the top three categories together account for {top3_share:.1f}%. Rent is a fixed monthly payment, so most discretionary spending sits in Shopping, Travel and ATM cash - the natural focus for card offers and partnerships.",
    f"Spending peaks in {R['peak_spend_month'][0]} (INR {R['peak_spend_month'][1]/1e6:,.2f} M) and is lowest in {R['low_spend_month'][0]} (INR {R['low_spend_month'][1]/1e6:,.2f} M); campaigns and liquidity planning can follow this pattern.",
    f"Excluding rent, the top 5% of debits (>= INR {R['hv']['p95']:,.0f}) carry {R['hv']['value_share_pct']}% of discretionary debit value, so a small set of transactions deserves extra monitoring.",
    f"K-Means splits the {R['accounts']} accounts into three tiers; the High-value tier is {R['segments']['High-value']['Share %']}% of accounts but {R['segment_spend_share']['High-value']}% of spending - the group to prioritise for relationship management.",
    f"{R['neg_balance_accounts']} accounts went below zero at least once ({R['neg_balance_pct']}% of transactions) - candidates for overdraft alerts and low-balance nudges.",
    f"Fraud is rare ({R['split']['fraud_pct_debit']}% of debits); {best} gave the best F1 ({comp[best]['f1']}) with the highest precision ({comp['Random Forest']['precision']}), while Logistic Regression caught more fraud (recall {comp['Logistic Regression']['recall']}) at the cost of many false alarms - the bank must choose the balance between missed fraud and review workload.",
    f"Unsupervised Isolation Forest ranks fraud above normal activity (ROC AUC {R['iso']['roc_auc']}) but its fixed {int(CONTAM*100)}% cut-off flagged {R['iso']['flagged']} debits and caught only {R['iso']['recall']*100:.1f}% of fraud, so it is best used to prioritise manual review when labels are unavailable, not as a stand-alone detector.",
]
R["insights"] = insights
for i, t in enumerate(insights, 1):
    print(f"{i}. {t}")

# %% [markdown]
# ## 21. Conclusion

# %%
conclusion = (
    f"The analysis of {R['clean_rows']:,} cleaned transactions from {R['accounts']} accounts shows regular, "
    f"salary-driven inflows and category-wise spending. K-Means produced three interpretable customer tiers "
    f"(silhouette {R['kmeans_final']['silhouette']}); Isolation Forest and Random Forest both detected unusual "
    f"transactions, with {best} performing best on F1. Results should be re-validated on real bank data."
)
R["conclusion"] = conclusion
print(conclusion)

with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
    json.dump(R, fh, indent=2, default=str)
print("\nAll results saved to", os.path.join(OUT_DIR, "results.json"))
