"""
NOTE: THIS IS AN OLD VERSION OF THE CODE.
Kept for now for reference only.

Data preprocessing is done in the Data_Pre_Processing.ipynb file.
Actual Q2 code is in the Q2.ipynb file.

"""

"""
RSM8341 Assignment 1 - Section 2: The Security Market Line

This script estimates CAPM betas for S&P 500 constituents using 10 years
of monthly excess returns, then plots individual stock scatter plots and
the Security Market Line (SML) to identify abnormal performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")


# Load pre-processed data
stock_excess = pd.read_csv(
    "./processed_data/stock_excess_monthly.csv", index_col="Date", parse_dates=True
)
mkt_excess = pd.read_csv(
    "./processed_data/mkt_excess_monthly.csv", index_col="Date", parse_dates=True
)
rf_m = pd.read_csv(
    "./processed_data/rf_monthly.csv", index_col="observation_date", parse_dates=True
)

# Run CAPM OLS regression for each stock: excess_ret = alpha + beta * mkt_excess + error
X = pd.DataFrame({"const": 1.0, "mkt": mkt_excess.values}, index=mkt_excess.index)

results = {}
for ticker in stock_excess.columns:
    y = stock_excess[ticker]
    mask = y.notna()
    # Require at least 24 months of data for a reliable estimate
    if mask.sum() < 24:
        continue
    try:
        model = sm.OLS(y[mask], X[mask]).fit()
        results[ticker] = {
            "alpha": model.params["const"],
            "beta": model.params["mkt"],
            "r2": model.rsquared,
            "se_beta": model.bse["mkt"],
            "tstat_alpha": model.tvalues["const"],
            "pval_alpha": model.pvalues["const"],
            "avg_excess_ret": y[mask].mean(),
        }
    except Exception:
        continue

capm_df = pd.DataFrame(results).T.sort_values("beta")

# Summary statistics
avg_mkt_monthly = mkt_excess.mean()
avg_mkt_annual = avg_mkt_monthly * 12
avg_rf_annual = rf_m.mean() * 12
T = len(mkt_excess)
N = len(capm_df)

print(f"Stocks: {N}, Monthly obs: {T}")
print(f"Avg monthly market excess return: {avg_mkt_monthly:.6f}")
print(f"Annualized market excess return:  {avg_mkt_annual * 100:.2f}%")
print(f"Avg annual risk-free rate:         {avg_rf_annual * 100:.2f}%")

# Task (b): Scatter plots for three representative stocks (low, mid, high beta)
rep = {
    "Low-beta: PG (Procter & Gamble)": "PG",
    "Mid-beta: FTNT (Fortinet)": "FTNT",
    "High-beta: APA (APA Corp)": "APA",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "CAPM Scatter: Monthly Excess Stock Return vs. Monthly Excess Market Return",
    fontsize=12,
    fontweight="bold",
)

for ax, (label, ticker) in zip(axes, rep.items()):
    y = stock_excess[ticker].dropna()
    x = mkt_excess.loc[y.index]
    b = capm_df.loc[ticker, "beta"]
    a = capm_df.loc[ticker, "alpha"]
    r2 = capm_df.loc[ticker, "r2"]

    ax.scatter(x, y, alpha=0.5, s=20, color="steelblue", edgecolors="none")
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(
        x_line,
        a + b * x_line,
        color="crimson",
        lw=2.0,
        label=f"$\\hat{{\\beta}}$ = {b:.2f}\n$\\hat{{\\alpha}}$ = {a:.4f}\n$R^2$ = {r2:.3f}",
    )
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Excess Market Return (monthly)")
    ax.set_ylabel("Excess Stock Return (monthly)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("task_b_scatter_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("Task (b) saved.")

# Task (c): Security Market Line plot with statistically significant alpha outliers labeled
sig = capm_df[capm_df["tstat_alpha"].abs() > 2]
top5 = sig.nlargest(5, "alpha")
bot5 = sig.nsmallest(5, "alpha")

fig, ax = plt.subplots(figsize=(13, 8))

# Color stocks by sign of their estimated alpha
pos = capm_df[capm_df["alpha"] >= 0]
neg = capm_df[capm_df["alpha"] < 0]

ax.scatter(
    neg["beta"],
    neg["avg_excess_ret"],
    color="#d62728",
    alpha=0.45,
    s=18,
    label="Negative $\\alpha$",
    zorder=3,
)
ax.scatter(
    pos["beta"],
    pos["avg_excess_ret"],
    color="#2ca02c",
    alpha=0.45,
    s=18,
    label="Positive $\\alpha$",
    zorder=3,
)

# Draw the theoretical SML using the realized average market excess return
beta_range = np.linspace(capm_df["beta"].min() - 0.2, capm_df["beta"].max() + 0.2, 400)
ax.plot(
    beta_range,
    avg_mkt_monthly * beta_range,
    color="navy",
    lw=2.2,
    label=f"Theoretical SML  (annualized market excess return = {avg_mkt_annual * 100:.1f}%)",
)

# Annotate the top and bottom 5 outliers by alpha magnitude
for ticker, row in pd.concat([top5, bot5]).iterrows():
    xoff = 8 if row["beta"] < 2.0 else -45
    ax.annotate(
        ticker,
        xy=(row["beta"], row["avg_excess_ret"]),
        xytext=(xoff, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color="#222222",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
    )

ax.set_xlabel("Estimated Beta ($\\hat{\\beta}$)", fontsize=12)
ax.set_ylabel("Average Monthly Excess Return", fontsize=12)
ax.set_title(
    "Security Market Line: Realized Excess Returns vs. Estimated Beta\n"
    "S&P 500 Constituents, Monthly Data, March 2015 - March 2025",
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)


# Add a secondary right-hand axis showing the equivalent annualized return scale
def to_annual(x):
    return x * 12 * 100


def to_monthly(x):
    return x / 12 / 100


secax = ax.secondary_yaxis("right", functions=(to_annual, to_monthly))
secax.set_ylabel("Annualized Excess Return (%)", fontsize=10)

plt.tight_layout()
plt.savefig("task_c_sml_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("Task (c) saved.")

# Print outlier summary tables
print("\nStocks above the SML (positive alpha, |t| > 2):")
print(
    top5[["alpha", "beta", "r2", "se_beta", "tstat_alpha", "avg_excess_ret"]]
    .round(5)
    .to_string()
)
print("\nStocks below the SML (negative alpha, |t| > 2):")
print(
    bot5[["alpha", "beta", "r2", "se_beta", "tstat_alpha", "avg_excess_ret"]]
    .round(5)
    .to_string()
)

# Save full results and outlier tables to CSV
capm_df.to_csv("capm_results_final.csv", float_format="%.6f")
pd.concat([top5, bot5]).to_csv("outliers_final.csv", float_format="%.6f")

print("\nDone.")
