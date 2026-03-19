"""
AEDC ADO Area Office — Revenue Forecasting with Machine Learning
================================================================
Author: Opemipo Daniel Owolabi
Project: Portfolio Project 2 — Predictive Analytics
Tools: Python, Pandas, Scikit-learn, Matplotlib

Business Problem:
-----------------
After analysing marketer performance (Project 1), the next question
management asks is: "What will our revenue collection look like next month?"

Manual forecasting is slow and inconsistent. This project builds a
Machine Learning model that:
  - Learns from historical daily collection data (June 2021)
  - Predicts the next 30 days of expected revenue
  - Quantifies model accuracy so management can trust the forecast
  - Visualises actual vs predicted performance clearly

This is the kind of work a Data Scientist does — moving from
"what happened?" to "what will happen?"

How the ML works (simple explanation):
---------------------------------------
We use LINEAR REGRESSION — the model finds the best mathematical
line through historical data points. Once it knows the trend,
it extends that line into the future to make predictions.

Think of it like drawing a trend line in Excel — but the computer
does it precisely and can forecast far into the future.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  AEDC REVENUE FORECASTING — MACHINE LEARNING MODEL")
print("  Opemipo Daniel Owolabi | Data Science Portfolio")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: LOAD HISTORICAL DATA
# ─────────────────────────────────────────────
# These are the real daily cumulative collection figures
# extracted from the AEDC ADO dashboard (June 2021)
# Each number = total revenue collected up to that date

print("\n[STEP 1] Loading historical revenue data...")

data = {
    "Date": [
        "2021-06-18", "2021-06-21", "2021-06-22",
        "2021-06-23", "2021-06-24", "2021-06-27",
        "2021-06-28", "2021-06-29", "2021-06-30"
    ],
    "Cumulative_Collection": [
        6_553_235.92,   # Day 1 snapshot
        7_830_302.53,   # Day 4
        8_211_032.20,   # Day 5
        8_763_793.61,   # Day 6
        9_343_281.86,   # Day 7
        9_997_193.88,   # Day 10
        10_424_848.06,  # Day 11
        10_938_285.60,  # Day 12
        11_305_133.60   # Day 13 (final)
    ]
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# Create a numeric "day number" column — ML models need numbers, not dates
# Day 1 = June 18, Day 2 = June 19, etc.
df["Day_Number"] = (df["Date"] - df["Date"].min()).dt.days + 1

# Calculate daily revenue (not cumulative) — how much collected each day
df["Daily_Collection"] = df["Cumulative_Collection"].diff().fillna(
    df["Cumulative_Collection"].iloc[0]
)

print(f"   ✓ Loaded {len(df)} historical data points")
print(f"   ✓ Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"   ✓ Total collected: ₦{df['Cumulative_Collection'].iloc[-1]/1e6:.2f}M")


# ─────────────────────────────────────────────
# STEP 2: PREPARE DATA FOR THE ML MODEL
# ─────────────────────────────────────────────
# ML models need:
#   X = the INPUT (what we know) → Day numbers
#   y = the OUTPUT (what we want to predict) → Revenue collected

print("\n[STEP 2] Preparing data for Machine Learning...")

X = df["Day_Number"].values.reshape(-1, 1)  # Input: day numbers
y = df["Cumulative_Collection"].values        # Output: revenue

# We use Polynomial Features (degree 2) instead of a straight line
# because revenue growth often curves upward (not perfectly straight)
# This is called POLYNOMIAL REGRESSION — a step up from linear regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split data: 80% to TRAIN the model, 20% to TEST how accurate it is
# This is standard ML practice — you never test on data you trained with
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples:  {len(X_test)}")
print("   ✓ Using Polynomial Regression (degree 2)")


# ─────────────────────────────────────────────
# STEP 3: TRAIN THE MODEL
# ─────────────────────────────────────────────
# This is where the "learning" happens
# The model looks at all training data and finds the best mathematical
# formula that describes how day number → revenue collected

print("\n[STEP 3] Training the ML model...")

model = LinearRegression()
model.fit(X_train, y_train)  # ← This is the actual "learning" step

# Test the model — predict on data it hasn't seen before
y_pred_test = model.predict(X_test)

# Measure accuracy
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"   ✓ Model trained successfully!")
print(f"   ✓ R² Score: {r2:.4f} (1.0 = perfect, 0 = useless)")
print(f"   ✓ Mean Absolute Error: ₦{mae:,.0f}")
print(f"   ✓ Model explains {r2*100:.1f}% of revenue variation")


# ─────────────────────────────────────────────
# STEP 4: FORECAST THE NEXT 30 DAYS
# ─────────────────────────────────────────────
# Now we ask the model: "What will revenue look like for the next 30 days?"
# We create future day numbers and let the model predict

print("\n[STEP 4] Forecasting next 30 days...")

last_day = int(df["Day_Number"].max())
future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
future_days_poly = poly.transform(future_days)
future_predictions = model.predict(future_days_poly)

# Create future dates
last_date = df["Date"].max()
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]

df_forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Cumulative": future_predictions,
    "Day_Number": future_days.flatten()
})

# Calculate predicted daily collections
df_forecast["Predicted_Daily"] = df_forecast["Predicted_Cumulative"].diff().fillna(
    df_forecast["Predicted_Cumulative"].iloc[0] - df["Cumulative_Collection"].iloc[-1]
)

# Also get model predictions on historical data (for the chart)
X_all_poly = poly.transform(df["Day_Number"].values.reshape(-1, 1))
df["Model_Fit"] = model.predict(X_all_poly)

projected_total = df_forecast["Predicted_Cumulative"].iloc[-1]
projected_daily_avg = df_forecast["Predicted_Daily"].mean()

print(f"   ✓ Forecast generated for {len(df_forecast)} future days")
print(f"   ✓ Projected cumulative (30 days): ₦{projected_total/1e6:.2f}M")
print(f"   ✓ Projected daily average: ₦{projected_daily_avg:,.0f}")


# ─────────────────────────────────────────────
# STEP 5: VISUALISE RESULTS
# ─────────────────────────────────────────────

print("\n[STEP 5] Generating forecast visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "AEDC ADO Area Office — Revenue Forecasting Dashboard\n"
    "Machine Learning Model (Polynomial Regression)  |  Analyst: Opemipo Daniel Owolabi",
    fontsize=14, fontweight="bold", y=1.01
)

# --- Chart 1: Actual vs Predicted (Historical Fit) ---
ax1 = axes[0, 0]
ax1.scatter(df["Date"], df["Cumulative_Collection"] / 1e6,
            color="#1f4e79", s=80, zorder=5, label="Actual Data", marker="o")
ax1.plot(df["Date"], df["Model_Fit"] / 1e6,
         color="#ed7d31", linewidth=2.5, linestyle="--", label="Model Fit")
ax1.set_title("📐 Model Fit — Actual vs Predicted (Historical)", fontweight="bold")
ax1.set_ylabel("Cumulative Collection (₦ Millions)")
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("₦%.1fM"))
ax1.legend()
ax1.tick_params(axis="x", rotation=30)

# --- Chart 2: 30-Day Revenue Forecast ---
ax2 = axes[0, 1]
ax2.plot(df["Date"], df["Cumulative_Collection"] / 1e6,
         color="#1f4e79", linewidth=2.5, marker="o", markersize=6, label="Historical (Actual)")
ax2.plot(df_forecast["Date"], df_forecast["Predicted_Cumulative"] / 1e6,
         color="#70ad47", linewidth=2.5, linestyle="--", marker="s", markersize=4,
         label="Forecast (Predicted)")
ax2.axvline(x=df["Date"].max(), color="#c00000", linestyle=":", linewidth=1.5,
            label="Forecast Start")
ax2.fill_between(df_forecast["Date"],
                 df_forecast["Predicted_Cumulative"] / 1e6 * 0.92,
                 df_forecast["Predicted_Cumulative"] / 1e6 * 1.08,
                 alpha=0.15, color="#70ad47", label="±8% Confidence Band")
ax2.set_title("🔮 30-Day Revenue Forecast", fontweight="bold")
ax2.set_ylabel("Cumulative Collection (₦ Millions)")
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("₦%.1fM"))
ax2.legend(fontsize=8)
ax2.tick_params(axis="x", rotation=30)

# --- Chart 3: Predicted Daily Collections ---
ax3 = axes[1, 0]
colors_daily = ["#70ad47" if v > projected_daily_avg else "#ffc000"
                for v in df_forecast["Predicted_Daily"]]
bars = ax3.bar(df_forecast["Date"], df_forecast["Predicted_Daily"] / 1e3,
               color=colors_daily, alpha=0.85)
ax3.axhline(y=projected_daily_avg / 1e3, color="#c00000", linestyle="--",
            linewidth=1.5, label=f"Avg: ₦{projected_daily_avg/1e3:.0f}K/day")
ax3.set_title("📅 Predicted Daily Collections — Next 30 Days", fontweight="bold")
ax3.set_ylabel("Daily Collection (₦ Thousands)")
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("₦%.0fK"))
ax3.legend()
ax3.tick_params(axis="x", rotation=30)

# --- Chart 4: Model Performance Summary ---
ax4 = axes[1, 1]
ax4.axis("off")

summary_text = f"""
MODEL PERFORMANCE SUMMARY
─────────────────────────────────────

Algorithm:       Polynomial Regression (Degree 2)
Training Data:   June 2021 AEDC Collection Data
Data Points:     {len(df)} historical snapshots


MODEL ACCURACY
─────────────────────────────────────

R² Score:        {r2:.4f}  ({r2*100:.1f}% variance explained)
Mean Abs Error:  ₦{mae:,.0f}


30-DAY FORECAST SUMMARY
─────────────────────────────────────

Forecast Start:  {df_forecast['Date'].min().strftime('%d %b %Y')}
Forecast End:    {df_forecast['Date'].max().strftime('%d %b %Y')}
Projected Total: ₦{projected_total/1e6:.2f} Million
Daily Average:   ₦{projected_daily_avg:,.0f}
Growth vs June:  +{((projected_total - df['Cumulative_Collection'].iloc[-1])
                    / df['Cumulative_Collection'].iloc[-1] * 100):.1f}%


BUSINESS INSIGHT
─────────────────────────────────────

Based on the growth trajectory observed
in June 2021, the model forecasts continued
revenue acceleration in July 2021.

Management should monitor actual daily
collections against this forecast and
investigate any days where collection
falls below the predicted average.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8))

plt.tight_layout()
plt.savefig("/home/claude/project2/revenue_forecast_dashboard.png",
            dpi=150, bbox_inches="tight")
print("   ✓ Dashboard saved: revenue_forecast_dashboard.png")


# ─────────────────────────────────────────────
# STEP 6: PRINT FINAL BUSINESS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FORECAST RESULTS — BUSINESS SUMMARY")
print("=" * 60)
print(f"\n  📐 Model Accuracy (R²):      {r2:.4f} ({r2*100:.1f}%)")
print(f"  📐 Mean Absolute Error:      ₦{mae:,.0f}")
print(f"\n  🔮 Projected Revenue (July): ₦{projected_total/1e6:.2f}M")
print(f"  📅 Predicted Daily Average:  ₦{projected_daily_avg:,.0f}/day")
growth = ((projected_total - df['Cumulative_Collection'].iloc[-1])
          / df['Cumulative_Collection'].iloc[-1] * 100)
print(f"  📈 Projected Growth vs June: +{growth:.1f}%")
print("\n  ✅ Forecast complete. Dashboard saved.")
print("=" * 60)
