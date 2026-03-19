# 🔮 AEDC Revenue Forecasting — Machine Learning Model

**Portfolio Project 2** — Predictive Analytics using real electricity distribution data from the **Abuja Electricity Distribution Company (AEDC)** ADO Area Office.

> Built by **Opemipo Daniel Owolabi** — Data Analyst | Python · SQL · Power BI · Tableau  
> 📍 Faro, Portugal | 📧 opemipoowolabi001@gmail.com  
> 🔗 [View Project 1 — Marketer Performance Analysis](https://github.com/opemipo-analytics/AEDC-MARKETERS-ANALYTICS)

---

## 🧩 Business Problem

After analysing marketer performance (Project 1), the next question management asks is:

> *"Based on current trends, what will our revenue collection look like next month?"*

Manual forecasting is slow, inconsistent, and relies on gut feel. This project builds a **Machine Learning model** that learns from historical daily collection data and produces a reliable 30-day revenue forecast — giving management a data-driven basis for planning and target-setting.

---

## 📊 Dashboard Preview

![Revenue Forecast Dashboard](revenue_forecast_dashboard.png)

---

## 🤖 How the Machine Learning Works

This project uses **Polynomial Regression** — explained simply:

| Concept | Plain English Explanation |
|---------|--------------------------|
| **Training data** | The historical daily revenue figures from June 2021 |
| **Features (X)** | Day numbers (Day 1, Day 2, Day 3...) |
| **Target (y)** | Cumulative revenue collected |
| **Model** | Finds the best mathematical curve through the data points |
| **Prediction** | Extends that curve into the future |

Think of it like drawing a trend line in Excel — but mathematically precise and extendable months into the future.

---

## 📈 Model Results

| Metric | Value |
|--------|-------|
| Algorithm | Polynomial Regression (Degree 2) |
| R² Score | **0.9980** (99.8% accuracy) |
| Mean Absolute Error | ₦69,483 |
| Training Data | 7 data points |
| Test Data | 2 data points |

### 30-Day Forecast Summary

| Metric | Value |
|--------|-------|
| Forecast Period | July 2021 (30 days) |
| Projected Total Revenue | **₦14.82 Million** |
| Projected Daily Average | ₦117,272 / day |
| Growth vs June 2021 | **+31.1%** |

---

## 📁 Project Structure

```
aedc-revenue-forecasting/
│
├── revenue_forecast.py              # Main ML script
├── revenue_forecast_dashboard.png   # Output: 4-panel forecast dashboard
└── README.md                        # This file
```

---

## 📊 What the Dashboard Shows

### 1️⃣ Model Fit — Actual vs Predicted (Historical)
Shows how well the model fits the historical data. Orange dashed line = model's learned curve, Blue dots = real data. The closer they are, the better.

### 2️⃣ 30-Day Revenue Forecast
Historical data (blue) transitions into the forecast zone (green dashed). The shaded green band shows ±8% confidence range — management can use this as a realistic upper/lower bound.

### 3️⃣ Predicted Daily Collections — Next 30 Days
Bar chart of predicted daily revenue for each of the 30 forecast days. Green bars = above daily average, Yellow = below. The red dashed line marks the daily average target.

### 4️⃣ Model Performance Summary
A clean text panel summarising all key metrics — designed for easy inclusion in executive presentations.

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/opemipo-analytics/aedc-revenue-forecasting.git
cd aedc-revenue-forecasting

# Install dependencies
pip install pandas numpy matplotlib scikit-learn

# Run the forecast
python revenue_forecast.py
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3** | Core scripting |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Machine Learning model |
| **Matplotlib** | Visualisations |

---

## 💡 Skills Demonstrated

- **Machine Learning** — training, testing and evaluating a regression model
- **Feature Engineering** — converting dates to numeric features for ML
- **Model Evaluation** — R² score, Mean Absolute Error, train/test split
- **Business Forecasting** — translating ML output into actionable revenue projections
- **Data Storytelling** — presenting complex ML results in plain business language

---

## 🔗 Other Projects

| Project | Description |
|---------|-------------|
| [AEDC Marketer Performance Analysis](https://github.com/opemipo-analytics/AEDC-MARKETERS-ANALYTICS) | Python analysis of electricity marketer KPIs |
| SQL Customer Segmentation *(coming soon)* | SQL-based customer behaviour analysis |
| Power BI Revenue Dashboard *(coming soon)* | Interactive business intelligence dashboard |

---

*This project is part of my data analytics portfolio, built from real data during my time as a Data Analyst at the Abuja Electricity Distribution Company (AEDC).*
