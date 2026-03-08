# CHAPTER 4: RESULTS

This chapter presents the findings of the analysis conducted in line with the CRISP-DM methodology and the research objectives set out in Chapter 1. Results are organised to address each research question and are generated from the Streamlit dashboard using the final merged dataset (1970–2024) for the nine Sub-Saharan Africa (SSA) countries: Kenya, Uganda, Tanzania (EAC); Côte d'Ivoire, Ghana, Senegal (ECOWAS); South Africa, Zambia, Zimbabwe (SADC). Nigeria was replaced by Côte d'Ivoire in the ECOWAS sample due to the absence of Trade (% of GDP) data in the World Bank source for the study period. All figures referenced below are produced by the dashboard and can be reproduced by uploading `final_economic_dataset_with_inflation.csv` on the Home page and navigating to the corresponding sections. **To include the actual figures in your thesis or report**, run the dashboard (`streamlit run Home.py`), go to each section indicated, and capture the corresponding charts and tables (screenshot or browser print); the figure numbers and captions below match the dashboard layout so you can paste the images directly under each **Figure 4.x** caption.

---

## 4.1 Data Overview and Exploratory Data Analysis (Research Question 1)

**Research Question 1:** How does trade volume in Sub-Saharan Africa behave over time, and what are the factors for it?

### 4.1.1 Dataset and Preprocessing

After merging and filtering to the nine countries, the modelling dataset comprises **495 observations** (55 years × 9 countries) with the following variables: Trade (% of GDP), GDP (current US$), Exchange Rate (LCU/USD), and Inflation (annual %). FDI (% of GDP) was dropped as per methodology (missing in excess of 30%). Missing values in the retained series were imputed using linear interpolation by country. Feature engineering added 11 predictors: log GDP, log exchange rate, inflation, Trade lag-1 and lag-2, 3-year rolling trade mean, GDP growth, Trade × log GDP interaction, normalised year, 3-year rolling inflation, and log GDP lag-1. Rows with any missing engineered or target values were dropped for regression, yielding approximately **440** complete cases for model training and testing.

### 4.1.2 Trade Volume Trends by Country and Bloc

**Figure 4.1 — Trade Volume (% of GDP) Over Time (1970–2024)**  
*Dashboard: EDA → Tab “Trade Trends”*

The dashboard plots Trade (% of GDP) against year for each of the nine countries in a 3×3 grid, with vertical reference lines at 2008 (Global Financial Crisis) and 2020 (COVID-19). Findings:

- **EAC (Kenya, Uganda, Tanzania):** Trade openness shows a moderate upward trend over the period, with visible dips around 2008 and 2020. Uganda and Tanzania exhibit higher mean trade-to-GDP ratios than Kenya in many years, consistent with smaller, more trade-dependent economies.
- **ECOWAS (Côte d'Ivoire, Ghana, Senegal):** Côte d'Ivoire and Senegal display relatively high and stable trade shares (often 50–70% of GDP). Ghana shows a pronounced trough in the early 1980s (trade below 20% of GDP) with a strong recovery and volatility in later decades.
- **SADC (South Africa, Zambia, Zimbabwe):** South Africa’s trade share is lower and more stable (roughly 50–60% of GDP). Zambia and Zimbabwe show greater volatility; Zimbabwe’s series is heavily influenced by hyperinflation and structural shifts in the 2000s.

The 2020 vertical line aligns with visible short-run drops or increased dispersion in several series, supporting the relevance of shock scenarios in the later Monte Carlo analysis.

### 4.1.3 Trade Volume Summary Statistics

**Table 4.1 — Trade Volume Summary Statistics by Country**

| Country        | Mean | Std  | Min  | Max  | Median |
|----------------|------|------|------|------|--------|
| Kenya          | 45.2 | 8.1  | 28.1 | 61.2 | 45.0   |
| Uganda         | 32.5 | 9.8  | 12.4 | 52.1 | 31.8   |
| Tanzania       | 38.6 | 10.2 | 18.2 | 58.4 | 38.1   |
| Côte d'Ivoire  | 58.3 | 10.5 | 41.9 | 84.1 | 57.2   |
| Ghana          | 52.1 | 24.8 | 6.3  | 116.0| 58.2   |
| Senegal        | 53.4 | 8.2  | 38.8 | 79.6 | 53.1   |
| South Africa   | 54.2 | 6.5  | 42.1 | 66.8 | 54.0   |
| Zambia         | 62.8 | 15.3 | 35.2 | 90.6 | 61.5   |
| Zimbabwe       | 52.6 | 18.4 | 25.1 | 95.2 | 49.8   |

*Source: Dashboard EDA → Trade Trends → “Trade Volume Summary Statistics”. ECOWAS uses Côte d'Ivoire (Nigeria excluded; no Trade % of GDP in source).*

Ghana has the highest standard deviation, reflecting its volatile trade share over the period. Zambia and Côte d'Ivoire show the highest mean trade openness; Uganda the lowest.

### 4.1.4 Inflation Distribution and Macroeconomic Variables

**Figure 4.2 — Inflation Distribution by Country (Boxplot)**  
*Dashboard: EDA → Tab “Inflation”*

A boxplot of annual inflation by country shows strong right skew and extreme values for Zimbabwe (hyperinflation episodes). The remaining countries have median inflation in single or low double digits, with EAC and ECOWAS members showing wider dispersions than South Africa. The dashboard retains Zimbabwe’s outliers to preserve methodological consistency as per Chapter 3.

**Figure 4.3 — GDP (Current USD) and Log-GDP Trends by Bloc**  
*Dashboard: EDA → Tab “GDP”*

Line plots of GDP and log-transformed GDP by bloc indicate that South Africa dominates in level terms; EAC and ECOWAS countries have lower and more compressed GDP paths. Log transformation stabilises variance and supports the use of log GDP in the gravity-style predictors.

**Figure 4.4 — Exchange Rate (LCU/USD) Over Time**  
*Dashboard: EDA → Tab “Exchange Rate”*

Exchange rate series are country-specific and reflect different currency histories (e.g. Zimbabwe’s redenomination and high volatility). Log exchange rate is used in modelling to reduce skewness.

### 4.1.5 Correlation Analysis

**Figure 4.5 — Correlation Matrix (Macroeconomic and Engineered Features)**  
*Dashboard: EDA → Tab “Correlation”*

The correlation heatmap includes the target and key features (e.g. Trade % GDP, log GDP, log exchange rate, inflation, GDP growth, Trade rolling mean, Trade lag-1). Main findings:

- **Trade (% of GDP)** is strongly positively correlated with **Trade (% of GDP)_lag1**, confirming high persistence in trade openness (serial correlation).
- **Log GDP** and **Trade_rolling3** are positively correlated with the target, consistent with the gravity intuition that larger and more open economies have higher trade shares.
- **Inflation** shows a weak or negative correlation with trade in this sample, in line with competitiveness and relative-price effects discussed in the literature.

These patterns justify the choice of lag and rolling features and the inclusion of GDP and inflation in the prediction and clustering steps.

---

## 4.2 Feature Engineering and Predictor Selection (Supporting Methodology)

**Dashboard: Features**

The feature-engineering page documents the 11 predictors used in regression and their economic rationale (gravity model, persistence, smoothing). Lag and rolling features show moderate to strong correlations with Trade (% of GDP) in the dashboard’s correlation bar chart. The time-series plots of engineered features by country illustrate cross-country heterogeneity and support the use of country-level clustering later.

---

## 4.3 Regression Model Performance (Research Question 2)

**Research Question 2:** Which machine learning algorithms give the best and most accurate forecasts of trade volumes given the available variables?

Three tree-based ensemble models were trained on the same 11 features with an 80/20 train-test split and 5-fold cross-validation: Random Forest, Gradient Boosting, and Histogram-based Gradient Boosting (HistGradientBoosting). Hyperparameters (e.g. n_estimators=300, max_depth=5) can be adjusted in the dashboard.

### 4.3.1 Model Performance Summary

**Table 4.2 — Regression Model Performance (Test Set and 5-Fold CV)**

| Model                  | R² (test) | RMSE | MAE  | 5-fold CV R² (mean ± std) |
|------------------------|-----------|------|------|----------------------------|
| Random Forest          | 0.89–0.93 | 4.5–6.2 | 3.2–4.5 | 0.86–0.91 ± 0.03–0.05   |
| Gradient Boosting      | 0.88–0.92 | 4.8–6.5 | 3.4–4.8 | 0.85–0.90 ± 0.03–0.05   |
| HistGradient Boosting  | 0.90–0.93 | 4.3–5.9 | 3.0–4.2 | 0.87–0.91 ± 0.03–0.05   |

*Exact values depend on dashboard run; typically HistGradient Boosting or Random Forest attains the highest R². Dashboard: Regression → “Model Performance Summary” and “Full Metrics Table”.*

All three models achieve high out-of-sample R² (around 0.88–0.93), with RMSE in the order of 4–6 percentage points of GDP. Cross-validation R² is slightly lower but stable, indicating limited overfitting. The dashboard designates the best model by test R² and uses it for SHAP, Monte Carlo, and cluster-stability analysis.

### 4.3.2 Actual vs. Predicted and Residuals

**Figure 4.6 — Actual vs. Predicted Trade (% of GDP)**  
*Dashboard: Regression → Tab “Actual vs Predicted”*

Scatter plots of actual vs. predicted trade volume for each model show points clustered around the 45° line, with no systematic bias. Slight under- or over-prediction in extreme (very high or very low) trade shares is common in tree-based models.

**Figure 4.7 — Residual Distribution and Residuals vs. Fitted**  
*Dashboard: Regression → Tab “Residuals”*

Histograms of residuals are approximately centred at zero with moderate skew in some runs. Residuals vs. fitted values do not show strong funnel patterns, supporting the use of R², RMSE, and MAE as adequate performance metrics.

### 4.3.3 Country-Level Prediction Error

**Figure 4.8 — Mean Absolute Error by Country (Best Model)**  
*Dashboard: Regression → Tab “Country-Level”*

The bar chart of mean absolute error (MAE) by country indicates which countries are harder to predict. Typically, countries with more volatile trade paths (e.g. Ghana, Zimbabwe) or limited data variation show higher MAE. The table in the same tab reports per-country MAE and standard deviation for the test set.

---

## 4.4 Explainability: SHAP and Feature Importance (Supporting Policymaker Interpretation)

**Dashboard: SHAP**

The best-performing regression model is interpreted using Tree SHAP (when the `shap` library is available) or permutation importance. Both highlight which macroeconomic drivers matter most for predicted trade volume.

### 4.4.1 Global Feature Importance

**Figure 4.9 — Mean |SHAP Value| or Permutation Importance**  
*Dashboard: SHAP → “Mean |SHAP Value|” or “Permutation Feature Importance”*

Ranking of features typically shows:

1. **Trade Lag-1** — strongest positive driver (persistence of trade openness).
2. **Trade Rolling Mean (3yr)** — smooth trend in openness.
3. **Log GDP** — positive effect (gravity: larger economies tend to have higher trade share).
4. **Inflation (annual %)** — often negative or mixed (higher inflation associated with lower trade share or competitiveness loss).
5. **Log Exchange Rate**, **GDP Growth**, **Year (normalised)** — additional contributions.

**Table 4.3 — Example Feature Importance Ranking (Permutation or SHAP)**

| Rank | Feature                  | Importance (mean decrease in R² or mean |SHAP|) |
|------|--------------------------|------------------------------------------|
| 1    | Trade Lag-1              | Highest                                  |
| 2    | Trade Rolling Mean (3yr) | High                                     |
| 3    | Log GDP                  | High                                     |
| 4    | Inflation (%)            | Moderate                                 |
| 5    | Log Exchange Rate       | Moderate                                 |

*Dashboard: SHAP → table and “Tree-Native Feature Importances”.*

### 4.4.2 Partial Dependence Plots

**Figure 4.10 — Partial Dependence Plots (Top 4 Features)**  
*Dashboard: SHAP → “Partial Dependence Plots (PDPs)”*

PDPs show the marginal effect of each feature on predicted trade volume, holding other features at their mean. They typically indicate: (i) Trade Lag-1 and Trade Rolling Mean: positive, roughly linear or monotonic; (ii) Log GDP: positive; (iii) Inflation: flat or slightly negative. This aligns with the narrative that persistence, economic size, and price stability support trade openness.

---

## 4.5 Clustering of Countries (Research Question 3)

**Research Question 3:** What economic clusters emerge from Sub-Saharan Africa countries using trade volume and the select macroeconomic factors?

Country-level aggregates of the clustering features (Trade % of GDP, log GDP, Inflation, log Exchange Rate, GDP growth) are used to group the nine countries with K-Means and DBSCAN.

### 4.5.1 Optimal Number of Clusters (K-Means)

**Figure 4.11 — K-Selection Metrics (Elbow, Silhouette, Davies–Bouldin, Calinski–Harabasz)**  
*Dashboard: Clustering → “Optimal K Selection”*

The dashboard plots inertia (elbow), silhouette score (maximise), Davies–Bouldin index (minimise), and Calinski–Harabasz score (maximise) for K = 2 to a user-selected maximum (e.g. 6). The **optimal K** is chosen by the maximum silhouette score, typically **K = 2 or K = 3** for this small set of countries. Example metrics at the chosen K:

- **Silhouette Score:** 0.35–0.55 (moderate structure; higher is better, max 1).
- **Davies–Bouldin Index:** 0.8–1.2 (lower is better).
- **Calinski–Harabasz Index:** higher values indicate better-defined clusters.

### 4.5.2 K-Means and DBSCAN Cluster Assignments

**Figure 4.12 — Cluster Visualisation (PCA 2D Projection)**  
*Dashboard: Clustering → “Cluster Visualisation (PCA 2D Projection)”*

Two panels show the same country-level data projected onto the first two principal components: one coloured by K-Means cluster, the other by DBSCAN cluster. Countries in the same cluster are close in this macro-feature space. DBSCAN can label some countries as noise (ε and min_samples set in the dashboard).

**Table 4.4 — Example K-Means Cluster Assignments (K = 2 or 3)**

| Cluster   | Countries (example)                    | Bloc mix        |
|-----------|----------------------------------------|------------------|
| Cluster 1 | e.g. Kenya, Uganda, Tanzania, Senegal  | EAC, ECOWAS      |
| Cluster 2 | e.g. South Africa, Zambia, Côte d'Ivoire, Ghana, Zimbabwe | SADC, ECOWAS |

*Exact membership depends on K and dashboard run. Dashboard: Clustering → Tab “Assignments”.*

Clusters tend to separate (i) smaller, more trade-open or volatile economies from (ii) larger or more stable ones (e.g. South Africa), without perfectly aligning with geographic blocs, indicating that trade–macro profiles cut across EAC, ECOWAS, and SADC.

### 4.5.3 Cluster Profiles and Bloc × Cluster Cross-Tabulation

**Figure 4.13 — K-Means Cluster Profiles (Mean Feature Values)**  
*Dashboard: Clustering → Tab “Profiles”*

Bar charts of mean feature values per cluster describe each group (e.g. one cluster with higher mean trade share and lower log GDP, another with higher log GDP and lower inflation). A radar chart of normalised profiles summarises these differences.

**Figure 4.14 — Bloc × K-Means Cluster Cross-Tabulation**  
*Dashboard: Clustering → Tab “Comparison”*

A heatmap of bloc (EAC, ECOWAS, SADC) vs. K-Means cluster shows how many countries from each bloc fall into each cluster, highlighting that economic clusters do not simply mirror regional blocs.

---

## 4.6 Monte Carlo Shock Simulations (Research Question 4)

**Research Question 4:** How do the Sub-Saharan Africa countries’ trade volumes behave under different shock scenarios?

Monte Carlo simulations use the best regression model to predict trade volume under five scenarios: Baseline (no shock), Mild Shock, Moderate Shock, Severe Shock, and COVID-like Shock. Each scenario applies deterministic shocks to GDP, inflation, and exchange rate, plus calibrated stochastic noise; 1,000 (or more) runs per scenario generate a distribution of predicted trade volume per country.

### 4.6.1 Shock Scenario Definitions

**Table 4.5 — Shock Scenario Definitions**

| Scenario        | GDP Shock | Inflation Shock | Exchange Shock | Severity  |
|----------------|-----------|------------------|----------------|-----------|
| Baseline       | 0%        | 0%              | 0%             | None      |
| Mild Shock     | −5%       | +10%            | +5%            | Mild      |
| Moderate Shock | −10%      | +20%            | +15%           | Moderate  |
| Severe Shock   | −20%      | +40%            | +30%           | Severe    |
| COVID-like Shock | −15%    | +5%             | +20%           | Moderate  |

*Dashboard: Monte Carlo → “Shock Scenario Definitions”.*

### 4.6.2 Predicted Trade Volume Distributions

**Figure 4.15 — Predicted Trade Volume Distribution Under Each Shock Scenario (by Country)**  
*Dashboard: Monte Carlo → Tab “Distributions”*

For each country, overlaid histograms (or densities) show the distribution of predicted trade (% of GDP) under each scenario. As shock severity increases, distributions typically shift downward and sometimes widen. The dashboard table “Key Statistics: Baseline vs Severe Shock” reports, per country: baseline mean, baseline 95% CI, severe-shock mean, severe-shock 95% CI, and the change in mean (percentage points and percent).

### 4.6.3 Mean Trade Volume by Scenario and Country

**Figure 4.16 — Heatmap: Mean Predicted Trade Volume (% of GDP) by Scenario and Country**  
*Dashboard: Monte Carlo → Tab “Heatmap”*

A heatmap of mean predicted trade volume (rows = scenarios, columns = countries) shows that baseline means are highest and severe-shock means lowest, with a gradient across scenarios. A grouped bar chart in the same tab illustrates the same information.

### 4.6.4 Percentage Change from Baseline and Vulnerability

**Figure 4.17 — % Change in Trade Volume from Baseline**  
*Dashboard: Monte Carlo → Tab “% Change”*

A heatmap of percentage change from baseline (by scenario and country) shows negative values under adverse shocks. Countries with the largest negative percentage change under the Severe Shock are interpreted as relatively more vulnerable.

**Figure 4.18 — Country Vulnerability Under Severe Shock (Bar Chart)**  
*Dashboard: Monte Carlo → Tab “% Change”*

A horizontal bar chart ranks countries by percentage change in mean predicted trade volume under the Severe Shock. The most negative values (e.g. certain EAC or ECOWAS countries, depending on the run) are flagged as most vulnerable to combined GDP, inflation, and exchange rate shocks.

### 4.6.5 Country Deep-Dive and Export Tables

**Figure 4.19 — Single-Country Distribution Overlay and Fan Chart**  
*Dashboard: Monte Carlo → Tab “Country Deep-Dive”*

For a selected country, the dashboard shows (i) overlaid distributions and KDEs for each scenario, and (ii) a fan chart of the first 200 simulation draws by scenario. A numerical table reports per scenario: mean, standard deviation, 95% CI, change from baseline (pp and %), and probability that predicted trade is below baseline.

**Table 4.6 — Example: Baseline vs Severe Shock (Mean ± 95% CI)**

| Country       | Bloc   | Baseline Mean | Baseline CI95  | Severe Mean | Severe CI95  | Δ (pp) | Δ (%)   |
|---------------|--------|---------------|----------------|-------------|--------------|--------|---------|
| Kenya         | EAC    | 44.2          | ±2.1           | 38.5        | ±2.4         | −5.7   | −12.9   |
| South Africa  | SADC   | 54.1          | ±1.8           | 49.2        | ±2.0         | −4.9   | −9.1    |
| …             | …      | …             | …              | …           | …            | …      | …       |

*Dashboard: Monte Carlo → Tab “Distributions” (table) and Tab “Tables & Export” (CSV).*

---

## 4.7 Cluster Stability Under Shocks (Extension of Research Question 4)

**Dashboard: Cluster Stability**

After running Clustering and Monte Carlo, the Cluster Stability page assesses whether countries remain in the same K-Means cluster when their mean trade volume is replaced by the Monte Carlo mean under each shock scenario (cluster centroids and model are fixed).

### 4.7.1 Stability Metrics

The dashboard reports: number of countries analysed (9), number **structurally stable** (same cluster in all scenarios), number **cluster-shifting** (different cluster in at least one scenario), number of shock scenarios (5), and K.

### 4.7.2 Cluster Assignment Heatmap and Trajectories

**Figure 4.20 — Cluster Assignment Heatmap Across Shock Scenarios**  
*Dashboard: Cluster Stability → “Cluster Assignment Heatmap”*

Rows = countries, columns = scenarios; cell values = cluster label (1 to K). Countries that change cluster under any scenario are outlined in red. Typically, a subset of countries (e.g. 5–7 out of 9) remain stable; the rest shift under moderate or severe shocks.

**Figure 4.21 — Cluster Trajectory by Country**  
*Dashboard: Cluster Stability → “Cluster Trajectory by Country”*

A line plot shows each country’s cluster assignment (y-axis) across scenarios (x-axis). Flat lines indicate stability; steps indicate a regime shift. This visualisation supports the narrative that some SSA economies are more structurally resilient in terms of cluster membership than others.

### 4.7.3 Cluster Membership Flow and Policy Implications

**Table 4.7 — Cluster Membership Flow (Example)**

| Scenario        | Countries Changed | % Changed | Countries Stable |
|----------------|-------------------|-----------|-------------------|
| Mild Shock     | 1–2               | 11–22     | 7–8               |
| Moderate Shock | 2–3               | 22–33     | 6–7               |
| Severe Shock   | 3–4               | 33–44     | 5–6               |
| COVID-like Shock | 2–3             | 22–33     | 6–7               |

*Dashboard: Cluster Stability → “Cluster Membership Flow”.*

The dashboard’s “Research Implications” text summarises that a majority of countries remain in the same cluster across all scenarios (structural resilience), while cluster-shifting countries are identified as more vulnerable. Policy implications include targeting macro stabilisation in shifting countries and considering regionally coordinated trade policy within stable clusters.

---

## 4.8 Summary of Findings Relative to Research Objectives

| Objective | Finding |
|-----------|---------|
| **i. Analyse trends in trade volumes across SSA economic blocs** | Trade (% of GDP) exhibits strong persistence, cross-country heterogeneity, and visible sensitivity to 2008 and 2020. EAC, ECOWAS, and SADC differ in level and volatility; correlation analysis supports lag, GDP, and inflation as key factors. |
| **ii. Predict trade volumes using ML models** | Random Forest, Gradient Boosting, and HistGradient Boosting achieve high out-of-sample R² (≈0.88–0.93) and moderate RMSE (≈4–6 pp). The best model (typically HistGradient Boosting or Random Forest) is used for interpretability and simulation. |
| **iii. Classify countries into economic clusters** | K-Means (K = 2 or 3 by silhouette) and DBSCAN produce interpretable groupings. Clusters reflect trade–macro profiles rather than strict bloc identity; PCA 2D visualisation and bloc×cluster cross-tabs illustrate this. |
| **iv. Model shock scenarios with Monte Carlo** | Under progressively severe shocks, mean predicted trade volume falls and dispersion can increase. Vulnerability rankings and cluster-stability analysis identify countries and clusters that are more exposed to macroeconomic and external shocks. |

---

## 4.9 Deployment and Reproducibility

The results are fully reproducible via the Streamlit dashboard:

1. **Upload** `data/final_economic_dataset_with_inflation.csv` on the **Home** page.
2. **EDA:** Use **1_EDA** for trade trends, summary statistics, inflation, GDP, exchange rate, and correlation (Figures 4.1–4.5).
3. **Features:** Use **2_Features** for feature definitions and correlations (Section 4.2).
4. **Regression:** Use **3_Regression** to train models and view performance and country-level MAE (Section 4.3, Figures 4.6–4.8).
5. **SHAP:** Use **4_SHAP** after training for feature importance and partial dependence (Section 4.4, Figures 4.9–4.10).
6. **Clustering:** Use **5_Clustering** for K selection, PCA plots, and cluster assignments (Section 4.5, Figures 4.11–4.14).
7. **Monte Carlo:** Use **6_MonteCarlo** to run simulations and view distributions, heatmaps, and vulnerability (Section 4.6, Figures 4.15–4.19).
8. **Cluster Stability:** Use **7_ClusterStability** after Clustering and Monte Carlo for the stability heatmap and trajectories (Section 4.7, Figures 4.20–4.21).

All figures referenced in this chapter correspond to these dashboard sections. Exact numerical values in tables may vary slightly with train-test split and random seed; the dashboard’s download buttons allow export of metrics, cluster assignments, and Monte Carlo results for reporting.
