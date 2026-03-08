"""3_Regression.py — Regression Modelling (RF · GBM · HistGBM)"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOC_COLORS,
    FEATURES, FEAT_DISPLAY, DISP_FEATURES, TARGET,
    dark_fig, style_ax, show, guard,
)

st.set_page_config(page_title="Regression · SSA Trade", page_icon="🤖", layout="wide")
inject_css()
guard(["df"], "⚠️ Please upload your dataset on the **Home** page first.")
df = st.session_state["df"]

st.markdown("## 🤖 Regression Modelling")
st.markdown("""
Predict trade volume (% of GDP) using three tree-based ensemble models.
All models use the same 11 engineered features, with 80/20 train-test split
and 5-fold cross-validation for robust evaluation.
""")

# ── Dataset summary ───────────────────────────────────────────────────────────
df_model = df[FEATURES + [TARGET, "Country Name", "Year"]].dropna()
st.markdown(
    f"**Modelling dataset:** {df_model.shape[0]} observations × {len(FEATURES)} features"
    f" (dropped {len(df) - df_model.shape[0]} rows with NaN)"
)

# ── Settings ──────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Hyperparameters")
c1, c2, c3, c4, c5 = st.columns(5)
test_size  = c1.slider("Test split %", 10, 30, 20, key="reg_test") / 100
n_trees    = c2.slider("n_estimators", 100, 600, 300, step=50, key="reg_n")
max_depth  = c3.slider("max_depth",    3,  12,   5,          key="reg_d")
lr         = c4.select_slider("GBM learning rate", [0.01, 0.05, 0.1, 0.2], 0.05, key="reg_lr")
run_cv     = c5.checkbox("Run 5-fold CV", value=True, key="reg_cv")

if st.button("🚀  Train All Models", key="reg_train"):
    X = df_model[FEATURES].values
    y = df_model[TARGET].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=n_trees, max_depth=max_depth,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_trees, learning_rate=lr,
            max_depth=max_depth, subsample=0.8, random_state=42
        ),
        "HistGradient Boosting": HistGradientBoostingRegressor(
            max_iter=n_trees, learning_rate=lr,
            max_depth=max_depth, random_state=42
        ),
    }

    results  = {}
    prog     = st.progress(0, "Training models…")
    for i, (name, mdl) in enumerate(models.items()):
        prog.progress(int(i / len(models) * 70), f"Training {name}…")
        mdl.fit(X_tr_s, y_tr)
        y_pred = mdl.predict(X_te_s)
        res = {
            "model":  mdl,
            "y_pred": y_pred,
            "R2":     r2_score(y_te, y_pred),
            "RMSE":   float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "MAE":    float(mean_absolute_error(y_te, y_pred)),
        }
        if run_cv:
            Xa = StandardScaler().fit_transform(df_model[FEATURES].values)
            ya = df_model[TARGET].values
            cv = cross_val_score(mdl, Xa, ya, cv=5, scoring="r2")
            res["CV_mean"] = float(cv.mean())
            res["CV_std"]  = float(cv.std())
        results[name] = res

    prog.progress(100, "Done!")
    best = max(results, key=lambda k: results[k]["R2"])
    st.session_state.update({
        "reg_results": results,
        "reg_y_test":  y_te,
        "reg_X_test":  X_te_s,
        "reg_best":    best,
        "best_model":  results[best]["model"],
        "scaler":      scaler,
        "df_model":    df_model,
    })
    st.success(f"✅ Training complete. Best model: **{best}**  (R²={results[best]['R2']:.4f})")

# ── Results ───────────────────────────────────────────────────────────────────
if "reg_results" not in st.session_state:
    st.info("👆 Configure hyperparameters above and click **Train All Models**.")
    st.stop()

results   = st.session_state["reg_results"]
y_te      = st.session_state["reg_y_test"]
X_te_s    = st.session_state["reg_X_test"]
best      = st.session_state["reg_best"]
MCOL      = {
    "Random Forest":         "#1E88E5",
    "Gradient Boosting":     "#FF7043",
    "HistGradient Boosting": "#43A047",
}

st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown("### Model Performance Summary")
cols = st.columns(3)
for col, (name, res) in zip(cols, results.items()):
    is_best = name == best
    c = MCOL[name]
    cv_html = (
        f"<div style='margin-top:8px;font-size:.8rem;color:#7A85A0;'>"
        f"5-fold CV R²: {res['CV_mean']:.3f} ± {res['CV_std']:.3f}</div>"
        if "CV_mean" in res else ""
    )
    with col:
        st.markdown(
            f"<div style='background:#13151F;"
            f"border:1px solid {'#43A047' if is_best else '#1E2235'};"
            f"border-top:4px solid {c};border-radius:10px;padding:18px;'>"
            f"<div style='font-weight:700;color:{c};font-size:1rem;'>"
            f"{name} {'🏆' if is_best else ''}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            f"gap:10px;margin-top:14px;text-align:center;'>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['R2']:.3f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>R²</div></div>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['RMSE']:.2f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>RMSE</div></div>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['MAE']:.2f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>MAE</div></div>"
            f"</div>{cv_html}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Tabs for detailed charts ───────────────────────────────────────────────────
rt1, rt2, rt3, rt4, rt5 = st.tabs([
    "📈 Actual vs Predicted",
    "📉 Residuals",
    "📊 Metrics Comparison",
    "🌐 Country-Level",
    "📋 Full Metrics Table",
])

with rt1:
    fig, axes = dark_fig(1, 3, figsize=(17, 5))
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(y_te, res["y_pred"], alpha=0.4, color=MCOL[name], s=22, zorder=3)
        mn, mx = y_te.min(), y_te.max()
        ax.plot([mn, mx], [mn, mx], "white", lw=1.3, ls="--", alpha=0.5)
        style_ax(ax, title=f"{name}\nR²={res['R2']:.3f}  RMSE={res['RMSE']:.2f}",
                 xlabel="Actual Trade % GDP", ylabel="Predicted")
    show(fig)

with rt2:
    fig, axes = dark_fig(1, 3, figsize=(17, 4))
    for ax, (name, res) in zip(axes, results.items()):
        resid = y_te - res["y_pred"]
        ax.hist(resid, bins=30, color=MCOL[name], alpha=0.82, edgecolor="#0D0F18")
        ax.axvline(0, color="white", lw=1.5, ls="--", alpha=0.6)
        ax.axvline(resid.mean(), color="#FDD835", lw=1.2, ls=":", alpha=0.8,
                   label=f"mean={resid.mean():.2f}")
        style_ax(ax, title=f"{name} Residuals\nSkew={float(pd.Series(resid).skew()):.3f}",
                 xlabel="Residual", ylabel="Count")
        ax.legend(fontsize=8, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
    show(fig)

    st.markdown("**Residuals vs Fitted Values**")
    fig, axes = dark_fig(1, 3, figsize=(17, 4))
    for ax, (name, res) in zip(axes, results.items()):
        resid = y_te - res["y_pred"]
        ax.scatter(res["y_pred"], resid, alpha=0.35, color=MCOL[name], s=18)
        ax.axhline(0, color="white", lw=1.3, ls="--", alpha=0.5)
        style_ax(ax, title=name, xlabel="Fitted Values", ylabel="Residual")
    show(fig)

with rt3:
    fig, axes = dark_fig(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
        names  = list(results.keys())
        vals   = [results[n][metric] for n in names]
        colors = [MCOL[n] for n in names]
        bars   = ax.bar(names, vals, color=colors, width=0.5, edgecolor="#0D0F18")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom",
                    color="white", fontsize=11, fontweight="bold")
        style_ax(ax, title=metric, ylabel=metric)
        ax.set_xticklabels(names, rotation=15, ha="right", color="#C5CAD6")
        if "CV_mean" in results[names[0]]:
            cv_vals = [results[n]["CV_mean"] for n in names]
            ax2 = ax.twinx()
            ax2.plot(names, cv_vals, "o--", color="#FDD835",
                     lw=1.5, markersize=8, label="CV R²")
            ax2.set_ylabel("CV R²", color="#FDD835", fontsize=8)
            ax2.tick_params(colors="#FDD835", labelsize=8)
    show(fig)

with rt4:
    st.markdown("**Per-country prediction analysis (test-set observations)**")
    df_model_local = st.session_state["df_model"].copy()
    # rebuild test indices
    from sklearn.model_selection import train_test_split as tts
    _, idx_te = tts(range(len(df_model_local)), test_size=test_size, random_state=42)
    df_te     = df_model_local.iloc[list(idx_te)].copy()
    df_te["Predicted"] = results[best]["y_pred"]
    df_te["Residual"]  = df_te[TARGET] - df_te["Predicted"]
    df_te["AbsError"]  = df_te["Residual"].abs()

    country_err = df_te.groupby("Country Name")["AbsError"].agg(
        MAE_test="mean", Std="std"
    ).round(3).sort_values("MAE_test")
    st.dataframe(country_err, use_container_width=True)

    fig, ax = dark_fig(figsize=(12, 4))
    ax.bar(country_err.index, country_err["MAE_test"],
           color=[BLOC_COLORS[COUNTRY_BLOC[c]] for c in country_err.index],
           edgecolor="#0D0F18", width=0.6)
    ax.errorbar(range(len(country_err)), country_err["MAE_test"],
                yerr=country_err["Std"], fmt="none",
                color="white", capsize=4, lw=1.2)
    style_ax(ax, title=f"Mean Absolute Error by Country — {best}",
             xlabel="Country", ylabel="MAE (pp of GDP)")
    ax.set_xticks(range(len(country_err)))
    ax.set_xticklabels(country_err.index, rotation=30, ha="right", color="#C5CAD6")
    show(fig)

with rt5:
    rows = []
    for name, res in results.items():
        row = {"Model": name, "R²": round(res["R2"],4),
               "RMSE": round(res["RMSE"],3), "MAE": round(res["MAE"],3)}
        if "CV_mean" in res:
            row["CV R² (mean)"] = round(res["CV_mean"],4)
            row["CV R² (std)"]  = round(res["CV_std"], 4)
        rows.append(row)
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download metrics", mdf.to_csv(index=False),
                       "model_metrics.csv", "text/csv")
