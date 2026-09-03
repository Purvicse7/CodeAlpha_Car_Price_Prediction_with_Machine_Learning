"""
CodeAlpha Data Science Internship
Task 3: Car Price Prediction with Machine Learning
------------------------------------------------
Goal: Predict the resale ("selling") price of used cars from features like
current showroom price, age, mileage driven, fuel type, seller type,
transmission, and ownership history.
Dataset: car_data.csv (Kaggle — vijayaadithyanvg/car-price-prediction-used-cars),
301 used-car listings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# --- Robust path handling: works no matter where this script sits ---
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filenames, search_root):
    """Look for any of `filenames` (list of possible names, e.g. with/without
    a space or underscore) in common spots relative to the script, then
    fall back to searching the whole project folder."""
    if isinstance(filenames, str):
        filenames = [filenames]
    candidates = []
    for filename in filenames:
        candidates += [
            os.path.join(search_root, "data", filename),
            os.path.join(search_root, "..", "data", filename),
            os.path.join(search_root, filename),
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # last resort: walk the project tree
    project_root = os.path.dirname(search_root)  # one level up from scripts/
    for root, _dirs, files in os.walk(project_root):
        for filename in filenames:
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find any of {filenames}. Make sure the file is inside a "
        f"'data' folder next to this script (or next to its parent folder)."
    )

OUT = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT, exist_ok=True)
CURRENT_YEAR = 2020  # dataset was compiled around 2020; used to derive car age

# ---------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------
df = pd.read_csv(find_file(["car_data.csv", "car data.csv", "car-data.csv"], SCRIPT_DIR))

# Feature engineering: age is far more predictive/interpretable than raw Year
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df = df.drop(columns=["Year"])

# brand = first word of Car_Name (goodwill proxy)
df["brand"] = df["Car_Name"].str.split().str[0].str.lower()
df = df.drop(columns=["Car_Name"])

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sum(), "total missing cells")
print("\nUnique brands:", df["brand"].nunique())
print(df.describe().T[["mean", "std", "min", "max"]])

# ---------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(df["Selling_Price"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Used-Car Selling Price (lakhs)")
plt.xlabel("Selling Price (lakh INR)")
plt.tight_layout()
plt.savefig(f"{OUT}/task3_price_distribution.png", dpi=150)
plt.close()

top_brands = df.groupby("brand")["Selling_Price"].mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(9, 5))
top_brands.plot(kind="bar", color="teal")
plt.title("Average Selling Price by Brand (Top 15 — Brand Goodwill Proxy)")
plt.ylabel("Average Selling Price (lakh INR)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(f"{OUT}/task3_avg_price_by_brand.png", dpi=150)
plt.close()

numeric_cols = ["Present_Price", "Driven_kms", "Owner", "Car_Age", "Selling_Price"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap — Numeric Features vs Selling Price")
plt.tight_layout()
plt.savefig(f"{OUT}/task3_correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Present_Price", "Car_Age", "Driven_kms"]):
    sns.scatterplot(data=df, x=col, y="Selling_Price", ax=ax, alpha=0.6, color="darkorange")
    ax.set_title(f"Selling Price vs {col}")
plt.tight_layout()
plt.savefig(f"{OUT}/task3_price_vs_key_features.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. Preprocessing: encode categoricals
# ---------------------------------------------------------
categorical_cols = ["Fuel_Type", "Selling_type", "Transmission", "brand"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=["Selling_Price"])
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
num_cols_to_scale = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols_to_scale] = scaler.fit_transform(X_train[num_cols_to_scale])
X_test_scaled[num_cols_to_scale] = scaler.transform(X_test[num_cols_to_scale])

# ---------------------------------------------------------
# 4. Train & compare regression models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=5000),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42, max_depth=3),
}

results = []
preds_store = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    preds_store[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE (lakh)": round(mae, 3), "RMSE (lakh)": round(rmse, 3), "R2 Score": round(r2, 4)})

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print("\nModel comparison:\n", results_df.to_string(index=False))
results_df.to_csv(f"{OUT}/task3_model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_preds = preds_store[best_name]
best_model = models[best_name]

# ---------------------------------------------------------
# 5. Plots for the best model
# ---------------------------------------------------------
plt.figure(figsize=(6, 5.5))
plt.scatter(y_test, best_preds, alpha=0.7, color="mediumseagreen")
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Selling Price (lakh INR)")
plt.ylabel("Predicted Selling Price (lakh INR)")
plt.title(f"Actual vs Predicted Selling Price — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/task3_actual_vs_predicted.png", dpi=150)
plt.close()

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 6))
    imp.sort_values().plot(kind="barh", color="steelblue")
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/task3_feature_importance.png", dpi=150)
    plt.close()
    imp.to_csv(f"{OUT}/task3_top_feature_importance.csv")

plt.figure(figsize=(7, 4))
sns.barplot(data=results_df, x="Model", y="R2 Score", hue="Model", palette="viridis", legend=False)
plt.title("Regression Model Comparison — Used-Car Price Prediction (R2 Score)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT}/task3_model_r2_comparison.png", dpi=150)
plt.close()

print(f"\nBest model: {best_name}")
print("Task 3 complete. Outputs saved to:", OUT)