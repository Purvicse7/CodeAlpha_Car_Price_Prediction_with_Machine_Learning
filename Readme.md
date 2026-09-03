# 🚗 Car Price Prediction with Machine Learning

> Predicting used car resale prices using exploratory data analysis, feature engineering, and regression machine learning algorithms.

---

## 📌 Project Overview

The objective of this project is to develop machine learning regression models to accurately predict the **selling price** (resale value in Lakh INR) of used vehicles based on various vehicle attributes such as current showroom price, age, mileage, fuel type, transmission type, seller type, and brand.

---

## 📊 Dataset Description

- **Source:** [Kaggle — Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-prediction-used-cars)
- **Size:** 301 records × 9 features
- **Target Variable:** `Selling_Price` (in Lakh INR)

### Features:
| Feature | Type | Description |
| :--- | :--- | :--- |
| `Car_Name` | Categorical | Brand and model name of the car |
| `Year` | Numeric | Year of manufacture / purchase |
| `Present_Price` | Numeric | Current showroom price of the vehicle (Lakh INR) |
| `Driven_kms` | Numeric | Total distance driven in kilometers |
| `Fuel_Type` | Categorical | Fuel type (`Petrol`, `Diesel`, `CNG`) |
| `Selling_type` | Categorical | Seller type (`Dealer`, `Individual`) |
| `Transmission` | Categorical | Transmission type (`Manual`, `Automatic`) |
| `Owner` | Numeric | Number of previous owners (0, 1, 3) |

---

## ⚙️ Methodology & Pipeline

1. **Data Preprocessing & Cleaning:**
   - Checked for null/missing values.
   - Handled relative file paths robustly to ensure the script runs across different folder structures.

2. **Feature Engineering:**
   - **`Car_Age`:** Computed car age using `2020 - Year` to make depreciation linear and interpretable.
   - **`brand`:** Extracted the brand name from `Car_Name` as a proxy for brand goodwill/prestige.

3. **Exploratory Data Analysis (EDA):**
   - Target variable distribution analysis.
   - Average resale price comparison across top brands.
   - Correlation heatmap to assess multi-collinearity and feature-target relationships.
   - Scatter plots of `Selling_Price` against key numeric indicators.

4. **Encoding & Feature Scaling:**
   - One-Hot Encoded categorical variables (`Fuel_Type`, `Selling_type`, `Transmission`, `brand`) with `drop_first=True`.
   - 80-20 Train-Test split (`random_state=42`).
   - Scaled numerical features using `StandardScaler` fitted strictly on training data.

5. **Model Training & Comparison:**
   Evaluated multiple regression models:
   - **Linear Regression**
   - **Ridge Regression** ($\alpha = 10$)
   - **Lasso Regression** ($\alpha = 0.1$)
   - **Random Forest Regressor** ($n = 300$)
   - **Gradient Boosting Regressor** ($n = 300, \text{max\_depth} = 3$)

---

## 📈 Evaluation Metrics

The models were evaluated and compared using the following metrics on the test set:
- **Mean Absolute Error (MAE):** Average magnitude of errors in Lakhs.
- **Root Mean Squared Error (RMSE):** Standard deviation of the residuals.
- **$R^2$ Score (Coefficient of Determination):** Proportion of variance explained by the model.

---

## 📁 Repository Structure

```text
├── data/
│   └── car_data.csv                   # Dataset
├── results/                           # Generated charts & metrics
│   ├── task3_price_distribution.png
│   ├── task3_avg_price_by_brand.png
│   ├── task3_correlation_heatmap.png
│   ├── task3_price_vs_key_features.png
│   ├── task3_model_comparison.csv
│   ├── task3_actual_vs_predicted.png
│   ├── task3_feature_importance.png
│   ├── task3_top_feature_importance.csv
│   └── task3_model_r2_comparison.png
├── car_price_prediction.py      # Main Python script
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```
# Execution
```
pip install -r requirements.txt
```

```
python car_price_prediction.py
```