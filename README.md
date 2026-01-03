# California_House_Price_Prediction

## 📌 Project Overview

This project is an end-to-end **Machine Learning web application** that predicts **median house prices** based on location and demographic features. The model is trained on the **California Housing Dataset**, where each data point represents a **census block group** (not an individual house).

The project covers the complete ML lifecycle:

* Data preprocessing
* Feature engineering
* Model training
* Model persistence
* Web deployment using Flask

---

## 📊 Dataset Description

The dataset contains information collected from the **1990 California Census**. Each row represents a **census block group** with aggregated housing and population data.

### 🔹 Input Features

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| longitude          | Longitude of the block group                |
| latitude           | Latitude of the block group                 |
| housing_median_age | Median age of houses                        |
| total_rooms        | Total number of rooms in the block group    |
| total_bedrooms     | Total number of bedrooms in the block group |
| population         | Total population                            |
| households         | Total households                            |
| median_income      | Median income (scaled)                      |
| ocean_proximity    | Distance from ocean (categorical)           |

### 🎯 Target Variable

* **median_house_value**: Median house price in USD

> ⚠️ Note: Values like `total_rooms` and `total_bedrooms` are high because they are **aggregated totals**, not per-house values.

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** – Data handling
* **Scikit-learn** – ML models & pipelines
* **RandomForestRegressor** – Prediction model
* **Flask** – Web framework
* **HTML & CSS** – Frontend
* **Joblib** – Model persistence

---

## 🔄 Machine Learning Pipeline

The project uses a **Scikit-learn Pipeline** to ensure consistent preprocessing:

### Numerical Features

* Missing value handling using **Median Imputation**
* Feature scaling using **StandardScaler**

### Categorical Features

* One-Hot Encoding using **OneHotEncoder**

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

---

## 🤖 Model Used

* **Random Forest Regressor**
* Robust to outliers
* Handles non-linear relationships well

The trained model and preprocessing pipeline are saved as:

* `model.pkl`
* `pipeline.pkl`

---

## 🌐 Web Application

A Flask-based web interface allows users to input housing features and get real-time predictions.

### Features

* User-friendly UI
* Real-time price prediction
* JSON-based API communication

---

## 📁 Project Structure

```
House_Price_Prediction/
│
├── main1.py
├── model.pkl
├── pipeline.pkl
├── housing.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## 📈 Future Improvements

* Feature engineering (rooms per household, population density)
* Model comparison (XGBoost, Linear Regression)
* Deployment on cloud platforms (Render, AWS, Railway)
* Add data validation and error handling

---

## 🧠 Key Learnings

* Importance of data preprocessing
* Using pipelines for reproducibility
* Handling categorical and numerical data
* Deploying ML models as web applications


