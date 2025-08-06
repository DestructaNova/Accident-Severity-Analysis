# 🚗 Road Traffic Accident Severity Prediction

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

> A machine learning project focused on analyzing and predicting the severity of road traffic accidents to uncover key contributing factors and inform road safety initiatives.

---

## 🎯 Project Objective

The primary goal of this project is to develop a robust classification model that can accurately predict the severity of a road traffic accident based on various features. The model's performance is evaluated using the **weighted f1-score**, focusing on its ability to distinguish between different levels of accident severity.

---

## 📖 Dataset Overview

The data for this analysis was sourced from the **Addis Ababa Sub-city police departments** and collected as part of a master's research thesis.

-   **Timeframe:** 2017 - 2020
-   **Instances:** 12,316 accidents
-   **Features:** 32 distinct attributes

All sensitive information was excluded during the data encoding process to ensure privacy.

---

## 🛠️ Project Workflow

This project follows a structured end-to-end machine learning pipeline, from data exploration to model deployment.

| Stage                          | Key Techniques & Libraries Used                                                                                             |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| **1. Exploratory Data Analysis** | `dabl` for high-level insights, `matplotlib` & `seaborn` for detailed custom plots.                                           |
| **2. Data Preprocessing** | `fillna` for handling missing data, One-Hot Encoding (`get_dummies`), and `SMOTENC` to resolve significant class imbalance. |
| **3. Feature Engineering** | `chi2` & `SelectKBest` for statistical feature selection, `PCA` for dimensionality reduction.                               |
| **4. Modeling** | `RandomForestClassifier` as a baseline, followed by `GridSearchCV` for hyperparameter tuning (`n_estimators`, `max_depth`). |

---

## 💻 Technology Stack

-   **Python**
-   **Pandas** for data manipulation
-   **Scikit-learn** for modeling and preprocessing
-   **Matplotlib & Seaborn** for data visualization
-   **imbalanced-learn** for handling imbalanced data
-   **dabl** for initial EDA
