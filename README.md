# 🏥 Health Insurance Premium Prediction
> **A Machine Learning & Business Intelligence Project**

---

### 📌 Project Overview
This project predicts individual health insurance premiums based on physical and clinical factors. By analyzing data from 986 policyholders, I built a predictive engine that automates pricing and identifies key cost drivers for the insurance provider.

### 📊 Key Visualizations (Tableau)
* **Avg Premium:** ₹24,337
* **Primary Drivers:** Age, BMI, and Chronic Health Conditions.
* [View Interactive Dashboard](https://public.tableau.com/shared/SDKFH67R5?:display_count=n&:origin=viz_share_link)

---

### 🧪 Statistical Insights (Hypothesis Testing)
We conducted rigorous testing to validate our features before modeling:
* **Chronic Diseases:** Using a T-test, we found a $p$-value of **0.0000**, proving chronic conditions significantly impact premiums.
* **Major Surgeries:** An ANOVA test confirmed that the number of surgeries is a critical predictor ($p < 0.05$).

### 🤖 Machine Learning Performance
I compared three models to find the best "Pricing Engine":

| Model | R² Score | MAE (Error) |
| :--- | :--- | :--- |
| **Random Forest** | **0.8899** | **₹1,093** |
| Gradient Boosting | 0.8657 | ₹1,476 |
| Linear Regression | 0.7148 | ₹2,580 |

---

### 🚀 Deployment
The final model is deployed via a **Streamlit Web Application**, allowing agents to input user data and receive an instant quote.

**How to run locally:**
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

---

### 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
