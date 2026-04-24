import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# --- DATA PREPROCESSING ---
df = pd.read_csv('/content/insurance-cost-prediction/insurance.csv')

# Feature Engineering: BMI Calculation
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# Feature Engineering: Risk Score (Sum of all binary health indicators)
health_cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 
               'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily']
df['RiskScore'] = df[health_cols].sum(axis=1)

# --- HYPOTHESIS TESTING ---
print("--- Statistical Analysis ---")
# Hypothesis: Does Chronic Disease lead to significantly higher premiums?
chronic_yes = df[df['AnyChronicDiseases'] == 1]['PremiumPrice']
chronic_no = df[df['AnyChronicDiseases'] == 0]['PremiumPrice']
t_stat, p_val = stats.ttest_ind(chronic_yes, chronic_no)
print(f"Chronic Disease T-test p-value: {p_val:.4f}")

# Hypothesis: Does the number of surgeries impact cost? (ANOVA)
surgery_groups = [group['PremiumPrice'].values for name, group in df.groupby('NumberOfMajorSurgeries')]
f_stat, p_anova = stats.f_oneway(*surgery_groups)
print(f"Surgeries ANOVA p-value: {p_anova:.4f}\n")

# --- MODEL SELECTION & TRAINING ---
X = df.drop(['PremiumPrice'], axis=1)
y = df['PremiumPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Random Forest was chosen for its ability to handle non-linear relationships
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(scaler.transform(X_test))
print(f"Model Performance (R2): {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):.2f}")

# --- SAVE ARTIFACTS FOR DEPLOYMENT ---
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Artifacts saved: rf_model.pkl, scaler.pkl, features.pkl")
