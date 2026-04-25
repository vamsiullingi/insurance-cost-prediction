import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# --- STATISTICAL ANALYSIS & VISUALIZATION ---
print("--- Statistical Analysis ---")

# 1. Global Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Insurance Risk Factors')
plt.show()

# 2. Hypothesis Testing: Chronic Disease
chronic_yes = df[df['AnyChronicDiseases'] == 1]['PremiumPrice']
chronic_no = df[df['AnyChronicDiseases'] == 0]['PremiumPrice']
t_stat, p_val = stats.ttest_ind(chronic_yes, chronic_no)
print(f"Chronic Disease T-test p-value: {p_val:.4f}")

# 3. Hypothesis Testing: Surgeries (ANOVA)
surgery_groups = [group['PremiumPrice'].values for name, group in df.groupby('NumberOfMajorSurgeries')]
f_stat, p_anova = stats.f_oneway(*surgery_groups)
print(f"Surgeries ANOVA p-value: {p_anova:.4f}\n")

# --- DATA SPLITTING & SCALING ---
X = df.drop(['PremiumPrice'], axis=1)
y = df['PremiumPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- THE MODEL SHOWDOWN (COMPARISON) ---
print("--- Comparing Models ---")
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, md in models.items():
    md.fit(X_train_scaled, y_train)
    preds = md.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name} -> R2 Score: {r2:.4f} | MAE: ₹{mae:.2f}")

# --- FINAL MODEL SELECTION ---
# Proceeding with Random Forest as the primary engine based on comparison results
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_train_scaled, y_train)

# Final Evaluation
y_pred = final_model.predict(X_test_scaled)
print(f"\nFinal Selected Model: Random Forest")
print(f"Final Performance (R2): {r2_score(y_test, y_pred):.4f}")
print(f"Final Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):.2f}")

# --- SAVE ARTIFACTS FOR DEPLOYMENT ---
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nDeployment artifacts saved successfully.")
