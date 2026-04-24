# 🏥 SmartHealth: AI-Driven Insurance Premium Prediction
> **Bridging the gap between Health Data and Actuarial Precision.**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)

---

## 🔗 Live Project Ecosystem
* **🚀 Live Calculator:** [Try the Web App](https://insurance-cost-prediction-b7jjmfgncbwxvajkwxvvd9.streamlit.app/)
* **📊 Analytics Hub:** [Explore the Tableau Dashboard](https://public.tableau.com/views/InsuranceCostPrediction_17770378945780/HealthInsurancePricingAnalysisRiskFactors)
* **✍️ Technical Deep-Dive:** [Detailed Medium Blog](https://medium.com/@vamsiullingi1729/predicting-the-unpredictable-how-i-built-an-ai-driven-insurance-pricing-engine-with-89-accuracy-76bc26a05641)

---

## 📖 The Problem
Traditional insurance pricing often relies on generic averages, leading to "premium leakage" for insurers and unfair rates for healthy individuals. **SmartHealth** uses a data-driven approach to analyze 986 policyholder profiles, providing **real-time, personalized risk assessments.**

---

## 🏗️ Project Architecture
1. **Data Discovery (Tableau):** Dissecting 11 health attributes to identify hidden cost drivers.
2. **Statistical Validation (SciPy):** Proving the significance of chronic conditions and surgical history.
3. **Feature Engineering:** Crafting custom `BMI` and `RiskScore` metrics to enhance model "vision."
4. **Predictive Modeling:** Training a high-precision Random Forest Regressor.
5. **Deployment:** Launching an interactive Streamlit interface for non-technical agents.

---

## 📊 Phase 1: Strategic Insights (Tableau)
Instead of static charts, I built a 4-layered interactive experience. 
* **The "Age-Chronic" Intersection:** Visualized how premiums spike exponentially after age 45 when combined with chronic conditions.
* **Risk Score Impact:** Found that each additional health condition increases premium costs by an average of **₹3,500**.

> **Pro-Tip:** Check out the "Health Condition Prevalence" sheet in my Tableau link to see how Diabetes and BP overlap.

---

## 🧪 Phase 2: The Science (Hypothesis Testing)
We validated our features to ensure the model focuses on what matters:
* **Chronic Disease:** A T-Test yielded a $p$-value of **0.0000**, confirming it as a mandatory predictor.
* **Surgical History:** ANOVA testing proved that individuals with 2+ surgeries form a distinct, high-cost bracket.

---

## 🤖 Phase 3: Machine Learning Engine
I achieved a high-accuracy model by moving beyond basic linear relationships.

### Performance Leaderboard:
| Model | R² Score | MAE (Error) |
| :--- | :--- | :--- |
| **🏆 Random Forest** | **0.8899** | **₹1,093** |
| Gradient Boosting | 0.8657 | ₹1,476 |
| Linear Regression | 0.7148 | ₹2,580 |

### 🛠️ Feature Importance
My Random Forest model identified **Age**, **History of Transplants**, and **BMI** as the top 3 influencers of premium costs.

---

## 🚀 Phase 4: Production Deployment
The final product is a **Live Web Calculator** designed for Insurance Agents.
* **Instant Quotes:** Input customer data and get a prediction in <1 second.
* **Dynamic Feedback:** The app calculates BMI and Risk Scores on the fly, providing data-backed health tips alongside the quote.

---

## 📂 Repository Contents
* `app.py`: The Streamlit frontend script.
* `model_training.py`: The full Python pipeline (Cleaning, Stats, Training).
* `insurance.csv`: The core dataset.
* `rf_model.pkl`: The serialized "brain" of our app.
* `requirements.txt`: List of dependencies for cloud hosting.

---

## 👨‍💻 Installation & Local Usage
1. Clone the repo: `git clone https://github.com/vamsiullingi/insurance-cost-prediction.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
