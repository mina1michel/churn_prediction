# churn_prediction
📉 Telecom Customer Churn Prediction
Predicting which customers are likely to leave a telecom company — enabling proactive retention strategies before churn happens.
---
  Problem Statement
Customer churn is one of the most costly problems in the telecom industry. Acquiring a new customer costs significantly more than retaining an existing one. This project builds a machine learning pipeline to identify high-risk customers before they leave, giving business teams the opportunity to intervene with targeted offers or support.
---
  Dataset
Source: IBM Telco Customer Churn Dataset
Size: 7,043 customers, 21 features
Target: `Churn` — whether a customer left within the last month (Yes/No)
Class distribution: ~73% Stay / ~27% Churn (imbalanced)
---
  Project Pipeline
```
Raw CSV → Data Cleaning → Feature Engineering → Train/Test Split
→ Model Training (3 models) → Evaluation → SHAP Analysis
```
Preprocessing steps:
Converted `TotalCharges` from object to numeric (hidden empty strings)
Dropped `CustomerID` (non-informative)
Encoded target as binary (Yes=1, No=0)
Applied `pd.get_dummies` for categorical encoding
Used `StandardScaler` for Logistic Regression
Applied `stratify=y` in train/test split to preserve class ratio
---
  Models Compared
Model	ROC-AUC	Churn Recall	Churn F1
Logistic Regression	0.835	0.80	0.61
XGBoost	0.820	0.69	0.59
Random Forest	0.818	0.49	0.55
> **Winner: Logistic Regression** — highest ROC-AUC and best churn recall after class balancing.
---
  Handling Class Imbalance
The dataset is imbalanced (~73% stay, ~27% churn). Without correction, models bias toward predicting "stay" and miss real churners.
Solution: Applied `class_weight='balanced'` to Logistic Regression and Random Forest, and `scale_pos_weight=3` to XGBoost.
Impact on Logistic Regression churn recall: 0.57 → 0.80
> In churn prediction, **recall matters more than precision**. Missing a churner costs a full customer. Falsely flagging a loyal customer costs only a small retention offer.
---
📊 Visualizations
Model Comparison
![Model Comparison](model_comparison.png)
Churn EDA
![Churn by Gender and Contract](churn_gender_contract.png)
![Churn by Senior Citizen and Monthly Charges](churn_senior_charges.png)
![Churn by Tenure and Internet Service](churn_tenure_internet.png)
SHAP Feature Importance
![SHAP Summary](shap_summary.png)
---
  Key Insights (SHAP Analysis)
SHAP (SHapley Additive exPlanations) reveals why the model makes each prediction — not just what it predicts.
Feature	Impact
Two-year contract	Strongest predictor of staying
Short tenure	New customers churn significantly more
High monthly charges	Pushes toward churn
Fiber optic internet	Associated with higher churn risk
No internet service	Strongly predicts staying
> These insights are directly actionable: focus retention efforts on month-to-month, fiber optic customers with high monthly charges in their first year.
---
  Tech Stack
Python — Pandas, NumPy
Scikit-learn — Logistic Regression, Random Forest, preprocessing, metrics
XGBoost — Gradient boosting
SHAP — Model explainability
Matplotlib / Seaborn — Visualization
---
📁 Repository Structure
```
├── churn_prediction.ipynb     # Full notebook
├── model_comparison.png       # ROC curves, F1 scores, confusion matrix
├── churn_gender_contract.png  # EDA plots
├── churn_senior_charges.png   # EDA plots
├── churn_tenure_internet.png  # EDA plots
├── shap_summary.png           # SHAP feature importance
└── README.md
```
---
🚀 How to Run
```bash
git clone https://github.com/mina1michel/telecom-churn-prediction
cd telecom-churn-prediction
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```
---
👤 Author
Mina Michel — AI/ML Engineer in training  
LinkedIn • GitHub
