# 🩺 Cancer Risk Level Prediction

A machine learning project that predicts a person's **cancer risk level (Low / Medium / High)** from lifestyle and health factors such as smoking, BMI, diet, family history and more.

> ⚠️ **Disclaimer:** The dataset looks synthetic (artificially generated). This is an **educational project**, not a medical tool. Do not use it for real health decisions.

---

## ✨ Highlights

- **Data leakage found and removed:** `Overall_Risk_Score` gives away the answer (the risk level is created from it), so it is dropped. Keeping it gives a fake ~100% accuracy.
- **Fair model comparison:** repeated stratified 5-fold cross-validation on training data only.
- **Imbalance handled correctly:** class weights, and SMOTE applied *inside* each fold (no leakage).
- **Hyperparameter tuning:** Optuna (40 trials) optimizing macro-F1.
- **Health-focused thresholding:** a boost factor on the *High* class, chosen on training data, so High-risk patients are missed less often.
- **Explainability:** permutation feature importance.
- **Deployment-ready:** one `.pkl` bundle (model + encoder + feature order + threshold) and a Streamlit demo app.

---

## 📊 Dataset

- **File:** `cancer-risk-factors.csv` (2,000 rows, 21 columns)
- **Target:** `Risk_Level` → Low (~16%), Medium (~79%), High (~5%) (**imbalanced**)
- **Features used (17):** Age, Gender, Smoking, Alcohol_Use, Obesity, Family_History, Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity, Air_Pollution, Occupational_Hazards, BRCA_Mutation, H_Pylori_Infection, Calcium_Intake, BMI, Physical_Activity_Level
- **Dropped:** `Patient_ID` (identifier), `Cancer_Type` (diagnosis label, not a risk factor), `Overall_Risk_Score` (**leakage**)

The CSV is not included in this repo. Download it from where you originally got it (for example, Kaggle) and place it in the project folder.

---

## 🧠 Models Compared

| Model | How imbalance is handled |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| Random Forest + SMOTE | synthetic samples inside each CV fold |
| XGBoost (plain) | none (baseline) |
| **XGBoost + class weights (final)** | balanced sample weights, tuned with Optuna |

Accuracy alone is misleading here (always predicting "Medium" gives ~79%), so the main metrics are **macro-F1** and **High-class recall**.

---

## 📈 Results

> Fill this table with the numbers from your own notebook run.

| Metric (test set) | Final model | With High-risk threshold |
|---|---|---|
| Accuracy | _…_ | _…_ |
| Macro-F1 | _…_ | _…_ |
| High recall | _…_ | _…_ |
| High precision | _…_ | _…_ |

The test set has only ~20 *High* patients, so one patient changes High recall by 5%. Treat the numbers as estimates.

---

## 🗂 Project Structure

```
.
├── Cancer_Risk_Prediction_Improved.ipynb   # full analysis, training and evaluation
├── app.py                                  # Streamlit demo app
├── requirements.txt
├── cancer-risk-factors.csv                 # (you add this, ignored by git)
└── cancer_risk_final_model.pkl             # (created by the notebook, ignored by git)
```

---

## 🚀 How to Run

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Put cancer-risk-factors.csv in this folder, then run the notebook
jupyter notebook Cancer_Risk_Prediction_Improved.ipynb
#    (Run all cells. This creates cancer_risk_final_model.pkl)

# 5. Start the demo app
streamlit run app.py
```

### Use the saved model in your own code

```python
import joblib, pandas as pd

bundle = joblib.load("cancer_risk_final_model.pkl")

patient = {  # one value for each of the 17 features
    "Age": 55, "Gender": 1, "Smoking": 7, "Alcohol_Use": 4, "Obesity": 6,
    "Diet_Red_Meat": 5, "Diet_Salted_Processed": 4, "Fruit_Veg_Intake": 3,
    "Physical_Activity": 2, "Air_Pollution": 7, "Occupational_Hazards": 5,
    "Calcium_Intake": 4, "Physical_Activity_Level": 3,
    "Family_History": 1, "BRCA_Mutation": 0, "H_Pylori_Infection": 0, "BMI": 29.5,
}

row = pd.DataFrame([patient])[bundle["feature_columns"]]
proba = bundle["model"].predict_proba(row)
proba[:, bundle["classes"].index("High")] *= bundle["high_boost"]
print(bundle["classes"][proba.argmax(axis=1)[0]])
```

---

## ⚠️ Limitations

- The dataset appears synthetic, so real-world performance is unknown.
- Very few *High* cases in the test set, so scores are noisy.
- `Risk_Level` seems to be derived from a score, which may put a natural ceiling on accuracy.
- Not validated for clinical use.

## 🔭 Future Work

- Try one-hot encoded `Cancer_Type` as an extra feature.
- Add SHAP explanations for individual patients.
- Deploy the app online (Streamlit Community Cloud or Hugging Face Spaces).
- Add a FastAPI endpoint and tests.

---

## 🛠 Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · Optuna · Matplotlib · Seaborn · Streamlit

## 📄 License

MIT. See [LICENSE](LICENSE).

## 👤 Author

**Surya**, B.E. Artificial Intelligence and Data Science student, Siddaganga Institute of Technology, Tumkur.
