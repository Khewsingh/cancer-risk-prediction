"""Streamlit demo app for the Cancer Risk Level model.

Run:  streamlit run app.py
Needs cancer_risk_final_model.pkl in the same folder (created by the notebook).
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cancer Risk Level Demo", page_icon="🩺")


@st.cache_resource
def load_bundle(path="cancer_risk_final_model.pkl"):
    return joblib.load(path)


def predict_risk(patient: dict, bundle: dict) -> dict:
    row = pd.DataFrame([patient])[bundle["feature_columns"]]
    proba = bundle["model"].predict_proba(row)
    boosted = proba.copy()
    boosted[:, bundle["classes"].index("High")] *= bundle["high_boost"]
    label = bundle["classes"][int(boosted.argmax(axis=1)[0])]
    return {"risk_level": label,
            "probabilities": {c: float(p) for c, p in zip(bundle["classes"], proba[0])}}


st.title("🩺 Cancer Risk Level Prediction")
st.warning("Educational demo built on a synthetic dataset. "
           "This is NOT medical advice and must not be used for real decisions.")

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("`cancer_risk_final_model.pkl` not found. Run the notebook first to create it.")
    st.stop()

scale_features = ["Smoking", "Alcohol_Use", "Obesity", "Diet_Red_Meat", "Diet_Salted_Processed",
                  "Fruit_Veg_Intake", "Physical_Activity", "Air_Pollution", "Occupational_Hazards",
                  "Calcium_Intake", "Physical_Activity_Level"]
binary_features = ["Family_History", "BRCA_Mutation", "H_Pylori_Infection"]

with st.form("patient"):
    c1, c2 = st.columns(2)
    patient = {}
    patient["Age"] = c1.number_input("Age", 1, 100, 50)
    patient["Gender"] = 1 if c2.selectbox("Gender", ["Female", "Male"]) == "Male" else 0
    patient["BMI"] = c1.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    st.markdown("**Lifestyle scores (0 = low, 10 = high)**")
    for i, f in enumerate(scale_features):
        patient[f] = (c1 if i % 2 == 0 else c2).slider(f.replace("_", " "), 0, 10, 5)
    st.markdown("**Medical history**")
    for f in binary_features:
        patient[f] = int(st.checkbox(f.replace("_", " ")))
    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_risk(patient, bundle)
    colour = {"Low": "green", "Medium": "orange", "High": "red"}[result["risk_level"]]
    st.markdown(f"### Predicted risk level: :{colour}[{result['risk_level']}]")
    st.bar_chart(pd.Series(result["probabilities"]))
