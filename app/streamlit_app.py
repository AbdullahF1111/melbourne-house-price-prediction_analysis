import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Model
# -----------------------------
try:
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "house_price_model.joblib")
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"⚠️ Could not load model file. Make sure 'house_price_model.joblib' exists in /model folder.\n\nError: {e}")
    st.stop()

# -----------------------------
# Feature Columns (must match training)
# -----------------------------
training_columns = [
    'Rooms', 'Bathroom', 'Car', 'Landsize', 'Distance',
    'Type_t', 'Type_u',
    'Regionname_northern Metropolitan',
    'Regionname_South-Eastern Metropolitan',
    'Regionname_Western Metropolitan'
]

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="🏠 Melbourne Housing Price Predictor", layout="wide")

st.title("🏠 Melbourne Housing Price Analysis & Prediction App")
st.markdown("""
Welcome to the **Melbourne Housing Price Predictor**!  
Use the sidebar to input property details and get an instant estimated price.  
This model is powered by **XGBoost**, trained on real Melbourne housing data.
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔧 Input House Attributes")

rooms = st.sidebar.slider("Number of Rooms", 1, 5, 3)
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 3, 2)
car = st.sidebar.slider("Number of Car Spaces", 0, 4, 2)
landsize = st.sidebar.number_input("Landsize (m²)", 50, 2000, 500)
distance = st.sidebar.number_input("Distance to City Center (km)", 1.0, 48.0, 10.0)

property_type = st.sidebar.selectbox("Property Type", ["House", "Unit", "Townhouse"])
region = st.sidebar.selectbox("Region", ["Northern Metropolitan", "Southern Metropolitan", "Western Metropolitan"])

# -----------------------------
# Encode Input
# -----------------------------
new_data = {
    "Rooms": rooms,
    "Bathroom": bathroom,
    "Car": car,
    "Landsize": landsize,
    "Distance": distance,
    "Type_t": 1 if property_type == "Townhouse" else 0,
    "Type_u": 1 if property_type == "Unit" else 0,
    "Regionname_South-Eastern Metropolitan": 1 if region == "Regionname_South-Eastern Metropolitan" else 0,
    "Regionname_northern Metropolitan": 1 if region == "Southern Metropolitan" else 0,
    "Regionname_Western Metropolitan": 1 if region == "Western Metropolitan" else 0,
}

# -----------------------------
# Prediction
# -----------------------------
X_new = pd.DataFrame([new_data])
X_new = X_new.reindex(columns=training_columns, fill_value=0)

try:
    pred_price = model.predict(X_new)[0]
    st.subheader("💰 Predicted House Price")
    st.success(f"Estimated Price: **${pred_price:,.0f} AUD**")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")
    st.stop()

# -----------------------------
# Model Statistics
# -----------------------------
st.markdown("### 📊 Model Performance Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="R² Score", value="0.73")
with col2:
    st.metric(label="RMSE", value="≈ 283,941 AUD")

# -----------------------------
# SHAP Explainability
# -----------------------------
st.markdown("### 🧠 SHAP Model Explainability")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)

    st.write("Feature Impact on Prediction:")
    shap.summary_plot(shap_values, X_new, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
except Exception:
    st.info("SHAP visualization only works locally with proper environment support.")

# -----------------------------
# Exploratory Analysis (Images)
# -----------------------------
st.markdown("### 📉 Exploratory Data Analysis Insights")
col1, col2 = st.columns(2)
col1.image("images/ca9df0c7-b8a3-4628-a6fe-b763c34093ca.png", caption="Landsize vs Price", use_container_width=True)
col2.image("images/f9a7203b-0e6e-493b-8e9d-be60e9c32106.png", caption="Categorical Feature Impact", use_container_width=True)

st.markdown("*(This app was created as part of a Data Analysis & Machine Learning portfolio project.)*")
