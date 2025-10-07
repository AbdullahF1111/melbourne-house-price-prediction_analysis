import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ================================
# 🔹 Load Model
# ================================
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "house_price_model.joblib")
model = joblib.load(model_path)

# ================================
# 🔹 Define training columns (must match model training)
# ================================
training_columns = [
    'Rooms', 'Bathroom', 'Car', 'Landsize', 'Distance',
    'Type_t', 'Type_u',
    'Regionname_Northern Metropolitan',
    'Regionname_South-Eastern Metropolitan',  # ✅ Added missing region
    'Regionname_Southern Metropolitan',
    'Regionname_Western Metropolitan'
]

# ================================
# 🔹 Streamlit Configuration
# ================================
st.set_page_config(page_title="🏠 Melbourne Housing Price Predictor", layout="wide")

st.title("🏠 Melbourne Housing Price Analysis & Prediction App")
st.markdown("""
This interactive app allows you to **analyze and predict house prices** in Melbourne using a trained **XGBoost model**.
Simply adjust the sliders and dropdowns in the sidebar to estimate a property’s market value.
""")

# ================================
# 🔹 Sidebar Inputs
# ================================
st.sidebar.header("🔧 Input House Attributes")

rooms = st.sidebar.slider("Number of Rooms", 1, 5, 3)
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 3, 2)
car = st.sidebar.slider("Number of Car Spaces", 0, 4, 2)
landsize = st.sidebar.number_input("Landsize (m²)", 50, 2000, 500)
distance = st.sidebar.number_input("Distance to City Center (km)", 1.0, 48.0, 10.0)

property_type = st.sidebar.selectbox("Property Type", ["House", "Unit", "Townhouse"])
region = st.sidebar.selectbox(
    "Region",
    [
        "Northern Metropolitan",
        "South-Eastern Metropolitan",  # ✅ Added missing region
        "Southern Metropolitan",
        "Western Metropolitan"
    ]
)

# ================================
# 🔹 Encode categorical values
# ================================
input_data = {
    "Rooms": rooms,
    "Bathroom": bathroom,
    "Car": car,
    "Landsize": landsize,
    "Distance": distance,
    "Type_t": 1 if property_type == "Townhouse" else 0,
    "Type_u": 1 if property_type == "Unit" else 0,
    "Regionname_Northern Metropolitan": 1 if region == "Northern Metropolitan" else 0,
    "Regionname_South-Eastern Metropolitan": 1 if region == "South-Eastern Metropolitan" else 0,
    "Regionname_Southern Metropolitan": 1 if region == "Southern Metropolitan" else 0,
    "Regionname_Western Metropolitan": 1 if region == "Western Metropolitan" else 0,
}

# ================================
# 🔹 Prepare input for prediction
# ================================
X_new = pd.DataFrame([input_data])
X_new = X_new.reindex(columns=training_columns, fill_value=0)  # Ensure alignment

# ================================
# 🔹 Make Prediction (Safe Execution)
# ================================
try:
    pred_price = model.predict(X_new)[0]
    st.subheader("💰 Predicted House Price")
    st.success(f"Estimated Price: **${pred_price:,.0f} AUD**")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")

# ================================
# 🔹 Model Statistics
# ================================
st.markdown("### 📊 Model Insights")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="R² Score", value="0.73")
    st.metric(label="RMSE", value="≈ 283,941 AUD")

with col2:
    st.image("images/cat_features.png", caption="Categorical Feature Analysis", use_container_width=True)

# ================================
# 🔹 SHAP Visualization
# ================================
st.markdown("### 🧠 Model Explainability (SHAP Analysis)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)
    st.write("Feature Impact Visualization:")
    shap.summary_plot(shap_values, X_new, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
except Exception:
    st.info("SHAP visualization is available only when model and data are locally compatible.")

# ================================
# 🔹 Static Visualizations
# ================================
st.markdown("### 📉 Exploratory Analysis")
col1, col2 = st.columns(2)
col1.image("images/ca9df0c7-b8a3-4628-a6fe-b763c34093ca.png", caption="Landsize vs SHAP Values", use_container_width=True)
col2.image("images/f9a7203b-0e6e-493b-8e9d-be60e9c32106.png", caption="Categorical Features Impact", use_container_width=True)
