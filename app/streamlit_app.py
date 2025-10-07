import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# =========================
# Load Model Safely
# =========================
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "house_price_model.joblib")

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'house_price_model.joblib' exists in the /model folder.")
    st.stop()

model = joblib.load(model_path)

# =========================
# Define training feature list
# =========================
training_columns = [
    'Rooms', 'Bathroom', 'Car', 'Landsize', 'Distance',
    'Type_t', 'Type_u',
    'Regionname_Northern Metropolitan',
    'Regionname_South-Eastern Metropolitan',
    'Regionname_Southern Metropolitan',
    'Regionname_Western Metropolitan'
]

# =========================
# Streamlit UI Setup
# =========================
st.set_page_config(page_title="🏠 Melbourne Housing Price Predictor", layout="wide")

st.title("🏠 Melbourne Housing Price Analysis & Prediction App")
st.markdown("""
Welcome to the **Melbourne Housing Price Prediction Dashboard**!  
This interactive tool allows you to:
- Input property details to **predict house prices**
- Explore **feature impacts and correlations**
- Understand model explainability using **SHAP analysis**
""")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Input House Attributes")

rooms = st.sidebar.slider("Number of Rooms", 1, 5, 3)
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 3, 2)
car = st.sidebar.slider("Number of Car Spaces", 0, 4, 2)
landsize = st.sidebar.number_input("Landsize (m²)", 50, 2000, 500)
distance = st.sidebar.number_input("Distance to City Center (km)", 1.0, 48.0, 10.0)

property_type = st.sidebar.selectbox("Property Type", ["House", "Unit", "Townhouse"])
region = st.sidebar.selectbox("Region", [
    "Northern Metropolitan",
    "South-Eastern Metropolitan",
    "Southern Metropolitan",
    "Western Metropolitan"
])

# =========================
# Encode Categorical Inputs
# =========================
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

# =========================
# Prediction
# =========================
X_new = pd.DataFrame([input_data])
X_new = X_new.reindex(columns=training_columns, fill_value=0)

try:
    pred_price = model.predict(X_new)[0]
    st.subheader("💰 Predicted House Price")
    st.success(f"Estimated Price: **${pred_price:,.0f} AUD**")
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")
    st.stop()

# =========================
# Model Statistics
# =========================
st.markdown("### 📊 Model Performance Summary")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="R² Score", value="0.73")
    st.metric(label="RMSE", value="≈ 283,941 AUD")

# =========================
# Load and Show Images Safely
# =========================
image_dir = os.path.join(os.path.dirname(__file__), "..", "images")

image_files = {
    "categorical": "categorical features.png",
    "numerical": "numerical features.png",
    "feature": "feature impact.png",
    "shapehouseprice": "shape houseprice.png",
    "shapelandsizedistance": "shape landsize & distance.png"
}

def safe_show_image(path, caption, explanation):
    """Safely display image or fallback plot with text."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
        st.caption(explanation)
    else:
        st.warning(f"⚠️ Image not found: {os.path.basename(path)} — showing placeholder.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Image Not Found", fontsize=16, ha='center', va='center')
        ax.axis("off")
        st.pyplot(fig)
        st.caption(explanation)

# =========================
# Analytical Visualizations Section
# =========================
st.markdown("## 📈 Exploratory Data Analysis (EDA)")

st.markdown("Below are visual insights that explain how various features relate to house prices in Melbourne:")

# --- Numerical relationships
st.markdown("### 🔹 Numerical Features and Price Relationships")
safe_show_image(
    os.path.join(image_dir, image_files["numerical"]),
    caption="Relationships Between Numerical Features and House Price",
    explanation="This figure shows how price correlates with Landsize, Distance, Rooms, and Bathrooms. "
                "Generally, larger landsize and more rooms increase the price, while distance from city lowers it."
)

# --- SHAP & Feature impact
st.markdown("### 🔹 SHAP Value & Feature Importance Analysis")
col1, col2 = st.columns(2)
with col1:
    safe_show_image(
        os.path.join(image_dir, image_files["shapelandsizedistance"]),
        caption="Landsize and Distance SHAP Analysis",
        explanation="This visualization shows the SHAP values of Landsize and Distance. "
                    "Larger Landsize contributes positively, while longer distances reduce predicted prices."
    )
with col2:
    safe_show_image(
        os.path.join(image_dir, image_files["feature"]),
        caption="Categorical Features Impact",
        explanation="SHAP summary showing how region and property type influence the prediction."
    )

# --- Categorical distributions
st.markdown("### 🔹 Categorical Features Analysis")
safe_show_image(
    os.path.join(image_dir, image_files["categorical"]),
    caption="Categorical Feature Distribution and Their Effect on Price",
    explanation="This plot illustrates how different property types and regions contribute to the price distribution."
)

# --- House price shape distribution
st.markdown("### 🔹 Price Distribution Overview")
safe_show_image(
    os.path.join(image_dir, image_files["shapehouseprice"]),
    caption="Distribution of Melbourne House Prices",
    explanation="This figure represents the distribution of house prices — showing that most properties "
                "fall within the median range, while a few luxury properties skew the upper tail."
)

# =========================
# SHAP Explainability
# =========================
st.markdown("## 🧠 Model Explainability (SHAP Analysis)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)
    st.write("Feature Importance (SHAP Summary):")
    shap.summary_plot(shap_values, X_new, plot_type="bar", show=False)
    st.pyplot(bbox_inches="tight")
except Exception:
    st.info("SHAP visualization skipped (requires local model compatibility).")
