import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
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
Use this app to analyze and predict house prices using an **XGBoost model** trained on Melbourne’s property dataset.
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
# Encode Inputs
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
# Image Display Function
# =========================
def safe_show_image(file_name, caption, explanation):
    """
    Safely display an image using PIL to ensure Streamlit Cloud compatibility.
    """
    image_path = os.path.join(os.path.dirname(__file__), "..", "images", file_name)
    try:
        img = Image.open(image_path)
        st.image(img, caption=caption, use_column_width=True)
        st.caption(explanation)
    except FileNotFoundError:
        st.warning(f"⚠️ Image not found: {file_name}")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Image Missing", fontsize=16, ha="center", va="center")
        ax.axis("off")
        st.pyplot(fig)
        st.caption(explanation)
    except Exception as e:
        st.error(f"⚠️ Could not display {file_name}: {e}")

# =========================
# EDA Section
# =========================
st.markdown("## 📈 Exploratory Data Analysis (EDA)")

st.markdown("Below are visual insights that explain how various features relate to house prices in Melbourne:")

safe_show_image(
    "numerical features.png",
    caption="Relationships Between Numerical Features and House Price",
    explanation="This figure shows correlations between Landsize, Rooms, Bathrooms, and Distance with price."
)

col1, col2 = st.columns(2)
with col1:
    safe_show_image(
        "shape landsize & distance.png",
        caption="Landsize & Distance SHAP Analysis",
        explanation="Larger landsize tends to increase prices, while greater distance decreases them."
    )
with col2:
    safe_show_image(
        "feature impact.png",
        caption="Categorical Features Impact",
        explanation="Shows how property type and region contribute to price variation."
    )

safe_show_image(
    "categorical features.png",
    caption="Categorical Feature Distribution",
    explanation="Shows the distribution of prices across different property types and regions."
)

safe_show_image(
    "shape houseprice.png",
    caption="Distribution of Melbourne House Prices",
    explanation="Displays the skewed distribution of house prices with few high-end outliers."
)

# =========================
# SHAP Explainability
# =========================
st.markdown("## 🧠 Model Explainability (SHAP Analysis)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)

    # Create static bar plot instead of interactive
    st.write("Feature Importance (SHAP Summary):")
    shap.summary_plot(shap_values, X_new, plot_type="bar", show=False)
    fig = plt.gcf()
    st.pyplot(fig, bbox_inches="tight")
    plt.clf()
except Exception as e:
    st.info("SHAP visualization skipped (requires local model compatibility).")

