# 🏠 Melbourne Housing Price Analysis & Prediction

“Melbourne House Price Prediction — an end-to-end data analysis and machine learning project built in Python (Pandas, Scikit-learn, XGBoost). It covers data cleaning, EDA, feature engineering, model training, and SHAP-based explainability, with deployment via Streamlit and Colab.”


## 📌 Objectives

- 📊 Analyze real estate market trends in Melbourne  
- 🧠 Identify key features influencing property prices  
- 🤖 Build accurate ML models for price prediction  
- 🌐 Deploy a user-friendly prediction & analysis interface using Streamlit

---

## 🧰 Tools & Technologies

| Category            | Tools Used                                               |
|---------------------|----------------------------------------------------------|
| Language        | Python                                                   |
| Libraries       | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |
| ML Models       | Linear Regression, Random Forest, XGBoost                |
| Visualization   | seaborn, matplotlib, SHAP                                |
| Deployment      | Streamlit                                      |
| Version Control | Git & GitHub                                             |

---

## 📊 Dataset

- [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) from Kaggle  
> A clean, public dataset suitable for regression tasks and real estate analysis.

---

## 🔍 Workflow

### **1️⃣ Data Cleaning & Preprocessing**

* **Handling Missing Values:**
  * We started by removing rows with an excessive number of missing values and dropping any remaining rows with missing data in key features.
  * This ensures our analysis is based on a complete and reliable dataset.

* **Outlier & Sanity Check:**
  * We performed a logical check to remove outliers and unrealistic data points that could negatively impact the model's performance.
  * **Distance:** Removed rows where `Distance` was 0, as this is a data entry error.
  * **Rooms:** Filtered out properties with fewer than 2 rooms or more than 6 rooms, representing rare and non-standard housing types.
  * **Bathroom:** Excluded properties with 0 bathrooms or more than 5, as these are considered outliers for residential properties.
  * **Car:** Removed rows with more than 5 car spots, which are statistically rare and not representative of the majority of the data.
  * **Landsize:** Filtered `Landsize` to a realistic range (10m² to 2000m²) to exclude errors and extreme values.

* **Categorical Data Filtering:**
  * To improve model robustness, we removed regions that had a very low number of occurrences in the dataset, such as 'Eastern Victoria', as they do not provide a strong signal for the model to learn from.

### 2️⃣ Exploratory Data Analysis (EDA)

Key Features Investigated:
Rooms, Bathroom, Car, Landsize, Distance, Regionname, Type, Price

Visualizations Used:
- 📈 Histograms (Price, Rooms, Car, Landsize)
- 📊 Boxplots & Scatterplots vs Price
- 🧪 Correlation heatmap
- 🔍 Missing value matrix (missingno)

Notable Findings:
- 🏠 More rooms & bathrooms typically → higher price  
- 🚗 1–2 car spots increase value, >2 → diminishing returns  
- 🏙 Closer to city → higher price (negative correlation with Distance)  
- 📐 Larger land boosts price up to ~1000 m², then flattens  
- 🌍 Region impacts pricing: Inner > Middle > Outer  
- 🏘 Type matters: House > Unit > Townhouse

### 3️⃣ Feature Engineering
-  Categorical encoding (e.g., Type, Regionname)
-  Selected top 7 impactful features based on EDA & model importance
-  Delete features like(YearBuilt,BuildingArea..) because have alot of null values

### 4️⃣ Modeling
- 📦 Trained and evaluated models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- 📏 Evaluated with MAE, RMSE, R²

### 5️⃣ SHAP Analysis (Model Explainability)
- 📊 SHAP summary & dependence plots revealed:
  - 🔗 Landsize × Distance: large land near city = high value
  - 🔗 Rooms × Type: room effect depends on property type

### 6️⃣ 🚀 Deployment & Usage
🌐 Live Streamlit App

- You can explore the interactive version of this project on Streamlit Cloud:
- 👉 [Melbourne House Price Prediction App](https://melbourne-house-price-predictionanalysis-qn43zobz8e3wvtlmlxqnw.streamlit.app/#6178c42d)

- What it does:

- Enter property details (rooms, bathrooms, land size, etc.)

- Instantly get a predicted price 💸

View exploratory visualizations and model insights (SHAP analysis)
---

## 💡 Key Insights

| Feature       | Impact on Price                                      |
|---------------|------------------------------------------------------|
| Rooms       | More rooms = higher price                            |
| Bathroom    | Same as above                                        |
| Car         | 1–2 spots valuable, more = less impact               |
| Distance    | Farther from CBD → lower prices                      |
| Landsize    | Adds value up to ~1000m², then plateaus              |
| Regionname  | Strong regional differences (Inner > Outer)          |
| Type        | Houses dominate in price (House > Unit > Townhouse) |

---

## 🚀 Future Enhancements

- 🌐 Add Google Maps API for geolocation visualization  
- 🔍 Integrate multilingual support (e.g., 🇩🇪 German, 🇸🇾 Arabic)  
- 📈 Update with more recent real estate data  

---

## 🛠 Run Locally

```bash
git clone https://github.com/yourusername/real-estate-price-prediction.git
cd real-estate-price-prediction
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🖼️ About The Images Folder

- The project includes an images/ folder that contains static visualizations used in the Streamlit app’s Exploratory Data Analysis (EDA) and Model Explainability sections.
These images were generated during the data analysis phase and are automatically loaded by the app.

| File                            | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| `numerical features.png`        | Correlations between numerical variables and house prices     |
| `categorical features.png`      | Comparison of price distributions by region and property type |
| `feature impact.png`            | Overall SHAP feature importance ranking                       |
| `shape landsize & distance.png` | SHAP analysis showing how distance and land size affect price |
| `shape houseprice.png`          | Distribution of predicted house prices across the dataset     |

