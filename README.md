# 🏠 Melbourne Housing Price Analysis & Prediction

A complete data science and ML project on Melbourne housing prices, covering EDA, data cleaning, feature engineering, and predictive modeling using Random Forest and XGBoost, with explainability via SHAP — all aimed at understanding key pricing factors and building an accurate price prediction model.

## 📌 Objectives

- 📊 Analyze real estate market trends in Melbourne  
- 🧠 Identify key features influencing property prices  
- 🤖 Build accurate ML models for price prediction  
- 🌐 (Optional) Deploy a user-friendly prediction interface using Streamlit

---

## 🧰 Tools & Technologies

| Category            | Tools Used                                               |
|---------------------|----------------------------------------------------------|
| Language        | Python                                                   |
| Libraries       | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |
| ML Models       | Linear Regression, Random Forest, XGBoost                |
| Visualization   | seaborn, matplotlib, SHAP                                |
| Deployment      | Streamlit (optional)                                     |
| Version Control | Git & GitHub                                             |

---

## 📊 Dataset

- [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) from Kaggle  
> A clean, public dataset suitable for regression tasks and real estate analysis.

---

## 🔍 Workflow

### 1️⃣ Data Cleaning
- ❌ Removed duplicates & missing values  
- 📏 Treated outliers in key features (Landsize, BuildingArea, etc.)
- Removed rows with:
  - `Distance of city center = 0`
  - `Bathroom = 0` or ≥ 6
  - `Rooms < 2` or ≥ 6
  - `Car > 5`
  - `Landsize < 10` or > 2000(m)
- Dropped missing values
- Removed rare regions: 'Eastern Victoria', etc.
- 
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
- 🧮 Categorical encoding (e.g., Type, Regionname)
- ✅ Selected top 7 impactful features based on EDA & model importance

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

### 6️⃣ Deployment (Optional)
- ⚙️ Streamlit app (coming soon):
  - User inputs house details → predicted price 💸

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
- 💡 Auto-interpret predictions via SHAP in Streamlit

---

## 🛠 Run Locally

```bash
git clone https://github.com/yourusername/real-estate-price-prediction.git
cd real-estate-price-prediction
pip install -r requirements.txt
streamlit run app/main.py
