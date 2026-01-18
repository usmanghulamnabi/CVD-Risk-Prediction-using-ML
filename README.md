# CVD Risk Prediction System

A **Python-based CVD (Cardiovascular Disease) Risk Prediction System** that predicts both the **CVD Risk Score** and **Risk Level** using machine learning models. The app provides a modern GUI built with **PyQt6**, allowing users to input patient data and visualize results with interactive **pie charts** and **feature importance graphs (SHAP)**.

---

## Features

- Predict **CVD Risk Score** (regression) and **CVD Risk Level** (classification)
- Interactive **pie chart** showing distribution of risk levels
- **SHAP feature importance** bar chart to understand model decisions
- Supports **custom input** for single or multiple patients
- Models include:
  - RandomForestRegressor for risk score
  - XGBoostClassifier for risk level
- Preprocessing pipeline with:
  - KNN Imputer for missing values
  - StandardScaler for normalization
  - SMOTE for handling class imbalance
- Save/load models with **joblib**
- Modern GUI using **PyQt6**
- Easy to extend with additional features or ML models
