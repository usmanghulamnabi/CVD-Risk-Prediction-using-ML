# =========================
# CVD Risk Prediction System
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import plotly.express as px
import shap
import plotly.graph_objects as go
import numpy as np
import joblib


# Load dataset
data = pd.read_csv('CVD Dataset.csv')

# Drop rows where target is missing
data = data.dropna(subset=['CVD Risk Score', 'CVD Risk Level'])


# Map categorical features to numeric
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['Smoking Status'] = data['Smoking Status'].map({'Y': 1, 'N': 0})
data['Diabetes Status'] = data['Diabetes Status'].map({'Y': 1, 'N': 0})
data['Family History of CVD'] = data['Family History of CVD'].map({'Y': 1, 'N': 0})
data['Physical Activity Level'] = data['Physical Activity Level'].map({
    'Low': 0,
    'Moderate': 1,
    'High': 2
})
data['Blood Pressure Category'] = data['Blood Pressure Category'].map({
    'Normal': 0,
    'Elevated': 1,
    'Hypertension Stage 1': 2,
    'Hypertension Stage 2': 3
})
data['CVD Risk Level'] = data['CVD Risk Level'].map({'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2})


# Features & targets
X = data[['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)',
          'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
          'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD',
          'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP', 'Blood Pressure Category', 'Estimated LDL (mg/dL)']]

y_score = data['CVD Risk Score'].values
y_level = data['CVD Risk Level'].values


# REGRESSION PIPELINE (CVD Risk Score)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
X_train_s = imputer.fit_transform(X_train_s)
X_test_s = imputer.transform(X_test_s)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_s)
X_test_s = scaler.transform(X_test_s)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=300, random_state=42)
regressor.fit(X_train_s, y_train_s)
y_pred_s = regressor.predict(X_test_s)

# Evaluation
'''print("=== REGRESSION (CVD Risk Score) ===")
print("Mean Squared Error:", mean_squared_error(y_test_s, y_pred_s))
print("R^2 Score:", r2_score(y_test_s, y_pred_s))
print("\n")'''

# CLASSIFICATION PIPELINE (CVD Risk Level)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_level, test_size=0.2, random_state=42, stratify=y_level
)

y_train_c = y_train_c.ravel()
y_test_c = y_test_c.ravel()

# Impute missing values
X_train_c = imputer.fit_transform(X_train_c)
X_test_c = imputer.transform(X_test_c)

# Scale features
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# Oversample with SMOTE
smote = SMOTE(random_state=42)
X_train_c, y_train_c = smote.fit_resample(X_train_c, y_train_c)

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)
xgb_clf.fit(X_train_c, y_train_c)
y_pred_c = xgb_clf.predict(X_test_c)

# Evaluation
'''
print(classification_report(y_test_c, y_pred_c))

# Print predictions with patient serial number
for i, (score, level) in enumerate(zip(y_pred_s, y_pred_c)):
    print(f"Patient {i+1}: Predicted Score = {score:.2f}, Predicted Level = {level}")'''

# Save the models
joblib.dump(regressor, "cvd_risk_score_model.pkl")
joblib.dump(xgb_clf, "cvd_risk_level_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")


# Map numeric levels back to labels for clarity
level_labels = {0: 'Low Risk', 1: 'Intermediary Risk', 2: 'High Risk'}
pred_labels = [level_labels[level] for level in y_pred_c]

# Count the occurrences of each level
level_counts = pd.Series(pred_labels).value_counts()

# Create a pie chart
level_labels = {0:'Low Risk',1:'Intermediary Risk',2:'High Risk'}
pred_labels = [level_labels[l] for l in y_pred_c]
level_counts = pd.Series(pred_labels).value_counts()

pie = go.Pie(
    labels=level_counts.index,
    values=level_counts.values,
    name='CVD Risk Level Distribution',
    marker=dict(colors=['green','orange','red']),
    domain=dict(x=[0,0.45])  # left half
)

# -------------------------------
# 4️⃣ SHAP for feature importance
# -------------------------------

feature_names = ['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)',
                 'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
                 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD',
                 'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP', 'Blood Pressure Category',
                 'Estimated LDL (mg/dL)']

# Convert X_test_c to DataFrame for SHAP (use explicit feature_names)
X_test_c_df = pd.DataFrame(X_test_c, columns=feature_names)

# SHAP explainer for classifier
explainer = shap.TreeExplainer(xgb_clf)
class_idx = 2  # explain HIGH risk class
n_features = len(feature_names)

def _extract_shap_class(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == n_features:
        # (n_samples, n_features)
        return arr
    if arr.ndim == 3:
        # try common shapes:
        # (n_samples, n_classes, n_features)
        if arr.shape[2] == n_features:
            return arr[:, class_idx, :]
        # (n_classes, n_samples, n_features)
        if arr.shape[2] == n_features:
            return arr[class_idx, :, :]
        # (n_samples, n_features, n_classes) — unlikely, but try to find features axis
        if arr.shape[1] == n_features:
            return arr[:, :, class_idx]
    raise ValueError(f"Unsupported SHAP array shape: {arr.shape}")

shap_class = None
# Try older API first (explainer.shap_values), fallback to newer explainer(...)
try:
    shap_vals = explainer.shap_values(X_test_c_df)
    # shap_vals may be list (per-class) or ndarray
    if isinstance(shap_vals, list):
        shap_class = np.asarray(shap_vals[class_idx])
    else:
        shap_class = _extract_shap_class(shap_vals)
except Exception:
    # Newer SHAP: Explanation object returned by explainer(X)
    shap_expl = explainer(X_test_c_df)
    vals = getattr(shap_expl, "values", None)
    if vals is None:
        raise RuntimeError("SHAP explainer did not return values.")
    shap_class = _extract_shap_class(vals)

# Mean absolute contribution per feature
mean_contrib = np.abs(shap_class).mean(axis=0)

# Create DataFrame for plotting
feature_contrib = pd.DataFrame({
    'Feature': feature_names,
    'Contribution': mean_contrib
}).sort_values(by='Contribution', ascending=True)  # ascending for horizontal bar


# Pie chart of predicted risk levels
level_labels = {0:'Low Risk', 1:'Intermediary Risk', 2:'High Risk'}
pred_labels = [level_labels[l] for l in y_pred_c]
level_counts = pd.Series(pred_labels).value_counts()

pie = go.Pie(
    labels=level_counts.index,
    values=level_counts.values,
    name='CVD Risk Level Distribution',
    marker=dict(colors=['green','orange','pink']),
    domain=dict(x=[0,0.45])
)

# Bar chart for SHAP feature importance
bar = go.Bar(
    x=feature_contrib['Contribution'],
    y=feature_contrib['Feature'],
    orientation='h',
    name='Feature Importance (SHAP)',
    marker_color='skyblue',
    xaxis='x2',
    yaxis='y2'
)

# Combine in a single figure with 2 subplots
fig = go.Figure()

fig.add_trace(pie)
fig.add_trace(bar)

fig.update_layout(
    title='CVD Risk Level Distribution & Feature Importance',
    height=780,
    width=1520,
    xaxis=dict(domain=[0, 0.45]),
    xaxis2=dict(domain=[0.6, 0.95]),
    yaxis=dict(domain=[0, 1]),
    yaxis2=dict(domain=[0, 1], anchor='x2')
)

fig.show()