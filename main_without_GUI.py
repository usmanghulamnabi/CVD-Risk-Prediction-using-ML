# =========================
# CVD Risk Prediction GUI - PyQt6
# =========================

import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
import joblib
import plotly.graph_objects as go
import shap

# ----------------------
# Load models and preprocessors
# ----------------------
regressor = joblib.load("cvd_risk_score_model.pkl")
classifier = joblib.load("cvd_risk_level_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

feature_cols = ['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)',
                'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
                'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD',
                'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP', 'Blood Pressure Category', 'Estimated LDL (mg/dL)']

# ----------------------
# Mapping for dropdowns
# ----------------------
sex_map = {'M':1,'F':0}
yn_map = {'Y':1,'N':0}
pa_map = {'Low':0,'Moderate':1,'High':2}
bp_map = {'Normal':0,'Elevated':1,'Hypertension Stage 1':2,'Hypertension Stage 2':3}

level_labels = {0:'Low Risk',1:'Intermediary Risk',2:'High Risk'}

# ----------------------
# GUI Class
# ----------------------
class CVDRiskApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVD Risk Prediction System")
        self.setGeometry(100,100,1400,800)
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout(self)
        
        # -------------------
        # Left panel - Input fields
        # -------------------
        left_panel = QVBoxLayout()
        
        self.inputs = {}
        for feature in feature_cols:
            h = QHBoxLayout()
            lbl = QLabel(feature)
            if feature in ['Sex','Smoking Status','Diabetes Status','Family History of CVD','Physical Activity Level','Blood Pressure Category']:
                combo = QComboBox()
                if feature=='Sex': combo.addItems(['M','F'])
                elif feature in ['Smoking Status','Diabetes Status','Family History of CVD']: combo.addItems(['Y','N'])
                elif feature=='Physical Activity Level': combo.addItems(['Low','Moderate','High'])
                elif feature=='Blood Pressure Category': combo.addItems(['Normal','Elevated','Hypertension Stage 1','Hypertension Stage 2'])
                self.inputs[feature] = combo
                h.addWidget(lbl)
                h.addWidget(combo)
            else:
                line = QLineEdit()
                self.inputs[feature] = line
                h.addWidget(lbl)
                h.addWidget(line)
            left_panel.addLayout(h)
        
        # Predict button
        predict_btn = QPushButton("Predict Risk")
        predict_btn.clicked.connect(self.predict_risk)
        left_panel.addWidget(predict_btn)
        
        # Labels for results
        self.score_label = QLabel("Predicted Risk Score: ")
        self.level_label = QLabel("Predicted Risk Level: ")
        left_panel.addWidget(self.score_label)
        left_panel.addWidget(self.level_label)
        
        layout.addLayout(left_panel, 1)
        
        # -------------------
        # Right panel - Plotly charts
        # -------------------
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view, 2)
        
    def predict_risk(self):
        # Collect inputs
        x = []
        for feature in feature_cols:
            widget = self.inputs[feature]
            if isinstance(widget,QLineEdit):
                val = float(widget.text())
            elif isinstance(widget,QComboBox):
                text = widget.currentText()
                if feature=='Sex': val = sex_map[text]
                elif feature in ['Smoking Status','Diabetes Status','Family History of CVD']: val = yn_map[text]
                elif feature=='Physical Activity Level': val = pa_map[text]
                elif feature=='Blood Pressure Category': val = bp_map[text]
            x.append(val)
        
        x_array = np.array(x).reshape(1,-1)
        x_imputed = imputer.transform(x_array)
        x_scaled = scaler.transform(x_imputed)
        
        # Predictions
        score_pred = regressor.predict(x_scaled)[0]
        level_pred = classifier.predict(x_scaled)[0]
        
        self.score_label.setText(f"Predicted Risk Score: {score_pred:.2f}")
        self.level_label.setText(f"Predicted Risk Level: {level_labels[level_pred]}")
        
        # -------------------
        # Create Pie Chart (1 patient, highlight risk level)
        # -------------------
        pie = go.Pie(
            labels=list(level_labels.values()),
            values=[1 if i==level_pred else 0 for i in range(3)],
            marker=dict(colors=['green','orange','red']),
            name='Predicted Risk Level'
        )
        
        # -------------------
        # SHAP Bar chart
        # -------------------
        X_input_df = pd.DataFrame(x_scaled, columns=feature_cols)
        explainer = shap.TreeExplainer(classifier)

        def _shap_for_class(shap_out, class_idx=2, n_features=len(feature_cols)):
            arr = np.asarray(shap_out)
            # (n_samples, n_features)
            if arr.ndim == 2 and arr.shape[1] == n_features:
                return arr
            # (n_samples, n_classes, n_features)
            if arr.ndim == 3:
                if arr.shape[2] == n_features:
                    return arr[:, class_idx, :]
                # (n_samples, n_features, n_classes)
                if arr.shape[1] == n_features:
                    return arr[:, :, class_idx]
                # (n_classes, n_samples, n_features)
                if arr.shape[0] != n_features and arr.shape[1] == n_features:
                    return arr[class_idx, :, :]
            raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")

        # Try older and newer SHAP APIs
        try:
            shap_vals = explainer.shap_values(X_input_df)
        except Exception:
            shap_expl = explainer(X_input_df)
            shap_vals = getattr(shap_expl, "values", shap_expl)

        # Extract contributions for the class of interest
        if isinstance(shap_vals, list):
            shap_class = np.asarray(shap_vals[2])  # list per class
        else:
            shap_class = _shap_for_class(shap_vals)

        # Aggregate per-feature contribution (mean over samples)
        mean_contrib = np.abs(shap_class).mean(axis=0)

        # Ensure lengths match feature list
        if mean_contrib.shape[0] != len(feature_cols):
            raise RuntimeError(f"SHAP feature count {mean_contrib.shape[0]} != features {len(feature_cols)}")

        feature_contrib = pd.DataFrame({
            'Feature': feature_cols,
            'Contribution': mean_contrib
        }).sort_values(by='Contribution', ascending=True)

        
        bar = go.Bar(
            x=feature_contrib['Contribution'],
            y=feature_contrib['Feature'],
            orientation='h',
            marker_color='skyblue',
            name='Feature Importance (SHAP)'
        )

        fig = go.Figure()
        fig.add_trace(bar)
        fig.add_trace(pie)
        fig.update_layout(title=
            'Predicted Risk & Feature Importance',
            height=700,
            width=900,
            xaxis=dict(domain=[0,0.45]),
            xaxis2=dict(domain=[0.55,0.95]),
            yaxis=dict(domain=[0,1]),
            yaxis2=dict(domain=[0,1], anchor='x2')
        )
        # Place bar chart on the left and pie chart on the right
        # ensure pie domain is on the right side
        pie.domain = dict(x=[0.6, 0.95], y=[0, 1])

        # Put bar first (left) then pie (right)
        fig = go.Figure(data=[bar, pie])
        fig.update_layout(
            title='Predicted Risk & Feature Importance',
            height=700,
            width=900,
            xaxis=dict(domain=[0, 0.55]),  # left area for bar
            margin=dict(l=40, r=20, t=60, b=40)
        )

        html = fig.to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html)

# ----------------------
# Run App
# ----------------------
if __name__=="__main__":
    app = QApplication(sys.argv)
    window = CVDRiskApp()
    window.show()
    sys.exit(app.exec())
