import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.utils import preprocess_input

router = APIRouter()

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# === Disease model file map ===
model_files = {
    "Diabetes Risk": os.path.join(MODELS_DIR, "xgboost_model_diabetes_risk.json"),
    "Cardiovascular Disease Risk": os.path.join(MODELS_DIR, "xgboost_model_cvd_risk.json"),
    "Chronic Kidney Disease (CKD)": os.path.join(MODELS_DIR, "xgboost_model_CKD.json"),
    "Autoimmune Disorder": os.path.join(MODELS_DIR, "xgboost_model_Autoimmune_Disorder.json"),
}

# === Load models ===
models = {}
for name, path in model_files.items():
    clf = xgb.XGBClassifier()
    clf.load_model(path)
    models[name] = clf

# === Feature list ===
ALL_FEATURES = [
    "AGE", "Smoking_Status", "Medication_Use", "PHQ_2", "BMI",
    "Blood_Glucose_HbA1c", "Hypertension_Systolic", "Hypertension_Diastolic",
    "CRP_Estimate", "missing_teeth_count", "gum_disease", "dental_visits_yearly",
    "has_cavities", "brushing_frequency", "plaque_level", "bleeding_on_brushing",
    "oral_lesions_present", "dry_mouth", "total_root_length_mm",
    "cej_to_bone_crest_mm", "bone_loss_percent"
]

# === Disease keys for chart route ===
key_map = {
    "diabetes_risk": "Diabetes Risk",
    "cvd_risk": "Cardiovascular Disease Risk",
    "CKD": "Chronic Kidney Disease (CKD)",
    "Autoimmune_Disorder": "Autoimmune Disorder"
}

# === Risk Label Logic ===
def get_risk_label(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# === Prediction logic ===
@router.post("/predict_all")
def predict_all_risks(user_input: dict):
    results = {}

    for disease, clf in models.items():
        row = {k: user_input.get(k, 0) for k in ALL_FEATURES if k != "bone_loss_percent"}

        # Compute derived feature
        cej = float(row.get("cej_to_bone_crest_mm", 0))
        root = float(row.get("total_root_length_mm", 0))
        bone_loss_percent = round((cej / root) * 100, 1) if root > 0 else 0.0
        row["bone_loss_percent"] = bone_loss_percent

        # Yes/No binary mapping
        yes_no_features = [
            "Medication_Use", "gum_disease", "has_cavities",
            "bleeding_on_brushing", "oral_lesions_present", "dry_mouth"
        ]
        for k in yes_no_features:
            val = str(row.get(k, "No")).strip().lower()
            row[k] = 1 if val == "yes" else 0

        # Numeric cast
        numeric_features = [
            "AGE", "PHQ_2", "BMI", "Blood_Glucose_HbA1c", "Hypertension_Systolic",
            "Hypertension_Diastolic", "CRP_Estimate", "missing_teeth_count",
            "dental_visits_yearly", "brushing_frequency", "total_root_length_mm",
            "cej_to_bone_crest_mm", "bone_loss_percent"
        ]
        for k in numeric_features:
            try:
                row[k] = float(row.get(k, 0))
            except:
                row[k] = 0.0

        # Categorical
        categorical_features = ["Smoking_Status", "plaque_level"]
        for k in categorical_features:
            row[k] = str(row.get(k, ""))

        try:
            processed = preprocess_input(row)
            X_input = np.array(processed, dtype=np.float32).reshape(1, -1)

            if X_input.shape[1] != clf.n_features_in_:
                raise ValueError(
                    f"{disease}: input feature mismatch — model expects {clf.n_features_in_}, got {X_input.shape[1]}"
                )

            prob = clf.predict_proba(X_input)[0][1]
            risk_level = get_risk_label(prob)
        except Exception as e:
            print(f"[{disease}] Error: {str(e)}")
            prob = 0.0
            risk_level = "Error"

        results[disease] = {
            "probability": round(float(prob), 2),
            "risk_level": risk_level
        }

    return results

# === Feature importance image route ===
@router.get("/feature-importance/{key}")
def feature_img(key: str):
    disease = key_map.get(key)
    clf = models.get(disease)
    if not clf:
        return JSONResponse(status_code=404, content={"error": "not found"})

    booster = clf.get_booster()
    importance_dict = booster.get_score(importance_type="gain")

    feature_map = {f"f{i}": ALL_FEATURES[i] for i in range(len(ALL_FEATURES))}
    top_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    feature_names = [feature_map.get(name, name) for name, _ in top_items]
    importances = [round(val, 4) for _, val in top_items]

    # Gradient colors
    colors = ['#800080', '#d63384', '#3399ff', '#cc66ff', '#9933cc']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(range(len(importances)), importances, color=colors[:len(importances)])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title(f"Top Features - {disease}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img = base64.b64encode(buf.getbuffer()).decode("utf-8")
    return {"image": img}
