# from pydantic import BaseModel
# from typing import Optional, Dict

# class PatientData(BaseModel):
#     AGE: float
#     GENDER: str
#     BMI: float
#     Smoking_Status: str
#     PHQ_2: float
#     CRP_Estimate: Optional[float] = None
#     Blood_Glucose_HbA1c: Optional[float] = None
#     Hypertension_Systolic: Optional[float] = None
#     Hypertension_Diastolic: Optional[float] = None
#     Medication_Use: Optional[str] = None
#     gum_disease: Optional[int] = None
#     oral_lesions_present: Optional[int] = None
#     brushing_frequency: Optional[float] = None
#     bleeding_on_brushing: Optional[int] = None
#     dry_mouth: Optional[int] = None
#     plaque_level: Optional[float] = None
#     cej_to_bone_crest_mm: Optional[float] = None
#     missing_teeth_count: Optional[int] = None

# class PredictionResponse(BaseModel):
#     predictions: Dict[str, str]  # e.g., {"Diabetes Risk": "Moderate Risk"}


from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel
from typing import Optional

class PatientData(BaseModel):
    AGE: float
    Smoking_Status: str
    Medication_Use: str  # Accepts "Yes"/"No" from frontend, converted in backend
    PHQ_2: float
    BMI: float
    Blood_Glucose_HbA1c: float
    Hypertension_Systolic: float
    Hypertension_Diastolic: float
    CRP_Estimate: float
    missing_teeth_count: int
    gum_disease: str  # Accepts "Yes"/"No" from frontend, converted in backend
    dental_visits_yearly: int
    has_cavities: str  # Accepts "Yes"/"No" from frontend, converted in backend
    brushing_frequency: float
    plaque_level: str  # "Low", "Medium", "High"
    bleeding_on_brushing: str  # Accepts "Yes"/"No" from frontend, converted in backend
    oral_lesions_present: str  # Accepts "Yes"/"No" from frontend, converted in backend
    dry_mouth: str  # Accepts "Yes"/"No" from frontend, converted in backend
    total_root_length_mm: float
    cej_to_bone_crest_mm: float
class RiskInfo(BaseModel):
    probability: float  # e.g., 0.82
    risk_level: str     # e.g., "High Risk"

class PredictionResponse(BaseModel):
    predictions: dict[str, RiskInfo]  # e.g., {"Diabetes Risk": {"probability": 0.82, "risk_level": "High Risk"}}
