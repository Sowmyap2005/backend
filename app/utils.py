from sklearn.preprocessing import LabelEncoder
import numpy as np
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv

# Global encoders per categorical field
label_encoders = {}

load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # Set JWT_SECRET_KEY in .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def safe_transform(le: LabelEncoder, val: str) -> int:
    """
    Ensures the label encoder can handle unseen labels by dynamically expanding classes_.
    """
    if val not in le.classes_:
        le.classes_ = np.append(le.classes_, val)
    return le.transform([val])[0]

def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocesses a dictionary of patient features into a numeric input vector.
    - Handles missing values (None, empty string)
    - Dynamically encodes categorical variables with LabelEncoder
    - Returns a flat float array suitable for model prediction
    """
    processed = []

    for key, val in data.items():
        # Handle missing or blank values
        if val is None or (isinstance(val, str) and val.strip() == ""):
            val = 0

        # Handle categorical features
        elif isinstance(val, str):
            val = val.strip()
            if key not in label_encoders:
                label_encoders[key] = LabelEncoder()
                label_encoders[key].fit([val])  # Initial fit
            le = label_encoders[key]
            val = safe_transform(le, val)

        processed.append(float(val))

    return np.array(processed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None