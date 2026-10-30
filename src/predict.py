"""Predict ME/CFS vs Depression vs Both for one patient record.

Usage:
    python src/predict.py

The example record below can be edited. Categorical values must match the
categories seen in training (see models/label_encoders.pkl); unknown values
raise an error instead of being silently replaced.
"""
import pickle
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_final.pkl"
ENCODER_PATH = ROOT / "models" / "label_encoders.pkl"

# Class codes used when the model was trained (see notebook, section C.1)
CLASS_LABELS = {0: "Depression", 1: "ME/CFS", 2: "Both"}

EXAMPLE_INPUT = {
    "age": 45,
    "gender": "Female",
    "sleep_quality_index": 3.0,
    "brain_fog_level": 6.0,
    "physical_pain_score": 4.0,
    "stress_level": 5.0,
    "depression_phq9_score": 12.0,
    "fatigue_severity_scale_score": 7.0,
    "pem_duration_hours": 24.0,
    "hours_of_sleep_per_night": 6.0,
    "pem_present": 1.0,
    "work_status": "Not working",
    "social_activity_level": "Very low",
    "exercise_frequency": "Sometimes",
    "meditation_or_mindfulness": "No",
}


def predict(record: dict) -> tuple[int, str]:
    model = joblib.load(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    df = pd.DataFrame([record])
    for col, enc in encoders.items():
        value = str(df.at[0, col])
        if value not in enc.classes_:
            raise ValueError(
                f"Invalid value {value!r} for '{col}'. Allowed: {list(enc.classes_)}"
            )
        df[col] = enc.transform(df[col].astype(str))

    # Keep the exact column order used at training time
    df = df[list(model.feature_names_in_)]
    code = int(model.predict(df)[0])
    return code, CLASS_LABELS[code]


if __name__ == "__main__":
    code, label = predict(EXAMPLE_INPUT)
    print(f"Prediction category code: {code}")
    print(f"Prediction category: {label}")
