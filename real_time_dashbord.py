import os
import json
import math
import time
import random
import pickle
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ICU-RT")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("ICU_PROJECT_ROOT", "/Users/bsoft/Desktop/hanane_project"))
DIABETES_MODEL_FILE = "Diabetes Risk.pkl"
DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
DEFAULT_MODEL_DIRS = [
    Path(os.getenv("ICU_MODELS_DIR", "")) if os.getenv("ICU_MODELS_DIR") else None,
    BASE_DIR / "saved",
    BASE_DIR,
    BASE_DIR / "project2" / "saved",
    PROJECT_ROOT / "project2" / "saved",
    PROJECT_ROOT,
    Path("/Users/bsoft/Desktop/hanane_project/project2/saved"),
    Path("/Users/bsoft/Desktop/hanane_project"),
]
MODEL_DIR_CANDIDATES = [p for p in DEFAULT_MODEL_DIRS if p is not None]
MODEL_FILES = {
    "Heart Rate Alert": "Heart Rate (bpm).pkl",
    "SpO2 Level Alert": "SpO2 Level (%).pkl",
    "Blood Pressure Alert": "Blood Pressure (mmHg).pkl",
    "Temperature Alert": "Body Temperature (°C).pkl",
    "Predicted Disease": "Predicted Disease.pkl",
    "Diabetes Risk": DIABETES_MODEL_FILE,
}

STREAM_INTERVAL_MS = 20
DISPLAY_INTERVAL_MS = 800
HISTORY_POINTS = 60
STORAGE_HISTORY_LIMIT = 500
STORAGE_DIR = BASE_DIR / "storage_historique"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_HISTORY_FILE = STORAGE_DIR / "patient_history.json"
NURSE_LOGS_FILE = STORAGE_DIR / "nurse_logs.json"
EMERGENCY_REQUESTS_FILE = STORAGE_DIR / "emergency_requests.json"

CLINICAL_THRESHOLDS = {
    "Heart Rate (bpm)": {"CL": 40, "WL": 50, "NL": 60, "NH": 100, "WH": 120, "CH": 150},
    "SpO2 Level (%)": {"CL": 85, "WL": 90, "NL": 95, "NH": 100, "WH": 100, "CH": 100},
    "Systolic Blood Pressure (mmHg)": {"CL": 70, "WL": 90, "NL": 90, "NH": 130, "WH": 140, "CH": 180},
    "Diastolic Blood Pressure (mmHg)": {"CL": 40, "WL": 60, "NL": 60, "NH": 90, "WH": 90, "CH": 120},
    "Body Temperature (°C)": {"CL": 34.0, "WL": 35.0, "NL": 36.1, "NH": 37.5, "WH": 38.3, "CH": 40.0},
}



def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


PATIENT_PROFILES = [
    {
        "id": "PT-REAL-001",
        "name": "Real Sensor Patient",
        "age": 23,
        "ward": "Prototype",
        "scenario": "real_sensor",
        "base": {
            "hr": 80,
            "spo2": 98,
            "sbp": 120,
            "dbp": 80,
            "temp": 37.0
        }
    },

    {"id": "PT-001", "name": "Ahmed Benali", "age": 58, "ward": "ICU-A", "scenario": "stable", "base": {"hr": 75, "spo2": 97, "sbp": 120, "dbp": 78, "temp": 36.8}},
    {"id": "PT-002", "name": "Fatima Zahra", "age": 42, "ward": "ICU-A", "scenario": "tachycardia", "base": {"hr": 108, "spo2": 94, "sbp": 145, "dbp": 92, "temp": 38.1}},
    {"id": "PT-003", "name": "Omar Mansouri", "age": 67, "ward": "ICU-B", "scenario": "hypoxia", "base": {"hr": 95, "spo2": 88, "sbp": 100, "dbp": 65, "temp": 37.9}},
    {"id": "PT-004", "name": "Nadia Haddad", "age": 35, "ward": "ICU-B", "scenario": "hypertensive_crisis", "base": {"hr": 88, "spo2": 96, "sbp": 172, "dbp": 115, "temp": 37.2}},
    {"id": "PT-005", "name": "Karim Boukhari", "age": 72, "ward": "ICU-C", "scenario": "bradycardia", "base": {"hr": 47, "spo2": 92, "sbp": 88, "dbp": 55, "temp": 36.2}},
    {"id": "PT-006", "name": "Salma Ouali", "age": 29, "ward": "ICU-C", "scenario": "sepsis", "base": {"hr": 118, "spo2": 91, "sbp": 85, "dbp": 50, "temp": 39.4}},
]

PATIENT_LOOKUP = {p["id"]: p for p in PATIENT_PROFILES}
SEVERITY_ORDER = ["NORMAL", "CAUTION", "WARNING", "CRITICAL"]
PATIENT_BUTTON_ALERTS = {"Urgent Help", "Breathing Difficulty", "Pain", "Need Help"}


def first_patient_button(*values: Optional[str]) -> str:
    """Return the first real patient request button from ESP32 fields."""
    for value in values:
        if value in PATIENT_BUTTON_ALERTS:
            return value
    return "None"


class SyntheticVitalGenerator:
    SCENARIO_PARAMS = {
        "stable": {
            "hr": {"noise": 1.2, "drift": 0.0, "event_prob": 0.002, "event_mag": 10},
            "spo2": {"noise": 0.3, "drift": 0.0, "event_prob": 0.001, "event_mag": -3},
            "sbp": {"noise": 2.0, "drift": 0.0, "event_prob": 0.002, "event_mag": 15},
            "dbp": {"noise": 1.5, "drift": 0.0, "event_prob": 0.002, "event_mag": 10},
            "temp": {"noise": 0.03, "drift": 0.0, "event_prob": 0.0005, "event_mag": 0.3},
        },
        "tachycardia": {
            "hr": {"noise": 2.5, "drift": 0.01, "event_prob": 0.005, "event_mag": 20},
            "spo2": {"noise": 0.5, "drift": -0.002, "event_prob": 0.003, "event_mag": -4},
            "sbp": {"noise": 3.0, "drift": 0.02, "event_prob": 0.003, "event_mag": 20},
            "dbp": {"noise": 2.0, "drift": 0.01, "event_prob": 0.003, "event_mag": 12},
            "temp": {"noise": 0.05, "drift": 0.001, "event_prob": 0.001, "event_mag": 0.5},
        },
        "hypoxia": {
            "hr": {"noise": 2.0, "drift": 0.02, "event_prob": 0.004, "event_mag": 15},
            "spo2": {"noise": 0.8, "drift": -0.01, "event_prob": 0.008, "event_mag": -5},
            "sbp": {"noise": 2.5, "drift": -0.02, "event_prob": 0.003, "event_mag": -15},
            "dbp": {"noise": 1.5, "drift": -0.01, "event_prob": 0.003, "event_mag": -8},
            "temp": {"noise": 0.06, "drift": 0.001, "event_prob": 0.001, "event_mag": 0.4},
        },
        "hypertensive_crisis": {
            "hr": {"noise": 1.5, "drift": 0.005, "event_prob": 0.003, "event_mag": 12},
            "spo2": {"noise": 0.3, "drift": -0.001, "event_prob": 0.001, "event_mag": -2},
            "sbp": {"noise": 4.0, "drift": 0.05, "event_prob": 0.006, "event_mag": 30},
            "dbp": {"noise": 2.5, "drift": 0.03, "event_prob": 0.006, "event_mag": 18},
            "temp": {"noise": 0.04, "drift": 0.0, "event_prob": 0.0005, "event_mag": 0.2},
        },
        "bradycardia": {
            "hr": {"noise": 1.0, "drift": -0.01, "event_prob": 0.005, "event_mag": -12},
            "spo2": {"noise": 0.6, "drift": -0.005, "event_prob": 0.004, "event_mag": -4},
            "sbp": {"noise": 2.0, "drift": -0.03, "event_prob": 0.004, "event_mag": -18},
            "dbp": {"noise": 1.2, "drift": -0.02, "event_prob": 0.004, "event_mag": -10},
            "temp": {"noise": 0.04, "drift": -0.001, "event_prob": 0.001, "event_mag": -0.3},
        },
        "sepsis": {
            "hr": {"noise": 3.0, "drift": 0.03, "event_prob": 0.007, "event_mag": 25},
            "spo2": {"noise": 1.0, "drift": -0.015, "event_prob": 0.008, "event_mag": -6},
            "sbp": {"noise": 4.0, "drift": -0.04, "event_prob": 0.007, "event_mag": -25},
            "dbp": {"noise": 2.5, "drift": -0.025, "event_prob": 0.007, "event_mag": -15},
            "temp": {"noise": 0.08, "drift": 0.003, "event_prob": 0.002, "event_mag": 0.8},
        },
    }

    BOUNDS = {
        "hr": (25, 220),
        "spo2": (70, 100),
        "sbp": (60, 220),
        "dbp": (30, 140),
        "temp": (33.0, 42.5),
    }

    def __init__(self):
        self.states: Dict[str, Dict[str, float]] = {}
        self.tick: Dict[str, int] = {}
        for p in PATIENT_PROFILES:
            self.states[p["id"]] = {k: float(v) for k, v in p["base"].items()}
            self.tick[p["id"]] = 0

    def _perturb(self, value: float, key: str, params: dict, tick: int) -> float:
        noise = np.random.normal(0, params["noise"])
        osc = params["noise"] * 0.5 * math.sin(tick * 0.02)
        drift = params["drift"]
        event = 0.0
        if random.random() < params["event_prob"]:
            event = params["event_mag"] * random.choice([-1, 1]) * random.uniform(0.5, 1.0)
        lo, hi = self.BOUNDS[key]
        return float(np.clip(value + noise + osc + drift + event, lo, hi))

    def next(self, patient_id: str) -> Dict[str, float]:
        profile = PATIENT_LOOKUP[patient_id]
        params = self.SCENARIO_PARAMS[profile["scenario"]]
        state = self.states[patient_id]
        tick = self.tick[patient_id]

        new_state = {
            "hr": self._perturb(state["hr"], "hr", params["hr"], tick),
            "spo2": self._perturb(state["spo2"], "spo2", params["spo2"], tick),
            "sbp": self._perturb(state["sbp"], "sbp", params["sbp"], tick),
            "dbp": self._perturb(state["dbp"], "dbp", params["dbp"], tick),
            "temp": self._perturb(state["temp"], "temp", params["temp"], tick),
        }
        if new_state["dbp"] >= new_state["sbp"]:
            new_state["dbp"] = float(np.clip(new_state["sbp"] - random.uniform(20, 35), 30, 140))

        self.states[patient_id] = new_state
        self.tick[patient_id] += 1
        return {
            "Heart Rate (bpm)": round(new_state["hr"], 1),
            "SpO2 Level (%)": round(new_state["spo2"], 1),
            "Systolic Blood Pressure (mmHg)": round(new_state["sbp"], 1),
            "Diastolic Blood Pressure (mmHg)": round(new_state["dbp"], 1),
            "Body Temperature (°C)": round(new_state["temp"], 2),
        }


class ModelRegistry:
    def __init__(self, candidates: List[Path]):
        self.candidates = candidates
        self.artifacts: Dict[str, dict] = {}
        self.models_dir = self._discover_models_dir()
        self._load_all()

    def _discover_models_dir(self) -> Path:
        for folder in self.candidates:
            if folder is None:
                continue
            if all((folder / filename).exists() for filename in MODEL_FILES.values()):
                logger.info("Using models directory: %s", folder)
                return folder
        for folder in self.candidates:
            if folder and folder.exists():
                logger.info("Using fallback models directory: %s", folder)
                return folder
        return BASE_DIR

    def _load_all(self):
        for target, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if not path.exists():
                logger.warning("Model not found: %s", path)
                continue
            try:
                with open(path, "rb") as f:
                    self.artifacts[target] = pickle.load(f)
                logger.info("Loaded model: %s", filename)
            except Exception as exc:
                logger.error("Failed to load %s: %s", filename, exc)

    def predict(self, target: str, feature_vec: np.ndarray) -> dict:
        artifact = self.artifacts.get(target)
        if artifact is None:
            return {"prediction": "N/A", "confidence": 0.0}
        try:
            model = artifact.get("calibrated") or artifact.get("model")
            scaler = artifact.get("scaler")
            label_encoder = artifact.get("label_encoder")
            X = feature_vec.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            conf = float(np.max(proba))
            if label_encoder is not None:
                label = label_encoder.inverse_transform([int(pred)])[0]
            else:
                label = "ABNORMAL" if int(pred) == 1 else "NORMAL"
            return {"prediction": label, "confidence": round(conf * 100, 1)}
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", target, exc)
            return {"prediction": "ERR", "confidence": 0.0}


def get_severity(vital: str, value: float) -> str:
    threshold = CLINICAL_THRESHOLDS.get(vital, {})
    if not threshold:
        return "NORMAL"
    if value <= threshold["CL"] or value >= threshold["CH"]:
        return "CRITICAL"
    if value <= threshold["WL"] or value >= threshold["WH"]:
        return "WARNING"
    if value < threshold["NL"] or value > threshold["NH"]:
        return "CAUTION"
    return "NORMAL"


def run_inference(registry: ModelRegistry, vitals: Dict[str, float]) -> dict:
    hr = vitals["Heart Rate (bpm)"]
    spo2 = vitals["SpO2 Level (%)"]
    sbp = vitals["Systolic Blood Pressure (mmHg)"]
    dbp = vitals["Diastolic Blood Pressure (mmHg)"]
    temp = vitals["Body Temperature (°C)"]

    pp = sbp - dbp
    mapp = dbp + pp / 3.0
    si = hr / max(sbp, 1)
    hs = hr * spo2 / 100.0
    bp_stage = 3 if (sbp >= 180 or dbp >= 120) else 2 if (sbp >= 140 or dbp >= 90) else 1 if (sbp >= 130 or dbp >= 80) else 0
    fever_stage = 4 if temp >= 40 else 3 if temp >= 39 else 2 if temp >= 38.3 else 1 if temp >= 37.5 else 0

    hr_pred = registry.predict("Heart Rate Alert", np.array([hr, si, hs], dtype=np.float32))
    spo2_pred = registry.predict("SpO2 Level Alert", np.array([spo2, hs], dtype=np.float32))
    bp_pred = registry.predict("Blood Pressure Alert", np.array([sbp, dbp, pp, mapp, bp_stage, si], dtype=np.float32))
    temp_pred = registry.predict("Temperature Alert", np.array([temp, fever_stage], dtype=np.float32))
    disease_pred = registry.predict(
        "Predicted Disease",
        np.array([hr, spo2, sbp, dbp, temp, pp, mapp, bp_stage, si, hs, fever_stage], dtype=np.float32),
    )

    threshold = {
        "hr": get_severity("Heart Rate (bpm)", hr),
        "spo2": get_severity("SpO2 Level (%)", spo2),
        "sbp": get_severity("Systolic Blood Pressure (mmHg)", sbp),
        "dbp": get_severity("Diastolic Blood Pressure (mmHg)", dbp),
        "temp": get_severity("Body Temperature (°C)", temp),
    }
    threshold["bp"] = SEVERITY_ORDER[max(SEVERITY_ORDER.index(threshold["sbp"]), SEVERITY_ORDER.index(threshold["dbp"]))]

    all_sev = list(threshold.values())
    for pred in [hr_pred, spo2_pred, bp_pred, temp_pred]:
        if pred["prediction"] == "ABNORMAL":
            all_sev.append("WARNING")
    overall = SEVERITY_ORDER[max(SEVERITY_ORDER.index(sev) for sev in all_sev)]

    return {
        "vitals": vitals,
        "threshold": threshold,
        "ml": {
            "hr": hr_pred,
            "spo2": spo2_pred,
            "bp": bp_pred,
            "temp": temp_pred,
            "disease": disease_pred,
        },
        "derived": {"pp": round(pp, 1), "map": round(mapp, 1), "si": round(si, 3)},
        "overall": overall,
        "ts": datetime.utcnow().isoformat(),
    }


def estimate_diabetes_type_distribution(payload: Dict[str, float]) -> dict:
    """Heuristic estimate because the dataset only provides binary Outcome labels."""
    age = float(payload.get("age", 0))
    bmi = float(payload.get("bmi", 0))
    insulin = float(payload.get("insulin", 0))
    glucose = float(payload.get("glucose", 0))
    pregnancies = float(payload.get("pregnancies", 0))

    type1_score = 1.0
    type2_score = 1.0

    if age <= 30:
        type1_score += 1.1
    else:
        type2_score += 1.0

    if bmi < 25:
        type1_score += 0.8
    elif bmi >= 30:
        type2_score += 1.2
    else:
        type2_score += 0.5

    if glucose >= 200 and insulin <= 90:
        type1_score += 0.8
    if insulin > 120:
        type2_score += 0.7
    if pregnancies > 0:
        type2_score += 0.3
    if age >= 45:
        type2_score += 0.8

    total = max(type1_score + type2_score, 1e-6)
    type1_pct = round(type1_score / total * 100, 1)
    type2_pct = round(100 - type1_pct, 1)
    predicted = "Type 1 Diabetes (estimated)" if type1_pct >= type2_pct else "Type 2 Diabetes (estimated)"
    return {
        "predicted_type": predicted,
        "type_percentages": {
            "Type 1 Diabetes": type1_pct,
            "Type 2 Diabetes": type2_pct,
        },
        "note": "Estimated from clinical profile because the available dataset has only binary diabetes-risk labels."
    }


def predict_diabetes_risk(registry: ModelRegistry, payload: Dict[str, float]) -> dict:
    artifact = registry.artifacts.get("Diabetes Risk")
    type_estimate = estimate_diabetes_type_distribution(payload)
    if artifact is None:
        return {
            "prediction": "MODEL_NOT_LOADED",
            "confidence": 0.0,
            "features_used": {},
            **type_estimate,
        }
    try:
        model = artifact.get("calibrated") or artifact.get("model")
        scaler = artifact.get("scaler")
        features = artifact.get("features") or DIABETES_FEATURES
        feature_map = {
            "Pregnancies": float(payload.get("pregnancies", 0)),
            "Glucose": float(payload.get("glucose", 0)),
            "BloodPressure": float(payload.get("blood_pressure", 0)),
            "SkinThickness": float(payload.get("skin_thickness", 0)),
            "Insulin": float(payload.get("insulin", 0)),
            "BMI": float(payload.get("bmi", 0)),
            "DiabetesPedigreeFunction": float(payload.get("diabetes_pedigree_function", 0)),
            "Age": float(payload.get("age", 0)),
        }
        X = np.array([[feature_map.get(name, 0.0) for name in features]], dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X)
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        return {
            "prediction": "Diabetes Risk" if pred == 1 else "Low Diabetes Risk",
            "confidence": round(confidence * 100, 1),
            "risk_percentages": {
                "Low Risk": round(float(proba[0]) * 100, 1),
                "Diabetes Risk": round(float(proba[1]) * 100, 1),
            },
            "features_used": feature_map,
            **type_estimate,
        }
    except Exception as exc:
        logger.warning("Diabetes prediction failed: %s", exc)
        return {
            "prediction": "ERR",
            "confidence": 0.0,
            "features_used": {},
            **type_estimate,
        }


class NurseAction(BaseModel):
    patient_id: str
    action_type: str
    item_name: str = ""
    note: str = ""
    performed_by: str = "Nurse"


class EmergencyRequest(BaseModel):
    patient_id: str
    request_type: str
    details: str = ""
    created_by: str = "Patient Button"


class EmergencyUpdate(BaseModel):
    status: str


class DiabetesRequest(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float
class RealSensorPayload(BaseModel):
    patient_id: str = "PT-REAL-001"

    # These can be None because the ESP32 sends null when the sensor is not detected.
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    systolic_bp: Optional[float] = 120
    diastolic_bp: Optional[float] = 80
    temperature: Optional[float] = None
    respiratory_rate: Optional[float] = None
    ecg_value: Optional[float] = None
    emergency_call: str = "None"
    piezo_vibration: Optional[float] = None
    source: str = "ESP32"

    # Extra hardware status fields sent by the ESP32
    bp_mode: str = "Simulated"
    system_status: Optional[str] = None
    wifi_status: Optional[str] = None
    max30102_status: Optional[str] = None
    finger_detected: Optional[str] = None
    ecg_status: Optional[str] = None
    temperature_status: Optional[str] = None
    piezo_status: Optional[str] = None
    buzzer_status: Optional[str] = None
    max30102_ir: Optional[int] = None
    max30102_red: Optional[int] = None

    # Extra button/monitoring fields sent by the new ESP32 5-button code
    monitoring_state: Optional[str] = None
    current_button: Optional[str] = None
    last_button_event: Optional[str] = None

    # Raw button debug values sent by ESP32
    btn_start_raw: Optional[int] = None
    btn_red_raw: Optional[int] = None
    btn_blue_raw: Optional[int] = None
    btn_yellow_raw: Optional[int] = None
    btn_white_raw: Optional[int] = None
    btn_start_state: Optional[str] = None
    btn_red_state: Optional[str] = None
    btn_blue_state: Optional[str] = None
    btn_yellow_state: Optional[str] = None
    btn_white_state: Optional[str] = None

generator = SyntheticVitalGenerator()
registry = ModelRegistry(MODEL_DIR_CANDIDATES)
loaded_history = load_json_file(PATIENT_HISTORY_FILE, {})
patient_history: Dict[str, List[dict]] = {
    p["id"]: list(loaded_history.get(p["id"], []))[-STORAGE_HISTORY_LIMIT:]
    for p in PATIENT_PROFILES
}
latest_state: Dict[str, dict] = {}
manual_overrides: Dict[str, bool] = {}
nurse_logs: List[dict] = list(load_json_file(NURSE_LOGS_FILE, []))[:50]
emergency_requests: List[dict] = list(load_json_file(EMERGENCY_REQUESTS_FILE, []))[:50]


def make_real_sensor_waiting_state() -> dict:
    """Create a visible PT-REAL-001 card before the first ESP32 POST arrives."""
    profile = PATIENT_LOOKUP["PT-REAL-001"]
    base = profile.get("base", {})
    inference_vitals = {
        "Heart Rate (bpm)": float(base.get("hr", 80)),
        "SpO2 Level (%)": float(base.get("spo2", 98)),
        "Systolic Blood Pressure (mmHg)": float(base.get("sbp", 120)),
        "Diastolic Blood Pressure (mmHg)": float(base.get("dbp", 80)),
        "Body Temperature (°C)": float(base.get("temp", 37.0)),
    }
    result = run_inference(registry, inference_vitals)
    result["vitals"] = {
        "Heart Rate (bpm)": None,
        "SpO2 Level (%)": None,
        "Systolic Blood Pressure (mmHg)": 120,
        "Diastolic Blood Pressure (mmHg)": 80,
        "Body Temperature (°C)": None,
    }
    result["threshold"]["hr"] = "NORMAL"
    result["threshold"]["spo2"] = "NORMAL"
    result["threshold"]["temp"] = "NORMAL"
    result["ml"]["hr"] = {"prediction": "N/A", "confidence": 0.0}
    result["ml"]["spo2"] = {"prediction": "N/A", "confidence": 0.0}
    result["ml"]["temp"] = {"prediction": "N/A", "confidence": 0.0}
    result.update({
        "patient_id": profile["id"],
        "patient_name": profile["name"],
        "patient_age": profile["age"],
        "ward": profile["ward"],
        "scenario": "real_sensor",
        "respiratory_rate": None,
        "ecg_value": None,
        "piezo_vibration": None,
        "emergency_call": "None",
        "source": "ESP32",
        "bp_mode": "Simulated",
        "system_status": "WAITING_START",
        "wifi_status": "Waiting ESP32",
        "max30102_status": "Waiting",
        "finger_detected": "No",
        "max30102_ir": None,
        "max30102_red": None,
        "ecg_status": "Waiting",
        "temperature_status": "Waiting",
        "piezo_status": "Waiting",
        "buzzer_status": "OFF",
        "monitoring_state": "WAITING_START",
        "current_button": "None",
        "last_button_event": "None",
        "btn_start_raw": None,
        "btn_red_raw": None,
        "btn_blue_raw": None,
        "btn_yellow_raw": None,
        "btn_white_raw": None,
        "btn_start_state": "—",
        "btn_red_state": "—",
        "btn_blue_state": "—",
        "btn_yellow_state": "—",
        "btn_white_state": "—",
        "last_update": None,
        "overall": "NORMAL",
    })
    return result


latest_state["PT-REAL-001"] = make_real_sensor_waiting_state()


async def simulation_loop():
    tick_interval = STREAM_INTERVAL_MS / 1000.0
    batch_ticks = max(1, DISPLAY_INTERVAL_MS // STREAM_INTERVAL_MS)

    while True:
        for _ in range(batch_ticks):
            for p in PATIENT_PROFILES:
                pid = p["id"]

                # Do not generate simulated values for the real ESP32 patient.
                # This patient is updated only through /api/esp32/vitals.
                if p.get("scenario") == "real_sensor":
                    continue

                # Do not overwrite patients currently controlled by real sensor data.
                if manual_overrides.get(pid):
                    continue

                vitals = generator.next(pid)
                result = run_inference(registry, vitals)
                result.update(
                    {
                        "patient_id": pid,
                        "patient_name": p["name"],
                        "patient_age": p["age"],
                        "ward": p["ward"],
                        "scenario": p["scenario"],
                    }
                )

                history = patient_history[pid]
                history.append(
                    {
                        "ts": result["ts"],
                        "hr": vitals["Heart Rate (bpm)"],
                        "spo2": vitals["SpO2 Level (%)"],
                        "sbp": vitals["Systolic Blood Pressure (mmHg)"],
                        "dbp": vitals["Diastolic Blood Pressure (mmHg)"],
                        "temp": vitals["Body Temperature (°C)"],
                        "sev": result["overall"],
                    }
                )

                if len(history) > STORAGE_HISTORY_LIMIT:
                    history.pop(0)

                latest_state[pid] = result

            save_json_file(PATIENT_HISTORY_FILE, patient_history)
            await asyncio.sleep(tick_interval)


app = FastAPI(title="ICU Real-Time Monitor", version="2026.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())
    logger.info("Dashboard started at http://127.0.0.1:8000")


@app.get("/", response_class=HTMLResponse)
async def serve():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/state")
async def get_state():
    payload = []
    for profile in PATIENT_PROFILES:
        pid = profile["id"]
        state = latest_state.get(pid)
        if state:
            entry = dict(state)
            entry["history"] = patient_history.get(pid, [])[-HISTORY_POINTS:]
            payload.append(entry)
    return JSONResponse(payload)


@app.get("/api/history/{patient_id}")
async def get_history(patient_id: str, limit: int = 200):
    records = list(patient_history.get(patient_id, []))[-max(1, min(limit, STORAGE_HISTORY_LIMIT)):]
    return JSONResponse({
        "patient_id": patient_id,
        "total_records": len(patient_history.get(patient_id, [])),
        "records": records,
        "nurse_logs_count": len(nurse_logs),
        "emergency_requests_count": len(emergency_requests),
    })


@app.get("/api/stream")
async def stream():
    async def event_gen() -> AsyncGenerator[str, None]:
        while True:
            payload = []
            for profile in PATIENT_PROFILES:
                pid = profile["id"]
                state = latest_state.get(pid)
                if state:
                    entry = dict(state)
                    entry["history"] = patient_history.get(pid, [])[-HISTORY_POINTS:]
                    payload.append(entry)
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(DISPLAY_INTERVAL_MS / 1000.0)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status")
async def status():
    return {
        "online": True,
        "models_dir": str(registry.models_dir),
        "models": list(registry.artifacts.keys()),
        "patients": len(PATIENT_PROFILES),
        "interval_ms": STREAM_INTERVAL_MS,
        "ts": datetime.utcnow().isoformat(),
    }
@app.post("/api/esp32/vitals")
async def receive_esp32_vitals(payload: RealSensorPayload):
    pid = payload.patient_id
    profile = PATIENT_LOOKUP.get(pid)

    if profile is None:
        return JSONResponse(
            {"ok": False, "error": "Unknown patient_id. Use PT-REAL-001 or PT-001 to PT-006."},
            status_code=400,
        )

    base = profile.get("base", {})

    # Values shown on the dashboard. These can be None, so the dashboard can display "None".
    display_vitals = {
        "Heart Rate (bpm)": round(payload.heart_rate, 1) if payload.heart_rate is not None else None,
        "SpO2 Level (%)": round(payload.spo2, 1) if payload.spo2 is not None else None,
        "Systolic Blood Pressure (mmHg)": round(payload.systolic_bp, 1) if payload.systolic_bp is not None else None,
        "Diastolic Blood Pressure (mmHg)": round(payload.diastolic_bp, 1) if payload.diastolic_bp is not None else None,
        "Body Temperature (°C)": round(payload.temperature, 2) if payload.temperature is not None else None,
    }

    # Values used only for AI inference. The AI models need numbers, so we use safe defaults
    # when a sensor is missing. The dashboard still shows None for missing sensors.
    inference_vitals = {
        "Heart Rate (bpm)": display_vitals["Heart Rate (bpm)"] if display_vitals["Heart Rate (bpm)"] is not None else float(base.get("hr", 80)),
        "SpO2 Level (%)": display_vitals["SpO2 Level (%)"] if display_vitals["SpO2 Level (%)"] is not None else float(base.get("spo2", 98)),
        "Systolic Blood Pressure (mmHg)": display_vitals["Systolic Blood Pressure (mmHg)"] if display_vitals["Systolic Blood Pressure (mmHg)"] is not None else float(base.get("sbp", 120)),
        "Diastolic Blood Pressure (mmHg)": display_vitals["Diastolic Blood Pressure (mmHg)"] if display_vitals["Diastolic Blood Pressure (mmHg)"] is not None else float(base.get("dbp", 80)),
        "Body Temperature (°C)": display_vitals["Body Temperature (°C)"] if display_vitals["Body Temperature (°C)"] is not None else float(base.get("temp", 37.0)),
    }

    result = run_inference(registry, inference_vitals)

    # Replace AI vitals with the real display values, so missing sensors appear as None.
    result["vitals"] = display_vitals

    # Missing sensors should not look abnormal on the dashboard.
    if display_vitals["Heart Rate (bpm)"] is None:
        result["threshold"]["hr"] = "NORMAL"
        result["ml"]["hr"] = {"prediction": "N/A", "confidence": 0.0}
    if display_vitals["SpO2 Level (%)"] is None:
        result["threshold"]["spo2"] = "NORMAL"
        result["ml"]["spo2"] = {"prediction": "N/A", "confidence": 0.0}
    if display_vitals["Body Temperature (°C)"] is None:
        result["threshold"]["temp"] = "NORMAL"
        result["ml"]["temp"] = {"prediction": "N/A", "confidence": 0.0}

    result.update(
        {
            "patient_id": pid,
            "patient_name": profile["name"],
            "patient_age": profile["age"],
            "ward": profile["ward"],
            "scenario": "real_sensor",
            "respiratory_rate": payload.respiratory_rate,
            "ecg_value": payload.ecg_value,
            "emergency_call": payload.emergency_call,
            "piezo_vibration": payload.piezo_vibration,
            "source": payload.source,
            "bp_mode": payload.bp_mode,
            "system_status": payload.system_status,
            "wifi_status": payload.wifi_status,
            "max30102_status": payload.max30102_status,
            "finger_detected": payload.finger_detected,
            "max30102_ir": payload.max30102_ir,
            "max30102_red": payload.max30102_red,
            "ecg_status": payload.ecg_status,
            "temperature_status": payload.temperature_status,
            "piezo_status": payload.piezo_status,
            "buzzer_status": payload.buzzer_status,
            "monitoring_state": payload.monitoring_state,
            "current_button": payload.current_button,
            "last_button_event": payload.last_button_event,
            "btn_start_raw": payload.btn_start_raw,
            "btn_red_raw": payload.btn_red_raw,
            "btn_blue_raw": payload.btn_blue_raw,
            "btn_yellow_raw": payload.btn_yellow_raw,
            "btn_white_raw": payload.btn_white_raw,
            "btn_start_state": payload.btn_start_state,
            "btn_red_state": payload.btn_red_state,
            "btn_blue_state": payload.btn_blue_state,
            "btn_yellow_state": payload.btn_yellow_state,
            "btn_white_state": payload.btn_white_state,
            "last_update": datetime.utcnow().isoformat(),
        }
    )

    active_request = first_patient_button(payload.emergency_call, payload.current_button, payload.last_button_event)

    # If a physical patient button is pressed on the ESP32, show the correct priority.
    # Urgent Help and Breathing Difficulty are critical.
    # Pain and Need Help are warning-level requests.
    if active_request != "None":
        if active_request in ["Urgent Help", "Breathing Difficulty"]:
            result["overall"] = "CRITICAL"
        elif active_request in ["Pain", "Need Help"]:
            if result["overall"] not in ["CRITICAL"]:
                result["overall"] = "WARNING"

        result["esp32_emergency"] = active_request

        emergency_entry = {
            "id": f"EM-{int(time.time() * 1000)}",
            "patient_id": pid,
            "patient_name": profile["name"],
            "ward": profile["ward"],
            "request_type": active_request,
            "details": "Emergency request sent from ESP32 physical button.",
            "created_by": "ESP32 Button",
            "status": "Pending",
            "created_at": datetime.utcnow().isoformat(),
        }

        duplicate = False
        if emergency_requests:
            last = emergency_requests[0]
            try:
                age_seconds = (
                    datetime.utcnow()
                    - datetime.fromisoformat(last.get("created_at", datetime.utcnow().isoformat()))
                ).total_seconds()
            except Exception:
                age_seconds = 999

            duplicate = (
                last.get("patient_id") == pid
                and last.get("request_type") == payload.emergency_call
                and age_seconds < 10
            )

        if not duplicate:
            emergency_requests.insert(0, emergency_entry)
            del emergency_requests[50:]
            save_json_file(EMERGENCY_REQUESTS_FILE, emergency_requests)

    history = patient_history[pid]
    history.append(
        {
            "ts": result["ts"],
            "hr": display_vitals["Heart Rate (bpm)"],
            "spo2": display_vitals["SpO2 Level (%)"],
            "sbp": display_vitals["Systolic Blood Pressure (mmHg)"],
            "dbp": display_vitals["Diastolic Blood Pressure (mmHg)"],
            "temp": display_vitals["Body Temperature (°C)"],
            "rr": payload.respiratory_rate,
            "ecg": payload.ecg_value,
            "emergency": payload.emergency_call,
            "current_button": payload.current_button,
            "last_button_event": payload.last_button_event,
            "monitoring_state": payload.monitoring_state,
            "max30102_ir": payload.max30102_ir,
            "max30102_red": payload.max30102_red,
            "buzzer_status": payload.buzzer_status,
            "btn_start_raw": payload.btn_start_raw,
            "btn_red_raw": payload.btn_red_raw,
            "btn_blue_raw": payload.btn_blue_raw,
            "btn_yellow_raw": payload.btn_yellow_raw,
            "btn_white_raw": payload.btn_white_raw,
            "sev": result["overall"],
        }
    )

    if len(history) > STORAGE_HISTORY_LIMIT:
        history.pop(0)

    latest_state[pid] = result
    manual_overrides[pid] = True
    save_json_file(PATIENT_HISTORY_FILE, patient_history)

    return JSONResponse(
        {
            "ok": True,
            "patient_id": pid,
            "overall": result["overall"],
            "vitals": display_vitals,
            "hardware_status": {
                "finger_detected": payload.finger_detected,
                "ecg_status": payload.ecg_status,
                "temperature_status": payload.temperature_status,
                "piezo_status": payload.piezo_status,
                "buzzer_status": payload.buzzer_status,
                "max30102_ir": payload.max30102_ir,
                "max30102_red": payload.max30102_red,
                "monitoring_state": payload.monitoring_state,
                "current_button": payload.current_button,
                "last_button_event": payload.last_button_event,
                "btn_start_raw": payload.btn_start_raw,
                "btn_red_raw": payload.btn_red_raw,
                "btn_blue_raw": payload.btn_blue_raw,
                "btn_yellow_raw": payload.btn_yellow_raw,
                "btn_white_raw": payload.btn_white_raw,
            },
        }
    )

@app.post("/api/diabetes/predict")
async def diabetes_predict(payload: DiabetesRequest):
    result = predict_diabetes_risk(registry, payload.model_dump())
    return JSONResponse(result)


@app.get("/api/nurse/logs")
async def get_nurse_logs():
    return JSONResponse(nurse_logs[:50])


@app.post("/api/nurse/logs")
async def add_nurse_log(payload: NurseAction):
    patient = PATIENT_LOOKUP.get(payload.patient_id, {"name": payload.patient_id, "ward": "Unknown"})
    entry = {
        "id": f"LOG-{int(time.time() * 1000)}",
        "patient_id": payload.patient_id,
        "patient_name": patient["name"],
        "ward": patient["ward"],
        "action_type": payload.action_type,
        "item_name": payload.item_name,
        "note": payload.note,
        "performed_by": payload.performed_by,
        "created_at": datetime.utcnow().isoformat(),
    }
    nurse_logs.insert(0, entry)
    del nurse_logs[50:]
    save_json_file(NURSE_LOGS_FILE, nurse_logs)
    return JSONResponse({"ok": True, "entry": entry})


@app.get("/api/emergency/requests")
async def get_emergency_requests():
    return JSONResponse(emergency_requests[:50])


@app.post("/api/emergency/requests")
async def add_emergency_request(payload: EmergencyRequest):
    patient = PATIENT_LOOKUP.get(payload.patient_id, {"name": payload.patient_id, "ward": "Unknown"})
    entry = {
        "id": f"EM-{int(time.time() * 1000)}",
        "patient_id": payload.patient_id,
        "patient_name": patient["name"],
        "ward": patient["ward"],
        "request_type": payload.request_type,
        "details": payload.details,
        "created_by": payload.created_by,
        "status": "Pending",
        "created_at": datetime.utcnow().isoformat(),
    }
    emergency_requests.insert(0, entry)
    del emergency_requests[50:]
    save_json_file(EMERGENCY_REQUESTS_FILE, emergency_requests)
    return JSONResponse({"ok": True, "entry": entry})


@app.post("/api/emergency/requests/{request_id}/status")
async def update_emergency_request(request_id: str, payload: EmergencyUpdate):
    for entry in emergency_requests:
        if entry["id"] == request_id:
            entry["status"] = payload.status
            entry["updated_at"] = datetime.utcnow().isoformat()
            save_json_file(EMERGENCY_REQUESTS_FILE, emergency_requests)
            return JSONResponse({"ok": True, "entry": entry})
    return JSONResponse({"ok": False, "error": "Request not found"}, status_code=404)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICU Real-Time Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#03080f;--surface:#050e1c;--card:#071220;--border:#0b1f3a;--border2:#133058;
  --accent:#00c8ff;--accent2:#0055d4;--green:#00e676;--green-d:rgba(0,230,118,.12);
  --yellow:#ffe000;--yellow-d:rgba(255,224,0,.1);--orange:#ff7700;--orange-d:rgba(255,119,0,.1);
  --red:#ff1744;--red-d:rgba(255,23,68,.12);--text:#b8d4f0;--text2:#5a7fa8;
  --mono:'JetBrains Mono',monospace;--sans:'Rajdhani',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;display:flex;flex-direction:column;overflow:hidden}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;background-image:linear-gradient(rgba(0,80,180,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(0,80,180,.035) 1px,transparent 1px);background-size:32px 32px;animation:bg 30s linear infinite}
body::after{content:'';position:fixed;top:-250px;left:-250px;width:700px;height:700px;background:radial-gradient(circle,rgba(0,85,212,.1) 0%,transparent 70%);pointer-events:none;z-index:0}
@keyframes bg{to{background-position:32px 32px,32px 32px}}

#topbar{position:sticky;top:0;z-index:50;background:rgba(3,8,15,.92);border-bottom:1px solid var(--border);backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:space-between;padding:10px 20px;gap:16px;flex-shrink:0}
.top-logo{display:flex;align-items:center;gap:10px}.top-logo .icon{width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,var(--accent2),var(--accent));display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px rgba(0,136,255,.35)}
.top-logo h1{font-size:18px;font-weight:700;letter-spacing:.5px;color:#fff}.top-meta{font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:1px}
.topbar-right{display:flex;align-items:center;gap:16px}.live-badge{display:flex;align-items:center;gap:7px;padding:5px 12px;border-radius:20px;border:1px solid rgba(0,230,118,.3);background:rgba(0,230,118,.06);font-family:var(--mono);font-size:10px;color:var(--green)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 1.2s ease-in-out infinite}.clock{font-family:var(--mono);font-size:13px;color:var(--accent);letter-spacing:.5px}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,230,118,.5)}50%{opacity:.6;box-shadow:0 0 0 5px rgba(0,230,118,0)}}

.nav-tabs{display:flex;gap:10px;flex-wrap:wrap}.tab-btn{background:var(--surface);border:1px solid var(--border2);color:var(--text2);padding:8px 16px;cursor:pointer;border-radius:4px;font-family:var(--mono);font-size:11px}
.tab-btn.active{background:var(--accent2);color:#fff;border-color:var(--accent)}

#alertBar{display:none;position:sticky;top:57px;z-index:40;background:rgba(255,23,68,.12);border-bottom:1px solid rgba(255,23,68,.4);backdrop-filter:blur(8px);padding:7px 20px;font-family:var(--mono);font-size:11px;color:var(--red);animation:flashBar 1s ease-in-out infinite}
@keyframes flashBar{0%,100%{background:rgba(255,23,68,.12)}50%{background:rgba(255,23,68,.22)}}

#app{flex:1;overflow-y:auto;overflow-x:hidden;padding:0 16px 20px;position:relative;z-index:1}
#app::-webkit-scrollbar{width:6px}#app::-webkit-scrollbar-track{background:var(--surface)}#app::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
.page-section{display:none;animation:fadeIn .3s ease-in}.page-section.active{display:block}@keyframes fadeIn{from{opacity:0}to{opacity:1}}

#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding-top:16px}
@media(max-width:1100px){#grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:680px){#grid{grid-template-columns:1fr}}

.pcard{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;transition:border-color .3s,box-shadow .3s;position:relative}
.pcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent2),var(--accent),transparent);opacity:.5}
.pcard.sev-CAUTION{border-color:rgba(255,224,0,.3)}.pcard.sev-WARNING{border-color:rgba(255,119,0,.4)}.pcard.sev-CRITICAL{border-color:rgba(255,23,68,.6);box-shadow:0 0 20px rgba(255,23,68,.15),inset 0 0 20px rgba(255,23,68,.03);animation:cardAlarm 1.5s ease-in-out infinite}
@keyframes cardAlarm{0%,100%{box-shadow:0 0 20px rgba(255,23,68,.15)}50%{box-shadow:0 0 35px rgba(255,23,68,.35)}}
.card-header{padding:12px 14px 10px;background:rgba(0,0,0,.25);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:8px}.patient-info{flex:1;min-width:0}.p-name{font-size:15px;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.p-meta{font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:2px}
.risk-pill{flex-shrink:0;font-family:var(--mono);font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:.8px;transition:all .3s;white-space:nowrap}
.risk-pill.NORMAL{background:var(--green-d);color:var(--green);border:1px solid rgba(0,230,118,.3)}.risk-pill.CAUTION{background:var(--yellow-d);color:var(--yellow);border:1px solid rgba(255,224,0,.3)}.risk-pill.WARNING{background:var(--orange-d);color:var(--orange);border:1px solid rgba(255,119,0,.35)}.risk-pill.CRITICAL{background:var(--red-d);color:var(--red);border:1px solid rgba(255,23,68,.4);animation:pillFlash 1s ease-in-out infinite}
@keyframes pillFlash{0%,100%{opacity:1}50%{opacity:.6}}

.vitals-row{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
.v-cell{background:var(--card);padding:9px 8px;text-align:center;transition:background .3s}.v-cell.sev-CAUTION{background:rgba(255,224,0,.07)}.v-cell.sev-WARNING{background:rgba(255,119,0,.09)}.v-cell.sev-CRITICAL{background:rgba(255,23,68,.1);animation:cellFlash 1.2s ease-in-out infinite}
@keyframes cellFlash{0%,100%{background:rgba(255,23,68,.1)}50%{background:rgba(255,23,68,.2)}}
.v-icon{font-size:14px;display:block;margin-bottom:3px}.v-label{font-family:var(--mono);font-size:8px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;display:block}.v-val{font-family:var(--mono);font-size:16px;font-weight:700;color:#fff;display:block;line-height:1.1;margin:2px 0}.v-unit{font-size:8px;color:var(--text2);display:block}
.v-sev{font-size:8px;font-family:var(--mono);font-weight:700;letter-spacing:.5px;display:inline-block;padding:1px 5px;border-radius:3px;margin-top:3px}.v-sev.NORMAL{color:var(--green);background:rgba(0,230,118,.1)}.v-sev.CAUTION{color:var(--yellow);background:rgba(255,224,0,.1)}.v-sev.WARNING{color:var(--orange);background:rgba(255,119,0,.1)}.v-sev.CRITICAL{color:var(--red);background:rgba(255,23,68,.15)}

.disease-strip{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;border-bottom:1px solid var(--border);gap:8px}.dis-label{font-family:var(--mono);font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:1px}.dis-name{font-family:var(--mono);font-size:12px;font-weight:700}.dis-conf{font-family:var(--mono);font-size:10px;color:var(--text2)}
.ml-row{display:flex;align-items:center;justify-content:space-around;padding:7px 10px;border-bottom:1px solid var(--border);gap:4px;flex-wrap:wrap}.ml-chip{font-family:var(--mono);font-size:8px;font-weight:700;padding:3px 7px;border-radius:4px;letter-spacing:.5px;white-space:nowrap}.ml-chip.ABN{background:rgba(255,23,68,.15);color:var(--red);border:1px solid rgba(255,23,68,.3)}.ml-chip.NRM{background:rgba(0,230,118,.08);color:var(--green);border:1px solid rgba(0,230,118,.2)}.ml-chip.NA{background:rgba(90,127,168,.1);color:var(--text2);border:1px solid var(--border2)}
.wave-wrap{padding:8px 10px 10px;position:relative}.wave-label{font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:5px;text-transform:uppercase;letter-spacing:1px}.wave-canvas{width:100%;height:52px;display:block;border-radius:6px;background:rgba(0,0,0,.3)}
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border)}.stat-cell{background:var(--card);padding:8px 10px;text-align:center}.stat-label{font-family:var(--mono);font-size:8px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;display:block}.stat-val{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--accent);display:block;margin-top:2px}

.panel-wrap{max-width:1200px;margin:18px auto 0;display:grid;gap:18px}
.section-card{background:var(--card);border:1px solid var(--border2);border-radius:14px;padding:18px}.section-title{font-size:24px;font-weight:700;color:#fff;margin-bottom:6px}.section-sub{font-family:var(--mono);font-size:11px;color:var(--text2);margin-bottom:14px}
.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}@media(max-width:800px){.form-grid{grid-template-columns:1fr}}
.input-group{display:flex;flex-direction:column;gap:6px}.input-group label{font-family:var(--mono);font-size:11px;color:var(--text2)}
.input-group input,.input-group select,.input-group textarea{background:var(--surface);color:var(--text);border:1px solid var(--border2);border-radius:8px;padding:10px 12px;font-family:var(--sans);font-size:14px}
.input-group textarea{min-height:110px;resize:vertical}.btn{background:linear-gradient(135deg,var(--accent2),var(--accent));color:#fff;border:none;border-radius:10px;padding:11px 18px;font-family:var(--mono);font-size:11px;cursor:pointer}.btn.secondary{background:var(--surface);border:1px solid var(--border2);color:var(--text)}
.log-list,.request-list{display:grid;gap:10px;margin-top:12px}.log-item,.request-item,.summary-box{background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:12px;padding:12px}.muted{font-family:var(--mono);font-size:11px;color:var(--text2)}
.request-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}.tag{display:inline-block;padding:3px 8px;border-radius:20px;font-family:var(--mono);font-size:10px}.tag.pending{background:rgba(255,119,0,.12);color:var(--orange);border:1px solid rgba(255,119,0,.35)}.tag.ack{background:rgba(255,224,0,.1);color:var(--yellow);border:1px solid rgba(255,224,0,.3)}.tag.done{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.3)}
.request-buttons{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}@media(max-width:800px){.request-buttons{grid-template-columns:repeat(2,1fr)}}
.quick-btn{background:rgba(0,0,0,.2);border:1px solid var(--border2);border-radius:12px;padding:16px;cursor:pointer;color:#fff;text-align:center;font-family:var(--mono);font-size:12px}.quick-btn:hover{border-color:var(--accent);box-shadow:0 0 0 1px rgba(0,200,255,.2) inset}
.ai-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}@media(max-width:1000px){.ai-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.ai-grid{grid-template-columns:1fr}}
.ai-table{width:100%;border-collapse:collapse}.ai-table th,.ai-table td{border-bottom:1px solid var(--border);padding:10px 8px;text-align:left}.ai-table th{font-family:var(--mono);font-size:11px;color:var(--text2)}
#alertQueue{position:fixed;bottom:20px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:100;pointer-events:none;max-width:280px}.alert-notif{padding:10px 14px;border-radius:10px;border-left:3px solid;background:rgba(3,8,15,.95);backdrop-filter:blur(12px);font-family:var(--mono);font-size:10px;pointer-events:all;animation:slideIn .3s ease}.alert-notif.CRITICAL{border-color:var(--red);color:var(--red)}.alert-notif.WARNING{border-color:var(--orange);color:var(--orange)}.alert-notif .n-title{font-weight:700;font-size:11px;margin-bottom:3px}.alert-notif .n-body{color:var(--text2)}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}.fade-out{animation:fadeOut .5s ease forwards}@keyframes fadeOut{to{opacity:0;transform:translateX(30px)}}
</style>
</head>
<body>
<div id="topbar">
  <div class="top-logo">
    <div class="icon">🏥</div>
    <div>
      <h1>Amar Telidji University-Laghouat</h1>
      <div class="top-meta">AI-ASSISTED VITAL SIGNS SURVEILLANCE · v2026.2</div>
    </div>
  </div>
  <div class="nav-tabs">
    <button class="tab-btn active" data-tab="dashboard" onclick="switchTab('dashboard', this)">MONITORING</button>
    <button class="tab-btn" data-tab="nurse" onclick="switchTab('nurse', this)">NURSE STATION</button>
    <button class="tab-btn" data-tab="ai" onclick="switchTab('ai', this)">AI ANALYTICS</button>
    <button class="tab-btn" data-tab="diabetes" onclick="switchTab('diabetes', this)">DIABETES</button>
    <button class="tab-btn" data-tab="emergency" onclick="switchTab('emergency', this)">EMERGENCY</button>
    <button class="tab-btn" data-tab="historique" onclick="switchTab('historique', this)">STORAGE HISTORIQUE</button>
  </div>
  <div class="topbar-right">
    <div class="live-badge"><div class="live-dot"></div>LIVE</div>
    <div class="clock" id="clock">--:--:--</div>
  </div>
</div>
<div id="alertBar">🚨 &nbsp;<span id="alertBarText"></span></div>
<div id="app">
  <section id="dashboard-section" class="page-section active">
    <div id="grid"></div>
  </section>

  <section id="nurse-section" class="page-section">
    <div class="panel-wrap">
      <div class="section-card">
        <div class="section-title">Nurse Dashboard</div>
        <div class="section-sub">Record injections, medications, observations, and bedside actions.</div>
        <form id="nurse-form" class="form-grid">
          <div class="input-group">
            <label>Patient</label>
            <select id="nurse-patient" required></select>
          </div>
          <div class="input-group">
            <label>Action Type</label>
            <select id="nurse-action" required>
              <option value="Medication">Medication</option>
              <option value="Injection">Injection</option>
              <option value="IV Change">IV Change</option>
              <option value="Observation">Observation</option>
              <option value="Procedure">Procedure</option>
            </select>
          </div>
          <div class="input-group">
            <label>Drug / Item / Action Name</label>
            <input id="nurse-item" placeholder="Example: Ceftriaxone 1g">
          </div>
          <div class="input-group">
            <label>Performed By</label>
            <input id="nurse-by" value="Nurse 1">
          </div>
          <div class="input-group" style="grid-column:1/-1">
            <label>Clinical Note</label>
            <textarea id="nurse-note" placeholder="Example: IV antibiotic given at 08:30. Patient tolerated well."></textarea>
          </div>
          <div><button class="btn" type="submit">SAVE NURSE ACTION</button></div>
        </form>
      </div>
      <div class="section-card">
        <div class="section-title" style="font-size:20px">Recent Nurse Actions</div>
        <div class="log-list" id="nurse-log-list"></div>
      </div>
    </div>
  </section>

  <section id="ai-section" class="page-section">
    <div class="panel-wrap">
      <div class="section-card">
        <div class="section-title">AI Analytics</div>
        <div class="section-sub">Real-time AI summary of risk levels, disease suggestion, and priority patients.</div>
        <div class="ai-grid">
          <div class="summary-box"><div class="muted">Critical Patients</div><div id="ai-critical" style="font-size:28px;color:var(--red);font-weight:700">0</div></div>
          <div class="summary-box"><div class="muted">Warning Patients</div><div id="ai-warning" style="font-size:28px;color:var(--orange);font-weight:700">0</div></div>
          <div class="summary-box"><div class="muted">Normal / Stable</div><div id="ai-normal" style="font-size:28px;color:var(--green);font-weight:700">0</div></div>
          <div class="summary-box"><div class="muted">Top Predicted Disease</div><div id="ai-top-disease" style="font-size:20px;color:var(--accent);font-weight:700">—</div></div>
        </div>
        <table class="ai-table">
          <thead><tr><th>Patient</th><th>Risk</th><th>AI Disease</th><th>Confidence</th><th>Ward</th></tr></thead>
          <tbody id="ai-table-body"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="diabetes-section" class="page-section">
    <div class="panel-wrap">
      <div class="section-card">
        <div class="section-title">Diabetes Risk Section</div>
        <div class="section-sub">This keeps your diabetes risk model and also estimates type distribution percentages from the clinical profile.</div>
        <form id="diabetes-form" class="form-grid">
          <div class="input-group"><label>Pregnancies</label><input id="db-pregnancies" type="number" step="1" value="1"></div>
          <div class="input-group"><label>Glucose</label><input id="db-glucose" type="number" step="0.1" value="140"></div>
          <div class="input-group"><label>Blood Pressure</label><input id="db-blood-pressure" type="number" step="0.1" value="70"></div>
          <div class="input-group"><label>Skin Thickness</label><input id="db-skin-thickness" type="number" step="0.1" value="20"></div>
          <div class="input-group"><label>Insulin</label><input id="db-insulin" type="number" step="0.1" value="80"></div>
          <div class="input-group"><label>BMI</label><input id="db-bmi" type="number" step="0.1" value="28.5"></div>
          <div class="input-group"><label>Diabetes Pedigree Function</label><input id="db-dpf" type="number" step="0.001" value="0.45"></div>
          <div class="input-group"><label>Age</label><input id="db-age" type="number" step="1" value="40"></div>
          <div><button class="btn" type="submit">PREDICT DIABETES RISK</button></div>
        </form>
      </div>
      <div class="section-card">
        <div class="section-title" style="font-size:20px">Diabetes Prediction Result</div>
        <div class="summary-box">
          <div class="muted">Prediction</div>
          <div id="diabetes-prediction" style="font-size:24px;font-weight:700;color:var(--accent)">—</div>
          <div class="muted" style="margin-top:10px">Confidence</div>
          <div id="diabetes-confidence" style="font-size:18px;color:#fff;font-weight:700">0%</div>
          <div class="muted" style="margin-top:10px">Estimated Type</div>
          <div id="diabetes-type" style="font-size:18px;color:#fff;font-weight:700">—</div>
          <div class="muted" style="margin-top:10px">Type Percentages</div>
          <div id="diabetes-type-percentages" style="font-size:14px;color:var(--text)">Type 1: 0% · Type 2: 0%</div>
          <div class="muted" id="diabetes-note" style="margin-top:12px">Note: this section predicts diabetes risk and estimates type percentages from the clinical profile.</div>
        </div>
      </div>
    </div>
  </section>

  <section id="emergency-section" class="page-section">
    <div class="panel-wrap">
      <div class="section-card">
        <div class="section-title">Emergency / Patient Call Section</div>
        <div class="section-sub">This is the button area you asked for. A patient can request help, and it appears instantly for nurses.</div>
        <div class="form-grid">
          <div class="input-group">
            <label>Patient</label>
            <select id="emergency-patient"></select>
          </div>
          <div class="input-group">
            <label>Extra Details</label>
            <input id="emergency-details" placeholder="Example: severe pain or cannot breathe well">
          </div>
        </div>
        <div class="request-buttons">
          <button class="quick-btn" onclick="sendEmergency('Need Help')">🆘 Need Help</button>
          <button class="quick-btn" onclick="sendEmergency('Pain')">⚡ Pain</button>
          <button class="quick-btn" onclick="sendEmergency('Breathing Difficulty')">🫁 Breathing Difficulty</button>
          <button class="quick-btn" onclick="sendEmergency('Urgent Help')">🚨 Urgent Help</button>
        </div>
      </div>
      <div class="section-card">
        <div class="section-title" style="font-size:20px">Emergency Queue</div>
        <div class="request-list" id="emergency-list"></div>
      </div>
    </div>
  </section>

  <section id="historique-section" class="page-section">
    <div class="panel-wrap">
      <div class="section-card">
        <div class="section-title">Storage Historique</div>
        <div class="section-sub">Stored vital-sign history for each patient, plus current storage counters.</div>
        <div class="form-grid">
          <div class="input-group">
            <label>Patient</label>
            <select id="history-patient" onchange="loadHistorique()"></select>
          </div>
          <div class="input-group">
            <label>Stored summary</label>
            <div class="summary-box">
              <div class="muted">Readings: <span id="history-total">0</span></div>
              <div class="muted">Nurse logs: <span id="history-nurse-count">0</span></div>
              <div class="muted">Emergency requests: <span id="history-emergency-count">0</span></div>
            </div>
          </div>
        </div>
        <table class="ai-table" style="margin-top:14px">
          <thead><tr><th>Timestamp</th><th>HR</th><th>SpO₂</th><th>SBP</th><th>DBP</th><th>Temp</th><th>Button</th><th>Mode</th><th>Severity</th></tr></thead>
          <tbody id="history-table-body"></tbody>
        </table>
      </div>
    </div>
  </section>
</div>
<div id="alertQueue"></div>
<script>
const HISTORY = 60;
const COLORS = {hr:'#ff4f80', spo2:'#00c8ff'};
const DIS_COLOR = {'Healthy':'#00e676','Heart Disease':'#ff1744','Hypertension':'#ff7700','Diabetes Mellitus':'#ffe000','Asthma':'#00b0ff','Unknown':'#5a7fa8','N/A':'#5a7fa8','ERR':'#ff1744'};
const DIS_ICON = {'Healthy':'💚','Heart Disease':'💔','Hypertension':'🔴','Diabetes Mellitus':'🟡','Asthma':'🫧','Unknown':'❓','N/A':'❓','ERR':'⚠️'};
const SEV_ORDER = ['NORMAL','CAUTION','WARNING','CRITICAL'];
let patients = {};
let prevSeverity = {};
let nurseLogs = [];
let emergencyRequests = [];
let notifCount = 0;

function tickClock(){ document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-GB', {hour12:false}); }
setInterval(tickClock, 1000); tickClock();

function switchTab(tabName, button){
  document.querySelectorAll('.page-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const section = document.getElementById(tabName + '-section');
  if(section){ section.classList.add('active'); }
  if(button){ button.classList.add('active'); }
  if(tabName === 'nurse'){ renderNurseLogs(); }
  if(tabName === 'ai'){ renderAI(); }
  if(tabName === 'emergency'){ renderEmergencyRequests(); }
  if(tabName === 'historique'){ loadHistorique(); }
}

function populatePatientSelects(){
  const options = Object.values(patients).sort((a,b) => a.patient_id.localeCompare(b.patient_id)).map(p => `<option value="${p.patient_id}">${p.patient_id} - ${p.patient_name}</option>`).join('');
  ['nurse-patient', 'emergency-patient', 'history-patient'].forEach(id => {
    const el = document.getElementById(id);
    if(el){
      const current = el.value;
      el.innerHTML = options;
      if(current && [...el.options].some(opt => opt.value === current)){ el.value = current; }
    }
  });
}

function updatePatient(p){
  patients[p.patient_id] = p;
  populatePatientSelects();
  const el = document.getElementById('card-' + p.patient_id);
  if(!el){ createCard(p); } else { refreshCard(el, p); }
  const prev = prevSeverity[p.patient_id] || 'NORMAL';
  const curr = p.overall;
  if(SEV_ORDER.indexOf(curr) > SEV_ORDER.indexOf(prev)){ pushNotification(p, curr); }
  prevSeverity[p.patient_id] = curr;
}

function createCard(p){
  const grid = document.getElementById('grid');
  const div = document.createElement('div');
  div.id = 'card-' + p.patient_id;
  div.className = 'pcard sev-' + p.overall;
  div.innerHTML = buildCardHTML(p);
  grid.appendChild(div);
  setTimeout(() => drawWave(div, p), 50);
}

function refreshCard(el, p){
  el.className = 'pcard sev-' + p.overall;
  el.innerHTML = buildCardHTML(p);
  drawWave(el, p);
}

function buildCardHTML(p){
  const v = p.vitals || {};
  const t = p.threshold || {};
  const ml = p.ml || {};
  const d = p.derived || {};
  const btnCurrent = p.current_button || 'None';
  const btnLast = p.last_button_event || 'None';
  const btnAlarm = (p.emergency_call && p.emergency_call !== 'None') ? p.emergency_call : (btnCurrent !== 'None' ? btnCurrent : btnLast);
  const btnActive = ['Urgent Help','Breathing Difficulty','Pain','Need Help'].includes(btnAlarm);
  const btnColor = btnActive ? 'var(--red)' : 'var(--green)';
  const dis = ml.disease || {};
  const disName = dis.prediction || '—';
  const disConf = dis.confidence || 0;
  const disColor = DIS_COLOR[disName] || DIS_COLOR['Unknown'];
  const disIcon = DIS_ICON[disName] || '❓';
  function mlChip(pred, label){
    pred = pred || {prediction:'N/A'};
    const cls = pred.prediction === 'ABNORMAL' ? 'ABN' : pred.prediction === 'NORMAL' ? 'NRM' : 'NA';
    return `<span class="ml-chip ${cls}">${label}: ${pred.prediction}</span>`;
  }
  return `
  <div class="card-header">
    <div class="patient-info">
      <div class="p-name">${p.patient_name}</div>
      <div class="p-meta">${p.patient_id} · Age ${p.patient_age} · ${p.ward} · <em>${p.scenario}</em></div>
    </div>
    <div class="risk-pill ${p.overall}">${riskIcon(p.overall)} ${p.overall}</div>
  </div>
  <div class="vitals-row">
    ${vCell('❤️','HR', v['Heart Rate (bpm)'], 'bpm', t.hr || 'NORMAL')}
    ${vCell('🫁','SpO₂', v['SpO2 Level (%)'], '%', t.spo2 || 'NORMAL')}
    ${vCell('🩸','SBP', v['Systolic Blood Pressure (mmHg)'], 'mmHg', t.bp || 'NORMAL')}
    ${vCell('🩸','DBP', v['Diastolic Blood Pressure (mmHg)'], 'mmHg', t.bp || 'NORMAL')}
    ${vCell('🌡️','TEMP', v['Body Temperature (°C)'], '°C', t.temp || 'NORMAL')}
  </div>
  <div class="disease-strip">
    <div><span class="dis-label">AI Diagnosis</span> <span class="dis-name" style="color:${disColor}">${disIcon} ${disName}</span></div>
    <span class="dis-conf">Conf: ${disConf}%</span>
  </div>
  <div class="ml-row">
    ${mlChip(ml.hr,'HR')}
    ${mlChip(ml.spo2,'SpO₂')}
    ${mlChip(ml.bp,'BP')}
    ${mlChip(ml.temp,'Temp')}
  </div>
  <div class="ml-row">
    <span class="ml-chip NA">Source: ${p.source || 'Simulation'}</span>
    <span class="ml-chip NA">Mode: ${p.monitoring_state || '—'}</span>
    <span class="ml-chip ${btnActive ? 'ABN' : 'NRM'}">Alarm: ${btnAlarm}</span>
    <span class="ml-chip NA">Current: ${btnCurrent}</span>
    <span class="ml-chip NA">Last: ${btnLast}</span>
    <span class="ml-chip NA">Finger: ${p.finger_detected || '—'}</span>
    <span class="ml-chip NA">IR: ${p.max30102_ir ?? '—'}</span>
    <span class="ml-chip NA">Red: ${p.max30102_red ?? '—'}</span>
    <span class="ml-chip NA">ECG: ${p.ecg_status || '—'}</span>
    <span class="ml-chip NA">Temp: ${p.temperature_status || '—'}</span>
    <span class="ml-chip NA">Piezo: ${p.piezo_status || '—'}</span>
    <span class="ml-chip ${p.buzzer_status === 'ON' ? 'ABN' : 'NA'}">Buzzer: ${p.buzzer_status || '—'}</span>
  </div>
  <div class="disease-strip">
    <div><span class="dis-label">ESP32 Button Event</span> <span class="dis-name" style="color:${btnColor}">${btnAlarm}</span></div>
    <span class="dis-conf">Current: ${btnCurrent} · Last: ${btnLast}</span>
  </div>
  <div class="wave-wrap">
    <div class="wave-label">❤ Heart Rate Waveform</div>
    <canvas class="wave-canvas" id="wave-hr-${p.patient_id}"></canvas>
    <div class="wave-label" style="margin-top:6px">🫁 SpO₂ Waveform</div>
    <canvas class="wave-canvas" id="wave-spo2-${p.patient_id}"></canvas>
  </div>
  <div class="stats-row">
    <div class="stat-cell"><span class="stat-label">Pulse Press.</span><span class="stat-val">${d.pp ?? '—'} mmHg</span></div>
    <div class="stat-cell"><span class="stat-label">MAP</span><span class="stat-val">${d.map ?? '—'} mmHg</span></div>
    <div class="stat-cell"><span class="stat-label">Shock Index</span><span class="stat-val">${d.si ?? '—'}</span></div>
  </div>

  <div class="stats-row">
    <div class="stat-cell"><span class="stat-label">Resp. Rate</span><span class="stat-val">${p.respiratory_rate ?? '—'} /min</span></div>
    <div class="stat-cell"><span class="stat-label">ECG Raw</span><span class="stat-val">${p.ecg_value ?? '—'}</span></div>
    <div class="stat-cell"><span class="stat-label">Piezo</span><span class="stat-val">${p.piezo_vibration ?? '—'}</span></div>
  </div>

  <div class="stats-row">
    <div class="stat-cell"><span class="stat-label">START</span><span class="stat-val">${p.btn_start_raw ?? '—'} / ${p.btn_start_state || '—'}</span></div>
    <div class="stat-cell"><span class="stat-label">RED</span><span class="stat-val">${p.btn_red_raw ?? '—'} / ${p.btn_red_state || '—'}</span></div>
    <div class="stat-cell"><span class="stat-label">BLUE</span><span class="stat-val">${p.btn_blue_raw ?? '—'} / ${p.btn_blue_state || '—'}</span></div>
  </div>
  <div class="stats-row">
    <div class="stat-cell"><span class="stat-label">YELLOW</span><span class="stat-val">${p.btn_yellow_raw ?? '—'} / ${p.btn_yellow_state || '—'}</span></div>
    <div class="stat-cell"><span class="stat-label">WHITE</span><span class="stat-val">${p.btn_white_raw ?? '—'} / ${p.btn_white_state || '—'}</span></div>
    <div class="stat-cell"><span class="stat-label">Last update</span><span class="stat-val">${p.last_update ? new Date(p.last_update).toLocaleTimeString('en-GB',{hour12:false}) : '—'}</span></div>
  </div>

  <div class="disease-strip">
    <div>
      <span class="dis-label">ESP32 Source</span>
      <span class="dis-name" style="color:var(--accent)">${p.source ?? 'Simulation'}</span>
    </div>
    <span class="dis-conf">Button: ${btnAlarm}</span>
  </div>`;
}

function vCell(icon, label, val, unit, sev){
  const display = typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(1)) : (val === null || val === undefined ? 'None' : val);
  const unitDisplay = (val === null || val === undefined) ? '' : unit;
  return `<div class="v-cell sev-${sev}"><span class="v-icon">${icon}</span><span class="v-label">${label}</span><span class="v-val">${display}</span><span class="v-unit">${unitDisplay}</span><span class="v-sev ${sev}">${sev}</span></div>`;
}
function riskIcon(sev){ return {NORMAL:'✅', CAUTION:'⚠️', WARNING:'🔶', CRITICAL:'🚨'}[sev] || ''; }

function drawWave(cardEl, p){
  const hist = p.history || [];
  drawSingleWave(cardEl, `wave-hr-${p.patient_id}`, hist.map(h => h.hr), COLORS.hr, 40, 200);
  drawSingleWave(cardEl, `wave-spo2-${p.patient_id}`, hist.map(h => h.spo2), COLORS.spo2, 70, 100);
}

function drawSingleWave(cardEl, id, data, color, yMin, yMax){
  data = (data || []).filter(v => typeof v === 'number' && isFinite(v));
  const canvas = cardEl.querySelector('#' + id);
  if(!canvas || !data || data.length < 2) return;
  const W = canvas.offsetWidth || 280;
  const H = canvas.offsetHeight || 52;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1;
  for(let y=0; y<=H; y+=H/4){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
  const grad = ctx.createLinearGradient(0,0,0,H); grad.addColorStop(0, color + '55'); grad.addColorStop(1, color + '00');
  const xStep = W / (HISTORY - 1); const yRange = yMax - yMin || 1;
  ctx.beginPath();
  data.forEach((v,i) => { const x=i*xStep; const y=H-((v-yMin)/yRange)*H*.85-H*.075; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.lineTo((data.length-1)*xStep, H); ctx.lineTo(0, H); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  data.forEach((v,i) => { const x=i*xStep; const y=H-((v-yMin)/yRange)*H*.85-H*.075; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.shadowColor = color; ctx.shadowBlur = 6; ctx.stroke(); ctx.shadowBlur = 0;
  const lv = data[data.length-1]; const lx = (data.length-1)*xStep; const ly = H-((lv-yMin)/yRange)*H*.85-H*.075;
  ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, Math.PI*2); ctx.fillStyle = '#fff'; ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fill(); ctx.shadowBlur = 0;
}

function updateAlertBar(){
  const criticals = Object.values(patients).filter(p => p.overall === 'CRITICAL').map(p => `${p.patient_id} (${p.patient_name})`);
  const warnings = Object.values(patients).filter(p => p.overall === 'WARNING').map(p => p.patient_id);
  const bar = document.getElementById('alertBar');
  if(criticals.length > 0){
    bar.style.display = 'block'; bar.style.background = 'rgba(255,23,68,.12)'; bar.style.borderBottomColor = 'rgba(255,23,68,.4)'; bar.style.color = 'var(--red)'; bar.style.animation = 'flashBar 1s ease-in-out infinite';
    document.getElementById('alertBarText').textContent = `CRITICAL ALERT — Patients requiring immediate attention: ${criticals.join(' · ')}`;
  } else if(warnings.length > 0){
    bar.style.display = 'block'; bar.style.background = 'rgba(255,119,0,.1)'; bar.style.borderBottomColor = 'rgba(255,119,0,.4)'; bar.style.color = 'var(--orange)'; bar.style.animation = 'none';
    document.getElementById('alertBarText').textContent = `WARNING — Patients with abnormal vitals: ${warnings.join(' · ')}`;
  } else { bar.style.display = 'none'; }
}

function pushNotification(p, severity){
  if(severity === 'NORMAL') return;
  const queue = document.getElementById('alertQueue');
  const div = document.createElement('div');
  div.className = `alert-notif ${severity}`;
  const v = p.vitals || {};
  div.innerHTML = `<div class="n-title">${riskIcon(severity)} ${severity} — ${p.patient_id}</div><div class="n-body">${p.patient_name} · ${p.ward}<br>HR: ${(v['Heart Rate (bpm)'] || 0).toFixed(0)} · SpO₂: ${(v['SpO2 Level (%)'] || 0).toFixed(1)}%<br>BP: ${(v['Systolic Blood Pressure (mmHg)'] || 0).toFixed(0)}/${(v['Diastolic Blood Pressure (mmHg)'] || 0).toFixed(0)}</div>`;
  queue.appendChild(div);
  setTimeout(() => { div.classList.add('fade-out'); setTimeout(() => div.remove(), 500); }, 5000);
  while(queue.children.length > 4){ queue.removeChild(queue.firstChild); }
}

async function loadNurseLogs(){
  const res = await fetch('/api/nurse/logs');
  nurseLogs = await res.json();
  renderNurseLogs();
}

function renderNurseLogs(){
  const box = document.getElementById('nurse-log-list');
  if(!box) return;
  if(nurseLogs.length === 0){ box.innerHTML = '<div class="log-item"><div class="muted">No nurse actions recorded yet.</div></div>'; return; }
  box.innerHTML = nurseLogs.map(item => `
    <div class="log-item">
      <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><strong>${item.patient_id} · ${item.patient_name}</strong><span class="muted">${formatTime(item.created_at)}</span></div>
      <div style="margin-top:6px"><span class="tag ack">${item.action_type}</span> <strong>${escapeHtml(item.item_name || 'General action')}</strong></div>
      <div class="muted" style="margin-top:6px">${escapeHtml(item.note || 'No note')}</div>
      <div class="muted" style="margin-top:6px">By: ${escapeHtml(item.performed_by)} · ${item.ward}</div>
    </div>`).join('');
}

async function loadEmergencyRequests(){
  const res = await fetch('/api/emergency/requests');
  emergencyRequests = await res.json();
  renderEmergencyRequests();
}

function statusClass(status){
  if(status === 'Pending') return 'pending';
  if(status === 'Acknowledged') return 'ack';
  return 'done';
}

function renderEmergencyRequests(){
  const box = document.getElementById('emergency-list');
  if(!box) return;
  if(emergencyRequests.length === 0){ box.innerHTML = '<div class="request-item"><div class="muted">No emergency requests yet.</div></div>'; return; }
  box.innerHTML = emergencyRequests.map(item => `
    <div class="request-item">
      <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><strong>${item.patient_id} · ${item.patient_name}</strong><span class="tag ${statusClass(item.status)}">${item.status}</span></div>
      <div style="margin-top:6px;font-size:16px;color:#fff">${escapeHtml(item.request_type)}</div>
      <div class="muted" style="margin-top:6px">${escapeHtml(item.details || 'No extra details')}</div>
      <div class="muted" style="margin-top:6px">${item.ward} · ${formatTime(item.created_at)}</div>
      <div class="request-actions">
        <button class="btn secondary" onclick="updateEmergencyStatus('${item.id}','Acknowledged')">ACKNOWLEDGE</button>
        <button class="btn" onclick="updateEmergencyStatus('${item.id}','Completed')">COMPLETE</button>
      </div>
    </div>`).join('');
}

async function updateEmergencyStatus(id, status){
  await fetch(`/api/emergency/requests/${id}/status`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({status})});
  await loadEmergencyRequests();
}

async function sendEmergency(requestType){
  const patient_id = document.getElementById('emergency-patient').value;
  const details = document.getElementById('emergency-details').value;
  if(!patient_id){ alert('Select a patient first'); return; }
  await fetch('/api/emergency/requests', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({patient_id, request_type: requestType, details})});
  document.getElementById('emergency-details').value = '';
  await loadEmergencyRequests();
  switchTab('emergency', document.querySelector('[data-tab="emergency"]'));
}

async function loadHistorique(){
  const select = document.getElementById('history-patient');
  const patientId = select?.value || Object.keys(patients).sort()[0];
  if(!patientId){ return; }
  if(select && !select.value){ select.value = patientId; }
  const res = await fetch(`/api/history/${patientId}?limit=200`);
  const payload = await res.json();
  document.getElementById('history-total').textContent = payload.total_records || 0;
  document.getElementById('history-nurse-count').textContent = payload.nurse_logs_count || 0;
  document.getElementById('history-emergency-count').textContent = payload.emergency_requests_count || 0;
  const tbody = document.getElementById('history-table-body');
  const records = payload.records || [];
  if(records.length === 0){
    tbody.innerHTML = '<tr><td colspan="7" class="muted">No stored history yet.</td></tr>';
    return;
  }
  tbody.innerHTML = records.slice().reverse().map(r => `
    <tr>
      <td>${formatTime(r.ts)}</td>
      <td>${r.hr}</td>
      <td>${r.spo2}</td>
      <td>${r.sbp}</td>
      <td>${r.dbp}</td>
      <td>${r.temp}</td>
      <td><span class="tag ${statusClassForRisk(r.sev)}">${r.sev}</span></td>
    </tr>`).join('');
}

function renderAI(){
  const items = Object.values(patients);
  const critical = items.filter(p => p.overall === 'CRITICAL').length;
  const warning = items.filter(p => p.overall === 'WARNING').length;
  const normal = items.filter(p => p.overall === 'NORMAL' || p.overall === 'CAUTION').length;
  document.getElementById('ai-critical').textContent = critical;
  document.getElementById('ai-warning').textContent = warning;
  document.getElementById('ai-normal').textContent = normal;
  const diseaseCounts = {};
  items.forEach(p => { const d = p.ml?.disease?.prediction || 'Unknown'; diseaseCounts[d] = (diseaseCounts[d] || 0) + 1; });
  const topDisease = Object.entries(diseaseCounts).sort((a,b) => b[1]-a[1])[0];
  document.getElementById('ai-top-disease').textContent = topDisease ? topDisease[0] : '—';
  const tbody = document.getElementById('ai-table-body');
  tbody.innerHTML = items.sort((a,b) => SEV_ORDER.indexOf(b.overall)-SEV_ORDER.indexOf(a.overall)).map(p => `
    <tr>
      <td>${p.patient_id} · ${p.patient_name}</td>
      <td><span class="tag ${statusClassForRisk(p.overall)}">${p.overall}</span></td>
      <td>${escapeHtml(p.ml?.disease?.prediction || '—')}</td>
      <td>${p.ml?.disease?.confidence || 0}%</td>
      <td>${p.ward}</td>
    </tr>`).join('');
}

function statusClassForRisk(risk){
  if(risk === 'CRITICAL') return 'pending';
  if(risk === 'WARNING') return 'ack';
  return 'done';
}

function formatTime(ts){
  try{ return new Date(ts).toLocaleString('en-GB'); } catch(e){ return ts; }
}

function escapeHtml(text){
  return String(text).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

const eventSource = new EventSource('/api/stream');
eventSource.onmessage = (e) => {
  try {
    const data = JSON.parse(e.data);
    data.forEach(p => updatePatient(p));
    updateAlertBar();
    renderAI();
    if(document.getElementById('historique-section')?.classList.contains('active')){ loadHistorique(); }
  } catch(err){ console.error(err); }
};
eventSource.onerror = () => { document.getElementById('clock').style.color = 'var(--red)'; };

document.getElementById('nurse-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    patient_id: document.getElementById('nurse-patient').value,
    action_type: document.getElementById('nurse-action').value,
    item_name: document.getElementById('nurse-item').value,
    note: document.getElementById('nurse-note').value,
    performed_by: document.getElementById('nurse-by').value || 'Nurse',
  };
  await fetch('/api/nurse/logs', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  document.getElementById('nurse-item').value = '';
  document.getElementById('nurse-note').value = '';
  await loadNurseLogs();
  switchTab('nurse', document.querySelector('[data-tab="nurse"]'));
});

document.getElementById('diabetes-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    pregnancies: Number(document.getElementById('db-pregnancies').value || 0),
    glucose: Number(document.getElementById('db-glucose').value || 0),
    blood_pressure: Number(document.getElementById('db-blood-pressure').value || 0),
    skin_thickness: Number(document.getElementById('db-skin-thickness').value || 0),
    insulin: Number(document.getElementById('db-insulin').value || 0),
    bmi: Number(document.getElementById('db-bmi').value || 0),
    diabetes_pedigree_function: Number(document.getElementById('db-dpf').value || 0),
    age: Number(document.getElementById('db-age').value || 0),
  };
  const res = await fetch('/api/diabetes/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const result = await res.json();
  const predEl = document.getElementById('diabetes-prediction');
  const confEl = document.getElementById('diabetes-confidence');
  const typeEl = document.getElementById('diabetes-type');
  const typePctEl = document.getElementById('diabetes-type-percentages');
  const noteEl = document.getElementById('diabetes-note');
  predEl.textContent = result.prediction || '—';
  confEl.textContent = `${result.confidence || 0}%`;
  typeEl.textContent = result.predicted_type || '—';
  const typePcts = result.type_percentages || {};
  typePctEl.textContent = `Type 1: ${typePcts['Type 1 Diabetes'] || 0}% · Type 2: ${typePcts['Type 2 Diabetes'] || 0}%`;
  noteEl.textContent = result.note || 'Estimated diabetes type percentages are based on the clinical profile.';
  predEl.style.color = (result.prediction === 'Diabetes Risk') ? 'var(--orange)' : 'var(--green)';
});

window.addEventListener('load', async () => {
  try {
    const stateRes = await fetch('/api/state');
    const data = await stateRes.json();
    data.forEach(p => updatePatient(p));
    updateAlertBar();
    renderAI();
    if(document.getElementById('historique-section')?.classList.contains('active')){ loadHistorique(); }
  } catch (e) {}
  await loadNurseLogs();
  await loadEmergencyRequests();
  await loadHistorique();
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  ICU REAL-TIME MONITORING DASHBOARD + NURSE + AI + DIABETES + EMERGENCY")
    print(f"  Models dir : {registry.models_dir}")
    print(f"  Models loaded: {list(registry.artifacts.keys())}")
    print(f"  Patients : {len(PATIENT_PROFILES)}")
    print(f"  Sim interval : {STREAM_INTERVAL_MS} ms")
    print(f"  Display update: {DISPLAY_INTERVAL_MS} ms")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")