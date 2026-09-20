import json
import os
import pickle
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURES, add_indices

BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "anemiasense.db"))
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

app = Flask(__name__)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    raise FileNotFoundError("Model files not found in models/. Run `python train_model.py` first.")
with open(MODEL_PATH, "rb") as f:  # only load pickles you created yourself
    model = pickle.load(f)
METRICS = json.loads(METRICS_PATH.read_text())

LIMITS = {"age": (1, 120), "hemoglobin": (3, 20), "rbc": (1, 8), "hematocrit": (10, 60)}
LABELS = {"age": "Age", "hemoglobin": "Hemoglobin", "rbc": "RBC count", "hematocrit": "Hematocrit"}
ADVICE = {
    "Low": "Values look consistent with no anemia. Continue routine check-ups.",
    "Moderate": "Some indicators suggest possible anemia. Repeat the blood test and consider iron studies (ferritin, B12, folate).",
    "High": "Strong indicators of anemia. Refer the patient to a physician promptly to find the cause and start treatment.",
}

# Tables follow the project's ER diagram (see docs/AnemiaSense_ERD.png)
SCHEMA = """
CREATE TABLE IF NOT EXISTS web_application (app_id INTEGER PRIMARY KEY, app_name TEXT, framework TEXT, deployment_date TEXT);
CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT, email TEXT);
CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, patient_name TEXT, age INTEGER, gender TEXT,
    contact_number TEXT, registration_date TEXT);
CREATE TABLE IF NOT EXISTS blood_tests (test_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id), hemoglobin REAL, rbc_count REAL, mcv REAL, mch REAL,
    mchc REAL, hematocrit REAL, test_date TEXT);
CREATE TABLE IF NOT EXISTS ml_models (model_id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT UNIQUE,
    algorithm_type TEXT, accuracy_score REAL, precision_score REAL, recall_score REAL, f1_score REAL, training_date TEXT);
CREATE TABLE IF NOT EXISTS dataset_records (record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL REFERENCES blood_tests(test_id), feature_vector TEXT, preprocessing_status TEXT, class_label INTEGER);
CREATE TABLE IF NOT EXISTS model_training_records (model_id INTEGER REFERENCES ml_models(model_id),
    record_id INTEGER REFERENCES dataset_records(record_id), PRIMARY KEY (model_id, record_id));
CREATE TABLE IF NOT EXISTS predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id), test_id INTEGER REFERENCES blood_tests(test_id),
    model_id INTEGER REFERENCES ml_models(model_id), prediction_result TEXT, confidence_score REAL, prediction_date TEXT);
CREATE TABLE IF NOT EXISTS evaluation_reports (report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id), accuracy REAL, classification_report TEXT,
    confusion_matrix TEXT, generated_date TEXT);
"""


def db_conn():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db():
    """Create tables, seed the web_application row, register the trained model. Returns model_id."""
    with db_conn() as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO web_application (app_name, framework, deployment_date) "
                   "SELECT 'AnemiaSense', 'Flask', ? WHERE NOT EXISTS (SELECT 1 FROM web_application)",
                   (date.today().isoformat(),))
        m = METRICS
        db.execute("INSERT OR IGNORE INTO ml_models (model_name, algorithm_type, accuracy_score, precision_score,"
                   " recall_score, f1_score, training_date) VALUES (?,?,?,?,?,?,?)",
                   (m["model_name"], m["algorithm_type"], m["accuracy"], m["precision"], m["recall"], m["f1"],
                    m["training_date"]))
        return db.execute("SELECT model_id FROM ml_models WHERE model_name=?", (m["model_name"],)).fetchone()[0]


def parse(data):
    try:
        v = {k: float(data.get(k, "")) for k in LIMITS}
    except (TypeError, ValueError):
        return None, "Age, hemoglobin, RBC count and hematocrit must be numbers."
    for k, (lo, hi) in LIMITS.items():
        if not lo <= v[k] <= hi:
            return None, f"{LABELS[k]} must be between {lo} and {hi}."
    if data.get("sex") not in ("male", "female"):
        return None, "Choose male or female."
    v["sex"] = data["sex"]
    v["patient_id"] = (data.get("patient_id") or "").strip()[:40]
    v["patient_name"] = (data.get("patient_name") or "").strip()[:80]
    return v, None


def assess(v):
    row = pd.DataFrame([{"age": v["age"], "sex": int(v["sex"] == "female"), "hemoglobin": v["hemoglobin"],
                         "rbc": v["rbc"], "hematocrit": v["hematocrit"]}])
    feats = add_indices(row)[FEATURES]
    prob = float(model.predict_proba(feats)[0][1])
    risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.7 else "High"
    cutoff = 13.0 if v["sex"] == "male" else 12.0  # WHO hemoglobin thresholds (g/dL)
    return {"probability": round(prob * 100, 1), "risk": risk, "advice": ADVICE[risk],
            "cutoff": cutoff, "below_cutoff": v["hemoglobin"] < cutoff,
            "mcv": round(float(feats["mcv"][0]), 1), "mch": round(float(feats["mch"][0]), 1),
            "mchc": round(float(feats["mchc"][0]), 1), "features": [float(x) for x in feats.iloc[0]]}


def save_record(v, r):
    """Save patient, blood test, dataset record, prediction and report (only when a patient ID is given)."""
    if not v["patient_id"]:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_conn() as db:
        db.execute("INSERT INTO patients (patient_id, patient_name, age, gender, registration_date) VALUES (?,?,?,?,?)"
                   " ON CONFLICT(patient_id) DO UPDATE SET age=excluded.age, gender=excluded.gender,"
                   " patient_name=COALESCE(NULLIF(excluded.patient_name, ''), patient_name)",
                   (v["patient_id"], v["patient_name"], int(v["age"]), v["sex"], now))
        test_id = db.execute("INSERT INTO blood_tests (patient_id, hemoglobin, rbc_count, mcv, mch, mchc, hematocrit,"
                             " test_date) VALUES (?,?,?,?,?,?,?,?)",
                             (v["patient_id"], v["hemoglobin"], v["rbc"], r["mcv"], r["mch"], r["mchc"],
                              v["hematocrit"], now)).lastrowid
        db.execute("INSERT INTO dataset_records (test_id, feature_vector, preprocessing_status, class_label)"
                   " VALUES (?,?,?,?)", (test_id, json.dumps(r["features"]), "indices derived",
                                         int(r["probability"] >= 50)))
        pred_id = db.execute("INSERT INTO predictions (patient_id, test_id, model_id, prediction_result,"
                             " confidence_score, prediction_date) VALUES (?,?,?,?,?,?)",
                             (v["patient_id"], test_id, MODEL_ID, r["risk"], r["probability"], now)).lastrowid
        db.execute("INSERT INTO evaluation_reports (prediction_id, accuracy, classification_report, confusion_matrix,"
                   " generated_date) VALUES (?,?,?,?,?)",
                   (pred_id, METRICS["accuracy"], METRICS["classification_report"],
                    json.dumps(METRICS["confusion_matrix"]), now))


def history(patient_id):
    if not patient_id:
        return []
    with db_conn() as db:
        return db.execute("SELECT p.prediction_date, b.hemoglobin, b.hematocrit, p.confidence_score,"
                          " p.prediction_result FROM predictions p JOIN blood_tests b ON b.test_id = p.test_id"
                          " WHERE p.patient_id=? ORDER BY p.prediction_id DESC LIMIT 10", (patient_id,)).fetchall()


MODEL_ID = init_db()


@app.route("/")
def index():
    return render_template("index.html", form={}, result=None, error=None, history=[])


@app.route("/assess", methods=["POST"])
def assess_form():
    v, error = parse(request.form)
    result = assess(v) if v else None
    if result:
        save_record(v, result)
    return render_template("index.html", form=request.form, result=result, error=error,
                           history=history(v["patient_id"]) if v else [])


@app.route("/api/assess", methods=["POST"])
def assess_api():
    v, error = parse(request.get_json(silent=True) or {})
    if error:
        return jsonify({"error": error}), 400
    result = assess(v)
    save_record(v, result)
    return jsonify({k: x for k, x in result.items() if k != "features"})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1")
