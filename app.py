import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURES, add_indices

BASE_DIR = Path(__file__).parent
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "anemiasense.db"))
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError("models/model.pkl not found. Run `python train_model.py` first.")
with open(MODEL_PATH, "rb") as f:  # only load pickles you created yourself
    model = pickle.load(f)

LIMITS = {"age": (1, 120), "hemoglobin": (3, 20), "rbc": (1, 8), "hematocrit": (10, 60)}
LABELS = {"age": "Age", "hemoglobin": "Hemoglobin", "rbc": "RBC count", "hematocrit": "Hematocrit"}
ADVICE = {
    "Low": "Values look consistent with no anemia. Continue routine check-ups.",
    "Moderate": "Some indicators suggest possible anemia. Repeat the blood test and consider iron studies (ferritin, B12, folate).",
    "High": "Strong indicators of anemia. Refer the patient to a physician promptly to find the cause and start treatment.",
}


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, patient_id TEXT,
            hemoglobin REAL, rbc REAL, hematocrit REAL, probability REAL, risk TEXT)""")


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
    return v, None


def assess(v):
    row = pd.DataFrame([{"age": v["age"], "sex": int(v["sex"] == "female"),
                         "hemoglobin": v["hemoglobin"], "rbc": v["rbc"],
                         "hematocrit": v["hematocrit"]}])
    prob = float(model.predict_proba(add_indices(row)[FEATURES])[0][1])
    risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.7 else "High"
    cutoff = 13.0 if v["sex"] == "male" else 12.0  # WHO hemoglobin thresholds (g/dL)
    return {"probability": round(prob * 100, 1), "risk": risk, "advice": ADVICE[risk],
            "cutoff": cutoff, "below_cutoff": v["hemoglobin"] < cutoff}


def save_record(v, r):
    if not v["patient_id"]:
        return
    with sqlite3.connect(DB_PATH) as db:
        db.execute("INSERT INTO records (created, patient_id, hemoglobin, rbc, hematocrit, probability, risk)"
                   " VALUES (?,?,?,?,?,?,?)",
                   (datetime.now().strftime("%Y-%m-%d %H:%M"), v["patient_id"], v["hemoglobin"],
                    v["rbc"], v["hematocrit"], r["probability"], r["risk"]))


def history(patient_id):
    if not patient_id:
        return []
    with sqlite3.connect(DB_PATH) as db:
        return db.execute("SELECT created, hemoglobin, hematocrit, probability, risk FROM records"
                          " WHERE patient_id=? ORDER BY id DESC LIMIT 10", (patient_id,)).fetchall()


init_db()


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
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1")
