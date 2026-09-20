# AnemiaSense

Flask + scikit-learn web app that screens for anemia from blood test values, gives a
Low/Moderate/High risk level with guidance, and tracks a patient's follow-up results.

**Demo:** <add your deployed link here>
**Note:** trained on synthetic data; a screening aid, not a diagnosis.

## Run locally
```
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

## API
```
curl -X POST http://127.0.0.1:5000/api/assess -H "Content-Type: application/json" \
  -d '{"age":30,"sex":"female","hemoglobin":9.5,"rbc":3.6,"hematocrit":29,"patient_id":"P-1"}'
```

## Structure
`app.py` (routes, validation, SQLite history) · `train_model.py` (data, training, report plot) ·
`features.py` (shared indices) · `templates/`, `static/` · `Procfile` (deployment)
