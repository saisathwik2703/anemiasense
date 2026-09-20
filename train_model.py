"""Train the anemia model. Run once:  python train_model.py

Uses SYNTHETIC data so the project runs anywhere. To use a real dataset,
replace make_data() with a function that returns the same columns plus
`anemic` (0/1). Sex: 0 = male, 1 = female.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import FEATURES, add_indices

BASE = Path(__file__).parent


def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    age = rng.integers(5, 90, n)
    anemic = rng.random(n) < np.where(sex == 1, 0.40, 0.25)
    healthy_hb = np.where(sex == 1, rng.normal(13.6, 0.9, n), rng.normal(15.0, 1.0, n))
    hb = np.clip(np.where(anemic, rng.normal(10.2, 1.4, n), healthy_hb), 4, 19)
    hct = np.clip(hb * 3 + rng.normal(0, 1.5, n), 12, 58)
    rbc = np.clip(np.where(anemic, rng.normal(3.8, 0.5, n), rng.normal(4.9, 0.4, n)), 2, 7)
    df = pd.DataFrame({"age": age, "sex": sex, "hemoglobin": hb, "rbc": rbc,
                       "hematocrit": hct, "anemic": anemic.astype(int)})
    return add_indices(df)


def main():
    df = make_data()
    X, y = df[FEATURES], df["anemic"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Accuracy: {accuracy_score(y_te, pred):.3f}")
    print(classification_report(y_te, pred, target_names=["Not anemic", "Anemic"]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues", ax=ax[0],
                xticklabels=["Not anemic", "Anemic"], yticklabels=["Not anemic", "Anemic"])
    ax[0].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(ax=ax[1], color="#1f5f7a")
    ax[1].set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(BASE / "static" / "model_report.png", dpi=120)

    (BASE / "models").mkdir(exist_ok=True)
    with open(BASE / "models" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved models/model.pkl and static/model_report.png")


if __name__ == "__main__":
    main()
