import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ======================
# Chargement du modèle
# ======================
MODEL_PATH = "modele/logistic_regression_model.pkl"
pipe = joblib.load(MODEL_PATH)

# ======================
# Colonnes attendues par le modèle
# ======================
FEATURES_NUM = [
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "SEX"
]

DUMMIES = [
    "EDUCATION_1", "EDUCATION_3", "EDUCATION_4",
    "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3"
]

ALL_FEATURES = FEATURES_NUM + DUMMIES

# ======================
# Fonctions utilitaires
# ======================
def to_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default

def to_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def validate_pay_values(row):
    """
    Règles PAY_* :
    - PAY_0, PAY_2, PAY_3, PAY_4 : -1 ou 1..8 (0 interdit)
    - PAY_5, PAY_6              : -1 ou 2..8 (0 et 1 interdits)
    """
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4"]:
        v = row[col]
        if v == 0 or v < -1 or v > 8:
            raise ValueError(
                f"{col} doit être -1 ou entre 1 et 8 (0 interdit). Valeur reçue : {v}"
            )

    for col in ["PAY_5", "PAY_6"]:
        v = row[col]
        if v in (0, 1) or v < -1 or v > 8:
            raise ValueError(
                f"{col} doit être -1 ou entre 2 et 8 (0 et 1 interdits). Valeur reçue : {v}"
            )


def build_row_from_form(form):
    """
    Construit une observation avec EXACTEMENT les colonnes attendues par le modèle.
    """
    row = {}

    # Numériques / entiers
    row["AGE"] = to_int(form.get("AGE"), 0)
    row["LIMIT_BAL"] = to_float(form.get("LIMIT_BAL"), 0.0)
    row["SEX"] = to_int(form.get("SEX"), 0)

    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        row[col] = to_int(form.get(col), 0)

    for col in [
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]:
        row[col] = to_float(form.get(col), 0.0)

    # EDUCATION (fusion 1 & 2)
    edu = to_int(form.get("EDUCATION"), 1)
    if edu not in (1, 3, 4):
        raise ValueError("EDUCATION doit être 1, 3 ou 4.")
    row["EDUCATION_1"] = 1 if edu == 1 else 0
    row["EDUCATION_3"] = 1 if edu == 3 else 0
    row["EDUCATION_4"] = 1 if edu == 4 else 0

    # MARRIAGE
    mar = to_int(form.get("MARRIAGE"), 1)
    if mar not in (1, 2, 3):
        raise ValueError("MARRIAGE doit être 1, 2 ou 3.")
    row["MARRIAGE_1"] = 1 if mar == 1 else 0
    row["MARRIAGE_2"] = 1 if mar == 2 else 0
    row["MARRIAGE_3"] = 1 if mar == 3 else 0

    # Sécurité : toutes les colonnes existent
    for c in ALL_FEATURES:
        if c not in row:
            row[c] = 0

    return row

# ======================
# Routes Flask
# ======================
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict-form")
def predict_form():
    try:
        row = build_row_from_form(request.form)
        validate_pay_values(row)

        X = pd.DataFrame([[row[c] for c in ALL_FEATURES]], columns=ALL_FEATURES)

        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= 0.5)

        return render_template("result.html", proba=proba, pred=pred, error=None)

    except Exception as e:
        return render_template(
            "result.html", proba=None, pred=None, error=str(e)
        ), 400


@app.post("/predict")
def predict_json():
    payload = request.get_json(force=True)
    if isinstance(payload, dict):
        payload = [payload]

    df = pd.DataFrame(payload)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": "Champs manquants", "missing": missing}), 400

    df = df[ALL_FEATURES]
    proba = pipe.predict_proba(df)[:, 1].astype(float).tolist()
    pred = [int(p >= 0.5) for p in proba]

    return jsonify(
        [{"proba_default": p, "prediction": y} for p, y in zip(proba, pred)]
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
