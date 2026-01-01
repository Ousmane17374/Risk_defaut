import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# =========================
# 1) Chargement du pipeline
# =========================
MODEL_PATH = "modele/credit_default_pipe.pkl"
pipe = joblib.load(MODEL_PATH)

# Colonnes attendues (si le pipeline a été fit sur un DataFrame)
EXPECTED_COLS = None
try:
    EXPECTED_COLS = list(pipe.feature_names_in_)
except Exception:
    # parfois c'est un step qui possède feature_names_in_
    try:
        EXPECTED_COLS = list(pipe.named_steps["scaler"].feature_names_in_)
    except Exception:
        EXPECTED_COLS = None

# =========================
# 2) Définition des features
# =========================
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


# =========================
# 3) Helpers conversion
# =========================
def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def build_row_from_inputs(get_value_func):
    """
    Construit une ligne dict à partir d'une source (form ou json)
    en créant exactement les colonnes attendues (ALL_FEATURES),
    avec dummies pour EDUCATION et MARRIAGE.
    """

    row = {}

    # Numériques / entiers
    row["AGE"] = to_int(get_value_func("AGE"), 0)
    row["LIMIT_BAL"] = to_float(get_value_func("LIMIT_BAL"), 0.0)

    # SEX (déjà encodée 0/1)
    row["SEX"] = to_int(get_value_func("SEX"), 0)

    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        row[col] = to_int(get_value_func(col), 0)

    for col in [
        "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
        "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
    ]:
        row[col] = to_float(get_value_func(col), 0.0)

    # EDUCATION (fusion 1 et 2 -> 1), valeurs possibles 1,3,4
    edu = to_int(get_value_func("EDUCATION"), 1)
    row["EDUCATION_1"] = 1 if edu == 1 else 0
    row["EDUCATION_3"] = 1 if edu == 3 else 0
    row["EDUCATION_4"] = 1 if edu == 4 else 0

    # MARRIAGE valeurs 1,2,3
    mar = to_int(get_value_func("MARRIAGE"), 1)
    row["MARRIAGE_1"] = 1 if mar == 1 else 0
    row["MARRIAGE_2"] = 1 if mar == 2 else 0
    row["MARRIAGE_3"] = 1 if mar == 3 else 0

    # Garantir toutes les colonnes
    for c in ALL_FEATURES:
        if c not in row:
            row[c] = 0

    return row


def build_X_from_row(row: dict) -> pd.DataFrame:
    """Crée un DataFrame 1 ligne, puis force l'ordre/présence des colonnes attendues."""
    X = pd.DataFrame([row])

    # Si le pipeline annonce les colonnes attendues, on respecte exactement
    if EXPECTED_COLS is not None:
        for c in EXPECTED_COLS:
            if c not in X.columns:
                X[c] = 0
        X = X[EXPECTED_COLS]
    else:
        # fallback : ton ordre défini
        X = X[ALL_FEATURES]

    return X


# =========================
# 4) Routes
# =========================
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict-form")
def predict_form():
    # Lecture depuis formulaire HTML
    row = build_row_from_inputs(lambda k: request.form.get(k))
    X = build_X_from_row(row)

    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    # Debug server logs
    app.logger.info(f"[FORM] proba={proba:.8f} pred={pred}")

    return render_template("result.html", proba=proba, pred=pred)


@app.post("/predict")
def predict_json():
    """
    Endpoint JSON.
    Accepte:
      - un dict (1 personne) avec les champs bruts (AGE, LIMIT_BAL, PAY_*, BILL_AMT*, PAY_AMT*, SEX, EDUCATION, MARRIAGE)
      - ou une liste de dicts
    """
    payload = request.get_json(force=True)

    if isinstance(payload, dict):
        payload = [payload]

    results = []
    for item in payload:
        row = build_row_from_inputs(lambda k: item.get(k))
        X = build_X_from_row(row)

        proba = float(pipe.predict_proba(X)[:, 1][0])
        pred = int(proba >= 0.5)

        results.append({"proba_default": proba, "prediction": pred})

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
