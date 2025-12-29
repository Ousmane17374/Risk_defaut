from pydantic import BaseModel # Utiliser pour la validation des données
import numpy as np
import pandas as pd
import joblib # utiliser pour charger le modèle sauvegarder
from flask import Flask, request, jsonify # Flask est un micro-FrameWork

app = Flask(__name__)

MODEL_PATH = "model/credit_default_pipe.pkl"
pipe = joblib.load(MODEL_PATH)

# Colonnes attendues par ton modèle (d'après ton df.info)
FEATURES_NUM = [
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "SEX"
]

# Dummies attendues
DUMMIES = [
    "EDUCATION_1", "EDUCATION_3", "EDUCATION_4",
    "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3"
]

ALL_FEATURES = FEATURES_NUM + DUMMIES

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

def build_row_from_form(form):
    """
    Construit une ligne (dict) contenant EXACTEMENT les features attendues (ALL_FEATURES),
    avec encodage dummies pour EDUCATION et MARRIAGE.
    """

    row = {}

    # ---- Numériques / entiers ----
    row["AGE"] = to_int(form.get("AGE"), 0)
    row["LIMIT_BAL"] = to_float(form.get("LIMIT_BAL"), 0.0)

    # SEX déjà encodée 0/1 dans ton dataset final
    row["SEX"] = to_int(form.get("SEX"), 0)

    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        row[col] = to_int(form.get(col), 0)

    for col in ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]:
        row[col] = to_float(form.get(col), 0.0)

    # ---- EDUCATION : tu as fusionné (1 et 2) => valeurs possibles : 1,3,4 uniquement ----
    edu = to_int(form.get("EDUCATION"), 1)  # défaut = 1
    row["EDUCATION_1"] = 1 if edu == 1 else 0
    row["EDUCATION_3"] = 1 if edu == 3 else 0
    row["EDUCATION_4"] = 1 if edu == 4 else 0

    # ---- MARRIAGE : valeurs proposées : 1,2,3 ----
    mar = to_int(form.get("MARRIAGE"), 1)  # défaut = 1
    row["MARRIAGE_1"] = 1 if mar == 1 else 0
    row["MARRIAGE_2"] = 1 if mar == 2 else 0
    row["MARRIAGE_3"] = 1 if mar == 3 else 0

    # S'assurer que toutes les colonnes existent
    for c in ALL_FEATURES:
        if c not in row:
            row[c] = 0

    return row

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict-form")
def predict_form():
    row = build_row_from_form(request.form)
    X = pd.DataFrame([[row[c] for c in ALL_FEATURES]], columns=ALL_FEATURES)

    proba = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    return render_template("result.html", proba=proba, pred=pred)

# (Optionnel) endpoint JSON si tu veux tester avec Postman/curl
@app.post("/predict")
def predict_json():
    payload = request.get_json(force=True)

    # accepte dict (1 client) ou liste de dicts
    if isinstance(payload, dict):
        payload = [payload]

    df = pd.DataFrame(payload)

    # Si l'utilisateur envoie EDUCATION/MARRIAGE au lieu des dummies, on peut les convertir
    # Ici on attend directement ALL_FEATURES pour simplifier (prod)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": "Champs manquants", "missing": missing}), 400

    df = df[ALL_FEATURES]
    proba = pipe.predict_proba(df)[:, 1].astype(float).tolist()
    pred = [int(p >= 0.5) for p in proba]

    return jsonify([{"proba_default": p, "prediction": y} for p, y in zip(proba, pred)])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
