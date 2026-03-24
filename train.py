import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATASET_PATH = "dataset_clean.csv"
MODEL_PATH = "modelo.pkl"
METRICS_PATH = "metrics.json"

# Campos que usará el formulario (para que sea entendible
FEATURES = [
    "age",
    "credit_amount",
    "month_duration",
    "payment_to_income_ratio",
    "years_employment",
    "status_account",
    "credit_history",
    "status_savings",
]

TARGET_COL = "target"   # good / bad


def main():
    # 1) Cargar dataset
    df = pd.read_csv(DATASET_PATH, sep=None, engine="python")

    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas en dataset.csv: "
            + ", ".join(missing)
            + "\nRevisa que descargaste el CSV correcto y lo renombraste a dataset.csv."
        )

    # 2) Convertir target
    df[TARGET_COL] = df[TARGET_COL].map({"good": 0, "bad": 1})
    if df[TARGET_COL].isna().any():
        raise ValueError("La columna target debe tener valores 'good' y 'bad'.")

    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()

    # 3) Columnas categóricas / numéricas
    categorical_cols = [
        "status_account",
        "credit_history",
        "status_savings",
        "years_employment"
    ]

    numeric_cols = [
        "age",
        "credit_amount",
        "month_duration",
        "payment_to_income_ratio"
    ]

    # 4) Preprocesamiento + modelo (Pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # 5) Split + entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    # 6) Métricas
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP],[FN, TP]]

    # Distribución
    counts = y.value_counts().to_dict()  # {0:..., 1:...}
    total = int(len(y))
    good = int(counts.get(0, 0))
    bad = int(counts.get(1, 0))

    # Opciones para selects (desde el dataset)
    options = {}
    for col in categorical_cols:
        # ordenamos para que quede bonito en el formulario
        options[col] = sorted(df[col].dropna().astype(str).unique().tolist())

    payload = {
        "dataset": {
            "total": total,
            "good": good,
            "bad": bad,
            "bad_rate": round((bad / total) if total else 0, 4),
        },
        "features": FEATURES,
        "categorical_cols": categorical_cols,
        "options": options,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": cm,
        }
    }

    # 7) Guardar modelo y métricas
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Modelo entrenado y guardado:", MODEL_PATH)
    print("Métricas guardadas:", METRICS_PATH)
    print("Siguiente paso: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
