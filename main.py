import json
import os
import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# =========================
# Paths robustos
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CLEAN = os.path.join(BASE_DIR, "dataset_clean.csv")
DATASET_RAW = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# =========================
# App
# =========================
app = FastAPI(title="Riesgo Crediticio - Taller")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================
# Cache simple
# =========================
_MODEL = None
_METRICS = None

# =========================
# Labels amigables (UI)
# - value (lo que se envía): código real (no se cambia)
# - label (lo que se muestra): texto entendible
# =========================
LABEL_MAPS = {
    "status_account": {
        "0 to < 200 DM": "Cuenta corriente: saldo bajo (₡0 a ₡199)",
        "< 0 DM": "Cuenta corriente: saldo negativo",
        ">= 200 DM": "Cuenta corriente: saldo alto (₡200 o más)",
        "no checking account": "No tiene cuenta corriente",
    },
    "credit_history": {
        "all credits at this bank paid back duly": "Pagos al día en este banco",
        "critical account/ other credits existing (not at this bank)": "Perfil crítico / otros créditos activos",
        "delay in paying off in the past": "Tuvo atrasos en el pasado",
        "existing credits paid back duly till now": "Créditos actuales al día",
        "no credits taken/ all credits paid back duly": "Sin créditos / historial limpio",
    },
    "status_savings": {
        "100 to < 500 DM": "Ahorros: bajos (₡100 a ₡499)",
        "500 to < 1000 DM": "Ahorros: medios (₡500 a ₡999)",
        "< 100 DM": "Ahorros: muy bajos (menos de ₡100)",
        ">= 1000 DM": "Ahorros: altos (₡1000 o más)",
        "unknown/ no savings account": "Ahorros: no disponible / sin cuenta",
    },
    "years_employment": {
        "1 to < 4 years": "Entre 1 y 3 años de empleo",
        "4 to < 7 years": "Entre 4 y 6 años de empleo",
        "< 1 year": "Menos de 1 año de empleo",
        ">= 7 years": "7 años o más de empleo",
        "unemployed": "Desempleado",
    },
}

def build_labels(options: dict) -> dict:
    """
    Crea labels amigables para el template SIN cambiar los valores reales.
    Si un código no está en el mapa, se muestra el mismo código (fallback seguro).
    """
    labels = {}
    for col, values in (options or {}).items():
        col_map = LABEL_MAPS.get(col, {})
        labels[col] = {v: col_map.get(v, v) for v in values}
    return labels

def _get_dataset_path():
    # Preferimos el dataset limpio si existe
    if os.path.exists(DATASET_CLEAN):
        return DATASET_CLEAN
    return DATASET_RAW


def load_metrics():
    global _METRICS
    if _METRICS is not None:
        return _METRICS
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        _METRICS = json.load(f)
    return _METRICS


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(MODEL_PATH):
        return None
    _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    metrics = load_metrics()
    model_exists = os.path.exists(MODEL_PATH)

    # Opciones para selects (si no hay metrics, las sacamos del dataset)
    options = {}
    dataset_path = _get_dataset_path()

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, sep=None, engine="python")
        cols = ["years_employment", "status_account", "credit_history", "status_savings"]
        for c in cols:
            if c in df.columns:
                options[c] = sorted(df[c].dropna().astype(str).unique().tolist())

    # Si ya hay metrics.json, preferimos las opciones del entrenamiento
    if metrics and "options" in metrics:
        options = metrics["options"]

    labels = build_labels(options)
    
    context = {
        "request": request,
        "model_exists": model_exists,
        "metrics": metrics,
        "options": options,
        "labels": labels
    }
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/predict")
async def predict(request: Request):
    model = load_model()
    metrics = load_metrics()

    if model is None or metrics is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Primero debes entrenar el modelo: ejecuta `py train.py`."}
        )

    data = await request.json()
    features = metrics.get("features", [])

    # Validación mínima
    missing = [f for f in features if f not in data]
    if missing:
        return JSONResponse(
            status_code=400,
            content={"error": f"Faltan campos: {', '.join(missing)}"}
        )

    numeric_fields = ["age", "credit_amount", "month_duration", "payment_to_income_ratio"]
    row = {}

    for f in features:
        if f in numeric_fields:
            row[f] = safe_float(data.get(f))
        else:
            row[f] = str(data.get(f)).strip()

    df_input = pd.DataFrame([row])

    pred = int(model.predict(df_input)[0])          # 0=good(bajo) / 1=bad(alto)
    proba_bad = float(model.predict_proba(df_input)[0][1])

    riesgo = "Alto" if pred == 1 else "Bajo"

    return {
        "riesgo": riesgo,
        "probabilidad_default": round(proba_bad, 4),
        "mensaje": (
            "Recomendación: revisar manualmente"
            if proba_bad >= 0.5
            else "Recomendación: aprobado (con reglas del negocio)"
        )
    }