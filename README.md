# Taller: Sistema Web de Evaluación de Riesgo Crediticio (Dataset real)

Este proyecto crea:
- Un **modelo de Machine Learning** (clasificación) para predecir riesgo crediticio (good/bad).
- Una **API** con FastAPI para hacer predicciones.
- Una **web** (dashboard + formulario) servida por la API.

## 1) Requisitos
- Python 3.9+ (ideal 3.10/3.11)
- VS Code (opcional)

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 2) Dataset (OBLIGATORIO)
Descarga `german_credit_data.csv` (Kaggle, versión “human-readable”) y **renómbralo** a:
```
dataset.csv
```
Colócalo en la carpeta raíz del proyecto (mismo nivel que `train.py`).

El dataset debe contener (al menos) estas columnas:
- target (good/bad)
- age, credit_amount, month_duration, payment_to_income_ratio
- years_employment, status_account, credit_history, status_savings

## 3) Entrenar el modelo
```bash
python train.py
```
Esto genera:
- `modelo.pkl` (modelo entrenado)
- `metrics.json` (métricas para el dashboard)

## 4) Ejecutar la web + API
```bash
uvicorn main:app --reload
```

Abrir en el navegador:
- http://localhost:8000

## 5) Evidencias sugeridas (capturas)
- Dashboard con métricas + distribución good/bad
- Predicción “Bajo riesgo” (good)
- Predicción “Alto riesgo” (bad)
- Terminal mostrando `python train.py` y `uvicorn ...`

---

### Notas
- Este proyecto es didáctico: rápido, entendible y funciona en laboratorio.
- El “dashboard” se hace con HTML/CSS (sin librerías externas) para evitar depender de internet.
