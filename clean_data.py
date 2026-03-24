import pandas as pd

INPUT_PATH = "dataset.csv"
OUTPUT_PATH = "dataset_clean.csv"

def main():
    print("Cargando dataset...")

    
    
    df = pd.read_csv(INPUT_PATH, sep=None, engine="python")

    # Limpiar espacios y pasar texto a minúscula
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Verificar target
    if "target" not in df.columns:
        raise ValueError("No existe columna target")

    print("Valores únicos en target antes de limpiar:")
    print(df["target"].unique())

    # Mantener solo good y bad
    df = df[df["target"].isin(["good", "bad"])]

    print("Valores únicos en target después de limpiar:")
    print(df["target"].unique())

    # Guardar limpio
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset limpio guardado como:", OUTPUT_PATH)

if __name__ == "__main__":
    main()