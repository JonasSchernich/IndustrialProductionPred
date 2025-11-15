# src/data/create_subsets.py
import pandas as pd
import pathlib

# --- Pfade (analog zu load_data.py) ---
BASE_DIR = pathlib.Path(__file__).parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Input-Datei
IN_CLEANED_FEATURES = PROCESSED_DIR / "cleaned_features.csv"

# Output-Dateien
OUT_INTUITIVE = PROCESSED_DIR / "intuitive_features.csv"
OUT_TOP_LEVEL = PROCESSED_DIR / "top_level_features.csv"
OUT_SECOND_LEVEL = PROCESSED_DIR / "second_level_features.csv"
OUT_INTUITIVE_SECOND = PROCESSED_DIR / "intuitive_second_level.csv"

# --- Feature-Definitionen ---

# 1. "Intuitive" Fragen (der Teil nach dem Punkt)
INTUITIVE_QUESTIONS = {
    "Geschäftsklima",
    "Geschäftslage_Beurteilung",
    "Nachfrage_gegen_Vormonat",
    "Produktion_gegen_Vormonat",
    "Produktionspläne",
    "Auftragsbestand_Beurteilung"  # Basierend auf Ihrem Beispiel
}

# 2. "Top-Level" Branchen (der Teil vor dem Punkt)
TOP_LEVEL_BRANCHES = {
    "Verarbeitendes_Gewerbe"
}

# 3. "Second-Level" Branchen (der Teil vor dem Punkt)
SECOND_LEVEL_BRANCHES = {
    "Verarbeitendes_Gewerbe_(ohne_Ernährungsgewerbe)",
    "Herstellung_von_Vorleistungsgütern",
    "Herstellung_von_Investitionsgütern",
    "Herstellung_von_Konsumgütern_(Ge-_und_Verbrauchsgüter)",
    "Herstellung_von_Gebrauchsgütern",
    "Herstellung_von_Verbrauchsgütern"
}


def split_feature_name(col_name: str):
    """Trennt 'Branche.Frage' am ersten Punkt."""
    try:
        branch, question = col_name.split('.', 1)
        return branch, question
    except ValueError:
        # Falls kein Punkt im Namen ist (z.B. 'date' oder 'IP_change')
        return None, None


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Lade die Basis-Datei
    try:
        df_clean = pd.read_csv(IN_CLEANED_FEATURES, parse_dates=["date"], index_col="date")
        print(f"'{IN_CLEANED_FEATURES.name}' erfolgreich geladen. Shape: {df_clean.shape}")
    except FileNotFoundError:
        print(f"FEHLER: '{IN_CLEANED_FEATURES.name}' nicht gefunden.")
        print(f"Bitte zuerst das Skript 'load_data.py' ausführen, um die Datei zu erstellen.")
        return

    all_cols = df_clean.columns.tolist()

    # Listen für die neuen Spalten-Sets
    cols_intuitive = []
    cols_top_level = []
    cols_second_level = []
    cols_intuitive_second = []

    # Gehe alle Spalten einmal durch und sortiere sie
    for col in all_cols:
        branch, question = split_feature_name(col)
        if branch is None:
            continue

        is_intuitive = question in INTUITIVE_QUESTIONS
        is_top_level = branch in TOP_LEVEL_BRANCHES
        is_second_level = branch in SECOND_LEVEL_BRANCHES

        # 1. Intuitive Features
        if is_intuitive:
            cols_intuitive.append(col)

        # 2. Top-Level Features
        if is_top_level:
            cols_top_level.append(col)

        # 3. Second-Level Features
        if is_second_level:
            cols_second_level.append(col)

        # 4. Intuitive + Second-Level
        if is_intuitive and is_second_level:
            cols_intuitive_second.append(col)

    # --- Speichern der Subsets ---

    # 1. Intuitive
    df_intuitive = df_clean[cols_intuitive]
    df_intuitive.to_csv(OUT_INTUITIVE)
    print(f"'{OUT_INTUITIVE.name}' gespeichert. Shape: {df_intuitive.shape}")

    # 2. Top-Level
    df_top = df_clean[cols_top_level]
    df_top.to_csv(OUT_TOP_LEVEL)
    print(f"'{OUT_TOP_LEVEL.name}' gespeichert. Shape: {df_top.shape}")

    # 3. Second-Level
    df_second = df_clean[cols_second_level]
    df_second.to_csv(OUT_SECOND_LEVEL)
    print(f"'{OUT_SECOND_LEVEL.name}' gespeichert. Shape: {df_second.shape}")

    # 4. Intuitive Second-Level
    df_intuitive_second = df_clean[cols_intuitive_second]
    df_intuitive_second.to_csv(OUT_INTUITIVE_SECOND)
    print(f"'{OUT_INTUITIVE_SECOND.name}' gespeichert. Shape: {df_intuitive_second.shape}")


if __name__ == "__main__":
    main()