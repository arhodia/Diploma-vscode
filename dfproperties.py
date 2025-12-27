import os
import pandas as pd

# 1) Βάλε εδώ το path του αρχείου σου
FILE_PATH = "C:\\Users\\arhod\\Desktop\\Diploma-vscode\\INC 5000 Companies 2019.csv"   # π.χ. r"C:\Users\...\INC 5000 Companies 2019.csv"
# 2) (προαιρετικό) φάκελος εξόδου για τα αποτελέσματα
OUT_DIR = "inc5000_column_stats"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(FILE_PATH)

# Case-insensitive αντιστοίχιση ονομάτων στηλών
col_map = {c.strip().lower(): c for c in df.columns}

targets = ["industry", "city", "state"]
missing = [c for c in targets if c not in col_map]
if missing:
    raise KeyError(
        f"Δεν βρέθηκαν οι στήλες {missing} στο αρχείο.\n"
        f"Διαθέσιμες στήλες: {list(df.columns)}"
    )

N = len(df)

def analyze_column(df: pd.DataFrame, col_name: str, out_dir: str) -> pd.DataFrame:
    """
    Επιστρέφει DataFrame με:
    - διαφορετικές τιμές (distinct values)
    - count (πόσες φορές εμφανίζεται)
    - percentage (ποσοστό επί του συνόλου)
    Και το αποθηκεύει σε CSV.
    """
    # Καθαρισμός: κρατάμε NaN ως <MISSING>, trim σε strings
    s = df[col_name].astype("string").str.strip()

    counts = s.value_counts(dropna=False)  # περιλαμβάνει και κενά/NaN
    stats = counts.rename("count").to_frame()
    stats.index = stats.index.fillna("<MISSING>")

    pct = (stats["count"] / N) * 100
    stats["percentage"] = pct.map(lambda x: f"{x:.4f}%")


    stats = (
        stats.reset_index()
             .rename(columns={"index": col_name})
             .sort_values("count", ascending=False)
             .reset_index(drop=True)
    )

    out_path = os.path.join(out_dir, f"{col_name}_stats.csv")
    stats.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Εκτύπωση βασικού summary
    print("=" * 80)
    print(f"Column: {col_name}")
    print(f"Total rows: {N}")
    print(f"Distinct values: {stats.shape[0]}")
    print(f"Saved: {out_path}")
    # Εκτύπωση ΟΛΩΝ των τιμών (όχι μόνο top 20)
    print("\nAll values:")
    print(stats.to_string(index=False))


    return stats


# --- Run analysis for industry, city, state ---
results = {}
for key in targets:
    col = col_map[key]            # το πραγματικό όνομα στήλης στο CSV
    results[key] = analyze_column(df, col, OUT_DIR)

# Αν θες να πάρεις ως λίστες τα distinct values:
industry_values = results["industry"][col_map["industry"]].tolist()
city_values     = results["city"][col_map["city"]].tolist()
state_values    = results["state"][col_map["state"]].tolist()

print("\nDone. Τα αναλυτικά αποτελέσματα είναι σε CSV μέσα στο:", OUT_DIR)
