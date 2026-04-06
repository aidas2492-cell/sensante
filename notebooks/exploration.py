import pandas as pd

# Charger les donnees
df = pd.read_csv("data/patients_dakar.csv")

# Apercu
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

print("\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f"  {diag:12s} : {temp:.1f} C")

print("\n" + "=" * 50)
print("Exploration terminee !")
print("=" * 50)
# ===== PATIENTS PAR SEXE ET DIAGNOSTIC =====
print("\n--- Patients par sexe et diagnostic ---")
sex_diag = df.groupby(["sexe", "diagnostic"]).size()
for (sexe, diag), count in sex_diag.items():
    print(f"  {sexe} - {diag:12s} : {count:3d} patients")