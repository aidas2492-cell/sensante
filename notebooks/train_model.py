import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Encoder les variables catégoriques en nombres
# Le modèle ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")       # (500,)
from sklearn.model_selection import train_test_split

# 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour le test
    random_state=42,    # Pour avoir les mêmes résultats à chaque fois
    stratify=y          # Garder les mêmes proportions de diagnostics
)

print(f"\nEntraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,   # 100 arbres de décision
    random_state=42     # Reproductibilité
)

# Entraîner sur les données d'entraînement
model.fit(X_train, y_train)

print("\nModèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print("\nComparaison des 10 premiers patients :")
print(comparison)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualiser la matrice de confusion (optionnel)
# Créer le dossier figures s'il n'existe pas
import os
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("\nFigure sauvegardée dans figures/confusion_matrix.png")
import joblib

# Créer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sérialiser le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder aussi les encodeurs (indispensables pour les nouvelles données)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features (pour référence)
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et métadonnées sauvegardés.")
# Recharger le modèle depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Tester avec un nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
prediction = model_loaded.predict([features])[0]
probabilites = model_loaded.predict_proba([features])[0]

print(f"\n--- Résultat pour le patient ---")
print(f"Âge: {nouveau_patient['age']}, Sexe: {nouveau_patient['sexe']}")
print(f"Température: {nouveau_patient['temperature']}°C")
print(f"Diagnostic prédit: {prediction}")
print(f"Probabilités:")
for classe, prob in zip(model_loaded.classes_, probabilites):
    print(f"  {classe}: {prob:.1%}")