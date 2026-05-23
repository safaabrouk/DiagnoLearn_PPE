"""
=============================================================================
SCRIPT D'ENTRAÎNEMENT DES MODÈLES ML
Système de prédiction des difficultés d'apprentissage en informatique
PPE - CRMEF Maroc
=============================================================================
Usage: python train_models.py
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, recall_score, precision_score,
    confusion_matrix
)
import pickle
import json

print("=" * 60)
print("  ENTRAÎNEMENT DES MODÈLES ML - DiagnoLearn")
print("=" * 60)

# ─── 1. Chargement du dataset ────────────────────────────────
print("\n📦 Chargement du dataset...")
df = pd.read_csv('dataset_difficultes_info.csv')
print(f"   ✓ {len(df)} élèves · {df.shape[1]-1} features")
print(f"   Distribution cible : {df['difficulte_future'].value_counts().to_dict()}")

# ─── 2. Préparation des données ──────────────────────────────
print("\n⚙️  Préparation des données...")

# Séparation features / cible
X = df.drop('difficulte_future', axis=1)
y = df['difficulte_future']

print(f"   Features : {list(X.columns)}")

# Split train/test stratifié (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train : {len(X_train)} | Test : {len(X_test)}")

# Normalisation (StandardScaler)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("   ✓ Normalisation StandardScaler appliquée")

# ─── 3. Définition des modèles ───────────────────────────────
print("\n🤖 Entraînement de 6 modèles ML...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'SVM':                 SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

# ─── 4. Entraînement et évaluation ──────────────────────────
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n   → {name}")

    # Entraînement
    model.fit(X_train_sc, y_train)
    trained_models[name] = model

    # Prédictions
    y_pred = model.predict(X_test_sc)

    # Métriques
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    cv_f1   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='f1').mean()
    cm      = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        'accuracy':  round(acc, 4),
        'f1':        round(f1, 4),
        'recall':    round(recall, 4),
        'precision': round(prec, 4),
        'cv_f1':     round(cv_f1, 4),
        'confusion_matrix': cm
    }

    print(f"     Accuracy={acc:.4f} | F1={f1:.4f} | Recall={recall:.4f} | CV-F1={cv_f1:.4f}")

# ─── 5. Sélection du meilleur modèle ────────────────────────
print("\n🏆 Sélection du meilleur modèle (critère : F1-Score)...")

# On priorise le recall (minimiser les faux négatifs)
best_name = max(results, key=lambda k: (results[k]['f1'], results[k]['recall']))
best_model = trained_models[best_name]
best_metrics = results[best_name]

print(f"   ✓ Meilleur modèle : {best_name}")
print(f"     F1={best_metrics['f1']:.4f} | Recall={best_metrics['recall']:.4f}")

# ─── 6. Sauvegarde ──────────────────────────────────────────
print("\n💾 Sauvegarde des fichiers...")

# Modèle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("   ✓ best_model.pkl sauvegardé")

# Scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ scaler.pkl sauvegardé")

# Résultats JSON
with open('model_results.json', 'w', encoding='utf-8') as f:
    json.dump({'results': results, 'best': best_name}, f, indent=2, ensure_ascii=False)
print("   ✓ model_results.json sauvegardé")

# ─── 7. Feature Importance (si Random Forest / Gradient Boosting) ──
if hasattr(best_model, 'feature_importances_'):
    fi = dict(zip(X.columns, best_model.feature_importances_.round(4)))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: -x[1]))
    print(f"\n📊 Top 5 features importantes :")
    for feat, imp in list(fi_sorted.items())[:5]:
        bar = '█' * int(imp * 100)
        print(f"   {feat:<35} {bar} {imp:.4f}")

# ─── 8. Rapport final ───────────────────────────────────────
print("\n" + "=" * 60)
print(f"  ✅ ENTRAÎNEMENT TERMINÉ — {best_name} sélectionné")
print("=" * 60)
print("\n  Lancez l'application : python app.py")
print("  Puis ouvrez : http://localhost:5000\n")
