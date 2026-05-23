# DiagnoLearn — Système de Prédiction des Difficultés d'Apprentissage
## PPE · CRMEF Maroc · Machine Learning + Flask

---

## 🎯 Description

Système intelligent de **prédiction précoce** des difficultés d'apprentissage en informatique au collège au Maroc, basé sur le Machine Learning.

**Fonctionnalités :**
- Formulaire diagnostique multi-étapes (profil, environnement numérique, académique, comportement)
- Comparaison de 6 modèles ML et sélection automatique du meilleur
- Affichage du niveau de risque (🟢 Faible / 🟡 Modéré / 🔴 Élevé)
- Recommandations pédagogiques personnalisées (IA Gemini ou base experte)
- Tableau de bord de comparaison des modèles

---

## 🏗️ Architecture

```
diagnolearn/
├── app.py                    # Serveur Flask (backend + API)
├── train_models.py           # Script d'entraînement ML
├── dataset_difficultes_info.csv  # Dataset (280 élèves, 19 features)
├── best_model.pkl            # Modèle Random Forest sérialisé
├── scaler.pkl                # StandardScaler sérialisé
├── requirements.txt          # Dépendances Python
└── templates/
    ├── index.html            # Page principale (formulaire + résultats)
    └── dashboard.html        # Page comparaison modèles ML
```

---

## 🚀 Installation et lancement

### 1. Prérequis
```bash
Python 3.8+
pip
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. (Optionnel) Réentraîner les modèles
```bash
python train_models.py
```

### 4. Lancer l'application
```bash
python app.py
```

### 5. Ouvrir dans le navigateur
```
http://localhost:5000
```

---

## 🤖 Recommandations IA (Gemini)

Pour activer les recommandations via Google Gemini API :

1. Obtenir une clé API sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Définir la variable d'environnement :

```bash
# Linux/Mac
export GEMINI_API_KEY="votre_cle_api"

# Windows
set GEMINI_API_KEY=votre_cle_api
```

Sans clé API, le système utilise des recommandations expertes prédéfinies.

---

## 📊 Modèles ML comparés

| Modèle | Accuracy | F1-Score | Recall | CV F1 |
|--------|----------|----------|--------|-------|
| Logistic Regression | 80.4% | 80.7% | 79.3% | 73.4% |
| Decision Tree | 62.5% | 63.2% | 62.1% | 67.8% |
| KNN | 82.1% | 81.5% | 75.9% | 74.3% |
| SVM | 85.7% | 85.7% | 82.8% | 75.4% |
| **Random Forest ✓** | **85.7%** | **86.7%** | **89.7%** | **76.5%** |
| Gradient Boosting | 85.7% | 85.7% | 82.8% | 70.2% |

**Random Forest** sélectionné pour son recall maximal (89.7%) — minimise les faux négatifs, crucial pour ne pas manquer un élève en difficulté.

---

## 📋 Dataset — Variables utilisées

### Profil (4 variables)
- `genre` : 0=Garçon / 1=Fille
- `age` : 11–17 ans
- `redoublement` : 0=Non / 1=Oui
- `zone` : 0=Rural / 1=Urbain

### Environnement numérique (4 variables)
- `acces_ordinateur`, `acces_internet` : 0/1
- `frequence_utilisation_pc` : 0–4
- `connaissance_pc` : 0–4

### Académique (2 variables)
- `moy_math`, `moy_generale` : 0–20

### Test diagnostique (4 variables)
- `score_logique`, `score_algo` : 0–10
- `temps_resolution` : minutes
- `nb_erreurs` : entier

### Comportement & Motivation (5 variables)
- `concentration`, `autonomie`, `participation` : 1–5
- `motivation`, `apprehension` : 1–5

### Variable cible
- `difficulte_future` : 0=Non / 1=Oui

---

## 🔒 Éthique et protection des données

- Aucune donnée personnelle identifiable stockée
- Conformité à la loi marocaine n° 09-08
- Le système est un outil d'aide à la décision, non un juge

---

## 👨‍💻 PPE — CRMEF Maroc
