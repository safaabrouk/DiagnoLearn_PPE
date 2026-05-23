"""
=============================================================================
SYSTÈME INTELLIGENT DE PRÉDICTION DES DIFFICULTÉS D'APPRENTISSAGE
PPE - CRMEF Maroc | Machine Learning + Flask + API IA
=============================================================================
Architecture:
  - Backend  : Flask (Python)
  - ML Model : Random Forest (meilleur modèle après comparaison)
  - Frontend : HTML/CSS/JS (templates Jinja2)
  - IA Reco  : Google Gemini API (recommandations pédagogiques)
=============================================================================
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import urllib.request
import urllib.error
import os

# ─── Initialisation Flask ───────────────────────────────────────────────────
app = Flask(__name__)

# ─── Chargement du modèle et du scaler ──────────────────────────────────────
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Résultats de comparaison des modèles (générés lors de l'entraînement)
MODEL_RESULTS = {
    "Logistic Regression": {"accuracy": 0.8036, "f1": 0.8070, "recall": 0.7931, "cv_f1": 0.7339},
    "Decision Tree":       {"accuracy": 0.6250, "f1": 0.6316, "recall": 0.6207, "cv_f1": 0.6777},
    "KNN":                 {"accuracy": 0.8214, "f1": 0.8148, "recall": 0.7586, "cv_f1": 0.7432},
    "SVM":                 {"accuracy": 0.8571, "f1": 0.8571, "recall": 0.8276, "cv_f1": 0.7543},
    "Random Forest":       {"accuracy": 0.8571, "f1": 0.8667, "recall": 0.8966, "cv_f1": 0.7650},
    "Gradient Boosting":   {"accuracy": 0.8571, "f1": 0.8571, "recall": 0.8276, "cv_f1": 0.7021},
}

# Importances des features (depuis l'entraînement Random Forest)
FEATURE_IMPORTANCES = {
    "score_algo": 0.1778,
    "moy_generale": 0.1296,
    "moy_math": 0.1158,
    "score_logique": 0.1104,
    "temps_resolution": 0.0926,
    "nb_erreurs": 0.0812,
    "autonomie": 0.0654,
    "concentration": 0.0589,
    "motivation": 0.0512,
    "participation": 0.0498,
    "apprehension": 0.0289,
    "connaissance_pc": 0.0198,
    "frequence_utilisation_pc": 0.0112,
    "age": 0.0067,
    "genre": 0.0005,
    "redoublement": 0.0000,
    "zone": 0.0000,
    "acces_ordinateur": 0.0000,
    "acces_internet": 0.0000,
}

# Ordre des features (doit correspondre exactement à l'ordre d'entraînement)
FEATURE_ORDER = [
    'genre', 'age', 'redoublement', 'zone',
    'acces_ordinateur', 'acces_internet', 'frequence_utilisation_pc', 'connaissance_pc',
    'moy_math', 'moy_generale',
    'score_logique', 'score_algo', 'temps_resolution', 'nb_erreurs',
    'concentration', 'autonomie', 'motivation', 'apprehension', 'participation'
]


# ─── Route principale : formulaire de saisie ────────────────────────────────
@app.route('/')
def index():
    """Page d'accueil avec le formulaire de diagnostic"""
    return render_template('index.html')


# ─── Route : tableau de bord des modèles ────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    """Page de synthèse pédagogique"""
    return render_template('dashboard.html', page='dashboard', results=MODEL_RESULTS, importances=FEATURE_IMPORTANCES)


@app.route('/models')
def models():
    """Page de comparaison des modèles ML"""
    return render_template('dashboard.html', page='models', results=MODEL_RESULTS, importances=FEATURE_IMPORTANCES)


@app.route('/about')
def about():
    """Page de présentation du projet PPE"""
    return render_template('about.html')


# ─── Route API : prédiction ──────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Reçoit les données du formulaire, effectue la prédiction ML
    et retourne le résultat + probabilité.

    Niveaux de risque:
      - prob < 0.35  → Faible risque (🟢)
      - prob < 0.65  → Risque modéré (🟡)
      - prob >= 0.65 → Risque élevé  (🔴)
    """
    try:
        data = request.get_json()

        # Construction du DataFrame avec l'ordre correct des features
        row = {}
        for feat in FEATURE_ORDER:
            val = data.get(feat)
            if val is None:
                return jsonify({'error': f'Champ manquant: {feat}'}), 400
            row[feat] = float(val)

        # Mise à l'échelle et prédiction (DataFrame pour conserver les noms de colonnes)
        X = pd.DataFrame([row])[FEATURE_ORDER]
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])  # Prob de difficulté

        # Détermination du niveau de risque
        if probability < 0.35:
            risk_level = "low"
            risk_label = "Faible Risque"
            risk_color = "#10b981"
        elif probability < 0.65:
            risk_level = "medium"
            risk_label = "Risque Modéré"
            risk_color = "#f59e0b"
        else:
            risk_level = "high"
            risk_label = "Risque Élevé"
            risk_color = "#ef4444"

        # Identification des facteurs de risque principaux
        risk_factors = _identify_risk_factors(data)

        return jsonify({
            'prediction': prediction,
            'probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'model_used': 'Random Forest'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Route API : recommandations IA ─────────────────────────────────────────
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Génère des recommandations pédagogiques personnalisées via Google Gemini API.
    Si la clé API n'est pas configurée, retourne des recommandations par défaut.
    """
    try:
        data = request.get_json()
        risk_level = data.get('risk_level', 'medium')
        risk_factors = data.get('risk_factors', [])
        student_profile = data.get('profile', {})

        # ── Tentative d'appel à Gemini API ──────────────────────────────────
        gemini_key = os.environ.get('GEMINI_API_KEY', '')

        if gemini_key:
            recommendations = _get_gemini_recommendations(
                gemini_key, risk_level, risk_factors, student_profile
            )
        else:
            # Recommandations par défaut (fallback sans API)
            recommendations = _get_default_recommendations(risk_level, risk_factors, student_profile)

        return jsonify({'recommendations': recommendations, 'source': 'gemini' if gemini_key else 'default'})

    except Exception as e:
        return jsonify({'error': str(e), 'recommendations': _get_default_recommendations('medium', [], {})}), 500


# ─── Fonctions utilitaires ───────────────────────────────────────────────────

def _identify_risk_factors(data):
    """
    Identifie les principaux facteurs de risque à partir des données de l'élève.
    Retourne une liste de dictionnaires {factor, value, severity}.
    """
    factors = []

    # Scores académiques
    if float(data.get('score_algo', 10)) < 4:
        factors.append({'factor': 'Score algorithmique faible', 'value': data.get('score_algo'), 'severity': 'high'})
    elif float(data.get('score_algo', 10)) < 6:
        factors.append({'factor': 'Score algorithmique moyen', 'value': data.get('score_algo'), 'severity': 'medium'})

    if float(data.get('score_logique', 10)) < 4:
        factors.append({'factor': 'Raisonnement logique faible', 'value': data.get('score_logique'), 'severity': 'high'})

    if float(data.get('moy_math', 20)) < 8:
        factors.append({'factor': 'Moyenne mathématiques basse', 'value': data.get('moy_math'), 'severity': 'high'})

    if float(data.get('moy_generale', 20)) < 8:
        factors.append({'factor': 'Moyenne générale basse', 'value': data.get('moy_generale'), 'severity': 'high'})

    # Comportement
    if int(data.get('motivation', 3)) <= 2:
        factors.append({'factor': 'Motivation faible', 'value': data.get('motivation'), 'severity': 'medium'})

    if int(data.get('concentration', 3)) <= 2:
        factors.append({'factor': 'Concentration insuffisante', 'value': data.get('concentration'), 'severity': 'medium'})

    if int(data.get('apprehension', 1)) >= 4:
        factors.append({'factor': 'Forte appréhension face à l\'informatique', 'value': data.get('apprehension'), 'severity': 'medium'})

    # Environnement numérique
    if int(data.get('acces_ordinateur', 1)) == 0:
        factors.append({'factor': 'Pas d\'ordinateur à domicile', 'value': 'Non', 'severity': 'medium'})

    if float(data.get('temps_resolution', 10)) > 25:
        factors.append({'factor': 'Temps de résolution très lent', 'value': f"{data.get('temps_resolution')} min", 'severity': 'medium'})

    if int(data.get('nb_erreurs', 0)) >= 6:
        factors.append({'factor': 'Nombre élevé d\'erreurs', 'value': data.get('nb_erreurs'), 'severity': 'high'})

    if int(data.get('redoublement', 0)) == 1:
        factors.append({'factor': 'Élève redoublant', 'value': 'Oui', 'severity': 'low'})

    return factors[:5]  # Retourner les 5 facteurs les plus importants


def _get_gemini_recommendations(api_key, risk_level, risk_factors, profile):
    """
    Appelle l'API Google Gemini pour générer des recommandations pédagogiques.
    """
    risk_map = {
        'low': 'faible risque de difficultés',
        'medium': 'risque modéré de difficultés',
        'high': 'risque élevé de difficultés'
    }

    factors_text = '\n'.join([f"- {f['factor']}" for f in risk_factors]) if risk_factors else "Aucun facteur critique identifié"

    prompt = f"""Tu es un expert en pédagogie de l'informatique au collège marocain.
Un élève présente un {risk_map.get(risk_level, 'risque modéré')} dans le cours d'informatique.

Facteurs identifiés:
{factors_text}

Génère exactement 4 recommandations concrètes et actionnables pour l'enseignant,
adaptées au contexte éducatif marocain (collège, ressources limitées, programme national).
Format JSON: [{{"titre": "...", "description": "...", "priorite": "haute/moyenne/faible"}}]
Réponds UNIQUEMENT avec le JSON, sans texte supplémentaire."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.loads(response.read().decode('utf-8'))
        text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text)


def _get_default_recommendations(risk_level, risk_factors, profile):
    """
    Recommandations pédagogiques par défaut selon le niveau de risque.
    Utilisées en fallback si Gemini API n'est pas disponible.
    """
    # Extraire les noms des facteurs pour personnaliser
    factor_names = [f.get('factor', '') for f in risk_factors] if risk_factors else []

    if risk_level == 'low':
        return [
            {
                "titre": "Enrichissement progressif des apprentissages",
                "description": "Proposer des mini-projets créatifs et gradués afin de consolider les acquis sans surcharge : création d'un quiz, mise en forme d'une page simple, ou résolution d'un défi algorithmique court. L'objectif est de maintenir l'engagement tout en développant l'autonomie.",
                "priorite": "moyenne"
            },
            {
                "titre": "Valorisation par le tutorat entre pairs",
                "description": "Confier ponctuellement à l'élève un rôle d'appui auprès d'un camarade durant les travaux pratiques. Cette posture renforce la verbalisation des procédures, la confiance en soi et la maîtrise des notions déjà acquises.",
                "priorite": "faible"
            },
            {
                "titre": "Défis courts pour stimuler la motivation",
                "description": "Intégrer des défis optionnels adaptés au collège, avec des critères de réussite clairs et un retour rapide. Les activités doivent encourager l'exploration, la persévérance et l'amélioration continue des compétences numériques.",
                "priorite": "faible"
            },
            {
                "titre": "Suivi positif et objectifs d'approfondissement",
                "description": "Prévoir un point de suivi léger toutes les deux à trois semaines pour valoriser les réussites, identifier les nouvelles compétences à développer et maintenir un lien pédagogique motivant avec l'apprenant.",
                "priorite": "faible"
            }
        ]
    elif risk_level == 'medium':
        recs = [
            {
                "titre": "Remédiation ciblée en logique algorithmique",
                "description": "Mettre en place des séances courtes centrées sur les séquences, conditions et boucles, avec manipulation visuelle, exemples contextualisés et exercices progressifs. Chaque séance doit se terminer par une tâche simple réussissable pour renforcer la confiance.",
                "priorite": "haute"
            },
            {
                "titre": "Ressources accessibles et travail guidé",
                "description": "Fournir des fiches structurées, exercices hors ligne et consignes étape par étape pour limiter l'impact des inégalités d'accès numérique. Les supports doivent permettre à l'élève de reprendre les notions essentielles à son rythme.",
                "priorite": "haute"
            },
            {
                "titre": "Projets signifiants pour renforcer l'engagement",
                "description": "Utiliser des situations proches de l'élève, comme un quiz culturel, une affiche numérique ou une mini-page sur un thème local. Donner du sens aux activités augmente l'implication et favorise la continuité de l'effort.",
                "priorite": "moyenne"
            },
            {
                "titre": "Accompagnement en binôme structuré",
                "description": "Associer l'élève à un pair complémentaire avec des rôles explicites : lecteur de consigne, opérateur, vérificateur. Cette organisation sécurise l'activité, réduit l'appréhension et développe progressivement l'autonomie.",
                "priorite": "moyenne"
            },
            {
                "titre": "Suivi personnalisé des progrès",
                "description": "Construire une grille de suivi simple avec deux ou trois indicateurs : compréhension des consignes, précision des réponses et persévérance. Les progrès observés doivent être communiqués à l'élève pour soutenir sa motivation.",
                "priorite": "moyenne"
            }
        ]
        return recs
    else:  # high risk
        return [
            {
                "titre": "Intervention prioritaire sur les prérequis",
                "description": "Commencer par des activités de raisonnement très guidées : classement, suites logiques, repérage d'étapes et résolution de petits problèmes. Avant de passer à la programmation, vérifier que l'élève comprend les notions de séquence, d'ordre et de condition.",
                "priorite": "haute"
            },
            {
                "titre": "Plan d'accompagnement personnalisé",
                "description": "Établir un plan individuel avec objectifs hebdomadaires très précis, tâches courtes, critères de réussite visibles et points d'étape réguliers. L'élève doit savoir ce qui est attendu, ce qu'il réussit déjà et quelle compétence travailler ensuite.",
                "priorite": "haute"
            },
            {
                "titre": "Réduction de l'appréhension face à l'informatique",
                "description": "Introduire des activités débranchées avant l'ordinateur afin de dédramatiser les erreurs. Utiliser la reformulation, la démonstration pas à pas et le droit à l'essai pour installer un climat de sécurité pédagogique.",
                "priorite": "haute"
            },
            {
                "titre": "Coordination avec l'équipe pédagogique",
                "description": "Partager les observations avec l'équipe pédagogique et la famille afin d'assurer une continuité de suivi. Si les difficultés logiques ou mathématiques sont importantes, prévoir un appui complémentaire coordonné avec les autres disciplines.",
                "priorite": "haute"
            },
            {
                "titre": "Engagement progressif de l'apprenant",
                "description": "Fixer des micro-objectifs atteignables, célébrer les progrès concrets et proposer des tâches courtes où l'élève peut réussir rapidement. Cette progression aide à reconstruire l'engagement et à limiter l'évitement.",
                "priorite": "haute"
            }
        ]


# ─── Point d'entrée ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
