
# TER M1 : Inférence en Langue Naturelle (NLI)

Projet de comparaison entre Fine-Tuning (CamemBERT) et Few-Shot Learning (Gemini) sur des tâches d'inférence en français.

## 📁 Structure
- `data/` : Données (GQNLI-FR et FraCaS)
- `experiments/` : Scripts Python
- `models/` : Modèles entraînés
- `results/` : Scores et logs

## 🚀 Utilisation Rapide

### 1. Installation

**Prérequis :**
- Python 3.9+
- Clé API Google (pour Gemini) : Créez un fichier `.env` à la racine contenant `GOOGLE_API_KEY=votre_clé`.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fine-Tuning CamemBERT (Recommandé)
Entraîne le modèle sur vos données (10 epochs).
```bash
python3 experiments/setup_data.py   # Étape 1 : Préparer les données
python3 experiments/eval_camembert.py   # Étape 2 : Entraîner
```
*Choisissez "GQNLI-FR" (1) pour les meilleurs résultats.*

### 3. Few-Shot Gemini
Test avec l'API Google (attention aux quotas).
```bash
python3 experiments/eval_gemini_few_shot.py
```

## 📊 Résultats Clés
Voir `RAPPORT_RESULTATS.md` pour le détail.
- **CamemBERT (GQNLI)** : 78% ✅
- **Gemini (0-shot)** : 90%
