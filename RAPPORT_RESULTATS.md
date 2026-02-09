# Rapport de Résultats : CamemBERT vs Gemini

Ce rapport synthétise les expériences menées sur l'inférence en langage naturel (NLI) en français.

## 1. Protocole Expérimental
Nous comparons deux approches sur deux datasets NLI français (très peu de données) :
- **CamemBERT (Fine-Tuning)** : Ré-entraînement complet du modèle sur les données Train.
- **Gemini 2.5 (Few-Shot)** : Prompting avec exemples (0, 1, 3, 5 exemples) sans ré-entraînement.

**Données utilisées :**
- **FraCaS GQ** : 48 train / 16 val (Très petit)
- **GQNLI-FR** : 180 train / 60 val (Petit)

## 2. Résultats CamemBERT (Fine-Tuning)
Le modèle a été entraîné sur 10 epochs.

| Dataset | Taille Train | Précision (Validation) | Observation |
| :--- | :--- | :--- | :--- |
| **FraCaS** | 48 ex | **50.00%** | Mieux que le hasard (33%), mais manque de données. |
| **GQNLI-FR** | 180 ex | **78.33%** | **Excellent résultat.** Le modèle apprend très bien avec un peu plus de données. |

✅ **Conclusion :** Le fine-tuning "Small Data" fonctionne très bien si on pousse le nombre d'epochs (10).

## 3. Résultats FlauBERT (Fine-Tuning)
Protocole identique à CamemBERT.

| Dataset | Taille Train | Précision (Validation) | Comparaison CamemBERT |
| :--- | :--- | :--- | :--- |
| **FraCaS** | 48 ex | **50.00%** | **Équivalent au hasard**. La performance chute avec les "optimisations". |
| **GQNLI-FR** | 180 ex | **41.67%** | **Pire que le hasard**. Échec complet d'apprentissage (Learning Rate 1e-5, 20 epochs). |

➡️ **Analyse Final :** FlauBERT n'est pas adapté pour ce dataset « Small Data ». Contrairement à CamemBERT qui performe immédiatement (78%), FlauBERT est instable et s'effondre vers la classe majoritaire dès qu'on tente de le stabiliser. **Nous abandonnons FlauBERT pour la suite.**

## 4. Résultats Gemini (Few-Shot)
Les tests sont impactés par des limitations techniques de l'API Gratuite.

| Config | Précision (GQNLI) | Problème rencontré |
| :--- | :--- | :--- |
| **0-shot** | **90.00%** | Excellent score de base (le modèle connait déjà la tâche). |
| **Few-shot** | **~40.00%** | **Chute anormale.** Causée par les erreurs de quota API (429). |

❌ **Problème identifié :** 
L'API gratuite limite à **20 requêtes/jour**. Dès que l'on dépasse, l'API renvoie une erreur. Le script comptait ces erreurs comme des "mauvaises réponses" (label neutre par défaut), ce qui a écrasé le score artificiellement.

**Solution déjà implémentée :** Un mécanisme de "Retry" (attendre et réessayer) a été ajouté au script pour contourner ce problème lors des prochains tests.

## 5. Conclusion Générale
- **CamemBERT est robuste et fiable** (78%) pour ce projet, même avec peu de données.
- **FlauBERT est instable** (échec sur GQNLI, mais bon sur FraCaS).
- **Gemini est potentiellement meilleur (90% en 0-shot)**, mais son évaluation est difficile sans payer l'API à cause des quotas très restrictifs.
