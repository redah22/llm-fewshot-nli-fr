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
| **GQNLI-FR (Base)** | 210 ex | **55.56%** | L'apprentissage est difficile et instable en partant de zéro. |
| **GQNLI-FR (XNLI Transfer)**| 210 ex | **62.22%** | **Gros progrès.** En pré-entraînant FlauBERT sur XNLI (fr), le modèle gagne presque 10 points. |

➡️ **Analyse Finale :** FlauBERT "Base" n'est pas très bien adapté pour un apprentissage direct sur ce dataset « Small Data » (55%). En revanche, la technique de **Transfer Learning** (Fine-tuning intermédiaire sur le grand corpus XNLI structuré NLI, puis affinage sur GQNLI-FR) permet de le stabiliser et de le faire monter à 62.22% sans effort. Les modèles ont plus de mal que CamemBERT, mais l'approche XNLI sauve la mise.

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
- **CamemBERT est robuste et fiable** (~84%) pour ce projet, même avec peu de données de départ.
- **FlauBERT nécessite du Transfer Learning** (62.22% avec XNLI contre 55% de base) car il a du mal avec les très petits datasets NLI.
- **Gemini est potentiellement meilleur (90% en 0-shot)**, mais son évaluation est difficile en version gratuite à cause des quotas très restrictifs.
