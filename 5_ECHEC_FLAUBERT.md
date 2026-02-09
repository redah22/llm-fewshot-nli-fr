# Rapport d'Analyse : Échec de FlauBERT sur GQNLI-FR

Ce document détaille les expériences menées avec le modèle **FlauBERT** (Base Cased) sur la tâche d'inférence en langue naturelle (NLI), comparé à **CamemBERT**.

## 1. Contexte
Dans le cadre du TER, nous cherchons à fine-tuner des modèles de langue sur des datasets NLI très petits ("Small Data") :
- **GQNLI-FR** : 180 exemples d'entraînement.
- **FraCaS** : 48 exemples d'entraînement.

CamemBERT a obtenu d'excellents résultats (**78%** sur GQNLI-FR). Nous avons voulu tester si FlauBERT pouvait faire mieux ou offrir une alternative.

## 2. Expériences & Résultats

### Tentative 1 : Configuration Standard (Baseline)
- **Paramètres** : Learning Rate 2e-5, 10 Epochs, Batch Size 8.
- **Résultats** :
  - **GQNLI-FR** : **50.00%** (Validation).
  - **FraCaS** : **68.75%** (Validation).
- **Observation** : Le score sur GQNLI est suspect (proche de la classe majoritaire). Le score sur FraCaS semble être un "coup de chance" dû à la volatilité sur un si petit set (16 exemples).

### Tentative 2 : Stabilisation & Optimisation
Pour contrer l'instabilité, nous avons ajusté le protocole :
1.  **Lower Learning Rate** : 1e-5 (pour éviter de "casser" les poids pré-entrainés).
2.  **More Epochs** : 20 (pour laisser le temps de converger doucement).
3.  **Tokenization Stricte** : Ajout explicite des `token_type_ids` pour séparer Prémisse/Hypothèse (spécifique à l'architecture XLM de FlauBERT).

- **Résultats** :
  - **GQNLI-FR** : **41.67%** (Validation).
  - **FraCaS** : **50.00%** (Validation).

## 3. Analyse de l'Échec

Le passage à une configuration plus "propre" et stable a paradoxalement dégradé les performances. Cela révèle que :

1.  **Instabilité Structurelle** : FlauBERT semble beaucoup moins robuste que CamemBERT dans un régime "Few-Shot / Small Data". Il n'arrive pas à extraire de signal stable.
2.  **Effondrement (Collapse)** : Avec un learning rate bas, le modèle ne converge pas vers une solution utile mais s'effondre vers une prédiction aléatoire ou majoritaire.
3.  **Complexité** : L'architecture de FlauBERT (XLM) est parfois plus difficile à fine-tuner que celle de CamemBERT (RoBERTa) sur des tâches de classification de phrases simples.

## 4. Conclusion pour le Mémoire
**Nous abandonnons FlauBERT au profit de CamemBERT.** 

Cet échec est un résultat scientifique intéressant en soi : il démontre la supériorité de la robustesse de CamemBERT pour des scénarios à très faibles ressources en français.
