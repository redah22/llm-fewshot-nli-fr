# Guide d'Utilisation Simplifié

## Commandes Essentielles

| Action | Commande | Note |
| :--- | :--- | :--- |
| **1. Préparer** | `python3 experiments/data_utils/setup_data.py` | À faire au début ou pour changer de dataset. |
| **2. CamemBERT** | `python3 experiments/fine_tuning/eval_camembert.py` | Tapez 1 (GQNLI) ou 2 (FraCaS). Durée : ~2 min. |
| **3. Gemini** | `python3 experiments/few_shot/eval_gemini_few_shot.py` | Nécessite API Key. Attention quotas. |

## FAQ Rapide

**Q: "Quota exceeded" avec Gemini ?**
R: C'est normal (limite gratuite). Le script attendra automatiquement. Laissez tourner ou relancez demain.

**Q: CamemBERT score bas ?**
R: Vérifiez que vous utilisez bien **GQNLI-FR** (180 exemples). FraCaS est trop petit pour avoir de bons résultats.

R: Dans le dossier `results/` (fichiers .json) ou affichés directement dans le terminal.

## 4. Nouveaux Datasets : DACCORD et RTE3-French

Les datasets DACCORD (1034 ex, contradiction vs compatibles) et RTE3-French (1600 ex, 3-way NLI) sont désormais disponibles.
1. Lancez `python3 experiments/data_utils/setup_data.py` et choisissez **4 (DACCORD)** ou **5 (RTE3-French)**.
2. Évaluez-les via `eval_camembert.py` ou `eval_flaubert.py` en choisissant l'option correspondante.

## 5. Pré-entraînement XNLI (Transfer Learning)

Pour obtenir de meilleurs résultats, vous pouvez d'abord entraîner FlauBERT sur un très grand dataset NLI (XNLI French), puis utiliser ce modèle "pré-entraîné" pour vos corpus plus petits (GQNLI, FraCaS).

**Étape A : Télécharger XNLI**
```bash
python3 experiments/data_utils/setup_xnli.py
```

**Étape B : Entraîner sur XNLI**
```bash
python3 experiments/fine_tuning/train_flaubert_xnli.py
```
*(Vous pourrez choisir la taille du dataset : 10k, 50k, complet. Le modèle final sera dans `models/flaubert_xnli_fr`)*

**Étape C : Transfer Learning vers le dataset final**
1. Lancez `python3 experiments/fine_tuning/eval_flaubert.py`
2. Choisissez le **Mode 1** (Train+Eval) ou le **Mode 3** (Same Dataset)
3. À la question "Quel modèle de base...", choisissez **2** et entrez : `models/flaubert_xnli_fr`
