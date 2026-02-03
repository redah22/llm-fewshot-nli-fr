# Guide d'Utilisation Simplifié

## Commandes Essentielles

| Action | Commande | Note |
| :--- | :--- | :--- |
| **1. Préparer** | `python3 experiments/setup_data.py` | À faire au début ou pour changer de dataset. |
| **2. CamemBERT** | `python3 experiments/eval_camembert.py` | Tapez 1 (GQNLI) ou 2 (FraCaS). Durée : ~2 min. |
| **3. Gemini** | `python3 experiments/eval_gemini_few_shot.py` | Nécessite API Key. Attention quotas. |

## FAQ Rapide

**Q: "Quota exceeded" avec Gemini ?**
R: C'est normal (limite gratuite). Le script attendra automatiquement. Laissez tourner ou relancez demain.

**Q: CamemBERT score bas ?**
R: Vérifiez que vous utilisez bien **GQNLI-FR** (180 exemples). FraCaS est trop petit pour avoir de bons résultats.

**Q: Où sont les résultats ?**
R: Dans le dossier `results/` (fichiers .json) ou affichés directement dans le terminal.
