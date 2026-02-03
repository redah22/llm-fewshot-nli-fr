# 📘 Guide Complet du Processus - TER French NLI

## 🎯 Vue d'Ensemble

Ce guide vous accompagne du début à la fin de votre projet TER.

## 📅 Processus en 5 Étapes

### Étape 1: Setup & Exploration (1 semaine)

**Objectif**: Préparer l'environnement et comprendre les données

#### Actions:
```bash
# 1. Installation
source venv/bin/activate
pip install -r requirements.txt

# 2. Tester le chargement
python3 test_fracas.py
```

#### Notebook:
📓 **`notebooks/01_fracas_exploration.ipynb`**
- Charger FraCaS
- Diviser en train/val/test (60/20/20)
- Analyser les statistiques
- Visualiser les distributions
- **⚠️ Sauvegarder les splits!**

**Livrables**:
- [x] Environnement installé
- [x] Dataset exploré
- [x] Splits créés et sauvegardés
- [x] Statistiques documentées

---

### Étape 2: Baseline & Few-Shot sur VALIDATION (1-2 semaines)

**Objectif**: Tester different nombres de few-shot **sur validation**

> **⚠️ CRITIQUE**: On utilise VALIDATION, PAS test!

#### Pourquoi Validation?

```python
# ✅ CORRECT - Comparer sur validation
train_data = load_from_disk('data/processed/fracas_split')['train']
val_data = load_from_disk('data/processed/fracas_split')['validation']

# Exemples few-shot depuis train
few_shot_5 = train_data.select(range(5))

# Tester 0-shot sur validation
val_acc_0shot = evaluate_zero_shot(model, val_data)
print(f"0-shot (Val): {val_acc_0shot:.2f}")

# Tester 5-shot sur validation
val_acc_5shot = evaluate_few_shot(model, val_data, few_shot_5)
print(f"5-shot (Val): {val_acc_5shot:.2f}")

# ✅ Pas de leakage car on n'a pas touché test!
```

#### Notebook ou Script:
📓 **`notebooks/02_few_shot_validation.ipynb`** (développement)  
OU  
🐍 **`scripts/run_few_shot.py`** (expériences reproductibles)

```python
# Exemple de comparaison
num_shots = [0, 1, 3, 5, 10]
val_results = []

for n in num_shots:
    # Few-shot depuis train
    examples = train_data.select(range(n)) if n > 0 else None
    
    # Évaluer sur validation
    val_acc = evaluate(model, val_data, examples)
    val_results.append(val_acc)
    
    print(f"{n}-shot: Validation accuracy = {val_acc:.2%}")

# Résultat: vous voyez l'amélioration sans toucher test!
```

**Livrables**:
- [x] Résultats 0-shot sur validation
- [x] Résultats few-shot (1, 3, 5, 10) sur validation  
- [x] Graphique comparatif
- [x] Meilleur nombre de shots identifié

---

### Étape 3: Fine-Tuning (Optionnel, 1-2 semaines)

**Objectif**: Entraîner CamemBERT sur train, monitorer sur validation

#### Script:
🐍 **`scripts/fine_tune_camembert.py`**

```python
from transformers import Trainer, TrainingArguments

# Configuration
training_args = TrainingArguments(
    output_dir='checkpoints/fracas',
    num_train_epochs=3,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# Entraîner
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,      # ✅ Train
    eval_dataset=val_data,         # ✅ Validation
    # PAS test_data!
)

trainer.train()

# Voir les résultats sur validation
val_results = trainer.evaluate(val_data)
print(f"Validation accuracy: {val_results['eval_accuracy']:.2%}")
```

**Livrables**:
- [x] Modèle entraîné
- [x] Courbes d'apprentissage (train/val)
- [x] Meilleur checkpoint sauvegardé
- [x] Performance sur validation

---

### Étape 4: Analyse des Résultats sur VALIDATION (1 semaine)

**Objectif**: Comparer toutes les approches sur validation

#### Notebook:
📓 **`notebooks/03_results_analysis.ipynb`**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Compiler les résultats sur VALIDATION
results = pd.DataFrame({
    'Approche': ['0-shot', '1-shot', '3-shot', '5-shot', '10-shot', 'Fine-tuned'],
    'Val_Accuracy': [0.45, 0.52, 0.61, 0.68, 0.71, 0.75]  # Vos résultats
})

# Visualiser
plt.figure(figsize=(10, 6))
plt.bar(results['Approche'], results['Val_Accuracy'])
plt.ylabel('Validation Accuracy')
plt.title('Comparaison des Approches (sur Validation)')
plt.ylim(0, 1)
plt.show()

# ✅ Tout est fait sur validation, test reste intact!
```

**Livrables**:
- [x] Tableau comparatif
- [x] Graphiques
- [x] Analyse d'erreurs
- [x] Meilleure approche identifiée

---

### Étape 5: Évaluation Finale sur TEST (1 jour)

**Objectif**: Rapporter les résultats finaux

> **⚠️ UNE SEULE FOIS! Ne plus modifier après!**

#### Script:
🐍 **`scripts/final_evaluation.py`**

```python
from datasets import DatasetDict

# Charger les splits
fracas = DatasetDict.load_from_disk('data/processed/fracas_split')
test_data = fracas['test']

# Charger le meilleur modèle (décidé sur validation)
best_model = load_best_model('checkpoints/fracas/best/')

# ✅ Évaluation finale sur TEST (première et dernière fois!)
test_results = evaluate(best_model, test_data)

print("="*60)
print("🎯 RÉSULTATS FINAUX")
print("="*60)
print(f"Test Accuracy: {test_results['accuracy']:.2%}")
print(f"Test F1-Macro: {test_results['f1_macro']:.2%}")
print("="*60)
print("⚠️  Ces résultats sont à rapporter dans le TER")
print("⚠️  NE PLUS MODIFIER le modèle!")
print("="*60)
```

**Livrables**:
- [x] Résultats test finaux
- [x] Rapport TER
- [x] (Optionnel) Publication

---

## 📊 Réponse à Votre Question

### "On peut comparer avant/après few-shot?"

**OUI, MAIS sur VALIDATION!**

```python
# ✅ CORRECT - Développement sur validation
val_data = load_validation()

# Baseline (0-shot)
acc_0 = evaluate_zero_shot(model, val_data)  # Ex: 45%

# Few-shot (5 exemples)
acc_5 = evaluate_few_shot(model, val_data, n=5)  # Ex: 68%

print(f"Amélioration: +{acc_5 - acc_0:.0%}")  # Ex: +23%

# ✅ Pas de leakage! On compare sur validation.
```

```python
# ❌ INCORRECT - NE PAS FAIRE ÇA!
test_data = load_test()  # ❌ Test!

acc_0 = evaluate_zero_shot(model, test_data)  # ❌ Leakage!
acc_5 = evaluate_few_shot(model, test_data, n=5)  # ❌  Leakage!

# ❌ Résultats biaisés car vous avez vu test!
```

### Workflow Correct:

```
1. DÉVELOPPEMENT (train + validation)
   ├─ Tester 0-shot sur validation → 45%
   ├─ Tester few-shot sur validation → 68%
   └─ Choisir le meilleur (ex: 5-shot)

2. ÉVALUATION FINALE (test - UNE FOIS)
   └─ Tester 5-shot sur test → 65% (résultat final)
```

**Pourquoi validation < test parfois?**
- Normal! Variance entre splits
- L'important: pas de leakage
- Résultats test = résultats officiels

---

## 🗂️ Organisation des Fichiers

```
TER_M1/
├── notebooks/
│   ├── 01_fracas_exploration.ipynb      # Étape 1
│   ├── 02_few_shot_validation.ipynb     # Étape 2
│   └── 03_results_analysis.ipynb        # Étape 4
│
├── scripts/
│   ├── test_fracas.py                   # Test rapide
│   ├── run_few_shot.py                  # Expériences few-shot
│   ├── fine_tune_camembert.py           # Fine-tuning
│   └── final_evaluation.py              # Évaluation finale
│
├── data/
│   └── processed/
│       └── fracas_split/                # Splits sauvegardés
│           ├── train/
│           ├── validation/
│           └── test/
│
└── results/
    ├── few_shot_validation.json         # Résultats validation
    └── final_test_results.json          # Résultats test
```

---

## ✅ Checklist Complète

### Étape 1: Setup
- [x] Environnement installé
- [x] FraCaS chargé
- [x] Notebook exploration exécuté
- [x] Splits créés et sauvegardés

### Étape 2: Validation
- [ ] 0-shot testé sur validation
- [ ] Few-shot testé sur validation (1, 3, 5, 10)
- [ ] Graphique comparatif créé
- [ ] Meilleur n_shots choisi

### Étape 3: Fine-Tuning (optionnel)
- [ ] Modèle entraîné sur train
- [ ] Monitored sur validation
- [ ] Checkpoint sauvegardé

### Étape 4: Analyse
- [ ] Résultats compilés
- [ ] Figures créées
- [ ] Analyse d'erreurs faite

### Étape 5: Test Final
- [ ] Modèle finalisé
- [ ] Test évalué (UNE FOIS)
- [ ] Résultats rapportés
- [ ] Rapport TER rédigé

---

## 🎯 En Résumé

1. **Explorez** avec `01_fracas_exploration.ipynb`
2. **Développez** sur validation (comparez 0-shot vs few-shot)
3. **Analysez** les résultats validation
4. **Évaluez** sur test (une fois!)
5. **Rapportez** les résultats test

**Test = compétition finale 🏆**  
**Validation = terrain d'entraînement ⚽**

Vous voyez l'amélioration sans tricher! 🎉
