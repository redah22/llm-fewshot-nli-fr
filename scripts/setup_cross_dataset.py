"""
Cross-Dataset Few-Shot Learning:
FraCaS (GENERALIZED QUANTIFIERS) → GQNLI-FR

Ce script montre comment faire du few-shot learning sur un topic spécifique
de FraCaS et évaluer sur GQNLI-FR.
"""

from datasets import load_dataset, DatasetDict
import pandas as pd

print("="*60)
print("CROSS-DATASET FEW-SHOT LEARNING")
print("Source: FraCaS (GENERALIZED QUANTIFIERS)")
print("Target: GQNLI-FR")
print("="*60)

# ============================================================================
# 1. CHARGER FRACAS ET FILTRER SUR GENERALIZED QUANTIFIERS
# ============================================================================

print("\n📦 Chargement de FraCaS...")
fracas = load_dataset('maximoss/fracas')
fracas_train = fracas['train']

print(f"Total FraCaS: {len(fracas_train)} exemples")

# Filtrer pour garder seulement GENERALIZED QUANTIFIERS
gq_examples = [
    ex for ex in fracas_train 
    if ex.get('topic') == 'GENERALIZED QUANTIFIERS'
]

print(f"\n✅ Filtrés sur GENERALIZED QUANTIFIERS: {len(gq_examples)} exemples")

# Afficher quelques exemples
print("\n📝 Exemples filtrés:")
for i, ex in enumerate(gq_examples[:3], 1):
    print(f"\n{i}. Label: {ex['label']}")
    print(f"   Prémisse: {ex['premises'][:80]}...")
    print(f"   Hypothèse: {ex['hypothesis'][:80]}...")

# ============================================================================
# 2. CHARGER GQNLI-FR (DATASET CIBLE)
# ============================================================================

print("\n" + "="*60)
print("📦 Chargement de GQNLI-FR...")

try:
    gqnli = load_dataset('maximoss/gqnli-fr')
    
    print(f"\n✅ GQNLI-FR chargé!")
    print(f"Splits disponibles: {list(gqnli.keys())}")
    
    for split_name in gqnli.keys():
        print(f"  {split_name}: {len(gqnli[split_name])} exemples")
    
    # Voir la structure
    if len(gqnli[list(gqnli.keys())[0]]) > 0:
        example = gqnli[list(gqnli.keys())[0]][0]
        print(f"\n📋 Colonnes GQNLI-FR: {list(example.keys())}")
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nAssurez-vous que maximoss/gqnli-fr existe sur HuggingFace")
    print("ou que vous avez les bons accès.")
    exit(1)

# ============================================================================
# 3. PRÉPARER LES SPLITS POUR L'EXPÉRIENCE
# ============================================================================

print("\n" + "="*60)
print("📊 Préparation des splits...")

# Diviser les exemples GQ de FraCaS en train/val
# (pour avoir des exemples few-shot et des données de développement)
from datasets import Dataset

gq_dataset = Dataset.from_dict({
    'premises': [ex['premises'] for ex in gq_examples],
    'hypothesis': [ex['hypothesis'] for ex in gq_examples],
    'label': [ex['label'] for ex in gq_examples],
})

# Mélanger et diviser (80% pour few-shot, 20% pour dev)
shuffled_gq = gq_dataset.shuffle(seed=42)
split_idx = int(len(shuffled_gq) * 0.8)

gq_few_shot = shuffled_gq.select(range(0, split_idx))
gq_dev = shuffled_gq.select(range(split_idx, len(shuffled_gq)))

print(f"\nFraCaS GQ splits:")
print(f"  Few-shot pool: {len(gq_few_shot)} exemples")
print(f"  Dev: {len(gq_dev)} exemples")

# Diviser GQNLI-FR si besoin
# (Supposons qu'il a déjà des splits)
if 'train' in gqnli:
    gqnli_train = gqnli['train']
    print(f"\nGQNLI-FR train: {len(gqnli_train)} exemples")

if 'validation' in gqnli:
    gqnli_val = gqnli['validation']
    print(f"GQNLI-FR validation: {len(gqnli_val)} exemples")
    
if 'test' in gqnli:
    gqnli_test = gqnli['test']
    print(f"GQNLI-FR test: {len(gqnli_test)} exemples")

# ============================================================================
# 4. STRATÉGIE D'ÉVALUATION
# ============================================================================

print("\n" + "="*60)
print("🎯 STRATÉGIE D'ÉVALUATION")
print("="*60)

strategy = """
1. FEW-SHOT EXAMPLES:
   - Source: FraCaS (GENERALIZED QUANTIFIERS uniquement)
   - Nombre: Tester 0, 1, 3, 5, 10 exemples
   - Sélection: Stratifiée (un de chaque label)

2. DÉVELOPPEMENT (sur validation):
   ✅ Tester différents n_shots sur GQNLI-FR validation
   ✅ Choisir le meilleur n_shots
   ❌ NE PAS toucher GQNLI-FR test!

3. ÉVALUATION FINALE (sur test):
   ✅ Évaluer avec le meilleur n_shots sur GQNLI-FR test
   ✅ UNE SEULE FOIS!

WORKFLOW:
   FraCaS GQ (few-shot) → GQNLI-FR (validation) → Choisir n
   FraCaS GQ (few-shot) → GQNLI-FR (test) → Résultats finaux
"""

print(strategy)

# ============================================================================
# 5. SÉLECTION D'EXEMPLES FEW-SHOT
# ============================================================================

print("\n" + "="*60)
print("📝 Sélection d'exemples few-shot (stratifiés)")
print("="*60)

# Obtenir les labels uniques
unique_labels = sorted(set(ex['label'] for ex in gq_few_shot))
print(f"\nLabels uniques dans FraCaS GQ: {unique_labels}")

# Sélectionner un exemple de chaque label (stratifié)
few_shot_examples = {}

for label in unique_labels:
    for ex in gq_few_shot:
        if ex['label'] == label:
            few_shot_examples[label] = ex
            break

print(f"\n✅ {len(few_shot_examples)} exemples few-shot sélectionnés:")
for label, ex in few_shot_examples.items():
    print(f"\nLabel {label}:")
    print(f"  P: {ex['premises'][:60]}...")
    print(f"  H: {ex['hypothesis'][:60]}...")

# ============================================================================
# 6. SAUVEGARDER LES DONNÉES POUR L'EXPÉRIENCE
# ============================================================================

print("\n" + "="*60)
print("💾 Sauvegarde des données...")

# Créer un DatasetDict pour FraCaS GQ
fracas_gq_split = DatasetDict({
    'few_shot': gq_few_shot,
    'dev': gq_dev,
})

# Sauvegarder
fracas_gq_split.save_to_disk('data/processed/fracas_gq_split')
print("✅ FraCaS GQ sauvegardé: data/processed/fracas_gq_split")

# Sauvegarder GQNLI-FR aussi pour référence
gqnli.save_to_disk('data/processed/gqnli_fr_split')
print("✅ GQNLI-FR sauvegardé: data/processed/gqnli_fr_split")

# ============================================================================
# 7. RÉSUMÉ
# ============================================================================

print("\n" + "="*60)
print("✅ SETUP COMPLET!")
print("="*60)

summary = f"""
📊 DONNÉES PRÉPARÉES:

Source (FraCaS - GENERALIZED QUANTIFIERS):
  • Few-shot pool: {len(gq_few_shot)} exemples
  • Dev: {len(gq_dev)} exemples
  • Labels: {unique_labels}

Cible (GQNLI-FR):
  • Validation: {len(gqnli_val) if 'validation' in gqnli else 'N/A'} exemples
  • Test: {len(gqnli_test) if 'test' in gqnli else 'N/A'} exemples

📁 FICHIERS SAUVEGARDÉS:
  • data/processed/fracas_gq_split/
  • data/processed/gqnli_fr_split/

🎯 PROCHAINES ÉTAPES:

1. Notebook: notebooks/02_cross_dataset_few_shot.ipynb
   → Tester 0-shot, 1-shot, 3-shot, 5-shot, 10-shot
   → Sur GQNLI-FR validation
   → Choisir le meilleur

2. Script: scripts/final_cross_dataset_eval.py
   → Évaluer sur GQNLI-FR test (une fois!)
   → Rapporter résultats

⚠️  RAPPEL: Ne toucher test qu'à la fin!
"""

print(summary)

print("="*60)
print("Pour lancer l'expérience, voir: notebooks/02_cross_dataset_few_shot.ipynb")
print("="*60)
