"""
Setup des données - Choisir le dataset

Choix: FraCaS ou GQNLI-FR
Splits: 60/20/20 (train/val/test)
"""

from datasets import load_dataset, Dataset, DatasetDict
import sys

print("="*60)
print("SETUP DONNÉES - Splits 60/20/20")
print("="*60)

# Demander quel dataset
print("\nQuel dataset voulez-vous utiliser?")
print("1. GQNLI-FR (300 exemples de NLI en français)")
print("2. FraCaS (346 exemples, topic GENERALIZED QUANTIFIERS)")
print("3. FraCaS (Lignes 0-74, pour tests)")

choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if choice == "1":
    print("\n📊 Dataset choisi: GQNLI-FR")
    
    # Charger GQNLI-FR
    print("Chargement de GQNLI-FR...")
    gqnli = load_dataset('maximoss/gqnli-fr')
    data = gqnli['test']
    
    print(f"Total: {len(data)} exemples")
    
    # Shuffle
    shuffled = data.shuffle(seed=42)
    total = len(shuffled)
    
    # Split 60/20/20
    train_size = int(total * 0.6)
    val_size = int(total * 0.2)
    
    train = shuffled.select(range(0, train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = shuffled.select(range(train_size + val_size, total))
    
    # Sauvegarder
    dataset_dict = DatasetDict({
        'train': train,
        'validation': val,
        'test': test,
    })
    
    dataset_dict.save_to_disk('data/processed/gqnli_fr')
    
    print(f"\nTrain: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"Val: {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"Test: {len(test)} ({len(test)/total*100:.1f}%)")
    print(f"\n✅ Sauvegardé: data/processed/gqnli_fr")
    


elif choice == "2":
    print("\n📊 Dataset choisi: FraCaS (GENERALIZED QUANTIFIERS)")
    
    # Charger FraCaS et filtrer
    print("Chargement de FraCaS...")
    fracas = load_dataset('maximoss/fracas')
    fracas_data = fracas['train']
    
    # Filtrer sur GQ
    gq_examples = [
        ex for ex in fracas_data 
        if ex.get('topic') == 'GENERALIZED QUANTIFIERS'
    ]
    
    print(f"Total GQ: {len(gq_examples)} exemples")
    
    # Convertir en Dataset
    dataset = Dataset.from_dict({
        'premises': [ex['premises'] for ex in gq_examples],
        'hypothesis': [ex['hypothesis'] for ex in gq_examples],
        'label': [ex['label'] for ex in gq_examples],
    })
    
    # Shuffle
    shuffled = dataset.shuffle(seed=42)
    total = len(shuffled)
    
    # Split 60/20/20
    train_size = int(total * 0.6)
    val_size = int(total * 0.2)
    
    train = shuffled.select(range(0, train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = shuffled.select(range(train_size + val_size, total))
    
    # Sauvegarder
    dataset_dict = DatasetDict({
        'train': train,
        'validation': val,
        'test': test,
    })
    
    dataset_dict.save_to_disk('data/processed/fracas_gq')
    
    print(f"\nTrain: {len(train)} ({len(train)/total*100:.1f}%)")
    print(f"Val: {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"Test: {len(test)} ({len(test)/total*100:.1f}%)")
    print(f"\n✅ Sauvegardé: data/processed/fracas_gq")

elif choice == "3":
    print("\n📊 Dataset choisi: FraCaS (Lignes 0-74)")
    
    # Charger FraCaS
    print("Chargement de FraCaS...")
    fracas = load_dataset('maximoss/fracas')
    fracas_data = fracas['train']
    
    # Selectionner 0 à 74
    # On prend les 75 premiers exemples SANS melanger d'abord pour garantir "lignes 0 à 74"
    subset = fracas_data.select(range(75))
    
    print(f"Total: {len(subset)} exemples")
    
    # Pour permettre le testing sur l'ensemble complet, on met TOUT dans chaque split
    # (Surtout utile pour 'validation' et 'test' si on veut evaluer sur ces 75 exemples spécifiques)
    train = subset
    val = subset
    test = subset
    
    # Sauvegarder
    dataset_dict = DatasetDict({
        'train': train,
        'validation': val,
        'test': test,
    })
    
    dataset_dict.save_to_disk('data/processed/fracas_subset_75')
    
    print(f"\n⚠️  NOTE: Pour ce subset, Train/Val/Test contiennent tous les {len(subset)} exemples.")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    print(f"\n✅ Sauvegardé: data/processed/fracas_subset_75")
    
else:
    print("❌ Choix invalide!")
    sys.exit(1)

print("\n" + "="*60)
print("✅ SETUP TERMINÉ")
print("="*60)

print("\n⚠️  RAPPEL:")
print("   - Train: Pour few-shot et fine-tuning")
print("   - Validation: Pour développement")
print("   - Test: NE PAS TOUCHER avant la fin!")
