"""
Test rapide pour GQNLI-FR
À lancer avec: source venv/bin/activate && python3 test_gqnli.py
"""

from datasets import load_dataset

print("Chargement de maximoss/gqnli-fr...")
print("-" * 50)

try:
    gqnli = load_dataset('maximoss/gqnli-fr')
    
    print(f"\nOk, dataset chargé")
    print(f"Splits: {list(gqnli.keys())}")
    
    total_rows = sum(len(gqnli[split]) for split in gqnli.keys())
    
    print(f"\nTotal: {total_rows:,} lignes")
    print("-" * 50)
    
    for split_name in gqnli.keys():
        split_data = gqnli[split_name]
        num_rows = len(split_data)
        percentage = (num_rows / total_rows * 100) if total_rows > 0 else 0
        
        print(f"\n{split_name.upper()}:")
        print(f"  Lignes: {num_rows:,} ({percentage:.1f}%)")
        
        if num_rows > 0:
            example = split_data[0]
            print(f"  Colonnes: {list(example.keys())}")
            
            print(f"\n  Premier exemple:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    
    print("\n" + "-" * 50)
    print(f"Résumé:")
    print(f"  Total: {total_rows:,} lignes")
    print(f"  Splits: {len(gqnli.keys())}")
    print(f"  Dataset: maximoss/gqnli-fr")
    print("-" * 50)
    
except Exception as e:
    print(f"\nErreur: {e}")
    print("\nVérifiez que:")
    print("1. venv est activé: source venv/bin/activate")
    print("2. datasets est installé: pip install datasets")
    print("3. Le dataset existe bien sur HuggingFace")
