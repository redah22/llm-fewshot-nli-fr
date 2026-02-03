"""
Script de test simple pour le dataset FraCaS.
Lancez avec: source venv/bin/activate && python3 test_fracas.py
"""

from datasets import load_dataset

print("🔄 Chargement du dataset maximoss/fracas...")
print("="*60)

try:
    # Charger le dataset
    fracas = load_dataset('maximoss/fracas')
    
    print(f"\n✅ Dataset chargé avec succès!")
    print(f"\n📊 Splits disponibles: {list(fracas.keys())}")
    
    # Compter le total de lignes
    total_rows = sum(len(fracas[split]) for split in fracas.keys())
    
    print(f"\n📈 TOTAL IMPORTÉ: {total_rows:,} lignes")
    print("="*60)
    
    # Informations détaillées sur chaque split
    for split_name in fracas.keys():
        split_data = fracas[split_name]
        num_rows = len(split_data)
        percentage = (num_rows / total_rows * 100) if total_rows > 0 else 0
        
        print(f"\n{split_name.upper()}:")
        print(f"  📊 Nombre de lignes: {num_rows:,} ({percentage:.1f}%)")
        
        if num_rows > 0:
            example = split_data[0]
            print(f"  📋 Colonnes: {list(example.keys())}")
            
            # Afficher le premier exemple
            print(f"\n  ✏️  Premier exemple:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    • {key}: {value[:100]}...")
                else:
                    print(f"    • {key}: {value}")
    
    print("\n" + "="*60)
    print(f"✅ RÉSUMÉ:")
    print(f"   Total importé: {total_rows:,} lignes")
    print(f"   Splits: {len(fracas.keys())}")
    print(f"   Dataset: maximoss/fracas")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    print("\nAssurez-vous d'avoir:")
    print("1. Activé l'environnement virtuel: source venv/bin/activate")
    print("2. Installé les dépendances: pip install -r requirements.txt")
