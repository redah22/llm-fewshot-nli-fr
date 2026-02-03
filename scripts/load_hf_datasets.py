"""
Quick setup script for loading French NLI datasets from Hugging Face.

If your datasets are already on Hugging Face Hub, simply update the 
DATASET_CONFIGS in this file with the correct Hugging Face dataset names.
"""

from src.data.loader import NLIDataLoader
from datasets import load_dataset

# =============================================================================
# CONFIGURATION - Update with your Hugging Face dataset names
# =============================================================================

HUGGINGFACE_DATASETS = {
    'xnli': {
        'hf_path': 'xnli',
        'config': 'fr',
        'splits': ['validation', 'test']
    },
    'fracas': {
        'hf_path': 'maximoss/fracas',
        'config': None,
        'splits': ['train', 'validation', 'test']
    },
    'gqnli-fr': {
        'hf_path': 'maximoss/gqnli-fr',
        'config': None,
        'splits': ['train', 'validation', 'test']  # À vérifier
    },
    # Ajoutez vos autres datasets ici
    # Exemple:
    # 'daccord': {
    #     'hf_path': 'nom-utilisateur/daccord',
    #     'config': None,
    #     'splits': ['train', 'validation', 'test']
    # },
    # 'rte3_fr': {
    #     'hf_path': 'nom-utilisateur/rte3-fr',
    #     'config': None,
    #     'splits': ['train', 'validation', 'test']
    # },
    # ... etc pour les autres datasets
}

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def download_all_datasets():
    """Télécharge tous les datasets depuis Hugging Face."""
    print("🔄 Téléchargement des datasets depuis Hugging Face...\n")
    
    for dataset_name, config in HUGGINGFACE_DATASETS.items():
        print(f"📦 Chargement de {dataset_name}...")
        try:
            hf_path = config['hf_path']
            hf_config = config.get('config')
            
            if hf_config:
                dataset = load_dataset(hf_path, hf_config)
            else:
                dataset = load_dataset(hf_path)
            
            print(f"   ✅ {dataset_name} chargé avec succès")
            print(f"   Splits disponibles: {list(dataset.keys())}")
            print(f"   Nombre d'exemples: {sum(len(dataset[s]) for s in dataset.keys())}\n")
            
        except Exception as e:
            print(f"   ❌ Erreur lors du chargement de {dataset_name}: {e}\n")


def test_dataset_loading(dataset_name: str):
    """
    Teste le chargement d'un dataset spécifique.
    
    Args:
        dataset_name: Nom du dataset à tester
    """
    if dataset_name not in HUGGINGFACE_DATASETS:
        print(f"❌ Dataset '{dataset_name}' non trouvé dans la configuration")
        print(f"Datasets disponibles: {list(HUGGINGFACE_DATASETS.keys())}")
        return
    
    print(f"\n🧪 Test de chargement: {dataset_name}")
    print("=" * 60)
    
    config = HUGGINGFACE_DATASETS[dataset_name]
    hf_path = config['hf_path']
    hf_config = config.get('config')
    
    try:
        # Charger le dataset
        if hf_config:
            dataset = load_dataset(hf_path, hf_config)
        else:
            dataset = load_dataset(hf_path)
        
        print(f"✅ Dataset chargé avec succès")
        print(f"\nSplits: {list(dataset.keys())}")
        
        # Afficher un exemple de chaque split
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n{split_name.upper()}:")
            print(f"  Nombre d'exemples: {len(split_data)}")
            
            if len(split_data) > 0:
                example = split_data[0]
                print(f"  Colonnes: {list(example.keys())}")
                print(f"\n  Premier exemple:")
                
                # Essayer d'afficher prémisse/hypothèse
                for key in ['premise', 'hypothesis', 'label']:
                    if key in example:
                        value = example[key]
                        if isinstance(value, str) and len(value) > 80:
                            value = value[:80] + "..."
                        print(f"    {key}: {value}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def show_dataset_example(dataset_name: str, split: str = 'test', index: int = 0):
    """
    Affiche un exemple d'un dataset.
    
    Args:
        dataset_name: Nom du dataset
        split: Split à utiliser
        index: Index de l'exemple
    """
    if dataset_name not in HUGGINGFACE_DATASETS:
        print(f"❌ Dataset '{dataset_name}' non trouvé")
        return
    
    config = HUGGINGFACE_DATASETS[dataset_name]
    hf_path = config['hf_path']
    hf_config = config.get('config')
    
    try:
        # Charger le dataset
        if hf_config:
            dataset = load_dataset(hf_path, hf_config, split=split)
        else:
            dataset = load_dataset(hf_path, split=split)
        
        if index >= len(dataset):
            print(f"❌ Index {index} hors limites (max: {len(dataset)-1})")
            return
        
        example = dataset[index]
        
        print(f"\n📄 Exemple #{index} de {dataset_name} ({split})")
        print("=" * 60)
        print(f"Prémisse: {example.get('premise', 'N/A')}")
        print(f"Hypothèse: {example.get('hypothesis', 'N/A')}")
        
        label_id = example.get('label', -1)
        label_names = ['entailment', 'neutral', 'contradiction']
        if 0 <= label_id < 3:
            print(f"Label: {label_names[label_id]} ({label_id})")
        else:
            print(f"Label: {label_id}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


# =============================================================================
# MAIN - Exemples d'utilisation
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Utilitaire pour charger les datasets French NLI depuis Hugging Face'
    )
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Télécharger tous les datasets configurés'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Tester le chargement d\'un dataset spécifique'
    )
    parser.add_argument(
        '--example',
        type=str,
        help='Afficher un exemple d\'un dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Split à utiliser (défaut: test)'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Index de l\'exemple à afficher (défaut: 0)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='Lister les datasets configurés'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Datasets configurés:")
        for name, config in HUGGINGFACE_DATASETS.items():
            print(f"  • {name}: {config['hf_path']}")
    
    elif args.download_all:
        download_all_datasets()
    
    elif args.test:
        test_dataset_loading(args.test)
    
    elif args.example:
        show_dataset_example(args.example, args.split, args.index)
    
    else:
        # Mode interactif par défaut
        print("\n🚀 Script de chargement des datasets French NLI")
        print("=" * 60)
        print("\nUtilisation:")
        print("  python scripts/load_hf_datasets.py --download-all")
        print("  python scripts/load_hf_datasets.py --test xnli")
        print("  python scripts/load_hf_datasets.py --example xnli --split test --index 5")
        print("  python scripts/load_hf_datasets.py --list")
        print("\nOu modifiez HUGGINGFACE_DATASETS dans ce fichier")
        print("avec vos noms de datasets Hugging Face.\n")
