"""
Utilitaire pour diviser un dataset en train/validation/test.

Exemple: dataset de 100 lignes -> 60% train, 20% val, 20% test
"""

from datasets import load_dataset, DatasetDict
import argparse


def split_dataset(
    dataset_name: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """
    Divise un dataset en train/validation/test.
    
    Args:
        dataset_name: Nom du dataset HuggingFace
        train_ratio: Proportion pour l'entraînement (défaut: 0.6 = 60%)
        val_ratio: Proportion pour la validation (défaut: 0.2 = 20%)
        test_ratio: Proportion pour le test (défaut: 0.2 = 20%)
        seed: Seed pour la reproductibilité
        
    Returns:
        DatasetDict avec les splits train/validation/test
    """
    # Vérifier que les ratios somment à 1
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Les ratios doivent sommer à 1.0 (actuel: {total})")
    
    print(f"📦 Chargement du dataset: {dataset_name}")
    
    # Charger le dataset
    # Si le dataset a déjà des splits, on les fusionne d'abord
    try:
        dataset = load_dataset(dataset_name)
        
        # Si c'est déjà un DatasetDict, fusionner tous les splits
        if isinstance(dataset, DatasetDict):
            from datasets import concatenate_datasets
            all_data = concatenate_datasets(list(dataset.values()))
            print(f"✓ Dataset fusionné: {len(all_data)} exemples au total")
        else:
            all_data = dataset
            
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None
    
    print(f"\n📊 Division du dataset:")
    print(f"  Total: {len(all_data)} exemples")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Validation: {val_ratio*100:.0f}%")
    print(f"  Test: {test_ratio*100:.0f}%")
    
    # Calculer les tailles
    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    # Le reste va au test pour éviter les erreurs d'arrondi
    test_size = total_size - train_size - val_size
    
    print(f"\n📏 Tailles calculées:")
    print(f"  Train: {train_size} exemples")
    print(f"  Validation: {val_size} exemples")
    print(f"  Test: {test_size} exemples")
    
    # Mélanger d'abord pour randomiser
    shuffled = all_data.shuffle(seed=seed)
    
    # Créer les splits
    train_dataset = shuffled.select(range(0, train_size))
    val_dataset = shuffled.select(range(train_size, train_size + val_size))
    test_dataset = shuffled.select(range(train_size + val_size, total_size))
    
    # Créer le DatasetDict
    split_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"\n✅ Dataset divisé avec succès!")
    print(f"  Splits: {list(split_datasets.keys())}")
    
    return split_datasets


def split_existing_split(
    dataset_name: str,
    split_name: str = 'train',
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """
    Divise un split existant d'un dataset.
    
    Utile si vous avez par exemple seulement un split 'train' 
    et voulez le diviser en train/val/test.
    
    Args:
        dataset_name: Nom du dataset HuggingFace
        split_name: Nom du split à diviser
        train_ratio, val_ratio, test_ratio: Proportions
        seed: Seed pour reproductibilité
        
    Returns:
        DatasetDict avec les nouveaux splits
    """
    print(f"📦 Chargement du split '{split_name}' de {dataset_name}")
    
    # Charger uniquement le split spécifié
    dataset = load_dataset(dataset_name, split=split_name)
    
    print(f"✓ Chargé: {len(dataset)} exemples")
    
    # Vérifier les ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Les ratios doivent sommer à 1.0 (actuel: {total})")
    
    # Calculer les tailles
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"\n📊 Division:")
    print(f"  Train: {train_size} ({train_ratio*100:.0f}%)")
    print(f"  Validation: {val_size} ({val_ratio*100:.0f}%)")
    print(f"  Test: {test_size} ({test_ratio*100:.0f}%)")
    
    # Mélanger et diviser
    shuffled = dataset.shuffle(seed=seed)
    
    train_dataset = shuffled.select(range(0, train_size))
    val_dataset = shuffled.select(range(train_size, train_size + val_size))
    test_dataset = shuffled.select(range(train_size + val_size, total_size))
    
    # Créer le DatasetDict
    split_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"\n✅ Split divisé avec succès!")
    
    return split_datasets


# Exemple d'utilisation dans du code
def example_usage():
    """Exemples d'utilisation."""
    
    print("="*60)
    print("EXEMPLE 1: Diviser un dataset entier")
    print("="*60)
    
    # Charger XNLI et le diviser
    from datasets import load_dataset
    
    # Charger tous les exemples
    xnli_full = load_dataset('xnli', 'fr')
    
    # Fusionner tous les splits
    from datasets import concatenate_datasets
    all_xnli = concatenate_datasets([xnli_full['validation'], xnli_full['test']])
    
    print(f"Total d'exemples: {len(all_xnli)}")
    
    # Mélanger
    shuffled = all_xnli.shuffle(seed=42)
    
    # Diviser: 60% train, 20% val, 20% test
    total = len(shuffled)
    train_size = int(total * 0.6)
    val_size = int(total * 0.2)
    
    train = shuffled.select(range(0, train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = shuffled.select(range(train_size + val_size, total))
    
    new_dataset = DatasetDict({
        'train': train,
        'validation': val,
        'test': test
    })
    
    print(f"Train: {len(new_dataset['train'])} exemples")
    print(f"Validation: {len(new_dataset['validation'])} exemples")
    print(f"Test: {len(new_dataset['test'])} exemples")
    
    print("\n" + "="*60)
    print("EXEMPLE 2: Méthode train_test_split intégrée")
    print("="*60)
    
    # La bibliothèque datasets a aussi une méthode intégrée!
    splits = all_xnli.train_test_split(test_size=0.4, seed=42)  # 60% train, 40% reste
    
    # Diviser le 'test' en validation et test
    remaining_splits = splits['test'].train_test_split(test_size=0.5, seed=42)  # 50/50 = 20% chacun
    
    final_dataset = DatasetDict({
        'train': splits['train'],
        'validation': remaining_splits['train'],
        'test': remaining_splits['test']
    })
    
    print(f"Train: {len(final_dataset['train'])} exemples")
    print(f"Validation: {len(final_dataset['validation'])} exemples")
    print(f"Test: {len(final_dataset['test'])} exemples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diviser un dataset en train/validation/test'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Nom du dataset HuggingFace'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Split spécifique à diviser (optionnel)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Ratio pour train (défaut: 0.6)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Ratio pour validation (défaut: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Ratio pour test (défaut: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed pour reproductibilité'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Montrer des exemples d\'utilisation'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Sauvegarder le dataset divisé (chemin)'
    )
    
    args = parser.parse_args()
    
    if args.example:
        example_usage()
    
    elif args.dataset:
        if args.split:
            # Diviser un split spécifique
            result = split_existing_split(
                dataset_name=args.dataset,
                split_name=args.split,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
        else:
            # Diviser le dataset entier
            result = split_dataset(
                dataset_name=args.dataset,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
        
        # Sauvegarder si demandé
        if args.save and result:
            print(f"\n💾 Sauvegarde dans: {args.save}")
            result.save_to_disk(args.save)
            print("✅ Sauvegardé!")
    
    else:
        print("Utilisez --dataset DATASET_NAME ou --example")
        print("\nExemples:")
        print("  python scripts/split_dataset.py --example")
        print("  python scripts/split_dataset.py --dataset xnli --split validation")
        print("  python scripts/split_dataset.py --dataset votre-nom/dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15")
