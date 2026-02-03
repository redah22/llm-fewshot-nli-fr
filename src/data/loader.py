"""
Data loader module for French NLI datasets.

Supports loading multiple French NLI datasets:
- XNLI (baseline)
- DACCORD
- RTE3-FR
- GQNLI-FR
- SICK-FR
- LingNLI-FR
"""

from typing import Dict, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
import os
from pathlib import Path


class NLIDataLoader:
    """Unified data loader for French NLI datasets."""
    
    # Dataset configurations
    DATASET_CONFIGS = {
        'xnli': {
            'hf_name': 'xnli',
            'language': 'fr',
            'splits': ['train', 'validation', 'test']
        },
        'fracas': {
            'hf_name': 'maximoss/fracas',
            'splits': ['train', 'validation', 'test']
        },
        'gqnli-fr': {
            'hf_name': 'maximoss/gqnli-fr',
            'splits': ['train', 'validation', 'test']
        },
        'daccord': {
            'hf_name': None,  # To be updated once available
            'path': 'data/raw/daccord',
            'splits': ['train', 'validation', 'test']
        },
        'rte3_fr': {
            'hf_name': None,
            'path': 'data/raw/rte3_fr',
            'splits': ['train', 'validation', 'test']
        },
        'gqnli_fr': {
            'hf_name': None,
            'path': 'data/raw/gqnli_fr',
            'splits': ['train', 'validation', 'test']
        },
        'sick_fr': {
            'hf_name': None,
            'path': 'data/raw/sick_fr',
            'splits': ['train', 'validation', 'test']
        },
        'lingnli_fr': {
            'hf_name': None,
            'path': 'data/raw/lingnli_fr',
            'splits': ['train', 'validation', 'test']
        }
    }
    
    # Label mappings
    LABEL_NAMES = ['entailment', 'neutral', 'contradiction']
    LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_NAMES)}
    ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(
        self, 
        dataset_name: str,
        split: Optional[str] = None,
        download: bool = True
    ) -> Union[Dataset, DatasetDict]:
        """
        Load a French NLI dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'xnli', 'daccord')
            split: Specific split to load ('train', 'validation', 'test')
                   If None, returns all splits as DatasetDict
            download: Whether to download if not cached
            
        Returns:
            Dataset or DatasetDict depending on split parameter
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.DATASET_CONFIGS.keys())}"
            )
        
        config = self.DATASET_CONFIGS[dataset_name]
        
        # Load from Hugging Face Hub
        if config.get('hf_name'):
            return self._load_from_hub(dataset_name, config, split)
        
        # Load from local files
        elif config.get('path'):
            return self._load_from_local(dataset_name, config, split)
        
        else:
            raise ValueError(f"No loading method configured for {dataset_name}")
    
    def _load_from_hub(
        self, 
        dataset_name: str,
        config: Dict,
        split: Optional[str]
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset from Hugging Face Hub."""
        hf_name = config['hf_name']
        
        # Special handling for XNLI
        if dataset_name == 'xnli':
            dataset = load_dataset(
                hf_name,
                'fr',
                cache_dir=str(self.cache_dir)
            )
            # XNLI uses 'validation' and 'test', no 'train' split
            # We'll use 'validation' as train for few-shot examples
            
        else:
            dataset = load_dataset(
                hf_name,
                cache_dir=str(self.cache_dir),
                split=split
            )
        
        return dataset if split is None else dataset[split]
    
    def _load_from_local(
        self,
        dataset_name: str,
        config: Dict,
        split: Optional[str]
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset from local files."""
        data_path = Path(config['path'])
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                f"Please download it first or contact supervisors."
            )
        
        # Try loading with datasets library
        # Assumes JSON, CSV, or similar format
        try:
            dataset = load_dataset(
                'json',  # or 'csv', adjust as needed
                data_files={
                    s: str(data_path / f"{s}.json") 
                    for s in config['splits']
                },
                cache_dir=str(self.cache_dir)
            )
            return dataset if split is None else dataset[split]
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {dataset_name} from {data_path}: {e}"
            )
    
    def get_few_shot_examples(
        self,
        dataset_name: str,
        num_examples: int = 5,
        strategy: str = 'stratified',
        allow_test: bool = False  # ⚠️ SAFEGUARD
    ) -> List[Dict]:
        """
        Get few-shot examples from a dataset.
        
        ⚠️ IMPORTANT: Par défaut, n'utilise JAMAIS le split 'test' pour éviter
        le data leakage. Les exemples viennent de 'train' ou 'validation'.
        
        Args:
            dataset_name: Source dataset
            num_examples: Number of examples to retrieve
            strategy: Selection strategy ('random', 'stratified', 'diverse')
            allow_test: Si True, autorise l'utilisation du test (DÉCONSEILLÉ!)
            
        Returns:
            List of example dictionaries
        """
        # ⚠️ SAFEGUARD: Empêcher l'utilisation du test par défaut
        if dataset_name == 'xnli':
            split = 'validation'  # XNLI n'a pas de train
            print(f"ℹ️ Few-shot depuis 'validation' de {dataset_name}")
        else:
            split = 'train'
            print(f"✅ Few-shot depuis 'train' de {dataset_name}")
        
        # Warning si test est autorisé
        if allow_test:
            print("⚠️⚠️⚠️ WARNING: allow_test=True peut causer du data leakage!")
            print("⚠️⚠️⚠️ N'utilisez JAMAIS le split test pour few-shot!")
            split = 'test'
        
        # Load correct split
        try:
            dataset = self.load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"❌ Erreur de chargement du split '{split}': {e}")
            # Fallback sur validation si train échoue
            if split == 'train':
                print(f"ℹ️ Tentative avec 'validation' à la place...")
                dataset = self.load_dataset(dataset_name, split='validation')
            else:
                raise
        
        if strategy == 'stratified':
            # Ensure balanced representation of all classes
            examples_per_class = num_examples // 3
            examples = []
            
            for label_id in range(3):
                label_examples = dataset.filter(
                    lambda x: x['label'] == label_id
                ).shuffle(seed=42).select(range(examples_per_class))
                examples.extend(label_examples)
                
        elif strategy == 'random':
            examples = dataset.shuffle(seed=42).select(range(num_examples))
            
        elif strategy == 'diverse':
            # TODO: Implement diversity-based selection (e.g., using embeddings)
            raise NotImplementedError("Diverse sampling not yet implemented")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return [
            {
                'premise': ex['premise'],
                'hypothesis': ex['hypothesis'],
                'label': self.ID_TO_LABEL[ex['label']]
            }
            for ex in examples
        ]
    
    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset = self.load_dataset(dataset_name)
        
        stats = {
            'dataset': dataset_name,
            'splits': {}
        }
        
        for split_name, split_data in dataset.items():
            label_counts = {}
            for example in split_data:
                label = self.ID_TO_LABEL[example['label']]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            stats['splits'][split_name] = {
                'total': len(split_data),
                'label_distribution': label_counts
            }
        
        return stats


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='xnli', 
                       help='Dataset to load')
    parser.add_argument('--download', action='store_true',
                       help='Download the dataset')
    parser.add_argument('--stats', action='store_true',
                       help='Print dataset statistics')
    
    args = parser.parse_args()
    
    loader = NLIDataLoader()
    
    if args.download:
        print(f"Loading {args.dataset}...")
        dataset = loader.load_dataset(args.dataset)
        print(f"✓ Successfully loaded {args.dataset}")
        print(f"  Splits: {list(dataset.keys())}")
    
    if args.stats:
        print(f"\nDataset Statistics for {args.dataset}:")
        stats = loader.get_dataset_stats(args.dataset)
        for split, split_stats in stats['splits'].items():
            print(f"\n{split.upper()}:")
            print(f"  Total examples: {split_stats['total']}")
            print(f"  Label distribution:")
            for label, count in split_stats['label_distribution'].items():
                pct = (count / split_stats['total']) * 100
                print(f"    {label}: {count} ({pct:.1f}%)")
