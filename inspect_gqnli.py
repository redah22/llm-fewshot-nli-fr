
from datasets import DatasetDict
import numpy as np

try:
    dataset = DatasetDict.load_from_disk("data/processed/gqnli_fr")
    
    print("GQNLI-FR Label Distribution:")
    for split in ['train', 'validation']:
        labels = dataset[split]['label']
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print(f"\n{split.upper()} ({total}):")
        for val, count in zip(unique, counts):
            print(f"  Label {val}: {count} ({count/total:.2%})")
            
except Exception as e:
    print(e)
