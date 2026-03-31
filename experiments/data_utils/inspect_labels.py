
from datasets import load_from_disk
import sys

print("=== GQNLI-FR Labels ===")
try:
    gqnli = load_from_disk('data/processed/gqnli_fr')
    train_data = gqnli['train']
    print(f"Features: {train_data.features['label']}")
    # Inspect first 10 examples
    for i in range(min(10, len(train_data))):
        ex = train_data[i]
        print(f"Label: {ex['label']} | Label Text: {ex.get('label_text', 'N/A')}")
except Exception as e:
    print(f"Error loading GQNLI: {e}")

print("\n=== FraCaS Subset Labels ===")
try:
    fracas = load_from_disk('data/processed/fracas_subset_75')
    test_data = fracas['validation'] # or test
    print(f"Features: {test_data.features['label']}")
    # Inspect first 10 examples
    for i in range(min(10, len(test_data))):
        ex = test_data[i]
        print(f"Label: {ex['label']}")
except Exception as e:
    print(f"Error loading FraCaS: {e}")
