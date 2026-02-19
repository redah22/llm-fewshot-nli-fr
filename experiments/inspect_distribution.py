
from datasets import load_from_disk
import sys

# GQNLI
try:
    print("=== GQNLI-FR ===")
    gqnli = load_from_disk('data/processed/gqnli_fr')
    train_data = gqnli['train']
    
    counts = {}
    for i in range(len(train_data)):
        ex = train_data[i]
        lbl = ex['label']
        if lbl not in counts: counts[lbl] = 0
        counts[lbl] += 1
    
    print(f"Total: {len(train_data)}")
    for k, v in counts.items():
        print(f"Label {k}: {v} ({v/len(train_data)*100:.1f}%)")

    # Sample mapping check
    print("\nSAMPLE CHECK:")
    for i in range(min(5, len(train_data))):
        ex = train_data[i]
        print(f"Text: {ex['premise'][:30]} -> {ex['hypothesis'][:30]} | Lbl: {ex['label']} (Txt: {ex.get('label_text', 'N/A')})")

except Exception as e:
    print(f"Error loading GQNLI: {e}")

# FraCaS Subset
try:
    print("\n=== FraCaS Subset (0-74) ===")
    fracas = load_from_disk('data/processed/fracas_subset_75')
    data = fracas['validation']
    
    counts = {}
    label_map = {'yes': 0, 'unknown': 1, 'no': 2}
    

    for i in range(len(data)):
        ex = data[i]
        lbl_str = ex['label']
        #lbl_int = label_map.get(lbl_str, -1)
        
        if lbl_str not in counts: counts[lbl_str] = 0
        counts[lbl_str] += 1
        
    print(f"Total: {len(data)}")
    for k, v in counts.items():
        print(f"Label '{k}' : {v} ({v/len(data)*100:.1f}%)")
        
    # Sample check
    print("\nSAMPLE CHECK:")
    for i in range(min(5, len(data))):
        ex = data[i]
        lbl_int = label_map.get(ex['label'], -1)
        print(f"Text: {ex['premises'][:30]} -> {ex['hypothesis'][:30]} | Lbl: {lbl_int} (Orig: {ex['label']})")
        
except Exception as e:
    print(f"Error loading FraCaS: {e}")
