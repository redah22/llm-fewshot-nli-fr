"""
Setup des données - Téléchargement et préparation de XNLI French

XNLI est un grand corpus d'inférence en langage naturel traduit dans plusieurs langues.
Ici, nous isolons la portion française pour pré-entraîner FlauBERT.
"""

from datasets import load_dataset
import os

print("="*60)
print("SETUP DONNÉES - XNLI (fr)")
print("="*60)

print("Chargement du dataset XNLI complet (langue: fr)...")

try:
    # Charge la version fr de xnli depuis HuggingFace
    xnli = load_dataset("xnli", "fr")
    
    print(f"Train: {len(xnli['train'])} exemples")
    print(f"Validation: {len(xnli['validation'])} exemples")
    print(f"Test: {len(xnli['test'])} exemples")
    
    # On sauvegarde tel quel dans le dossier data/processed
    os.makedirs('data/processed/xnli_fr', exist_ok=True)
    xnli.save_to_disk('data/processed/xnli_fr')
    
    print("\n✅ Dataset XNLI-fr sauvegardé avec succès dans 'data/processed/xnli_fr'")
    print("Vous pouvez maintenant lancer l'entraînement via train_flaubert_xnli.py")

except Exception as e:
    print(f"❌ Erreur lors du chargement ou de la sauvegarde de XNLI: {e}")
