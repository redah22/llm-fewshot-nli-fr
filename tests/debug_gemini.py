"""
Debug Gemini - Voir les prédictions réelles
"""

from datasets import DatasetDict
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Charger données
gqnli = DatasetDict.load_from_disk('data/processed/gqnli_split')
fracas_gq = DatasetDict.load_from_disk('data/processed/fracas_gq')

gqnli_dev = gqnli['dev']
few_shot_pool = fracas_gq['few_shot']

# Prendre 1 exemple few-shot
few_shot_ex = few_shot_pool[0]

print("="*60)
print("DEBUG GEMINI")
print("="*60)

print("\nExemple few-shot utilisé:")
print(f"Prémisse: {few_shot_ex['premises']}")
print(f"Hypothèse: {few_shot_ex['hypothesis']}")
print(f"Label: {few_shot_ex['label']}")

# Tester sur 3 exemples de GQNLI-FR
model = genai.GenerativeModel('gemini-2.5-flash')

for i in range(3):
    ex = gqnli_dev[i]
    
    print(f"\n{'='*60}")
    print(f"Exemple GQNLI-FR {i+1}:")
    print(f"Prémisse: {ex['premise']}")
    print(f"Hypothèse: {ex['hypothesis']}")
    print(f"Vrai label: {ex['label']} ({ex['label_text']})")
    
    # Créer prompt simple
    prompt = f"""Tu es un expert en inférence logique.

Pour la paire suivante, détermine la relation:
- 0 = entailment (l'hypothèse découle de la prémisse)
- 1 = neutral (pas de relation claire)
- 2 = contradiction (l'hypothèse contredit la prémisse)

Exemple:
Prémisse: {few_shot_ex['premises']}
Hypothèse: {few_shot_ex['hypothesis']}
Réponse: {few_shot_ex['label']}

Nouvelle paire:
Prémisse: {ex['premise']}
Hypothèse: {ex['hypothesis']}
Réponse (seulement 0, 1 ou 2):"""
    
    try:
        response = model.generate_content(prompt)
        raw_response = response.text
        
        print(f"\nRéponse brute Gemini: '{raw_response}'")
        
        # Tenter de parser
        try:
            pred = int(raw_response.strip())
            print(f"Parsé comme: {pred}")
        except:
            print(f"❌ Échec parsing!")
            
    except Exception as e:
        print(f"Erreur: {e}")

print("\n" + "="*60)
print("Vérifiez si:")
print("1. Les labels correspondent (FraCaS vs GQNLI-FR)")
print("2. Gemini répond bien juste un chiffre")
print("3. Le prompt est clair")
