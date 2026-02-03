"""
Évaluation Few-Shot avec Gemini

Choisir le dataset: GQNLI-FR ou FraCaS
"""

from datasets import DatasetDict
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import time

# Charger .env
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Demander quel dataset
print("="*60)
print("ÉVALUATION FEW-SHOT - GEMINI")
print("="*60)

print("\nQuel dataset?")
print("1. GQNLI-FR")
print("2. FraCaS GQ")

choice = input("\nVotre choix (1 ou 2): ").strip()

if choice == "1":
    dataset_name = "gqnli_fr"
    dataset_path = "data/processed/gqnli_fr"
    premise_key = "premise"
    print("\n📊 Dataset: GQNLI-FR")
elif choice == "2":
    dataset_name = "fracas_gq"
    dataset_path = "data/processed/fracas_gq"
    premise_key = "premises"
    print("\n📊 Dataset: FraCaS GQ")
else:
    print("❌ Choix invalide!")
    exit(1)

# Charger les données
print(f"Chargement de {dataset_name}...")
dataset = DatasetDict.load_from_disk(dataset_path)

train_data = dataset['train']
val_data = dataset['validation']

print(f"Train: {len(train_data)} exemples")
print(f"Validation: {len(val_data)} exemples")

# Sélectionner des exemples stratifiés pour few-shot
def select_stratified_examples(data, n_shots):
    if n_shots == 0:
        return []
    
    labels_seen = set()
    examples = []
    
    for ex in data:
        label = ex['label']
        if label not in labels_seen:
            examples.append(ex)
            labels_seen.add(label)
            if len(examples) >= n_shots:
                break
    
    return examples

# Créer le prompt
def create_prompt(premise, hypothesis, few_shot_examples, prem_key):
    prompt = "Tu es un expert en inférence en langage naturel (NLI).\n\n"
    prompt += "Pour chaque paire prémisse/hypothèse, détermine la relation:\n"
    prompt += "- 0 = entailment (l'hypothèse découle de la prémisse)\n"
    prompt += "- 1 = neutral (pas de relation claire)\n"
    prompt += "- 2 = contradiction (l'hypothèse contredit la prémisse)\n\n"
    
    if few_shot_examples:
        prompt += "Exemples:\n\n"
        for i, ex in enumerate(few_shot_examples, 1):
            prompt += f"Exemple {i}:\n"
            prompt += f"Prémisse: {ex[prem_key]}\n"
            prompt += f"Hypothèse: {ex['hypothesis']}\n"
            prompt += f"Réponse: {ex['label']}\n\n"
    
    prompt += "Nouvelle paire à évaluer:\n"
    prompt += f"Prémisse: {premise}\n"
    prompt += f"Hypothèse: {hypothesis}\n"
    prompt += "Réponse (seulement le chiffre 0, 1 ou 2): "
    
    return prompt

# Évaluer avec Gemini
def evaluate_gemini(data, few_shot_examples, prem_key, model_name='gemini-2.5-flash'):
    model = genai.GenerativeModel(model_name)
    
    predictions = []
    labels = []
    
    for ex in tqdm(data, desc=f"Éval {len(few_shot_examples)}-shot"):
        prompt = create_prompt(ex[prem_key], ex['hypothesis'], few_shot_examples, prem_key)
        
        max_retries = 5
        retry_delay = 30  # Secondes
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                pred_text = response.text.strip()
                
                try:
                    pred = int(pred_text)
                    if pred not in [0, 1, 2]:
                        pred = 1
                except:
                    pred = 1
                
                predictions.append(pred)
                labels.append(ex['label'])
                
                # Petit délai de courtoisie même si succès
                time.sleep(2)
                break  # Succès, on sort de la boucle de retry
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Quota exceeded" in error_str:
                    print(f"\n⚠️ Quota dépassé. Attente de {retry_delay}s avant retry ({attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"\n❌ Erreur non-quota: {e}")
                    predictions.append(1) # Cas d'erreur fatal, on met neutral
                    labels.append(ex['label'])
                    break
    
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels)
    
    return accuracy, predictions, labels

# Expérience
print("\n" + "="*60)
print("DÉBUT DE L'EXPÉRIENCE")
print("="*60)
print(f"1. Les splits sont chargés depuis le disque (figés).")
print(f"2. Exemples pour le contexte -> tirés de TRAIN uniquement.")
print(f"3. Exemples à évaluer -> tirés de VALIDATION uniquement.")
print(f"4. Le modèle n'a pas de mémoire entre les appels.\n")

results = {}

# 1. Baseline 0-shot
print("-" * 30)
print("PHASE 1: BASELINE (0-shot)")
print("-" * 30)
print("Le modèle ne voit AUCUN exemple d'entraînement.")
print(f"Évaluation sur {len(val_data[:20])} questions de VALIDATION...\n")

few_shot_examples = []
test_subset = val_data.select(range(min(20, len(val_data))))
acc_0, _, _ = evaluate_gemini(test_subset, few_shot_examples, premise_key)
results['0-shot'] = {'accuracy': acc_0}

print(f"\n>> Précision Baseline (0-shot): {acc_0:.2%}")


# 2. Few-shot
print("\n" + "-" * 30)
print("PHASE 2: FEW-SHOT LEARNING")
print("-" * 30)

for n_shots in [1, 3, 5]:
    print(f"\n--- Test avec {n_shots} exemple(s) de contexte ---")
    print(f"Source: {n_shots} exemple(s) pris aléatoirement dans TRAIN")
    
    few_shot_examples = select_stratified_examples(train_data, n_shots)
    acc, _, _ = evaluate_gemini(test_subset, few_shot_examples, premise_key)
    
    results[f'{n_shots}-shot'] = {'accuracy': acc}
    gain = acc - acc_0
    print(f">> Précision {n_shots}-shot: {acc:.2%} (Gain: {gain:+.2%})")

# Résumé
print("\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)
print(f"Dataset: {dataset_name}")
print(f"{'Shot':<10} | {'Précision':<10} | {'Gain':<10}")
print("-" * 35)
for k, v in results.items():
    acc = v['accuracy']
    base = results['0-shot']['accuracy']
    gain = acc - base
    print(f"{k:<10} | {acc:.2%}    | {gain:+.2%}")

os.makedirs('results', exist_ok=True)
with open(f'results/gemini_{dataset_name}_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Résultats: results/gemini_{dataset_name}_results.json")
