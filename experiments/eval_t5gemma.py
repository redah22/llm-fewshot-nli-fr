"""
Évaluation Few-Shot avec T5Gemma 2 (google/t5gemma-2-1b-1b)

T5Gemma est un modèle encoder-decoder multilingue basé sur Gemma 3.
Il génère du texte en sortie, donc on utilise une approche de prompting
similaire à Gemini, mais en local (pas d'API nécessaire).

Référence : https://huggingface.co/google/t5gemma-2-1b-1b

Utilisation :
    python3 experiments/eval_t5gemma.py

Prérequis :
    pip install -U transformers torch
    (Optionnel) pip install accelerate  # pour GPU multi-carte
"""

from datasets import DatasetDict
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torch
import json
import os
from tqdm import tqdm

"Pour se connecter et installer le modèle, il faut créer un token READ sur HuggingFace, puis taper : hf auth login et rentrer le token quand c'est demandé"

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────


MODEL_NAME = "google/t5gemma-2-1b-1b" 


print("=" * 60)
print("ÉVALUATION FEW-SHOT - T5GEMMA 2 (1B-1B)")
print("=" * 60)
print(f"Modèle : {MODEL_NAME}")

# Détecter le device disponible
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device : {device}")

# ─────────────────────────────────────────────
# 2. Choix du dataset
# ─────────────────────────────────────────────
print("\nQuel dataset?")
print("1. GQNLI-FR  (180 train / 60 val) — Recommandé")
print("2. FraCaS GQ (48 train  / 16 val)")

choice = input("\nVotre choix (1 ou 2): ").strip()

if choice == "1":
    dataset_name = "gqnli_fr"
    dataset_path = "data/processed/gqnli_fr"
    premise_key = "premise"
    print("\n📊 Dataset : GQNLI-FR")
elif choice == "2":
    dataset_name = "fracas_gq"
    dataset_path = "data/processed/fracas_gq"
    premise_key = "premises"
    print("\n📊 Dataset : FraCaS GQ")
else:
    print("❌ Choix invalide!")
    exit(1)

# ─────────────────────────────────────────────
# 3. Chargement des données
# ─────────────────────────────────────────────
print(f"\nChargement de {dataset_name}...")
dataset = DatasetDict.load_from_disk(dataset_path)

train_data = dataset["train"]
val_data   = dataset["validation"]

print(f"Train      : {len(train_data)} exemples")
print(f"Validation : {len(val_data)} exemples")

# ─────────────────────────────────────────────
# 4. Chargement du modèle T5Gemma
# ─────────────────────────────────────────────
print(f"\nChargement de {MODEL_NAME}...")
print("(Premier téléchargement ~2 Go — soyez patient)")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # Économise la VRAM
    device_map="auto",            # Répartit automatiquement sur GPU/CPU
)
model.eval()

print(f"✅ Modèle chargé sur {device}")

# ─────────────────────────────────────────────
# 5. Fonctions utilitaires
# ─────────────────────────────────────────────

def select_stratified_examples(data, n_shots):
    """Sélectionne n_shots exemples représentatifs de chaque classe."""
    if n_shots == 0:
        return []

    labels_seen = set()
    examples = []

    for ex in data:
        label = ex["label"]
        if label not in labels_seen:
            examples.append(ex)
            labels_seen.add(label)
            if len(examples) >= n_shots:
                break

    return examples


def create_prompt(premise, hypothesis, few_shot_examples, prem_key):
    """
    Construit un prompt NLI pour T5Gemma.

    T5Gemma est un modèle de génération de texte : on lui demande
    de répondre par '0', '1' ou '2'.
    """
    prompt = (
        "Tu es un expert en inférence en langage naturel (NLI).\n\n"
        "Pour chaque paire prémisse/hypothèse, détermine la relation :\n"
        "- 0 = entailment (l'hypothèse découle de la prémisse)\n"
        "- 1 = neutral (pas de relation claire)\n"
        "- 2 = contradiction (l'hypothèse contredit la prémisse)\n\n"
    )

    if few_shot_examples:
        prompt += "Exemples :\n\n"
        for i, ex in enumerate(few_shot_examples, 1):
            prompt += f"Exemple {i} :\n"
            prompt += f"Prémisse  : {ex[prem_key]}\n"
            prompt += f"Hypothèse : {ex['hypothesis']}\n"
            prompt += f"Réponse   : {ex['label']}\n\n"

    prompt += "Nouvelle paire à évaluer :\n"
    prompt += f"Prémisse  : {premise}\n"
    prompt += f"Hypothèse : {hypothesis}\n"
    prompt += "Réponse (seulement le chiffre 0, 1 ou 2) : "

    return prompt


def parse_prediction(text: str) -> int:
    """Extrait la prédiction (0, 1 ou 2) depuis la réponse du modèle."""
    text = text.strip()
    # Chercher le premier chiffre valide dans la réponse
    for char in text:
        if char in ("0", "1", "2"):
            return int(char)
    return 1  # Neutre par défaut si aucun chiffre trouvé


def evaluate_t5gemma(data, few_shot_examples, prem_key, max_new_tokens=10):
    """
    Évalue T5Gemma sur un dataset NLI en mode few-shot.

    Paramètres
    ----------
    data              : Dataset HuggingFace à évaluer
    few_shot_examples : Liste d'exemples pour le contexte few-shot
    prem_key          : Clé de la prémisse dans le dataset
    max_new_tokens    : Nombre max de tokens générés (10 suffit pour '0'/'1'/'2')

    Retourne
    --------
    accuracy, predictions, labels
    """
    predictions = []
    labels = []

    for ex in tqdm(data, desc=f"Éval {len(few_shot_examples)}-shot"):
        prompt = create_prompt(ex[prem_key], ex["hypothesis"], few_shot_examples, prem_key)

        # Tokenisation (texte uniquement, pas d'image)
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Déterministe (greedy)
                num_beams=1,
            )

        # Décoder la réponse
        response_text = processor.decode(output_ids[0], skip_special_tokens=True)
        pred = parse_prediction(response_text)

        predictions.append(pred)
        labels.append(ex["label"])

    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(labels) if labels else 0.0

    return accuracy, predictions, labels


# ─────────────────────────────────────────────
# 6. Expériences
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DÉBUT DE L'EXPÉRIENCE")
print("=" * 60)
print("1. Les splits sont chargés depuis le disque (figés).")
print("2. Exemples pour le contexte → tirés de TRAIN uniquement.")
print("3. Exemples à évaluer        → tirés de VALIDATION uniquement.")
print("4. Le modèle est exécuté localement (pas d'API).\n")

# Sous-ensemble de validation pour aller plus vite (20 exemples)
# Augmentez cette valeur pour une évaluation plus complète
EVAL_SIZE = min(20, len(val_data))
test_subset = val_data.select(range(EVAL_SIZE))
print(f"Évaluation sur {EVAL_SIZE} exemples de validation.\n")

results = {}

# ── Phase 1 : Baseline 0-shot ──────────────────
print("-" * 40)
print("PHASE 1 : BASELINE (0-shot)")
print("-" * 40)
print("Le modèle ne voit AUCUN exemple d'entraînement.\n")

acc_0, preds_0, labels_0 = evaluate_t5gemma(test_subset, [], premise_key)
results["0-shot"] = {"accuracy": acc_0}
print(f"\n>> Précision Baseline (0-shot) : {acc_0:.2%}")

# ── Phase 2 : Few-Shot ─────────────────────────
print("\n" + "-" * 40)
print("PHASE 2 : FEW-SHOT LEARNING")
print("-" * 40)

for n_shots in [1, 3, 5]:
    print(f"\n--- Test avec {n_shots} exemple(s) de contexte ---")
    few_shot_examples = select_stratified_examples(train_data, n_shots)

    acc, _, _ = evaluate_t5gemma(test_subset, few_shot_examples, premise_key)
    results[f"{n_shots}-shot"] = {"accuracy": acc}

    gain = acc - acc_0
    print(f">> Précision {n_shots}-shot : {acc:.2%}  (Gain : {gain:+.2%})")

# ─────────────────────────────────────────────
# 7. Résumé et sauvegarde
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(f"Modèle  : {MODEL_NAME}")
print(f"Dataset : {dataset_name}")
print(f"{'Shot':<10} | {'Précision':<10} | {'Gain':<10}")
print("-" * 35)
for k, v in results.items():
    acc  = v["accuracy"]
    base = results["0-shot"]["accuracy"]
    gain = acc - base
    print(f"{k:<10} | {acc:.2%}    | {gain:+.2%}")

os.makedirs("results", exist_ok=True)
output_path = f"results/t5gemma_{dataset_name}_results.json"
with open(output_path, "w") as f:
    json.dump(
        {
            "model": MODEL_NAME,
            "dataset": dataset_name,
            "eval_size": EVAL_SIZE,
            "results": results,
        },
        f,
        indent=2,
    )

print(f"\n✅ Résultats sauvegardés : {output_path}")
