"""
Fine-Tuning T5Gemma 2 (270M) pour la tâche NLI en français

T5Gemma est un modèle encoder-decoder : contrairement à CamemBERT
(classification directe), on l'entraîne à GÉNÉRER le bon label ("0", "1" ou "2")
à partir du texte de la prémisse et de l'hypothèse.

Référence modèle : https://huggingface.co/google/t5gemma-2-270m-270m

Utilisation :
    python3 experiments/fine_tuning/finetune_t5gemma.py

Prérequis :
    pip install -U transformers torch accelerate
    hf auth login   # Token HuggingFace avec accès à google/t5gemma-2-270m-270m
"""

from datasets import DatasetDict
from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch
import numpy as np
import json
import os

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
MODEL_NAME = "google/t5gemma-2-270m-270m"

# Mapping label → texte généré par le modèle
# T5Gemma génère du texte, donc on lui apprend à générer "0", "1" ou "2"
LABEL_MAP = {0: "0", 1: "1", 2: "2"}
LABEL_MAP_STR = {"yes": "0", "entailment": "0",
                 "unknown": "1", "undef": "1", "neutral": "1",
                 "no": "2", "contradiction": "2"}

print("=" * 60)
print("FINE-TUNING T5GEMMA 2 (270M)")
print("=" * 60)
print(f"Modèle : {MODEL_NAME}")

# Détecter le device
if torch.cuda.is_available():
    device = "cuda"
    print(f"Device : GPU ({torch.cuda.get_device_name(0)})")
else:
    device = "cpu"
    print("Device : CPU (plus lent, ~20-30 min)")

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
    premise_key  = "premise"
    print("\n📊 Dataset : GQNLI-FR")
elif choice == "2":
    dataset_name = "fracas_gq"
    dataset_path = "data/processed/fracas_gq"
    premise_key  = "premises"
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
eval_data  = dataset["validation"]

print(f"Train      : {len(train_data)} exemples")
print(f"Validation : {len(eval_data)} exemples")

# ─────────────────────────────────────────────
# 4. Chargement du modèle et du processor
# ─────────────────────────────────────────────
print(f"\nChargement de {MODEL_NAME}...")
print("(Premier téléchargement ~600 Mo — soyez patient)")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer  # Le tokenizer texte du processor

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)
# Forcer float32 sur tous les poids (device_map peut ignorer torch_dtype)
model = model.float()

print(f"✅ Modèle chargé")

# ─────────────────────────────────────────────
# 5. Préparation des données
# ─────────────────────────────────────────────

def normalize_label(label) -> str:
    """Convertit n'importe quel format de label en '0', '1' ou '2'."""
    if isinstance(label, int):
        return str(label) if label in [0, 1, 2] else "1"
    label_str = str(label).lower().strip()
    if label_str in LABEL_MAP_STR:
        return LABEL_MAP_STR[label_str]
    try:
        val = int(label_str)
        return str(val) if val in [0, 1, 2] else "1"
    except ValueError:
        return "1"  # Neutre par défaut


def preprocess(examples):
    """
    Prépare les exemples pour T5Gemma.

    Entrée  (input_ids)  : "nli: <prémisse> </s> <hypothèse>"
    Sortie  (labels)     : "0" / "1" / "2"
    """
    premises    = examples[premise_key]
    hypotheses  = examples["hypothesis"]
    raw_labels  = examples["label"]

    # Construire les prompts d'entrée
    inputs = [
        f"nli: {p} </s> {h}"
        for p, h in zip(premises, hypotheses)
    ]

    # Tokeniser les entrées
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding=False,
    )

    # Tokeniser les labels (texte "0", "1" ou "2")
    target_texts = [normalize_label(l) for l in raw_labels]
    labels = tokenizer(
        text_target=target_texts,
        max_length=4,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("\nTokenisation des données...")
train_dataset = train_data.map(
    preprocess,
    batched=True,
    remove_columns=train_data.column_names,
    desc="Train",
)
eval_dataset = eval_data.map(
    preprocess,
    batched=True,
    remove_columns=eval_data.column_names,
    desc="Validation",
)

print(f"✅ Train tokenisé : {len(train_dataset)} exemples")
print(f"✅ Val tokenisé   : {len(eval_dataset)} exemples")

# ─────────────────────────────────────────────
# 6. Métriques
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Calcule l'accuracy en décodant les tokens générés."""
    predictions, labels = eval_pred

    # Remplacer les -100 (padding) par le token pad
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Décoder prédictions et labels
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)

    # Nettoyer les espaces
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Calculer l'accuracy
    correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_labels) if decoded_labels else 0.0

    return {"accuracy": accuracy}


# ─────────────────────────────────────────────
# 7. Arguments d'entraînement
# ─────────────────────────────────────────────
output_dir = f"checkpoints/t5gemma_{dataset_name}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=3e-4,
    per_device_train_batch_size=1,       # ⬇️ Réduit de 4 → 1 pour économiser la RAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,       # Simule batch_size=4 sans charger 4x en RAM
    num_train_epochs=3,                  # Réduit 10 → 3 (plus rapide, moins de RAM)
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=4,
    logging_steps=10,
    report_to="none",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,         # Économise ~30% de RAM supplémentaire
    dataloader_pin_memory=False,         # Désactive pin_memory (inutile sur CPU)
)

# ─────────────────────────────────────────────
# 8. Trainer
# ─────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None,   # None = ne pas appeler prepare_decoder_input_ids_from_labels (incompatible avec T5Gemma)
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ─────────────────────────────────────────────
# 9. Évaluation AVANT entraînement (Baseline)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 : BASELINE (avant entraînement)")
print("=" * 60)
print("Évaluation du modèle 'vierge' sur le validation set...")

baseline = trainer.evaluate()
print(f">> Précision Baseline : {baseline['eval_accuracy']:.2%}  (Attendu : ~33% hasard)")

# ─────────────────────────────────────────────
# 10. Fine-Tuning
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING")
print("=" * 60)
print(f"Epochs      : {training_args.num_train_epochs}")
print(f"Batch size  : {training_args.per_device_train_batch_size}")
print(f"Learning rate : {training_args.learning_rate}")
print(f"Exemples train : {len(train_dataset)}\n")

trainer.train()

print("\n✅ Fine-tuning terminé!")

# ─────────────────────────────────────────────
# 11. Évaluation finale
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉVALUATION FINALE")
print("=" * 60)

eval_results = trainer.evaluate()
print(f"Accuracy finale : {eval_results['eval_accuracy']:.2%}")

# ─────────────────────────────────────────────
# 12. Sauvegarde du modèle et des résultats
# ─────────────────────────────────────────────
print("\nSauvegarde du modèle...")
model_save_path = f"models/t5gemma_{dataset_name}"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✅ Modèle sauvegardé : {model_save_path}")

os.makedirs("results", exist_ok=True)
results_path = f"results/t5gemma_finetune_{dataset_name}_results.json"
with open(results_path, "w") as f:
    json.dump(
        {
            "model": MODEL_NAME,
            "dataset": dataset_name,
            "train_size": len(train_data),
            "eval_size": len(eval_data),
            "baseline_accuracy": baseline["eval_accuracy"],
            "final_accuracy": eval_results["eval_accuracy"],
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
        },
        f,
        indent=2,
    )

print(f"✅ Résultats : {results_path}")

print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print(f"Modèle   : {MODEL_NAME}")
print(f"Dataset  : {dataset_name}")
print(f"Baseline : {baseline['eval_accuracy']:.2%}")
print(f"Après FT : {eval_results['eval_accuracy']:.2%}")
print(f"Gain     : {eval_results['eval_accuracy'] - baseline['eval_accuracy']:+.2%}")
