"""
Fine-tuning FlauBERT - Adapté de CamemBERT

Choisir le dataset: GQNLI-FR ou FraCaS
"""

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score
import json
import os
import torch

print("="*60)
print("FINE-TUNING FLAUBERT")
print("="*60)

# Demander quel dataset
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
try:
    dataset = DatasetDict.load_from_disk(dataset_path)
except Exception as e:
    print(f"❌ Erreur: Impossible de charger {dataset_path}")
    print("Avez-vous lancé 'python3 experiments/setup_data.py' ?")
    exit(1)

train_data = dataset['train']
eval_data = dataset['validation']

print(f"Train: {len(train_data)} exemples")
print(f"Validation: {len(eval_data)} exemples")

# Charger FlauBERT
print("\nChargement de FlauBERT...")
model_name = 'flaubert/flaubert_base_cased' 
# Alternative: flaubert/flaubert_base_uncased, flaubert/flaubert_large_cased
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

print(f"✅ Modèle chargé: {model_name}")

# Tokenization et préparation
def preprocess_function(examples):
    # Tokenize
    result = tokenizer(
        examples[premise_key],
        examples['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_token_type_ids=True  # ⚠️ Added for FlauBERT
    )
    
    # Mapping des labels
    label_map = {
        'yes': 0, 'entailment': 0,
        'unknown': 1, 'undef': 1, 'neutral': 1,
        'no': 2, 'contradiction': 2
    }
    
    def map_label(l):
        if isinstance(l, int): return l
        # Nettoyer et mapper
        l_str = str(l).lower().strip()
        if l_str in label_map:
            return label_map[l_str]
        # Essayer de convertir en int si possible
        try:
            return int(l_str)
        except:
            return 1  # Par défaut neutral
            
    # Ajouter labels (forcer int via mapping)
    result['labels'] = [map_label(l) for l in examples['label']]
    return result

print("\nTokenization et préparation...")
train_dataset = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names
)

eval_dataset = eval_data.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_data.column_names
)

# Set format avec colonnes explicites
# Set format avec colonnes explicites
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])

# Métriques
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Training args
training_args = TrainingArguments(
    output_dir=f'checkpoints/flaubert_{dataset_name}',
    eval_strategy='epoch',
    save_strategy='no',
    learning_rate=1e-5,  # Reduced from 2e-5 for better stability
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,  # Increased from 10 to allow convergence
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    logging_dir=f'logs/flaubert_{dataset_name}',
    logging_steps=2,  # More frequent logging to monitor loss
    report_to='none',
    remove_unused_columns=False,
)

# Trainer
from transformers import DataCollatorWithPadding

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 0. Évaluation AVANT entraînement (Baseline)
print("\n" + "="*60)
print("PHASE 1: BASELINE (Avant entraînement)")
print("="*60)
print("Évaluation du modèle 'vierge' sur le validation set...")
baseline_metrics = trainer.evaluate()
print(f">> Précision Baseline: {baseline_metrics['eval_accuracy']:.2%} (Attendu: ~33% hasard)")

# 1. Fine-tuning
print("\n" + "="*60)
print("PHASE 2: FINE-TUNING")
print("="*60)
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")

trainer.train()

print("\n✅ Fine-tuning terminé!")

# Évaluation
print("\n" + "="*60)
print("ÉVALUATION")
print("="*60)

eval_results = trainer.evaluate()

print(f"Accuracy: {eval_results['eval_accuracy']:.2%}")

# Sauvegarder
print("\nSauvegarde...")
# Create dirs
os.makedirs(f'models/flaubert_{dataset_name}', exist_ok=True)

trainer.save_model(f'models/flaubert_{dataset_name}')
tokenizer.save_pretrained(f'models/flaubert_{dataset_name}')

print(f"✅ Modèle: models/flaubert_{dataset_name}")

# Résultats
os.makedirs('results', exist_ok=True)
with open(f'results/flaubert_{dataset_name}_results.json', 'w') as f:
    json.dump({
        'model': model_name,
        'dataset': dataset_name,
        'train_size': len(train_data),
        'eval_size': len(eval_data),
        'accuracy': eval_results['eval_accuracy'],
        'epochs': training_args.num_train_epochs,
    }, f, indent=2)

print(f"✅ Résultats: results/flaubert_{dataset_name}_results.json")
