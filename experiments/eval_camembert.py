"""
Fine-tuning CamemBERT - VERSION CORRIGÉE

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

print("="*60)
print("FINE-TUNING CAMEMBERT")
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
dataset = DatasetDict.load_from_disk(dataset_path)

train_data = dataset['train']
eval_data = dataset['validation']

print(f"Train: {len(train_data)} exemples")
print(f"Validation: {len(eval_data)} exemples")

# Charger CamemBERT
print("\nChargement de CamemBERT...")
model_name = 'camembert-base'
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
        max_length=128
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

print(f"Colonnes train: {train_dataset.column_names}")

# Set format avec colonnes explicites
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Vérification debug
ex0 = train_dataset[0]
print("Exemple de donnée train (keys):", ex0.keys())
print("Type label:", type(ex0['labels']), ex0['labels'])


# Métriques
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Training args
training_args = TrainingArguments(
    output_dir=f'checkpoints/camembert_{dataset_name}',
    eval_strategy='epoch',
    save_strategy='no',  # ⚠️ Désactivé pour éviter le crash "segmentation fault" sur Mac
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Augmenté à 10 pour mieux apprendre (small data)
    weight_decay=0.01,
    load_best_model_at_end=False,  # Désactivé car on ne sauvegarde plus de checkpoints
    metric_for_best_model='accuracy',
    logging_dir=f'logs/camembert_{dataset_name}',
    logging_steps=5,
    report_to='none',
    remove_unused_columns=False,  # ⚠️ IMPORTANT
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

# Vérification debug
print("\nExemple de donnée train (keys):", train_dataset[0].keys())

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
trainer.save_model(f'models/camembert_{dataset_name}')
tokenizer.save_pretrained(f'models/camembert_{dataset_name}')

print(f"✅ Modèle: models/camembert_{dataset_name}")

# Résultats
import json
import os

os.makedirs('results', exist_ok=True)
with open(f'results/camembert_{dataset_name}_results.json', 'w') as f:
    json.dump({
        'model': model_name,
        'dataset': dataset_name,
        'train_size': len(train_data),
        'eval_size': len(eval_data),
        'accuracy': eval_results['eval_accuracy'],
        'epochs': training_args.num_train_epochs,
    }, f, indent=2)

print(f"✅ Résultats: results/camembert_{dataset_name}_results.json")
