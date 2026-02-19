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
print("FINE-TUNING / EVALUATION FLAUBERT")
print("="*60)

# Mode selection
print("\nQuel mode?")
print("1. Fine-tuning + Evaluation (Default)")
print("2. Evaluation Only (Load Checkpoint)")
mode_choice = input("\nVotre choix (1 ou 2): ").strip()

if mode_choice == "2":
    mode = "eval_only"
    print("\n🔹 Mode: Evaluation Only")
else:
    mode = "train_eval"
    print("\n🔹 Mode: Train & Eval")

# Dataset selection
print("\nQuel dataset pour l'évaluation?")
print("1. GQNLI-FR")
print("2. FraCaS GQ")
print("3. FraCaS (Lignes 0-74)")

choice = input("\nVotre choix (1, 2 ou 3): ").strip()

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
elif choice == "3":
    dataset_name = "fracas_subset_75"
    dataset_path = "data/processed/fracas_subset_75"
    premise_key = "premises"
    print("\n📊 Dataset: FraCaS (Lignes 0-74)")
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

train_data = dataset.get('train')
eval_data = dataset.get('validation')
test_data = dataset.get('test') # Added for completeness

print(f"Train: {len(train_data) if train_data else 0} exemples")
print(f"Validation: {len(eval_data) if eval_data else 0} exemples")
print(f"Test: {len(test_data) if test_data else 0} exemples")

# Select model
if mode == "eval_only":
    default_ckpt = f"models/flaubert_gqnli_fr"
    print(f"\nChemin du checkpoint (défaut: {default_ckpt})")
    print("Exemple: models/flaubert_gqnli_fr")
    ckpt_path = input("Chemin: ").strip()
    if not ckpt_path:
        ckpt_path = default_ckpt
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Erreur: Le checkpoint {ckpt_path} n'existe pas.")
        exit(1)
        
    print(f"Chargement du checkpoint depuis {ckpt_path}...")
    model_name = ckpt_path
else:
    # Training mode
    print("\nChargement de FlauBERT (Base)...")
    model_name = 'flaubert/flaubert_base_cased' 

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
        return_token_type_ids=True
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
tokenized_datasets = {}

if train_data:
    tokenized_datasets['train'] = train_data.map(
        preprocess_function, batched=True, remove_columns=train_data.column_names
    )
if eval_data:
    tokenized_datasets['validation'] = eval_data.map(
        preprocess_function, batched=True, remove_columns=eval_data.column_names
    )

# Set format
for split in tokenized_datasets:
    tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])

train_dataset = tokenized_datasets.get('train')
eval_dataset = tokenized_datasets.get('validation')


# Métriques
from sklearn.metrics import accuracy_score, confusion_matrix

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate confusion matrix for detailed analysis
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    print("\n\n📊 CONFUSION MATRIX:")
    print("Pred ->  0 (Ent)  1 (Neu)  2 (Con)")
    print(f"True 0:  {cm[0][0]:<8} {cm[0][1]:<8} {cm[0][2]:<8}")
    print(f"True 1:  {cm[1][0]:<8} {cm[1][1]:<8} {cm[1][2]:<8}")
    print(f"True 2:  {cm[2][0]:<8} {cm[2][1]:<8} {cm[2][2]:<8}")
    
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Training args
training_args = TrainingArguments(
    output_dir=f'checkpoints/flaubert_{dataset_name}',
    eval_strategy='epoch',
    save_strategy='no',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    logging_dir=f'logs/flaubert_{dataset_name}',
    logging_steps=2,
    report_to='none',
    remove_unused_columns=False,
    save_total_limit=1,
)

# Trainer
from transformers import DataCollatorWithPadding

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if mode == "train_eval" else None,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

if mode == "train_eval":
    # 0. Évaluation AVANT entraînement (Baseline)
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (Avant entraînement)")
    print("="*60)
    print("Évaluation du modèle 'vierge' sur le validation set...")
    if eval_dataset:
        baseline_metrics = trainer.evaluate()
        print(f">> Précision Baseline: {baseline_metrics['eval_accuracy']:.2%} (Attendu: ~33% hasard)")
    else:
        print("⚠️ Pas de dataset de validation, impossible d'évaluer la baseline.")

    # 1. Fine-tuning
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING")
    print("="*60)
    if train_dataset:
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        
        trainer.train()
        print("\n✅ Fine-tuning terminé!")
        
        # Sauvegarder
        print("\nSauvegarde...")
        os.makedirs(f'models/flaubert_{dataset_name}', exist_ok=True)
        trainer.save_model(f'models/flaubert_{dataset_name}')
        tokenizer.save_pretrained(f'models/flaubert_{dataset_name}')
        print(f"✅ Modèle: models/flaubert_{dataset_name}")
    else:
        print("⚠️ Pas de dataset d'entraînement!")


# Évaluation Final (Common to both modes)
print("\n" + "="*60)
print("ÉVALUATION FINALE")
print("="*60)

if eval_dataset:
    # DEBUG: Inspect data
    print("\n🔍 DEBUG: Inspecting first 5 examples...")
    for i in range(min(5, len(eval_dataset))):
        example = eval_dataset[i]
        input_ids = example['input_ids']
        label = example['labels']
        decoded = tokenizer.decode(input_ids)
        print(f"\nExample {i}:")
        print(f"  Label (Int): {label}")
        print(f"  Text: {decoded[:200]}...") # Truncate for readability
        
        # Predict validation
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0).to(model.device), attention_mask=example['attention_mask'].unsqueeze(0).to(model.device))
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            print(f"  Prediction: {pred} (Logits: {logits.cpu().numpy()})")
            
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['eval_accuracy']:.2%}")
    
    # Résultats
    os.makedirs('results', exist_ok=True)
    result_filename = f'results/flaubert_{dataset_name}_{mode}.json'
    with open(result_filename, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'mode': mode,
            'eval_size': len(eval_dataset),
            'accuracy': eval_results['eval_accuracy'],
        }, f, indent=2)
    print(f"✅ Résultats: {result_filename}")
else:
    print("⚠️ Pas de dataset d'évaluation!")


