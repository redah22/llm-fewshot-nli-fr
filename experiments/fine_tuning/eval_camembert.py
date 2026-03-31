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
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os
import torch

print("="*60)
print("FINE-TUNING / EVALUATION CAMEMBERT")
print("="*60)

# Mode selection
print("\nQuel mode?")
print("1. Fine-tuning + Evaluation (cross-dataset)")
print("2. Evaluation Only (Load Checkpoint)")
print("3. Fine-tuning + Evaluation (même dataset, split 70/15/15)")
mode_choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if mode_choice == "2":
    mode = "eval_only"
    print("\n🔹 Mode: Evaluation Only")
elif mode_choice == "3":
    mode = "same_dataset"
    print("\n🔹 Mode: Same-Dataset Split (70% train / 15% val / 15% test)")
else:
    mode = "train_eval"
    print("\n🔹 Mode: Train & Eval (cross-dataset)")

custom_result_name = input("\nNom du fichier de résultats (laissez vide pour générer automatiquement) : ").strip()


# ---------------------------------------------------------
# SÉLECTION DES DATASETS
# ---------------------------------------------------------

def ask_dataset(type_str="l'évaluation"):
    print(f"\nQuel dataset pour {type_str}?")
    print("1. GQNLI-FR")
    print("2. FraCaS GQ")
    print("3. FraCaS (Lignes 0-74)")
    print("4. DACCORD")
    print("5. RTE3-French")
    
    c = input(f"\nVotre choix (1, 2, 3, 4 ou 5): ").strip()
    
    name = ""
    path = ""
    pkey = ""
    
    if c == "1":
        name = "gqnli_fr"
        path = "data/processed/gqnli_fr"
        pkey = "premise"
        print(f"\n📊 Dataset {type_str}: GQNLI-FR")
    elif c == "2":
        name = "fracas_gq"
        path = "data/processed/fracas_gq"
        pkey = "premises"
        print(f"\n📊 Dataset {type_str}: FraCaS GQ")
    elif c == "3":
        name = "fracas_subset_75"
        path = "data/processed/fracas_subset_75"
        pkey = "premises"
        print(f"\n📊 Dataset {type_str}: FraCaS (Lignes 0-74)")
    elif c == "4":
        name = "daccord"
        path = "data/processed/daccord"
        pkey = "premise"
        print(f"\n📊 Dataset {type_str}: DACCORD")
    elif c == "5":
        name = "rte3_fr"
        path = "data/processed/rte3_fr"
        pkey = "premise"
        print(f"\n📊 Dataset {type_str}: RTE3-French")
    else:
        print("❌ Choix invalide!")
        exit(1)
    return name, path, pkey

train_dataset_name = None
train_data = None
train_premise_key = None

eval_dataset_name = None
eval_data = None
eval_premise_key = None
test_data = None


# Variables for same_dataset mode
test_dataset = None

if mode == "train_eval":
    # 1. Training Dataset
    train_dataset_name, train_path, train_premise_key = ask_dataset("l'entraînement")
    try:
        ds_train = DatasetDict.load_from_disk(train_path)
        train_data = ds_train['train']
    except Exception as e:
        print(f"❌ Erreur chargement TRAIN: {e}")
        exit(1)

    # 2. Evaluation Dataset
    eval_dataset_name, eval_path, eval_premise_key = ask_dataset("l'évaluation")
    try:
        ds_eval = DatasetDict.load_from_disk(eval_path)
        eval_data = ds_eval['validation']
        test_data = ds_eval.get('test')
    except Exception as e:
        print(f"❌ Erreur chargement EVAL: {e}")
        exit(1)

    dataset_name = train_dataset_name

elif mode == "same_dataset":
    # Single dataset split into train / val / test (70 / 15 / 15)
    eval_dataset_name, eval_path, eval_premise_key = ask_dataset("le split")
    train_dataset_name = eval_dataset_name
    train_premise_key = eval_premise_key
    dataset_name = eval_dataset_name

    try:
        ds_all = DatasetDict.load_from_disk(eval_path)
        # Concatenate all available splits to get the full dataset
        from datasets import concatenate_datasets
        all_splits = list(ds_all.values())
        full_ds = concatenate_datasets(all_splits).shuffle(seed=42)
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        exit(1)

    n = len(full_ds)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))
    n_train = n - n_test - n_val

    train_data = full_ds.select(range(n_train))
    eval_data  = full_ds.select(range(n_train, n_train + n_val))
    test_data  = full_ds.select(range(n_train + n_val, n))

    print(f"\n✂️  Split: {n_train} train / {n_val} val / {len(test_data)} test (seed=42)")

else:
    # Eval Only
    eval_dataset_name, eval_path, eval_premise_key = ask_dataset("l'évaluation")
    try:
        ds_eval = DatasetDict.load_from_disk(eval_path)
        eval_data = ds_eval['validation']
        test_data = ds_eval.get('test')
    except Exception as e:
        print(f"❌ Erreur chargement EVAL: {e}")
        exit(1)

    dataset_name = eval_dataset_name

print(f"\nTrain: {len(train_data) if train_data else 0} exemples")
print(f"Validation: {len(eval_data) if eval_data else 0} exemples")
if mode == "same_dataset" and test_data is not None:
    print(f"Test (hold-out): {len(test_data)} exemples")



# Select model
if mode == "eval_only":
    default_ckpt = f"models/camembert_gqnli_fr"
    print(f"\nChemin du checkpoint (défaut: {default_ckpt})")
    print("Exemple: models/camembert_gqnli_fr")
    print("Ou tapez 'base' pour tester CamemBERT sans fine-tuning.")
    ckpt_path = input("Chemin: ").strip()
    
    if not ckpt_path:
        ckpt_path = default_ckpt
    
    if ckpt_path.lower() in ["base", "camembert-base"]:
        model_name = "camembert-base"
        print(f"Chargement de {model_name} (Baseline)...")
    else:
        if not os.path.exists(ckpt_path):
            print(f"❌ Erreur: Le checkpoint {ckpt_path} n'existe pas.")
            exit(1)
            
        print(f"Chargement du checkpoint depuis {ckpt_path}...")
        model_name = ckpt_path
else:
    # Training mode
    print("\nChargement de CamemBERT (Base)...")
    model_name = 'camembert-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

print(f"✅ Modèle chargé: {model_name}")


# Tokenization et préparation
def get_tokenize_function(p_key):
    def tokenize_func(examples):
        # Tokenize
        result = tokenizer(
            examples[p_key],
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
    return tokenize_func

print("\nTokenization et préparation...")
tokenized_datasets = {}

if train_data:
    print(f"Tokenizing TRAIN ({len(train_data)} examples using key '{train_premise_key}')")
    tokenized_datasets['train'] = train_data.map(
        get_tokenize_function(train_premise_key),
        batched=True,
        remove_columns=train_data.column_names
    )
if eval_data:
    print(f"Tokenizing EVAL ({len(eval_data)} examples using key '{eval_premise_key}')")
    tokenized_datasets['validation'] = eval_data.map(
        get_tokenize_function(eval_premise_key),
        batched=True,
        remove_columns=eval_data.column_names
    )
if mode == "same_dataset" and test_data is not None:
    print(f"Tokenizing TEST ({len(test_data)} examples using key '{eval_premise_key}')")
    tokenized_datasets['test'] = test_data.map(
        get_tokenize_function(eval_premise_key),
        batched=True,
        remove_columns=test_data.column_names
    )


# Set format
for split in tokenized_datasets:
    tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset = tokenized_datasets.get('train')
eval_dataset = tokenized_datasets.get('validation')
test_dataset = tokenized_datasets.get('test')

# Métriques
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
    output_dir=f'checkpoints/camembert_{dataset_name}',
    eval_strategy='epoch',
    save_strategy='no',  # ⚠️ Désactivé pour éviter le crash "segmentation fault" sur Mac
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Augmenté à 10 pour mieux apprendre (small data)
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    logging_dir=f'logs/camembert_{dataset_name}',
    logging_steps=5,
    report_to='none',
    remove_unused_columns=False,  # ⚠️ IMPORTANT
    save_total_limit=1,
)

# Trainer
from transformers import DataCollatorWithPadding

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if mode in ("train_eval", "same_dataset") else None,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)


# 0. Évaluation AVANT entraînement (Baseline)
baseline_acc = None

if mode in ("train_eval", "same_dataset"):
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (Avant entraînement)")
    print("="*60)
    eval_target_for_baseline = eval_dataset
    print("Évaluation du modèle 'vierge' sur le validation set...")
    if eval_target_for_baseline:
        baseline_metrics = trainer.evaluate()
        baseline_acc = baseline_metrics['eval_accuracy']
        print(f">> Précision Baseline: {baseline_acc:.2%} (Attendu: ~33% hasard)")
    else:
        print("⚠️ Pas de dataset de validation, impossible d'évaluer la baseline.")

    # Fine-tuning
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
        os.makedirs(f'models/camembert_{dataset_name}', exist_ok=True)
        trainer.save_model(f'models/camembert_{dataset_name}')
        tokenizer.save_pretrained(f'models/camembert_{dataset_name}')
        print(f"✅ Modèle: models/camembert_{dataset_name}")
    else:
        print("⚠️ Pas de dataset d'entraînement!")

# Évaluation Finale
print("\n" + "="*60)
print("ÉVALUATION FINALE")
print("="*60)

# In same_dataset mode, evaluate on the held-out test set
# In other modes, evaluate on the validation set as before
final_eval_dataset = test_dataset if (mode == "same_dataset" and test_dataset is not None) else eval_dataset
final_eval_split_name = "test (hold-out)" if (mode == "same_dataset" and test_dataset is not None) else "validation"

if final_eval_dataset:
    print(f"\n📋 Évaluation sur le set: {final_eval_split_name} ({len(final_eval_dataset)} exemples)")

    # Pour évaluer sur le test set en mode same_dataset, on remplace temporairement
    # le eval_dataset du Trainer
    if mode == "same_dataset" and test_dataset is not None:
        trainer.eval_dataset = test_dataset

    # DEBUG: Inspect first 5 examples
    print("\n🔍 DEBUG: Inspecting first 5 examples...")
    for i in range(min(5, len(final_eval_dataset))):
        example = final_eval_dataset[i]
        input_ids = example['input_ids']
        label = example['labels']
        decoded = tokenizer.decode(input_ids)
        print(f"\nExample {i}:")
        print(f"  Label (Int): {label}")
        print(f"  Text: {decoded[:200]}...")

        with torch.no_grad():
            outputs = model(
                input_ids.unsqueeze(0).to(model.device),
                attention_mask=example['attention_mask'].unsqueeze(0).to(model.device)
            )
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            print(f"  Prediction: {pred} (Logits: {logits.cpu().numpy()})")

    eval_results = trainer.evaluate()
    final_accuracy = eval_results['eval_accuracy']
    print(f"Accuracy ({final_eval_split_name}): {final_accuracy:.2%}")
    if baseline_acc is not None:
        print(f"(Baseline validation était: {baseline_acc:.2%})")

    # Résultats
    os.makedirs('results', exist_ok=True)

    is_baseline_run = (mode == "eval_only" and model_name == "camembert-base")
    baseline_suffix = "_baseline" if is_baseline_run else ""
    
    if custom_result_name:
        if not custom_result_name.endswith('.json'):
            custom_result_name += '.json'
        result_filename = f'results/{custom_result_name}'
    else:
        result_filename = f'results/camembert_{dataset_name}_{mode}{baseline_suffix}.json'
        
    status = "baseline" if is_baseline_run else "fine-tuned"

    data_to_save = {
        'model': model_name,
        'dataset': dataset_name,
        'mode': mode,
        'training_status': status,
        'eval_split': final_eval_split_name,
        'eval_size': len(final_eval_dataset),
        'accuracy': final_accuracy,
    }

    if mode == "same_dataset":
        data_to_save['split_info'] = {
            'train': len(train_data),
            'val': len(eval_data),
            'test': len(test_data),
            'seed': 42
        }

    if baseline_acc is not None:
        data_to_save['baseline_accuracy'] = baseline_acc

    with open(result_filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f"✅ Résultats: {result_filename}")
else:
    print("⚠️ Pas de dataset d'évaluation!")
