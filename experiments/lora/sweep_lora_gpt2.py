"""
Script de Sweep WandB pour GPT-2 French (Architecture Decoder-only 117M as Classifier).
Même taille de réseau que CamemBERT pour une comparaison parfaitement équitable !
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force 1 seul GPU

import sys
import numpy as np
import torch
import torch.nn as nn
import wandb
import shutil

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Modele de 117M parametres entraine 100% en Francais
BASE_MODEL = "ClassCat/gpt2-base-french"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA.")
    exit(1)

# 1. TÉLÉCHARGEMENT & PRÉPARATION

def get_dataset(name):
    print(f"Téléchargement de {name}...")
    if name == "gqnli_fr":
        gqnli = load_dataset('maximoss/gqnli-fr')['test']
        train_idx = list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))
        val_idx   = list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))
        test_idx  = list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))
        ds = DatasetDict({
            'train': gqnli.select(train_idx).shuffle(seed=42),
            'validation': gqnli.select(val_idx).shuffle(seed=42),
            'test': gqnli.select(test_idx).shuffle(seed=42)
        })
        return ds, "premise"
        
    elif name == "fracas_75":
        fracas = load_dataset('maximoss/fracas')['train']
        ds = DatasetDict({
            'train': fracas.select(range(75)), 
            'validation': fracas.select(range(75, 100)), 
            'test': fracas.select(range(100, 150))
        })
        return ds, "premises"
        
    elif name == "daccord":
        data = load_dataset('maximoss/daccord-contradictions')['train']
        # DACCORD contient des labels (0, 1). 1 = Contradiction. On ramène à 2 pour standardiser multi-classes.
        def fix_daccord_label(ex):
            if ex["label"] == 1:
                ex["label"] = 2
            return ex
        data = data.map(fix_daccord_label)
        data = data.shuffle(seed=42)
        total = len(data)
        train_size, val_size = int(total * 0.6), int(total * 0.2)
        ds = DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        })
        return ds, "premise"
        
    elif name == "rte3_fr":
        splits = list(load_dataset('maximoss/rte3-french').values())
        data = concatenate_datasets(splits).shuffle(seed=42)
        total = len(data)
        train_size, val_size = int(total * 0.6), int(total * 0.2)
        ds = DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        })
        return ds, "premise"
        
    raise ValueError(f"Dataset {name} inconnu.")

# Balayage strictement identique a celui de CamemBERT
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [16]},
        "lora_alpha": {"values": [32]},
        "learning_rate": {"values": [3e-4, 5e-4]},
        "lora_dropout": {"values": [0.1]},
    }
}

if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
    print(f"\nVotre choix: {exp_choice} (via argument)")
else:
    print("\nQuelle expérience LoRA GPT-2 utiliser ?")
    print("1. FraCaS (0-74)  →  test GQNLI-FR")
    print("2. GQNLI-FR       →  test FraCaS (0-74)")
    print("3. RTE3-DEV       →  test DACCORD (Binaire)")
    exp_choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if exp_choice == "1":
    EXP_NAME = "sweep_fracas_to_gqnli_gpt2"
    train_ds_name, test_ds_name = "fracas_75", "gqnli_fr"
elif exp_choice == "2":
    EXP_NAME = "sweep_gqnli_to_fracas_gpt2"
    train_ds_name, test_ds_name = "gqnli_fr", "fracas_75"
elif exp_choice == "3":
    EXP_NAME = "sweep_rte3_to_daccord_gpt2"
    train_ds_name, test_ds_name = "rte3_fr", "daccord"
else:
    print("Choix invalide!"); exit(1)

TRAIN_DICT, TRAIN_PKEY = get_dataset(train_ds_name)
TEST_DICT, TEST_PKEY = get_dataset(test_ds_name)

is_binary = (exp_choice == "3")

if is_binary:
    # Retour aux valeurs normales et stables comme CamemBERT
    SWEEP_CONFIG["parameters"]["loss_penalty"] = {"values": [1.0, 3.0, 5.0, 10.0]}
    
if is_binary:
    LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 0, "undef": 0, "neutral": 0, "no": 1, "contradiction": 1}
else:
    LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 1, "undef": 1, "neutral": 1, "no": 2, "contradiction": 2}

global_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Important pour decoder: Fixer le pad token sur l'eos
if global_tokenizer.pad_token is None:
    global_tokenizer.pad_token = global_tokenizer.eos_token
# Pour un Classifier, on met generalement la padding side a la fin (right) sur les Decodeurs
global_tokenizer.padding_side = "right"

def map_label(label):
    if isinstance(label, int):
        if is_binary:
            return 1 if label == 2 else 0
        return label if label in [0, 1, 2] else 1
    s = str(label).lower().strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    return 1 if not is_binary else 0

def tokenize_fn(examples, p_key):
    prompts = [f"Prémisse : {p}\nHypothèse : {h}\n" for p, h in zip(examples[p_key], examples["hypothesis"])]
    res = global_tokenizer(prompts, truncation=True, max_length=256)
    res["labels"] = [map_label(l) for l in examples["label"]]
    return res

print("\nTokenisation des datasets...")
train_data = TRAIN_DICT['train'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = TRAIN_DICT['validation'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_untokenized = TEST_DICT['test']
test_data = test_untokenized.map(lambda ex: tokenize_fn(ex, TEST_PKEY), batched=True, remove_columns=test_untokenized.column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    try:
        cm_labels = [0, 1] if is_binary else [0, 1, 2]
        cm = confusion_matrix(labels, predictions, labels=cm_labels)
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception: pass
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="macro")
    }
    return metrics

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        penalty = float(getattr(wandb.config, "loss_penalty", 1.0))
        
        if is_binary:
            weight = torch.tensor([1.0, penalty], device=labels.device, dtype=torch.float)
        else:
            weight = None
            
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 4. FONCTION D'ENTRAÎNEMENT WANDB
def train_gpt2_lora():
    run = wandb.init()
    config = wandb.config

    if is_binary:
        run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}_p{config.loss_penalty}"
    else:
        run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
        
    print(f"\n{'='*60}\nSWEEP GPT-2 RUN: {run_label}\n{'='*60}")

    print("Chargement du modèle Décodeur pur...")
    num_labels = 2 if is_binary else 3
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        device_map="auto"
        # PAS DE BitsAndBytesConfig (Pas de QLoRA) car le modele fait exactement la meme taille (110M) que Camembert !
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = global_tokenizer.pad_token_id
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["c_attn"], # Dans GPT-2 la couche d'attention principale s'appelle c_attn
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    args = TrainingArguments(
        output_dir=f"/tmp/ckpt_{EXP_NAME}_{run_label}", 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20, # On re-augmente a 20 car le modele de 117M s'entraine extremement vite (comme CamemBERT)
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nÉvaluation Cross-Dataset Finale...")
    test_results = trainer.evaluate(test_data, metric_key_prefix="test")
    test_acc = test_results["test_accuracy"]
    test_f1 = test_results["test_f1_score"]
    
    print(f"FINAL TEST ACCURACY : {test_acc:.2%}")
    print(f"FINAL TEST F1-SCORE : {test_f1:.4f}")

    # Le Trainer avec son WandbCallback vient d'envoyer automatiquement les resultats "test_accuracy" a WandB !
    wandb.summary["final_test_accuracy"] = test_acc
    wandb.summary["final_test_f1_score"] = test_f1

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    total_runs = 1
    for param_name, param in SWEEP_CONFIG["parameters"].items():
        if "values" in param:
            total_runs *= len(param["values"])
    
    if len(sys.argv) > 1:
        confirm = "o" 
    else:
        confirm = input(f"\nLancer {total_runs} sweeps GPT-2 LoRA ? (o/n): ").strip().lower()
        
    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_gpt2_lora, count=total_runs)
        print("\nTerminé ! Consultez les résultats sur WandB.")
    else:
        print("Annulé.")
