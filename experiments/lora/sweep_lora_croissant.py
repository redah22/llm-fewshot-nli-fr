"""
Script de Sweep WandB pour CroissantLLM (Architecture Decoder-only).
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force 1 seul GPU

import sys
import numpy as np
import torch
import wandb
import shutil

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_MODEL = "croissantllm/CroissantLLMBase"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA pour QLoRA.")
    print("Veuillez activer le GPU T4 x2 (ou P100) dans les paramètres de votre Notebook Kaggle.")
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
        fracas = load_dataset('maximoss/fracas')['train'].select(range(75))
        ds = DatasetDict({'train': fracas, 'validation': fracas, 'test': fracas})
        return ds, "premises"
        
    elif name == "daccord":
        data = load_dataset('maximoss/daccord-contradictions')['train'].shuffle(seed=42)
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

# Grille ciblée: 16 runs. On a éliminé lr=1e-4 et lora_dropout=0.0
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [8, 16]},
        "lora_alpha": {"values": [32, 64]},
        "learning_rate": {"values": [5e-4, 1e-3]},
        "lora_dropout": {"values": [0.05, 0.1]},
    }
}

if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
    print(f"\nVotre choix (1, 2 ou 3): {exp_choice} (via argument)")
else:
    print("\nQuelle expérience QLoRA CroissantLLM utiliser ?")
    print("1. FraCaS (0-74)  →  test GQNLI-FR")
    print("2. GQNLI-FR       →  test FraCaS (0-74)")
    print("3. RTE3-DEV       →  test DACCORD + RTE3-TEST")
    exp_choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if exp_choice == "1":
    EXP_NAME = "sweep_fracas_to_gqnli_croissant"
    train_ds_name, test_ds_name = "fracas_75", "gqnli_fr"
elif exp_choice == "2":
    EXP_NAME = "sweep_gqnli_to_fracas_croissant"
    train_ds_name, test_ds_name = "gqnli_fr", "fracas_75"
elif exp_choice == "3":
    EXP_NAME = "sweep_rte3_to_daccord_croissant"
    train_ds_name, test_ds_name = "rte3_fr", "daccord"
else:
    print("Choix invalide!"); exit(1)

TRAIN_DICT, TRAIN_PKEY = get_dataset(train_ds_name)
TEST_DICT, TEST_PKEY = get_dataset(test_ds_name)

global_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Causal LMs n'ont généralement pas de token de padding par défaut
if global_tokenizer.pad_token is None:
    global_tokenizer.pad_token = global_tokenizer.eos_token

LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 1, "undef": 1, "neutral": 1, "no": 2, "contradiction": 2}

def map_label(label):
    if isinstance(label, int) and label in [0, 1, 2]: return label
    s = str(label).lower().strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try:
        val = int(s)
        return val if val in [0, 1, 2] else 1
    except:
        return 1

def tokenize_fn(examples, p_key):
    # Pour un modèle causal (Decoder-Only), on lui donne la structure complete de gauche à droite
    prompts = [f"Prémisse : {p}\nHypothèse : {h}\n" for p, h in zip(examples[p_key], examples["hypothesis"])]
    res = global_tokenizer(prompts, truncation=True, padding="max_length", max_length=256)
    res["labels"] = [map_label(l) for l in examples["label"]]
    return res

train_data = TRAIN_DICT['train'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = TRAIN_DICT['validation'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_untokenized = concatenate_datasets(list(TEST_DICT.values()))
test_data = test_untokenized.map(lambda ex: tokenize_fn(ex, TEST_PKEY), batched=True, remove_columns=test_untokenized.column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception: pass
    return {"accuracy": accuracy_score(labels, predictions)}

# 4. FONCTION D'ENTRAÎNEMENT WANDB

def train_croissant_qlora():
    run = wandb.init()
    config = wandb.config

    run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
    print(f"\n{'='*60}\nSWEEP CROISSANT-LLM RUN: {run_label}\n{'='*60}")

    print("Chargement en QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Nous utilisons AutoModelForSequenceClassification sur un CausalLM. 
    # C'est la méthode de l'état de l'art pour extraire de l'accuracy pure d'un LLM sans générer de texte.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        device_map="auto",
        quantization_config=bnb_config
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = global_tokenizer.pad_token_id
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"], # Projections classiques ciblées pour les LLM
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {trainable:,} ({100 * trainable / total:.2f}%)")

    args = TrainingArguments(
        output_dir=f"/tmp/ckpt_{EXP_NAME}_{run_label}", 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    trainer.train()

    print("\nÉvaluation Cross-Dataset Finale...")
    test_results = trainer.evaluate(test_data)
    test_acc = test_results["eval_accuracy"]
    print(f"FINAL TEST ACCURACY : {test_acc:.2%}")

    wandb.log({"test/cross_dataset_accuracy": test_acc})
    wandb.summary["test_cross_dataset_accuracy"] = test_acc

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    total_runs = 1
    for param in SWEEP_CONFIG["parameters"].values():
        total_runs *= len(param["values"])
    
    if len(sys.argv) > 1:
        confirm = "o" 
    else:
        confirm = input(f"\nLancer {total_runs} sweeps CroissantLLM QLoRA ? (o/n): ").strip().lower()
        
    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_croissant_qlora, count=total_runs)
        print("\nTerminé ! Consultez les résultats sur WandB.")
    else:
        print("Annulé.")
