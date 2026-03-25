"""
WandB Sweep — Recherche automatique des meilleurs hyperparamètres LoRA
=====================================================================

Ce script lance automatiquement plusieurs entraînements LoRA avec
différentes combinaisons d'hyperparamètres (rang r, alpha, learning_rate,
dropout, epochs) et enregistre tout sur WandB pour comparaison visuelle.

Utilisation :
    python3 experiments/lora/sweep_lora.py

Résultats visibles en temps réel sur : https://wandb.ai/<votre-user>/fewshot-nli-fr
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import json
import numpy as np
import torch
import wandb

from datasets import DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION DU SWEEP
# ─────────────────────────────────────────────────────────

# Définition de l'espace de recherche des hyperparamètres
SWEEP_CONFIG = {
    "method": "grid",  # grid = teste TOUTES les combinaisons (exhaustif)
    "metric": {
        "name": "eval/accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "lora_r": {
            "values": [4, 8, 16]           # Rang des matrices LoRA
        },
        "lora_alpha": {
            "values": [8, 16, 32]          # Facteur d'échelle
        },
        "learning_rate": {
            "values": [5e-4]  # Taux d'apprentissage
        },
        "lora_dropout": {
            "values": [0.1]          # Dropout LoRA
        },
    }
}
# Total combinaisons grid: 3 × 3 × 3 × 2 = 54 runs
# ⚠️ Sur CPU c'est long ! Réduisez les "values" si besoin.

# ─────────────────────────────────────────────────────────
# 2. CHOIX DE L'EXPÉRIENCE
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("WANDB SWEEP — RECHERCHE HYPERPARAMÈTRES LORA")
print("=" * 60)

BASE_MODEL = "camembert-base"
MODEL_SHORT = "camembert"

DATASETS = {
    "gqnli_fr": {"path": "data/processed/gqnli_fr", "pkey": "premise", "label": "GQNLI-FR"},
    "fracas_75": {"path": "data/processed/fracas_subset_75", "pkey": "premises", "label": "FraCaS (0-74)"},
    "daccord": {"path": "data/processed/daccord", "pkey": "premise", "label": "DACCORD"},
    "rte3_fr": {"path": "data/processed/rte3_fr", "pkey": "premise", "label": "RTE3-French"},
}

print("\nQuelle expérience utiliser pour le sweep ?")
print("1. FraCaS (0-74)  →  test GQNLI-FR")
print("2. GQNLI-FR       →  test FraCaS (0-74)")
print("3. RTE3-DEV       →  test DACCORD + RTE3-TEST")

exp_choice = input("\nVotre choix (1, 2 ou 3): ").strip()
if exp_choice not in ["1", "2", "3"]:
    print("❌ Choix invalide!")
    exit(1)

# Déterminer les datasets train/val/test selon le choix
if exp_choice == "1":
    TRAIN_DS_KEY, TRAIN_SPLIT = "fracas_75", "train"
    VAL_DS_KEY, VAL_SPLIT = "fracas_75", "validation"
    TEST_DS_KEY, TEST_MERGE = "gqnli_fr", True
    EXP_NAME = "sweep_fracas_to_gqnli"
elif exp_choice == "2":
    TRAIN_DS_KEY, TRAIN_SPLIT = "gqnli_fr", "train"
    VAL_DS_KEY, VAL_SPLIT = "gqnli_fr", "validation"
    TEST_DS_KEY, TEST_MERGE = "fracas_75", True
    EXP_NAME = "sweep_gqnli_to_fracas"
else:
    TRAIN_DS_KEY, TRAIN_SPLIT = "rte3_fr", "train"
    VAL_DS_KEY, VAL_SPLIT = "rte3_fr", "validation"
    TEST_DS_KEY, TEST_MERGE = "daccord", True
    EXP_NAME = "sweep_rte3_to_daccord"

# ─────────────────────────────────────────────────────────
# 3. FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────

LABEL_MAP = {
    "yes": 0, "entailment": 0,
    "unknown": 1, "undef": 1, "neutral": 1,
    "no": 2, "contradiction": 2
}

def map_label(label):
    if isinstance(label, int) and label in [0, 1, 2]:
        return label
    return LABEL_MAP.get(str(label).lower().strip(), 1)

def make_tokenize_fn(tokenizer, premise_key):
    def tokenize(examples):
        result = tokenizer(
            examples[premise_key], examples["hypothesis"],
            truncation=True, padding="max_length", max_length=128,
        )
        result["labels"] = [map_label(l) for l in examples["label"]]
        return result
    return tokenize

def load_split(path, split, premise_key, tokenizer):
    ds = DatasetDict.load_from_disk(path)
    data = ds[split]
    tokenized = data.map(make_tokenize_fn(tokenizer, premise_key), batched=True, remove_columns=data.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def load_all_merged(path, premise_key, tokenizer):
    ds = DatasetDict.load_from_disk(path)
    merged = concatenate_datasets(list(ds.values()))
    tokenized = merged.map(make_tokenize_fn(tokenizer, premise_key), batched=True, remove_columns=merged.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ─────────────────────────────────────────────────────────
# 4. FONCTION D'ENTRAÎNEMENT (Appelée par le Sweep Agent)
# ─────────────────────────────────────────────────────────

def train_one_run():
    """
    Fonction appelée automatiquement par wandb.agent() pour chaque
    combinaison d'hyperparamètres. Les paramètres sont injectés via
    wandb.config.
    """
    # Initialiser le run WandB  
    run = wandb.init()
    config = wandb.config

    # Récupérer les hyperparamètres choisis par le sweep
    lora_r = config.lora_r
    lora_alpha = config.lora_alpha
    learning_rate = config.learning_rate
    lora_dropout = config.lora_dropout

    run_label = f"r{lora_r}_a{lora_alpha}_lr{learning_rate}_d{lora_dropout}"
    print(f"\n{'='*60}")
    print(f"🔄 SWEEP RUN: r={lora_r}, α={lora_alpha}, lr={learning_rate}, dropout={lora_dropout}")
    print(f"{'='*60}")

    # Charger le modèle et appliquer LoRA avec les paramètres du sweep
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📐 Paramètres entraînables : {trainable:,} ({100 * trainable / total:.2f}%)")

    # Charger les données
    train_info = DATASETS[TRAIN_DS_KEY]
    val_info = DATASETS[VAL_DS_KEY]
    test_info = DATASETS[TEST_DS_KEY]

    train_ds = load_split(train_info["path"], TRAIN_SPLIT, train_info["pkey"], tokenizer)
    val_ds = load_split(val_info["path"], VAL_SPLIT, val_info["pkey"], tokenizer)

    if TEST_MERGE:
        test_ds = load_all_merged(test_info["path"], test_info["pkey"], tokenizer)
    else:
        test_ds = load_split(test_info["path"], "test", test_info["pkey"], tokenizer)

    # Configurer le Trainer
    args = TrainingArguments(
        output_dir=f"checkpoints/sweep_{run_label}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Entraîner
    trainer.train()

    # Test final cross-dataset
    test_results = trainer.evaluate(test_ds)
    test_acc = test_results["eval_accuracy"]
    print(f"\n🎯 Test cross-dataset accuracy : {test_acc:.2%}")

    # Logger la métrique finale dans WandB (apparaîtra dans le tableau comparatif)
    wandb.log({"test/cross_dataset_accuracy": test_acc})
    wandb.summary["test_cross_dataset_accuracy"] = test_acc

    # Nettoyer les checkpoints pour économiser l'espace disque
    import shutil
    checkpoint_dir = f"checkpoints/sweep_{run_label}"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    wandb.finish()

# ─────────────────────────────────────────────────────────
# 5. LANCEMENT DU SWEEP
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🚀 Lancement du Sweep WandB pour l'expérience : {EXP_NAME}")
    print(f"   Méthode : {SWEEP_CONFIG['method']}")
    
    total_runs = 1
    for param in SWEEP_CONFIG["parameters"].values():
        total_runs *= len(param["values"])
    print(f"   Nombre total de runs : {total_runs}")
    
    confirm = input(f"\n⚠️  Lancer {total_runs} entraînements automatiques ? (o/n): ").strip().lower()
    if confirm != "o":
        print("❌ Annulé.")
        exit(0)

    # Créer le sweep sur WandB  
    sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")

    # Lancer l'agent qui va exécuter chaque run automatiquement
    print(f"\n✅ Sweep créé ! ID: {sweep_id}")
    print(f"📊 Suivez en direct : https://wandb.ai/colin-dievart-facult-des-sciences-montpellier/fewshot-nli-fr/sweeps/{sweep_id}")
    
    wandb.agent(sweep_id, function=train_one_run, count=total_runs)

    print("\n✅ Sweep terminé ! Consultez les résultats sur WandB.")
