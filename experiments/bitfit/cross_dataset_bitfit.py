"""
Script de BitFit CROSS-DATASET pour CamemBERT et FlauBERT (Classification NLI).
Entraîne sur un dataset (train_ds) et évalue sur un autre (test_ds).
BitFit = On gèle TOUS les paramètres sauf les biais (bias).
"""
import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import torch
import torch.nn as nn
import wandb
import json
import shutil
import argparse
import numpy as np

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

def get_dataset(name):
    print(f"Téléchargement et structuration de {name}...")
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
        ds = DatasetDict({
            'train': fracas, 'validation': fracas, 'test': fracas
        })
        return ds, "premises"
        
    elif name == "daccord":
        data = load_dataset('maximoss/daccord-contradictions')['train'].shuffle(seed=42)
        # Convertir label 1 -> 2 (faux/contradiction) pour le format 3-class standard
        data = data.map(lambda ex: {"label": 2 if ex["label"] == 1 else 0})
        total = len(data)
        train_size, val_size = int(total * 0.6), int(total * 0.2)
        return DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        }), "premise"

    elif name == "sick":
        ds = load_dataset("maximoss/sick-fr")
        def convert(ex):
            lbl = str(ex["entailment_label"]).strip().upper()
            l_val = 0 if lbl == "ENTAILMENT" else (1 if lbl == "NEUTRAL" else 2)
            return {"premise": ex["sentence_A"], "hypothesis": ex["sentence_B"], "label": l_val}
        return DatasetDict({
            "train": ds["train"].map(convert),
            "validation": ds["validation"].map(convert),
            "test": ds["test"].map(convert)
        }), "premise"

    elif name == "rte3":
        ds = load_dataset("maximoss/rte3-french")
        # RTE3-french sur HF a "validation" et "test"
        return DatasetDict({
            "train": ds["validation"],
            "validation": ds["test"],
            "test": ds["test"]
        }), "premise"

    raise ValueError(f"Dataset {name} inconnu.")

# ============================================================
# 2. NORMALISATION DES LABELS
# ============================================================

LABEL_MAP = {
    "yes": 0, "entailment": 0,
    "unknown": 1, "undef": 1, "neutral": 1,
    "no": 2, "contradiction": 2
}

def map_label(label):
    if isinstance(label, int):
        return label if label in [0, 1, 2] else 1
    s = str(label).lower().strip()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        val = int(s)
        return val if val in [0, 1, 2] else 1
    except:
        return 1

# ============================================================
# 3. TOKENISATION
# ============================================================

global_tokenizer = None

def tokenize_fn(examples, p_key):
    res = global_tokenizer(
        examples[p_key], examples["hypothesis"],
        truncation=True, padding="max_length", max_length=128
    )
    res["labels"] = [map_label(l) for l in examples["label"]]
    return res

# ============================================================
# 4. MÉTRIQUES
# ============================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    labels_order = [0, 1, 2]
    cm = confusion_matrix(labels, predictions, labels=labels_order)
    print(f"\n📊 MATRICE DE CONFUSION:")
    print(f"             Préd:vrai   Préd:neutre   Préd:faux")
    print(f"Vrai vrai:   {cm[0][0]:<11} {cm[0][1]:<13} {cm[0][2]}")
    print(f"Vrai neutre: {cm[1][0]:<11} {cm[1][1]:<13} {cm[1][2]}")
    print(f"Vrai faux:   {cm[2][0]:<11} {cm[2][1]:<13} {cm[2][2]}\n")
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="macro"),
    }
    return metrics

# ============================================================
# 5. BITFIT : Geler tout sauf les biais
# ============================================================

def apply_bitfit(model):
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "bias" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    
    # La tête de classification doit TOUJOURS être entraînable
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            param.requires_grad = True
            if "bias" not in name:
                trainable_params += param.numel()
    
    pct = 100 * trainable_params / total_params
    print(f"🔧 BitFit activé : {trainable_params:,} params entraînables / {total_params:,} total ({pct:.4f}%)")
    return model

# ============================================================
# 6. CONFIGS À TESTER
# ============================================================

CONFIGS = {
    "A": {"lr": 3e-4, "epochs": 10, "desc": "BitFit lr=3e-4, 10 epochs"},
    "B": {"lr": 5e-4, "epochs": 10, "desc": "BitFit lr=5e-4, 10 epochs"},
    "C": {"lr": 1e-3, "epochs": 10, "desc": "BitFit lr=1e-3, 10 epochs"},
}

def run_config(config_name, config, model_id, train_data, val_data, tokenizer, train_ds_name, test_ds_name):
    print(f"\n{'='*70}")
    print(f"  🔬 CONFIG {config_name} : {config['desc']}")
    print(f"  🔄 CROSS DATASET : {train_ds_name} -> {test_ds_name}")
    print(f"{'='*70}")
    
    run = wandb.init(project="fewshot-nli-fr", reinit=True)
    run.name = f"bitfit_{model_id.split('/')[-1]}_{config_name}_{train_ds_name}2{test_ds_name}"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = apply_bitfit(model)

    args = TrainingArguments(
        output_dir=f"/tmp/bitfit_{config_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config["lr"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=False,  # FIX: Disabled to avoid HF loading bug
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    # Évaluation finale: extraction manuelle des vrais résultats pour contourner le bug
    best_acc = 0.0
    best_f1 = 0.0
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            best_acc = max(best_acc, log["eval_accuracy"])
        if "eval_f1_score" in log:
            best_f1 = max(best_f1, log["eval_f1_score"])
            
    print(f"\n✅ Config {config_name} → Vraie Best Accuracy: {best_acc:.2%} | Vrai Best F1: {best_f1:.4f}")
    
    result = {
        "config_name": config_name,
        "config": config,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
    }
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    del model, trainer
    torch.cuda.empty_cache()
    
    wandb.finish()
    return result

# ============================================================
# 7. MAIN
# ============================================================

def main():
    global global_tokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="camembert", help="camembert ou flaubert")
    parser.add_argument("--train_ds", type=str, default="rte3", help="Dataset d'entraînement")
    parser.add_argument("--test_ds", type=str, default="daccord", help="Dataset d'évaluation")
    parser.add_argument("--config", type=str, default="all", help="A, B, C, ou all")
    args = parser.parse_args()

    if args.model == "flaubert":
        model_id = "flaubert/flaubert_base_cased"
    elif args.model == "camembert_xnli":
        model_id = "BaptisteDoyen/camembert-base-xnli"
    else:
        model_id = "camembert-base"

    print(f"🚀 BitFit CROSS-DATASET sur {model_id}")

    global_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = global_tokenizer

    # Données
    train_ds_dict, train_pkey = get_dataset(args.train_ds)
    test_ds_dict, test_pkey = get_dataset(args.test_ds)
    
    train_data = train_ds_dict["train"].map(
        lambda ex: tokenize_fn(ex, train_pkey), batched=True,
        remove_columns=train_ds_dict["train"].column_names
    )
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    val_split = "validation" if "validation" in test_ds_dict else "test"
    val_data = test_ds_dict[val_split].map(
        lambda ex: tokenize_fn(ex, test_pkey), batched=True,
        remove_columns=test_ds_dict[val_split].column_names
    )
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(f"📊 Train: {len(train_data)} | Val/Test Cible: {len(val_data)}")

    # Configs à tester
    configs_to_run = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    
    all_results = {}
    for name, cfg in configs_to_run.items():
        result = run_config(name, cfg, model_id, train_data, val_data, tokenizer, args.train_ds, args.test_ds)
        all_results[name] = result
        print(f"\n📈 Config {name} terminée : Accuracy = {result['best_accuracy']:.2%}")

    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    res_path = f"results/bitfit_{args.model}_{args.train_ds}to{args.test_ds}.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ RÉSULTATS SAUVEGARDÉS DANS : {res_path}")

if __name__ == "__main__":
    main()
