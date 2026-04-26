"""
Script BitFit pour CamemBERT et Flaubert (Classification NLI).
BitFit = On gèle TOUS les paramètres sauf les biais (bias).
C'est la méthode PEFT la plus légère possible.

Même pipeline de données que Colin (LoRA) et Tala (Prefix-Tuning)
pour une comparaison équitable.

Usage :
  python experiments/bitfit/sweep_bitfit.py --model camembert --dataset gqnli_fr
  python experiments/bitfit/sweep_bitfit.py --model flaubert --dataset gqnli_fr
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

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ============================================================
# 1. CHARGEMENT DES DONNÉES (identique à Colin / sweep_lora.py)
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
        fracas = load_dataset('maximoss/fracas')['train']
        ds = DatasetDict({
            'train': fracas.select(range(75)), 
            'validation': fracas.select(range(75, 100)), 
            'test': fracas.select(range(100, 150))
        })
        return ds, "premises"

    raise ValueError(f"Dataset {name} inconnu.")

# ============================================================
# 2. NORMALISATION DES LABELS (identique à Colin)
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
    label_names = ["vrai", "neutre", "faux"]
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
    """
    BitFit : Gèle TOUS les paramètres du modèle SAUF ceux
    qui contiennent 'bias' dans leur nom.
    C'est la technique PEFT la plus simple et la plus légère.
    """
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
    "A": {"lr": 3e-4, "epochs": 20, "desc": "BitFit lr=3e-4, 20 epochs"},
    "B": {"lr": 5e-4, "epochs": 20, "desc": "BitFit lr=5e-4, 20 epochs"},
    "C": {"lr": 1e-3, "epochs": 20, "desc": "BitFit lr=1e-3, 20 epochs"},
}

def run_config(config_name, config, model_id, train_data, val_data, tokenizer, dataset_name):
    print(f"\n{'='*70}")
    print(f"  🔬 CONFIG {config_name} : {config['desc']}")
    print(f"{'='*70}")
    
    run = wandb.init(project="fewshot-nli-fr", reinit=True)
    run.name = f"bitfit_{model_id.split('/')[-1]}_{config_name}_{dataset_name}"
    
    # Charger le modèle frais à chaque config
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Appliquer BitFit
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
        load_best_model_at_end=True,
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
    
    # Évaluation finale
    eval_results = trainer.evaluate()
    best_acc = eval_results.get("eval_accuracy", 0.0)
    best_f1 = eval_results.get("eval_f1_score", 0.0)
    
    print(f"\n✅ Config {config_name} → Accuracy: {best_acc:.2%} | F1: {best_f1:.4f}")
    
    result = {
        "config_name": config_name,
        "config": config,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
    }
    
    # Nettoyage
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
    parser.add_argument("--dataset", type=str, default="gqnli_fr", help="gqnli_fr, fracas_75")
    parser.add_argument("--config", type=str, default="all", help="A, B, C, ou all")
    args = parser.parse_args()

    # Modèles
    if args.model == "flaubert":
        model_id = "flaubert/flaubert_base_cased"
    else:
        model_id = "camembert-base"

    print(f"🚀 BitFit sur {model_id}")

    # Tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = global_tokenizer

    # Données
    ds_dict, p_key = get_dataset(args.dataset)
    
    train_data = ds_dict["train"].map(
        lambda ex: tokenize_fn(ex, p_key), batched=True,
        remove_columns=ds_dict["train"].column_names
    )
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    val_data = ds_dict["validation"].map(
        lambda ex: tokenize_fn(ex, p_key), batched=True,
        remove_columns=ds_dict["validation"].column_names
    )
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    print(f"📊 Train: {len(train_data)} | Val: {len(val_data)}")

    # Configs à tester
    configs_to_run = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    
    all_results = {}
    for name, cfg in configs_to_run.items():
        result = run_config(name, cfg, model_id, train_data, val_data, tokenizer, args.dataset)
        all_results[name] = result
        print(f"\n📈 Config {name} terminée : Accuracy = {result['best_accuracy']:.2%}")

    # Sauvegarde
    os.makedirs("results", exist_ok=True)
    model_short = args.model
    res_path = f"results/bitfit_{model_short}_{args.dataset}.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ RÉSULTATS SAUVEGARDÉS DANS : {res_path}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("  📊 RÉSUMÉ COMPARATIF BITFIT")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"  Config {name} ({res['config']['desc']})")
        print(f"    → Accuracy: {res['best_accuracy']:.2%}")
        print(f"    → F1 Score: {res['best_f1']:.4f}")
        print()


if __name__ == "__main__":
    main()
