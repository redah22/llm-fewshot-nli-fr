"""
Expériences Cross-Dataset NLI — Plan du tuteur
===============================================

Ce script implémente les 4 expériences définies par le tuteur :

  EXP 0 — Baseline : test du modèle vierge sur TOUS les datasets
  EXP 1 — Fine-tuning sur FraCaS (0-74)  → test sur tout GQNLI-FR
  EXP 2 — Fine-tuning sur GQNLI-FR       → test sur FraCaS (0-74)
  EXP 3 — Fine-tuning sur RTE3-DEV       → test sur DACCORD + RTE3-TEST

La deuxième étape consistera à reproduire ces mêmes expériences
avec LoRA-PEFT pour comparer les résultats.

Utilisation :
    python3 experiments/cross_dataset_eval.py

Prérequis (données) :
    python3 experiments/setup_data.py   # Choix 1 (GQNLI-FR)
    python3 experiments/setup_data.py   # Choix 3 (FraCaS 0-74)
    python3 experiments/setup_data.py   # Choix 4 (DACCORD)
    python3 experiments/setup_data.py   # Choix 5 (RTE3-French)
"""

import os
import json
import numpy as np
import torch

from datasets import DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────
# 1. CHOIX DU MODÈLE
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("EXPÉRIENCES CROSS-DATASET NLI")
print("Plan du tuteur — Fine-Tuning sans PEFT")
print("=" * 60)

print("\nQuel modèle de base utiliser ?")
print("1. CamemBERT (camembert-base)  — recommandé")
print("2. FlauBERT  (flaubert/flaubert_base_uncased)")

model_choice = input("\nVotre choix (1 ou 2, défaut=1): ").strip() or "1"

if model_choice == "2":
    BASE_MODEL = "flaubert/flaubert_base_uncased"
    MODEL_SHORT = "flaubert"
    print("📦 Modèle : FlauBERT")
else:
    BASE_MODEL = "camembert-base"
    MODEL_SHORT = "camembert"
    print("📦 Modèle : CamemBERT")

# ─────────────────────────────────────────────────────────
# 2. CHOIX DE L'EXPÉRIENCE
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Quelle expérience ?")
print("-" * 60)
print("0. Baseline — Modèle vierge testé sur TOUS les datasets")
print("1. Fine-tuning FraCaS (0-74)  →  test GQNLI-FR (complet)")
print("2. Fine-tuning GQNLI-FR       →  test FraCaS (0-74)")
print("3. Fine-tuning RTE3-DEV       →  test DACCORD + RTE3-TEST")
print("4. Toutes les expériences (1, 2 et 3) à la suite")

exp_choice = input("\nVotre choix (0, 1, 2, 3 ou 4): ").strip()

if exp_choice not in ["0", "1", "2", "3", "4"]:
    print("❌ Choix invalide!")
    exit(1)

# ─────────────────────────────────────────────────────────
# 3. FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────

# Mapping des labels texte → entier
LABEL_MAP = {
    "yes": 0, "entailment": 0,
    "unknown": 1, "undef": 1, "neutral": 1,
    "no": 2, "contradiction": 2,
}

def map_label(l):
    if isinstance(l, int):
        return l
    s = str(l).lower().strip()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        return int(s)
    except ValueError:
        return 1  # neutre par défaut


def make_tokenize_fn(tokenizer, premise_key):
    """Retourne une fonction de tokenisation pour le Trainer HuggingFace."""
    def tokenize(examples):
        result = tokenizer(
            examples[premise_key],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        result["labels"] = [map_label(l) for l in examples["label"]]
        return result
    return tokenize


def load_split(path, split, premise_key, tokenizer):
    """Charge et tokenise un split d'un dataset sauvegardé sur disque."""
    ds = DatasetDict.load_from_disk(path)
    data = ds[split]
    tokenized = data.map(
        make_tokenize_fn(tokenizer, premise_key),
        batched=True,
        remove_columns=data.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized, data


def load_all_splits_merged(path, premise_key, tokenizer):
    """Charge et concatène tous les splits d'un dataset (pour baseline)."""
    ds = DatasetDict.load_from_disk(path)
    merged = concatenate_datasets(list(ds.values()))
    tokenized = merged.map(
        make_tokenize_fn(tokenizer, premise_key),
        batched=True,
        remove_columns=merged.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized, merged


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # Matrice de confusion
    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        print("\n📊 MATRICE DE CONFUSION:")
        print("         Préd:0  Préd:1  Préd:2")
        print(f"  Vrai 0: {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]}")
        print(f"  Vrai 1: {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]}")
        print(f"  Vrai 2: {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]}")
    except Exception:
        pass
    return {"accuracy": accuracy_score(labels, predictions)}


class MetricsCollectorCallback(TrainerCallback):
    """Collecte loss/grad_norm/lr par step et eval_loss/accuracy par epoch."""
    def __init__(self):
        self.train_history = []
        self.eval_history  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self.train_history.append({
                "step":          state.global_step,
                "epoch":         round(logs.get("epoch", 0), 4),
                "loss":          logs.get("loss"),
                "grad_norm":     logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
            })
        if "eval_loss" in logs:
            self.eval_history.append({
                "epoch":         round(logs.get("epoch", 0), 4),
                "eval_loss":     logs.get("eval_loss"),
                "eval_accuracy": logs.get("eval_accuracy"),
            })


def make_trainer(model, tokenizer, train_ds, eval_ds, run_name, epochs=20):
    """Crée un Trainer avec early stopping (patience=3) et collecte de métriques."""
    metrics_cb = MetricsCollectorCallback()
    args = TrainingArguments(
        output_dir=f"checkpoints/{MODEL_SHORT}_{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), metrics_cb],
    )
    # Attacher le callback au trainer pour y accéder après l'entraînement
    trainer._metrics_cb = metrics_cb
    return trainer


def fresh_model(tokenizer_name):
    """Charge un modèle CamemBERT/FlauBERT vierge (poids originaux)."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, num_labels=3
    )
    return tok, mdl


def save_results(results: dict, filename: str, trainer=None):
    """Sauvegarde les résultats en JSON, avec l'historique d'entraînement si disponible."""
    # Ajouter l'historique du callback si le trainer est passé
    if trainer is not None and hasattr(trainer, "_metrics_cb"):
        cb = trainer._metrics_cb
        best_idx = np.argmax([e["eval_accuracy"] for e in cb.eval_history]) if cb.eval_history else 0
        results["training_history"] = {
            "train_steps":    cb.train_history,
            "eval_epochs":    cb.eval_history,
            "epochs_trained": len(cb.eval_history),
            "best_epoch":     cb.eval_history[best_idx]["epoch"] if cb.eval_history else None,
            "best_val_acc":   cb.eval_history[best_idx]["eval_accuracy"] if cb.eval_history else None,
        }
    os.makedirs("results/metrics", exist_ok=True)
    path = f"results/metrics/{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\u2705 Résultats sauvegardés : {path}")
    if trainer is not None and hasattr(trainer, "_metrics_cb") and trainer._metrics_cb.eval_history:
        print(f"   → Pour les graphiques : python3 experiments/plot_training.py {path}")


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# DATASETS — chemins et clés de prémisse
# ─────────────────────────────────────────────────────────
DATASETS = {
    "gqnli_fr": {
        "path": "data/processed/gqnli_fr",
        "pkey": "premise",
        "label": "GQNLI-FR",
    },
    "fracas_75": {
        "path": "data/processed/fracas_subset_75",
        "pkey": "premises",
        "label": "FraCaS (0-74)",
    },
    "daccord": {
        "path": "data/processed/daccord",
        "pkey": "premise",
        "label": "DACCORD",
    },
    "rte3_fr": {
        "path": "data/processed/rte3_fr",
        "pkey": "premise",
        "label": "RTE3-French",
    },
}


def check_dataset(key):
    """Vérifie qu'un dataset existe sur disque et affiche une aide si non."""
    info = DATASETS[key]
    disk_path = info["path"]
    if not os.path.isdir(disk_path):
        print(f"❌ Dataset '{info['label']}' introuvable : {disk_path}")
        print(f"   → Lancez : python3 experiments/setup_data.py")
        return False
    return True


# ─────────────────────────────────────────────────────────
# EXP 0 — BASELINE : modèle vierge sur tous les datasets
# ─────────────────────────────────────────────────────────

def run_exp0():
    print_header("EXP 0 — BASELINE (modèle vierge sur tous les datasets)")

    tokenizer, model = fresh_model(BASE_MODEL)
    print(f"✅ {MODEL_SHORT} chargé (poids d'origine, sans fine-tuning)")

    all_results = {}

    for ds_key, ds_info in DATASETS.items():
        if not check_dataset(ds_key):
            continue

        print(f"\n🔹 Évaluation sur : {ds_info['label']}")
        try:
            tokenized_all, raw = load_all_splits_merged(
                ds_info["path"], ds_info["pkey"], tokenizer
            )
        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")
            continue

        print(f"   Exemples : {len(raw)}")
        trainer = make_trainer(model, tokenizer, None, tokenized_all, f"baseline_{ds_key}", epochs=1)
        metrics = trainer.evaluate()
        acc = metrics["eval_accuracy"]
        print(f"   → Précision baseline : {acc:.2%}")
        all_results[ds_key] = {"dataset": ds_info["label"], "accuracy": acc}

    print_header("RÉSUMÉ BASELINE")
    for k, v in all_results.items():
        print(f"  {v['dataset']:<25} : {v['accuracy']:.2%}")

    save_results(
        {"model": BASE_MODEL, "experiment": "baseline", "results": all_results},
        f"{MODEL_SHORT}_exp0_baseline",
    )


# ─────────────────────────────────────────────────────────
# EXP 1 — Fine-tuning FraCaS (0-74) → test GQNLI-FR complet
# ─────────────────────────────────────────────────────────

def run_exp1():
    print_header("EXP 1 — Fine-tuning FraCaS (0-74)  →  test GQNLI-FR")

    if not check_dataset("fracas_75") or not check_dataset("gqnli_fr"):
        return

    tokenizer, model = fresh_model(BASE_MODEL)

    # --- Train : FraCaS (0-74) split train ---
    print("\n📂 Chargement TRAIN : FraCaS (0-74)")
    train_tok, train_raw = load_split(
        DATASETS["fracas_75"]["path"], "train",
        DATASETS["fracas_75"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    # --- Eval (pendant le FT) : FraCaS validation ---
    val_tok, val_raw = load_split(
        DATASETS["fracas_75"]["path"], "validation",
        DATASETS["fracas_75"]["pkey"], tokenizer
    )

    trainer = make_trainer(model, tokenizer, train_tok, val_tok, "exp1_fracas75_to_gqnli")

    # Baseline avant FT
    print("\n🔹 Baseline (avant fine-tuning) sur FraCaS val :")
    base_metrics = trainer.evaluate()
    baseline_acc = base_metrics["eval_accuracy"]
    print(f"   → {baseline_acc:.2%}")

    # Fine-tuning
    print("\n🔹 Fine-tuning sur FraCaS (0-74) train...")
    trainer.train()
    print("✅ Fine-tuning terminé")

    # Sauvegarde du modèle
    model_path = f"models/{MODEL_SHORT}_exp1_fracas75"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

    # --- Test : TOUS les splits de GQNLI-FR ---
    print("\n🔹 Test sur GQNLI-FR (tous splits) :")
    gqnli_tok, gqnli_raw = load_all_splits_merged(
        DATASETS["gqnli_fr"]["path"], DATASETS["gqnli_fr"]["pkey"], tokenizer
    )
    print(f"   {len(gqnli_raw)} exemples")

    trainer.eval_dataset = gqnli_tok
    test_metrics = trainer.evaluate()
    test_acc = test_metrics["eval_accuracy"]
    print(f"   → Précision sur GQNLI-FR : {test_acc:.2%}")

    save_results(
        {
            "model": BASE_MODEL,
            "experiment": "exp1",
            "train_on": "FraCaS (0-74)",
            "test_on": "GQNLI-FR (complet)",
            "train_size": len(train_raw),
            "test_size": len(gqnli_raw),
            "baseline_accuracy": baseline_acc,
            "final_accuracy": test_acc,
            "gain": test_acc - baseline_acc,
        },
        f"{MODEL_SHORT}_exp1_fracas75_to_gqnli",
        trainer=trainer,
    )


# ─────────────────────────────────────────────────────────
# EXP 2 — Fine-tuning GQNLI-FR → test FraCaS (0-74)
# ─────────────────────────────────────────────────────────

def run_exp2():
    print_header("EXP 2 — Fine-tuning GQNLI-FR  →  test FraCaS (0-74)")

    if not check_dataset("gqnli_fr") or not check_dataset("fracas_75"):
        return

    tokenizer, model = fresh_model(BASE_MODEL)

    # --- Train : GQNLI-FR split train ---
    print("\n📂 Chargement TRAIN : GQNLI-FR")
    train_tok, train_raw = load_split(
        DATASETS["gqnli_fr"]["path"], "train",
        DATASETS["gqnli_fr"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    # --- Eval (pendant FT) : GQNLI validation ---
    val_tok, val_raw = load_split(
        DATASETS["gqnli_fr"]["path"], "validation",
        DATASETS["gqnli_fr"]["pkey"], tokenizer
    )

    trainer = make_trainer(model, tokenizer, train_tok, val_tok, "exp2_gqnli_to_fracas75")

    # Baseline
    print("\n🔹 Baseline (avant fine-tuning) sur GQNLI val :")
    base_metrics = trainer.evaluate()
    baseline_acc = base_metrics["eval_accuracy"]
    print(f"   → {baseline_acc:.2%}")

    # Fine-tuning
    print("\n🔹 Fine-tuning sur GQNLI-FR train...")
    trainer.train()
    print("✅ Fine-tuning terminé")

    # Sauvegarde
    model_path = f"models/{MODEL_SHORT}_exp2_gqnli"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

    # --- Test : tous les splits de FraCaS (0-74) ---
    print("\n🔹 Test sur FraCaS (0-74) (tous splits) :")
    fracas_tok, fracas_raw = load_all_splits_merged(
        DATASETS["fracas_75"]["path"], DATASETS["fracas_75"]["pkey"], tokenizer
    )
    print(f"   {len(fracas_raw)} exemples")

    trainer.eval_dataset = fracas_tok
    test_metrics = trainer.evaluate()
    test_acc = test_metrics["eval_accuracy"]
    print(f"   → Précision sur FraCaS (0-74) : {test_acc:.2%}")

    save_results(
        {
            "model": BASE_MODEL,
            "experiment": "exp2",
            "train_on": "GQNLI-FR",
            "test_on": "FraCaS (0-74) complet",
            "train_size": len(train_raw),
            "test_size": len(fracas_raw),
            "baseline_accuracy": baseline_acc,
            "final_accuracy": test_acc,
            "gain": test_acc - baseline_acc,
        },
        f"{MODEL_SHORT}_exp2_gqnli_to_fracas75",
        trainer=trainer,
    )


# ─────────────────────────────────────────────────────────
# EXP 3 — Fine-tuning RTE3-DEV → test DACCORD + RTE3-TEST
# ─────────────────────────────────────────────────────────

def run_exp3():
    print_header("EXP 3 — Fine-tuning RTE3-DEV  →  test DACCORD + RTE3-TEST")

    if not check_dataset("rte3_fr") or not check_dataset("daccord"):
        return

    tokenizer, model = fresh_model(BASE_MODEL)

    # --- Train : RTE3 train (≈ le "dev" du tuteur = notre split train) ---
    print("\n📂 Chargement TRAIN : RTE3-French (split train)")
    train_tok, train_raw = load_split(
        DATASETS["rte3_fr"]["path"], "train",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    # Eval pendant le FT : RTE3 validation
    val_tok, val_raw = load_split(
        DATASETS["rte3_fr"]["path"], "validation",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )

    trainer = make_trainer(model, tokenizer, train_tok, val_tok, "exp3_rte3_to_daccord")

    # Baseline
    print("\n🔹 Baseline (avant fine-tuning) sur RTE3 val :")
    base_metrics = trainer.evaluate()
    baseline_acc = base_metrics["eval_accuracy"]
    print(f"   → {baseline_acc:.2%}")

    # Fine-tuning
    print("\n🔹 Fine-tuning sur RTE3-DEV (train)...")
    trainer.train()
    print("✅ Fine-tuning terminé")

    # Sauvegarde
    model_path = f"models/{MODEL_SHORT}_exp3_rte3"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

    results_exp3 = {
        "model": BASE_MODEL,
        "experiment": "exp3",
        "train_on": "RTE3-French (train)",
        "baseline_accuracy": baseline_acc,
    }

    # --- Test 1 : DACCORD (tous splits) ---
    print("\n🔹 Test sur DACCORD :")
    dacc_tok, dacc_raw = load_all_splits_merged(
        DATASETS["daccord"]["path"], DATASETS["daccord"]["pkey"], tokenizer
    )
    print(f"   {len(dacc_raw)} exemples")
    trainer.eval_dataset = dacc_tok
    dacc_metrics = trainer.evaluate()
    dacc_acc = dacc_metrics["eval_accuracy"]
    print(f"   → Précision sur DACCORD : {dacc_acc:.2%}")
    results_exp3["daccord"] = {"test_size": len(dacc_raw), "accuracy": dacc_acc}

    # --- Test 2 : RTE3-TEST ---
    print("\n🔹 Test sur RTE3-TEST (split test) :")
    rte3_test_tok, rte3_test_raw = load_split(
        DATASETS["rte3_fr"]["path"], "test",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )
    print(f"   {len(rte3_test_raw)} exemples")
    trainer.eval_dataset = rte3_test_tok
    rte3_metrics = trainer.evaluate()
    rte3_test_acc = rte3_metrics["eval_accuracy"]
    print(f"   → Précision sur RTE3-TEST : {rte3_test_acc:.2%}")
    results_exp3["rte3_test"] = {"test_size": len(rte3_test_raw), "accuracy": rte3_test_acc}

    print_header("RÉSUMÉ EXP 3")
    print(f"  Baseline RTE3 val : {baseline_acc:.2%}")
    print(f"  DACCORD           : {dacc_acc:.2%}")
    print(f"  RTE3-TEST         : {rte3_test_acc:.2%}")

    save_results(results_exp3, f"{MODEL_SHORT}_exp3_rte3_to_daccord", trainer=trainer)


# ─────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────

if exp_choice == "0":
    run_exp0()

elif exp_choice == "1":
    run_exp1()

elif exp_choice == "2":
    run_exp2()

elif exp_choice == "3":
    run_exp3()

elif exp_choice == "4":
    print("\n🚀 Lancement de toutes les expériences (1, 2, 3)...")
    run_exp1()
    run_exp2()
    run_exp3()

print("\n" + "=" * 60)
print("✅ EXPÉRIENCE(S) TERMINÉE(S)")
print(f"   Résultats dans : results/{MODEL_SHORT}_exp*.json")
print("=" * 60)
