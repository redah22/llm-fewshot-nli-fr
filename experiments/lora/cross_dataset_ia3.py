"""
Expériences Cross-Dataset NLI — avec IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
==========================================================================================================

Reproduction exacte des expériences de cross_dataset_lora.py,
mais avec IA³ à la place de LoRA.

IA³ DIFFÈRE DE LORA :
  - LoRA ajoute des matrices A×B dans les couches d'attention.
  - IA³ ajoute uniquement des VECTEURS de mise à l'échelle (l_k, l_v, l_ff).
    Ces vecteurs amplifient ou inhibent les activations internes du modèle.
  - Résultat : 100-1000× moins de paramètres entraînables que LoRA,
    mais souvent des scores comparables ou meilleurs en Few-Shot.

  EXP 0 — Baseline : test du modèle vierge sur TOUS les datasets
  EXP 1 — Fine-tuning IA³ sur FraCaS (0-74)  → test sur tout GQNLI-FR
  EXP 2 — Fine-tuning IA³ sur GQNLI-FR       → test sur FraCaS (0-74)
  EXP 3 — Fine-tuning IA³ sur RTE3-DEV       → test sur DACCORD + RTE3-TEST

Utilisation :
    python3 experiments/lora/cross_dataset_ia3.py

Prérequis :
    pip install peft   (déjà installé)
    Datasets préparés via setup_data.py (choix 1, 3, 4, 5)
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"  # Même projet que LoRA pour comparer côte à côte

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
from peft import (
    get_peft_model,
    IA3Config,
    TaskType,
)
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────
# 1. CHOIX DU MODÈLE
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("EXPÉRIENCES CROSS-DATASET NLI — IA³ PEFT")
print("=" * 60)

print("\nQuel modèle de base utiliser ?")
print("1. CamemBERT (camembert-base)  — recommandé")
print("2. FlauBERT  (flaubert/flaubert_base_cased)")
print("3. FlauBERT + XNLI Full (models/flaubert_xnli_fr_full)")

model_choice = input("\nVotre choix (1, 2 ou 3, défaut=1): ").strip() or "1"

if model_choice == "2":
    BASE_MODEL = "flaubert/flaubert_base_cased"
    MODEL_SHORT = "flaubert"
    # IA³ cible les Keys (k), Values (v) dans l'attention, et la couche FF intermédiaire
    # Pour FlauBERT (architecture XLM), les modules sont :
    IA3_FF_MODULES     = ["lin1"]   # couche Feed-Forward intermédiaire
    IA3_TARGET_MODULES = ["k_lin", "v_lin", "out_lin"] + IA3_FF_MODULES
    print("📦 Modèle : FlauBERT (cased)")
elif model_choice == "3":
    if not os.path.isdir("models/flaubert_xnli_fr_full"):
        print("❌ Erreur : Le modèle local models/flaubert_xnli_fr_full n'existe pas. Utilisez 1 ou 2.")
        exit(1)
    BASE_MODEL = "models/flaubert_xnli_fr_full"
    MODEL_SHORT = "flaubert_xnli"
    IA3_FF_MODULES     = ["lin1"]
    IA3_TARGET_MODULES = ["k_lin", "v_lin", "out_lin"] + IA3_FF_MODULES
    print("📦 Modèle : FlauBERT + XNLI Full")
else:
    BASE_MODEL = "camembert-base"
    MODEL_SHORT = "camembert"
    # Pour CamemBERT (architecture RoBERTa), les modules sont :
    IA3_FF_MODULES     = ["intermediate.dense"]
    IA3_TARGET_MODULES = ["key", "value", "query", "output.dense"] + IA3_FF_MODULES
    print("📦 Modèle : CamemBERT")

# ─────────────────────────────────────────────────────────
# 2. CHOIX DE L'EXPÉRIENCE
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Quelle expérience ?")
print("-" * 60)
print("0. Baseline — Modèle vierge testé sur TOUS les datasets")
print("1. IA³ FraCaS (0-74)  →  test GQNLI-FR (complet)")
print("2. IA³ GQNLI-FR       →  test FraCaS (0-74)")
print("3. IA³ RTE3-DEV       →  test DACCORD + RTE3-TEST")
print("4. Toutes les expériences (1, 2 et 3) à la suite")

exp_choice = input("\nVotre choix (0, 1, 2, 3 ou 4): ").strip()

if exp_choice not in ["0", "1", "2", "3", "4"]:
    print("❌ Choix invalide!")
    exit(1)

# ─────────────────────────────────────────────────────────
# 3. FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────

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
        return 1


def make_tokenize_fn(tokenizer, premise_key):
    needs_token_type_ids = "flaubert" in BASE_MODEL.lower()
    def tokenize(examples):
        kwargs = dict(
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        if needs_token_type_ids:
            kwargs["return_token_type_ids"] = True
        result = tokenizer(examples[premise_key], examples["hypothesis"], **kwargs)
        result["labels"] = [map_label(l) for l in examples["label"]]
        return result
    return tokenize


def load_split(path, split, premise_key, tokenizer):
    ds = DatasetDict.load_from_disk(path)
    data = ds[split]
    tokenized = data.map(
        make_tokenize_fn(tokenizer, premise_key),
        batched=True,
        remove_columns=data.column_names,
    )
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        cols.append("token_type_ids")
    tokenized.set_format(type="torch", columns=cols)
    return tokenized, data


def load_all_splits_merged(path, premise_key, tokenizer):
    ds = DatasetDict.load_from_disk(path)
    merged = concatenate_datasets(list(ds.values()))
    tokenized = merged.map(
        make_tokenize_fn(tokenizer, premise_key),
        batched=True,
        remove_columns=merged.column_names,
    )
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        cols.append("token_type_ids")
    tokenized.set_format(type="torch", columns=cols)
    return tokenized, merged


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
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


def make_ia3_model(base_model_name):
    """
    Charge le modèle de base et applique IA³.

    IA³ insère des vecteurs (l_k, l_v, l_ff) qui MULTIPLIENT (amplifient
    ou inhibent) les activations des couches clés et valeurs de l'attention,
    et de la couche Feed-Forward intermédiaire.

    Contrairement à LoRA qui ADDITIONNE des matrices basses-dimensions,
    IA³ MULTIPLIE par des scalaires appris — beaucoup plus léger.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
    )

    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=IA3_TARGET_MODULES,         # Couches q/k/v de l'attention
        feedforward_modules=IA3_FF_MODULES,        # Couche FF intermédiaire
        modules_to_save=["classifier"],            # Entraîner aussi la tête de classification
    )

    model = get_peft_model(base_model, ia3_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n📐 Paramètres IA³ :")
    print(f"   Entraînables : {trainable:,} ({100 * trainable / total:.4f}% du total)")
    print(f"   Gelés        : {total - trainable:,}")
    print(f"   Total        : {total:,}")
    print(f"   ℹ️  IA³ entraîne {total // trainable}× moins de params que le full fine-tuning")

    return tokenizer, model


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


def make_trainer(model, tokenizer, train_ds, eval_ds, run_name, epochs=40):
    """Crée un Trainer IA³ avec early stopping (patience=10) et collecte de métriques."""
    metrics_cb = MetricsCollectorCallback()
    args = TrainingArguments(
        output_dir=f"checkpoints/{MODEL_SHORT}_ia3_{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # IA³ : learning rate plus élevé que LoRA car les vecteurs sont initialisés à 1
        # et doivent s'ajuster rapidement. 3e-3 est recommandé dans le papier original.
        learning_rate=3e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",  # Envoie les courbes en direct à WandB (même dashboard que LoRA)
        run_name=f"{MODEL_SHORT}-ia3-{run_name}",  # Préfixe "ia3" pour distinguer de "lora" dans WandB
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10), metrics_cb],
    )
    trainer._metrics_cb = metrics_cb
    return trainer


def save_results(results: dict, filename: str, trainer=None):
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
    print(f"✅ Résultats sauvegardés : {path}")
    if trainer is not None and hasattr(trainer, "_metrics_cb") and trainer._metrics_cb.eval_history:
        print(f"   → Pour les graphiques d'entraînement : python3 experiments/utils/plot_training.py {path}")
        print(f"   → Pour la comparaison globale       : python3 experiments/utils/plot_comparison.py")


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# DATASETS
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
    info = DATASETS[key]
    if not os.path.isdir(info["path"]):
        print(f"❌ Dataset '{info['label']}' introuvable : {info['path']}")
        print(f"   → Lancez : python3 experiments/data_utils/setup_data.py")
        return False
    return True


# ─────────────────────────────────────────────────────────
# EXP 0 — BASELINE
# ─────────────────────────────────────────────────────────

def run_exp0():
    print_header("EXP 0 — BASELINE IA³ (modèle vierge sur tous les datasets)")

    tokenizer, model = make_ia3_model(BASE_MODEL)
    print(f"\n✅ {MODEL_SHORT} + IA³ chargé (avant entraînement)")

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
        trainer = make_trainer(model, tokenizer, None, tokenized_all,
                               f"baseline_{ds_key}", epochs=1)
        metrics = trainer.evaluate()
        acc = metrics["eval_accuracy"]
        print(f"   → Précision baseline : {acc:.2%}")
        all_results[ds_key] = {"dataset": ds_info["label"], "accuracy": acc}

    print_header("RÉSUMÉ BASELINE IA³")
    for k, v in all_results.items():
        print(f"  {v['dataset']:<25} : {v['accuracy']:.2%}")

    save_results(
        {
            "model": BASE_MODEL,
            "peft_method": "IA3",
            "experiment": "baseline_ia3",
            "results": all_results,
        },
        f"{MODEL_SHORT}_ia3_exp0_baseline",
    )


# ─────────────────────────────────────────────────────────
# EXP 1 — IA³ FraCaS (0-74) → test GQNLI-FR complet
# ─────────────────────────────────────────────────────────

def run_exp1():
    print_header("EXP 1 IA³ — Fine-tuning FraCaS (0-74)  →  test GQNLI-FR")

    if not check_dataset("fracas_75") or not check_dataset("gqnli_fr"):
        return

    tokenizer, model = make_ia3_model(BASE_MODEL)

    print("\n📂 Chargement TRAIN : FraCaS (0-74)")
    train_tok, train_raw = load_split(
        DATASETS["fracas_75"]["path"], "train",
        DATASETS["fracas_75"]["pkey"], tokenizer
    )
    val_tok, _ = load_split(
        DATASETS["fracas_75"]["path"], "validation",
        DATASETS["fracas_75"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    trainer = make_trainer(model, tokenizer, train_tok, val_tok,
                           "exp1_fracas75_to_gqnli")

    print("\n🔹 Baseline avant IA³ (FraCaS val) :")
    base_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → {base_acc:.2%}")

    print("\n🔹 Fine-tuning IA³ sur FraCaS (0-74)...")
    trainer.train()
    print("✅ Fine-tuning IA³ terminé")

    model_path = f"models/{MODEL_SHORT}_ia3_exp1_fracas75"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Adaptateurs IA³ sauvegardés : {model_path}")
    print(f"   ℹ️  Taille très légère — seuls les vecteurs IA³ sont sauvegardés.")

    print("\n🔹 Test sur GQNLI-FR (tous splits) :")
    gqnli_tok, gqnli_raw = load_all_splits_merged(
        DATASETS["gqnli_fr"]["path"], DATASETS["gqnli_fr"]["pkey"], tokenizer
    )
    print(f"   {len(gqnli_raw)} exemples")
    trainer.eval_dataset = gqnli_tok
    test_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → Précision sur GQNLI-FR : {test_acc:.2%}")

    save_results(
        {
            "model": BASE_MODEL,
            "peft_method": "IA3",
            "experiment": "exp1_ia3",
            "train_on": "FraCaS (0-74)",
            "test_on": "GQNLI-FR (complet)",
            "train_size": len(train_raw),
            "test_size": len(gqnli_raw),
            "baseline_accuracy": base_acc,
            "final_accuracy": test_acc,
            "gain": test_acc - base_acc,
        },
        f"{MODEL_SHORT}_ia3_exp1_fracas75_to_gqnli",
        trainer=trainer,
    )


# ─────────────────────────────────────────────────────────
# EXP 2 — IA³ GQNLI-FR → test FraCaS (0-74)
# ─────────────────────────────────────────────────────────

def run_exp2():
    print_header("EXP 2 IA³ — Fine-tuning GQNLI-FR  →  test FraCaS (0-74)")

    if not check_dataset("gqnli_fr") or not check_dataset("fracas_75"):
        return

    tokenizer, model = make_ia3_model(BASE_MODEL)

    print("\n📂 Chargement TRAIN : GQNLI-FR")
    train_tok, train_raw = load_split(
        DATASETS["gqnli_fr"]["path"], "train",
        DATASETS["gqnli_fr"]["pkey"], tokenizer
    )
    val_tok, _ = load_split(
        DATASETS["gqnli_fr"]["path"], "validation",
        DATASETS["gqnli_fr"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    trainer = make_trainer(model, tokenizer, train_tok, val_tok,
                           "exp2_gqnli_to_fracas75")

    print("\n🔹 Baseline avant IA³ (GQNLI val) :")
    base_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → {base_acc:.2%}")

    print("\n🔹 Fine-tuning IA³ sur GQNLI-FR...")
    trainer.train()
    print("✅ Fine-tuning IA³ terminé")

    model_path = f"models/{MODEL_SHORT}_ia3_exp2_gqnli"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Adaptateurs IA³ sauvegardés : {model_path}")

    print("\n🔹 Test sur FraCaS (0-74) (tous splits) :")
    fracas_tok, fracas_raw = load_all_splits_merged(
        DATASETS["fracas_75"]["path"], DATASETS["fracas_75"]["pkey"], tokenizer
    )
    print(f"   {len(fracas_raw)} exemples")
    trainer.eval_dataset = fracas_tok
    test_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → Précision sur FraCaS (0-74) : {test_acc:.2%}")

    save_results(
        {
            "model": BASE_MODEL,
            "peft_method": "IA3",
            "experiment": "exp2_ia3",
            "train_on": "GQNLI-FR",
            "test_on": "FraCaS (0-74) complet",
            "train_size": len(train_raw),
            "test_size": len(fracas_raw),
            "baseline_accuracy": base_acc,
            "final_accuracy": test_acc,
            "gain": test_acc - base_acc,
        },
        f"{MODEL_SHORT}_ia3_exp2_gqnli_to_fracas75",
        trainer=trainer,
    )


# ─────────────────────────────────────────────────────────
# EXP 3 — IA³ RTE3-DEV → test DACCORD + RTE3-TEST
# ─────────────────────────────────────────────────────────

def run_exp3():
    print_header("EXP 3 IA³ — Fine-tuning RTE3-DEV  →  test DACCORD + RTE3-TEST")

    if not check_dataset("rte3_fr") or not check_dataset("daccord"):
        return

    tokenizer, model = make_ia3_model(BASE_MODEL)

    print("\n📂 Chargement TRAIN : RTE3-French (split train)")
    train_tok, train_raw = load_split(
        DATASETS["rte3_fr"]["path"], "train",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )
    val_tok, _ = load_split(
        DATASETS["rte3_fr"]["path"], "validation",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )
    print(f"   {len(train_raw)} exemples d'entraînement")

    trainer = make_trainer(model, tokenizer, train_tok, val_tok,
                           "exp3_rte3_to_daccord")

    print("\n🔹 Baseline avant IA³ (RTE3 val) :")
    base_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → {base_acc:.2%}")

    print("\n🔹 Fine-tuning IA³ sur RTE3-DEV...")
    trainer.train()
    print("✅ Fine-tuning IA³ terminé")

    model_path = f"models/{MODEL_SHORT}_ia3_exp3_rte3"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Adaptateurs IA³ sauvegardés : {model_path}")

    results_exp3 = {
        "model": BASE_MODEL,
        "peft_method": "IA3",
        "experiment": "exp3_ia3",
        "train_on": "RTE3-French (train)",
        "baseline_accuracy": base_acc,
    }

    print("\n🔹 Test sur DACCORD :")
    dacc_tok, dacc_raw = load_all_splits_merged(
        DATASETS["daccord"]["path"], DATASETS["daccord"]["pkey"], tokenizer
    )
    print(f"   {len(dacc_raw)} exemples")
    trainer.eval_dataset = dacc_tok
    dacc_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → Précision sur DACCORD : {dacc_acc:.2%}")
    results_exp3["daccord"] = {"test_size": len(dacc_raw), "accuracy": dacc_acc}

    print("\n🔹 Test sur RTE3-TEST :")
    rte3_test_tok, rte3_test_raw = load_split(
        DATASETS["rte3_fr"]["path"], "test",
        DATASETS["rte3_fr"]["pkey"], tokenizer
    )
    print(f"   {len(rte3_test_raw)} exemples")
    trainer.eval_dataset = rte3_test_tok
    rte3_test_acc = trainer.evaluate()["eval_accuracy"]
    print(f"   → Précision sur RTE3-TEST : {rte3_test_acc:.2%}")
    results_exp3["rte3_test"] = {"test_size": len(rte3_test_raw), "accuracy": rte3_test_acc}

    print_header("RÉSUMÉ EXP 3 IA³")
    print(f"  Baseline RTE3 val : {base_acc:.2%}")
    print(f"  DACCORD           : {dacc_acc:.2%}")
    print(f"  RTE3-TEST         : {rte3_test_acc:.2%}")

    save_results(results_exp3, f"{MODEL_SHORT}_ia3_exp3_rte3_to_daccord", trainer=trainer)


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
    print("\n🚀 Lancement de toutes les expériences IA³ (1, 2, 3)...")
    run_exp1()
    run_exp2()
    run_exp3()

print("\n" + "=" * 60)
print("✅ EXPÉRIENCE(S) IA³ TERMINÉE(S)")
print(f"   Résultats dans : results/metrics/{MODEL_SHORT}_ia3_exp*.json")
print(f"   Graphiques     : python3 experiments/utils/plot_comparison.py")
print("=" * 60)
