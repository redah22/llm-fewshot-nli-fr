"""
Entraînement et évaluation CamemBERT sur DACCORD — Classifieur BINAIRE
=======================================================================

DACCORD est un dataset BINAIRE :
  - 0 = compatible  (les phrases ne se contredisent pas)
  - 1 = contradiction

Ce script entraîne CamemBERT avec num_labels=2 (et non 3 comme en NLI classique),
avec early stopping et collecte des métriques par epoch pour générer des graphiques.

Expériences :
  1. Baseline    : évaluation du modèle vierge AVANT entraînement
  2. Fine-tuning : entraînement avec early stopping (patience=3)
  3. Évaluation  : test sur le split test de DACCORD

Les matrices de confusion AVANT et APRÈS + métriques par epoch sont dans le JSON.
Pour générer les graphiques : python3 experiments/plot_training.py

Utilisation :
    python3 experiments/eval_daccord_binary.py

Prérequis :
    python3 experiments/setup_data.py   # Choix 4 (DACCORD)
"""

import os
import json
import numpy as np
import torch

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────

DATASET_PATH = "data/processed/daccord"
NUM_LABELS   = 2   # Binaire ! 0=compatible, 1=contradiction
LABEL_NAMES  = ["compatible (0)", "contradiction (1)"]

print("=" * 60)
print("DACCORD — CLASSIFIEUR BINAIRE")
print("=" * 60)
print(f"Labels : 0=compatible  |  1=contradiction")

# Choix du modèle
print("\nQuel modèle ?")
print("1. CamemBERT (camembert-base)  — recommandé")
print("2. FlauBERT  (flaubert/flaubert_base_uncased)")

mc = input("\nVotre choix (1 ou 2, défaut=1): ").strip() or "1"
if mc == "2":
    BASE_MODEL   = "flaubert/flaubert_base_uncased"
    MODEL_SHORT  = "flaubert"
else:
    BASE_MODEL   = "camembert-base"
    MODEL_SHORT  = "camembert"
print(f"📦 Modèle : {BASE_MODEL}")

# ─────────────────────────────────────────────────────────
# 2. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────

if not os.path.isdir(DATASET_PATH):
    print(f"\n❌ Dataset DACCORD introuvable : {DATASET_PATH}")
    print("   → Lancez : python3 experiments/setup_data.py  (choix 4)")
    exit(1)

print(f"\n📂 Chargement de DACCORD...")
ds = DatasetDict.load_from_disk(DATASET_PATH)

train_data = ds["train"]
val_data   = ds["validation"]
test_data  = ds["test"]

print(f"   Train      : {len(train_data)} exemples")
print(f"   Validation : {len(val_data)} exemples")
print(f"   Test       : {len(test_data)} exemples")

# Afficher la distribution des classes
from collections import Counter
for split_name, split_data in [("train", train_data), ("test", test_data)]:
    cnt = Counter(int(l) for l in split_data["label"])
    total = len(split_data)
    print(f"   Distribution {split_name} : "
          f"compatible={cnt[0]} ({cnt[0]/total:.0%})  "
          f"contradiction={cnt[1]} ({cnt[1]/total:.0%})")

# ─────────────────────────────────────────────────────────
# 3. TOKENISATION
# ─────────────────────────────────────────────────────────

print(f"\n🔤 Chargement du tokenizer {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(examples):
    result = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    # Labels déjà entiers 0/1 dans DACCORD
    result["labels"] = [int(l) for l in examples["label"]]
    return result

print("Tokenisation...")
train_tok = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
val_tok   = val_data.map(tokenize,   batched=True, remove_columns=val_data.column_names)
test_tok  = test_data.map(tokenize,  batched=True, remove_columns=test_data.column_names)

for tok in [train_tok, val_tok, test_tok]:
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("✅ Tokenisation terminée")

# ─────────────────────────────────────────────────────────
# 4. MODÈLE (num_labels=2)
# ─────────────────────────────────────────────────────────

print(f"\n🤖 Chargement de {BASE_MODEL} (num_labels=2)...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)
print("✅ Modèle chargé")

# ─────────────────────────────────────────────────────────
# 5. MÉTRIQUES + MATRICE DE CONFUSION
# ─────────────────────────────────────────────────────────

# Stockage des matrices (remplies lors des évaluations)
confusion_matrices = {}

def compute_metrics_and_store(eval_pred, phase="eval"):
    """Calcule accuracy + matrice de confusion, et la stocke dans confusion_matrices."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Affichage dans le terminal
    print(f"\n📊 MATRICE DE CONFUSION ({phase}) :")
    print(f"                  Préd:0 (compat)  Préd:1 (contra)")
    print(f"  Vrai 0 (compat):  {cm[0][0]:<17} {cm[0][1]}")
    print(f"  Vrai 1 (contra):  {cm[1][0]:<17} {cm[1][1]}")

    # Precision / Recall par classe
    print(f"\n  Rappel compatible   : {cm[0][0]/(cm[0][0]+cm[0][1]+1e-9):.1%}")
    print(f"  Rappel contradiction: {cm[1][1]/(cm[1][0]+cm[1][1]+1e-9):.1%}")

    acc = accuracy_score(labels, preds)

    # Sauvegarder pour le JSON final
    confusion_matrices[phase] = {
        "matrix": cm.tolist(),
        "labels": ["compatible (0)", "contradiction (1)"],
        "recall_compatible":    float(cm[0][0] / (cm[0][0] + cm[0][1] + 1e-9)),
        "recall_contradiction": float(cm[1][1] / (cm[1][0] + cm[1][1] + 1e-9)),
    }

    return {"accuracy": acc}


def make_metrics_fn(phase):
    """Wrapper pour passer la phase à compute_metrics_and_store."""
    def fn(eval_pred):
        return compute_metrics_and_store(eval_pred, phase=phase)
    return fn

# ─────────────────────────────────────────────────────────
# 6. CALLBACK — Collecte des métriques par epoch/step
# ─────────────────────────────────────────────────────────

class MetricsCollectorCallback(TrainerCallback):
    """
    Collecte toutes les métriques d'entraînement et d'évaluation.
    - train_history : loss, grad_norm, learning_rate (par logging step)
    - eval_history  : eval_loss, eval_accuracy (par epoch)
    """
    def __init__(self):
        self.train_history = []   # Métriques de train (par step)
        self.eval_history  = []   # Métriques d'eval (par epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Métriques de train (contiennent 'loss' mais pas 'eval_loss')
        if "loss" in logs and "eval_loss" not in logs:
            self.train_history.append({
                "step":          state.global_step,
                "epoch":         round(logs.get("epoch", 0), 4),
                "loss":          logs.get("loss"),
                "grad_norm":     logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
            })
        # Métriques d'évaluation (fin de chaque epoch)
        if "eval_loss" in logs:
            self.eval_history.append({
                "epoch":         round(logs.get("epoch", 0), 4),
                "eval_loss":     logs.get("eval_loss"),
                "eval_accuracy": logs.get("eval_accuracy"),
            })

metrics_callback = MetricsCollectorCallback()

# ─────────────────────────────────────────────────────────
# 7. TRAINER
# ─────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=f"checkpoints/{MODEL_SHORT}_daccord_binary",
    eval_strategy="epoch",
    save_strategy="epoch",             # Nécessaire pour early stopping
    save_total_limit=2,                # Garde seulement les 2 meilleurs checkpoints
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,               # Augmenté : l'early stopping arrêtera au bon moment
    weight_decay=0.01,
    load_best_model_at_end=True,       # Recharge le meilleur modèle à la fin
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    logging_steps=20,
    report_to="none",
    remove_unused_columns=False,
)

# ─────────────────────────────────────────────────────────
# 7. ÉVALUATION BASELINE (avant entraînement)
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 : BASELINE (modèle vierge)")
print("=" * 60)

baseline_trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_tok,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=make_metrics_fn("baseline_test"),
)

baseline_metrics = baseline_trainer.evaluate()
baseline_acc = baseline_metrics["eval_accuracy"]
print(f"\n>> Précision Baseline (test) : {baseline_acc:.2%}  (Attendu : ~50% = hasard binaire)")

# ─────────────────────────────────────────────────────────
# 8. FINE-TUNING
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 : FINE-TUNING sur DACCORD train")
print("=" * 60)
print(f"Exemples train : {len(train_data)}")
print(f"Epochs : {training_args.num_train_epochs}")
print(f"Batch size : {training_args.per_device_train_batch_size}")

# EarlyStoppingCallback : arrête si eval_accuracy ne s'améliore pas pendant 3 epochs
early_stop = EarlyStoppingCallback(early_stopping_patience=3)

ft_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=make_metrics_fn("validation_during_training"),
    callbacks=[metrics_callback, early_stop],
)

ft_trainer.train()
print("\n✅ Fine-tuning terminé")

# Sauvegarde du modèle
model_path = f"models/{MODEL_SHORT}_daccord_binary"
os.makedirs(model_path, exist_ok=True)
ft_trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"✅ Modèle sauvegardé : {model_path}")

# ─────────────────────────────────────────────────────────
# 9. ÉVALUATION FINALE (sur test set)
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3 : ÉVALUATION FINALE (test set)")
print("=" * 60)

# Remplacer le eval_dataset par le test set
ft_trainer.eval_dataset = test_tok
ft_trainer.compute_metrics = make_metrics_fn("final_test")

final_metrics = ft_trainer.evaluate()
final_acc = final_metrics["eval_accuracy"]
print(f"\n>> Précision finale (test) : {final_acc:.2%}")
print(f">> Gain vs baseline        : {final_acc - baseline_acc:+.2%}")

# ─────────────────────────────────────────────────────────
# 10. SAUVEGARDE DES RÉSULTATS
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print(f"  Modèle              : {BASE_MODEL}")
print(f"  Dataset             : DACCORD (binaire)")
print(f"  Baseline (test)     : {baseline_acc:.2%}")
print(f"  Après fine-tuning   : {final_acc:.2%}")
print(f"  Gain                : {final_acc - baseline_acc:+.2%}")
print(f"\n  Matrices de confusion :")
for phase, data in confusion_matrices.items():
    print(f"    [{phase}]")
    print(f"      compatible   rappel: {data['recall_compatible']:.1%}")
    print(f"      contradiction rappel: {data['recall_contradiction']:.1%}")

# Récupérer l'epoch où le meilleur modèle a été sauvegardé
best_epoch = metrics_callback.eval_history[
    np.argmax([e["eval_accuracy"] for e in metrics_callback.eval_history])
]["epoch"] if metrics_callback.eval_history else None

best_val_acc = max(
    (e["eval_accuracy"] for e in metrics_callback.eval_history),
    default=None
)

os.makedirs("results/metrics", exist_ok=True)
results_path = f"results/metrics/{MODEL_SHORT}_daccord_binary.json"
results = {
    "model": BASE_MODEL,
    "dataset": "DACCORD",
    "task": "binary_classification",
    "num_labels": NUM_LABELS,
    "labels": LABEL_NAMES,
    "train_size": len(train_data),
    "val_size": len(val_data),
    "test_size": len(test_data),
    "max_epochs": training_args.num_train_epochs,
    "epochs_trained": len(metrics_callback.eval_history),
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_acc,
    "learning_rate": training_args.learning_rate,
    "early_stopping_patience": 3,
    "baseline_accuracy": baseline_acc,
    "final_accuracy": final_acc,
    "gain": final_acc - baseline_acc,
    "confusion_matrices": confusion_matrices,
    # Historique complet pour les graphiques
    "training_history": {
        "train_steps": metrics_callback.train_history,   # loss, grad_norm, lr par step
        "eval_epochs": metrics_callback.eval_history,    # eval_loss, eval_accuracy par epoch
    },
}
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Résultats complets sauvegardés : {results_path}")
print(f"   → Meilleure epoch    : {best_epoch}")
print(f"   → Meilleure val acc  : {best_val_acc:.2%}")
print(f"   → Pour les graphiques : python3 experiments/plot_training.py")
