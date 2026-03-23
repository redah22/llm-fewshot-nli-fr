"""
Expériences Cross-Dataset NLI avec T5Gemma 2 (270M) - Fine-tuning Complet
========================================================================

Reproduction des expériences cross-dataset pour le modèle génératif T5Gemma.

  EXP 0 — Baseline : test du modèle vierge sur TOUS les datasets
  EXP 1 — Fine-tuning FraCaS (0-74)  → test sur tout GQNLI-FR
  EXP 2 — Fine-tuning GQNLI-FR       → test sur FraCaS (0-74)
  EXP 3 — Fine-tuning RTE3-DEV       → test sur DACCORD + RTE3-TEST

Utilisation :
    python3 experiments/fine_tuning/cross_dataset_t5gemma.py
"""

import os
import json
import numpy as np
import torch

from datasets import DatasetDict, concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("EXPÉRIENCES CROSS-DATASET NLI — T5GEMMA (270M)")
print("=" * 60)

BASE_MODEL = "google/t5gemma-2-270m-270m"
MODEL_SHORT = "t5gemma"

if torch.cuda.is_available():
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CPU détecté. L'entraînement sera très lent.")

print("\n" + "=" * 60)
print("Quelle expérience ?")
print("-" * 60)
print("0. Baseline — Modèle vierge testé sur TOUS les datasets")
print("1. FraCaS (0-74)  →  test GQNLI-FR (complet)")
print("2. GQNLI-FR       →  test FraCaS (0-74)")
print("3. RTE3-DEV       →  test DACCORD + RTE3-TEST")
print("4. Toutes les expériences (1, 2 et 3) à la suite")

exp_choice = input("\nVotre choix (0, 1, 2, 3 ou 4): ").strip()
if exp_choice not in ["0", "1", "2", "3", "4"]:
    print("❌ Choix invalide!")
    exit(1)

# ─────────────────────────────────────────────────────────
# 2. DATASETS & FONCTIONS
# ─────────────────────────────────────────────────────────

DATASETS = {
    "gqnli_fr": {"path": "data/processed/gqnli_fr", "pkey": "premise", "label": "GQNLI-FR"},
    "fracas_75": {"path": "data/processed/fracas_subset_75", "pkey": "premises", "label": "FraCaS (0-74)"},
    "daccord": {"path": "data/processed/daccord", "pkey": "premise", "label": "DACCORD"},
    "rte3_fr": {"path": "data/processed/rte3_fr", "pkey": "premise", "label": "RTE3-French"},
}

def check_dataset(key):
    if not os.path.isdir(DATASETS[key]["path"]):
        print(f"❌ Dataset '{DATASETS[key]['label']}' introuvable. Lancez setup_data.py.")
        return False
    return True

LABEL_MAP_STR = {
    "yes": "0", "entailment": "0",
    "unknown": "1", "undef": "1", "neutral": "1",
    "no": "2", "contradiction": "2"
}

def normalize_label(label) -> str:
    if isinstance(label, int):
        return str(label) if label in [0, 1, 2] else "1"
    label_str = str(label).lower().strip()
    if label_str in LABEL_MAP_STR:
        return LABEL_MAP_STR[label_str]
    try:
        val = int(label_str)
        return str(val) if val in [0, 1, 2] else "1"
    except ValueError:
        return "1"

class T5GemmaMetricsCb(TrainerCallback):
    def __init__(self):
        self.train_history, self.eval_history = [], []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if "loss" in logs and "eval_loss" not in logs:
            self.train_history.append({
                "step": state.global_step, "epoch": round(logs.get("epoch", 0), 4),
                "loss": logs.get("loss"), "learning_rate": logs.get("learning_rate"),
            })
        if "eval_loss" in logs:
            self.eval_history.append({
                "epoch": round(logs.get("epoch", 0), 4),
                "eval_loss": logs.get("eval_loss"),
                "eval_accuracy": logs.get("eval_accuracy", 0),
            })

def save_results(results: dict, filename: str, trainer=None):
    if trainer and hasattr(trainer, "_metrics_cb"):
        cb = trainer._metrics_cb
        best_idx = np.argmax([e["eval_accuracy"] for e in cb.eval_history]) if cb.eval_history else 0
        results["training_history"] = {
            "train_steps": cb.train_history,
            "eval_epochs": cb.eval_history,
            "epochs_trained": len(cb.eval_history),
            "best_epoch": cb.eval_history[best_idx]["epoch"] if cb.eval_history else None,
            "best_val_acc": cb.eval_history[best_idx]["eval_accuracy"] if cb.eval_history else None,
        }
    os.makedirs("results/metrics", exist_ok=True)
    path = f"results/metrics/{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\u2705 Résultats dans : {path}")

# ─────────────────────────────────────────────────────────
# 3. PIPELINE SEQ2SEQ
# ─────────────────────────────────────────────────────────

def get_model_and_tokenizer():
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    tok = processor.tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map="auto")
    model = model.float()
    return tok, model

global_tokenizer = None

def make_preprocess(premise_key):
    def preprocess(examples):
        inputs = [f"nli: {p} </s> {h}" for p, h in zip(examples[premise_key], examples["hypothesis"])]
        model_inputs = global_tokenizer(inputs, max_length=256, truncation=True, padding=False)
        targets = [normalize_label(l) for l in examples["label"]]
        labels = global_tokenizer(text_target=targets, max_length=4, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess

def load_and_prep(ds_key, split=None, merge_all=False):
    ds = DatasetDict.load_from_disk(DATASETS[ds_key]["path"])
    raw_data = concatenate_datasets(list(ds.values())) if merge_all else ds[split]
    prep_fn = make_preprocess(DATASETS[ds_key]["pkey"])
    tokenized = raw_data.map(prep_fn, batched=True, remove_columns=raw_data.column_names)
    return tokenized, raw_data

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    dec_preds = [p.strip() for p in global_tokenizer.batch_decode(preds, skip_special_tokens=True)]
    dec_labels = [l.strip() for l in global_tokenizer.batch_decode(labels, skip_special_tokens=True)]
    
    # Try to clean preds if model hallucinates slightly
    cleaned_preds = [p if p in ["0","1","2"] else "1" for p in dec_preds]
    int_preds = [int(p) for p in cleaned_preds]
    int_labels = [int(l) if l in ["0","1","2"] else 1 for l in dec_labels]
    
    try:
        cm = confusion_matrix(int_labels, int_preds, labels=[0, 1, 2])
        print("\n📊 MATRICE DE CONFUSION:")
        print(f"  Vrai 0: {cm[0][0]:<4} {cm[0][1]:<4} {cm[0][2]}")
        print(f"  Vrai 1: {cm[1][0]:<4} {cm[1][1]:<4} {cm[1][2]}")
        print(f"  Vrai 2: {cm[2][0]:<4} {cm[2][1]:<4} {cm[2][2]}")
    except Exception: pass
    
    acc = sum(p == l for p, l in zip(cleaned_preds, dec_labels)) / max(1, len(dec_labels))
    return {"accuracy": acc}

def make_trainer(model, tok, train_ds, eval_ds, run_name, epochs=15):
    cb = T5GemmaMetricsCb()
    args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{MODEL_SHORT}_{run_name}",
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        predict_with_generate=True, generation_max_length=4,
        logging_steps=10, report_to="none", save_total_limit=2,
        gradient_checkpointing=True, dataloader_pin_memory=False
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=None, padding=True, pad_to_multiple_of=8)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4), cb]
    )
    trainer._metrics_cb = cb
    return trainer

# ─────────────────────────────────────────────────────────
# 4. EXPERIENCES
# ─────────────────────────────────────────────────────────

def run_exp0():
    print("\n=== EXP 0 : BASELINE TOUS DATASETS ===")
    global global_tokenizer
    global_tokenizer, model = get_model_and_tokenizer()
    res = {}
    for k in DATASETS:
        if not check_dataset(k): continue
        tok_ds, raw = load_and_prep(k, merge_all=True)
        trainer = make_trainer(model, global_tokenizer, None, tok_ds, f"base_{k}", epochs=1)
        acc = trainer.evaluate()
        print(f" {DATASETS[k]['label']} -> {acc['eval_accuracy']:.2%}")
        res[k] = acc["eval_accuracy"]
    save_results({"model": BASE_MODEL, "exp": "baseline", "res": res}, f"{MODEL_SHORT}_exp0_baseline")

def run_exp1():
    print("\n=== EXP 1 : FraCaS(0-74) -> GQNLI ===")
    global global_tokenizer
    global_tokenizer, model = get_model_and_tokenizer()
    train_tok, train_raw = load_and_prep("fracas_75", "train")
    val_tok, _ = load_and_prep("fracas_75", "validation")
    
    trainer = make_trainer(model, global_tokenizer, train_tok, val_tok, "exp1", epochs=15)
    base_acc = trainer.evaluate()["eval_accuracy"]
    trainer.train()
    
    gqnli_tok, gqnli_raw = load_and_prep("gqnli_fr", merge_all=True)
    trainer.eval_dataset = gqnli_tok
    test_acc = trainer.evaluate()["eval_accuracy"]
    
    save_results({
        "model": BASE_MODEL, "exp": "exp1_t5gemma", 
        "baseline_val": base_acc, "final_test": test_acc
    }, f"{MODEL_SHORT}_exp1_fracas_to_gqnli", trainer)

def run_exp2():
    print("\n=== EXP 2 : GQNLI -> FraCaS(0-74) ===")
    global global_tokenizer
    global_tokenizer, model = get_model_and_tokenizer()
    train_tok, train_raw = load_and_prep("gqnli_fr", "train")
    val_tok, _ = load_and_prep("gqnli_fr", "validation")
    
    trainer = make_trainer(model, global_tokenizer, train_tok, val_tok, "exp2", epochs=15)
    base_acc = trainer.evaluate()["eval_accuracy"]
    trainer.train()
    
    fracas_tok, fracas_raw = load_and_prep("fracas_75", merge_all=True)
    trainer.eval_dataset = fracas_tok
    test_acc = trainer.evaluate()["eval_accuracy"]
    
    save_results({
        "model": BASE_MODEL, "exp": "exp2_t5gemma", 
        "baseline_val": base_acc, "final_test": test_acc
    }, f"{MODEL_SHORT}_exp2_gqnli_to_fracas", trainer)

def run_exp3():
    print("\n=== EXP 3 : RTE3-DEV -> DACCORD & RTE3-TEST ===")
    global global_tokenizer
    global_tokenizer, model = get_model_and_tokenizer()
    train_tok, train_raw = load_and_prep("rte3_fr", "train")
    val_tok, _ = load_and_prep("rte3_fr", "validation")
    
    trainer = make_trainer(model, global_tokenizer, train_tok, val_tok, "exp3", epochs=15)
    base_acc = trainer.evaluate()["eval_accuracy"]
    trainer.train()
    
    dacc_tok, _ = load_and_prep("daccord", merge_all=True)
    trainer.eval_dataset = dacc_tok
    dacc_acc = trainer.evaluate()["eval_accuracy"]
    
    rte3t_tok, _ = load_and_prep("rte3_fr", "test")
    trainer.eval_dataset = rte3t_tok
    rte3t_acc = trainer.evaluate()["eval_accuracy"]
    
    save_results({
        "model": BASE_MODEL, "exp": "exp3_t5gemma", 
        "baseline_val": base_acc, "test_daccord": dacc_acc, "test_rte3": rte3t_acc
    }, f"{MODEL_SHORT}_exp3_rte3_to_daccord", trainer)

if exp_choice == "0": run_exp0()
elif exp_choice == "1": run_exp1()
elif exp_choice == "2": run_exp2()
elif exp_choice == "3": run_exp3()
elif exp_choice == "4":
    run_exp1(); run_exp2(); run_exp3()

print("\n✅ Terminé !")
