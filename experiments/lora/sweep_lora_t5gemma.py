"""
WandB Sweep — QLoRA T5Gemma 2 (270M)
=====================================================================

Script autonome optimisé pour Kaggle/Colab (GPU requis pour QLoRA).
Télécharge et prépare dynamiquement les datasets sans dépendre de setup_data.py.
Lance un sweep automatique sur WandB pour trouver les meilleurs hyperparamètres.

Utilisation :
    python3 experiments/lora/sweep_lora_t5gemma.py
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force 1 seul GPU (Corrige le bug Kaggle T4x2)

import json
import numpy as np
import torch
import wandb
import shutil

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("WANDB SWEEP — QLORA T5GEMMA (KAGGLE EDITION)")
print("=" * 60)

BASE_MODEL = "google/t5gemma-2-270m-270m"
MODEL_SHORT = "t5gemma"

if not torch.cuda.is_available():
    print("❌ ERREUR CRITIQUE : Ce script nécessite un GPU CUDA pour QLoRA.")
    print("Veuillez activer le GPU T4 x2 (ou P100) dans les paramètres de votre Notebook Kaggle.")
    exit(1)

# ─────────────────────────────────────────────────────────
# 1. TÉLÉCHARGEMENT & PRÉPARATION DE DONNÉES À LA VOLÉE
# ─────────────────────────────────────────────────────────

def get_dataset(name):
    """Télécharge et segmente dynamiquement le jeu de données depuis Hub."""
    print(f"📥 Téléchargement et structuration de {name}...")
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
        total = len(data)
        train_size = int(total * 0.6)
        val_size = int(total * 0.2)
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
        train_size = int(total * 0.6)
        val_size = int(total * 0.2)
        ds = DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        })
        return ds, "premise"
        
    raise ValueError(f"Dataset {name} inconnu.")

# ─────────────────────────────────────────────────────────
# 2. CONFIGURATION DU SWEEP (RÉDUIT)
# ─────────────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [4, 8, 16]},
        "lora_alpha": {"values": [8, 16]},
        "learning_rate": {"values": [3e-4, 5e-4]},
        "lora_dropout": {"values": [0.1]},
    }
} # Total : 3 x 2 x 2 x 1 = 12 runs (idéal pour tester rapidement la robustesse)

import sys

# Si on passe un argument (ex: python script.py 2), on prend cet argument sans demander
if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
    print(f"\nVotre choix (1, 2 ou 3): {exp_choice} (via argument)")
else:
    print("\nQuelle expérience (Q)LoRA T5Gemma utiliser pour le sweep ?")
    print("1. FraCaS (0-74)  →  test GQNLI-FR")
    print("2. GQNLI-FR       →  test FraCaS (0-74)")
    print("3. RTE3-DEV       →  test DACCORD + RTE3-TEST")
    exp_choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if exp_choice == "1":
    EXP_NAME = "sweep_fracas_to_gqnli_t5gemma"
    train_ds_name, test_ds_name = "fracas_75", "gqnli_fr"
elif exp_choice == "2":
    EXP_NAME = "sweep_gqnli_to_fracas_t5gemma"
    train_ds_name, test_ds_name = "gqnli_fr", "fracas_75"
elif exp_choice == "3":
    EXP_NAME = "sweep_rte3_to_daccord_t5gemma"
    train_ds_name, test_ds_name = "rte3_fr", "daccord"
else:
    print("❌ Choix invalide!"); exit(1)

# On télécharge les datasets cibles
TRAIN_DICT, TRAIN_PKEY = get_dataset(train_ds_name)
TEST_DICT, TEST_PKEY = get_dataset(test_ds_name)

# ─────────────────────────────────────────────────────────
# 3. PRE-PROCESSING
# ─────────────────────────────────────────────────────────

global_tokenizer = AutoProcessor.from_pretrained(BASE_MODEL).tokenizer
LABEL_MAP_STR = {"yes": "0", "entailment": "0", "unknown": "1", "undef": "1", "neutral": "1", "no": "2", "contradiction": "2"}

def normalize_label(label):
    if isinstance(label, int): return str(label) if label in [0, 1, 2] else "1"
    label_str = str(label).lower().strip()
    if label_str in LABEL_MAP_STR: return LABEL_MAP_STR[label_str]
    try:
        val = int(label_str)
        return str(val) if val in [0, 1, 2] else "1"
    except ValueError:
        return "1"

def preprocess_fn(examples, p_key):
    inputs = [f"nli: {p} </s> {h}" for p, h in zip(examples[p_key], examples["hypothesis"])]
    model_inputs = global_tokenizer(inputs, max_length=256, truncation=True, padding=False)
    targets = [normalize_label(l) for l in examples["label"]]
    labels = global_tokenizer(text_target=targets, max_length=4, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# On tokenise directement
train_data = TRAIN_DICT['train'].map(lambda ex: preprocess_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['train'].column_names)
val_data = TRAIN_DICT['validation'].map(lambda ex: preprocess_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['validation'].column_names)

# Le test est la concaténation de tous les splits cibles
test_untokenized = concatenate_datasets(list(TEST_DICT.values()))
test_data = test_untokenized.map(lambda ex: preprocess_fn(ex, TEST_PKEY), batched=True, remove_columns=test_untokenized.column_names)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    dec_preds = [p.strip() for p in global_tokenizer.batch_decode(preds, skip_special_tokens=True)]
    dec_labels = [l.strip() for l in global_tokenizer.batch_decode(labels, skip_special_tokens=True)]
    
    cleaned_preds = [p if p in ["0","1","2"] else "1" for p in dec_preds]
    int_preds = [int(p) for p in cleaned_preds]
    int_labels = [int(l) if l in ["0","1","2"] else 1 for l in dec_labels]
    
    try:
        cm = confusion_matrix(int_labels, int_preds, labels=[0, 1, 2])
        print(f"\n📊 CM (Vrai En Ligne, Préd En Colonne):\n{cm}")
    except Exception: pass
    
    acc = sum(p == l for p, l in zip(cleaned_preds, dec_labels)) / max(1, len(dec_labels))
    return {"accuracy": acc}

# ─────────────────────────────────────────────────────────
# 4. FONCTION D'ENTRAÎNEMENT (POUR WANDB SWEEP)
# ─────────────────────────────────────────────────────────

def train_t5_qlora():
    run = wandb.init()
    config = wandb.config

    run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
    print(f"\n{'='*60}\n🔄 SWEEP T5GEMMA RUN: {run_label}\n{'='*60}")

    print("🚀 Chargement en QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📐 Paramètres entraînables : {trainable:,} ({100 * trainable / total:.2f}%)")

    args = Seq2SeqTrainingArguments(
        output_dir=f"/tmp/checkpoints_{EXP_NAME}_{run_label}", # Kaggle limite l'espace disque permanent
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=16,  # 🚀 Batch 8x plus grand (remplit le GPU, soulage le CPU)
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        predict_with_generate=True,
        generation_max_length=4,
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        dataloader_num_workers=2,        # 🧵 Multithreading CPU pour préparer les données
        dataloader_pin_memory=True
    )

    collator = DataCollatorForSeq2Seq(tokenizer=global_tokenizer, model=None, padding=True, pad_to_multiple_of=8)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    trainer.train()

    # Evaluation Finale sur Test Cross-Dataset
    print("\n🎯 Évaluation Cross-Dataset Finale...")
    test_results = trainer.evaluate(test_data)
    test_acc = test_results["eval_accuracy"]
    print(f"✅ FINAL TEST ACCURACY : {test_acc:.2%}")

    wandb.log({"test/cross_dataset_accuracy": test_acc})
    wandb.summary["test_cross_dataset_accuracy"] = test_acc

    # Nettoyage des checkpoints locaux de ce run (Kaggle n'a que 20 Go par défaut)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    wandb.finish()

# ─────────────────────────────────────────────────────────
# 5. DÉMARRAGE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_runs = 1
    for param in SWEEP_CONFIG["parameters"].values():
        total_runs *= len(param["values"])
    
    if len(sys.argv) > 1:
        confirm = "o" # Mode automatique sans confirmation
    else:
        confirm = input(f"\n⚠️  KAGGLE: Lancer automatiquement {total_runs} sweeps T5Gemma QLoRA ? (o/n): ").strip().lower()
        
    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\n✅ Sweep créé ! ID: {sweep_id}")
        print(f"📊 Suivez en direct : https://wandb.ai/votre_profil/fewshot-nli-fr/sweeps/{sweep_id}")
        wandb.agent(sweep_id, function=train_t5_qlora, count=total_runs)
        print("\n✅ Terminé ! Consultez les résultats sur WandB.")
    else:
        print("❌ Annulé.")
