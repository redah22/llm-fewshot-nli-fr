"""
Script de Sweep WandB pour T5Gemma (Architecture Encoder-Decoder).
Configuration unifiée multi-dataset selon la distribution CLAUDE.md.

Distribution :
  TRAIN : RTE3-DEV (90%) + SICK Train + DACCORD (1ère moitié) + GQNLI (80-100, 180-200, 280-300)
  VAL   : SICK-VAL + RTE3-DEV (10%) + GQNLI (60-80, 160-180, 260-280)
  TEST  : RTE3-test + SICK-Test + DACCORD (2ème moitié) + GQNLI (0-60, 100-160, 200-260)
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
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
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import confusion_matrix, f1_score

BASE_MODEL = "google/t5gemma-2-270m-270m"
MODEL_SHORT = "t5gemma"
EXP_NAME = "sweep_unified_t5gemma"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA pour QLoRA.")
    exit(1)

# 1. CHARGEMENT UNIFIÉ DES DONNÉES


def _normalize_columns(ds, premise_key="premise"):
    """Normalise un dataset pour avoir uniquement premise, hypothesis, label."""
    cols_to_remove = [c for c in ds.column_names if c not in {premise_key, "hypothesis", "label"}]
    ds = ds.remove_columns(cols_to_remove)
    if premise_key != "premise":
        ds = ds.rename_column(premise_key, "premise")
    return ds


def load_unified_dataset():
    """Charge et combine les datasets selon la distribution unifiée."""
    print("Chargement unifié des datasets...")
    all_train, all_val, all_test = [], [], []

    # --- GQNLI-FR (ranges sans chevauchement) ---
    print("  → GQNLI-FR...")
    gqnli = load_dataset('maximoss/gqnli-fr')['test']
    gqnli_train = gqnli.select(list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300)))
    gqnli_val = gqnli.select(list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280)))
    gqnli_test = gqnli.select(list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260)))
    all_train.append(_normalize_columns(gqnli_train))
    all_val.append(_normalize_columns(gqnli_val))
    all_test.append(_normalize_columns(gqnli_test))

    # --- RTE3-FR (DEV 90% train / 10% val, test séparé) ---
    print("  → RTE3-FR...")
    rte3 = load_dataset('maximoss/rte3-french')
    dev_keys = [k for k in rte3.keys() if k != 'test']
    rte3_dev = concatenate_datasets([rte3[k] for k in dev_keys]).shuffle(seed=42)
    split_90 = int(len(rte3_dev) * 0.9)
    all_train.append(_normalize_columns(rte3_dev.select(range(0, split_90))))
    all_val.append(_normalize_columns(rte3_dev.select(range(split_90, len(rte3_dev)))))
    if 'test' in rte3:
        all_test.append(_normalize_columns(rte3['test']))

    # --- SICK-FR (splits originaux) ---
    print("  → SICK-FR...")
    sick = load_dataset('maximoss/sick-fr')

    def convert_sick(ex):
        lbl = str(ex['entailment_label']).strip().upper()
        if lbl == 'ENTAILMENT': label_id = 0
        elif lbl == 'NEUTRAL': label_id = 1
        elif lbl == 'CONTRADICTION': label_id = 2
        else: label_id = 1
        return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}

    for split_name, target in [('train', all_train), ('validation', all_val), ('test', all_test)]:
        if split_name in sick:
            converted = sick[split_name].map(convert_sick, remove_columns=sick[split_name].column_names)
            target.append(converted)

    # --- DACCORD (1ère moitié = train, 2ème moitié = test) ---
    print("  → DACCORD...")
    daccord = load_dataset('maximoss/daccord-contradictions')['train'].shuffle(seed=42)

    def fix_daccord_label(ex):
        if ex["label"] == 1:
            ex["label"] = 2  # Contradiction: 1 → 2 (standard NLI)
        return ex

    daccord = daccord.map(fix_daccord_label)
    half = len(daccord) // 2
    all_train.append(_normalize_columns(daccord.select(range(0, half))))
    all_test.append(_normalize_columns(daccord.select(range(half, len(daccord)))))

    # Combinaison finale
    train_ds = concatenate_datasets(all_train).shuffle(seed=42)
    val_ds = concatenate_datasets(all_val).shuffle(seed=42)
    test_ds = concatenate_datasets(all_test).shuffle(seed=42)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})


# 2. CONFIGURATION DU SWEEP

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [32]},
        "lora_alpha": {"values": [64]},
        "learning_rate": {"values": [5e-4]},
        "lora_dropout": {"values": [0.1]},
    }
}

# 3. CHARGEMENT & TOKENISATION

UNIFIED = load_unified_dataset()
global_tokenizer = AutoProcessor.from_pretrained(BASE_MODEL).tokenizer

LABEL_MAP = {
    "yes": "vrai", "entailment": "vrai", "0": "vrai",
    "unknown": "neutre", "undef": "neutre", "neutral": "neutre", "1": "neutre",
    "no": "faux", "contradiction": "faux", "2": "faux"
}


def normalize_label(label):
    if isinstance(label, int):
        if label == 0: return "vrai"
        if label == 1: return "neutre"
        if label == 2: return "faux"
        return "neutre"
    s = str(label).lower().strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try:
        val = int(s)
        if val == 0: return "vrai"
        if val == 1: return "neutre"
        if val == 2: return "faux"
        return "neutre"
    except Exception:
        return "neutre"


def preprocess_fn(examples):
    consigne = "Consigne : Prédire si l'hypothèse est vraie, fausse ou neutre d'après la prémisse.\n"
    inputs = [
        f"{consigne}Prémisse : {p}\nHypothèse : {h}\nRéponse :"
        for p, h in zip(examples["premise"], examples["hypothesis"])
    ]
    model_inputs = global_tokenizer(inputs, max_length=256, truncation=True, padding=False)
    targets = [normalize_label(l) for l in examples["label"]]
    labels = global_tokenizer(text_target=targets, max_length=8, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data = UNIFIED['train'].map(preprocess_fn, batched=True, remove_columns=UNIFIED['train'].column_names)
val_data = UNIFIED['validation'].map(preprocess_fn, batched=True, remove_columns=UNIFIED['validation'].column_names)
test_data = UNIFIED['test'].map(preprocess_fn, batched=True, remove_columns=UNIFIED['test'].column_names)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    dec_preds = [p.strip() for p in global_tokenizer.batch_decode(preds, skip_special_tokens=True)]
    dec_labels = [l.strip() for l in global_tokenizer.batch_decode(labels, skip_special_tokens=True)]

    print(f"\n[DEBUG] Extraits générés : {dec_preds[:5]}")
    print(f"[DEBUG] Extraits cibles  : {dec_labels[:5]}")

    LABEL_TO_INT = {"vrai": 0, "neutre": 1, "faux": 2}

    import re
    cleaned_preds = []
    for p in dec_preds:
        match = re.search(r'(vrai|faux|neutre)', p.lower())
        cleaned_preds.append(match.group(1) if match else "neutre")

    int_preds = [LABEL_TO_INT[p] for p in cleaned_preds]
    int_labels = [LABEL_TO_INT.get(l.lower(), 1) for l in dec_labels]

    try:
        cm = confusion_matrix(int_labels, int_preds, labels=[0, 1, 2])
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception:
        pass

    acc = sum(p == l for p, l in zip(cleaned_preds, dec_labels)) / max(1, len(dec_labels))
    return {
        "accuracy": acc,
        "f1_score": f1_score(int_labels, int_preds, average="macro")
    }


# 4. FONCTION D'ENTRAÎNEMENT (SWEEP WANDB)

def train_t5_qlora():
    run = wandb.init()
    config = wandb.config

    run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
    print(f"\n{'='*60}\nSWEEP T5GEMMA UNIFIÉ: {run_label}\n{'='*60}")

    print("Chargement en QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, device_map="auto", quantization_config=bnb_config
    )
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules="all-linear"
    )

    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {trainable:,} ({100 * trainable / total:.2f}%)")

    args = Seq2SeqTrainingArguments(
        output_dir=f"/tmp/checkpoints_{EXP_NAME}_{run_label}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        predict_with_generate=True,
        generation_max_length=8,
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False
    )

    collator = DataCollatorForSeq2Seq(tokenizer=global_tokenizer, model=None, padding=True, pad_to_multiple_of=8)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # Évaluation zero-shot (avant entraînement)
    print(f"\n[ZERO-SHOT] Évaluation du modèle vierge sur {len(test_data)} exemples...")
    zs_results = trainer.evaluate(test_data, metric_key_prefix="zeroshot")
    zs_acc = zs_results["zeroshot_accuracy"]
    zs_f1 = zs_results.get("zeroshot_f1_score", 0.0)
    print(f"ZERO-SHOT ACCURACY : {zs_acc:.2%}")
    print(f"ZERO-SHOT F1-SCORE : {zs_f1:.4f}")
    wandb.summary["zeroshot_accuracy"] = zs_acc
    wandb.summary["zeroshot_f1_score"] = zs_f1

    trainer.train()

    # Évaluation finale sur Test
    print(f"\n[EVAL FINALE] {len(test_data)} exemples...")
    test_results = trainer.evaluate(test_data, metric_key_prefix="test")
    test_acc = test_results["test_accuracy"]
    print(f"TEST ACCURACY : {test_acc:.2%}")
    if "test_f1_score" in test_results:
        test_f1 = test_results["test_f1_score"]
        print(f"TEST F1-SCORE : {test_f1:.4f}")
        wandb.summary["final_test_f1_score"] = test_f1
    wandb.summary["final_test_accuracy"] = test_acc

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
        confirm = input(f"\nLancer {total_runs} sweeps T5Gemma QLoRA unifié ? (o/n): ").strip().lower()

    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_t5_qlora, count=total_runs)
        print("\nTerminé ! Consultez les résultats sur WandB.")
    else:
        print("Annulé.")
