"""
Script de Sweep WandB pour T5Gemma (Architecture Encoder-Decoder).
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

BASE_MODEL = "google/t5gemma-2-270m-270m"
MODEL_SHORT = "t5gemma"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA pour QLoRA.")
    print("Veuillez activer le GPU T4 x2 (ou P100) dans les paramètres de votre Notebook Kaggle.")
    exit(1)

# 1. TÉLÉCHARGEMENT & PRÉPARATION DE DONNÉES À LA VOLÉE

def get_dataset(name):
    """Télécharge et segmente dynamiquement le jeu de données depuis Hub."""
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
        
    elif name == "fracas_full":
        fracas = load_dataset('maximoss/fracas')['train']
        # On retire tous les 'undef'
        fracas = fracas.filter(lambda x: str(x['label']).strip().lower() != "undef")
        shuffled = fracas.shuffle(seed=42)
        total = len(shuffled)
        train_size = int(total * 0.8)
        ds = DatasetDict({
            'train': shuffled.select(range(0, train_size)),
            'validation': shuffled.select(range(train_size, total)),
            'test': shuffled.select(range(train_size, total))
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
        
    elif name == "sick_fr":
        sick = load_dataset('maximoss/sick-fr')
        if len(sick.keys()) > 1:
            data = concatenate_datasets(list(sick.values()))
        else:
            data = list(sick.values())[0]
            
        def convert_sick(ex):
            lbl = str(ex['entailment_label']).strip().upper()
            if lbl == 'ENTAILMENT': label_id = 0
            elif lbl == 'NEUTRAL': label_id = 1
            elif lbl == 'CONTRADICTION': label_id = 2
            else: label_id = 1
            return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}
            
        data = data.map(convert_sick, remove_columns=data.column_names)
        
        # --- SÉLECTION BALANCÉE GLOBALE ---
        vrai_pool = data.filter(lambda x: x['label'] == 0)
        neutre_pool = data.filter(lambda x: x['label'] == 1)
        faux_pool = data.filter(lambda x: x['label'] == 2)
        
        # Le nombre max correspond à la taille de la plus petite classe
        max_per_class = min(len(vrai_pool), len(neutre_pool), len(faux_pool))
        
        test_vrai = vrai_pool.select(range(max_per_class))
        test_neutre = neutre_pool.select(range(max_per_class))
        test_faux = faux_pool.select(range(max_per_class))
        
        from datasets import concatenate_datasets
        balanced_data = concatenate_datasets([test_vrai, test_neutre, test_faux]).shuffle(seed=42)
        
        total = len(balanced_data)
        train_size = int(total * 0.6)
        val_size = int(total * 0.2)
        
        ds = DatasetDict({
            'train': balanced_data.select(range(0, train_size)),
            'validation': balanced_data.select(range(train_size, train_size + val_size)),
            'test': balanced_data.select(range(train_size + val_size, total))
        })
        return ds, "premise"
        
    raise ValueError(f"Dataset {name} inconnu.")

# 2. CONFIGURATION DU SWEEP (RÉDUIT)

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [32]},
        "lora_alpha": {"values": [64]},
        "learning_rate": {"values": [3e-4, 5e-4]},
        "lora_dropout": {"values": [0.05, 0.1]},
    }
}

import sys

if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
    print(f"\nChoix : {exp_choice}")
else:
    print("\nQuelle expérience (Q)LoRA T5Gemma utiliser pour le sweep ?")
    print("1. FraCaS (TOTAL)  →  test GQNLI-FR")
    print("2. GQNLI-FR        →  test FraCaS (TOTAL)")
    print("3. RTE3-DEV        →  test DACCORD + RTE3-TEST")
    print("4. FraCaS (TOTAL)  →  test SICK-FR")
    print("5. SICK-FR         →  test SICK-FR (Intra-dataset)")
    exp_choice = input("\nVotre choix (1 à 5): ").strip()

if exp_choice == "1":
    EXP_NAME = "sweep_fracas_to_gqnli_t5gemma"
    train_ds_name, test_ds_name = "fracas_full", "gqnli_fr"
elif exp_choice == "2":
    EXP_NAME = "sweep_gqnli_to_fracas_t5gemma"
    train_ds_name, test_ds_name = "gqnli_fr", "fracas_full"
elif exp_choice == "3":
    EXP_NAME = "sweep_rte3_to_daccord_t5gemma"
    train_ds_name, test_ds_name = "rte3_fr", "daccord"
elif exp_choice == "4":
    EXP_NAME = "sweep_fracas_to_sick_t5gemma"
    train_ds_name, test_ds_name = "fracas_full", "sick_fr"
elif exp_choice == "5":
    EXP_NAME = "sweep_sick_to_sick_t5gemma"
    train_ds_name, test_ds_name = "sick_fr", "sick_fr"
else:
    print("Choix invalide!"); exit(1)

# On télécharge les datasets cibles
TRAIN_DICT, TRAIN_PKEY = get_dataset(train_ds_name)
TEST_DICT, TEST_PKEY = get_dataset(test_ds_name)

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
    return "neutre"

def preprocess_fn(examples, p_key):
    inputs = [
        f"Consigne : Prédire si l'hypothèse est vraie, fausse ou neutre d'après la prémisse.\nPrémisse : {p}\nHypothèse : {h}\nRéponse :"
        for p, h in zip(examples[p_key], examples["hypothesis"])
    ]
    model_inputs = global_tokenizer(inputs, max_length=256, truncation=True, padding=False)
    targets = [normalize_label(l) for l in examples["label"]]
    labels = global_tokenizer(text_target=targets, max_length=8, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# On tokenise directement
train_data = TRAIN_DICT['train'].map(lambda ex: preprocess_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['train'].column_names)
val_data = TRAIN_DICT['validation'].map(lambda ex: preprocess_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['validation'].column_names)

# Le test se fait uniquement sur le split officiel (pour éviter d'attendre 2h sur 10 000 exemples)
test_untokenized = TEST_DICT['test']
test_data = test_untokenized.map(lambda ex: preprocess_fn(ex, TEST_PKEY), batched=True, remove_columns=test_untokenized.column_names)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    dec_preds = [p.strip() for p in global_tokenizer.batch_decode(preds, skip_special_tokens=True)]
    dec_labels = [l.strip() for l in global_tokenizer.batch_decode(labels, skip_special_tokens=True)]
    
    print(f"\n[DEBUG LORA] Extraits générés : {dec_preds[:5]}")
    print(f"[DEBUG LORA] Extraits cibles  : {dec_labels[:5]}")
    
    LABEL_TO_INT = {"vrai": 0, "neutre": 1, "faux": 2}
    
    import re
    cleaned_preds = []
    for p in dec_preds:
        match = re.search(r'(vrai|faux|neutre)', p.lower())
        if match:
            cleaned_preds.append(match.group(1))
        else:
            cleaned_preds.append("neutre")
    int_preds = [LABEL_TO_INT[p] for p in cleaned_preds]
    int_labels = [LABEL_TO_INT.get(l.lower(), 1) for l in dec_labels]
    
    try:
        cm = confusion_matrix(int_labels, int_preds, labels=[0, 1, 2])
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception: pass
    
    acc = sum(p == l for p, l in zip(cleaned_preds, dec_labels)) / max(1, len(dec_labels))
    return {"accuracy": acc}

# 4. FONCTION D'ENTRAÎNEMENT (POUR WANDB SWEEP)

def train_t5_qlora():
    run = wandb.init()
    config = wandb.config

    run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
    print(f"\n{'='*60}\nSWEEP T5GEMMA RUN: {run_label}\n{'='*60}")

    print("Chargement en QLoRA 4-bit...")
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=50,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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

    trainer.train()

    # Evaluation Finale sur Test Cross-Dataset
    print(f"\n🔍 [WANDB SWEEP RUN] Démarrage EVALUATION FINALE sur {len(test_data)} exemples (SICK-FR Equilibré)...")
    import sys; sys.stdout.flush()
    test_results = trainer.evaluate(test_data)
    print(f"✅ EVALUATION TERMINÉE !")
    test_acc = test_results["eval_accuracy"]
    print(f"FINAL TEST ACCURACY : {test_acc:.2%}")

    wandb.log({"test/cross_dataset_accuracy": test_acc})
    wandb.summary["test_cross_dataset_accuracy"] = test_acc

    # Nettoyage des checkpoints locaux de ce run (Kaggle n'a que 20 Go par défaut)
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
        confirm = input(f"\nLancer {total_runs} sweeps T5Gemma QLoRA ? (o/n): ").strip().lower()
        
    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        print(f"📊 Suivez en direct : https://wandb.ai/votre_profil/fewshot-nli-fr/sweeps/{sweep_id}")
        wandb.agent(sweep_id, function=train_t5_qlora, count=total_runs)
        print("\nTerminé ! Consultez les résultats sur WandB.")
    else:
        print("Annulé.")
