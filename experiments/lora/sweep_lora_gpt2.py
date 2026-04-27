"""
Script de Sweep WandB pour GPT-2 French (Architecture Decoder-only 117M as Classifier).
Même taille de réseau que CamemBERT pour une comparaison parfaitement équitable !
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force 1 seul GPU

import sys
import numpy as np
import torch
import torch.nn as nn
import wandb
import shutil

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Modele de 117M parametres entraine 100% en Francais
BASE_MODEL = "ClassCat/gpt2-base-french"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA.")
    exit(1)

# 1. TÉLÉCHARGEMENT & PRÉPARATION

def get_dataset(name):
    print(f"Téléchargement de {name}...")
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
        
    elif name == "fracas":
        fracas = load_dataset('maximoss/fracas')['train']
        fracas = fracas.filter(lambda x: str(x['label']).strip().lower() != "undef")
        
        train_idx = list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))
        val_idx   = list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))
        test_idx  = list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))
        
        ds = DatasetDict({
            'train': fracas.select(train_idx),
            'validation': fracas.select(val_idx),
            'test': fracas.select(test_idx)
        })
        return ds, "premises"
        
    elif name == "daccord":
        data = load_dataset('maximoss/daccord-contradictions')['train']
        # DACCORD contient des labels (0, 1). 1 = Contradiction. On ramène à 2 pour standardiser multi-classes.
        def fix_daccord_label(ex):
            if ex["label"] == 1:
                ex["label"] = 2
            return ex
        data = data.map(fix_daccord_label)
        data = data.shuffle(seed=42)
        total = len(data)
        train_size, val_size = int(total * 0.6), int(total * 0.2)
        ds = DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data  # 100% du dataset DACCORD — pas de fuite (train sur RTE3)
        })
        return ds, "premise"
        
    elif name == "rte3_fr":
        splits = list(load_dataset('maximoss/rte3-french').values())
        data = concatenate_datasets(splits).shuffle(seed=42)
        total = len(data)
        train_size, val_size = int(total * 0.6), int(total * 0.2)
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
        
        total = len(data)
        train_size = int(total * 0.6)
        val_size = int(total * 0.2)
        
        ds = DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        })
        return ds, "premise"
        
    elif name == "fracas_sick_mix":
        ds_fracas, _ = get_dataset("fracas")
        ds_sick, _ = get_dataset("sick_fr")
        
        LABEL_MAP_LOCAL = {"yes": 0, "entailment": 0, "unknown": 1, "undef": 1, "neutral": 1, "no": 2, "contradiction": 2}
        def align_fracas(ex):
            s = str(ex["label"]).lower().strip()
            l_id = LABEL_MAP_LOCAL.get(s, 1)
            if s not in LABEL_MAP_LOCAL:
                try: l_id = int(s)
                except: pass
            return {"premise": ex["premises"], "hypothesis": ex["hypothesis"], "label": l_id}
            
        ds_fracas = ds_fracas.map(align_fracas, remove_columns=ds_fracas['train'].column_names)
        
        mix_train = concatenate_datasets([ds_fracas['train'], ds_sick['train']]).shuffle(seed=42)
        mix_val = concatenate_datasets([ds_fracas['validation'], ds_sick['validation']]).shuffle(seed=42)
        
        ds = DatasetDict({
            'train': mix_train,
            'validation': mix_val,
            'test': ds_sick['test']
        })
        return ds, "premise"

    raise ValueError(f"Dataset {name} inconnu.")

# Balayage strictement identique a celui de CamemBERT
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r": {"values": [16]},
        "lora_alpha": {"values": [32]},
        "learning_rate": {"values": [3e-4, 5e-4]},
        "lora_dropout": {"values": [0.1]},
    }
}

if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
    print(f"\nVotre choix: {exp_choice} (via argument)")
else:
    print("\nQuelle expérience LoRA GPT-2 utiliser ?")
    print("1. FraCaS (0-74)  →  test GQNLI-FR")
    print("2. GQNLI-FR       →  test FraCaS (0-74)")
    print("3. RTE3-DEV       →  test DACCORD (Binaire)")
    print("4. FraCaS (TOTAL) →  test SICK-FR")
    print("5. SICK-FR        →  test SICK-FR")
    print("6. Mix (FraCaS + SICK-FR) → test SICK-FR")
    exp_choice = input("\nVotre choix (1 à 6): ").strip()

if exp_choice == "1":
    EXP_NAME = "sweep_fracas_to_gqnli_gpt2"
    train_ds_name, test_ds_name = "fracas", "gqnli_fr"
elif exp_choice == "2":
    EXP_NAME = "sweep_gqnli_to_fracas_gpt2"
    train_ds_name, test_ds_name = "gqnli_fr", "fracas"
elif exp_choice == "3":
    EXP_NAME = "sweep_rte3_to_daccord_gpt2"
    train_ds_name, test_ds_name = "rte3_fr", "daccord"
elif exp_choice == "4":
    EXP_NAME = "sweep_fracas_to_sick_gpt2"
    train_ds_name, test_ds_name = "fracas", "sick_fr"
elif exp_choice == "5":
    EXP_NAME = "sweep_sick_to_sick_gpt2"
    train_ds_name, test_ds_name = "sick_fr", "sick_fr"
elif exp_choice == "6":
    EXP_NAME = "sweep_mix_to_sick_gpt2"
    train_ds_name, test_ds_name = "fracas_sick_mix", "sick_fr"
else:
    print("Choix invalide!"); exit(1)

TRAIN_DICT, TRAIN_PKEY = get_dataset(train_ds_name)
TEST_DICT, TEST_PKEY = get_dataset(test_ds_name)

is_binary = (exp_choice == "3")
is_option_6 = (exp_choice == "6")

if is_binary or is_option_6:
    # Retour aux valeurs normales et stables comme CamemBERT
    SWEEP_CONFIG["parameters"]["loss_penalty"] = {"values": [1.0, 3.0, 5.0, 10.0]}
    
if is_binary:
    LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 0, "undef": 0, "neutral": 0, "no": 1, "contradiction": 1}
else:
    LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 1, "undef": 1, "neutral": 1, "no": 2, "contradiction": 2}

global_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Important pour decoder: Fixer le pad token sur l'eos
if global_tokenizer.pad_token is None:
    global_tokenizer.pad_token = global_tokenizer.eos_token
# Pour un Classifier, on met generalement la padding side a la fin (right) sur les Decodeurs
global_tokenizer.padding_side = "right"

def map_label(label):
    if isinstance(label, int):
        if is_binary:
            return 1 if label == 2 else 0
        return label if label in [0, 1, 2] else 1
    s = str(label).lower().strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    return 1 if not is_binary else 0

def tokenize_fn(examples, p_key):
    prompts = [f"Prémisse : {p}\nHypothèse : {h}\n" for p, h in zip(examples[p_key], examples["hypothesis"])]
    res = global_tokenizer(prompts, truncation=True, max_length=256)
    res["labels"] = [map_label(l) for l in examples["label"]]
    return res

print("\nTokenisation des datasets...")
train_data = TRAIN_DICT['train'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = TRAIN_DICT['validation'].map(lambda ex: tokenize_fn(ex, TRAIN_PKEY), batched=True, remove_columns=TRAIN_DICT['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_untokenized = TEST_DICT['test']
test_data = test_untokenized.map(lambda ex: tokenize_fn(ex, TEST_PKEY), batched=True, remove_columns=test_untokenized.column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    try:
        cm_labels = [0, 1] if is_binary else [0, 1, 2]
        cm = confusion_matrix(labels, predictions, labels=cm_labels)
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception: pass
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="macro")
    }
    return metrics

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        penalty = float(getattr(wandb.config, "loss_penalty", 1.0))
        
        if is_binary:
            weight = torch.tensor([1.0, penalty], device=labels.device, dtype=torch.float)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif is_option_6:
            weight = torch.tensor([1.0, max(1.0, penalty/2.0), penalty], device=labels.device, dtype=torch.float)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# 4. FONCTION D'ENTRAÎNEMENT WANDB
def train_gpt2_lora():
    run = wandb.init()
    config = wandb.config

    if is_binary:
        run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}_p{config.loss_penalty}"
    else:
        run_label = f"r{config.lora_r}_a{config.lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}"
        
    print(f"\n{'='*60}\nSWEEP GPT-2 RUN: {run_label}\n{'='*60}")

    print("Chargement du modèle Décodeur pur...")
    num_labels = 2 if is_binary else 3
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        device_map="auto"
        # PAS DE BitsAndBytesConfig (Pas de QLoRA) car le modele fait exactement la meme taille (110M) que Camembert !
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = global_tokenizer.pad_token_id
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["c_attn"], # Dans GPT-2 la couche d'attention principale s'appelle c_attn
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    args = TrainingArguments(
        output_dir=f"/tmp/ckpt_{EXP_NAME}_{run_label}", 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nÉvaluation Cross-Dataset Finale...")
    test_results = trainer.evaluate(test_data, metric_key_prefix="test")
    test_acc = test_results["test_accuracy"]
    test_f1 = test_results["test_f1_score"]
    
    print(f"FINAL TEST ACCURACY : {test_acc:.2%}")
    print(f"FINAL TEST F1-SCORE : {test_f1:.4f}")

    # Le Trainer avec son WandbCallback vient d'envoyer automatiquement les resultats "test_accuracy" a WandB !
    wandb.summary["final_test_accuracy"] = test_acc
    wandb.summary["final_test_f1_score"] = test_f1

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    total_runs = 1
    for param_name, param in SWEEP_CONFIG["parameters"].items():
        if "values" in param:
            total_runs *= len(param["values"])
    
    if len(sys.argv) > 1:
        confirm = "o" 
    else:
        confirm = input(f"\nLancer {total_runs} sweeps GPT-2 LoRA ? (o/n): ").strip().lower()
        
    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_gpt2_lora, count=total_runs)
        print("\nTerminé ! Consultez les résultats sur WandB.")
    else:
        print("Annulé.")
