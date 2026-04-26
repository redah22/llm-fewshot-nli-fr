"""
Script de Prefix-Tuning CROSS-DATASET pour LLM Causaux (GPT-2, Mistral, LLaMA).
Entraîne sur un dataset (train_ds) et évalue sur un autre (test_ds).
"""
import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import torch
import wandb
import re
import numpy as np
import shutil
import argparse

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, PrefixTuningConfig, TaskType

# ============================================================
# 1. CHARGEMENT DES DONNÉES (identique à sweep_lora.py)
# ============================================================

def get_dataset(name):
    """Télécharge et segmente dynamiquement le jeu de données depuis Hub."""
    print(f"Téléchargement et structuration de {name}...")
    if name == "gqnli_fr":
        gqnli = load_dataset('maximoss/gqnli-fr')['test']
        # Split STRATIFIÉ : on regroupe par label d'abord pour équilibrer
        by_label = {0: [], 1: [], 2: []}
        for i, ex in enumerate(gqnli):
            by_label[int(ex['label'])].append(i)
        train_idx, val_idx, test_idx = [], [], []
        for label, indices in by_label.items():
            n = len(indices)
            t1 = int(n * 0.6)
            t2 = int(n * 0.8)
            train_idx.extend(indices[:t1])
            val_idx.extend(indices[t1:t2])
            test_idx.extend(indices[t2:])
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
        train_size, val_size = int(total * 0.6), int(total * 0.2)
        return DatasetDict({
            'train': data.select(range(0, train_size)),
            'validation': data.select(range(train_size, train_size + val_size)),
            'test': data.select(range(train_size + val_size, total))
        }), "premise"

    raise ValueError(f"Dataset {name} inconnu.")

# ============================================================
# 2. NORMALISATION DES LABELS (identique à sweep_lora_t5gemma.py)
# ============================================================

def normalize_label(label):
    LABEL_MAP = {
        "yes": "vrai", "entailment": "vrai", 0: "vrai", "0": "vrai",
        "unknown": "neutre", "undef": "neutre", "neutral": "neutre", 1: "neutre", "1": "neutre",
        "no": "faux", "contradiction": "faux", 2: "faux", "2": "faux"
    }
    s = str(label).lower().strip()
    return LABEL_MAP.get(s, "neutre")

# ============================================================
# 3. PRÉPARATION CAUSAL LM (prompt masqué dans les labels)
# ============================================================

def preprocess_causal(examples, p_key, tokenizer):
    all_input_ids = []
    all_labels = []
    target_labels = []
    prompt_only_texts = []

    for p, h, l in zip(examples[p_key], examples["hypothesis"], examples["label"]):
        prompt = f"Consigne : Prédire si l'hypothèse est vraie, fausse ou neutre d'après la prémisse.\nPrémisse : {p}\nHypothèse : {h}\nRéponse : "
        label_text = normalize_label(l)
        full_text = prompt + label_text + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, max_length=256, truncation=True, add_special_tokens=False)["input_ids"]

        # MASQUAGE : -100 sur le prompt, vrais tokens seulement sur le label
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        target_labels.append(label_text)
        prompt_only_texts.append(prompt)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "target_label": target_labels,
        "prompt_only": prompt_only_texts,
    }

# ============================================================
# 4. CALLBACK D'ÉVALUATION PAR PROBABILITÉS
# ============================================================

class ProbabilityEvalCallback(TrainerCallback):
    """
    Évalue par comparaison de logits sur les 3 tokens (vrai/faux/neutre).
    """
    def __init__(self, val_dataset, tokenizer):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.label_tokens = {
            "vrai": tokenizer.encode("vrai", add_special_tokens=False)[0],
            "faux": tokenizer.encode("faux", add_special_tokens=False)[0],
            "neutre": tokenizer.encode("neutre", add_special_tokens=False)[0],
        }
        print(f"📌 Token IDs des labels : {self.label_tokens}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print("\n[🎯 EVALUATION PAR PROBABILITÉS (CROSS DATASET)]")
        model.eval()

        sample_size = min(len(self.val_dataset), 50)
        sample = self.val_dataset.select(range(sample_size))

        correct = 0
        y_true = []
        y_pred = []
        
        for i in range(sample_size):
            ex = sample[i]
            prompt = ex["prompt_only"]
            true_label = ex["target_label"]
            y_true.append(true_label)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)

            next_token_logits = outputs.logits[0, -1, :]
            scores = {label: next_token_logits[tid].item() for label, tid in self.label_tokens.items()}
            pred = max(scores, key=scores.get)
            y_pred.append(pred)

            if pred == true_label:
                correct += 1
            if i < 3:
                print(f"  → Scores: vrai={scores['vrai']:.2f} faux={scores['faux']:.2f} neutre={scores['neutre']:.2f} | Prédit: '{pred}' | Vrai: '{true_label}'")

        acc = correct / sample_size
        
        # --- MATRICE DE CONFUSION ---
        from sklearn.metrics import confusion_matrix
        labels_order = ["vrai", "neutre", "faux"]
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        print("\n📊 MATRICE DE CONFUSION:")
        print(f"             Préd:vrai   Préd:neutre   Préd:faux")
        print(f"Vrai vrai:   {cm[0][0]:<11} {cm[0][1]:<13} {cm[0][2]}")
        print(f"Vrai neutre: {cm[1][0]:<11} {cm[1][1]:<13} {cm[1][2]}")
        print(f"Vrai faux:   {cm[2][0]:<11} {cm[2][1]:<13} {cm[2][2]}\n")
        
        self.last_accuracy = acc
        self.last_cm = cm.tolist()
        
        wandb.log({"eval/accuracy": acc})
        print(f"✅ Accuracy Validation (Epoch {state.epoch}): {acc:.2%}")
        model.train()

# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2, mistral, ou llama")
    parser.add_argument("--train_ds", type=str, default="gqnli_fr", help="Dataset d'entraînement (gqnli_fr, fracas_75, daccord)")
    parser.add_argument("--test_ds", type=str, default="fracas_75", help="Dataset d'évaluation (gqnli_fr, fracas_75, daccord)")
    args = parser.parse_args()

    # Modèles
    if args.model == "gpt2":
        model_id = "gpt2"
        use_4bit = False
    elif args.model == "llama":
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        use_4bit = True
    else:
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        use_4bit = True

    print(f"🚀 Lancement Prefix-Tuning sur {model_id} (4-bit: {use_4bit})")
    print(f"🔄 CROSS DATASET : Train={args.train_ds} -> Test={args.test_ds}")

    # WANDB
    run = wandb.init(project="fewshot-nli-fr")
    config = wandb.config
    lr = config.learning_rate if "learning_rate" in config else 5e-5
    v_tokens = config.virtual_tokens if "virtual_tokens" in config else 10
    run.name = f"prefix_{args.model}_{args.train_ds}2{args.test_ds}_v{v_tokens}_lr{lr}"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Données CROSS DATASET
    train_dict, train_pkey = get_dataset(args.train_ds)
    test_dict, test_pkey = get_dataset(args.test_ds)
    
    train_ds = train_dict["train"].map(
        lambda ex: preprocess_causal(ex, train_pkey, tokenizer), 
        batched=True, remove_columns=train_dict["train"].column_names
    )
    
    # Évaluation sur le dataset de test
    val_split = "validation" if "validation" in test_dict else "test"
    val_ds = test_dict[val_split].map(
        lambda ex: preprocess_causal(ex, test_pkey, tokenizer), 
        batched=True, remove_columns=test_dict[val_split].column_names
    )

    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Modèle
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    ) if use_4bit else None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=v_tokens,
        prefix_projection=False,   # Sans projection = Config A (la meilleure, évite le collapse)
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Entraînement
    training_args = TrainingArguments(
        output_dir=f"/tmp/prefix_{args.model}",
        learning_rate=lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,         # Régularisation
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        report_to="wandb",
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[ProbabilityEvalCallback(val_ds, tokenizer)]
    )

    print("\n🎬 Début de l'entraînement Prefix-Tuning CROSS DATASET...")
    trainer.train()
    
    # SAUVEGARDE DES RÉSULTATS DANS /results
    import json
    os.makedirs("results", exist_ok=True)
    res_path = f"results/prefix_{args.model}_{args.train_ds}to{args.test_ds}_v{v_tokens}.json"
    
    eval_callback = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, ProbabilityEvalCallback)][0]
    results_data = {
        "model": args.model,
        "train_dataset": args.train_ds,
        "test_dataset": args.test_ds,
        "virtual_tokens": v_tokens,
        "final_accuracy": getattr(eval_callback, 'last_accuracy', 0.0),
        "confusion_matrix": getattr(eval_callback, 'last_cm', []),
        "labels_order": ["vrai", "neutre", "faux"]
    }
    
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ RÉSULTATS OFFICIELS SAUVEGARDÉS DANS : {res_path}")
    
    # Nettoyage
    if os.path.exists(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
