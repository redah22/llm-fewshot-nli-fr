"""
Script de Prefix-Tuning OPTIMISÉ pour la soutenance.
Teste 4 configurations pour trouver celle qui évite le collapse :
  - Config A : Sans projection, 10 tokens virtuels (~600K params)
  - Config B : Sans projection, 30 tokens virtuels (~2M params)  
  - Config C : Avec projection, 10 tokens virtuels (~23M params)
  - Config D : Sans projection, 10 tokens + Loss Penalty sur neutre (comme Colin/LoRA)

Chaque config est entraînée 5 epochs max.
Les résultats sont sauvegardés dans /results.
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

from datasets import load_dataset, DatasetDict
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
# 1. CHARGEMENT DES DONNÉES
# ============================================================

def get_dataset(name):
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

    raise ValueError(f"Dataset {name} inconnu.")

# ============================================================
# 2. NORMALISATION DES LABELS
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
# 3. PRÉPARATION CAUSAL LM
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
# 4. CALLBACK D'ÉVALUATION
# ============================================================

class ProbabilityEvalCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.label_tokens = {
            "vrai": tokenizer.encode("vrai", add_special_tokens=False)[0],
            "faux": tokenizer.encode("faux", add_special_tokens=False)[0],
            "neutre": tokenizer.encode("neutre", add_special_tokens=False)[0],
        }
        self.best_accuracy = 0.0
        self.all_results = []  # Historique par epoch
        print(f"📌 Token IDs des labels : {self.label_tokens}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print("\n[🎯 EVALUATION PAR PROBABILITÉS]")
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
        self.all_results.append({
            "epoch": state.epoch,
            "accuracy": acc,
            "confusion_matrix": cm.tolist()
        })
        
        if acc > self.best_accuracy:
            self.best_accuracy = acc
        
        wandb.log({"eval/accuracy": acc})
        print(f"✅ Accuracy Validation (Epoch {state.epoch}): {acc:.2%}  (Best: {self.best_accuracy:.2%})")
        model.train()

# ============================================================
# 5. MAIN
# ============================================================

CONFIGS = {
    "A": {"num_virtual_tokens": 10, "prefix_projection": False, "lr": 5e-5, "epochs": 5, "loss_penalty": 1.0, "desc": "Sans projection, 10 tokens, lr=5e-5"},
    "B": {"num_virtual_tokens": 30, "prefix_projection": False, "lr": 5e-5, "epochs": 5, "loss_penalty": 1.0, "desc": "Sans projection, 30 tokens, lr=5e-5"},
    "C": {"num_virtual_tokens": 10, "prefix_projection": True,  "lr": 5e-5, "epochs": 5, "loss_penalty": 1.0, "desc": "Avec projection, 10 tokens, lr=5e-5"},
    "D": {"num_virtual_tokens": 10, "prefix_projection": False, "lr": 5e-5, "epochs": 5, "loss_penalty": 5.0, "desc": "Sans projection, 10 tokens + Loss Penalty neutre=5.0"},
}

# ============================================================
# 5b. WEIGHTED TRAINER (méthode de Colin adaptée pour CausalLM)
# ============================================================

class WeightedCausalTrainer(Trainer):
    """
    Trainer personnalisé qui applique une pénalité (loss_penalty) sur le token 'neutre'.
    Technique identique à celle de Colin (WeightedTrainer dans sweep_lora_gpt2.py),
    adaptée pour un modèle CausalLM (génératif) au lieu de SequenceClassification.
    """
    def __init__(self, *args, neutre_token_id=None, loss_penalty=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.neutre_token_id = neutre_token_id
        self.loss_penalty = loss_penalty
        print(f"⚖️  Loss Penalty activée : poids={loss_penalty} sur token neutre (ID={neutre_token_id})")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Décaler les logits et labels comme pour un CausalLM standard
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        vocab_size = shift_logits.size(-1)
        
        if self.loss_penalty > 1.0 and self.neutre_token_id is not None:
            # Créer un vecteur de poids pour TOUT le vocabulaire
            # Poids = 1.0 pour tous les tokens, sauf neutre qui a un poids plus élevé
            weight = torch.ones(vocab_size, device=shift_logits.device, dtype=shift_logits.dtype)
            weight[self.neutre_token_id] = self.loss_penalty
            loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def run_config(config_name, config, model_id, use_4bit, train_ds, val_ds, tokenizer, dataset_name):
    print(f"\n{'='*70}")
    print(f"  🔬 CONFIG {config_name} : {config['desc']}")
    print(f"{'='*70}")
    
    run = wandb.init(project="fewshot-nli-fr", reinit=True)
    run.name = f"prefix_opt_{config_name}_{dataset_name}_v{config['num_virtual_tokens']}"
    
    # Charger le modèle frais à chaque config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    ) if use_4bit else None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config["num_virtual_tokens"],
        prefix_projection=config["prefix_projection"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    eval_callback = ProbabilityEvalCallback(val_ds, tokenizer)
    
    # Récupérer l'ID du token neutre pour la pénalité
    neutre_token_id = tokenizer.encode("neutre", add_special_tokens=False)[0]
    penalty = config.get("loss_penalty", 1.0)

    training_args = TrainingArguments(
        output_dir=f"/tmp/prefix_opt_{config_name}",
        learning_rate=config["lr"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=5,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # Utiliser le WeightedCausalTrainer (comme le WeightedTrainer de Colin)
    trainer = WeightedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[eval_callback],
        neutre_token_id=neutre_token_id,
        loss_penalty=penalty,
    )

    trainer.train()
    
    # Résultat
    result = {
        "config_name": config_name,
        "config": config,
        "best_accuracy": eval_callback.best_accuracy,
        "final_accuracy": eval_callback.last_accuracy,
        "final_confusion_matrix": eval_callback.last_cm,
        "all_epochs": eval_callback.all_results,
        "labels_order": ["vrai", "neutre", "faux"]
    }
    
    # Nettoyage
    if os.path.exists(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)
    del model, base_model, trainer
    torch.cuda.empty_cache()
    
    wandb.finish()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral", help="gpt2, mistral, ou llama")
    parser.add_argument("--dataset", type=str, default="gqnli_fr", help="gqnli_fr, fracas_75")
    parser.add_argument("--config", type=str, default="all", help="A, B, C, D, ou all")
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

    print(f"🚀 Prefix-Tuning OPTIMISÉ sur {model_id}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Données
    ds_dict, p_key = get_dataset(args.dataset)
    train_ds = ds_dict["train"].map(
        lambda ex: preprocess_causal(ex, p_key, tokenizer), 
        batched=True, remove_columns=ds_dict["train"].column_names
    )
    val_ds = ds_dict["validation"].map(
        lambda ex: preprocess_causal(ex, p_key, tokenizer), 
        batched=True, remove_columns=ds_dict["validation"].column_names
    )
    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Configs à tester
    configs_to_run = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    
    all_results = {}
    for name, cfg in configs_to_run.items():
        result = run_config(name, cfg, model_id, use_4bit, train_ds, val_ds, tokenizer, args.dataset)
        all_results[name] = result
        print(f"\n📈 Config {name} terminée : Best accuracy = {result['best_accuracy']:.2%}")

    # Sauvegarde globale
    os.makedirs("results", exist_ok=True)
    res_path = f"results/prefix_optimized_{args.model}_{args.dataset}.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ TOUS LES RÉSULTATS SAUVEGARDÉS DANS : {res_path}")
    
    # Résumé final
    print(f"\n{'='*70}")
    print("  📊 RÉSUMÉ COMPARATIF")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"  Config {name} ({res['config']['desc']})")
        print(f"    → Best accuracy: {res['best_accuracy']:.2%}")
        print(f"    → Final accuracy: {res['final_accuracy']:.2%}")
        cm = res['final_confusion_matrix']
        print(f"    → Matrice finale: vrai→{cm[0]}, neutre→{cm[1]}, faux→{cm[2]}")
        print()


if __name__ == "__main__":
    main()
