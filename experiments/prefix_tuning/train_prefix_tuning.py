"""
Script de Prefix-Tuning pour LLM Causaux (GPT-2, Mistral, LLaMA).
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
    DataCollatorForLanguageModeling,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, PrefixTuningConfig, TaskType
from sklearn.metrics import confusion_matrix

def get_dataset(name):
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

def normalize_label(label):
    LABEL_MAP = {
        "yes": "vrai", "entailment": "vrai", 0: "vrai", "0": "vrai",
        "unknown": "neutre", "neutral": "neutre", 1: "neutre", "1": "neutre",
        "no": "faux", "contradiction": "faux", 2: "faux", "2": "faux"
    }
    s = str(label).lower().strip()
    return LABEL_MAP.get(s, "neutre")

def preprocess_causal(examples, p_key, tokenizer):
    inputs = []
    for p, h, l in zip(examples[p_key], examples["hypothesis"], examples["label"]):
        prompt = f"Consigne : Prédire si l'hypothèse est vraie, fausse ou neutre d'après la prémisse.\nPrémisse : {p}\nHypothèse : {h}\nRéponse : "
        label_text = normalize_label(l)
        inputs.append(prompt + label_text + tokenizer.eos_token)
        
    # On tokenise directement tout le texte (prompt + réponse + eos)
    # L'objectif Causal LM va tenter de prédire chaque token suivant.
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=False)
    # On ajoute la cible en texte brut pour l'évaluer plus tard avec la Regex
    model_inputs["target_label"] = [normalize_label(l) for l in examples["label"]]
    model_inputs["prompt_only"] = [
        f"Consigne : Prédire si l'hypothèse est vraie, fausse ou neutre d'après la prémisse.\nPrémisse : {p}\nHypothèse : {h}\nRéponse : "
        for p, h in zip(examples[p_key], examples["hypothesis"])
    ]
    return model_inputs

class GenerationAccuracyCallback(TrainerCallback):
    """
    Callback pour évaluer un modèle Causal LM par génération à la fin de chaque Epoch.
    Utilise Expressions Régulières (Regex) pour éviter les hallucinations comme abordé dans le rapport M1.
    """
    def __init__(self, val_dataset, tokenizer):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print("\n[🎯 EVALUATION GENERATIVE PAR REGEX]")
        model.eval()
        correct, total = 0, len(self.val_dataset)
        
        # On va tester sur un echantillon si le dataset est trop gros pour gagner du temps
        sample_size = min(total, 50)
        sample = self.val_dataset.select(range(sample_size))
        
        LABEL_TO_INT = {"vrai": 0, "neutre": 1, "faux": 2}
        all_preds, all_labels = [], []
        
        for i in range(sample_size):
            ex = sample[i]
            prompt = ex["prompt_only"]
            true_label = ex["target_label"]
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=5, # Seulement 5 tokens max pour couper un éventuel bégaiment ("fauxfauxfaux")
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
            decoded = self.tokenizer.decode(out_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # Application de la REGEX du "Couteau Suisse" (Rapport M1)
            match = re.search(r'(vrai|faux|neutre)', decoded.lower())
            pred = match.group(1) if match else "neutre"
            
            all_preds.append(pred)
            all_labels.append(true_label)
            
            if pred == true_label: correct += 1
            if i < 2:
                print(f"  → Généré brut : '{decoded}' | Regex a gardé : '{pred}' | Vrai : '{true_label}'")
                
        metrics = {"eval/accuracy": correct / sample_size}
        wandb.log(metrics)
        print(f"✅ Accuracy Générative (Epoch {state.epoch}): {metrics['eval/accuracy']:.2%}")
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2, mistral, ou llama")
    parser.add_argument("--train_ds", type=str, default="fracas_75")
    parser.add_argument("--test_ds", type=str, default="gqnli_fr")
    args = parser.parse_args()

    # Modèles
    if args.model == "gpt2":
        model_id = "gpt2"
        use_4bit = False
    elif args.model == "meta-llama/Llama-2-7b-hf" or args.model == "llama":
        model_id = "meta-llama/Llama-2-7b-hf"
        use_4bit = True
    else:
        model_id = "mistralai/Mistral-7B-v0.1"
        use_4bit = True

    print(f"🚀 Lancement Prefix-Tuning sur {model_id} (4-bit: {use_4bit})")

    # WANDB INIT
    run = wandb.init(project="fewshot-nli-fr")
    config = wandb.config
    
    # Paramètres dynamiques du Sweep (ou par défaut)
    lr = config.learning_rate if "learning_rate" in config else 5e-4
    v_tokens = config.virtual_tokens if "virtual_tokens" in config else 30
    run.name = f"prefix_{args.model}_v{v_tokens}_lr{lr}"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding mieux pour la génération Causal LM

    # Préparation Données
    train_dict, t_key = get_dataset(args.train_ds)
    train_ds = train_dict["train"].map(lambda ex: preprocess_causal(ex, t_key, tokenizer), batched=True, remove_columns=train_dict["train"].column_names)
    val_ds = train_dict["validation"].map(lambda ex: preprocess_causal(ex, t_key, tokenizer), batched=True, remove_columns=train_dict["validation"].column_names)

    # Chargement Modèle (4-bit pour Mistral/Llama sur Kaggle)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if use_4bit else None
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    # CONFIGURATION PREFIX-TUNING
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=v_tokens
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"/tmp/prefix_{args.model}",
        learning_rate=lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=15,
        save_strategy="epoch",
        logging_steps=5,
        report_to="wandb",
        remove_unused_columns=True # Fix du crash: le Trainer ignorera "prompt_only" et "target_label" pour la collate_fn, val_ds les conserve quand même pour le Callback !
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[GenerationAccuracyCallback(val_ds, tokenizer)]
    )

    print("\n🎬 Début de l'entraînement Prefix-Tuning...")
    trainer.train()
    
    if os.path.exists(training_args.output_dir):
        shutil.rmtree(training_args.output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
