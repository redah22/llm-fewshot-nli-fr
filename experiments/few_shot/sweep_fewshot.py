"""
Sweep Few-Shot NLI — LoRA & IA³
================================

Ce script explore l'impact du NOMBRE D'EXEMPLES D'ENTRAÎNEMENT
sur la performance few-shot des modèles NLI français.

Contrairement aux scripts de Colin (qui utilisent tout le dataset),
ici on tire au sort N exemples STRATIFIÉS (même proportion de classes),
sans jamais les inclure dans le test set.

Axes du sweep WandB :
  - n_shots    : nombre d'exemples few-shot (8, 16, 32, 64, 128)
  - peft_method: "lora" ou "ia3"
  - model      : camembert-base, flaubert, flaubert+xnli
  - seed_data  : graine pour tirer les exemples (42, 123, 999)

Modes d'expérience :
  - intra  : train et test sur le même dataset (split stratifié)
  - cross  : train sur dataset A, test intégral sur dataset B

Utilisation :
    python3 experiments/few_shot/sweep_fewshot.py
    python3 experiments/few_shot/sweep_fewshot.py 1   # EXP 1 via argument
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import sys
import random
import numpy as np
import torch
import wandb
import shutil
from collections import defaultdict

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, IA3Config, TaskType
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION DES MODÈLES DISPONIBLES
# ─────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "camembert": {
        "name": "camembert-base",
        "short": "camembert",
        "lora_targets": ["query", "value"],
        "ia3_targets":  ["key", "value", "query", "output.dense", "intermediate.dense"],
        "ia3_ff":       ["intermediate.dense"],
        "pad_side": "right",
    },
    "flaubert": {
        "name": "flaubert/flaubert_base_cased",
        "short": "flaubert",
        "lora_targets": ["query", "value"],
        "ia3_targets":  ["k_lin", "v_lin", "out_lin", "lin1"],
        "ia3_ff":       ["lin1"],
        "pad_side": "right",
    },
    "flaubert_xnli": {
        "name": "models/flaubert_xnli_fr_full",
        "short": "flaubert_xnli",
        "lora_targets": ["query", "value"],
        "ia3_targets":  ["k_lin", "v_lin", "out_lin", "lin1"],
        "ia3_ff":       ["lin1"],
        "pad_side": "right",
    },
}

# ─────────────────────────────────────────────────────────
# 2. CHARGEMENT DES DATASETS
# ─────────────────────────────────────────────────────────

LABEL_MAP = {
    "yes": 0, "entailment": 0,
    "unknown": 1, "undef": 1, "neutral": 1,
    "no": 2, "contradiction": 2,
}

def normalize_label(label):
    if isinstance(label, int):
        return label if label in [0, 1, 2] else 1
    s = str(label).lower().strip()
    if s in ['0', '1', '2']:
        return int(s)
    return LABEL_MAP.get(s, 1)

def get_dataset(name):
    """Télécharge et unifie un dataset depuis HuggingFace Hub."""
    print(f"  Chargement de {name}...")
    if name == "gqnli_fr":
        data = load_dataset("maximoss/gqnli-fr")["test"]
        # Découpage identique à celui de Colin pour cohérence
        train_idx = list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))
        val_idx   = list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))
        test_idx  = list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))
        ds = DatasetDict({
            "train": data.select(train_idx).shuffle(seed=42),
            "validation": data.select(val_idx).shuffle(seed=42),
            "test": data.select(test_idx).shuffle(seed=42),
        })
        return ds, "premise"

    elif name == "fracas_full":
        data = load_dataset("maximoss/fracas")["train"]
        data = data.filter(lambda x: str(x["label"]).strip().lower() != "undef")
        shuffled = data.shuffle(seed=42)
        n = len(shuffled)
        n_train, n_val = int(n * 0.6), int(n * 0.2)
        ds = DatasetDict({
            "train": shuffled.select(range(0, n_train)),
            "validation": shuffled.select(range(n_train, n_train + n_val)),
            "test": shuffled.select(range(n_train + n_val, n)),
        })
        return ds, "premises"

    elif name == "sick_fr":
        raw = load_dataset("maximoss/sick-fr")
        data = concatenate_datasets(list(raw.values())) if len(raw) > 1 else list(raw.values())[0]
        def convert_sick(ex):
            lbl = str(ex["entailment_label"]).strip().upper()
            lid = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}.get(lbl, 1)
            return {"premise": ex["sentence_A"], "hypothesis": ex["sentence_B"], "label": lid}
        data = data.map(convert_sick, remove_columns=data.column_names)
        n = len(data)
        n_train, n_val = int(n * 0.6), int(n * 0.2)
        ds = DatasetDict({
            "train": data.select(range(0, n_train)),
            "validation": data.select(range(n_train, n_train + n_val)),
            "test": data.select(range(n_train + n_val, n)),
        })
        return ds, "premise"

    elif name == "daccord":
        data = load_dataset("maximoss/daccord-contradictions")["train"].shuffle(seed=42)
        def fix_daccord(ex):
            if ex["label"] == 1: ex["label"] = 2
            return ex
        data = data.map(fix_daccord)
        n = len(data)
        n_train, n_val = int(n * 0.6), int(n * 0.2)
        ds = DatasetDict({
            "train": data.select(range(0, n_train)),
            "validation": data.select(range(n_train, n_train + n_val)),
            "test": data.select(range(n_train + n_val, n)),
        })
        return ds, "premise"

    elif name == "rte3_fr":
        splits = list(load_dataset("maximoss/rte3-french").values())
        data = concatenate_datasets(splits).shuffle(seed=42)
        n = len(data)
        n_train, n_val = int(n * 0.6), int(n * 0.2)
        ds = DatasetDict({
            "train": data.select(range(0, n_train)),
            "validation": data.select(range(n_train, n_train + n_val)),
            "test": data.select(range(n_train + n_val, n)),
        })
        return ds, "premise"

    raise ValueError(f"Dataset inconnu : {name}")


# ─────────────────────────────────────────────────────────
# 3. ÉCHANTILLONNAGE STRATIFIÉ FEW-SHOT
# ─────────────────────────────────────────────────────────

def sample_few_shot(dataset, n_shots, seed, premise_key, excluded_indices=None):
    """
    Tire n_shots exemples STRATIFIÉS depuis le dataset.
    Les exemples sélectionnés ne se retrouveront jamais dans le test set.

    Returns:
        few_shot_dataset : Dataset avec n_shots exemples
        few_shot_indices : Les indices utilisés (pour exclusion)
    """
    rng = random.Random(seed)
    excluded = set(excluded_indices or [])

    # Grouper par classe (label normalisé)
    class_to_indices = defaultdict(list)
    for i, ex in enumerate(dataset):
        if i in excluded:
            continue
        lbl = normalize_label(ex["label"])
        class_to_indices[lbl].append(i)

    num_classes = len(class_to_indices)
    if num_classes == 0:
        raise ValueError("Aucune classe disponible dans le dataset.")

    n_per_class = n_shots // num_classes
    remainder   = n_shots % num_classes

    selected_indices = []
    for cls in sorted(class_to_indices.keys()):
        pool = class_to_indices[cls]
        rng.shuffle(pool)
        count = n_per_class + (1 if remainder > 0 else 0)
        remainder -= 1
        # Si pas assez d'exemples, on prend tout ce qu'on a
        count = min(count, len(pool))
        selected_indices.extend(pool[:count])

    rng.shuffle(selected_indices)
    shot_ds = dataset.select(selected_indices)

    print(f"  ✅ {len(shot_ds)} exemples few-shot tirés (stratifié, seed={seed})")
    label_counts = defaultdict(int)
    for ex in shot_ds:
        label_counts[normalize_label(ex["label"])] += 1
    print(f"     Distribution : {dict(sorted(label_counts.items()))}")

    return shot_ds, selected_indices


# ─────────────────────────────────────────────────────────
# 4. CHOIX DE L'EXPÉRIENCE
# ─────────────────────────────────────────────────────────

EXPERIMENTS = {
    "1": {
        "name": "intra_gqnli",
        "mode": "intra",
        "train_ds": "gqnli_fr",
        "test_ds": "gqnli_fr",
        "label": "Few-Shot Intra — GQNLI",
    },
    "2": {
        "name": "intra_sick",
        "mode": "intra",
        "train_ds": "sick_fr",
        "test_ds": "sick_fr",
        "label": "Few-Shot Intra — SICK-FR",
    },
    "3": {
        "name": "cross_fracas_to_gqnli",
        "mode": "cross",
        "train_ds": "fracas_full",
        "test_ds": "gqnli_fr",
        "label": "Few-Shot Cross — FraCaS → GQNLI",
    },
    "4": {
        "name": "cross_gqnli_to_sick",
        "mode": "cross",
        "train_ds": "gqnli_fr",
        "test_ds": "sick_fr",
        "label": "Few-Shot Cross — GQNLI → SICK-FR",
    },
    "5": {
        "name": "cross_fracas_to_sick",
        "mode": "cross",
        "train_ds": "fracas_full",
        "test_ds": "sick_fr",
        "label": "Few-Shot Cross — FraCaS → SICK-FR",
    },
}

if len(sys.argv) > 1:
    exp_choice = sys.argv[1].strip()
else:
    print("\n" + "=" * 60)
    print("SWEEP FEW-SHOT NLI — LoRA & IA³")
    print("=" * 60)
    print("\nQuelle expérience ?")
    for k, v in EXPERIMENTS.items():
        print(f"  {k}. {v['label']}")
    exp_choice = input("\nVotre choix : ").strip()

if exp_choice not in EXPERIMENTS:
    print("❌ Choix invalide."); exit(1)

EXP = EXPERIMENTS[exp_choice]
print(f"\n🔬 Expérience : {EXP['label']}")

# ─────────────────────────────────────────────────────────
# 5. CONFIGURATION DU SWEEP WANDB
# ─────────────────────────────────────────────────────────

# Pour tester les 3 modèles, mettez les 3 dans la liste.
# Pour commencer rapidement, gardez juste "camembert".
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "test/f1_score", "goal": "maximize"},
    "parameters": {
        "n_shots":     {"values": [8, 16, 32, 64, 128]},
        "peft_method": {"values": ["lora", "ia3"]},
        "model_key":   {"values": ["camembert", "flaubert", "flaubert_xnli"]},
        "seed_data":   {"values": [42, 123, 999]},
        # Hyperparamètres PEFT fixes (meilleurs d'après les sweeps de Colin)
        "lora_r":      {"values": [16]},
        "lora_alpha":  {"values": [32]},
        "lora_dropout":{"values": [0.1]},
        "learning_rate":{"values": [3e-4]},
    }
}

# ─────────────────────────────────────────────────────────
# 6. CHARGEMENT INITIAL DES DATASETS (une seule fois)
# ─────────────────────────────────────────────────────────

print("\n📂 Chargement des datasets...")
TRAIN_DS, TRAIN_PKEY = get_dataset(EXP["train_ds"])
if EXP["mode"] == "cross" and EXP["test_ds"] != EXP["train_ds"]:
    TEST_DS, TEST_PKEY = get_dataset(EXP["test_ds"])
else:
    TEST_DS, TEST_PKEY = TRAIN_DS, TRAIN_PKEY

print(f"  Train pool : {len(TRAIN_DS['train'])} exemples")
print(f"  Test set   : {len(TEST_DS['test'])} exemples")


# ─────────────────────────────────────────────────────────
# 7. TOKENISATION
# ─────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer, premise_key):
    """Retourne une fonction de tokenisation compatible avec .map()"""
    def tokenize(examples):
        result = tokenizer(
            examples[premise_key],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        result["labels"] = [normalize_label(l) for l in examples["label"]]
        return result
    return tokenize


# ─────────────────────────────────────────────────────────
# 8. CONSTRUCTION DU MODÈLE PEFT
# ─────────────────────────────────────────────────────────

def make_peft_model(model_cfg, peft_method, lora_r, lora_alpha, lora_dropout):
    """Charge le modèle base et applique LoRA ou IA³."""
    base = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"], num_labels=3
    )

    if peft_method == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=model_cfg["lora_targets"],
            modules_to_save=["classifier"],
            bias="none",
        )
    elif peft_method == "ia3":
        config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=model_cfg["ia3_targets"],
            feedforward_modules=model_cfg["ia3_ff"],
            modules_to_save=["classifier"],
        )
    else:
        raise ValueError(f"peft_method inconnu : {peft_method}")

    model = get_peft_model(base, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  📐 {peft_method.upper()} — Params entraînables : {trainable:,} ({100*trainable/total:.3f}%)")

    return model


# ─────────────────────────────────────────────────────────
# 9. MÉTRIQUES
# ─────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        print(f"\n  Matrice de confusion:\n{cm}")
    except Exception:
        pass
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="macro"),
    }


# ─────────────────────────────────────────────────────────
# 10. FONCTION D'ENTRAÎNEMENT (APPELÉE PAR WANDB AGENT)
# ─────────────────────────────────────────────────────────

def train_one_run():
    run = wandb.init()
    cfg = wandb.config

    model_cfg = MODEL_REGISTRY[cfg.model_key]
    run_label = (
        f"{model_cfg['short']}-{cfg.peft_method}"
        f"-{cfg.n_shots}shots-seed{cfg.seed_data}"
    )
    print(f"\n{'='*60}")
    print(f"  RUN: {run_label}")
    print(f"{'='*60}")

    # — Tokenizer —
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_cfg.get("pad_side", "right")

    tokenize_train = make_tokenize_fn(tokenizer, TRAIN_PKEY)
    tokenize_test  = make_tokenize_fn(tokenizer, TEST_PKEY)

    # — Échantillonnage Few-Shot Stratifié —
    # En mode INTRA, les indices few-shot doivent être exclus du test set.
    # On tire depuis TRAIN_DS['train'] pour ne pas toucher au test split.
    print(f"\n  Tirage de {cfg.n_shots} exemples (seed={cfg.seed_data})...")
    few_shot_raw, few_shot_indices = sample_few_shot(
        TRAIN_DS["train"],
        n_shots=cfg.n_shots,
        seed=cfg.seed_data,
        premise_key=TRAIN_PKEY,
    )

    # Tokenisation du train few-shot
    few_shot_tok = few_shot_raw.map(
        tokenize_train, batched=True,
        remove_columns=few_shot_raw.column_names
    )
    few_shot_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Tokenisation de la validation (depuis le split val du train dataset)
    val_raw = TRAIN_DS["validation"]
    val_tok = val_raw.map(
        tokenize_train, batched=True,
        remove_columns=val_raw.column_names
    )
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Tokenisation du test
    # En mode INTRA, on vérifie que les indices few-shot ne sont pas dans le test set.
    # (Le test split et le train split sont disjoints par construction du DatasetDict.)
    test_raw = TEST_DS["test"]
    if EXP["mode"] == "intra":
        # Sécurité : vérifier qu'aucun exemple identique ne se retrouve dans les deux
        # (dans notre cas c'est garanti par la séparation train/test du DatasetDict)
        print(f"  ✅ Mode INTRA — Test set ({len(test_raw)} ex) disjoint du few-shot pool.")
    test_tok = test_raw.map(
        tokenize_test, batched=True,
        remove_columns=test_raw.column_names
    )
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # — Modèle PEFT —
    model = make_peft_model(
        model_cfg, cfg.peft_method,
        cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
    )

    # — TrainingArguments —
    # Learning rate plus élevé pour IA³ (recommandé dans le papier original)
    lr = 3e-3 if cfg.peft_method == "ia3" else float(cfg.learning_rate)

    output_dir = f"/tmp/fewshot_{EXP['name']}_{run_label}"
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=5,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=few_shot_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    trainer.train()

    # — Évaluation finale sur le test set cross-dataset —
    print(f"\n  🔍 Évaluation finale sur {len(test_raw)} exemples ({EXP['test_ds']})...")
    test_results = trainer.evaluate(test_tok, metric_key_prefix="test")
    test_acc = test_results["test_accuracy"]
    test_f1  = test_results.get("test_f1_score", 0.0)

    print(f"  ✅ TEST ACCURACY : {test_acc:.2%}")
    print(f"  ✅ TEST F1-SCORE : {test_f1:.4f}")

    # Logguer dans le résumé WandB pour faciliter les comparaisons
    wandb.summary["final_test_accuracy"] = test_acc
    wandb.summary["final_test_f1_score"] = test_f1
    wandb.summary["n_shots"] = cfg.n_shots
    wandb.summary["model"] = model_cfg["short"]
    wandb.summary["peft_method"] = cfg.peft_method

    # Nettoyage checkpoints temporaires
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    wandb.finish()


# ─────────────────────────────────────────────────────────
# 11. LANCEMENT DU SWEEP
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Calcul du nombre total de runs
    total_runs = 1
    for param in SWEEP_CONFIG["parameters"].values():
        if "values" in param:
            total_runs *= len(param["values"])

    print(f"\n{'='*60}")
    print(f"  Sweep : {EXP['label']}")
    print(f"  Nombre de runs prévus : {total_runs}")
    print(f"  (Paramètres : {list(SWEEP_CONFIG['parameters'].keys())})")
    print(f"{'='*60}")

    if len(sys.argv) > 1:
        confirm = "o"
    else:
        print("\n⚠️  Note : Le sweep complet (3 modèles × 2 PEFT × 5 n_shots × 3 seeds) = 90 runs.")
        print("   Pour tester, réduisez les listes dans SWEEP_CONFIG['parameters'].")
        confirm = input(f"\nLancer {total_runs} runs ? (o/n) : ").strip().lower()

    if confirm == "o":
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"\n  Sweep ID : {sweep_id}")
        print(f"  Dashboard : https://wandb.ai/reda300050-univer/fewshot-nli-fr/sweeps/{sweep_id}")
        wandb.agent(sweep_id, function=train_one_run, count=total_runs)
        print("\n✅ Sweep terminé ! Consultez WandB pour les courbes n_shots → accuracy.")
    else:
        print("Annulé.")
