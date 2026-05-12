"""
Script unifié de Sweep WandB pour les classifieurs NLI (~110M params).
Modèles : CamemBERT-XNLI, GPT-2 French, FlauBERT-XNLI
Mêmes splits de données pour une comparaison équitable.

Usage:
    python3 sweep_classifiers.py <modele> <experience> [auto]

    Modèles : camembert-xnli | gpt2 | flaubert-xnli
    Expériences :
      1. FraCaS quantifiers + GQNLI (logique formelle + grammaire)
      2. SICK few-shot (60 ex) → test SICK + FraCaS
      3. FraCaS formel → test SICK naturel (transfert cross-style)
      4. GQNLI pur (grammaire) → eval FraCaS
      5. Distribution unifiée multi-dataset (identique à Gemma)
      6. RTE3-FR binaire → DACCORD (non-contradiction vs contradiction)
      7. RTE3-FR intra 3 classes
      8. Courbe few-shot N-shot sur SICK (N = 10, 25, 50, 100, 200)

Sweeps EXP 1-7 : 24 combinaisons d'hyperparamètres LoRA (grid)
Sweep  EXP 8   : 10 combinaisons (5 N-shots × 2 lr) — courbe d'apprentissage
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION MODÈLES
# ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "camembert-xnli": {
        "hf_name": "almanach/camembert-base",
        "short": "camembert_xnli",
        "target_modules": ["query", "value"],
        "modules_to_save": ["classifier"],
        "is_decoder": False,
    },
    "gpt2": {
        "hf_name": "ClassCat/gpt2-base-french",
        "short": "gpt2",
        "target_modules": ["c_attn"],
        "modules_to_save": None,
        "is_decoder": True,
    },
    "flaubert-xnli": {
        "hf_name": "moussaKam/flaubert-base-cased-xnli",
        "short": "flaubert_xnli",
        "target_modules": ["q_lin", "v_lin"],
        "modules_to_save": ["classifier"],
        "is_decoder": False,
    },
}

# ─────────────────────────────────────────────────────────
# 2. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────

LABEL_MAP = {"yes": 0, "entailment": 0, "neutral": 1, "no": 2, "contradiction": 2}
INVALID_LABEL = -1  # Marqueur pour les labels inconnus (unknown, undef) — filtrés de toutes les phases


def map_label(label):
    """Retourne 0/1/2 pour les labels valides, INVALID_LABEL pour unknown/undef/non reconnus."""
    if isinstance(label, int):
        return label if label in [0, 1, 2] else INVALID_LABEL
    s = str(label).lower().strip()
    if s in ("unknown", "undef"):
        return INVALID_LABEL
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    try:
        v = int(s)
        return v if v in [0, 1, 2] else INVALID_LABEL
    except Exception:
        return INVALID_LABEL


def filter_valid(ds):
    """Supprime les exemples avec un label invalide (unknown/undef)."""
    before = len(ds)
    ds = ds.filter(lambda ex: ex["label"] != INVALID_LABEL)
    removed = before - len(ds)
    if removed > 0:
        print(f"    → {removed} exemple(s) inconnus/undef supprimés ({before} → {len(ds)})")
    return ds


def load_fracas():
    """Charge FraCaS filtré (sans undef), retourne (dataset, premise_key)."""
    fracas = load_dataset('maximoss/fracas')['train']
    fracas = fracas.filter(lambda x: str(x['label']).strip().lower() != "undef")
    return fracas, "premises"


def load_gqnli():
    """Charge GQNLI-FR, retourne (dataset, premise_key)."""
    return load_dataset('maximoss/gqnli-fr')['test'], "premise"


def load_sick():
    """Charge SICK-FR avec conversion de labels, retourne dict de splits."""
    sick = load_dataset('maximoss/sick-fr')

    def convert_sick(ex):
        lbl = str(ex['entailment_label']).strip().upper()
        if lbl == 'ENTAILMENT':
            label_id = 0
        elif lbl == 'NEUTRAL':
            label_id = 1
        elif lbl == 'CONTRADICTION':
            label_id = 2
        else:
            label_id = 1
        return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}

    result = {}
    for split_name in sick:
        result[split_name] = sick[split_name].map(convert_sick, remove_columns=sick[split_name].column_names)
    return result


def normalize_columns(ds, premise_key="premise"):
    """Garde uniquement premise, hypothesis, label. Filtre les unknown/undef."""
    cols_to_remove = [c for c in ds.column_names if c not in {premise_key, "hypothesis", "label"}]
    ds = ds.remove_columns(cols_to_remove)
    if premise_key != "premise":
        ds = ds.rename_column(premise_key, "premise")
    ds = ds.map(lambda ex: {"label": map_label(ex["label"])})
    return filter_valid(ds)


# ─────────────────────────────────────────────────────────
# 3. DÉFINITION DES EXPÉRIENCES
# ─────────────────────────────────────────────────────────

def build_experiment_1():
    """
    EXP 1 — FraCaS quantifiers + GQNLI (logique formelle + grammaire)
    Train : FraCaS 0-50 + GQNLI 80-100, 180-200, 280-300
    Val   : FraCaS 50-60 + GQNLI 61-80, 161-180, 261-280
    Test  : GQNLI 0-60, 100-160, 200-260 + FraCaS 60-72
    Eval séparée : FraCaS 0-72 complet + GQNLI train ranges
    """
    fracas, fkey = load_fracas()
    gqnli, gkey = load_gqnli()

    train_ds = concatenate_datasets([
        normalize_columns(fracas.select(range(0, 50)), fkey),
        normalize_columns(gqnli.select(list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))), gkey),
    ]).shuffle(seed=42)

    val_ds = concatenate_datasets([
        normalize_columns(fracas.select(range(50, 60)), fkey),
        normalize_columns(gqnli.select(list(range(61, 80)) + list(range(161, 180)) + list(range(261, 280))), gkey),
    ]).shuffle(seed=42)

    test_ds = concatenate_datasets([
        normalize_columns(gqnli.select(list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))), gkey),
        normalize_columns(fracas.select(range(60, 72)), fkey),
    ]).shuffle(seed=42)

    # Évaluations séparées
    eval_extra = {
        "fracas_full": normalize_columns(fracas.select(range(0, 72)), fkey),
        "gqnli_train_ranges": normalize_columns(
            gqnli.select(list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))), gkey
        ),
    }

    print(f"  EXP1 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return "fracas_gqnli_formal", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_2():
    """
    EXP 2 — SICK compositional (sémantique compositionnelle)
    Train : SICK train (60 premiers exemples few-shot)
    Val   : SICK validation
    Test  : SICK test + FraCaS 0-72 (transfert naturel → formel)
    Eval séparée : SICK test seul + FraCaS seul
    """
    sick = load_sick()
    fracas, fkey = load_fracas()

    train_ds = sick['train'].select(range(60)).shuffle(seed=42)
    val_ds = sick['validation']

    test_ds = concatenate_datasets([
        sick['test'],
        normalize_columns(fracas.select(range(0, 72)), fkey),
    ]).shuffle(seed=42)

    eval_extra = {
        "sick_test_only": sick['test'],
        "fracas_full": normalize_columns(fracas.select(range(0, 72)), fkey),
    }

    print(f"  EXP2 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return "sick_few_shot", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_3():
    """
    EXP 3 — Cross-style : FraCaS formel → SICK naturel
    Train : FraCaS 0-50 (quantificateurs formels)
    Val   : FraCaS 50-72
    Test  : SICK test (phrases naturelles compositionnelles)
    Eval séparée : SICK test + FraCaS 0-72
    Teste si la logique formelle aide à comprendre le langage naturel.
    """
    fracas, fkey = load_fracas()
    sick = load_sick()

    train_ds = normalize_columns(fracas.select(range(0, 50)), fkey).shuffle(seed=42)
    val_ds = normalize_columns(fracas.select(range(50, 72)), fkey)
    test_ds = sick['test']

    eval_extra = {
        "sick_test": sick['test'],
        "fracas_full": normalize_columns(fracas.select(range(0, 72)), fkey),
    }

    print(f"  EXP3 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return "fracas_to_sick_transfer", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_4():
    """
    EXP 4 — GQNLI pur (grammaire et acceptabilité)
    Train : GQNLI 80-100, 180-200, 280-300
    Val   : GQNLI 61-80, 161-180, 261-280
    Test  : GQNLI 0-60, 100-160, 200-260
    Eval séparée : FraCaS 0-72 (transfert grammaire → logique)
    Teste la capacité du modèle sur la grammaire pure.
    """
    gqnli, gkey = load_gqnli()
    fracas, fkey = load_fracas()

    train_ds = normalize_columns(
        gqnli.select(list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))), gkey
    ).shuffle(seed=42)

    val_ds = normalize_columns(
        gqnli.select(list(range(61, 80)) + list(range(161, 180)) + list(range(261, 280))), gkey
    )

    test_ds = normalize_columns(
        gqnli.select(list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))), gkey
    )

    eval_extra = {
        "fracas_full": normalize_columns(fracas.select(range(0, 72)), fkey),
    }

    print(f"  EXP4 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return "gqnli_grammar_pure", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_5():
    """
    EXP 5 — Distribution unifiée multi-dataset (identique à Gemma)
    Train : RTE3-DEV (90%) + SICK Train + DACCORD (1ère moitié) + GQNLI (80-100, 180-200, 280-300)
    Val   : SICK-VAL + RTE3-DEV (10%) + GQNLI (60-80, 160-180, 260-280)
    Test  : RTE3-test + SICK-Test + DACCORD (2ème moitié) + GQNLI (0-60, 100-160, 200-260)
    Même distribution que T5Gemma pour comparaison directe entre modèles.
    """
    gqnli, gkey = load_gqnli()
    sick = load_sick()

    # GQNLI
    gqnli_train = normalize_columns(gqnli.select(list(range(80, 100)) + list(range(180, 200)) + list(range(280, 300))), gkey)
    gqnli_val = normalize_columns(gqnli.select(list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))), gkey)
    gqnli_test = normalize_columns(gqnli.select(list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))), gkey)

    # RTE3-FR
    rte3 = load_dataset('maximoss/rte3-french')
    dev_keys = [k for k in rte3.keys() if k != 'test']
    rte3_dev = concatenate_datasets([rte3[k] for k in dev_keys]).shuffle(seed=42)
    split_90 = int(len(rte3_dev) * 0.9)
    rte3_train = rte3_dev.select(range(0, split_90)).rename_column("premise", "premise")
    rte3_val = rte3_dev.select(range(split_90, len(rte3_dev)))
    rte3_test_ds = rte3['test'] if 'test' in rte3 else None

    def normalize_rte3(ds):
        cols = [c for c in ds.column_names if c not in {"premise", "hypothesis", "label"}]
        ds = ds.remove_columns(cols)
        ds = ds.map(lambda ex: {"label": map_label(ex["label"])})
        return filter_valid(ds)

    # DACCORD
    daccord = load_dataset('maximoss/daccord-contradictions')['train'].shuffle(seed=42)
    daccord = daccord.map(lambda ex: {"label": 2 if ex["label"] == 1 else ex["label"]})
    half = len(daccord) // 2
    daccord_cols = [c for c in daccord.column_names if c not in {"premise", "hypothesis", "label"}]
    daccord = daccord.remove_columns(daccord_cols)
    daccord_train = daccord.select(range(0, half))
    daccord_test = daccord.select(range(half, len(daccord)))

    all_train = [gqnli_train, normalize_rte3(rte3_train), sick['train'], daccord_train]
    all_val = [gqnli_val, normalize_rte3(rte3_val), sick['validation']]
    all_test = [gqnli_test, sick['test'], daccord_test]
    if rte3_test_ds:
        all_test.append(normalize_rte3(rte3_test_ds))

    train_ds = concatenate_datasets(all_train).shuffle(seed=42)
    val_ds = concatenate_datasets(all_val).shuffle(seed=42)
    test_ds = concatenate_datasets(all_test).shuffle(seed=42)

    print(f"  EXP5 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return "unified_multi_dataset", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), {}, 3


def build_experiment_6():
    """
    EXP 6 — RTE3-FR (binaire) → DACCORD
    Classification binaire : non-contradiction (0) vs contradiction (1)
    Train : RTE3-FR validation (80%), labels remappés (entailment/neutral→0, contradiction→1)
    Val   : RTE3-FR validation (20%), même remapping
    Test  : DACCORD complet (labels originaux 0/1 — 0=non-contradiction, 1=contradiction)
    """
    rte3 = load_dataset('maximoss/rte3-french')
    daccord = load_dataset('maximoss/daccord-contradictions')['train'].shuffle(seed=42)

    def to_binary(ex):
        lbl = map_label(ex["label"])
        if lbl == INVALID_LABEL:
            return {"label": INVALID_LABEL}
        return {"label": 1 if lbl == 2 else 0}

    # RTE3 validation → train/val 80/20
    rte3_dev = rte3['validation'].map(to_binary)
    rte3_cols = [c for c in rte3_dev.column_names if c not in {"premise", "hypothesis", "label"}]
    rte3_dev = filter_valid(rte3_dev.remove_columns(rte3_cols))
    split = int(len(rte3_dev) * 0.8)
    train_ds = rte3_dev.select(range(0, split)).shuffle(seed=42)
    val_ds = rte3_dev.select(range(split, len(rte3_dev)))

    # DACCORD : labels originaux 0/1 (1=contradiction), garder tel quel
    daccord_cols = [c for c in daccord.column_names if c not in {"premise", "hypothesis", "label"}]
    test_ds = daccord.remove_columns(daccord_cols)

    print(f"  EXP6 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Labels train: {set(train_ds['label'])} | Labels test: {set(test_ds['label'])}")
    return "rte3_binary_to_daccord", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), {}, 2


def build_experiment_7():
    """
    EXP 7 — RTE3-FR intra (3 classes)
    Train : RTE3-FR validation (80%), 3 classes standard
    Val   : RTE3-FR validation (20%)
    Test  : RTE3-FR test complet (800 ex)
    Référence intra-dataset pour mesurer la capacité d'apprentissage pure sur RTE3.
    """
    rte3 = load_dataset('maximoss/rte3-french')

    rte3_dev = rte3['validation']
    rte3_cols = [c for c in rte3_dev.column_names if c not in {"premise", "hypothesis", "label"}]
    rte3_dev = filter_valid(rte3_dev.remove_columns(rte3_cols).map(lambda ex: {"label": map_label(ex["label"])}))
    split = int(len(rte3_dev) * 0.8)
    train_ds = rte3_dev.select(range(0, split)).shuffle(seed=42)
    val_ds = rte3_dev.select(range(split, len(rte3_dev)))

    rte3_test = filter_valid(rte3['test'].remove_columns(rte3_cols).map(lambda ex: {"label": map_label(ex["label"])}))

    print(f"  EXP7 — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(rte3_test)}")
    return "rte3_intra_3classes", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': rte3_test}), {}, 3


def build_experiment_8():
    """
    EXP 8 — Courbe few-shot N-shot sur SICK (data efficiency)
    Train : N premiers exemples de SICK train (N = 10, 25, 50, 100, 200 — paramètre de sweep)
    Val   : SICK validation complet
    Test  : SICK test complet
    Objectif : mesurer à partir de combien d'exemples chaque modèle "décolle".
    La sélection effective de N exemples se fait dans train_run() via config.n_shots.
    Retourne le dataset SICK complet — la troncature est appliquée au moment de l'entraînement.
    """
    sick = load_sick()
    print(f"  EXP8 — SICK complet Train: {len(sick['train'])} | Val: {len(sick['validation'])} | Test: {len(sick['test'])}")
    print(f"  (La courbe few-shot varie N via le paramètre n_shots dans le sweep)")
    eval_extra = {"sick_test": sick['test']}
    return "sick_nshot_curve", DatasetDict({'train': sick['train'], 'validation': sick['validation'], 'test': sick['test']}), eval_extra, 3


EXPERIMENTS = {
    "1": build_experiment_1, "2": build_experiment_2, "3": build_experiment_3,
    "4": build_experiment_4, "5": build_experiment_5,
    "6": build_experiment_6, "7": build_experiment_7,
    "8": build_experiment_8,
}

# ─────────────────────────────────────────────────────────
# 4. SWEEP CONFIGS
# ─────────────────────────────────────────────────────────
#
# EXP 1-7 : 24 combinaisons grid
#   lora_r      : [8, 16, 32]   — rang LoRA, capacité vs. surapprentissage
#   lora_alpha  : [16, 64]      — scaling effectif (alpha/r = 2 ou 8)
#   learning_rate: [5e-5, 1e-4, 3e-4, 5e-4]  — le paramètre le + critique
#   lora_dropout : [0.1]         — fixé, peu d'impact sur ~100M params
#   Total : 3 × 2 × 4 × 1 = 24 runs
#
# EXP 8 : courbe few-shot — n_shots varie dans le sweep
#   n_shots     : [10, 25, 50, 100, 200]
#   learning_rate: [1e-4, 3e-4]  — 2 valeurs pour vérifier stabilité
#   lora_r/alpha/dropout : fixés aux valeurs standard
#   Total : 5 × 2 × 1 × 1 × 1 = 10 runs

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r":        {"values": [8, 16, 32]},
        # lora_alpha est déduit automatiquement dans train_run() : alpha = 2 * r
        # (convention LoRA standard — évite les combinaisons redondantes r/alpha)
        "learning_rate": {"values": [5e-5, 1e-4, 3e-4, 5e-4]},
        "lora_dropout":  {"values": [0.05, 0.1, 0.2]},
    }
}
# Total EXP 1-7 : 3 × 4 × 3 = 36 runs

# Sweep dédié EXP 8 : n_shots est le paramètre principal
SWEEP_CONFIG_EXP8 = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r":        {"values": [16]},
        # lora_alpha = 2 * r = 32 (déduit automatiquement)
        "learning_rate": {"values": [1e-4, 3e-4]},
        "lora_dropout":  {"values": [0.1]},
        "n_shots":       {"values": [10, 25, 50, 100, 200]},
    }
}
# Total EXP 8 : 1 × 2 × 1 × 5 = 10 runs

# ─────────────────────────────────────────────────────────
# 5. PARSING DES ARGUMENTS
# ─────────────────────────────────────────────────────────

if len(sys.argv) < 3:
    print("Usage: python3 sweep_classifiers.py <modele> <experience> [auto]")
    print(f"\nModèles disponibles : {', '.join(MODEL_CONFIGS.keys())}")
    print("Expériences :")
    print("  1. FraCaS quantifiers + GQNLI (logique formelle + grammaire)        — 24 runs")
    print("  2. SICK few-shot (60 ex) → test SICK + FraCaS                        — 24 runs")
    print("  3. FraCaS formel → test SICK naturel (transfert cross-style)         — 24 runs")
    print("  4. GQNLI pur (grammaire) → eval FraCaS                               — 24 runs")
    print("  5. Distribution unifiée multi-dataset (identique à Gemma)            — 24 runs")
    print("  6. RTE3-FR binaire → DACCORD (non-contradiction vs contradiction)    — 24 runs")
    print("  7. RTE3-FR intra 3 classes                                            — 24 runs")
    print("  8. Courbe few-shot N-shot sur SICK (N = 10, 25, 50, 100, 200)        — 10 runs")
    exit(1)

model_choice = sys.argv[1].strip().lower()
exp_choice = sys.argv[2].strip()

if model_choice not in MODEL_CONFIGS:
    print(f"Modèle '{model_choice}' inconnu. Choix : {', '.join(MODEL_CONFIGS.keys())}")
    exit(1)
if exp_choice not in EXPERIMENTS:
    print(f"Expérience '{exp_choice}' inconnue. Choix : {', '.join(sorted(EXPERIMENTS.keys()))}")
    exit(1)

MODEL_CFG = MODEL_CONFIGS[model_choice]
print(f"\n{'='*60}")
print(f"MODÈLE : {MODEL_CFG['hf_name']} | EXPÉRIENCE : {exp_choice}")
print(f"{'='*60}")

if not torch.cuda.is_available():
    print("ERREUR : GPU CUDA requis.")
    exit(1)

# ─────────────────────────────────────────────────────────
# 6. CHARGEMENT DONNÉES + TOKENISATION
# ─────────────────────────────────────────────────────────

EXP_NAME, UNIFIED, EVAL_EXTRA, NUM_LABELS = EXPERIMENTS[exp_choice]()

global_tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG["hf_name"])

# GPT-2 : fixer le pad token
if MODEL_CFG["is_decoder"]:
    if global_tokenizer.pad_token is None:
        global_tokenizer.pad_token = global_tokenizer.eos_token
    global_tokenizer.padding_side = "right"


def tokenize_fn(examples):
    if MODEL_CFG["is_decoder"]:
        prompts = [f"Prémisse : {p}\nHypothèse : {h}\n" for p, h in zip(examples["premise"], examples["hypothesis"])]
        res = global_tokenizer(prompts, truncation=True, max_length=256)
    else:
        res = global_tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)
    # Pour EXP 6 (binaire), les labels sont déjà 0/1 — on les passe tels quels
    res["labels"] = [int(l) for l in examples["label"]]
    return res


print("\nTokenisation...")
train_data = UNIFIED['train'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = UNIFIED['validation'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_data = UNIFIED['test'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['test'].column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Tokeniser les évaluations séparées
eval_extra_tokenized = {}
for name, ds in EVAL_EXTRA.items():
    tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_extra_tokenized[name] = tok

# ─────────────────────────────────────────────────────────
# 7. MÉTRIQUES (avec confusion matrix + per-class dans WandB)
# ─────────────────────────────────────────────────────────

LABEL_NAMES_3 = ["entailment", "neutral", "contradiction"]
LABEL_NAMES_2 = ["non-contradiction", "contradiction"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")

    label_ids = list(range(NUM_LABELS))
    label_names = LABEL_NAMES_2 if NUM_LABELS == 2 else LABEL_NAMES_3

    try:
        cm = confusion_matrix(labels, predictions, labels=label_ids)
        print(f"\nMatrice de confusion:\n{cm}")
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=labels.tolist(), preds=predictions.tolist(),
                class_names=label_names
            )
        })
        prec, rec, f1_per, sup = precision_recall_fscore_support(labels, predictions, labels=label_ids, zero_division=0)
        for i, name in enumerate(label_names):
            wandb.log({
                f"{name}_precision": prec[i],
                f"{name}_recall": rec[i],
                f"{name}_f1": f1_per[i],
                f"{name}_support": int(sup[i]),
            })
    except Exception as e:
        print(f"Erreur logging confusion matrix: {e}")

    return {"accuracy": acc, "f1_score": f1}


# ─────────────────────────────────────────────────────────
# 8. FONCTION D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────

def train_run():
    run = wandb.init()
    config = wandb.config

    # lora_alpha = 2 × r (convention standard, non balayé)
    lora_alpha = 2 * config.lora_r

    # EXP 8 : n_shots est un paramètre du sweep
    n_shots = getattr(config, "n_shots", None)
    nshots_suffix = f"_n{n_shots}" if n_shots is not None else ""
    run_label = f"{MODEL_CFG['short']}_exp{exp_choice}_r{config.lora_r}_a{lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}{nshots_suffix}"

    # Dataset d'entraînement effectif (troncature pour EXP 8)
    if n_shots is not None:
        current_train_data = train_data.select(range(min(n_shots, len(train_data))))
        print(f"\n[EXP8] N-shot = {n_shots} → {len(current_train_data)} exemples d'entraînement")
    else:
        current_train_data = train_data

    print(f"\n{'='*60}\n{run_label}\n{'='*60}")

    # Chargement du modèle
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CFG["hf_name"], num_labels=NUM_LABELS, device_map="auto"
    )
    base_model.config.use_cache = False
    if MODEL_CFG["is_decoder"]:
        base_model.config.pad_token_id = global_tokenizer.pad_token_id

    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "r": config.lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": MODEL_CFG["target_modules"],
        "bias": "none",
    }
    if MODEL_CFG["modules_to_save"]:
        lora_kwargs["modules_to_save"] = MODEL_CFG["modules_to_save"]

    model = get_peft_model(base_model, LoraConfig(**lora_kwargs))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {trainable:,} ({100 * trainable / total:.2f}%)")

    args = TrainingArguments(
        output_dir=f"/tmp/ckpt_{EXP_NAME}_{run_label}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=current_train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
    )

    # ── Zero-shot ──
    print(f"\n[ZERO-SHOT] Évaluation sur Test ({len(test_data)} exemples)...")
    zs_results = trainer.evaluate(test_data, metric_key_prefix="zeroshot")
    zs_acc = zs_results["zeroshot_accuracy"]
    zs_f1 = zs_results.get("zeroshot_f1_score", 0.0)
    print(f"ZERO-SHOT — Accuracy: {zs_acc:.2%} | F1: {zs_f1:.4f}")
    wandb.summary["zeroshot_accuracy"] = zs_acc
    wandb.summary["zeroshot_f1_score"] = zs_f1

    # Zero-shot sur évals séparées
    for eval_name, eval_ds in eval_extra_tokenized.items():
        print(f"\n[ZERO-SHOT] Éval séparée : {eval_name} ({len(eval_ds)} ex)...")
        zs_extra = trainer.evaluate(eval_ds, metric_key_prefix=f"zeroshot_{eval_name}")
        wandb.summary[f"zeroshot_{eval_name}_accuracy"] = zs_extra[f"zeroshot_{eval_name}_accuracy"]
        wandb.summary[f"zeroshot_{eval_name}_f1_score"] = zs_extra.get(f"zeroshot_{eval_name}_f1_score", 0.0)

    # ── Entraînement ──
    trainer.train()

    # ── Évaluation finale sur Test ──
    print(f"\n[TEST FINAL] {len(test_data)} exemples...")
    test_results = trainer.evaluate(test_data, metric_key_prefix="test")
    test_acc = test_results["test_accuracy"]
    test_f1 = test_results.get("test_f1_score", 0.0)
    print(f"TEST — Accuracy: {test_acc:.2%} | F1: {test_f1:.4f}")
    wandb.summary["final_test_accuracy"] = test_acc
    wandb.summary["final_test_f1_score"] = test_f1

    # Évals séparées post-entraînement
    for eval_name, eval_ds in eval_extra_tokenized.items():
        print(f"\n[EVAL POST-TRAIN] {eval_name} ({len(eval_ds)} ex)...")
        extra_results = trainer.evaluate(eval_ds, metric_key_prefix=f"final_{eval_name}")
        wandb.summary[f"final_{eval_name}_accuracy"] = extra_results[f"final_{eval_name}_accuracy"]
        wandb.summary[f"final_{eval_name}_f1_score"] = extra_results.get(f"final_{eval_name}_f1_score", 0.0)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    wandb.finish()


# ─────────────────────────────────────────────────────────
# 9. LANCEMENT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sélection de la config de sweep selon l'expérience
    active_sweep_config = SWEEP_CONFIG_EXP8 if exp_choice == "8" else SWEEP_CONFIG

    total_runs = 1
    for param in active_sweep_config["parameters"].values():
        if "values" in param:
            total_runs *= len(param["values"])

    if exp_choice == "8":
        print(f"\nEXP 8 — Courbe few-shot : {total_runs} runs")
        print(f"  N-shots  : {active_sweep_config['parameters']['n_shots']['values']}")
        print(f"  LR       : {active_sweep_config['parameters']['learning_rate']['values']}")
    else:
        print(f"\nEXP {exp_choice} — Sweep hyperparamètres : {total_runs} runs")
        r_vals = active_sweep_config['parameters']['lora_r']['values']
        print(f"  lora_r   : {r_vals}  →  lora_alpha (2×r) : {[2*r for r in r_vals]}")
        print(f"  lr       : {active_sweep_config['parameters']['learning_rate']['values']}")
        print(f"  dropout  : {active_sweep_config['parameters']['lora_dropout']['values']}")

    auto = len(sys.argv) > 3
    if auto:
        confirm = "o"
    else:
        confirm = input(f"\nLancer {total_runs} run(s) {MODEL_CFG['short']} EXP{exp_choice} ? (o/n): ").strip().lower()

    if confirm == "o":
        sweep_id = wandb.sweep(sweep=active_sweep_config, project="fewshot-nli-fr")
        print(f"\nSweep créé ! ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_run, count=total_runs)
        print("\nTerminé !")
    else:
        print("Annulé.")
