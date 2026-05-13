"""
Script unifié de Sweep WandB pour les classifieurs NLI (~110M params).
Modèles : CamemBERT-XNLI, GPT-2 French, FlauBERT-XNLI, FlauBERT-custom

Expériences (spécifications tuteur — Mémoire Master 1 GL) :
  1. FraCaS(GQ)[0:50]+GQNLI[80-100,180-200,280-300] → val=FraCaS[50:60]+GQNLI[60-80,160-180,260-280]
     test=GQNLI[0-60,100-160,200-260]+FraCaS[60:]   (logique formelle + grammaire)
  2. SICK_train[0:60] → val=SICK_val / test=SICK_test+FraCaS_GQ   (few-shot sémantique)
  3. FraCaS_GQ[0:50] → val=FraCaS_GQ[50:] / test=SICK_test        (transfert cross-style)
  4. GQNLI[80-100,180-200,280-300] → val=GQNLI[60-80,...] / test=GQNLI[0-60,...]+FraCaS_GQ (grammaire→logique)
  5. RTE3-dev(90%)+SICK_train+DACCORD(½)+GQNLI[80-100,...] → val=RTE3-dev(10%)+SICK_val+GQNLI[60-80,...]
     test=RTE3-test+SICK_test+DACCORD(½)+GQNLI[0-60,...]           (distribution unifiée / comparaison Gemma)
  6. RTE3-dev(80%, binaire) → val=RTE3-dev(20%, binaire) / test=DACCORD complet  (transfert binaire)
  7. RTE3-dev(80%, 3-class) → val=RTE3-dev(20%) / test=RTE3-test   (intra-dataset 3 classes)
  8. Courbe N-shot SICK (N=10,25,50,100,200) — balanced_select      (data efficiency)

Splitting GQNLI (plages tuteur) :
  test  : [0:60,  100:160, 200:260]  →  180 ex
  val   : [60:80, 160:180, 260:280]  →   60 ex
  train : [80:100,180:200, 280:300]  →   60 ex

Autres règles :
  - DACCORD : split proportionnel par thème (a/b/c) + même prémisse = même split
  - RTE3     : même prémisse = même split
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
# Multi-GPU : laisser le Trainer utiliser tous les GPUs disponibles (DataParallel auto)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # décommenter pour forcer GPU unique

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
import shutil

from collections import defaultdict
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATION MODÈLES
# ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "camembert-xnli": {
        "hf_name": "mtheo/camembert-base-xnli",
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
    # Modèle FlauBERT custom entraîné par un ami — remplacer par le repo HuggingFace
    # ou un chemin local (ex: "/kaggle/input/custom-flaubert/model")
    "flaubert-custom": {
        "hf_name": os.environ.get("CUSTOM_FLAUBERT_MODEL", "TODO/flaubert-xnli-custom"),
        "short": "flaubert_custom",
        "target_modules": ["q_lin", "v_lin"],
        "modules_to_save": ["classifier"],
        "is_decoder": False,
    },
}

# ─────────────────────────────────────────────────────────
# 2. SPLIT GQNLI PAR PLAGES D'INDEX (tuteur)
# ─────────────────────────────────────────────────────────

def gqnli_by_index(gqnli_ds):
    """
    Splits GQNLI-FR (300 ex, 3 pages × 100) selon les plages tuteur :
      test  : [0:60,  100:160, 200:260]  → 180 ex
      val   : [60:80, 160:180, 260:280]  →  60 ex
      train : [80:100,180:200, 280:300]  →  60 ex
    """
    test_idx  = list(range(0, 60))  + list(range(100, 160)) + list(range(200, 260))
    val_idx   = list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))
    train_idx = list(range(80,100)) + list(range(180, 200)) + list(range(280, 300))
    return (gqnli_ds.select(train_idx),
            gqnli_ds.select(val_idx),
            gqnli_ds.select(test_idx))


def load_daccord_full(binary=True):
    """Charge DACCORD complet avec labels natifs (0=compatible, 1=contradiction)."""
    dacc = load_dataset('maximoss/daccord-contradictions')['train']
    cols = [c for c in dacc.column_names if c not in {'premise', 'hypothesis', 'label'}]
    dacc = dacc.remove_columns(cols)
    if not binary:
        dacc = dacc.map(lambda ex: {'label': 2 if ex['label'] == 1 else 0})
    return dacc

# ─────────────────────────────────────────────────────────
# 3. UTILITAIRES DE LABELS
# ─────────────────────────────────────────────────────────

LABEL_MAP = {"yes": 0, "entailment": 0, "neutral": 1, "no": 2, "contradiction": 2}
INVALID_LABEL = -1


def map_label(label):
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
    before = len(ds)
    ds = ds.filter(lambda ex: ex["label"] != INVALID_LABEL)
    removed = before - len(ds)
    if removed > 0:
        print(f"    → {removed} exemple(s) invalides supprimés ({before} → {len(ds)})")
    return ds


def normalize_columns(ds, premise_key="premise"):
    cols_to_remove = [c for c in ds.column_names if c not in {premise_key, "hypothesis", "label"}]
    ds = ds.remove_columns(cols_to_remove)
    if premise_key != "premise":
        ds = ds.rename_column(premise_key, "premise")
    ds = ds.map(lambda ex: {"label": map_label(ex["label"])})
    return filter_valid(ds)


def print_dist(ds, name):
    from collections import Counter
    c = Counter(ds["label"])
    total = len(ds)
    parts = [f"{n}={c.get(n,0)}({100*c.get(n,0)/total:.0f}%)" for n in ["0-ent","1-neu","2-con"]]
    # Simplified
    ent, neu, con = c.get(0,0), c.get(1,0), c.get(2,0)
    print(f"    {name} [{total}]: ent={ent}({100*ent/total:.0f}%) neu={neu}({100*neu/total:.0f}%) con={con}({100*con/total:.0f}%)")

# ─────────────────────────────────────────────────────────
# 4. FONCTIONS DE CHARGEMENT ET SPLITTING
# ─────────────────────────────────────────────────────────

def load_fracas_gq():
    """Charge FraCaS filtré sur Quantificateurs Généralisés (topic GQ, sans undef), 80 ex."""
    fracas = load_dataset('maximoss/fracas')['train']
    fracas = fracas.filter(lambda x: x['topic'] == 'GENERALIZED QUANTIFIERS'
                           and str(x['label']).strip().lower() != 'undef')
    return normalize_columns(fracas, premise_key="premises")


def load_fracas_all():
    """Charge tout FraCaS (tous phénomènes, sans undef), ~335 ex."""
    fracas = load_dataset('maximoss/fracas')['train']
    fracas = fracas.filter(lambda x: str(x['label']).strip().lower() != 'undef')
    return normalize_columns(fracas, premise_key="premises")


def load_gqnli():
    """Charge GQNLI-FR complet (300 ex)."""
    ds = load_dataset('maximoss/gqnli-fr')['test']
    return normalize_columns(ds, premise_key="premise")




def load_sick():
    """Charge SICK-FR avec conversion de labels standard."""
    sick = load_dataset('maximoss/sick-fr')

    def convert(ex):
        lbl = str(ex['entailment_label']).strip().upper()
        if lbl == 'ENTAILMENT':    label_id = 0
        elif lbl == 'NEUTRAL':     label_id = 1
        elif lbl == 'CONTRADICTION': label_id = 2
        else:                      label_id = 1
        return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}

    result = {}
    for split in sick:
        result[split] = sick[split].map(convert, remove_columns=sick[split].column_names)
    return result


def sick_sample(sick_split_ds, ratio, seed=42):
    """Sélection stratifiée d'un ratio du dataset SICK."""
    idx = list(range(len(sick_split_ds)))
    labels = sick_split_ds['label']
    _, idx_keep = train_test_split(idx, test_size=ratio, random_state=seed, stratify=labels)
    return sick_split_ds.select(sorted(idx_keep))


def split_rte3_by_premise(train_r=0.8, seed=42, binary=False):
    """
    Charge RTE3-dev et le split 80/20 en respectant les groupes de prémisses.
    Retourne (train_ds, val_ds, test_ds).
    """
    rng = random.Random(seed)
    rte3 = load_dataset('maximoss/rte3-french')
    dev_keys = [k for k in rte3.keys() if k != 'test']

    # Agréger le dev et grouper par prémisse
    all_rows = []
    prem_groups = defaultdict(list)
    for k in dev_keys:
        for ex in rte3[k]:
            idx = len(all_rows)
            all_rows.append({'premise': ex['premise'], 'hypothesis': ex['hypothesis'],
                              'label': map_label(ex['label'])})
            prem_groups[ex['premise']].append(idx)

    groups = list(prem_groups.values())
    rng.shuffle(groups)

    n_train = int(len(all_rows) * train_r)
    train_idx, val_idx = [], []
    for g in groups:
        if len(train_idx) < n_train:
            train_idx.extend(g)
        else:
            val_idx.extend(g)

    def make_ds(indices, rows, binary):
        ds = Dataset.from_list([rows[i] for i in indices])
        ds = filter_valid(ds)
        if binary:
            ds = ds.map(lambda ex: {'label': 1 if ex['label'] == 2 else 0})
        return ds

    # Test set
    test_rows = [{'premise': ex['premise'], 'hypothesis': ex['hypothesis'],
                  'label': map_label(ex['label'])} for ex in rte3['test']]
    test_ds = Dataset.from_list(test_rows)
    test_ds = filter_valid(test_ds)
    if binary:
        test_ds = test_ds.map(lambda ex: {'label': 1 if ex['label'] == 2 else 0})

    return make_ds(train_idx, all_rows, binary), make_ds(val_idx, all_rows, binary), test_ds


def split_daccord_by_theme(train_r=0.8, val_r=0.2, seed=42, binary=False):
    """
    Charge DACCORD et split en respectant :
    - Proportionnalité par thème (a=Russie-Ukraine, b=COVID, c=Climat)
    - Même prémisse = même split
    Retourne (train_ds, val_ds) ou (train_ds, val_ds, test_ds) si train_r+val_r < 1.
    """
    rng = random.Random(seed)
    dacc = load_dataset('maximoss/daccord-contradictions')['train']

    # Grouper par thème puis par prémisse
    themes = defaultdict(lambda: defaultdict(list))
    for i, ex in enumerate(dacc):
        theme = str(ex['id'])[0]
        themes[theme][ex['premise']].append(i)

    split_idx = {'train': [], 'val': [], 'test': []}

    for theme, prem_groups in themes.items():
        groups = list(prem_groups.values())
        rng.shuffle(groups)
        theme_total = sum(len(g) for g in groups)
        n_train = int(theme_total * train_r)
        n_val   = int(theme_total * val_r)
        t_tr, t_vl, t_te = [], [], []
        for g in groups:
            if len(t_tr) < n_train:
                t_tr.extend(g)
            elif len(t_vl) < n_val:
                t_vl.extend(g)
            else:
                t_te.extend(g)
        split_idx['train'].extend(t_tr)
        split_idx['val'].extend(t_vl)
        split_idx['test'].extend(t_te)

    def make_ds(indices):
        ds = dacc.select(sorted(indices))
        cols = [c for c in ds.column_names if c not in {'premise', 'hypothesis', 'label'}]
        ds = ds.remove_columns(cols)
        if binary:
            # 0=non-contradiction, 1=contradiction (labels DACCORD natifs : 0=compatibles, 1=contradiction)
            return ds
        else:
            # Remap vers 3-class : 0=entailment, 2=contradiction
            return ds.map(lambda ex: {'label': 2 if ex['label'] == 1 else 0})

    result = {k: make_ds(v) for k, v in split_idx.items() if v}
    for k, ds in result.items():
        print_dist(ds, f"DACCORD-{k}")
    return result


def balanced_select(ds, n, seed=42):
    """Sélectionne exactement n//3 exemples par classe pour un train parfaitement équilibré."""
    rng = random.Random(seed)
    per_class = n // 3
    by_label = defaultdict(list)
    for i, label in enumerate(ds['label']):
        if label in [0, 1, 2]:
            by_label[label].append(i)
    selected = []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        selected += indices[:per_class]
    rng.shuffle(selected)
    print(f"    balanced_select: {per_class} ex/classe → {len(selected)} total")
    return ds.select(selected)

# ─────────────────────────────────────────────────────────
# 5. DÉFINITION DES EXPÉRIENCES (PDF tuteur)
# ─────────────────────────────────────────────────────────

def build_experiment_1():
    """EXP 1 — FraCaS quantifiers + GQNLI (logique formelle + grammaire)"""
    fracas = load_fracas_gq()
    gqnli  = load_gqnli()
    gq_train, gq_val, gq_test = gqnli_by_index(gqnli)
    n = len(fracas)
    train_ds = concatenate_datasets([fracas.select(range(min(50, n))), gq_train]).shuffle(seed=42)
    val_ds   = concatenate_datasets([fracas.select(range(min(50, n), min(60, n))), gq_val]).shuffle(seed=42)
    test_ds  = concatenate_datasets([gq_test, fracas.select(range(min(60, n), n))]).shuffle(seed=42)
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)]:
        print_dist(ds, name)
    print(f"  EXP1 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    return "fracas_gq_gqnli_mixed", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), {}, 3


def build_experiment_2():
    """EXP 2 — SICK few-shot 60 ex (sémantique compositionnelle)"""
    sick   = load_sick()
    fracas = load_fracas_gq()
    train_ds = sick['train'].select(range(min(60, len(sick['train'])))).shuffle(seed=42)
    val_ds   = sick['validation']
    test_ds  = concatenate_datasets([sick['test'], fracas]).shuffle(seed=42)
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)]:
        print_dist(ds, name)
    print(f"  EXP2 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    eval_extra = {"sick_test_only": sick['test'], "fracas_gq_only": fracas}
    return "sick_60shot_fracas_test", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_3():
    """EXP 3 — FraCaS formel → SICK naturel (transfert cross-style)"""
    fracas = load_fracas_gq()
    sick   = load_sick()
    n50    = min(50, len(fracas))
    train_ds = fracas.select(range(n50)).shuffle(seed=42)
    val_ds   = fracas.select(range(n50, len(fracas)))
    test_ds  = sick['test']
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)]:
        print_dist(ds, name)
    print(f"  EXP3 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    eval_extra = {"fracas_gq": fracas}
    return "fracas_gq_to_sick", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_4():
    """EXP 4 — GQNLI pur (grammaire → logique) + eval séparée FraCaS"""
    fracas = load_fracas_gq()
    gqnli  = load_gqnli()
    gq_train, gq_val, gq_test = gqnli_by_index(gqnli)
    for name, ds in [("TRAIN", gq_train), ("VAL", gq_val), ("TEST", gq_test)]:
        print_dist(ds, name)
    print(f"  EXP4 — Train:{len(gq_train)} Val:{len(gq_val)} Test:{len(gq_test)}")
    eval_extra = {"fracas_gq": fracas}
    return "gqnli_pure", DatasetDict({'train': gq_train.shuffle(seed=42), 'validation': gq_val, 'test': gq_test}), eval_extra, 3


def build_experiment_5():
    """EXP 5 — Distribution unifiée multi-dataset (comparaison Gemma)"""
    sick   = load_sick()
    gqnli  = load_gqnli()
    dacc   = split_daccord_by_theme(train_r=0.5, val_r=0.0, seed=42, binary=False)
    gq_train, gq_val, gq_test = gqnli_by_index(gqnli)
    rte3_train, rte3_val, rte3_test = split_rte3_by_premise(train_r=0.9, seed=42, binary=False)
    train_ds = concatenate_datasets([rte3_train, sick['train'], dacc['train'], gq_train]).shuffle(seed=42)
    val_ds   = concatenate_datasets([rte3_val, sick['validation'], gq_val]).shuffle(seed=42)
    test_ds  = concatenate_datasets([rte3_test, sick['test'], dacc.get('test', sick['test'].select([])), gq_test]).shuffle(seed=42)
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)]:
        print_dist(ds, name)
    print(f"  EXP5 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    return "unified_multidataset", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), {}, 3


def build_experiment_6():
    """EXP 6 — RTE3-FR binaire → DACCORD (classifieurs uniquement)"""
    rte3_train, rte3_val, _ = split_rte3_by_premise(train_r=0.8, seed=42, binary=True)
    dacc_test = load_daccord_full(binary=True)
    for name, ds in [("TRAIN", rte3_train), ("VAL", rte3_val), ("TEST", dacc_test)]:
        print_dist(ds, name)
    print(f"  EXP6 — Train:{len(rte3_train)} Val:{len(rte3_val)} Test:{len(dacc_test)}")
    return "rte3_binary_to_daccord", DatasetDict({'train': rte3_train, 'validation': rte3_val, 'test': dacc_test}), {}, 2


def build_experiment_7():
    """EXP 7 — RTE3-FR intra 3 classes (classifieurs uniquement)"""
    rte3_train, rte3_val, rte3_test = split_rte3_by_premise(train_r=0.8, seed=42, binary=False)
    for name, ds in [("TRAIN", rte3_train), ("VAL", rte3_val), ("TEST", rte3_test)]:
        print_dist(ds, name)
    print(f"  EXP7 — Train:{len(rte3_train)} Val:{len(rte3_val)} Test:{len(rte3_test)}")
    return "rte3_intra_3class", DatasetDict({'train': rte3_train, 'validation': rte3_val, 'test': rte3_test}), {}, 3


def build_experiment_8():
    """EXP 8 — Courbe few-shot N-shot sur SICK (data efficiency — balanced_select)"""
    sick = load_sick()
    print(f"  EXP8 — SICK Train:{len(sick['train'])} Val:{len(sick['validation'])} Test:{len(sick['test'])}")
    return "sick_nshot_curve", DatasetDict({'train': sick['train'], 'validation': sick['validation'], 'test': sick['test']}), {}, 3


def build_experiment_9():
    """EXP 9 — FraCaS GQ (85/15) → test=GQNLI complet (split pur, sans mélange)"""
    fracas = load_fracas_gq()
    gqnli  = load_gqnli()
    labels = fracas['label']
    idx    = list(range(len(fracas)))
    idx_tr, idx_vl = train_test_split(idx, test_size=0.15, random_state=42, stratify=labels)
    train_ds = fracas.select(idx_tr).shuffle(seed=42)
    val_ds   = fracas.select(idx_vl)
    test_ds  = gqnli
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)]:
        print_dist(ds, name)
    print(f"  EXP9 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    return "fracas_gq_split_gqnli_test", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), {}, 3


def build_experiment_10():
    """EXP 10 — FraCaS GQ(85%)+SICK(20%) → val=FraCaS(15%)+SICK(15%) / test=GQNLI+SICK_test"""
    fracas = load_fracas_gq()
    gqnli  = load_gqnli()
    sick   = load_sick()
    idx    = list(range(len(fracas)))
    idx_tr, idx_vl = train_test_split(idx, test_size=0.15, random_state=42, stratify=fracas['label'])
    sick_tr = sick_sample(sick['train'],      ratio=0.20)
    sick_vl = sick_sample(sick['validation'], ratio=0.15)
    train_ds = concatenate_datasets([fracas.select(idx_tr), sick_tr]).shuffle(seed=42)
    val_ds   = concatenate_datasets([fracas.select(idx_vl), sick_vl]).shuffle(seed=42)
    test_ds  = concatenate_datasets([gqnli, sick['test']]).shuffle(seed=42)
    eval_extra = {"gqnli_only": gqnli, "sick_test_only": sick['test']}
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds)]:
        print_dist(ds, name)
    print(f"  EXP10 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    return "fracas_sick_aug_gqnli_test", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


def build_experiment_11():
    """EXP 11 — GQNLI(80%)+SICK(25%) → val=GQNLI(20%)+SICK(25%) / test=FraCaS GQ+SICK_test"""
    fracas = load_fracas_gq()
    gqnli  = load_gqnli()
    sick   = load_sick()
    gq_train, gq_val, _ = gqnli_by_index(gqnli)
    sick_tr = sick_sample(sick['train'],      ratio=0.25)
    sick_vl = sick_sample(sick['validation'], ratio=0.25)
    train_ds = concatenate_datasets([gq_train, sick_tr]).shuffle(seed=42)
    val_ds   = concatenate_datasets([gq_val,   sick_vl]).shuffle(seed=42)
    test_ds  = concatenate_datasets([fracas, sick['test']]).shuffle(seed=42)
    eval_extra = {"fracas_gq_only": fracas, "sick_test_only": sick['test']}
    for name, ds in [("TRAIN", train_ds), ("VAL", val_ds)]:
        print_dist(ds, name)
    print(f"  EXP11 — Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    return "gqnli_sick_aug_fracas_test", DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds}), eval_extra, 3


EXPERIMENTS = {
    "1":  build_experiment_1,  "2":  build_experiment_2,
    "3":  build_experiment_3,  "4":  build_experiment_4,
    "5":  build_experiment_5,  "6":  build_experiment_6,
    "7":  build_experiment_7,  "8":  build_experiment_8,
    "9":  build_experiment_9,  "10": build_experiment_10,
    "11": build_experiment_11,
}

# ─────────────────────────────────────────────────────────
# 6. SWEEP CONFIGS
# EXP 1-7, 9-11 : 8 runs (r∈{8,16} × lr∈{5e-5,1e-4,3e-4,5e-4} × dropout fixé 0.1)
# EXP 8         : 10 runs (5 N-shots × 2 lr)
# ─────────────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r":        {"values": [8, 16]},
        "learning_rate": {"values": [5e-5, 1e-4, 3e-4, 5e-4]},
        "lora_dropout":  {"values": [0.1]},
    }
}
# Total EXP 1-7, 9-11 : 2 × 4 × 1 = 8 runs

SWEEP_CONFIG_EXP8 = {
    "method": "grid",
    "metric": {"name": "eval/f1_score", "goal": "maximize"},
    "parameters": {
        "lora_r":        {"values": [16]},
        "learning_rate": {"values": [1e-4, 3e-4]},
        "lora_dropout":  {"values": [0.1]},
        "n_shots":       {"values": [10, 25, 50, 100, 200]},
    }
}
# Total EXP 8 : 1 × 2 × 1 × 5 = 10 runs

# ─────────────────────────────────────────────────────────
# 7. PARSING DES ARGUMENTS
# ─────────────────────────────────────────────────────────

if len(sys.argv) < 3:
    print("Usage: python3 sweep_classifiers.py <modele> <experience> [auto]")
    print(f"\nModèles : {', '.join(MODEL_CONFIGS.keys())}")
    print("Expériences :")
    for k, fn in sorted(EXPERIMENTS.items(), key=lambda x: int(x[0])):
        print(f"  {k:>2}. {fn.__doc__.strip().split(chr(10))[0].strip()}")
    exit(1)

model_choice = sys.argv[1].strip().lower()
exp_choice   = sys.argv[2].strip()

if model_choice not in MODEL_CONFIGS:
    print(f"Modèle '{model_choice}' inconnu.")
    exit(1)
if exp_choice not in EXPERIMENTS:
    print(f"Expérience '{exp_choice}' inconnue.")
    exit(1)

MODEL_CFG = MODEL_CONFIGS[model_choice]
print(f"\n{'='*60}")
print(f"MODÈLE : {MODEL_CFG['hf_name']} | EXPÉRIENCE : {exp_choice}")
print(f"{'='*60}")

if not torch.cuda.is_available():
    print("ERREUR : GPU CUDA requis.")
    exit(1)

# ─────────────────────────────────────────────────────────
# 8. CHARGEMENT DONNÉES + TOKENISATION
# ─────────────────────────────────────────────────────────

EXP_NAME, UNIFIED, EVAL_EXTRA, NUM_LABELS = EXPERIMENTS[exp_choice]()

global_tokenizer = AutoTokenizer.from_pretrained(MODEL_CFG["hf_name"])

if MODEL_CFG["is_decoder"]:
    if global_tokenizer.pad_token is None:
        global_tokenizer.pad_token = global_tokenizer.eos_token
    global_tokenizer.padding_side = "right"


def tokenize_fn(examples):
    if MODEL_CFG["is_decoder"]:
        prompts = [f"Prémisse : {p}\nHypothèse : {h}\n"
                   for p, h in zip(examples["premise"], examples["hypothesis"])]
        res = global_tokenizer(prompts, truncation=True, max_length=256)
    else:
        res = global_tokenizer(examples["premise"], examples["hypothesis"],
                               truncation=True, padding="max_length", max_length=128)
    res["labels"] = [int(l) for l in examples["label"]]
    return res


print("\nTokenisation...")
train_data = UNIFIED['train'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = UNIFIED['validation'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_data = UNIFIED['test'].map(tokenize_fn, batched=True, remove_columns=UNIFIED['test'].column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

eval_extra_tokenized = {}
for name, ds in EVAL_EXTRA.items():
    tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_extra_tokenized[name] = tok

# ─────────────────────────────────────────────────────────
# 9. MÉTRIQUES
# ─────────────────────────────────────────────────────────

LABEL_NAMES_3 = ["entailment", "neutral", "contradiction"]
LABEL_NAMES_2 = ["non-contradiction", "contradiction"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="macro")
    label_ids   = list(range(NUM_LABELS))
    label_names = LABEL_NAMES_2 if NUM_LABELS == 2 else LABEL_NAMES_3
    try:
        cm = confusion_matrix(labels, predictions, labels=label_ids)
        print(f"\nMatrice de confusion:\n{cm}")
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=labels.tolist(), preds=predictions.tolist(),
            class_names=label_names
        )})
        prec, rec, f1_per, sup = precision_recall_fscore_support(
            labels, predictions, labels=label_ids, zero_division=0)
        for i, name in enumerate(label_names):
            wandb.log({f"{name}_precision": prec[i], f"{name}_recall": rec[i],
                       f"{name}_f1": f1_per[i], f"{name}_support": int(sup[i])})
    except Exception as e:
        print(f"Erreur confusion matrix: {e}")
    return {"accuracy": acc, "f1_score": f1}

# ─────────────────────────────────────────────────────────
# 10. FONCTION D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────

def train_run():
    run = wandb.init()
    config = wandb.config
    lora_alpha = 2 * config.lora_r

    n_shots = getattr(config, "n_shots", None)
    nshots_suffix = f"_n{n_shots}" if n_shots is not None else ""
    run_label = f"{MODEL_CFG['short']}_exp{exp_choice}_r{config.lora_r}_a{lora_alpha}_lr{config.learning_rate}_d{config.lora_dropout}{nshots_suffix}"

    # EXP 8 : balanced_select (N//3 par classe)
    if n_shots is not None:
        current_train_data = balanced_select(train_data, n_shots)
        print(f"\n[EXP8] N-shot={n_shots} → {len(current_train_data)} ex ({n_shots//3}/classe)")
    else:
        current_train_data = train_data

    print(f"\n{'='*60}\n{run_label}\n{'='*60}")

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
        per_device_train_batch_size=32,   # 32 par GPU × 2 GPU = 64 effectif
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        fp16=True,                         # Mixed precision pour accélérer sur T4
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=current_train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=global_tokenizer),
        compute_metrics=compute_metrics,
    )

    # ── Zero-shot avant entraînement ──
    print(f"\n[ZERO-SHOT] Test ({len(test_data)} ex)...")
    zs = trainer.evaluate(test_data, metric_key_prefix="zeroshot")
    wandb.summary["zeroshot_accuracy"]  = zs["zeroshot_accuracy"]
    wandb.summary["zeroshot_f1_score"]  = zs.get("zeroshot_f1_score", 0.0)
    print(f"ZERO-SHOT — Acc:{zs['zeroshot_accuracy']:.2%} F1:{zs.get('zeroshot_f1_score',0):.4f}")

    for eval_name, eval_ds in eval_extra_tokenized.items():
        print(f"\n[ZERO-SHOT] {eval_name} ({len(eval_ds)} ex)...")
        zs_e = trainer.evaluate(eval_ds, metric_key_prefix=f"zeroshot_{eval_name}")
        wandb.summary[f"zeroshot_{eval_name}_accuracy"] = zs_e[f"zeroshot_{eval_name}_accuracy"]
        wandb.summary[f"zeroshot_{eval_name}_f1_score"] = zs_e.get(f"zeroshot_{eval_name}_f1_score", 0.0)

    # ── Entraînement ──
    trainer.train()

    # ── Évaluation finale ──
    print(f"\n[TEST FINAL] {len(test_data)} ex...")
    res = trainer.evaluate(test_data, metric_key_prefix="test")
    wandb.summary["final_test_accuracy"]  = res["test_accuracy"]
    wandb.summary["final_test_f1_score"]  = res.get("test_f1_score", 0.0)
    print(f"TEST — Acc:{res['test_accuracy']:.2%} F1:{res.get('test_f1_score',0):.4f}")

    for eval_name, eval_ds in eval_extra_tokenized.items():
        print(f"\n[EVAL POST-TRAIN] {eval_name}...")
        r_e = trainer.evaluate(eval_ds, metric_key_prefix=f"final_{eval_name}")
        wandb.summary[f"final_{eval_name}_accuracy"] = r_e[f"final_{eval_name}_accuracy"]
        wandb.summary[f"final_{eval_name}_f1_score"] = r_e.get(f"final_{eval_name}_f1_score", 0.0)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    wandb.finish()


# ─────────────────────────────────────────────────────────
# 11. LANCEMENT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    active_sweep_config = SWEEP_CONFIG_EXP8 if exp_choice == "8" else SWEEP_CONFIG

    total_runs = 1
    for param in active_sweep_config["parameters"].values():
        if "values" in param:
            total_runs *= len(param["values"])

    print(f"\nEXP {exp_choice} — {total_runs} runs")
    if exp_choice == "8":
        print(f"  N-shots : {active_sweep_config['parameters']['n_shots']['values']}")
        print(f"  LR      : {active_sweep_config['parameters']['learning_rate']['values']}")
    else:
        r_vals = active_sweep_config['parameters']['lora_r']['values']
        print(f"  lora_r  : {r_vals}  →  lora_alpha (2×r) : {[2*r for r in r_vals]}")
        print(f"  lr      : {active_sweep_config['parameters']['learning_rate']['values']}")
        print(f"  dropout : {active_sweep_config['parameters']['lora_dropout']['values']}")

    auto = len(sys.argv) > 3
    confirm = "o" if auto else input(f"\nLancer {total_runs} run(s) ? (o/n): ").strip().lower()

    if confirm == "o":
        sweep_id = wandb.sweep(sweep=active_sweep_config, project="fewshot-nli-fr")
        print(f"Sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=train_run, count=total_runs)
        print("Terminé !")
    else:
        print("Annulé.")
