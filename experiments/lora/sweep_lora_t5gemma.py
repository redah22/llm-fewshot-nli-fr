"""
Script de Sweep WandB pour T5Gemma (Architecture Encoder-Decoder).
Configuration unifiée multi-dataset — splitting respectueux des groupes.

Règles de splitting appliquées :
  - GQNLI   : groupes de prémisses (tuteur) — même groupe = même split
  - DACCORD : split proportionnel par thème (a/b/c) + même prémisse = même split
  - RTE3    : même prémisse = même split (split 80/10/10 intra-dev + test séparé)
  - SICK    : splits originaux du dataset

Distribution résultante :
  TRAIN : RTE3-dev(80%) + SICK-train + DACCORD(70%) + GQNLI(groupes 8-10 ≈ 20%)
  VAL   : RTE3-dev(10%) + SICK-val   + DACCORD(15%) + GQNLI(groupes 6-7  ≈ 18%)
  TEST  : RTE3-test     + SICK-test  + DACCORD(15%) + GQNLI(groupes 1-5  ≈ 62%)
"""

import os
os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import random
import json
import numpy as np
import torch
import wandb
import shutil

from collections import defaultdict
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

BASE_MODEL = "google/t5gemma-2-270m-270m"
MODEL_SHORT = "t5gemma"
EXP_NAME = "sweep_unified_t5gemma"

if not torch.cuda.is_available():
    print("ERREUR CRITIQUE : Ce script nécessite un GPU CUDA pour QLoRA.")
    exit(1)

# ─────────────────────────────────────────────────────────
# 1. GROUPES GQNLI (définis par le tuteur)
# 10 groupes × 3 pages (offsets 0, 100, 200) = 300 exemples
# ─────────────────────────────────────────────────────────

GQNLI_GROUPS = [
    list(range(0, 20))  + list(range(100, 120)) + list(range(200, 220)),   # G1 : 60 ex
    list(range(20, 29)) + list(range(120, 129)) + list(range(220, 229)),   # G2 : 27 ex
    list(range(29, 41)) + list(range(129, 141)) + list(range(229, 241)),   # G3 : 36 ex
    list(range(41, 50)) + list(range(141, 150)) + list(range(241, 250)),   # G4 : 27 ex
    list(range(50, 62)) + list(range(150, 162)) + list(range(250, 262)),   # G5 : 36 ex
    list(range(62, 67)) + list(range(162, 167)) + list(range(262, 267)),   # G6 : 15 ex
    list(range(67, 80)) + list(range(167, 180)) + list(range(267, 280)),   # G7 : 39 ex
    list(range(80, 85)) + list(range(180, 185)) + list(range(280, 285)),   # G8 : 15 ex
    list(range(85, 91)) + list(range(185, 191)) + list(range(285, 291)),   # G9 : 18 ex
    list(range(91, 100)) + list(range(191, 200)) + list(range(291, 300)),  # G10: 27 ex
]  # Total : 300 exemples

# ─────────────────────────────────────────────────────────
# 2. UTILITAIRES
# ─────────────────────────────────────────────────────────

def _normalize_columns(ds, premise_key="premise"):
    """Normalise un dataset pour avoir uniquement premise, hypothesis, label."""
    cols_to_remove = [c for c in ds.column_names if c not in {premise_key, "hypothesis", "label"}]
    ds = ds.remove_columns(cols_to_remove)
    if premise_key != "premise":
        ds = ds.rename_column(premise_key, "premise")
    return ds


def _print_dist(ds, name):
    from collections import Counter
    c = Counter(ds["label"])
    total = max(len(ds), 1)
    ent, neu, con = c.get(0, 0), c.get(1, 0), c.get(2, 0)
    print(f"    {name} [{total}]: ent={ent}({100*ent/total:.0f}%) neu={neu}({100*neu/total:.0f}%) con={con}({100*con/total:.0f}%)")


# ─────────────────────────────────────────────────────────
# 3. FONCTIONS DE CHARGEMENT ET SPLITTING
# ─────────────────────────────────────────────────────────

def _load_gqnli_unified(seed=42):
    """
    Charge GQNLI et le split en respectant l'intégrité des groupes.
    Groupes 1-5 → test (~62%), groupes 6-7 → val (~18%), groupes 8-10 → train (~20%).
    """
    gqnli = load_dataset('maximoss/gqnli-fr')['test']
    gqnli = _normalize_columns(gqnli, premise_key="premise")

    # Assignation fixe des groupes (index 0-based dans GQNLI_GROUPS)
    test_groups  = [0, 1, 2, 3, 4]   # G1-G5 : ~186 ex
    val_groups   = [5, 6]             # G6-G7 : ~54 ex
    train_groups = [7, 8, 9]          # G8-G10: ~60 ex

    def collect_idx(group_indices):
        idx = []
        for g in group_indices:
            idx.extend(GQNLI_GROUPS[g])
        return sorted(idx)

    train_idx = collect_idx(train_groups)
    val_idx   = collect_idx(val_groups)
    test_idx  = collect_idx(test_groups)

    train_ds = gqnli.select(train_idx) if train_idx else None
    val_ds   = gqnli.select(val_idx)   if val_idx   else None
    test_ds  = gqnli.select(test_idx)  if test_idx  else None

    print("  GQNLI-FR (group-aware):")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds:
            _print_dist(ds, f"    gqnli-{name}")

    return train_ds, val_ds, test_ds


def _load_rte3_unified(seed=42):
    """
    Charge RTE3 et split le dev en 80/10/10 par groupes de prémisses.
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
            lbl = ex['label']
            if isinstance(lbl, str):
                lbl = {"yes": 0, "entailment": 0, "neutral": 1, "unknown": 1, "no": 2, "contradiction": 2}.get(lbl.lower().strip(), 1)
            idx = len(all_rows)
            all_rows.append({'premise': ex['premise'], 'hypothesis': ex['hypothesis'], 'label': int(lbl)})
            prem_groups[ex['premise']].append(idx)

    groups = list(prem_groups.values())
    rng.shuffle(groups)

    n_total = len(all_rows)
    n_train = int(n_total * 0.80)
    n_val   = int(n_total * 0.10)

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) < n_train:
            train_idx.extend(g)
        elif len(val_idx) < n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    def make(indices):
        return Dataset.from_list([all_rows[i] for i in indices])

    # Test set séparé du dataset original
    test_rows = []
    if 'test' in rte3:
        for ex in rte3['test']:
            lbl = ex['label']
            if isinstance(lbl, str):
                lbl = {"yes": 0, "entailment": 0, "neutral": 1, "unknown": 1, "no": 2, "contradiction": 2}.get(lbl.lower().strip(), 1)
            test_rows.append({'premise': ex['premise'], 'hypothesis': ex['hypothesis'], 'label': int(lbl)})
    rte3_test = Dataset.from_list(test_rows) if test_rows else make(test_idx)

    print("  RTE3-FR (premise-aware 80/10/10):")
    for name, ds in [("train", make(train_idx)), ("val", make(val_idx)), ("test", rte3_test)]:
        _print_dist(ds, f"    rte3-{name}")

    return make(train_idx), make(val_idx), rte3_test


def _load_sick_unified():
    """Charge SICK-FR avec conversion des labels."""
    sick = load_dataset('maximoss/sick-fr')

    def convert(ex):
        lbl = str(ex['entailment_label']).strip().upper()
        if lbl == 'ENTAILMENT':      label_id = 0
        elif lbl == 'NEUTRAL':       label_id = 1
        elif lbl == 'CONTRADICTION': label_id = 2
        else:                        label_id = 1
        return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}

    result = {}
    for split in sick:
        result[split] = sick[split].map(convert, remove_columns=sick[split].column_names)

    print("  SICK-FR:")
    for name in ['train', 'validation', 'test']:
        if name in result:
            _print_dist(result[name], f"    sick-{name}")

    return result


def _load_daccord_unified(seed=42):
    """
    Charge DACCORD et split en 70/15/15 par thème + groupes de prémisses.
    Remap labels : 1 (contradiction native) → 2 (NLI standard).
    """
    rng = random.Random(seed)
    dacc = load_dataset('maximoss/daccord-contradictions')['train']

    # Grouper par thème (1er caractère de l'id) puis par prémisse
    themes = defaultdict(lambda: defaultdict(list))
    for i, ex in enumerate(dacc):
        theme = str(ex['id'])[0]
        themes[theme][ex['premise']].append(i)

    split_idx = {'train': [], 'val': [], 'test': []}
    train_r, val_r = 0.70, 0.15

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
        # Remap : 1 (contradiction native DACCORD) → 2 (NLI standard)
        return ds.map(lambda ex: {'label': 2 if ex['label'] == 1 else ex['label']})

    result = {k: make_ds(v) for k, v in split_idx.items() if v}

    print("  DACCORD (theme+premise-aware 70/15/15):")
    for name, ds in result.items():
        _print_dist(ds, f"    daccord-{name}")

    return result


def load_unified_dataset():
    """Charge et combine les datasets selon les règles de splitting du tuteur."""
    print("Chargement unifié des datasets...")
    all_train, all_val, all_test = [], [], []

    # GQNLI-FR
    gq_train, gq_val, gq_test = _load_gqnli_unified()
    if gq_train: all_train.append(gq_train)
    if gq_val:   all_val.append(gq_val)
    if gq_test:  all_test.append(gq_test)

    # RTE3-FR
    rte_train, rte_val, rte_test = _load_rte3_unified()
    all_train.append(rte_train)
    all_val.append(rte_val)
    all_test.append(rte_test)

    # SICK-FR
    sick = _load_sick_unified()
    all_train.append(sick['train'])
    all_val.append(sick['validation'])
    all_test.append(sick['test'])

    # DACCORD
    dacc = _load_daccord_unified()
    all_train.append(dacc['train'])
    all_val.append(dacc['val'])
    all_test.append(dacc['test'])

    train_ds = concatenate_datasets(all_train).shuffle(seed=42)
    val_ds   = concatenate_datasets(all_val).shuffle(seed=42)
    test_ds  = concatenate_datasets(all_test).shuffle(seed=42)

    print(f"\nDataset unifié → Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")
    _print_dist(train_ds, "TRAIN total")
    _print_dist(val_ds,   "VAL total")
    _print_dist(test_ds,  "TEST total")

    return DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})


# ─────────────────────────────────────────────────────────
# 4. CONFIGURATION DU SWEEP
# ─────────────────────────────────────────────────────────

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/accuracy", "goal": "maximize"},
    "parameters": {
        "lora_r":        {"values": [32]},
        "lora_alpha":    {"values": [64]},
        "learning_rate": {"values": [5e-4]},
        "lora_dropout":  {"values": [0.1]},
    }
}

# ─────────────────────────────────────────────────────────
# 5. CHARGEMENT & TOKENISATION
# ─────────────────────────────────────────────────────────

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
val_data   = UNIFIED['validation'].map(preprocess_fn, batched=True, remove_columns=UNIFIED['validation'].column_names)
test_data  = UNIFIED['test'].map(preprocess_fn, batched=True, remove_columns=UNIFIED['test'].column_names)


# ─────────────────────────────────────────────────────────
# 6. MÉTRIQUES
# ─────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    dec_preds  = [p.strip() for p in global_tokenizer.batch_decode(preds,  skip_special_tokens=True)]
    dec_labels = [l.strip() for l in global_tokenizer.batch_decode(labels, skip_special_tokens=True)]

    print(f"\n[DEBUG] Extraits générés : {dec_preds[:5]}")
    print(f"[DEBUG] Extraits cibles  : {dec_labels[:5]}")

    LABEL_TO_INT = {"vrai": 0, "neutre": 1, "faux": 2}

    import re
    cleaned_preds = []
    for p in dec_preds:
        match = re.search(r'(vrai|faux|neutre)', p.lower())
        cleaned_preds.append(match.group(1) if match else "neutre")

    int_preds  = [LABEL_TO_INT[p] for p in cleaned_preds]
    int_labels = [LABEL_TO_INT.get(l.lower(), 1) for l in dec_labels]

    try:
        cm = confusion_matrix(int_labels, int_preds, labels=[0, 1, 2])
        print(f"\nMatrice de confusion:\n{cm}")
        label_names = ["entailment", "neutral", "contradiction"]
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=int_labels, preds=int_preds,
                class_names=label_names
            )
        })
        prec, rec, f1_per, sup = precision_recall_fscore_support(
            int_labels, int_preds, labels=[0, 1, 2], zero_division=0)
        for i, name in enumerate(label_names):
            wandb.log({
                f"{name}_precision": prec[i],
                f"{name}_recall":    rec[i],
                f"{name}_f1":        f1_per[i],
                f"{name}_support":   int(sup[i]),
            })
    except Exception as e:
        print(f"Erreur logging confusion matrix: {e}")

    acc = sum(p == l for p, l in zip(cleaned_preds, dec_labels)) / max(1, len(dec_labels))
    return {
        "accuracy": acc,
        "f1_score": f1_score(int_labels, int_preds, average="macro")
    }


# ─────────────────────────────────────────────────────────
# 7. FONCTION D'ENTRAÎNEMENT (SWEEP WANDB)
# ─────────────────────────────────────────────────────────

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
    total     = sum(p.numel() for p in model.parameters())
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

    collator = DataCollatorForSeq2Seq(
        tokenizer=global_tokenizer, model=None, padding=True, pad_to_multiple_of=8
    )

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
    zs_f1  = zs_results.get("zeroshot_f1_score", 0.0)
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


# ─────────────────────────────────────────────────────────
# 8. LANCEMENT
# ─────────────────────────────────────────────────────────

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
