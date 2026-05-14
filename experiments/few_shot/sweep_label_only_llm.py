"""
Label-Only Few-Shot NLI — LLMs génératifs (HuggingFace)

Ce script évalue des LLMs en mode Few-Shot STRICT (sans Chain of Thought).
Il prend des exemples de `train_dataset` et évalue sur `eval_dataset`.

Usage :
  python3 sweep_label_only_llm.py --model llama3 --train_dataset gqnli --eval_dataset sick --n_shots 5
  python3 sweep_label_only_llm.py --model qwen2.5 --sweep
"""

import os, sys, random, argparse
from collections import defaultdict

os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────
# 1. IMPORTS DES LOADERS EXISTANTS
# ─────────────────────────────────────────────────────────
# On suppose que ce script est lancé depuis la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.fewshot_cot.run_fewshot_cot import (
    MODEL_CONFIGS, DATASETS, get_train_test, LABEL_NAMES_2, LABEL_NAMES_3, parse_label
)

# ─────────────────────────────────────────────────────────
# 2. CONSTRUCTION DU PROMPT "LABEL-ONLY" (Sans Raisonnement)
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en inférence de langue naturelle (NLI) en français.
Ta tâche est de déterminer la relation logique entre une prémisse et une hypothèse.
Les labels possibles sont :
- entailment : l'hypothèse découle logiquement de la prémisse
- neutral : l'hypothèse n'est ni confirmée ni contredite par la prémisse
- contradiction : l'hypothèse contredit la prémisse

Réponds UNIQUEMENT avec le label exact, rien d'autre."""

SYSTEM_PROMPT_BINARY = """Tu es un expert en détection de contradictions en français.
Ta tâche est de déterminer si une hypothèse contredit une prémisse.
Les labels possibles sont :
- non-contradiction : l'hypothèse ne contredit pas la prémisse
- contradiction : l'hypothèse contredit la prémisse

Réponds UNIQUEMENT avec le label exact, rien d'autre."""

def format_example_label_only(ex: dict, num_labels: int, with_answer: bool = True) -> str:
    label_names = LABEL_NAMES_2 if num_labels == 2 else LABEL_NAMES_3
    text = f"Prémisse : {ex['premise']}\nHypothèse : {ex['hypothesis']}"
    if with_answer:
        label_str = label_names.get(ex["label"], "unknown")
        text += f"\nLabel : {label_str}"
    else:
        text += "\nLabel :"
    return text

def build_prompt_label_only(fewshot_examples: list, test_example: dict, num_labels: int) -> list:
    """Construit le prompt au format chat pour le mode Label-Only."""
    system = SYSTEM_PROMPT_BINARY if num_labels == 2 else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if fewshot_examples:
        examples_text = "\n\n---\n\n".join(
            format_example_label_only(ex, num_labels, with_answer=True)
            for ex in fewshot_examples
        )
        messages.append({
            "role": "user",
            "content": f"Voici {len(fewshot_examples)} exemple(s) :\n\n{examples_text}\n\n---\n\nMaintenant, classe cet exemple :\n\n{format_example_label_only(test_example, num_labels, with_answer=False)}"
        })
    else:
        messages.append({
            "role": "user",
            "content": format_example_label_only(test_example, num_labels, with_answer=False)
        })

    return messages

# ─────────────────────────────────────────────────────────
# 3. SÉLECTION EXEMPLES FEW-SHOT
# ─────────────────────────────────────────────────────────

def select_fewshot_examples(train_ds, n_shots: int, num_labels: int, seed: int = 42) -> list:
    if n_shots == 0:
        return []
    rng = random.Random(seed)
    per_class = max(1, n_shots // num_labels)
    by_label = defaultdict(list)
    for ex in train_ds:
        by_label[ex["label"]].append(ex)
    selected = []
    for label in range(num_labels):
        pool = by_label.get(label, [])
        if not pool: continue
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return selected[:n_shots]

def harmonize_label(label, source_num_labels, target_num_labels):
    """Harmonise un label si on passe du 3-classes au 2-classes (ex: DACCORD)."""
    if label == -1: return -1
    if source_num_labels == 3 and target_num_labels == 2:
        # RTE/GQNLI/SICK (0,1,2) -> DACCORD (0,1)
        # 0 (entailment), 1 (neutral) -> 0 (non-contradiction/compatible)
        # 2 (contradiction) -> 1 (contradiction)
        return 1 if label == 2 else 0
    if source_num_labels == 2 and target_num_labels == 3:
        # DACCORD (0,1) -> RTE/GQNLI/SICK (0,1,2)
        # 0 (non-contradiction) -> 0 (entailment) -- Choix par défaut
        # 1 (contradiction) -> 2 (contradiction)
        return 2 if label == 1 else 0
    return label

def balanced_eval_sample(test_ds, max_samples: int, num_labels: int, seed: int = 42) -> Dataset:
    if len(test_ds) <= max_samples:
        return test_ds
    rng = random.Random(seed)
    per_class = max_samples // num_labels
    by_label = defaultdict(list)
    for i, ex in enumerate(test_ds):
        by_label[ex["label"]].append(i)
    selected = []
    for label in range(num_labels):
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return test_ds.select(sorted(selected))

# ─────────────────────────────────────────────────────────
# 4. CHARGEMENT MODÈLE & INFÉRENCE
# ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_cfg: dict):
    model_name = model_cfg["hf_name"]
    print(f"Chargement de {model_name}...")

    # Récupération du token HF (Secrets Kaggle ou Env)
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_cfg.get("use_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            token=token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16, token=token
        )

    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 15) -> str:
    # Pour Label-Only, on a besoin de très peu de tokens
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except Exception:
        prompt = "\n".join(m["content"] for m in messages) + "\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def compute_and_log_metrics(labels_true, labels_pred, num_labels, prefix="test"):
    valid_mask = [i for i, p in enumerate(labels_pred) if p != -1]
    if not valid_mask:
        print("ERREUR : aucune prédiction valide extraite.")
        return {}

    y_true = [labels_true[i] for i in valid_mask]
    y_pred = [labels_pred[i] for i in valid_mask]
    parse_rate = len(valid_mask) / len(labels_true)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    label_ids   = list(range(num_labels))
    label_names = [LABEL_NAMES_2[i] for i in label_ids] if num_labels == 2 else [LABEL_NAMES_3[i] for i in label_ids]

    prec, rec, f1_per, sup = precision_recall_fscore_support(y_true, y_pred, labels=label_ids, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    metrics = {
        f"{prefix}/accuracy":   acc,
        f"{prefix}/f1_macro":   f1,
        f"{prefix}/parse_rate": parse_rate,
    }
    for i, name in enumerate(label_names):
        metrics[f"{prefix}/{name}_f1"]       = f1_per[i]
        metrics[f"{prefix}/{name}_precision"] = prec[i]
        metrics[f"{prefix}/{name}_recall"]    = rec[i]

    try:
        wandb.log(metrics)
        wandb.log({f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=label_names)})
        wandb.summary.update({k: v for k, v in metrics.items()})
    except Exception as e:
        print(f"[WARN] WandB log failed: {e}")

    print(f"\nAccuracy : {acc:.2%} | F1 macro : {f1:.4f} | Parse rate : {parse_rate:.2%}")
    return metrics

# ─────────────────────────────────────────────────────────
# 5. MAIN SWEEP
# ─────────────────────────────────────────────────────────

SHOTS_SWEEP_VALUES = [0, 1, 3, 5, 10]
SEEDS_VALUES = [42, 123, 999]
DATASETS_LIST = list(DATASETS.keys())

# Si on veut tout tester dans un seul Sweep (intra + cross) :
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "test/f1_macro", "goal": "maximize"},
    "parameters": {
        "n_shots": {"values": SHOTS_SWEEP_VALUES},
        "train_dataset": {"values": DATASETS_LIST},
        "eval_dataset": {"values": DATASETS_LIST},
        "seed": {"values": SEEDS_VALUES},
    },
}

_G_MODEL = None
_G_TOKENIZER = None
_G_MODEL_CFG = None
_G_ARGS = None
_G_DATASETS_CACHE = {}

def get_cached_dataset(ds_name):
    if ds_name not in _G_DATASETS_CACHE:
        train_ds, test_ds, num_labels = get_train_test(ds_name)
        _G_DATASETS_CACHE[ds_name] = {"train": train_ds, "test": test_ds, "num_labels": num_labels}
    return _G_DATASETS_CACHE[ds_name]

def eval_run():
    run = wandb.init()
    config = wandb.config
    n_shots = config.n_shots
    train_ds_name = config.train_dataset
    eval_ds_name = config.eval_dataset
    seed = config.seed

    run_name = f"{_G_MODEL_CFG['short']}_train-{train_ds_name}_eval-{eval_ds_name}_n{n_shots}_s{seed}"
    run.name = run_name
    
    # Chargement
    train_info = get_cached_dataset(train_ds_name)
    eval_info = get_cached_dataset(eval_ds_name)
    
    # Si num_labels diffèrent (ex: daccord(2) et gqnli(3)), on se base sur le dataset d'EVAL
    target_num_labels = eval_info["num_labels"]
    source_num_labels = train_info["num_labels"]

    run.config.update({
        "model": _G_MODEL_CFG["hf_name"],
        "mode": "intra" if train_ds_name == eval_ds_name else "cross",
        "num_labels": target_num_labels,
        "use_cot": False
    })

    # Train (Few-Shot)
    fewshot_examples = select_fewshot_examples(train_info["train"], n_shots, source_num_labels, seed)
    # Harmonisation des labels few-shot pour correspondre à la tâche d'eval
    for ex in fewshot_examples:
        ex["label"] = harmonize_label(ex["label"], source_num_labels, target_num_labels)
    
    # Eval
    test_ds = eval_info["test"]
    if _G_ARGS.max_eval_samples > 0 and len(test_ds) > _G_ARGS.max_eval_samples:
        test_ds = balanced_eval_sample(test_ds, _G_ARGS.max_eval_samples, eval_info["num_labels"], seed)

    print(f"\n--- {run_name} ---")
    print(f"Test sur {len(test_ds)} exemples...")

    labels_true, labels_pred, raw_outputs = [], [], []

    for i, ex in enumerate(test_ds):
        messages = build_prompt_label_only(fewshot_examples, ex, target_num_labels)
        response = generate_response(_G_MODEL, _G_TOKENIZER, messages, max_new_tokens=15)
        predicted = parse_label(response, target_num_labels)

        labels_true.append(ex["label"])
        labels_pred.append(predicted)
        raw_outputs.append({"true": ex["label"], "pred": predicted, "response": response})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_ds)}]...")

    metrics = compute_and_log_metrics(labels_true, labels_pred, target_num_labels)

    # ── Sauvegarde CSV locale (récupérable depuis l'Output Kaggle) ─────────
    import csv, os as _os
    csv_path = "/kaggle/working/label_only_results.csv" if _os.path.exists("/kaggle") else "label_only_results.csv"
    write_header = not _os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "model", "train_ds", "eval_ds",
                                               "mode", "n_shots", "seed",
                                               "accuracy", "f1_macro", "parse_rate"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_name":   run_name,
            "model":      _G_MODEL_CFG["short"],
            "train_ds":   train_ds_name,
            "eval_ds":    eval_ds_name,
            "mode":       "intra" if train_ds_name == eval_ds_name else "cross",
            "n_shots":    n_shots,
            "seed":       seed,
            "accuracy":   metrics.get("test/accuracy", ""),
            "f1_macro":   metrics.get("test/f1_macro", ""),
            "parse_rate": metrics.get("test/parse_rate", ""),
        })
    print(f"💾 Résultat sauvegardé dans {csv_path}")
    # ──────────────────────────────────────────────────────────────────────

    try:
        table = wandb.Table(columns=["true_label", "pred_label", "response"])
        for r in raw_outputs[:20]:
            table.add_data(r["true"], r["pred"], r["response"])
        wandb.log({"predictions_sample": table})
        wandb.finish()
    except Exception as e:
        print(f"[WARN] WandB finish failed: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--train_dataset", type=str, choices=DATASETS_LIST)
    p.add_argument("--eval_dataset", type=str, choices=DATASETS_LIST)
    p.add_argument("--n_shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true", help="Lance un Grid Sweep WandB complet (très long !)")
    p.add_argument("--max_eval_samples", type=int, default=300)
    p.add_argument("--auto", action="store_true")
    return p.parse_args()

def main():
    global _G_MODEL, _G_TOKENIZER, _G_MODEL_CFG, _G_ARGS
    _G_ARGS = parse_args()
    _G_MODEL_CFG = MODEL_CONFIGS[_G_ARGS.model]

    if not _G_ARGS.auto:
        print(f"Modèle : {_G_MODEL_CFG['hf_name']}")
        if _G_ARGS.sweep:
            print(f"Lancement d'un Sweep complet intra et cross dataset ({len(DATASETS_LIST)}x{len(DATASETS_LIST)} x 5 shots x 3 seeds)")

    _G_MODEL, _G_TOKENIZER = load_model_and_tokenizer(_G_MODEL_CFG)

    if _G_ARGS.sweep:
        project_name = os.environ.get("WANDB_PROJECT", "fewshot-nli-fr")
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=project_name)
        wandb.agent(sweep_id, function=eval_run)
    else:
        if not _G_ARGS.train_dataset or not _G_ARGS.eval_dataset:
            print("Erreur: en mode run simple, spécifiez --train_dataset et --eval_dataset")
            sys.exit(1)
        SWEEP_CONFIG_SINGLE = {
            "method": "grid",
            "metric": {"name": "test/f1_macro", "goal": "maximize"},
            "parameters": {
                "n_shots": {"values": [_G_ARGS.n_shots]},
                "train_dataset": {"values": [_G_ARGS.train_dataset]},
                "eval_dataset": {"values": [_G_ARGS.eval_dataset]},
                "seed": {"values": [_G_ARGS.seed]},
            },
        }
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG_SINGLE, project="fewshot-nli-fr")
        wandb.agent(sweep_id, function=eval_run, count=1)

if __name__ == "__main__":
    main()
