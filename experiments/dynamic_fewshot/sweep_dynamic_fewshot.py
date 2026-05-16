"""
Dynamic Few-Shot NLI — LLMs génératifs (HuggingFace)

Ce script évalue des LLMs en mode Dynamic Few-Shot.
Au lieu de prendre 5 exemples au hasard, il utilise SentenceTransformers
pour calculer la similarité cosinus entre la question de test et la base
d'entraînement, et injecte les 5 exemples les plus pertinents sémantiquement.

Usage :
  python3 sweep_dynamic_fewshot.py --model mistral --train_dataset sick --eval_dataset sick --n_shots 5
"""

import os, sys, random, argparse, csv, traceback
from collections import defaultdict

os.environ.setdefault("WANDB_PROJECT", "fewshot-nli-fr")

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────
# 1. CONSTANTES ET CONFIGURATIONS (Totalement isolé)
# ─────────────────────────────────────────────────────────
LABEL_NAMES_3 = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES_2 = {0: "non-contradiction", 1: "contradiction"}

DATASETS_LIST = ["sick", "rte3", "gqnli", "fracas-gq", "daccord"]

MODEL_CONFIGS = {
    "mistral": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "short": "mistral_7b",
        "use_4bit": True,
        "backend": "hf",
    },
    "llama3": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "short": "llama3_8b",
        "use_4bit": True,
        "backend": "hf",
    },
    "gpt-4o-mini": {
        "hf_name": "gpt-4o-mini",
        "short": "gpt4o_mini",
        "use_4bit": False,
        "backend": "openai",
    }
}

LABEL_ALIASES = {
    "entailment": 0, "entail": 0, "oui": 0, "vrai": 0, "yes": 0,
    "neutral": 1, "neutre": 1, "indeterminate": 1,
    "contradiction": 2, "contradicts": 2, "non": 2, "faux": 2, "no": 2,
    "non-contradiction": 0,
}

def parse_label(text: str, num_labels: int) -> int:
    import re
    text_low = text.lower()
    text_low = re.sub(r"\*+", "", text_low).strip()
    for line in text_low.split("\n"):
        for kw in ["label :", "label:", "étiquette :", "étiquette:", "réponse :", "réponse:"]:
            if kw in line:
                remainder = line.split(kw, 1)[-1].strip()
                for alias, val in LABEL_ALIASES.items():
                    if alias in remainder:
                        if num_labels == 2 and val == 2: return 1
                        if num_labels == 2 and val == 1: return 0
                        return val
    for alias, val in LABEL_ALIASES.items():
        if alias in text_low:
            if num_labels == 2 and val == 2: return 1
            if num_labels == 2 and val == 1: return 0
            return val
    return -1

def harmonize_label(label, source_num_labels, target_num_labels):
    if label == -1: return -1
    if source_num_labels == 3 and target_num_labels == 2:
        return 1 if label == 2 else 0
    if source_num_labels == 2 and target_num_labels == 3:
        return 2 if label == 1 else 0
    return label

def get_train_test(dataset_name: str):
    from datasets import load_dataset
    import random
    
    LABEL_MAP = {"yes": 0, "entailment": 0, "neutral": 1, "no": 2, "contradiction": 2}
    def map_lbl(label):
        if isinstance(label, int): return label if label in [0, 1, 2] else -1
        s = str(label).lower().strip()
        if s in ("0", "1", "2"): return int(s)
        return LABEL_MAP.get(s, -1)

    if dataset_name == "sick":
        splits = load_dataset("maximoss/sick-fr")
        def convert(ex):
            lbl = str(ex["entailment_label"]).strip().upper()
            return {"premise": ex["sentence_A"], "hypothesis": ex["sentence_B"], "label": 0 if lbl == "ENTAILMENT" else (1 if lbl == "NEUTRAL" else 2)}
        return splits["train"].map(convert, remove_columns=splits["train"].column_names).filter(lambda x: x["label"] != -1), \
               splits["test"].map(convert, remove_columns=splits["test"].column_names).filter(lambda x: x["label"] != -1), 3
               
    elif dataset_name == "rte3":
        raw = load_dataset("maximoss/rte3-french")
        def convert(ex): return {"premise": ex["premise"], "hypothesis": ex["hypothesis"], "label": map_lbl(ex["label"])}
        return raw["dev"].map(convert, remove_columns=raw["dev"].column_names).filter(lambda x: x["label"] != -1), \
               raw["test"].map(convert, remove_columns=raw["test"].column_names).filter(lambda x: x["label"] != -1), 3
               
    elif dataset_name == "gqnli":
        ds = load_dataset("maximoss/gqnli-fr")["test"]
        def convert(ex): return {"premise": ex["premise"], "hypothesis": ex["hypothesis"], "label": map_lbl(ex["label"])}
        ds = ds.map(convert, remove_columns=ds.column_names).filter(lambda x: x["label"] != -1)
        train_idx = list(range(80,100)) + list(range(180,200)) + list(range(280,300))
        test_idx = list(range(300))
        return ds.select(train_idx), ds.select(test_idx), 3
        
    elif dataset_name == "fracas-gq":
        ds = load_dataset("maximoss/fracas")["train"]
        ds = ds.filter(lambda x: x["topic"] == "GENERALIZED QUANTIFIERS" and str(x["label"]).strip().lower() != "undef")
        def convert(ex): return {"premise": ex["premises"], "hypothesis": ex["hypothesis"], "label": map_lbl(ex["label"])}
        ds = ds.map(convert, remove_columns=ds.column_names).filter(lambda x: x["label"] != -1)
        n = len(ds)
        return ds.select(range(int(n * 0.8))), ds.select(range(int(n * 0.8), n)), 3
        
    elif dataset_name == "daccord":
        ds = load_dataset("maximoss/daccord-contradictions")["train"]
        cols = [c for c in ds.column_names if c not in {"premise", "hypothesis", "label"}]
        ds = ds.remove_columns(cols)
        n = len(ds)
        return ds.select(range(int(n * 0.5))), ds.select(range(int(n * 0.5), n)), 2
        
    raise ValueError(f"Dataset inconnu : {dataset_name}")

# ─────────────────────────────────────────────────────────
# 2. CONSTRUCTION DU PROMPT (Identique à Label-Only)
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

def format_example(ex: dict, num_labels: int, with_answer: bool = True) -> str:
    label_names = LABEL_NAMES_2 if num_labels == 2 else LABEL_NAMES_3
    text = f"Prémisse : {ex['premise']}\nHypothèse : {ex['hypothesis']}"
    if with_answer:
        label_str = label_names.get(ex["label"], "unknown")
        text += f"\nLabel : {label_str}"
    else:
        text += "\nLabel :"
    return text

def build_prompt(dynamic_examples: list, test_example: dict, num_labels: int) -> list:
    system = SYSTEM_PROMPT_BINARY if num_labels == 2 else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]

    if dynamic_examples:
        examples_text = "\n\n---\n\n".join(
            format_example(ex, num_labels, with_answer=True)
            for ex in dynamic_examples
        )
        messages.append({
            "role": "user",
            "content": f"Voici {len(dynamic_examples)} exemple(s) similaires :\n\n{examples_text}\n\n---\n\nMaintenant, classe cet exemple :\n\n{format_example(test_example, num_labels, with_answer=False)}"
        })
    else:
        messages.append({
            "role": "user",
            "content": format_example(test_example, num_labels, with_answer=False)
        })

    return messages

# ─────────────────────────────────────────────────────────
# 3. SÉLECTION DYNAMIQUE AVEC EMBEDDINGS (RAG)
# ─────────────────────────────────────────────────────────

def compute_train_embeddings(train_ds, embedder):
    """Précalcule les embeddings de toutes les phrases d'entraînement pour aller plus vite."""
    print(f"Calcul des embeddings pour les {len(train_ds)} exemples d'entraînement...")
    texts = [f"{ex['premise']} {ex['hypothesis']}" for ex in train_ds]
    embeddings = embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return embeddings

def select_dynamic_examples(test_ex: dict, train_ds, train_embeddings, embedder, n_shots: int, source_num_labels: int, target_num_labels: int):
    """Sélectionne dynamiquement les n_shots exemples les plus pertinents."""
    if n_shots == 0:
        return []
        
    test_text = f"{test_ex['premise']} {test_ex['hypothesis']}"
    test_emb = embedder.encode(test_text, convert_to_tensor=True)
    
    # Cosine similarity entre le test et tout le train
    cos_scores = util.cos_sim(test_emb, train_embeddings)[0]
    
    # Pour éviter de biaiser le LLM avec une seule classe, on prend les plus similaires,
    # mais on essaie de garder une certaine diversité (ex: au moins 1 de chaque classe si possible).
    # Mais la méthode classique consiste simplement à prendre le Top K.
    top_results = torch.topk(cos_scores, k=min(n_shots * 3, len(train_ds))) # On regarde un peu plus large pour équilibrer
    
    selected = []
    seen_labels = set()
    
    # 1. On essaie de prendre au moins un exemple de chaque classe parmi les plus proches
    for idx in top_results.indices:
        ex = train_ds[int(idx)]
        harmonized_label = ex["label"]
        if source_num_labels != target_num_labels:
             if source_num_labels == 3 and target_num_labels == 2:
                 harmonized_label = 1 if ex["label"] == 2 else 0
             elif source_num_labels == 2 and target_num_labels == 3:
                 harmonized_label = 2 if ex["label"] == 1 else 0
        
        if harmonized_label not in seen_labels and len(selected) < n_shots:
            # On copie pour éviter de modifier le dataset original
            ex_copy = ex.copy()
            ex_copy["label"] = harmonized_label
            selected.append(ex_copy)
            seen_labels.add(harmonized_label)
            
    # 2. S'il manque des exemples pour atteindre n_shots, on complète avec les plus hauts scores restants
    for idx in top_results.indices:
        if len(selected) >= n_shots:
            break
        ex = train_ds[int(idx)]
        harmonized_label = ex["label"]
        # Même logique de conversion
        if source_num_labels != target_num_labels:
             if source_num_labels == 3 and target_num_labels == 2:
                 harmonized_label = 1 if ex["label"] == 2 else 0
             elif source_num_labels == 2 and target_num_labels == 3:
                 harmonized_label = 2 if ex["label"] == 1 else 0
                 
        ex_copy = ex.copy()
        ex_copy["label"] = harmonized_label
        
        # On s'assure de ne pas ajouter des doublons stricts
        if ex_copy not in selected:
            selected.append(ex_copy)
            
    return selected[:n_shots]

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
    token = os.environ.get("HF_TOKEN")
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
            model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16, token=token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, token=token)

    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 15) -> str:
    try:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        input_ids = inputs["input_ids"]
    except Exception:
        prompt = "\n".join(m["content"] for m in messages) + "\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, pad_token_id=tokenizer.pad_token_id,
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

    metrics = {f"{prefix}/accuracy": acc, f"{prefix}/f1_macro": f1, f"{prefix}/parse_rate": parse_rate}
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

SHOTS_SWEEP_VALUES = [5] # Concentrons-nous sur 5 exemples comme demandé
SEEDS_VALUES = [42]

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "test/f1_macro", "goal": "maximize"},
    "parameters": {
        "n_shots":       {"values": SHOTS_SWEEP_VALUES},
        "train_dataset": {"values": DATASETS_LIST},
        "eval_dataset":  {"values": DATASETS_LIST},
        "seed":          {"values": SEEDS_VALUES},
    },
}

_G_MODEL         = None
_G_TOKENIZER     = None
_G_MODEL_CFG     = None
_G_ARGS          = None
_G_DATASETS_CACHE = {}
_G_EMBEDDER      = None

def get_cached_dataset(ds_name):
    if ds_name not in _G_DATASETS_CACHE:
        train_ds, test_ds, num_labels = get_train_test(ds_name)
        
        # On calcule les embeddings du train une seule fois pour ce dataset
        global _G_EMBEDDER
        train_embeddings = compute_train_embeddings(train_ds, _G_EMBEDDER)
        
        _G_DATASETS_CACHE[ds_name] = {
            "train": train_ds, 
            "test": test_ds, 
            "num_labels": num_labels,
            "train_embeddings": train_embeddings
        }
    return _G_DATASETS_CACHE[ds_name]

def eval_run():
    try:
        run = wandb.init()
        config = wandb.config
        n_shots       = config.n_shots
        train_ds_name = config.train_dataset
        eval_ds_name  = config.eval_dataset
        seed          = config.seed

        run_name = f"DYN_{_G_MODEL_CFG['short']}_train-{train_ds_name}_eval-{eval_ds_name}_n{n_shots}_s{seed}"
        run.name = run_name

        train_info = get_cached_dataset(train_ds_name)
        eval_info  = get_cached_dataset(eval_ds_name)

        target_num_labels = eval_info["num_labels"]
        source_num_labels = train_info["num_labels"]

        run.config.update({
            "model":      _G_MODEL_CFG["hf_name"],
            "mode":       "intra_dynamic" if train_ds_name == eval_ds_name else "cross_dynamic",
            "num_labels": target_num_labels,
            "method":     "dynamic_fewshot"
        })

        test_ds = eval_info["test"]
        if _G_ARGS.max_eval_samples > 0 and len(test_ds) > _G_ARGS.max_eval_samples:
            test_ds = balanced_eval_sample(test_ds, _G_ARGS.max_eval_samples, eval_info["num_labels"], seed)

        print(f"\n--- {run_name} ---")
        print(f"Test sur {len(test_ds)} exemples avec recherche de similarité...")

        labels_true, labels_pred, raw_outputs = [], [], []

        for i, ex in enumerate(test_ds):
            # Sélection DYNAMIQUE pour CE test_example
            dynamic_examples = select_dynamic_examples(
                ex, 
                train_info["train"], 
                train_info["train_embeddings"], 
                _G_EMBEDDER, 
                n_shots, 
                source_num_labels, 
                target_num_labels
            )
            
            messages  = build_prompt(dynamic_examples, ex, target_num_labels)
            response  = generate_response(_G_MODEL, _G_TOKENIZER, messages, max_new_tokens=15)
            predicted = parse_label(response, target_num_labels)

            ex_label_harmonized = harmonize_label(ex["label"], source_num_labels, target_num_labels)

            labels_true.append(ex_label_harmonized)
            labels_pred.append(predicted)
            dyn_text = "\n\n".join([f"Ex {idx+1}:\nP: {d['premise']}\nH: {d['hypothesis']}\nL: {d['label']}" for idx, d in enumerate(dynamic_examples)])
            raw_outputs.append({
                "test_premise": ex["premise"],
                "test_hypothesis": ex["hypothesis"],
                "dynamic_examples_used": dyn_text,
                "true": ex_label_harmonized, 
                "pred": predicted, 
                "response": response
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(test_ds)}]...")

        metrics = compute_and_log_metrics(labels_true, labels_pred, target_num_labels)

        csv_path = "/kaggle/working/dynamic_fewshot_results.csv" if os.path.exists("/kaggle") else "dynamic_fewshot_results.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_name", "model", "train_ds", "eval_ds",
                "mode", "n_shots", "seed",
                "accuracy", "f1_macro", "parse_rate"
            ])
            if write_header: writer.writeheader()
            writer.writerow({
                "run_name":   run_name,
                "model":      _G_MODEL_CFG["short"],
                "train_ds":   train_ds_name,
                "eval_ds":    eval_ds_name,
                "mode":       "intra_dynamic" if train_ds_name == eval_ds_name else "cross_dynamic",
                "n_shots":    n_shots,
                "seed":       seed,
                "accuracy":   metrics.get("test/accuracy", ""),
                "f1_macro":   metrics.get("test/f1_macro", ""),
                "parse_rate": metrics.get("test/parse_rate", ""),
            })

        try:
            table = wandb.Table(columns=["test_premise", "test_hypothesis", "dynamic_examples_used", "true_label", "pred_label", "response"])
            for r in raw_outputs[:20]: 
                table.add_data(r["test_premise"], r["test_hypothesis"], r["dynamic_examples_used"], r["true"], r["pred"], r["response"])
            wandb.log({"predictions_sample": table})
        except Exception:
            pass
        wandb.finish()

    except Exception as e:
        print(f"❌ CRASH dans eval_run: {str(e)}\n{traceback.format_exc()}")
        try: wandb.finish(exit_code=1)
        except: pass
        raise e

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--train_dataset",    type=str,      choices=DATASETS_LIST)
    p.add_argument("--eval_dataset",     type=str,      choices=DATASETS_LIST)
    p.add_argument("--n_shots",          type=int,      default=5)
    p.add_argument("--seed",             type=int,      default=42)
    p.add_argument("--sweep",            action="store_true")
    p.add_argument("--max_eval_samples", type=int,      default=300)
    p.add_argument("--embed_model",      type=str,      default="paraphrase-multilingual-MiniLM-L12-v2")
    return p.parse_args()

def main():
    global _G_MODEL, _G_TOKENIZER, _G_MODEL_CFG, _G_ARGS, _G_EMBEDDER
    _G_ARGS      = parse_args()
    _G_MODEL_CFG = MODEL_CONFIGS[_G_ARGS.model]

    print(f"Chargement du modèle d'embedding : {_G_ARGS.embed_model} ...")
    _G_EMBEDDER = SentenceTransformer(_G_ARGS.embed_model)

    print(f"Modèle : {_G_MODEL_CFG['hf_name']}")
    _G_MODEL, _G_TOKENIZER = load_model_and_tokenizer(_G_MODEL_CFG)

    project_name = os.environ.get("WANDB_PROJECT", "fewshot-nli-fr")

    if _G_ARGS.sweep:
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=project_name)
        wandb.agent(sweep_id, function=eval_run)
    else:
        sweep_config_single = {
            "method": "grid",
            "metric": {"name": "test/f1_macro", "goal": "maximize"},
            "parameters": {
                "n_shots":       {"values": [_G_ARGS.n_shots]},
                "train_dataset": {"values": [_G_ARGS.train_dataset]},
                "eval_dataset":  {"values": [_G_ARGS.eval_dataset]},
                "seed":          {"values": [_G_ARGS.seed]},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_config_single, project=project_name)
        wandb.agent(sweep_id, function=eval_run, count=1)

if __name__ == "__main__":
    main()
